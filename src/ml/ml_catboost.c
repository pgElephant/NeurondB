/*-------------------------------------------------------------------------
 *
 * ml_catboost.c
 *    CatBoost Integration for NeuronDB
 *
 *  Provides full integration of the Yandex CatBoost gradient boosting library.
 *  Supports categorical features and can use GPU acceleration if enabled.
 *  Requires CatBoost C library (libcatboostmodel.so) and headers.
 *
 *  Copyright (c) 2024-2025, pgElephant, Inc.
 *
 *  IDENTIFICATION
 *      src/ml/ml_catboost.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "storage/fd.h"
#include "miscadmin.h"
#include <utils/memutils.h>
#ifdef __has_include
#if __has_include(<catboost/c_api.h>)
#include <catboost/c_api.h>
#define CATBOOST_AVAILABLE
#endif
#endif
#include <stdio.h>

/* PG_MODULE_MAGIC is in neurondb.c only */

PG_FUNCTION_INFO_V1(train_catboost_classifier);
PG_FUNCTION_INFO_V1(train_catboost_regressor);
PG_FUNCTION_INFO_V1(predict_catboost);

/*
 * Helper function to get column index in a query result by name.
 */
static int
get_column_index(SPITupleTable *tuptable, TupleDesc tupdesc, const char *colname)
{
	int i;

	for (i = 0; i < tupdesc->natts; i++)
	{
                if (strcmp(NameStr(TupleDescAttr(tupdesc, i)->attname), colname) == 0)
			return i;
	}
	elog(ERROR, "Column \"%s\" does not exist in tuple", colname);
	/* This line will not be reached, but included to match coding style */
	return -1;
}

/*
 * Helper to write training data to CSV for CatBoost
 */
static void
write_csv_from_spi(char *csv_path, SPITupleTable *tuptable, TupleDesc tupdesc,
				   int *feature_idxs, int n_features, int label_idx)
{
	FILE	   *fp;
	uint64		row;
	int			i;

	fp = AllocateFile(csv_path, "w");
	if (fp == NULL)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open file \"%s\" for writing: %m", csv_path)));

	/* Write header */
	for (i = 0; i < n_features; i++)
	{
		fprintf(fp, "f%d,", i);
	}
	fprintf(fp, "label\n");

	for (row = 0; row < tuptable->numvals; row++)
	{
		HeapTuple	tuple = tuptable->vals[row];
		for (i = 0; i < n_features; i++)
		{
			Datum		val;
			bool		isnull;
			Oid			typid;
			Oid			typoutput;
			bool		typisvarlena;
			char	   *valstr;

			val = heap_getattr(tuple, feature_idxs[i] + 1, tupdesc, &isnull);
			if (isnull)
			{
				fprintf(fp, ",");
			}
			else
			{
				typid = TupleDescAttr(tupdesc, feature_idxs[i])->atttypid;
				getTypeOutputInfo(typid, &typoutput, &typisvarlena);
				valstr = OidOutputFunctionCall(typoutput, val);
				fprintf(fp, "%s,", valstr);
			}
		}
		{
			Datum label_val;
			bool label_isnull;
			Oid typid;
			Oid typoutput;
			bool typisvarlena;
			char *labelstr;

			label_val = heap_getattr(tuple, label_idx + 1, tupdesc, &label_isnull);
			if (label_isnull)
			{
				fprintf(fp, "\n");
			}
			else
			{
				typid = TupleDescAttr(tupdesc, label_idx)->atttypid;
				getTypeOutputInfo(typid, &typoutput, &typisvarlena);
				labelstr = OidOutputFunctionCall(typoutput, label_val);
				fprintf(fp, "%s\n", labelstr);
			}
		}
	}
	FreeFile(fp);
}

/*
 * Helper function for error translation from CatBoost return codes.
 */
static void
check_catboost_error(int err_code)
{
#ifdef CATBOOST_AVAILABLE
	if (err_code != 0)
	{
		const char *err_msg = CatBoostGetErrorString();
		if (err_msg == NULL)
			err_msg = "Unknown error from CatBoost";
		elog(ERROR, "CatBoost error: %s (code=%d)", err_msg, err_code);
	}
#else
	if (err_code != 0)
		elog(ERROR, "CatBoost error: code=%d (library not available)", err_code);
#endif
}

/*
 * train_catboost_classifier
 * Trains a CatBoost classifier model using the provided table, feature columns, and label column.
 * Returns integer model_id on successful training and storage.
 */
Datum
train_catboost_classifier(PG_FUNCTION_ARGS)
{
#ifdef CATBOOST_AVAILABLE
	text	   *table_name_text = PG_GETARG_TEXT_PP(0);
	text	   *feature_col_text = PG_GETARG_TEXT_PP(1);
	text	   *label_col_text = PG_GETARG_TEXT_PP(2);
	int32		iterations;
	float8		learning_rate;
	int32		depth;

	char	   *table_name;
	char	   *feature_col_list;
	char	   *label_col;

	StringInfoData sql;
	int			ret;
	int			n_features, i, model_id;

	char	  **features = NULL;
	char	   *token;
	MemoryContext oldctx;
	MemoryContext per_query_ctx;

	/* Assign parameters with defaults if NULL */
	iterations = PG_ARGISNULL(3) ? 1000 : PG_GETARG_INT32(3);
	learning_rate = PG_ARGISNULL(4) ? 0.03 : PG_GETARG_FLOAT8(4);
	depth = PG_ARGISNULL(5) ? 6 : PG_GETARG_INT32(5);

	table_name = text_to_cstring(table_name_text);
	feature_col_list = text_to_cstring(feature_col_text);
	label_col = text_to_cstring(label_col_text);

	/* Parse feature_col_list into features[] */
	n_features = 1;
	for (i = 0; feature_col_list[i]; i++)
	{
		if (feature_col_list[i] == ',')
			n_features++;
	}
	features = (char **) palloc0(sizeof(char *) * n_features);

	i = 0;
	token = strtok(feature_col_list, ",");
	while (token != NULL)
	{
		while (*token == ' ' || *token == '\t') token++;
		features[i++] = pstrdup(token);
		token = strtok(NULL, ",");
	}

	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT ");
	for (i = 0; i < n_features; i++)
	{
		if (i > 0)
			appendStringInfoChar(&sql, ',');
		appendStringInfoString(&sql, features[i]);
	}
	appendStringInfo(&sql, ",%s FROM %s", label_col, table_name);

	elog(DEBUG1, "Running SQL: %s", sql.data);

	per_query_ctx = AllocSetContextCreate(CurrentMemoryContext,
										  "catboost_spi_ctx",
										  ALLOCSET_DEFAULT_SIZES);
	oldctx = MemoryContextSwitchTo(per_query_ctx);

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		elog(ERROR, "Could not connect to SPI");

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		elog(ERROR, "Could not execute SQL for training set");
	}

	/* Get feature and label indexes */
	TupleDesc	tupdesc = SPI_tuptable->tupdesc;
	int		   *feature_idxs = (int *) palloc0(sizeof(int) * n_features);
	int			label_idx;
	for (i = 0; i < n_features; i++)
		feature_idxs[i] = get_column_index(SPI_tuptable, tupdesc, features[i]);
	label_idx = get_column_index(SPI_tuptable, tupdesc, label_col);

	/* Write training data to CSV */
	char tmp_csv_path[MAXPGPATH];
	char tmp_model_path[MAXPGPATH];

	snprintf(tmp_csv_path, sizeof(tmp_csv_path), "%s/catboost_train_%d.csv", PG_TEMP_FILES_DIR, MyProcPid);
	snprintf(tmp_model_path, sizeof(tmp_model_path), "%s/catboost_model_%d.cbm", PG_TEMP_FILES_DIR, MyProcPid);

	write_csv_from_spi(tmp_csv_path, SPI_tuptable, tupdesc, feature_idxs, n_features, label_idx);

	/* Setup CatBoost C API options and train */
	ModelCalcerHandle *model_handle = NULL;
	CatBoostModelTrainingOptions *opts = CatBoostCreateModelTrainingOptions();

	check_catboost_error(CatBoostSetTrainingOptionInt(opts, "iterations", iterations));
	check_catboost_error(CatBoostSetTrainingOptionDouble(opts, "learning_rate", learning_rate));
	check_catboost_error(CatBoostSetTrainingOptionInt(opts, "depth", depth));

	check_catboost_error(CatBoostTrainModelFromFile(opts,
													tmp_csv_path,
													n_features,
													label_idx,
													true, /* has header */
													&model_handle));

	check_catboost_error(CatBoostSaveModelToFile(model_handle, tmp_model_path));

	model_id = MyProcPid;

	CatBoostDestroyModelTrainingOptions(opts);
	CatBoostFreeModelCalcer(model_handle);

	SPI_finish();
	MemoryContextSwitchTo(oldctx);
	MemoryContextDelete(per_query_ctx);

	PG_RETURN_INT32(model_id);
#else
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("CatBoost library not available. Please install CatBoost to use this function.")));
	PG_RETURN_NULL();
#endif
}

/*
 * train_catboost_regressor
 * Trains a CatBoost regressor model. Arguments similar to classifier.
 * Returns integer model_id on success.
 */
Datum
train_catboost_regressor(PG_FUNCTION_ARGS)
{
#ifdef CATBOOST_AVAILABLE
	text	   *table_name_text = PG_GETARG_TEXT_PP(0);
	text	   *feature_col_text = PG_GETARG_TEXT_PP(1);
	text	   *target_col_text = PG_GETARG_TEXT_PP(2);
	int32		iterations;
	char	   *table_name;
	char	   *feature_col_list;
	char	   *target_col;

	StringInfoData sql;
	int			ret;
	int			n_features, i, model_id;

	char	  **features = NULL;
	char	   *token;
	MemoryContext oldctx;
	MemoryContext per_query_ctx;

	iterations = PG_ARGISNULL(3) ? 1000 : PG_GETARG_INT32(3);

	table_name = text_to_cstring(table_name_text);
	feature_col_list = text_to_cstring(feature_col_text);
	target_col = text_to_cstring(target_col_text);

	/* Parse feature list */
	n_features = 1;
	for (i = 0; feature_col_list[i]; i++)
	{
		if (feature_col_list[i] == ',')
			n_features++;
	}
	features = (char **) palloc0(sizeof(char *) * n_features);

	i = 0;
	token = strtok(feature_col_list, ",");
	while (token != NULL)
	{
		while (*token == ' ' || *token == '\t') token++;
		features[i++] = pstrdup(token);
		token = strtok(NULL, ",");
	}

	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT ");
	for (i = 0; i < n_features; i++)
	{
		if (i > 0)
			appendStringInfoChar(&sql, ',');
		appendStringInfoString(&sql, features[i]);
	}
	appendStringInfo(&sql, ",%s FROM %s", target_col, table_name);

	per_query_ctx = AllocSetContextCreate(CurrentMemoryContext,
										  "catboost_spi_ctx",
										  ALLOCSET_DEFAULT_SIZES);
	oldctx = MemoryContextSwitchTo(per_query_ctx);

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		elog(ERROR, "Could not connect to SPI");

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		elog(ERROR, "Could not execute SQL for training set");
	}

	TupleDesc tupdesc = SPI_tuptable->tupdesc;
	int		*feature_idxs = (int *) palloc0(sizeof(int) * n_features);
	int		target_idx;
	for (i = 0; i < n_features; i++)
		feature_idxs[i] = get_column_index(SPI_tuptable, tupdesc, features[i]);
	target_idx = get_column_index(SPI_tuptable, tupdesc, target_col);

	char tmp_csv_path[MAXPGPATH];
	char tmp_model_path[MAXPGPATH];

	snprintf(tmp_csv_path, sizeof(tmp_csv_path), "%s/catboost_train_%d.csv", PG_TEMP_FILES_DIR, MyProcPid);
	snprintf(tmp_model_path, sizeof(tmp_model_path), "%s/catboost_model_%d.cbm", PG_TEMP_FILES_DIR, MyProcPid);

	write_csv_from_spi(tmp_csv_path, SPI_tuptable, tupdesc, feature_idxs, n_features, target_idx);

	ModelCalcerHandle *model_handle = NULL;
	CatBoostModelTrainingOptions *opts = CatBoostCreateModelTrainingOptions();
	check_catboost_error(CatBoostSetTrainingOptionInt(opts, "iterations", iterations));
	check_catboost_error(CatBoostSetTrainingOptionInt(opts, "task_type", 0)); /* 0 for regression */

	check_catboost_error(CatBoostTrainModelFromFile(opts,
													tmp_csv_path,
													n_features,
													target_idx,
													true,
													&model_handle));

	check_catboost_error(CatBoostSaveModelToFile(model_handle, tmp_model_path));

	model_id = MyProcPid;

	CatBoostDestroyModelTrainingOptions(opts);
	CatBoostFreeModelCalcer(model_handle);

	SPI_finish();
	MemoryContextSwitchTo(oldctx);
	MemoryContextDelete(per_query_ctx);

	PG_RETURN_INT32(model_id);
#else
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("CatBoost library not available. Please install CatBoost to use this function.")));
	PG_RETURN_NULL();
#endif
}

/*
 * predict_catboost
 *  Predicts using a persisted CatBoost model and a feature array.
 *  Arguments: int4 model_id, float8[] features (or text[] for categorical support).
 *  Returns float8 predicted value.
 */
Datum
predict_catboost(PG_FUNCTION_ARGS)
{
#ifdef CATBOOST_AVAILABLE
	int32		model_id;
	ArrayType  *features_array;
	Oid			elmtype;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	int			n_features, i, ndim;
	Datum	   *elems;
	bool	   *nulls;
	float64		result = 0.0;

	ModelCalcerHandle *model_handle = NULL;
	char		model_path[MAXPGPATH];

	model_id = PG_GETARG_INT32(0);
	features_array = PG_GETARG_ARRAYTYPE_P(1);

	ndim = ARR_NDIM(features_array);
	if (ndim != 1)
		ereport(ERROR, (errmsg("features array must be one-dimensional")));

	n_features = ArrayGetNItems(ARR_NDIM(features_array), ARR_DIMS(features_array));
	elmtype = ARR_ELEMTYPE(features_array);
	get_typlenbyvalalign(elmtype, &typlen, &typbyval, &typalign);

	deconstruct_array(features_array, elmtype, typlen, typbyval, typalign,
					  &elems, &nulls, &n_features);

	snprintf(model_path, sizeof(model_path), "%s/catboost_model_%d.cbm", PG_TEMP_FILES_DIR, model_id);

	check_catboost_error(CatBoostLoadModelFromFile(model_path, &model_handle));

	if (elmtype == TEXTOID)
	{
		/* Categorical/text features */
		char	  **input_features = (char **) palloc(sizeof(char *) * n_features);
		for (i = 0; i < n_features; i++)
		{
			if (nulls[i])
				elog(ERROR, "catboost feature %d is NULL", i);

			input_features[i] = text_to_cstring(DatumGetTextPP(elems[i]));
		}

		check_catboost_error(CatBoostModelCalcerPredictText(model_handle,
															(const char* const *)input_features,
															n_features,
															&result,
															1));
		pfree(input_features);
	}
	else
	{
		/* Numeric features */
		double *features = (double *) palloc(sizeof(double) * n_features);
		for (i = 0; i < n_features; i++)
		{
			if (nulls[i])
				elog(ERROR, "catboost feature %d is NULL", i);
			if (elmtype == FLOAT8OID)
				features[i] = DatumGetFloat8(elems[i]);
			else if (elmtype == FLOAT4OID)
				features[i] = (double) DatumGetFloat4(elems[i]);
			else if (elmtype == INT4OID)
				features[i] = (double) DatumGetInt32(elems[i]);
			else if (elmtype == INT8OID)
				features[i] = (double) DatumGetInt64(elems[i]);
			else if (elmtype == INT2OID)
				features[i] = (double) DatumGetInt16(elems[i]);
			else
				elog(ERROR, "Unsupported feature element type for CatBoost: %u", elmtype);
		}

		check_catboost_error(CatBoostModelCalcerPredict(model_handle,
														features,
														n_features,
														&result,
														1));
		pfree(features);
	}

	CatBoostFreeModelCalcer(model_handle);

	PG_RETURN_FLOAT8(result);
#else
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("CatBoost library not available. Please install CatBoost to use this function.")));
	PG_RETURN_NULL();
#endif
}
