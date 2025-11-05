/*-------------------------------------------------------------------------
 *
 * ml_xgboost.c
 *	  XGBoost Integration for NeuronDB
 *
 * Provides XGBoost gradient boosting for classification and regression.
 * Requires XGBoost C library (libxgboost.so).
 *
 * Conditional compilation: if XGBoost headers not found, builds stub
 * implementations that error gracefully.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *	  src/ml/ml_xgboost.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/memutils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Check if XGBoost C API is available */
#if __has_include(<xgboost/c_api.h>)
#include <xgboost/c_api.h>
#define HAVE_XGBOOST 1
#else
#define HAVE_XGBOOST 0
#endif

PG_FUNCTION_INFO_V1(train_xgboost_classifier);
PG_FUNCTION_INFO_V1(train_xgboost_regressor);
PG_FUNCTION_INFO_V1(predict_xgboost);

#if HAVE_XGBOOST

/*
 * Load feature matrix and label vector from table using SPI.
 */
static void
load_training_data(const char *table, const char *feature_col, const char *label_col,
				   float **out_features, float **out_labels,
				   int *out_nrows, int *out_ncols)
{
	int				ret;
	int				i;
	int				j;
	int				nrows;
	int				ncols;
	StringInfoData	query;
	float		   *features = NULL;
	float		   *labels = NULL;
	TupleDesc		tupdesc;
	HeapTuple		tuple;
	bool			isnull;
	Datum			feat_datum;
	ArrayType	   *feat_arr = NULL;

	initStringInfo(&query);

	/* Construct query to select feature and label columns */
	appendStringInfo(&query, "SELECT %s, %s FROM %s", feature_col, label_col, table);

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	ret = SPI_execute(query.data, true, 0);

	if (ret != SPI_OK_SELECT)
		elog(ERROR, "SPI_execute failed for training data");

	nrows = SPI_processed;
	if (nrows <= 0)
		elog(ERROR, "No training rows found");

	/*
	 * Determine dimension (features can be a vector column).
	 * We expect features to be either PostgreSQL array or a single float column.
	 * We'll support 1-D and N-D, check first row.
	 */
	tupdesc = SPI_tuptable->tupdesc;
	tuple = SPI_tuptable->vals[0];

	feat_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
	if (isnull)
		elog(ERROR, "Null feature vector found");

	if (tupdesc->attrs[0]->atttypid == FLOAT4ARRAYOID ||
		tupdesc->attrs[0]->atttypid == FLOAT8ARRAYOID)
	{
		feat_arr = DatumGetArrayTypeP(feat_datum);
		ncols = ArrayGetNItems(ARR_NDIM(feat_arr), ARR_DIMS(feat_arr));
	}
	else
	{
		ncols = 1;
	}

	features = (float *) palloc(sizeof(float) * nrows * ncols);
	labels = (float *) palloc(sizeof(float) * nrows);

	for (i = 0; i < nrows; i++)
	{
		HeapTuple	current_tuple;
		bool		isnull_feat;
		bool		isnull_label;
		Datum		featval;
		Datum		labelval;

		current_tuple = SPI_tuptable->vals[i];

		/* Features */
		featval = SPI_getbinval(current_tuple, tupdesc, 1, &isnull_feat);
		if (isnull_feat)
			elog(ERROR, "Null feature vector in row %d", i);

		if (feat_arr)
		{
			ArrayType  *curr_arr;
			int			arr_len;
			float8	   *fdat;

			curr_arr = DatumGetArrayTypeP(featval);

			if (ARR_NDIM(curr_arr) == 1)
			{
				arr_len = ArrayGetNItems(ARR_NDIM(curr_arr), ARR_DIMS(curr_arr));
				if (arr_len != ncols)
					elog(ERROR, "Unexpected dimension of feature array");
				fdat = (float8 *) ARR_DATA_PTR(curr_arr);
				for (j = 0; j < ncols; j++)
					features[i * ncols + j] = (float) fdat[j];
			}
			else
			{
				elog(ERROR, "Feature arrays must be 1D");
			}
		}
		else
		{
			if (tupdesc->attrs[0]->atttypid == FLOAT8OID)
				features[i * ncols] = (float) DatumGetFloat8(featval);
			else if (tupdesc->attrs[0]->atttypid == FLOAT4OID)
				features[i * ncols] = (float) DatumGetFloat4(featval);
			else
				elog(ERROR, "Unsupported feature column type");
		}

		/* Labels */
		labelval = SPI_getbinval(current_tuple, tupdesc, 2, &isnull_label);
		if (isnull_label)
			elog(ERROR, "Null label/target in row %d", i);

		if (tupdesc->attrs[1]->atttypid == INT4OID)
			labels[i] = (float) DatumGetInt32(labelval);
		else if (tupdesc->attrs[1]->atttypid == FLOAT4OID)
			labels[i] = (float) DatumGetFloat4(labelval);
		else if (tupdesc->attrs[1]->atttypid == FLOAT8OID)
			labels[i] = (float) DatumGetFloat8(labelval);
		else
			elog(ERROR, "Unsupported label/target column type");
	}

	*out_features = features;
	*out_labels = labels;
	*out_nrows = nrows;
	*out_ncols = ncols;

	SPI_finish();
}

/*
 * Save XGBoost model binary to ml_models.
 */
static int32
store_xgboost_model(const void *model_bytes, size_t model_len)
{
	int				ret;
	int32			model_id = 0;
	Oid				argtypes[2] = {BYTEAOID, TEXTOID};
	Datum			values[2];
	char			nulls[2] = {' ', ' '};
	char		   *insert_cmd =
						"INSERT INTO ml_models(model, provider) VALUES ($1, $2) RETURNING id";

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed for model insert");

	values[0] = PointerGetDatum(cstring_to_bytea((const char *) model_bytes, model_len));
	values[1] = CStringGetTextDatum("xgboost");

	ret = SPI_execute_with_args(insert_cmd, 2, argtypes, values, nulls, false, 1);
	if (ret != SPI_OK_INSERT_RETURNING)
		elog(ERROR, "SPI_execute_with_args failed to insert XGBoost model");

	if (SPI_processed > 0)
	{
		HeapTuple	tup = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		bool		isnull;
		Datum		iddat;

		iddat = SPI_getbinval(tup, tupdesc, 1, &isnull);
		if (isnull)
			elog(ERROR, "Null model ID returned");
		model_id = DatumGetInt32(iddat);
	}
	else
	{
		elog(ERROR, "No model id returned from insert");
	}

	SPI_finish();

	return model_id;
}

/*
 * Retrieve XGBoost model binary from ml_models.
 */
static void *
fetch_xgboost_model(int32 model_id, size_t *model_size)
{
	int				ret;
	char			select_cmd[256];
	HeapTuple		tup;
	TupleDesc		tupdesc;
	bool			isnull;
	Datum			modeldat;
	bytea		   *model_bytea;
	size_t			len;
	void		   *data;

	snprintf(select_cmd, sizeof(select_cmd),
			 "SELECT model FROM ml_models WHERE id = %d", model_id);

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed for model fetch");

	ret = SPI_execute(select_cmd, true, 1);

	if (ret != SPI_OK_SELECT)
		elog(ERROR, "SPI_execute failed for model select");

	if (SPI_processed == 0)
		elog(ERROR, "Model with id %d not found in ml_models", model_id);

	tup = SPI_tuptable->vals[0];
	tupdesc = SPI_tuptable->tupdesc;

	modeldat = SPI_getbinval(tup, tupdesc, 1, &isnull);
	if (isnull)
		elog(ERROR, "Null model returned");

	model_bytea = DatumGetByteaP(modeldat);
	len = VARSIZE(model_bytea) - VARHDRSZ;
	data = palloc(len);
	memcpy(data, VARDATA(model_bytea), len);

	*model_size = len;

	SPI_finish();
	return data;
}

/*
 * train_xgboost_classifier() - Train XGBoost classifier
 * Parameters:
 *   table_name TEXT - Training data table
 *   feature_col TEXT - Feature column name
 *   label_col TEXT - Label column name
 *   n_estimators INTEGER - Number of trees (default 100)
 *   max_depth INTEGER - Maximum tree depth (default 6)
 *   learning_rate FLOAT - Learning rate (default 0.3)
 * Returns: model_id INTEGER (stored in ml_models)
 */
Datum
train_xgboost_classifier(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *feature_col = PG_GETARG_TEXT_PP(1);
	text	   *label_col = PG_GETARG_TEXT_PP(2);
	int32		n_estimators = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);
	int32		max_depth = PG_ARGISNULL(4) ? 6 : PG_GETARG_INT32(4);
	float8		learning_rate = PG_ARGISNULL(5) ? 0.3 : PG_GETARG_FLOAT8(5);

	char	   *table_str = text_to_cstring(table_name);
	char	   *feature_str = text_to_cstring(feature_col);
	char	   *label_str = text_to_cstring(label_col);

	float	   *features = NULL;
	float	   *labels = NULL;
	int			nrows = 0;
	int			ncols = 0;
	DMatrixHandle dtrain = NULL;
	BoosterHandle booster = NULL;
	char		num_class_str[16];
	char		eta_str[32];
	char		md_str[16];
	const char *keys[6];
	const char *vals[6];
	int			param_count = 6;
	int			i, iter;
	float		max_label = 0.0f;
	int			num_class;
	bst_ulong	out_len = 0;
	char	   *out_bytes = NULL;
	int32		model_id;

	elog(NOTICE, "XGBoost Classifier: table=%s, feature=%s, label=%s, n_estimators=%d, max_depth=%d, learning_rate=%.3f",
		 table_str, feature_str, label_str, n_estimators, max_depth, learning_rate);

	load_training_data(table_str, feature_str, label_str,
					  &features, &labels, &nrows, &ncols);

	if (XGDMatrixCreateFromMat(features, nrows, ncols, (float) NAN, &dtrain) != 0)
		elog(ERROR, "Failed to create DMatrix");

	if (XGDMatrixSetFloatInfo(dtrain, "label", labels, nrows) != 0)
		elog(ERROR, "Failed to set DMatrix labels");

	/* Determine number of classes as max(label) + 1 */
	max_label = labels[0];
	for (i = 1; i < nrows; i++)
	{
		if (labels[i] > max_label)
			max_label = labels[i];
	}
	num_class = (int) max_label + 1;

	snprintf(num_class_str, sizeof(num_class_str), "%d", num_class);
	snprintf(eta_str, sizeof(eta_str), "%f", learning_rate);
	snprintf(md_str, sizeof(md_str), "%d", max_depth);

	keys[0] = "objective";
	vals[0] = "multi:softmax";
	keys[1] = "num_class";
	vals[1] = num_class_str;
	keys[2] = "booster";
	vals[2] = "gbtree";
	keys[3] = "eta";
	vals[3] = eta_str;
	keys[4] = "max_depth";
	vals[4] = md_str;
	keys[5] = "verbosity";
	vals[5] = "1";

	if (XGBoosterCreate(&dtrain, 1, &booster) != 0)
		elog(ERROR, "Failed to create XGBoost booster");
	for (i = 0; i < param_count; i++)
	{
		if (XGBoosterSetParam(booster, keys[i], vals[i]) != 0)
			elog(ERROR, "Failed to set XGBoost booster parameter: %s", keys[i]);
	}

	for (iter = 0; iter < n_estimators; iter++)
	{
		if (XGBoosterUpdateOneIter(booster, iter, dtrain) != 0)
			elog(ERROR, "Failed during XGBoost training iteration %d", iter);
	}

	if (XGBoosterSaveModelToBuffer(booster, &out_len, (const char **) &out_bytes) != 0)
		elog(ERROR, "Failed to serialize XGBoost model");

	model_id = store_xgboost_model(out_bytes, out_len);

	(void) XGBoosterFree(booster);
	(void) XGDMatrixFree(dtrain);
	pfree(features);
	pfree(labels);

	PG_RETURN_INT32(model_id);
}

/*
 * train_xgboost_regressor() - Train XGBoost regressor
 * Parameters:
 *   table_name TEXT - Training data table
 *   feature_col TEXT - Feature column name
 *   target_col TEXT - Regression target column name
 *   n_estimators INTEGER - #trees (default 100)
 *   max_depth INTEGER
 *   learning_rate FLOAT
 * Returns: model_id INTEGER (stored in ml_models)
 */
Datum
train_xgboost_regressor(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *feature_col = PG_GETARG_TEXT_PP(1);
	text	   *target_col = PG_GETARG_TEXT_PP(2);
	int32		n_estimators = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);
	int32		max_depth = PG_ARGISNULL(4) ? 6 : PG_GETARG_INT32(4);
	float8		learning_rate = PG_ARGISNULL(5) ? 0.3 : PG_GETARG_FLOAT8(5);

	char	   *table_str = text_to_cstring(table_name);
	char	   *feature_str = text_to_cstring(feature_col);
	char	   *target_str = text_to_cstring(target_col);

	float	   *features = NULL;
	float	   *labels = NULL;
	int			nrows = 0;
	int			ncols = 0;
	DMatrixHandle dtrain = NULL;
	BoosterHandle booster = NULL;
	char		eta_str[32];
	char		md_str[16];
	const char *keys[5];
	const char *vals[5];
	int			param_count = 5;
	int			i, iter;
	bst_ulong	out_len = 0;
	char	   *out_bytes = NULL;
	int32		model_id;

	elog(NOTICE, "XGBoost Regressor: table=%s, feature=%s, target=%s, n_estimators=%d, max_depth=%d, learning_rate=%.3f",
		 table_str, feature_str, target_str, n_estimators, max_depth, learning_rate);

	load_training_data(table_str, feature_str, target_str,
					  &features, &labels, &nrows, &ncols);

	if (XGDMatrixCreateFromMat(features, nrows, ncols, (float) NAN, &dtrain) != 0)
		elog(ERROR, "Failed to create DMatrix");

	if (XGDMatrixSetFloatInfo(dtrain, "label", labels, nrows) != 0)
		elog(ERROR, "Failed to set DMatrix regression targets");

	snprintf(eta_str, sizeof(eta_str), "%f", learning_rate);
	snprintf(md_str, sizeof(md_str), "%d", max_depth);

	keys[0] = "objective";
	vals[0] = "reg:squarederror";
	keys[1] = "booster";
	vals[1] = "gbtree";
	keys[2] = "eta";
	vals[2] = eta_str;
	keys[3] = "max_depth";
	vals[3] = md_str;
	keys[4] = "verbosity";
	vals[4] = "1";

	if (XGBoosterCreate(&dtrain, 1, &booster) != 0)
		elog(ERROR, "Failed to create XGBoost booster");
	for (i = 0; i < param_count; i++)
	{
		if (XGBoosterSetParam(booster, keys[i], vals[i]) != 0)
			elog(ERROR, "Failed to set XGBoost booster parameter: %s", keys[i]);
	}
	for (iter = 0; iter < n_estimators; iter++)
	{
		if (XGBoosterUpdateOneIter(booster, iter, dtrain) != 0)
			elog(ERROR, "Failed during XGBoost training iteration %d", iter);
	}

	if (XGBoosterSaveModelToBuffer(booster, &out_len, (const char **) &out_bytes) != 0)
		elog(ERROR, "Failed to serialize XGBoost model");

	model_id = store_xgboost_model(out_bytes, out_len);

	(void) XGBoosterFree(booster);
	(void) XGDMatrixFree(dtrain);
	pfree(features);
	pfree(labels);

	PG_RETURN_INT32(model_id);
}

/*
 * predict_xgboost() - Predict with XGBoost model
 * Arguments:
 *   model_id INT
 *   features FLOAT8[]
 * Returns:
 *   prediction (FLOAT8)
 */
Datum
predict_xgboost(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	ArrayType  *features_array = PG_GETARG_ARRAYTYPE_P(1);
	int			n_dims;
	float8	   *features = NULL;
	float	   *feat_f = NULL;
	DMatrixHandle dmat = NULL;
	BoosterHandle booster = NULL;
	size_t		model_size;
	void	   *mod_bytes = NULL;
	bst_ulong	out_len = 0;
	const float *out_result = NULL;
	int			i;
	float8		pred;

	if (ARR_NDIM(features_array) != 1)
		elog(ERROR, "features must be a 1-dimensional array");

	n_dims = (int) ArrayGetNItems(ARR_NDIM(features_array), ARR_DIMS(features_array));
	features = (float8 *) ARR_DATA_PTR(features_array);

	feat_f = (float *) palloc(sizeof(float) * n_dims);
	for (i = 0; i < n_dims; i++)
		feat_f[i] = (float) features[i];

	mod_bytes = fetch_xgboost_model(model_id, &model_size);

	if (XGBoosterCreate(NULL, 0, &booster) != 0)
		elog(ERROR, "Failed to create XGBoost booster");

	if (XGBoosterLoadModelFromBuffer(booster, mod_bytes, model_size) != 0)
		elog(ERROR, "Failed to load XGBoost model from buffer");

	if (XGDMatrixCreateFromMat(feat_f, 1, n_dims, (float) NAN, &dmat) != 0)
		elog(ERROR, "Failed to create DMatrix for prediction");

	if (XGBoosterPredict(booster, dmat, 0, 0, 0, &out_len, &out_result) != 0)
		elog(ERROR, "XGBoost prediction failed");

	pred = (out_len > 0) ? (float8) out_result[0] : 0.0;

	(void) XGBoosterFree(booster);
	(void) XGDMatrixFree(dmat);
	pfree(feat_f);
	pfree(mod_bytes);

	PG_RETURN_FLOAT8(pred);
}

#else /* !HAVE_XGBOOST */

/* Stub implementations when XGBoost library is not available */

Datum
train_xgboost_classifier(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("XGBoost is not available"),
			 errhint("Install libxgboost and recompile NeuronDB to enable XGBoost support.")));
	PG_RETURN_INT32(-1);
}

Datum
train_xgboost_regressor(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("XGBoost is not available"),
			 errhint("Install libxgboost and recompile NeuronDB to enable XGBoost support.")));
	PG_RETURN_INT32(-1);
}

Datum
predict_xgboost(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("XGBoost is not available"),
			 errhint("Install libxgboost and recompile NeuronDB to enable XGBoost support.")));
	PG_RETURN_FLOAT8(0.0);
}

#endif /* HAVE_XGBOOST */
