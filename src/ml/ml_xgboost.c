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
load_training_data(const char *table,
	const char *feature_col,
	const char *label_col,
	float **out_features,
	float **out_labels,
	int *out_nrows,
	int *out_ncols)
{
	int ret;
	int i;
	int j;
	int nrows;
	int ncols;
	StringInfoData query;
	float *features = NULL;
	float *labels = NULL;
	TupleDesc tupdesc;
	HeapTuple tuple;
	bool isnull;
	Datum feat_datum;
	ArrayType *feat_arr = NULL;

	initStringInfo(&query);

	/* Construct query to select feature and label columns */
	appendStringInfo(
		&query, "SELECT %s, %s FROM %s", feature_col, label_col, table);

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

	/* Defensive: Validate SPI_tuptable */
	if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL
		|| SPI_tuptable->vals == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_tuptable is invalid")));

	feat_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
	if (isnull)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("Null feature vector found")));

	if (tupdesc->attrs[0]->atttypid == FLOAT4ARRAYOID
		|| tupdesc->attrs[0]->atttypid == FLOAT8ARRAYOID)
	{
		feat_arr = DatumGetArrayTypeP(feat_datum);
		ncols = ArrayGetNItems(ARR_NDIM(feat_arr), ARR_DIMS(feat_arr));
	} else
	{
		ncols = 1;
	}

	/* Defensive: Validate dimensions */
	if (ncols <= 0 || ncols > 100000 || nrows > 10000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid training data dimensions: %d rows, %d cols",
					nrows, ncols)));

	features = (float *)palloc(sizeof(float) * nrows * ncols);
	labels = (float *)palloc(sizeof(float) * nrows);

	/* Defensive: Validate allocations */
	if (features == NULL || labels == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate training data arrays")));

	for (i = 0; i < nrows; i++)
	{
		HeapTuple current_tuple;
		bool isnull_feat;
		bool isnull_label;
		Datum featval;
		Datum labelval;

		current_tuple = SPI_tuptable->vals[i];

		/* Features */
		/* Defensive: Validate tuple */
		if (current_tuple == NULL)
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("NULL tuple at row %d", i)));

		featval =
			SPI_getbinval(current_tuple, tupdesc, 1, &isnull_feat);
		if (isnull_feat)
			ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					errmsg("Null feature vector in row %d", i)));

		if (feat_arr)
		{
			ArrayType *curr_arr;
			int arr_len;
			float8 *fdat;

			curr_arr = DatumGetArrayTypeP(featval);

			/* Defensive: Check NULL array */
			if (curr_arr == NULL)
				ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
						errmsg("Null feature array in row %d", i)));

			if (ARR_NDIM(curr_arr) == 1)
			{
				arr_len = ArrayGetNItems(
					ARR_NDIM(curr_arr), ARR_DIMS(curr_arr));
				if (arr_len != ncols)
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("Unexpected dimension of feature array: expected %d, got %d",
								ncols, arr_len)));
				fdat = (float8 *)ARR_DATA_PTR(curr_arr);

				/* Defensive: Validate pointer */
				if (fdat == NULL)
					ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
							errmsg("NULL array data pointer in row %d", i)));

				for (j = 0; j < ncols; j++)
				{
					/* Defensive: Check for NaN/Inf */
					if (isnan(fdat[j]) || isinf(fdat[j]))
						ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								errmsg("Feature contains NaN or Infinity in row %d, col %d",
									i, j)));
					features[i * ncols + j] = (float)fdat[j];
				}
			} else
			{
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("Feature arrays must be 1D")));
			}
		} else
		{
			float8 feat_val;

			if (tupdesc->attrs[0]->atttypid == FLOAT8OID)
				feat_val = DatumGetFloat8(featval);
			else if (tupdesc->attrs[0]->atttypid == FLOAT4OID)
				feat_val = (float8)DatumGetFloat4(featval);
			else
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("Unsupported feature column type")));

			/* Defensive: Check for NaN/Inf */
			if (isnan(feat_val) || isinf(feat_val))
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("Feature contains NaN or Infinity in row %d", i)));

			features[i * ncols] = (float)feat_val;
		}

		/* Labels */
		labelval =
			SPI_getbinval(current_tuple, tupdesc, 2, &isnull_label);
		if (isnull_label)
			elog(ERROR, "Null label/target in row %d", i);

		if (tupdesc->attrs[1]->atttypid == INT4OID)
		{
			labels[i] = (float)DatumGetInt32(labelval);
		}
		else if (tupdesc->attrs[1]->atttypid == FLOAT4OID)
		{
			float4 label_val = DatumGetFloat4(labelval);

			/* Defensive: Check for NaN/Inf */
			if (isnan(label_val) || isinf(label_val))
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("Label contains NaN or Infinity in row %d", i)));

			labels[i] = (float)label_val;
		}
		else if (tupdesc->attrs[1]->atttypid == FLOAT8OID)
		{
			float8 label_val = DatumGetFloat8(labelval);

			/* Defensive: Check for NaN/Inf */
			if (isnan(label_val) || isinf(label_val))
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("Label contains NaN or Infinity in row %d", i)));

			labels[i] = (float)label_val;
		}
		else
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("Unsupported label/target column type")));
		}
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
	int ret;
	int32 model_id = 0;
	Oid argtypes[2] = { BYTEAOID, TEXTOID };
	Datum values[2];
	char nulls[2] = { ' ', ' ' };
	char *insert_cmd = "INSERT INTO ml_models(model, provider) VALUES ($1, "
			   "$2) RETURNING id";

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed for model insert");

	values[0] = PointerGetDatum(
		cstring_to_bytea((const char *)model_bytes, model_len));
	values[1] = CStringGetTextDatum("xgboost");

	ret = SPI_execute_with_args(
		insert_cmd, 2, argtypes, values, nulls, false, 1);
	if (ret != SPI_OK_INSERT_RETURNING)
		elog(ERROR,
			"SPI_execute_with_args failed to insert XGBoost model");

	if (SPI_processed > 0)
	{
		HeapTuple tup = SPI_tuptable->vals[0];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		bool isnull;
		Datum iddat;

		iddat = SPI_getbinval(tup, tupdesc, 1, &isnull);
		if (isnull)
			elog(ERROR, "Null model ID returned");
		model_id = DatumGetInt32(iddat);
	} else
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
	int ret;
	char select_cmd[256];
	HeapTuple tup;
	TupleDesc tupdesc;
	bool isnull;
	Datum modeldat;
	bytea *model_bytea;
	size_t len;
	void *data;

	snprintf(select_cmd,
		sizeof(select_cmd),
		"SELECT model FROM ml_models WHERE id = %d",
		model_id);

	/* Defensive: Validate model_id */
	if (model_id <= 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("model_id must be positive, got %d", model_id)));

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_connect failed for model fetch")));

	ret = SPI_execute(select_cmd, true, 1);

	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_execute failed for model select")));

	if (SPI_processed == 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Model with id %d not found in ml_models",
					model_id)));

	/* Defensive: Validate SPI_tuptable */
	if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL
		|| SPI_tuptable->vals == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_tuptable is invalid")));

	tup = SPI_tuptable->vals[0];
	tupdesc = SPI_tuptable->tupdesc;

	modeldat = SPI_getbinval(tup, tupdesc, 1, &isnull);
	if (isnull)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("Null model returned")));

	model_bytea = DatumGetByteaP(modeldat);

	/* Defensive: Validate bytea */
	if (model_bytea == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("Invalid model bytea")));

	len = VARSIZE(model_bytea) - VARHDRSZ;

	/* Defensive: Validate size */
	if (len <= 0 || len > 100000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Invalid model size: %zu", len)));

	data = palloc(len);

	/* Defensive: Validate allocation */
	if (data == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate model buffer")));

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
	text *table_name;
	text *feature_col;
	text *label_col;
	int32 n_estimators;
	int32 max_depth;
	float8 learning_rate;
	char *table_str;
	char *feature_str;
	char *label_str;
	float *features = NULL;
	float *labels = NULL;
	int nrows = 0;
	int ncols = 0;
	DMatrixHandle dtrain = NULL;
	BoosterHandle booster = NULL;
	char num_class_str[16];
	char eta_str[32];
	char md_str[16];
	const char *keys[6];
	const char *vals[6];
	int param_count = 6;
	int i;
	int iter;
	float max_label = 0.0f;
	int num_class;
	bst_ulong out_len = 0;
	char *out_bytes = NULL;
	int32 model_id;

	CHECK_NARGS_RANGE(3, 6);
	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);
	n_estimators = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);
	max_depth = PG_ARGISNULL(4) ? 6 : PG_GETARG_INT32(4);
	learning_rate = PG_ARGISNULL(5) ? 0.3 : PG_GETARG_FLOAT8(5);

	/* Defensive: Check NULL inputs */
	if (table_name == NULL || feature_col == NULL || label_col == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("table_name, feature_col, and label_col cannot be NULL")));

	/* Defensive: Validate parameters */
	if (n_estimators <= 0 || n_estimators > 10000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("n_estimators must be in range [1, 10000], got %d",
					n_estimators)));
	if (max_depth <= 0 || max_depth > 100)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("max_depth must be in range [1, 100], got %d",
					max_depth)));
	if (learning_rate <= 0.0 || learning_rate > 10.0 || isnan(learning_rate)
		|| isinf(learning_rate))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("learning_rate must be in range (0, 10], got %f",
					learning_rate)));

	table_str = text_to_cstring(table_name);
	feature_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	/* Defensive: Validate allocations */
	if (table_str == NULL || feature_str == NULL || label_str == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate strings")));

	load_training_data(table_str,
		feature_str,
		label_str,
		&features,
		&labels,
		&nrows,
		&ncols);

	/* Defensive: Validate training data */
	if (features == NULL || labels == NULL)
	{
		if (table_str)
			pfree(table_str);
		if (feature_str)
			pfree(feature_str);
		if (label_str)
			pfree(label_str);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("failed to load training data")));
	}
	if (nrows <= 0 || ncols <= 0)
	{
		if (features)
			pfree(features);
		if (labels)
			pfree(labels);
		if (table_str)
			pfree(table_str);
		if (feature_str)
			pfree(feature_str);
		if (label_str)
			pfree(label_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("training data must have at least one row and column")));
	}

	if (XGDMatrixCreateFromMat(features, nrows, ncols, (float)NAN, &dtrain)
		!= 0)
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
	num_class = (int)max_label + 1;

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
			elog(ERROR,
				"Failed to set XGBoost booster parameter: %s",
				keys[i]);
	}

	for (iter = 0; iter < n_estimators; iter++)
	{
		if (XGBoosterUpdateOneIter(booster, iter, dtrain) != 0)
			elog(ERROR,
				"Failed during XGBoost training iteration %d",
				iter);
	}

	if (XGBoosterSaveModelToBuffer(
		    booster, &out_len, (const char **)&out_bytes)
		!= 0)
		elog(ERROR, "Failed to serialize XGBoost model");

	model_id = store_xgboost_model(out_bytes, out_len);

	(void)XGBoosterFree(booster);
	(void)XGDMatrixFree(dtrain);
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
	text *table_name;
	text *feature_col;
	text *target_col;
	int32 n_estimators;
	int32 max_depth;
	float8 learning_rate;
	char *table_str;
	char *feature_str;
	char *target_str;
	float *features = NULL;
	float *labels = NULL;
	int nrows = 0;
	int ncols = 0;
	DMatrixHandle dtrain = NULL;
	BoosterHandle booster = NULL;
	char eta_str[32];
	char md_str[16];
	const char *keys[5];
	const char *vals[5];
	int param_count = 5;
	int i;
	int iter;
	bst_ulong out_len = 0;
	char *out_bytes = NULL;
	int32 model_id;

	CHECK_NARGS_RANGE(3, 6);
	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);
	n_estimators = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);
	max_depth = PG_ARGISNULL(4) ? 6 : PG_GETARG_INT32(4);
	learning_rate = PG_ARGISNULL(5) ? 0.3 : PG_GETARG_FLOAT8(5);

	/* Defensive: Check NULL inputs */
	if (table_name == NULL || feature_col == NULL || target_col == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("table_name, feature_col, and target_col cannot be NULL")));

	/* Defensive: Validate parameters */
	if (n_estimators <= 0 || n_estimators > 10000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("n_estimators must be in range [1, 10000], got %d",
					n_estimators)));
	if (max_depth <= 0 || max_depth > 100)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("max_depth must be in range [1, 100], got %d",
					max_depth)));
	if (learning_rate <= 0.0 || learning_rate > 10.0 || isnan(learning_rate)
		|| isinf(learning_rate))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("learning_rate must be in range (0, 10], got %f",
					learning_rate)));

	table_str = text_to_cstring(table_name);
	feature_str = text_to_cstring(feature_col);
	target_str = text_to_cstring(target_col);

	/* Defensive: Validate allocations */
	if (table_str == NULL || feature_str == NULL || target_str == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate strings")));

	float *features = NULL;
	float *labels = NULL;
	int nrows = 0;
	int ncols = 0;
	DMatrixHandle dtrain = NULL;
	BoosterHandle booster = NULL;
	char eta_str[32];
	char md_str[16];
	const char *keys[5];
	const char *vals[5];
	int param_count = 5;
	int i, iter;
	bst_ulong out_len = 0;
	char *out_bytes = NULL;
	int32 model_id;

	elog(NOTICE,
		"XGBoost Regressor: table=%s, feature=%s, target=%s, "
		"n_estimators=%d, max_depth=%d, learning_rate=%.3f",
		table_str,
		feature_str,
		target_str,
		n_estimators,
		max_depth,
		learning_rate);

	load_training_data(table_str,
		feature_str,
		target_str,
		&features,
		&labels,
		&nrows,
		&ncols);

	if (XGDMatrixCreateFromMat(features, nrows, ncols, (float)NAN, &dtrain)
		!= 0)
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
			elog(ERROR,
				"Failed to set XGBoost booster parameter: %s",
				keys[i]);
	}
	for (iter = 0; iter < n_estimators; iter++)
	{
		if (XGBoosterUpdateOneIter(booster, iter, dtrain) != 0)
			elog(ERROR,
				"Failed during XGBoost training iteration %d",
				iter);
	}

	if (XGBoosterSaveModelToBuffer(
		    booster, &out_len, (const char **)&out_bytes)
		!= 0)
		elog(ERROR, "Failed to serialize XGBoost model");

	model_id = store_xgboost_model(out_bytes, out_len);

	(void)XGBoosterFree(booster);
	(void)XGDMatrixFree(dtrain);
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
	int32 model_id;
	ArrayType *features_array;
	int n_dims;
	float8 *features = NULL;
	float *feat_f = NULL;
	DMatrixHandle dmat = NULL;
	BoosterHandle booster = NULL;
	size_t model_size;
	void *mod_bytes = NULL;
	bst_ulong out_len = 0;
	const float *out_result = NULL;
	int i;
	float8 pred;

	CHECK_NARGS(2);
	model_id = PG_GETARG_INT32(0);
	features_array = PG_GETARG_ARRAYTYPE_P(1);

	/* Defensive: Check NULL input */
	if (features_array == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("features array cannot be NULL")));

	/* Defensive: Validate model_id */
	if (model_id <= 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("model_id must be positive, got %d", model_id)));

	if (ARR_NDIM(features_array) != 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("features must be a 1-dimensional array")));

	n_dims = (int)ArrayGetNItems(
		ARR_NDIM(features_array), ARR_DIMS(features_array));

	/* Defensive: Validate dimensions */
	if (n_dims <= 0 || n_dims > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("feature dimension must be in range [1, 100000], got %d",
					n_dims)));

	features = (float8 *)ARR_DATA_PTR(features_array);

	feat_f = (float *)palloc(sizeof(float) * n_dims);

	/* Defensive: Validate allocation */
	if (feat_f == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate feature array")));

	for (i = 0; i < n_dims; i++)
	{
		/* Defensive: Check for NaN/Inf */
		if (isnan(features[i]) || isinf(features[i]))
		{
			pfree(feat_f);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("features cannot contain NaN or Infinity")));
		}
		feat_f[i] = (float)features[i];
	}

	mod_bytes = fetch_xgboost_model(model_id, &model_size);

	if (XGBoosterCreate(NULL, 0, &booster) != 0)
		elog(ERROR, "Failed to create XGBoost booster");

	if (XGBoosterLoadModelFromBuffer(booster, mod_bytes, model_size) != 0)
		elog(ERROR, "Failed to load XGBoost model from buffer");

	if (XGDMatrixCreateFromMat(feat_f, 1, n_dims, (float)NAN, &dmat) != 0)
		elog(ERROR, "Failed to create DMatrix for prediction");

	if (XGBoosterPredict(booster, dmat, 0, 0, 0, &out_len, &out_result)
		!= 0)
		elog(ERROR, "XGBoost prediction failed");

	pred = (out_len > 0) ? (float8)out_result[0] : 0.0;

	(void)XGBoosterFree(booster);
	(void)XGDMatrixFree(dmat);
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
			errhint("Install libxgboost and recompile NeuronDB to "
				"enable XGBoost support.")));
	PG_RETURN_INT32(-1);
}

Datum
train_xgboost_regressor(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("XGBoost is not available"),
			errhint("Install libxgboost and recompile NeuronDB to "
				"enable XGBoost support.")));
	PG_RETURN_INT32(-1);
}

Datum
predict_xgboost(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("XGBoost is not available"),
			errhint("Install libxgboost and recompile NeuronDB to "
				"enable XGBoost support.")));
	PG_RETURN_FLOAT8(0.0);
}

#endif /* HAVE_XGBOOST */

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for XGBoost
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"

void
neurondb_gpu_register_xgboost_model(void)
{
	elog(DEBUG1, "XGBoost GPU Model Ops registration skipped - not yet implemented");
}
