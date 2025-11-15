/*-------------------------------------------------------------------------
 *
 * ml_knn.c
 *    K-Nearest Neighbors (KNN) implementation for classification and regression
 *
 * Implements KNN using Euclidean distance for vector similarity.
 * Supports both classification (voting) and regression (averaging).
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_knn.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/memutils.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_catalog.h"
#include "neurondb_cuda_knn.h"
#include "vector/vector_types.h"

#include <math.h>
#include <float.h>

/* Structure to hold a training sample with distance */
typedef struct
{
	float *features;
	double label;
	double distance;
	int dim;
} KNNSample;

/* Comparison function for qsort */
static int
compare_samples(const void *a, const void *b)
{
	const KNNSample *sa = (const KNNSample *)a;
	const KNNSample *sb = (const KNNSample *)b;

	if (sa->distance < sb->distance)
		return -1;
	if (sa->distance > sb->distance)
		return 1;
	return 0;
}

/* Compute Euclidean distance between two vectors */
static double
euclidean_distance(float *a, float *b, int dim)
{
	double sum = 0.0;
	int i;

	/* Assert: Internal invariants */
	Assert(a != NULL);
	Assert(b != NULL);
	Assert(dim > 0);

	/* Defensive: Check for NULL pointers */
	if (a == NULL || b == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("euclidean_distance: NULL vector pointer")));

	if (dim <= 0 || dim > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("euclidean_distance: invalid dimension: %d", dim)));

	for (i = 0; i < dim; i++)
	{
		/* Defensive: Check for NaN/Inf */
		if (isnan(a[i]) || isnan(b[i]) || isinf(a[i]) || isinf(b[i]))
		{
			elog(WARNING, "euclidean_distance: NaN or Infinity detected at index %d", i);
		}
		double diff = a[i] - b[i];
		sum += diff * diff;
	}

	/* Defensive: Check for overflow */
	if (isnan(sum) || isinf(sum))
		ereport(ERROR,
			(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				errmsg("euclidean_distance: distance calculation overflow")));

	return sqrt(sum);
}

/*
 * knn_classify
 *
 * Classifies a single sample using K-Nearest Neighbors
 * Returns class label (0 or 1 for binary classification)
 */
PG_FUNCTION_INFO_V1(knn_classify);

Datum
knn_classify(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *feature_col;
	text *label_col;
	Vector *query_vector;
	int k;
	char *tbl_str;
	char *feat_str;
	char *label_str;
	StringInfoData query;
	int ret;
	int nvec = 0;
	int dim;
	KNNSample *samples = NULL;
	double class_votes[2] = { 0.0, 0.0 }; /* For binary classification */
	int predicted_class;
	int i;
	MemoryContext oldcontext;

	CHECK_NARGS(5);

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);
	query_vector = PG_GETARG_VECTOR_P(3);
	k = PG_GETARG_INT32(4);

	/* Defensive: Check NULL inputs */
	if (table_name == NULL || feature_col == NULL || label_col == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("knn_classify: table_name, feature_col, and label_col cannot be NULL")));

	if (query_vector == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("knn_classify: query_vector cannot be NULL")));

	/* Defensive: Validate k */
	if (k < 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("k must be at least 1, got %d", k)));

	if (k > 1000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("k must be at most 1000, got %d", k)));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	/* Defensive: Validate allocations */
	if (tbl_str == NULL || feat_str == NULL || label_str == NULL)
	{
		if (tbl_str)
			pfree(tbl_str);
		if (feat_str)
			pfree(feat_str);
		if (label_str)
			pfree(label_str);
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate strings")));
	}

	dim = query_vector->dim;

	/* Defensive: Validate vector dimension */
	if (dim <= 0 || dim > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("knn_classify: invalid query vector dimension: %d", dim)));

	elog(DEBUG1, "knn_classify: Starting classification with k=%d, dim=%d", k, dim);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
	{
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_connect failed")));
	}

	/* Build query to fetch all training samples */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		feat_str,
		label_str,
		tbl_str,
		feat_str,
		label_str);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("Query failed: %s", query.data)));
	}

	nvec = SPI_processed;

	/* Defensive: Validate sample count */
	if (nvec < k)
	{
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				errmsg("Need at least k=%d samples, but found only %d",
					k,
					nvec)));
	}

	if (nvec > 10000000)
	{
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Too many samples: %d (maximum 10000000)", nvec)));
	}

	/* Assert: SPI_tuptable should be valid after SPI_OK_SELECT */
	Assert(SPI_tuptable != NULL);
	Assert(SPI_tuptable->tupdesc != NULL);
	Assert(SPI_tuptable->vals != NULL);

	elog(DEBUG1, "knn_classify: Loaded %d training samples", nvec);

	/* Allocate samples array in caller's context */
	MemoryContextSwitchTo(oldcontext);
	samples = (KNNSample *)palloc(sizeof(KNNSample) * nvec);

	/* Defensive: Validate allocation */
	if (samples == NULL)
	{
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate samples array")));
	}

	/* Extract data and compute distances */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum feat_datum;
		Datum label_datum;
		bool feat_null;
		bool label_null;
		Vector *vec;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);

		if (feat_null || label_null)
			continue;

		vec = DatumGetVector(feat_datum);

		if (vec->dim != dim)
			ereport(ERROR,
				(errmsg("Dimension mismatch: expected %d, got "
					"%d",
					dim,
					vec->dim)));

		/* Copy features */
		samples[i].features = (float *)palloc(sizeof(float) * dim);
		memcpy(samples[i].features, vec->data, sizeof(float) * dim);
		samples[i].dim = dim;

		/* Get label */
		samples[i].label = DatumGetFloat8(label_datum);

		/* Compute distance */
		samples[i].distance = euclidean_distance(
			query_vector->data, samples[i].features, dim);
	}

	SPI_finish();

	/* Sort samples by distance */
	qsort(samples, nvec, sizeof(KNNSample), compare_samples);

	/* Vote among k nearest neighbors */
	for (i = 0; i < k && i < nvec; i++)
	{
		int label_class = (int)samples[i].label;
		if (label_class >= 0 && label_class < 2)
			class_votes[label_class] += 1.0;
	}

	/* Determine predicted class */
	predicted_class = (class_votes[1] > class_votes[0]) ? 1 : 0;

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		pfree(samples[i].features);
	pfree(samples);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(label_str);

	PG_RETURN_INT32(predicted_class);
}

/*
 * knn_regress
 *
 * Predicts a continuous value using K-Nearest Neighbors
 * Returns average of k nearest neighbors' values
 */
PG_FUNCTION_INFO_V1(knn_regress);

Datum
knn_regress(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *feature_col;
	text *target_col;
	Vector *query_vector;
	int k;
	char *tbl_str;
	char *feat_str;
	char *targ_str;
	StringInfoData query;
	int ret;
	int nvec = 0;
	int dim;
	KNNSample *samples = NULL;
	double prediction = 0.0;
	int i;
	MemoryContext oldcontext;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);
	query_vector = PG_GETARG_VECTOR_P(3);
	k = PG_GETARG_INT32(4);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	dim = query_vector->dim;

	if (k < 1)
		ereport(ERROR, (errmsg("k must be at least 1, got %d", k)));

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("SPI_connect failed")));

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		feat_str,
		targ_str,
		tbl_str,
		feat_str,
		targ_str);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR, (errmsg("Query failed")));

	nvec = SPI_processed;

	if (nvec < k)
		ereport(ERROR,
			(errmsg("Need at least k=%d samples, but found only %d",
				k,
				nvec)));

	/* Allocate samples array */
	MemoryContextSwitchTo(oldcontext);
	samples = (KNNSample *)palloc(sizeof(KNNSample) * nvec);

	/* Extract data and compute distances */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum feat_datum;
		Datum targ_datum;
		bool feat_null;
		bool targ_null;
		Vector *vec;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		vec = DatumGetVector(feat_datum);

		if (vec->dim != dim)
			ereport(ERROR,
				(errmsg("Dimension mismatch: expected %d, got "
					"%d",
					dim,
					vec->dim)));

		/* Copy features */
		samples[i].features = (float *)palloc(sizeof(float) * dim);
		memcpy(samples[i].features, vec->data, sizeof(float) * dim);
		samples[i].dim = dim;

		/* Get target value */
		samples[i].label = DatumGetFloat8(targ_datum);

		/* Compute distance */
		samples[i].distance = euclidean_distance(
			query_vector->data, samples[i].features, dim);
	}

	SPI_finish();

	/* Sort by distance */
	qsort(samples, nvec, sizeof(KNNSample), compare_samples);

	/* Average k nearest neighbors */
	for (i = 0; i < k && i < nvec; i++)
		prediction += samples[i].label;
	prediction /= k;

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		pfree(samples[i].features);
	pfree(samples);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(targ_str);

	PG_RETURN_FLOAT8(prediction);
}

/*
 * evaluate_knn_classifier
 *
 * Evaluates KNN classifier performance on a test set
 * Returns: [accuracy, precision, recall, f1_score]
 */
PG_FUNCTION_INFO_V1(evaluate_knn_classifier);

Datum
evaluate_knn_classifier(PG_FUNCTION_ARGS)
{
	text *train_table;
	text *test_table;
	text *feature_col;
	text *label_col;
	int k;
	char *train_str;
	char *test_str;
	char *feat_str;
	char *label_str;
	StringInfoData query;
	int ret;
	int ntest = 0;
	int tp = 0, tn = 0, fp = 0, fn = 0;
	double accuracy, precision, recall, f1_score;
	int i;
	Datum *result_datums;
	ArrayType *result_array;
	MemoryContext oldcontext;

	train_table = PG_GETARG_TEXT_PP(0);
	test_table = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);
	k = PG_GETARG_INT32(4);

	train_str = text_to_cstring(train_table);
	test_str = text_to_cstring(test_table);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("SPI_connect failed")));

	/* Build query to fetch test samples */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		feat_str,
		label_str,
		test_str,
		feat_str,
		label_str);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR, (errmsg("Query failed")));

	ntest = SPI_processed;

	/* For each test sample, classify using KNN */
	for (i = 0; i < ntest; i++)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum feat_datum;
		Datum label_datum;
		bool feat_null;
		bool label_null;
		Vector *vec;
		int y_true;
		int y_pred;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);

		if (feat_null || label_null)
			continue;

		vec = DatumGetVector(feat_datum);
		y_true = (int)DatumGetFloat8(label_datum);

		/* Call knn_classify for this sample */
		y_pred = DatumGetInt32(DirectFunctionCall5(knn_classify,
			PointerGetDatum(train_table),
			PointerGetDatum(feature_col),
			PointerGetDatum(label_col),
			PointerGetDatum(vec),
			Int32GetDatum(k)));

		/* Update confusion matrix */
		if (y_true == 1)
		{
			if (y_pred == 1)
				tp++;
			else
				fn++;
		} else
		{
			if (y_pred == 1)
				fp++;
			else
				tn++;
		}
	}

	SPI_finish();

	/* Compute metrics */
	accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
	precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
	recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
	f1_score = (precision + recall > 0)
		? 2.0 * precision * recall / (precision + recall)
		: 0.0;

	/* Build result array */
	MemoryContextSwitchTo(oldcontext);

	result_datums = (Datum *)palloc(sizeof(Datum) * 4);
	result_datums[0] = Float8GetDatum(accuracy);
	result_datums[1] = Float8GetDatum(precision);
	result_datums[2] = Float8GetDatum(recall);
	result_datums[3] = Float8GetDatum(f1_score);

	result_array = construct_array(result_datums,
		4,
		FLOAT8OID,
		sizeof(float8),
		FLOAT8PASSBYVAL,
		'd');

	pfree(result_datums);
	pfree(train_str);
	pfree(test_str);
	pfree(feat_str);
	pfree(label_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * predict_knn_model_id
 *
 * Predicts using a KNN model stored in the catalog by model_id.
 * This is the CPU fallback implementation.
 *
 * Arguments:
 *   model_id INT - model ID from neurondb.ml_models
 *   features REAL[] - feature vector for prediction
 * Returns:
 *   prediction (FLOAT8)
 */
/*
 * train_knn_model_id
 *
 * Trains KNN (lazy learner) - stores metadata in catalog, returns model_id
 * KNN doesn't actually train, it just stores the training table reference
 */
PG_FUNCTION_INFO_V1(train_knn_model_id);

Datum
train_knn_model_id(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *feature_col;
	text *label_col;
	int k_value;
	char *tbl_str;
	char *feat_str;
	char *label_str;
	int nvec, dim;
	int i;
	bytea *model_data = NULL;
	MLCatalogModelSpec spec;
	Jsonb *metrics;
	StringInfoData metrics_json;
	int32 model_id;
	StringInfoData model_buf;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);
	k_value = PG_GETARG_INT32(3);

	if (k_value < 1 || k_value > 1000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("k must be between 1 and 1000, got %d", k_value)));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	/* Load training data and labels together using SPI */
	{
		StringInfoData query;
		int ret;
		Vector *first_vec;
		bool isnull;
		Datum vec_datum, label_datum;
		
		if (SPI_connect() != SPI_OK_CONNECT)
			ereport(ERROR, (errmsg("SPI_connect failed")));
		
		initStringInfo(&query);
		appendStringInfo(&query, "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
			feat_str, label_str, tbl_str, feat_str, label_str);
		
		ret = SPI_execute(query.data, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			SPI_finish();
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			pfree(query.data);
			ereport(ERROR, (errmsg("Failed to fetch training data")));
		}
		
		nvec = SPI_processed;
		if (nvec < k_value)
		{
			SPI_finish();
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			pfree(query.data);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("KNN: need at least %d samples, got %d", k_value, nvec)));
		}
		
		/* Get dimension from first vector */
		vec_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
		if (isnull)
		{
			SPI_finish();
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			pfree(query.data);
			ereport(ERROR, (errmsg("NULL vector in first row")));
		}
		first_vec = DatumGetVector(vec_datum);
		dim = first_vec->dim;
		
		/* Verify all vectors have same dimension (we don't need to store the data) */
		for (i = 0; i < nvec; i++)
		{
			Vector *vec;
			
			vec_datum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &isnull);
			if (isnull)
			{
				SPI_finish();
				pfree(tbl_str);
				pfree(feat_str);
				pfree(label_str);
				pfree(query.data);
				ereport(ERROR, (errmsg("NULL vector at row %d", i)));
			}
			vec = DatumGetVector(vec_datum);
			if (vec->dim != dim)
			{
				SPI_finish();
				pfree(tbl_str);
				pfree(feat_str);
				pfree(label_str);
				pfree(query.data);
				ereport(ERROR, (errmsg("Inconsistent vector dimension at row %d", i)));
			}
		}
		
		SPI_finish();
		/* query.data will be cleaned up when function returns */
	}

	/* For KNN (lazy learner), store minimal model data: just k and dimensions */
	/* The actual training data stays in the table */
	initStringInfo(&model_buf);
	appendBinaryStringInfo(&model_buf, (char *)&k_value, sizeof(int));
	appendBinaryStringInfo(&model_buf, (char *)&nvec, sizeof(int));
	appendBinaryStringInfo(&model_buf, (char *)&dim, sizeof(int));
	
	/* Store table name, feature col, label col as strings for CPU prediction */
	appendBinaryStringInfo(&model_buf, tbl_str, strlen(tbl_str) + 1);
	appendBinaryStringInfo(&model_buf, feat_str, strlen(feat_str) + 1);
	appendBinaryStringInfo(&model_buf, label_str, strlen(label_str) + 1);

	{
		int total_size = VARHDRSZ + model_buf.len;
		model_data = (bytea *)palloc(total_size);
		SET_VARSIZE(model_data, total_size);
		memcpy(VARDATA(model_data), model_buf.data, model_buf.len);
		pfree(model_buf.data);
	}

	/* Build metrics JSONB */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json, "{\"storage\": \"cpu\", \"k\": %d, \"n_samples\": %d, \"n_features\": %d}",
		k_value, nvec, dim);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metrics_json.data)));
	pfree(metrics_json.data);

	/* Store model in catalog */
	memset(&spec, 0, sizeof(MLCatalogModelSpec));
	spec.project_name = NULL;
	spec.algorithm = "knn";
	spec.training_table = tbl_str;
	spec.training_column = label_str;
	spec.model_data = model_data;
	spec.metrics = metrics;
	spec.num_samples = nvec;
	spec.num_features = dim;

	model_id = ml_catalog_register_model(&spec);

	/* Cleanup */
	pfree(tbl_str);
	pfree(feat_str);
	pfree(label_str);

	PG_RETURN_INT32(model_id);
}

PG_FUNCTION_INFO_V1(predict_knn_model_id);

Datum
predict_knn_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	ArrayType *features_array;
	bytea *model_data = NULL;
	Jsonb *metrics = NULL;
	const char *base;
	NdbCudaKnnModelHeader *hdr;
	const float *training_features;
	const double *training_labels;
	float *query_features = NULL;
	float *distances = NULL;
	int *indices = NULL;
	int i;
	int j;
	int ndims;
	int *dims;
	int nelems;
	float8 *features = NULL;
	float8 *features_allocated = NULL; /* For float4->float8 conversion */
	double prediction = 0.0;
	MemoryContext oldcontext;
	MemoryContext callcontext;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("predict_knn_model_id: requires 2 arguments, got %d",
					PG_NARGS()),
				errhint("Usage: predict_knn_model_id(model_id, features[])")));

	model_id = PG_GETARG_INT32(0);
	features_array = PG_GETARG_ARRAYTYPE_P(1);

	/* Validate features array */
	ndims = ARR_NDIM(features_array);
	dims = ARR_DIMS(features_array);
	nelems = ArrayGetNItems(ndims, dims);
	
		/* Handle both float4[] (real[]) and float8[] arrays */
		{
			Oid elem_type = ARR_ELEMTYPE(features_array);
			if (elem_type == FLOAT4OID)
			{
				/* Convert float4[] to float8[] */
				float4 *features_f4 = (float4 *)ARR_DATA_PTR(features_array);
				features_allocated = (float8 *)palloc(sizeof(float8) * nelems);
				for (i = 0; i < nelems; i++)
					features_allocated[i] = (float8)features_f4[i];
				features = features_allocated;
			}
			else
			{
				/* Already float8[] */
				features = (float8 *)ARR_DATA_PTR(features_array);
			}
		}

	if (nelems == 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("predict_knn_model_id: features array cannot be empty")));

	/* Fetch model data from catalog */
	callcontext = AllocSetContextCreate(CurrentMemoryContext,
		"predict_knn_model_id context",
		ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("predict_knn_model_id: model %d not found", model_id)));
	}

	if (model_data == NULL)
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("predict_knn_model_id: model %d has no model_data", model_id)));
	}

	/* Ensure bytea is in current function's memory context */
	{
		int data_len = VARSIZE(model_data);
		bytea *copy = (bytea *)palloc(data_len);
		memcpy(copy, model_data, data_len);
		model_data = copy;
	}

	base = VARDATA(model_data);
	
	/* Check if this is CPU model (starts with k, n_samples, n_features) or GPU model */
	/* CPU model: k (int), n_samples (int), n_features (int), table_name (string), feature_col (string), label_col (string) */
	/* GPU model: NdbCudaKnnModelHeader */
	if (VARSIZE(model_data) - VARHDRSZ < sizeof(int) * 3)
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);
		if (metrics)
			pfree(metrics);
		if (model_data)
			pfree(model_data);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("predict_knn_model_id: model %d has invalid header (too small)", model_id)));
	}

	/* Try to detect CPU vs GPU format */
	/* CPU format: first int is k (usually 1-100), GPU format: first bytes are magic/version */
	{
		int first_int = *(int *)base;
		/* If first int is reasonable k value (1-1000) and model size matches CPU format, use CPU */
		if (first_int >= 1 && first_int <= 1000)
		{
			/* CPU model format */
			int k_cpu, n_samples_cpu, n_features_cpu;
			char *table_name_cpu, *feature_col_cpu, *label_col_cpu;
			const char *str_ptr;
			Vector *query_vector;
			text *table_text, *feat_text, *label_text;
			
			memcpy(&k_cpu, base, sizeof(int));
			memcpy(&n_samples_cpu, base + sizeof(int), sizeof(int));
			memcpy(&n_features_cpu, base + sizeof(int) * 2, sizeof(int));
			
			if (nelems != n_features_cpu)
			{
				MemoryContextSwitchTo(oldcontext);
				MemoryContextDelete(callcontext);
				if (metrics)
					pfree(metrics);
				if (model_data)
					pfree(model_data);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Feature dimension mismatch: expected %d, got %d", n_features_cpu, nelems)));
			}
			
			/* Extract strings */
			str_ptr = base + sizeof(int) * 3;
			table_name_cpu = (char *)str_ptr;
			str_ptr += strlen(table_name_cpu) + 1;
			feature_col_cpu = (char *)str_ptr;
			str_ptr += strlen(feature_col_cpu) + 1;
			label_col_cpu = (char *)str_ptr;
			
			/* Call knn_classify via SPI SQL to avoid memory context issues */
			/* knn_classify uses SPI internally, so calling it via SQL is safer */
			StringInfoData sql_query;
			StringInfoData tbl_lit, feat_lit, label_lit;
			int ret;
			int j;
			
			/* Quote strings for SQL */
			initStringInfo(&tbl_lit);
			appendStringInfoChar(&tbl_lit, '\'');
			for (j = 0; table_name_cpu[j]; j++)
			{
				if (table_name_cpu[j] == '\'')
					appendStringInfoChar(&tbl_lit, '\'');
				appendStringInfoChar(&tbl_lit, table_name_cpu[j]);
			}
			appendStringInfoChar(&tbl_lit, '\'');
			
			initStringInfo(&feat_lit);
			appendStringInfoChar(&feat_lit, '\'');
			for (j = 0; feature_col_cpu[j]; j++)
			{
				if (feature_col_cpu[j] == '\'')
					appendStringInfoChar(&feat_lit, '\'');
				appendStringInfoChar(&feat_lit, feature_col_cpu[j]);
			}
			appendStringInfoChar(&feat_lit, '\'');
			
			initStringInfo(&label_lit);
			appendStringInfoChar(&label_lit, '\'');
			for (j = 0; label_col_cpu[j]; j++)
			{
				if (label_col_cpu[j] == '\'')
					appendStringInfoChar(&label_lit, '\'');
				appendStringInfoChar(&label_lit, label_col_cpu[j]);
			}
			appendStringInfoChar(&label_lit, '\'');
			
			/* Build vector literal string */
			initStringInfo(&sql_query);
			appendStringInfo(&sql_query, "SELECT knn_classify(%s, %s, %s, '",
				tbl_lit.data, feat_lit.data, label_lit.data);
			
			/* Build vector literal */
			appendStringInfoChar(&sql_query, '[');
			for (i = 0; i < nelems; i++)
			{
				if (i > 0)
					appendStringInfoString(&sql_query, ", ");
				/* Validate feature value before formatting */
				if (!isfinite(features[i]))
				{
					SPI_finish();
					pfree(sql_query.data);
					pfree(tbl_lit.data);
					pfree(feat_lit.data);
					pfree(label_lit.data);
					MemoryContextSwitchTo(oldcontext);
					MemoryContextDelete(callcontext);
					if (metrics)
						pfree(metrics);
					if (model_data)
						pfree(model_data);
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("Non-finite value in features array at index %d", i)));
				}
				appendStringInfo(&sql_query, "%.6f", features[i]);
			}
			appendStringInfo(&sql_query, "]'::vector, %d)", k_cpu);
			
			/* Execute via SPI */
			if (SPI_connect() != SPI_OK_CONNECT)
			{
				pfree(sql_query.data);
				pfree(tbl_lit.data);
				pfree(feat_lit.data);
				pfree(label_lit.data);
				MemoryContextSwitchTo(oldcontext);
				MemoryContextDelete(callcontext);
				if (metrics)
					pfree(metrics);
				if (model_data)
					pfree(model_data);
				ereport(ERROR, (errmsg("SPI_connect failed in predict_knn_model_id")));
			}
			
			ret = SPI_execute(sql_query.data, true, 0);
			if (ret != SPI_OK_SELECT || SPI_processed == 0)
			{
				SPI_finish();
				pfree(sql_query.data);
				pfree(tbl_lit.data);
				pfree(feat_lit.data);
				pfree(label_lit.data);
				MemoryContextSwitchTo(oldcontext);
				MemoryContextDelete(callcontext);
				if (metrics)
					pfree(metrics);
				if (model_data)
					pfree(model_data);
				ereport(ERROR, (errmsg("knn_classify failed in predict_knn_model_id")));
			}
			
			/* Get result */
			{
				bool isnull;
				Datum result_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
				if (isnull)
					prediction = 0.0;
				else
					prediction = (double)DatumGetInt32(result_datum);
			}
			
			SPI_finish();
			pfree(sql_query.data);
			pfree(tbl_lit.data);
			pfree(feat_lit.data);
			pfree(label_lit.data);
			
			/* Free features_allocated before deleting callcontext */
			if (features_allocated)
				pfree(features_allocated);
			
			/* Free model_data and metrics before switching contexts */
			if (model_data)
				pfree(model_data);
			if (metrics)
				pfree(metrics);
			
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(callcontext);
			
			PG_RETURN_FLOAT8(prediction);
		}
	}

	/* GPU model format */
	if (VARSIZE(model_data) - VARHDRSZ < sizeof(NdbCudaKnnModelHeader))
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);
		if (metrics)
			pfree(metrics);
		if (model_data)
			pfree(model_data);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("predict_knn_model_id: model %d has invalid header",
					model_id)));
	}

	hdr = (NdbCudaKnnModelHeader *)base;

	/* Validate model header */
	if (hdr->n_samples <= 0 || hdr->n_samples > 100000000 ||
		hdr->n_features <= 0 || hdr->n_features > 1000000 ||
		hdr->k <= 0 || hdr->k > hdr->n_samples ||
		(hdr->task_type != 0 && hdr->task_type != 1))
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("predict_knn_model_id: model %d has invalid header",
					model_id)));
	}

	/* Validate feature dimension */
	if (nelems != hdr->n_features)
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("predict_knn_model_id: feature dimension mismatch (expected %d, got %d)",
					hdr->n_features, nelems)));
	}

	/* Extract training data from model */
	training_features = (const float *)(base + sizeof(NdbCudaKnnModelHeader));
	training_labels = (const double *)(base + sizeof(NdbCudaKnnModelHeader) +
		sizeof(float) * (size_t)hdr->n_samples * (size_t)hdr->n_features);

	/* Convert query features to float */
	query_features = (float *)palloc(sizeof(float) * hdr->n_features);
	for (i = 0; i < hdr->n_features; i++)
	{
		if (!isfinite(features[i]))
		{
			pfree(query_features);
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(callcontext);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("predict_knn_model_id: non-finite value in features array")));
		}
		query_features[i] = (float)features[i];
	}

	/* Allocate distance array and indices */
	distances = (float *)palloc(sizeof(float) * hdr->n_samples);
	indices = (int *)palloc(sizeof(int) * hdr->n_samples);

	/* Compute distances to all training samples */
	for (i = 0; i < hdr->n_samples; i++)
	{
		double sum = 0.0;
		const float *train_feat = training_features + (i * hdr->n_features);

		for (j = 0; j < hdr->n_features; j++)
		{
			double diff = query_features[j] - train_feat[j];
			sum += diff * diff;
		}
		distances[i] = (float)sqrt(sum);
		indices[i] = i;
	}

	/* Sort by distance (simple selection sort for k nearest) */
	for (i = 0; i < hdr->k && i < hdr->n_samples; i++)
	{
		int min_idx = i;
		for (j = i + 1; j < hdr->n_samples; j++)
		{
			if (distances[j] < distances[min_idx])
				min_idx = j;
		}
		if (min_idx != i)
		{
			float tmp_dist = distances[i];
			int tmp_idx = indices[i];
			distances[i] = distances[min_idx];
			indices[i] = indices[min_idx];
			distances[min_idx] = tmp_dist;
			indices[min_idx] = tmp_idx;
		}
	}

	/* Compute prediction from k nearest neighbors */
	if (hdr->task_type == 0)
	{
		/* Classification: majority vote */
		double class_votes[2] = {0.0, 0.0};
		for (i = 0; i < hdr->k && i < hdr->n_samples; i++)
		{
			int label = (int)training_labels[indices[i]];
			if (label >= 0 && label < 2)
				class_votes[label] += 1.0;
		}
		prediction = (class_votes[1] > class_votes[0]) ? 1.0 : 0.0;
	}
	else
	{
		/* Regression: average */
		for (i = 0; i < hdr->k && i < hdr->n_samples; i++)
			prediction += training_labels[indices[i]];
		prediction /= hdr->k;
	}

	/* Cleanup */
	pfree(query_features);
	pfree(distances);
	pfree(indices);
	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(callcontext);

	PG_RETURN_FLOAT8(prediction);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for KNN
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"

/* Stub function to satisfy linker - full implementation needed */
void
neurondb_gpu_register_knn_model(void)
{
	/* KNN GPU Model Ops not yet implemented - will use CPU fallback */
	elog(DEBUG1, "KNN GPU Model Ops registration skipped - not yet implemented");
}
