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
#include "utils/jsonb.h"
#include "utils/memutils.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_catalog.h"
#include "neurondb_cuda_knn.h"
#include "vector/vector_types.h"
#include "ml_gpu_registry.h"
#ifdef NDB_GPU_CUDA
#include "neurondb_gpu_model.h"
#include "neurondb_gpu.h"
#endif

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif

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
euclidean_distance(const float *a, const float *b, int dim)
{
	double sum = 0.0;
	int i;

	for (i = 0; i < dim; i++)
	{
		double diff = a[i] - b[i];
		sum += diff * diff;
	}

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

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);
	query_vector = PG_GETARG_VECTOR_P(3);
	k = PG_GETARG_INT32(4);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	dim = query_vector->dim;

	if (k < 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: k must be at least 1, got %d", k)));

	oldcontext = CurrentMemoryContext;

	/* Build query to fetch all training samples before SPI */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quote_identifier(feat_str),
		quote_identifier(label_str),
		quote_identifier(tbl_str),
		quote_identifier(feat_str),
		quote_identifier(label_str));
	elog(DEBUG1, "knn_classify: executing query: %s", query.data);

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
	{
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: SPI_connect failed")));
	}

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: query failed: %s", query.data)));

	nvec = SPI_processed;

	if (nvec < k)
	{
		SPI_finish();
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		ereport(ERROR,
			(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				errmsg("neurondb: need at least k=%d samples, but found only %d",
					k,
					nvec)));
	}

	/* Allocate samples array in caller's context */
	MemoryContextSwitchTo(oldcontext);
	samples = (KNNSample *)palloc(sizeof(KNNSample) * nvec);

	/* Extract data and compute distances - track valid sample count */
	{
		int nsamples = 0;

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
			if (vec == NULL || vec->dim <= 0)
				continue;

			if (vec->dim != dim)
			{
				int j;
				SPI_finish();
				pfree(query.data);
				for (j = 0; j < nsamples; j++)
					pfree(samples[j].features);
				pfree(samples);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(label_str);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: dimension mismatch: expected %d, got %d",
							dim, vec->dim)));
			}

			/* Copy features */
			samples[nsamples].features = (float *)palloc(sizeof(float) * dim);
			memcpy(samples[nsamples].features, vec->data, sizeof(float) * dim);
			samples[nsamples].dim = dim;

			/* Get label */
			samples[nsamples].label = DatumGetFloat8(label_datum);

			/* Compute distance */
			samples[nsamples].distance = euclidean_distance(
				query_vector->data, samples[nsamples].features, dim);
			nsamples++;
		}

		if (nsamples < k)
		{
			SPI_finish();
			pfree(query.data);
			for (i = 0; i < nsamples; i++)
				pfree(samples[i].features);
			pfree(samples);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
					errmsg("neurondb: need at least k=%d valid samples, but found only %d",
						k,
						nsamples)));
		}

		SPI_finish();
		pfree(query.data);

		/* Sort samples by distance */
		qsort(samples, nsamples, sizeof(KNNSample), compare_samples);

		/* Vote among k nearest neighbors */
		for (i = 0; i < k && i < nsamples; i++)
		{
			int label_class = (int)samples[i].label;
			if (label_class >= 0 && label_class < 2)
				class_votes[label_class] += 1.0;
		}

		/* Determine predicted class */
		predicted_class = (class_votes[1] > class_votes[0]) ? 1 : 0;

		/* Cleanup */
		for (i = 0; i < nsamples; i++)
			pfree(samples[i].features);
		pfree(samples);
	}
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
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: k must be at least 1, got %d", k)));

	oldcontext = CurrentMemoryContext;

	/* Build query before SPI */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quote_identifier(feat_str),
		quote_identifier(targ_str),
		quote_identifier(tbl_str),
		quote_identifier(feat_str),
		quote_identifier(targ_str));
	elog(DEBUG1, "knn_regress: executing query: %s", query.data);

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
	{
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: SPI_connect failed")));
	}

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: query failed")));

	nvec = SPI_processed;

	if (nvec < k)
	{
		SPI_finish();
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
			(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				errmsg("neurondb: need at least k=%d samples, but found only %d",
					k,
					nvec)));
	}

	/* Allocate samples array */
	MemoryContextSwitchTo(oldcontext);
	samples = (KNNSample *)palloc(sizeof(KNNSample) * nvec);

	/* Extract data and compute distances - track valid sample count */
	{
		int nsamples = 0;

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
			if (vec == NULL || vec->dim <= 0)
				continue;

			if (vec->dim != dim)
			{
				int j;
				SPI_finish();
				pfree(query.data);
				for (j = 0; j < nsamples; j++)
					pfree(samples[j].features);
				pfree(samples);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(targ_str);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: dimension mismatch: expected %d, got %d",
							dim, vec->dim)));
			}

			/* Copy features */
			samples[nsamples].features = (float *)palloc(sizeof(float) * dim);
			memcpy(samples[nsamples].features, vec->data, sizeof(float) * dim);
			samples[nsamples].dim = dim;

			/* Get target value */
			samples[nsamples].label = DatumGetFloat8(targ_datum);

			/* Compute distance */
			samples[nsamples].distance = euclidean_distance(
				query_vector->data, samples[nsamples].features, dim);
			nsamples++;
		}

		if (nsamples < k)
		{
			SPI_finish();
			pfree(query.data);
			for (i = 0; i < nsamples; i++)
				pfree(samples[i].features);
			pfree(samples);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
					errmsg("neurondb: need at least k=%d valid samples, but found only %d",
						k,
						nsamples)));
		}

		SPI_finish();
		pfree(query.data);

		/* Sort by distance */
		qsort(samples, nsamples, sizeof(KNNSample), compare_samples);

		/* Average k nearest neighbors */
		for (i = 0; i < k && i < nsamples; i++)
			prediction += samples[i].label;
		prediction /= k;

		/* Cleanup */
		for (i = 0; i < nsamples; i++)
			pfree(samples[i].features);
		pfree(samples);
	}
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

	/* Build query to fetch test samples before SPI */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quote_identifier(feat_str),
		quote_identifier(label_str),
		quote_identifier(test_str),
		quote_identifier(feat_str),
		quote_identifier(label_str));
	elog(DEBUG1, "evaluate_knn_classifier: executing query: %s", query.data);

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
	{
		pfree(query.data);
		pfree(train_str);
		pfree(test_str);
		pfree(feat_str);
		pfree(label_str);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: SPI_connect failed")));
	}

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: query failed")));

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
		int y_true;
		int y_pred;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);

		if (feat_null || label_null)
			continue;

		y_true = (int)DatumGetFloat8(label_datum);

		/* Call knn_classify for this sample - pass feat_datum, not vec */
		y_pred = DatumGetInt32(DirectFunctionCall5(knn_classify,
			PointerGetDatum(train_table),
			PointerGetDatum(feature_col),
			PointerGetDatum(label_col),
			feat_datum,
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
	pfree(query.data);

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
		Datum vec_datum;
		
		if (SPI_connect() != SPI_OK_CONNECT)
			ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: SPI_connect failed")));
		
		initStringInfo(&query);
		appendStringInfo(&query, "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
			quote_identifier(feat_str),
			quote_identifier(label_str),
			quote_identifier(tbl_str),
			quote_identifier(feat_str),
			quote_identifier(label_str));
		
		ret = SPI_execute(query.data, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			SPI_finish();
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			pfree(query.data);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("neurondb: failed to fetch training data")));
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
		
		/* Get dimension from first row - handle both array and vector types */
		{
			TupleDesc tupdesc = SPI_tuptable->tupdesc;
			Oid feat_type_oid = SPI_gettypeid(tupdesc, 1);
			bool feat_is_array = (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID);

			vec_datum = SPI_getbinval(SPI_tuptable->vals[0], tupdesc, 1, &isnull);
			if (isnull)
			{
				SPI_finish();
				pfree(tbl_str);
				pfree(feat_str);
				pfree(label_str);
				pfree(query.data);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: NULL feature in first row")));
			}

			if (feat_is_array)
			{
				ArrayType *arr = DatumGetArrayTypeP(vec_datum);
				if (ARR_NDIM(arr) != 1)
				{
					SPI_finish();
					pfree(tbl_str);
					pfree(feat_str);
					pfree(label_str);
					pfree(query.data);
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("neurondb: feature array must be 1-dimensional")));
				}
				dim = ARR_DIMS(arr)[0];
			}
			else
			{
				first_vec = DatumGetVector(vec_datum);
				if (first_vec == NULL)
				{
					SPI_finish();
					pfree(tbl_str);
					pfree(feat_str);
					pfree(label_str);
					pfree(query.data);
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("neurondb: invalid vector in first row")));
				}
				dim = first_vec->dim;
			}
		}
		
		/* Verify all features have same dimension (we don't need to store the data) */
		{
			TupleDesc tupdesc = SPI_tuptable->tupdesc;
			Oid feat_type_oid = SPI_gettypeid(tupdesc, 1);
			bool feat_is_array = (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID);

			for (i = 0; i < nvec; i++)
			{
				vec_datum = SPI_getbinval(SPI_tuptable->vals[i], tupdesc, 1, &isnull);
				if (isnull)
				{
					SPI_finish();
					pfree(tbl_str);
					pfree(feat_str);
					pfree(label_str);
					pfree(query.data);
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("neurondb: NULL feature at row %d", i)));
				}

				if (feat_is_array)
				{
					ArrayType *arr = DatumGetArrayTypeP(vec_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != dim)
					{
						SPI_finish();
						pfree(tbl_str);
						pfree(feat_str);
						pfree(label_str);
						pfree(query.data);
						ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								errmsg("neurondb: inconsistent feature array dimension at row %d (expected %d)",
									i, dim)));
					}
				}
				else
				{
					Vector *vec = DatumGetVector(vec_datum);
					if (vec == NULL || vec->dim != dim)
					{
						SPI_finish();
						pfree(tbl_str);
						pfree(feat_str);
						pfree(label_str);
						pfree(query.data);
						ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								errmsg("neurondb: inconsistent vector dimension at row %d (expected %d)",
									i, dim)));
					}
				}
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
	elog(DEBUG1, "predict_knn_model_id: creating memory context");
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
			StringInfoData sql_query;
			StringInfoData tbl_lit, feat_lit, label_lit;
			int ret;
			int k;
			
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
			
			/* Quote strings for SQL */
			initStringInfo(&tbl_lit);
			appendStringInfoChar(&tbl_lit, '\'');
			for (k = 0; table_name_cpu[k]; k++)
			{
				if (table_name_cpu[k] == '\'')
					appendStringInfoChar(&tbl_lit, '\'');
				appendStringInfoChar(&tbl_lit, table_name_cpu[k]);
			}
			appendStringInfoChar(&tbl_lit, '\'');
			
			initStringInfo(&feat_lit);
			appendStringInfoChar(&feat_lit, '\'');
			for (k = 0; feature_col_cpu[k]; k++)
			{
				if (feature_col_cpu[k] == '\'')
					appendStringInfoChar(&feat_lit, '\'');
				appendStringInfoChar(&feat_lit, feature_col_cpu[k]);
			}
			appendStringInfoChar(&feat_lit, '\'');
			
			initStringInfo(&label_lit);
			appendStringInfoChar(&label_lit, '\'');
			for (k = 0; label_col_cpu[k]; k++)
			{
				if (label_col_cpu[k] == '\'')
					appendStringInfoChar(&label_lit, '\'');
				appendStringInfoChar(&label_lit, label_col_cpu[k]);
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
					pfree(sql_query.data);
					pfree(tbl_lit.data);
					pfree(feat_lit.data);
					pfree(label_lit.data);
					if (features_allocated)
						pfree(features_allocated);
					MemoryContextSwitchTo(oldcontext);
					MemoryContextDelete(callcontext);
					if (metrics)
						pfree(metrics);
					if (model_data)
						pfree(model_data);
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("neurondb: non-finite value in features array at index %d", i)));
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
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("neurondb: SPI_connect failed in predict_knn_model_id")));
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
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("neurondb: knn_classify failed in predict_knn_model_id")));
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
	if (features_allocated)
		pfree(features_allocated);
	if (model_data)
		pfree(model_data);
	if (metrics)
		pfree(metrics);
	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(callcontext);

	PG_RETURN_FLOAT8(prediction);
}

/*
 * knn_predict_batch
 *
 * Helper function to predict a batch of samples using KNN model.
 * Updates confusion matrix.
 */
static void
knn_predict_batch(int32 model_id,
	const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	int *tp_out,
	int *tn_out,
	int *fp_out,
	int *fn_out)
{
	int i;
	int tp = 0;
	int tn = 0;
	int fp = 0;
	int fn = 0;
	bytea *model_data = NULL;
	Jsonb *metrics = NULL;
	const char *base;
	int k_value = 0;

	if (features == NULL || labels == NULL || n_samples <= 0)
	{
		if (tp_out)
			*tp_out = 0;
		if (tn_out)
			*tn_out = 0;
		if (fp_out)
			*fp_out = 0;
		if (fn_out)
			*fn_out = 0;
		return;
	}

	/* Load model to get k value */
	if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
	{
		if (tp_out)
			*tp_out = 0;
		if (tn_out)
			*tn_out = 0;
		if (fp_out)
			*fp_out = 0;
		if (fn_out)
			*fn_out = 0;
		return;
	}

	if (model_data == NULL)
	{
		if (metrics)
			pfree(metrics);
		if (tp_out)
			*tp_out = 0;
		if (tn_out)
			*tn_out = 0;
		if (fp_out)
			*fp_out = 0;
		if (fn_out)
			*fn_out = 0;
		return;
	}

	base = VARDATA(model_data);

	/* Detect CPU vs GPU format and get k_value */
	{
		int first_int = *(int *)base;
		if (first_int >= 1 && first_int <= 1000)
		{
			/* CPU format */
			memcpy(&k_value, base, sizeof(int));
		}
		else
		{
			/* GPU format */
			const NdbCudaKnnModelHeader *gpu_hdr;

			if (VARSIZE(model_data) - VARHDRSZ < (int)sizeof(NdbCudaKnnModelHeader))
			{
				if (metrics)
					pfree(metrics);
				if (model_data)
					pfree(model_data);
				if (tp_out)
					*tp_out = 0;
				if (tn_out)
					*tn_out = 0;
				if (fp_out)
					*fp_out = 0;
				if (fn_out)
					*fn_out = 0;
				return;
			}
			gpu_hdr = (const NdbCudaKnnModelHeader *)base;
			k_value = gpu_hdr->k;
		}
	}

	if (k_value < 1)
	{
		if (metrics)
			pfree(metrics);
		if (model_data)
			pfree(model_data);
		if (tp_out)
			*tp_out = 0;
		if (tn_out)
			*tn_out = 0;
		if (fp_out)
			*fp_out = 0;
		if (fn_out)
			*fn_out = 0;
		return;
	}

	/* Predict directly - avoid calling predict_knn_model_id which uses SPI */
	{
		int first_int = *(int *)base;
		bool is_gpu = (first_int < 1 || first_int > 1000);

		if (is_gpu)
		{
			/* GPU model: use payload directly */
			const NdbCudaKnnModelHeader *gpu_hdr;
			const float *training_features;
			const double *training_labels;

			if (VARSIZE(model_data) - VARHDRSZ < (int)sizeof(NdbCudaKnnModelHeader))
			{
				if (metrics)
					pfree(metrics);
				if (model_data)
					pfree(model_data);
				if (tp_out)
					*tp_out = 0;
				if (tn_out)
					*tn_out = 0;
				if (fp_out)
					*fp_out = 0;
				if (fn_out)
					*fn_out = 0;
				return;
			}

			gpu_hdr = (const NdbCudaKnnModelHeader *)base;
			training_features = (const float *)(base + sizeof(NdbCudaKnnModelHeader));
			training_labels = (const double *)(base + sizeof(NdbCudaKnnModelHeader) +
				sizeof(float) * (size_t)gpu_hdr->n_samples * (size_t)gpu_hdr->n_features);

			/* For each test sample, compute distances and predict */
			for (i = 0; i < n_samples; i++)
			{
				const float *row = features + (i * feature_dim);
				double y_true = labels[i];
				int true_class;
				double prediction = 0.0;
				int pred_class;
				float *distances_local = NULL;
				int *indices_local = NULL;
				int j;

				if (!isfinite(y_true))
					continue;

				true_class = (int)rint(y_true);
				if (true_class < 0)
					true_class = 0;
				if (true_class > 1)
					true_class = 1;

				/* Compute distances to all training samples */
				distances_local = (float *)palloc(sizeof(float) * gpu_hdr->n_samples);
				indices_local = (int *)palloc(sizeof(int) * gpu_hdr->n_samples);

				for (j = 0; j < gpu_hdr->n_samples; j++)
				{
					double sum = 0.0;
					const float *train_feat = training_features + (j * gpu_hdr->n_features);
					int k;

					for (k = 0; k < feature_dim; k++)
					{
						double diff = row[k] - train_feat[k];
						sum += diff * diff;
					}
					distances_local[j] = (float)sqrt(sum);
					indices_local[j] = j;
				}

				/* Selection sort for k nearest */
				for (j = 0; j < k_value && j < gpu_hdr->n_samples; j++)
				{
					int min_idx = j;
					int k;

					for (k = j + 1; k < gpu_hdr->n_samples; k++)
					{
						if (distances_local[k] < distances_local[min_idx])
							min_idx = k;
					}
					if (min_idx != j)
					{
						float tmp_dist = distances_local[j];
						int tmp_idx = indices_local[j];
						distances_local[j] = distances_local[min_idx];
						indices_local[j] = indices_local[min_idx];
						distances_local[min_idx] = tmp_dist;
						indices_local[min_idx] = tmp_idx;
					}
				}

				/* Majority vote */
				{
					double class_votes[2] = {0.0, 0.0};
					int k;

					for (k = 0; k < k_value && k < gpu_hdr->n_samples; k++)
					{
						int label = (int)training_labels[indices_local[k]];
						if (label >= 0 && label < 2)
							class_votes[label] += 1.0;
					}
					prediction = (class_votes[1] > class_votes[0]) ? 1.0 : 0.0;
				}

				pred_class = (int)rint(prediction);
				if (pred_class < 0)
					pred_class = 0;
				if (pred_class > 1)
					pred_class = 1;

				pfree(distances_local);
				pfree(indices_local);

				/* Update confusion matrix */
				if (true_class == 1 && pred_class == 1)
					tp++;
				else if (true_class == 0 && pred_class == 0)
					tn++;
				else if (true_class == 0 && pred_class == 1)
					fp++;
				else if (true_class == 1 && pred_class == 0)
					fn++;
			}
		}
		else
		{
			/* CPU model: load training data and predict */
			char *table_name_cpu, *feature_col_cpu, *label_col_cpu;
			const char *str_ptr;
			StringInfoData query;
			int ret;
			int n_train = 0;
			float *train_features = NULL;
			double *train_labels = NULL;
			int train_valid = 0;
			int j;
			MemoryContext outer_context;

			/* Extract strings from model */
			str_ptr = base + sizeof(int) * 3; /* Skip k, n_samples, n_features */
			table_name_cpu = (char *)str_ptr;
			str_ptr += strlen(table_name_cpu) + 1;
			feature_col_cpu = (char *)str_ptr;
			str_ptr += strlen(feature_col_cpu) + 1;
			label_col_cpu = (char *)str_ptr;

			/* Save outer memory context before nested SPI */
			outer_context = CurrentMemoryContext;

			/* Build query in outer context before nested SPI */
			initStringInfo(&query);
			appendStringInfo(&query,
				"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
				quote_identifier(feature_col_cpu),
				quote_identifier(label_col_cpu),
				quote_identifier(table_name_cpu),
				quote_identifier(feature_col_cpu),
				quote_identifier(label_col_cpu));

			/* Load training data using nested SPI (PostgreSQL supports this) */
			if (SPI_connect() != SPI_OK_CONNECT)
			{
				pfree(query.data);
				if (metrics)
					pfree(metrics);
				if (model_data)
					pfree(model_data);
				if (tp_out)
					*tp_out = 0;
				if (tn_out)
					*tn_out = 0;
				if (fp_out)
					*fp_out = 0;
				if (fn_out)
					*fn_out = 0;
				return;
			}

			ret = SPI_execute(query.data, true, 0);
			if (ret == SPI_OK_SELECT)
			{
				n_train = SPI_processed;
				if (n_train >= k_value)
				{
					TupleDesc tupdesc;
					Oid feat_type_oid;
					bool feat_is_array;
					MemoryContext nested_context;

					/* Save nested SPI context */
					nested_context = CurrentMemoryContext;

					/* Allocate in outer context so it survives SPI_finish() */
					MemoryContextSwitchTo(outer_context);
					train_features = (float *)palloc(sizeof(float) * (size_t)n_train * (size_t)feature_dim);
					train_labels = (double *)palloc(sizeof(double) * (size_t)n_train);
					MemoryContextSwitchTo(nested_context); /* Back to nested SPI context */

					tupdesc = SPI_tuptable->tupdesc;
					feat_type_oid = SPI_gettypeid(tupdesc, 1);
					feat_is_array = (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID);

					for (i = 0; i < n_train; i++)
					{
						HeapTuple tuple = SPI_tuptable->vals[i];
						Datum feat_datum;
						Datum label_datum;
						bool feat_null;
						bool label_null;
						Vector *vec;
						ArrayType *arr;
						float *train_row;

						feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
						label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);

						if (feat_null || label_null)
							continue;

						train_row = train_features + (train_valid * feature_dim);
						train_labels[train_valid] = DatumGetFloat8(label_datum);

						if (feat_is_array)
						{
							arr = DatumGetArrayTypeP(feat_datum);
							if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feature_dim)
								continue;
							if (feat_type_oid == FLOAT8ARRAYOID)
							{
								float8 *data = (float8 *)ARR_DATA_PTR(arr);
								for (j = 0; j < feature_dim; j++)
									train_row[j] = (float)data[j];
							}
							else
							{
								float4 *data = (float4 *)ARR_DATA_PTR(arr);
								memcpy(train_row, data, sizeof(float) * feature_dim);
							}
						}
						else
						{
							vec = DatumGetVector(feat_datum);
							if (vec == NULL || vec->dim != feature_dim)
								continue;
							memcpy(train_row, vec->data, sizeof(float) * feature_dim);
						}

						train_valid++;
					}
				}
			}

			SPI_finish();
			/* Free query.data after SPI_finish() - it's allocated in outer context */
			pfree(query.data);

			if (train_valid < k_value)
			{
				if (train_features)
					pfree(train_features);
				if (train_labels)
					pfree(train_labels);
				if (metrics)
					pfree(metrics);
				if (model_data)
					pfree(model_data);
				if (tp_out)
					*tp_out = 0;
				if (tn_out)
					*tn_out = 0;
				if (fp_out)
					*fp_out = 0;
				if (fn_out)
					*fn_out = 0;
				return;
			}

			/* For each test sample, compute distances and predict */
			for (i = 0; i < n_samples; i++)
			{
				const float *row = features + (i * feature_dim);
				double y_true = labels[i];
				int true_class;
				double prediction = 0.0;
				int pred_class;
				KNNSample *samples_local = NULL;

				if (!isfinite(y_true))
					continue;

				true_class = (int)rint(y_true);
				if (true_class < 0)
					true_class = 0;
				if (true_class > 1)
					true_class = 1;

				/* Compute distances to all training samples */
				samples_local = (KNNSample *)palloc(sizeof(KNNSample) * train_valid);
				for (j = 0; j < train_valid; j++)
				{
					samples_local[j].features = train_features + (j * feature_dim);
					samples_local[j].label = train_labels[j];
					samples_local[j].distance = euclidean_distance(
						row, samples_local[j].features, feature_dim);
					samples_local[j].dim = feature_dim;
				}

				/* Sort by distance */
				qsort(samples_local, train_valid, sizeof(KNNSample), compare_samples);

				/* Majority vote */
				{
					double class_votes[2] = {0.0, 0.0};
					int k;

					for (k = 0; k < k_value && k < train_valid; k++)
					{
						int label = (int)samples_local[k].label;
						if (label >= 0 && label < 2)
							class_votes[label] += 1.0;
					}
					prediction = (class_votes[1] > class_votes[0]) ? 1.0 : 0.0;
				}

				pred_class = (int)rint(prediction);
				if (pred_class < 0)
					pred_class = 0;
				if (pred_class > 1)
					pred_class = 1;

				pfree(samples_local);

				/* Update confusion matrix */
				if (true_class == 1 && pred_class == 1)
					tp++;
				else if (true_class == 0 && pred_class == 0)
					tn++;
				else if (true_class == 0 && pred_class == 1)
					fp++;
				else if (true_class == 1 && pred_class == 0)
					fn++;
			}

			if (train_features)
				pfree(train_features);
			if (train_labels)
				pfree(train_labels);
		}
	}

	if (metrics)
		pfree(metrics);
	if (model_data)
		pfree(model_data);

	if (tp_out)
		*tp_out = tp;
	if (tn_out)
		*tn_out = tn;
	if (fp_out)
		*fp_out = fp;
	if (fn_out)
		*fn_out = fn;
}

/*
 * evaluate_knn_by_model_id
 *
 * Evaluates KNN model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: accuracy, precision, recall, f1_score, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_knn_by_model_id);

Datum
evaluate_knn_by_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	text *table_name;
	text *feature_col;
	text *label_col;
	char *tbl_str;
	char *feat_str;
	char *targ_str;
	int ret;
	int nvec = 0;
	int i;
	int j;
	int valid_rows = 0;
	Oid feat_type_oid = InvalidOid;
	bool feat_is_array = false;
	double accuracy = 0.0;
	double precision = 0.0;
	double recall = 0.0;
	double f1_score = 0.0;
	int tp = 0;
	int tn = 0;
	int fp = 0;
	int fn = 0;
	MemoryContext oldcontext;
	MemoryContext callcontext;
	StringInfoData query;
	StringInfoData jsonbuf;
	Jsonb *result_jsonb = NULL;
	bytea *gpu_payload = NULL;
	Jsonb *gpu_metrics = NULL;
	bool is_gpu_model = false;
	int feat_dim = 0;
	/* BULLETPROOF: Copy strings immediately to preserve them for error messages */
	char *tbl_str_copy = NULL;
	char *feat_str_copy = NULL;
	char *targ_str_copy = NULL;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_knn_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_knn_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;
	
	/* BULLETPROOF: Copy strings immediately to preserve them for error messages */
	tbl_str_copy = pstrdup(tbl_str);
	feat_str_copy = pstrdup(feat_str);
	targ_str_copy = pstrdup(targ_str);

	/* Build query in caller context, before SPI */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quote_identifier(feat_str),
		quote_identifier(targ_str),
		quote_identifier(tbl_str),
		quote_identifier(feat_str),
		quote_identifier(targ_str));

	/* Load model from catalog to determine feature dimension */
	if (!ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
	{
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_knn_by_model_id: model %d not found",
					model_id)));
	}

	/* Check if GPU model and extract feature dimension */
	if (gpu_payload != NULL)
	{
		const char *base = VARDATA(gpu_payload);
		int first_int = *(int *)base;

		if (first_int >= 1 && first_int <= 1000)
		{
			/* CPU format: k, n_samples, n_features */
			int k_cpu, n_samples_cpu, n_features_cpu;

			memcpy(&k_cpu, base, sizeof(int));
			memcpy(&n_samples_cpu, base + sizeof(int), sizeof(int));
			memcpy(&n_features_cpu, base + sizeof(int) * 2, sizeof(int));
			feat_dim = n_features_cpu;
			is_gpu_model = false;
		}
		else
		{
			/* GPU format */
			const NdbCudaKnnModelHeader *gpu_hdr;

			if (VARSIZE(gpu_payload) - VARHDRSZ >= (int)sizeof(NdbCudaKnnModelHeader))
			{
				gpu_hdr = (const NdbCudaKnnModelHeader *)base;
				feat_dim = gpu_hdr->n_features;
				is_gpu_model = true;
			}
			else
			{
				pfree(query.data);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(targ_str);
				if (gpu_payload)
					pfree(gpu_payload);
				if (gpu_metrics)
					pfree(gpu_metrics);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_knn_by_model_id: model %d has invalid header",
							model_id)));
			}
		}
	}

	if (feat_dim <= 0)
	{
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		if (gpu_payload)
			pfree(gpu_payload);
		if (gpu_metrics)
			pfree(gpu_metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_knn_by_model_id: could not determine feature dimension")));
	}

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
	{
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		if (gpu_payload)
			pfree(gpu_payload);
		if (gpu_metrics)
			pfree(gpu_metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_knn_by_model_id: SPI_connect failed")));
	}

	/* Save SPI context for later cleanup */
	callcontext = CurrentMemoryContext;

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		if (gpu_payload)
			pfree(gpu_payload);
		if (gpu_metrics)
			pfree(gpu_metrics);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_knn_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		/* Check if table/view exists and has any rows at all */
		StringInfoData check_query;
		int check_ret;
		int total_rows = 0;
		
		initStringInfo(&check_query);
		appendStringInfo(&check_query,
			"SELECT COUNT(*) FROM %s",
			quote_identifier(tbl_str));
		check_ret = SPI_execute(check_query.data, true, 0);
		if (check_ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool isnull;
			Datum count_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
			if (!isnull)
				total_rows = DatumGetInt64(count_datum);
		}
		pfree(check_query.data);
		
		pfree(query.data);
		/* Free original strings before SPI_finish() */
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		if (gpu_payload)
			pfree(gpu_payload);
		if (gpu_metrics)
			pfree(gpu_metrics);
		SPI_finish();
		
		/* Now use copies for error messages */
		if (total_rows == 0)
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_knn_by_model_id: table/view '%s' has no rows",
						tbl_str_copy),
					errhint("Ensure the table/view exists and contains data")));
		}
		else
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_knn_by_model_id: no valid rows found in '%s' (query returned %d rows, but all were filtered out due to NULL values)",
						tbl_str_copy, total_rows),
					errhint("Ensure columns '%s' and '%s' are not NULL",
						feat_str_copy, targ_str_copy)));
		}
	}

	/* Determine feature column type */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/* GPU batch evaluation path for GPU models - uses optimized evaluation kernel */
	if (is_gpu_model && neurondb_gpu_is_available())
	{
#ifdef NDB_GPU_CUDA
		const NdbCudaKnnModelHeader *gpu_hdr;
		double *h_labels = NULL;
		float *h_features = NULL;
		size_t payload_size;
		valid_rows = 0;

		/* Defensive check: validate payload size */
		payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
		if (payload_size < sizeof(NdbCudaKnnModelHeader))
		{
				 elog(DEBUG1,
				 	"neurondb: evaluate_knn_by_model_id: GPU payload too small (%zu bytes), falling back to CPU",
				 payload_size);
			goto cpu_evaluation_path;
		}

		/* Load GPU model header with defensive checks */
		gpu_hdr = (const NdbCudaKnnModelHeader *)VARDATA(gpu_payload);
		if (gpu_hdr == NULL)
		{
			elog(DEBUG1,
			     "neurondb: evaluate_knn_by_model_id: NULL GPU header, falling back to CPU");
			goto cpu_evaluation_path;
		}

		if (gpu_hdr->n_features != feat_dim)
		{
				 elog(DEBUG1,
				 	"neurondb: evaluate_knn_by_model_id: feature dimension mismatch (model has %d, expected %d), falling back to CPU",
				 gpu_hdr->n_features, feat_dim);
			goto cpu_evaluation_path;
		}

		if (gpu_hdr->n_features <= 0 || gpu_hdr->n_features > 100000)
		{
				 elog(DEBUG1,
				 	"evaluate_knn_by_model_id: invalid feature_dim (%d), falling back to CPU",
				 gpu_hdr->n_features);
			goto cpu_evaluation_path;
		}

		/* Allocate host buffers for features and labels with size checks */
		{
			size_t features_size = sizeof(float) * (size_t)nvec * (size_t)feat_dim;
			size_t labels_size = sizeof(double) * (size_t)nvec;

			if (features_size > MaxAllocSize || labels_size > MaxAllocSize)
			{
					 elog(DEBUG1,
					 	"neurondb: evaluate_knn_by_model_id: allocation size too large (features=%zu, labels=%zu), falling back to CPU",
					 features_size, labels_size);
				goto cpu_evaluation_path;
			}

			h_features = (float *)palloc(features_size);
			h_labels = (double *)palloc(labels_size);

			if (h_features == NULL || h_labels == NULL)
			{
				elog(DEBUG1,
					"neurondb: evaluate_knn_by_model_id: memory allocation failed, falling back to CPU");
				if (h_features)
					pfree(h_features);
				if (h_labels)
					pfree(h_labels);
				goto cpu_evaluation_path;
			}
		}

		/* Extract features and labels from SPI results - optimized batch extraction */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc tupdesc = SPI_tuptable->tupdesc;

			if (tupdesc == NULL)
			{
				elog(DEBUG1,
				     "neurondb: evaluate_knn_by_model_id: NULL TupleDesc, falling back to CPU");
				pfree(h_features);
				pfree(h_labels);
				goto cpu_evaluation_path;
			}

			for (i = 0; i < nvec; i++)
			{
				HeapTuple tuple;
				Datum feat_datum;
				Datum targ_datum;
				bool feat_null;
				bool targ_null;
				Vector *vec;
				ArrayType *arr;
				float *feat_row;

				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || i >= SPI_processed)
					break;

				tuple = SPI_tuptable->vals[i];
				if (tuple == NULL)
					continue;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				/* Bounds check */
				if (valid_rows >= nvec)
				{
					elog(DEBUG1,
					     "neurondb: evaluate_knn_by_model_id: valid_rows overflow, breaking");
					break;
				}

				feat_row = h_features + (valid_rows * feat_dim);
				if (feat_row == NULL || feat_row < h_features || feat_row >= h_features + (nvec * feat_dim))
				{
					elog(DEBUG1,
					     "neurondb: evaluate_knn_by_model_id: feat_row out of bounds, skipping row");
					continue;
				}

				h_labels[valid_rows] = DatumGetFloat8(targ_datum);

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						/* Optimized: bulk conversion with loop unrolling hint */
						float8 *data = (float8 *)ARR_DATA_PTR(arr);
						int j_remain = feat_dim % 4;
						int j_end = feat_dim - j_remain;

						/* Process 4 elements at a time for better cache locality */
						for (j = 0; j < j_end; j += 4)
						{
							feat_row[j] = (float)data[j];
							feat_row[j + 1] = (float)data[j + 1];
							feat_row[j + 2] = (float)data[j + 2];
							feat_row[j + 3] = (float)data[j + 3];
						}
						/* Handle remaining elements */
						for (j = j_end; j < feat_dim; j++)
							feat_row[j] = (float)data[j];
					}
					else
					{
						/* FLOAT4ARRAYOID: direct memcpy (already optimal) */
						float4 *data = (float4 *)ARR_DATA_PTR(arr);
						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					/* Vector type: direct memcpy (already optimal) */
					vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_rows++;
			}
		}

		if (valid_rows == 0)
		{
			pfree(h_features);
			pfree(h_labels);
			if (gpu_payload)
				pfree(gpu_payload);
			if (gpu_metrics)
				pfree(gpu_metrics);
			pfree(query.data);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_knn_by_model_id: no valid rows found")));
		}

		/* Use optimized GPU batch evaluation */
		{
			int rc;
			char *gpu_errstr = NULL;

			/* Defensive checks before GPU call */
			if (h_features == NULL || h_labels == NULL || valid_rows <= 0 || feat_dim <= 0)
			{
					 elog(DEBUG1,
					 	"neurondb: evaluate_knn_by_model_id: invalid inputs for GPU evaluation (features=%p, labels=%p, rows=%d, dim=%d), falling back to CPU",
					 (void *)h_features, (void *)h_labels, valid_rows, feat_dim);
				pfree(h_features);
				pfree(h_labels);
				goto cpu_evaluation_path;
			}

			PG_TRY();
			{
				rc = ndb_cuda_knn_evaluate_batch(gpu_payload,
					h_features,
					h_labels,
					valid_rows,
					feat_dim,
					&accuracy,
					&precision,
					&recall,
					&f1_score,
					&gpu_errstr);

				if (rc == 0)
				{
					/* Success - build result and return */
					initStringInfo(&jsonbuf);
					appendStringInfo(&jsonbuf,
						"{\"accuracy\":%.6f,\"precision\":%.6f,\"recall\":%.6f,\"f1_score\":%.6f,\"n_samples\":%d}",
						accuracy,
						precision,
						recall,
						f1_score,
						valid_rows);

					result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
						CStringGetDatum(jsonbuf.data)));

					pfree(jsonbuf.data);
					
					/* BULLETPROOF: Copy result_jsonb to caller's context before SPI_finish() */
					MemoryContextSwitchTo(oldcontext);
					result_jsonb = (Jsonb *)PG_DETOAST_DATUM_COPY(JsonbPGetDatum(result_jsonb));
					
					/* Now safe to finish SPI and free SPI-allocated memory */
					MemoryContextSwitchTo(callcontext);
					pfree(h_features);
					pfree(h_labels);
					if (gpu_payload)
						pfree(gpu_payload);
					if (gpu_metrics)
						pfree(gpu_metrics);
					/* Note: gpu_errstr is allocated by GPU function, don't free it */
					pfree(query.data);
					SPI_finish();
					
					/* Switch back to oldcontext and free strings allocated before SPI_connect */
					MemoryContextSwitchTo(oldcontext);
					pfree(tbl_str);
					pfree(feat_str);
					pfree(targ_str);
					
					PG_RETURN_JSONB_P(result_jsonb);
				}
				else
				{
					/* GPU evaluation failed - fall back to CPU */
						 elog(DEBUG1,
						 	"neurondb: evaluate_knn_by_model_id: GPU batch evaluation failed: %s, falling back to CPU",
						 gpu_errstr ? gpu_errstr : "unknown error");
					if (gpu_errstr)
						pfree(gpu_errstr);
					pfree(h_features);
					pfree(h_labels);
					goto cpu_evaluation_path;
				}
			}
			PG_CATCH();
			{
				elog(DEBUG1,
				     "neurondb: evaluate_knn_by_model_id: exception during GPU evaluation, falling back to CPU");
				if (h_features)
					pfree(h_features);
				if (h_labels)
					pfree(h_labels);
				goto cpu_evaluation_path;
			}
			PG_END_TRY();
		}
#endif	/* NDB_GPU_CUDA */
	}

cpu_evaluation_path:

	/* CPU evaluation path */
	/* Use optimized batch prediction */
	{
		float *h_features = NULL;
		double *h_labels = NULL;
		valid_rows = 0;

		/* Allocate host buffers for features and labels */
		h_features = (float *)palloc(sizeof(float) * (size_t)nvec * (size_t)feat_dim);
		h_labels = (double *)palloc(sizeof(double) * (size_t)nvec);

		/* Extract features and labels from SPI results - optimized batch extraction */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc tupdesc = SPI_tuptable->tupdesc;

			for (i = 0; i < nvec; i++)
			{
				HeapTuple tuple = SPI_tuptable->vals[i];
				Datum feat_datum;
				Datum targ_datum;
				bool feat_null;
				bool targ_null;
				Vector *vec;
				ArrayType *arr;
				float *feat_row;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				feat_row = h_features + (valid_rows * feat_dim);
				h_labels[valid_rows] = DatumGetFloat8(targ_datum);

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						/* Optimized: bulk conversion with loop unrolling hint */
						float8 *data = (float8 *)ARR_DATA_PTR(arr);
						int j_remain = feat_dim % 4;
						int j_end = feat_dim - j_remain;

						/* Process 4 elements at a time for better cache locality */
						for (j = 0; j < j_end; j += 4)
						{
							feat_row[j] = (float)data[j];
							feat_row[j + 1] = (float)data[j + 1];
							feat_row[j + 2] = (float)data[j + 2];
							feat_row[j + 3] = (float)data[j + 3];
						}
						/* Handle remaining elements */
						for (j = j_end; j < feat_dim; j++)
							feat_row[j] = (float)data[j];
					}
					else
					{
						/* FLOAT4ARRAYOID: direct memcpy (already optimal) */
						float4 *data = (float4 *)ARR_DATA_PTR(arr);
						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					/* Vector type: direct memcpy (already optimal) */
					vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_rows++;
			}
		}

		if (valid_rows == 0)
		{
			/* Check if we had rows but they were all filtered out */
			if (nvec > 0)
			{
				/* Try to determine why rows were filtered */
				HeapTuple tuple = SPI_tuptable->vals[0];
				TupleDesc tupdesc = SPI_tuptable->tupdesc;
				Datum feat_datum;
				bool feat_null;
				Vector *vec;
				int actual_dim = 0;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				if (!feat_null)
				{
					if (feat_is_array)
					{
						ArrayType *arr = DatumGetArrayTypeP(feat_datum);
						if (ARR_NDIM(arr) == 1)
							actual_dim = ARR_DIMS(arr)[0];
					}
					else
					{
						vec = DatumGetVector(feat_datum);
						if (vec != NULL)
							actual_dim = vec->dim;
					}
				}

				pfree(h_features);
				pfree(h_labels);
				if (gpu_payload)
					pfree(gpu_payload);
				if (gpu_metrics)
					pfree(gpu_metrics);
				pfree(query.data);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(targ_str);
				SPI_finish();
				if (actual_dim > 0 && actual_dim != feat_dim)
				{
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("neurondb: evaluate_knn_by_model_id: feature dimension mismatch"),
							errdetail("Model expects %d features but test data has %d features. All %d rows were filtered out due to dimension mismatch.",
								feat_dim,
								actual_dim,
								nvec)));
				}
				else
				{
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("neurondb: evaluate_knn_by_model_id: no valid rows found"),
							errdetail("Query returned %d rows but all were filtered out (NULL values or dimension mismatches).",
								nvec)));
				}
			}
			else
			{
				pfree(h_features);
				pfree(h_labels);
				if (gpu_payload)
					pfree(gpu_payload);
				if (gpu_metrics)
					pfree(gpu_metrics);
				pfree(query.data);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(targ_str);
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_knn_by_model_id: no valid rows found"),
						errdetail("Query returned 0 rows from table '%s'.", tbl_str_copy)));
			}
		}

		/* Use batch prediction helper */
		knn_predict_batch(model_id,
			h_features,
			h_labels,
			valid_rows,
			feat_dim,
			&tp,
			&tn,
			&fp,
			&fn);

		/* Compute metrics */
		if (valid_rows > 0)
		{
			int total_predictions = tp + tn + fp + fn;
			
			/* Defensive check: ensure we have predictions */
			if (total_predictions == 0)
			{
					elog(DEBUG1,
						"neurondb: evaluate_knn_by_model_id: no valid predictions made for %d valid rows (model_id=%d, feature_dim=%d)",
					valid_rows, model_id, feat_dim);
				accuracy = 0.0;
				precision = 0.0;
				recall = 0.0;
				f1_score = 0.0;
			}
			else
			{
				/* Use valid_rows for accuracy (original behavior) */
				accuracy = (double)(tp + tn) / (double)valid_rows;
				
				/* Warn if some rows were skipped during prediction */
				if (total_predictions < valid_rows)
				{
						elog(DEBUG1,
							"neurondb: evaluate_knn_by_model_id: %d rows skipped during prediction (labels out of range?)",
						valid_rows - total_predictions);
				}

				if ((tp + fp) > 0)
					precision = (double)tp / (double)(tp + fp);
				else
					precision = 0.0;

				if ((tp + fn) > 0)
					recall = (double)tp / (double)(tp + fn);
				else
					recall = 0.0;

				if ((precision + recall) > 0.0)
					f1_score = 2.0 * (precision * recall) / (precision + recall);
				else
					f1_score = 0.0;
			}
		}

		/* Cleanup */
		pfree(h_features);
		pfree(h_labels);
		if (gpu_payload)
			pfree(gpu_payload);
		if (gpu_metrics)
			pfree(gpu_metrics);
	}

	/* Build jsonb result in SPI context */
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"accuracy\":%.6f,\"precision\":%.6f,\"recall\":%.6f,\"f1_score\":%.6f,\"n_samples\":%d}",
		accuracy,
		precision,
		recall,
		f1_score,
		valid_rows > 0 ? valid_rows : nvec);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
		CStringGetDatum(jsonbuf.data)));

	pfree(jsonbuf.data);
	
	/* BULLETPROOF: Copy result_jsonb to caller's context before SPI_finish() */
	MemoryContextSwitchTo(oldcontext);
	result_jsonb = (Jsonb *)PG_DETOAST_DATUM_COPY(JsonbPGetDatum(result_jsonb));
	
	/* Now safe to finish SPI and free SPI-allocated memory */
	SPI_finish();
	pfree(query.data);
	
	/* Free strings allocated before SPI_connect (in oldcontext) */
	pfree(tbl_str);
	pfree(feat_str);
	pfree(targ_str);
	
	/* Free string copies used for error messages */
	if (tbl_str_copy)
		pfree(tbl_str_copy);
	if (feat_str_copy)
		pfree(feat_str_copy);
	if (targ_str_copy)
		pfree(targ_str_copy);
	
	PG_RETURN_JSONB_P(result_jsonb);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for KNN
 *-------------------------------------------------------------------------
 */

/* Stub function to satisfy linker - full implementation needed */
void
neurondb_gpu_register_knn_model(void)
{
	/* KNN GPU Model Ops not yet implemented - will use CPU fallback */
	return;
}
