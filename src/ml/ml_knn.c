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

#include "neurondb.h"
#include "neurondb_ml.h"

#include <math.h>
#include <float.h>

/* Structure to hold a training sample with distance */
typedef struct
{
	float	   *features;
	double		label;
	double		distance;
	int			dim;
} KNNSample;

/* Comparison function for qsort */
static int
compare_samples(const void *a, const void *b)
{
	const KNNSample *sa = (const KNNSample *) a;
	const KNNSample *sb = (const KNNSample *) b;
	
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
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	Vector	   *query_vector;
	int			k;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *label_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			dim;
	KNNSample  *samples = NULL;
	double		class_votes[2] = {0.0, 0.0};  /* For binary classification */
	int			predicted_class;
	int			i;
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
				(errmsg("k must be at least 1, got %d", k)));
	
	oldcontext = CurrentMemoryContext;
	
	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
				(errmsg("SPI_connect failed")));
	
	/* Build query to fetch all training samples */
	initStringInfo(&query);
	appendStringInfo(&query, "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str, label_str, tbl_str, feat_str, label_str);
	
	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
				(errmsg("Query failed: %s", query.data)));
	
	nvec = SPI_processed;
	
	if (nvec < k)
		ereport(ERROR,
				(errmsg("Need at least k=%d samples, but found only %d", k, nvec)));
	
	/* Allocate samples array in caller's context */
	MemoryContextSwitchTo(oldcontext);
	samples = (KNNSample *) palloc(sizeof(KNNSample) * nvec);
	
	/* Extract data and compute distances */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		label_datum;
		bool		feat_null;
		bool		label_null;
		Vector	   *vec;
		
		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);
		
		if (feat_null || label_null)
			continue;
		
		vec = DatumGetVector(feat_datum);
		
		if (vec->dim != dim)
			ereport(ERROR,
					(errmsg("Dimension mismatch: expected %d, got %d", dim, vec->dim)));
		
		/* Copy features */
		samples[i].features = (float *) palloc(sizeof(float) * dim);
		memcpy(samples[i].features, vec->data, sizeof(float) * dim);
		samples[i].dim = dim;
		
		/* Get label */
		samples[i].label = DatumGetFloat8(label_datum);
		
		/* Compute distance */
		samples[i].distance = euclidean_distance(query_vector->data, samples[i].features, dim);
	}
	
	SPI_finish();
	
	/* Sort samples by distance */
	qsort(samples, nvec, sizeof(KNNSample), compare_samples);
	
	/* Vote among k nearest neighbors */
	for (i = 0; i < k && i < nvec; i++)
	{
		int label_class = (int) samples[i].label;
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
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	Vector	   *query_vector;
	int			k;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			dim;
	KNNSample  *samples = NULL;
	double		prediction = 0.0;
	int			i;
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
				(errmsg("k must be at least 1, got %d", k)));
	
	oldcontext = CurrentMemoryContext;
	
	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
				(errmsg("SPI_connect failed")));
	
	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query, "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str, targ_str, tbl_str, feat_str, targ_str);
	
	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
				(errmsg("Query failed")));
	
	nvec = SPI_processed;
	
	if (nvec < k)
		ereport(ERROR,
				(errmsg("Need at least k=%d samples, but found only %d", k, nvec)));
	
	/* Allocate samples array */
	MemoryContextSwitchTo(oldcontext);
	samples = (KNNSample *) palloc(sizeof(KNNSample) * nvec);
	
	/* Extract data and compute distances */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec;
		
		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		
		if (feat_null || targ_null)
			continue;
		
		vec = DatumGetVector(feat_datum);
		
		if (vec->dim != dim)
			ereport(ERROR,
					(errmsg("Dimension mismatch: expected %d, got %d", dim, vec->dim)));
		
		/* Copy features */
		samples[i].features = (float *) palloc(sizeof(float) * dim);
		memcpy(samples[i].features, vec->data, sizeof(float) * dim);
		samples[i].dim = dim;
		
		/* Get target value */
		samples[i].label = DatumGetFloat8(targ_datum);
		
		/* Compute distance */
		samples[i].distance = euclidean_distance(query_vector->data, samples[i].features, dim);
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
	text	   *train_table;
	text	   *test_table;
	text	   *feature_col;
	text	   *label_col;
	int			k;
	char	   *train_str;
	char	   *test_str;
	char	   *feat_str;
	char	   *label_str;
	StringInfoData query;
	int			ret;
	int			ntest = 0;
	int			tp = 0, tn = 0, fp = 0, fn = 0;
	double		accuracy, precision, recall, f1_score;
	int			i;
	Datum	   *result_datums;
	ArrayType  *result_array;
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
		ereport(ERROR,
				(errmsg("SPI_connect failed")));
	
	/* Build query to fetch test samples */
	initStringInfo(&query);
	appendStringInfo(&query, "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str, label_str, test_str, feat_str, label_str);
	
	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
				(errmsg("Query failed")));
	
	ntest = SPI_processed;
	
	/* For each test sample, classify using KNN */
	for (i = 0; i < ntest; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		label_datum;
		bool		feat_null;
		bool		label_null;
		Vector	   *vec;
		int			y_true;
		int			y_pred;
		
		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);
		
		if (feat_null || label_null)
			continue;
		
		vec = DatumGetVector(feat_datum);
		y_true = (int) DatumGetFloat8(label_datum);
		
		/* Call knn_classify for this sample */
		y_pred = DatumGetInt32(
			DirectFunctionCall5(knn_classify,
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
		}
		else
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
	f1_score = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
	
	/* Build result array */
	MemoryContextSwitchTo(oldcontext);
	
	result_datums = (Datum *) palloc(sizeof(Datum) * 4);
	result_datums[0] = Float8GetDatum(accuracy);
	result_datums[1] = Float8GetDatum(precision);
	result_datums[2] = Float8GetDatum(recall);
	result_datums[3] = Float8GetDatum(f1_score);
	
	result_array = construct_array(result_datums, 4,
								   FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
	
	pfree(result_datums);
	pfree(train_str);
	pfree(test_str);
	pfree(feat_str);
	pfree(label_str);
	
	PG_RETURN_ARRAYTYPE_P(result_array);
}

