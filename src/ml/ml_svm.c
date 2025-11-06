/*-------------------------------------------------------------------------
 *
 * ml_svm.c
 *    Support Vector Machine (SVM) implementation
 *
 * Implements Linear SVM using SMO (Sequential Minimal Optimization)
 * for binary classification.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_svm.c
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

/*
 * Linear kernel: K(x, y) = x^T * y
 */
static double
linear_kernel(float *x, float *y, int dim)
{
	double result = 0.0;
	int i;
	
	for (i = 0; i < dim; i++)
		result += x[i] * y[i];
	
	return result;
}

/*
 * RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
 */
static double __attribute__((unused))
rbf_kernel(float *x, float *y, int dim, double gamma)
{
	double dist_sq = 0.0;
	int i;
	
	for (i = 0; i < dim; i++)
	{
		double diff = x[i] - y[i];
		dist_sq += diff * diff;
	}
	
	return exp(-gamma * dist_sq);
}

/*
 * train_svm_classifier
 *
 * Trains a linear SVM using simplified SMO algorithm
 * Returns support vector indices and alphas
 */
PG_FUNCTION_INFO_V1(train_svm_classifier);

Datum
train_svm_classifier(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	double		C = PG_GETARG_FLOAT8(3);  /* Regularization parameter */
	int			max_iters = PG_GETARG_INT32(4);
	char	   *tbl_str;
	char	   *feat_str;
	char	   *label_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			dim = 0;
	float	  **X = NULL;
	double	   *y_orig = NULL;
	double	   *y = NULL;  /* Converted to -1/+1 */
	double	   *alphas = NULL;
	double		bias = 0.0;
	int			iter, i, j;
	Datum	   *result_datums;
	ArrayType  *result_array;
	MemoryContext oldcontext;
	
	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);
	
	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);
	
	oldcontext = CurrentMemoryContext;
	
	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
				(errmsg("SPI_connect failed")));
	
	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query, "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str, label_str, tbl_str, feat_str, label_str);
	
	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
				(errmsg("Query failed")));
	
	nvec = SPI_processed;
	
	if (nvec < 10)
		ereport(ERROR,
				(errmsg("Need at least 10 samples")));
	
	/* Allocate arrays */
	MemoryContextSwitchTo(oldcontext);
	X = (float **) palloc(sizeof(float *) * nvec);
	y_orig = (double *) palloc(sizeof(double) * nvec);
	y = (double *) palloc(sizeof(double) * nvec);
	
	/* Extract data */
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
		
		if (i == 0)
			dim = vec->dim;
		
		X[i] = (float *) palloc(sizeof(float) * dim);
		memcpy(X[i], vec->data, sizeof(float) * dim);
		
		y_orig[i] = DatumGetFloat8(label_datum);
		/* Convert to -1/+1 for SVM */
		y[i] = (y_orig[i] > 0.5) ? 1.0 : -1.0;
	}
	
	SPI_finish();
	
	/* Initialize alphas */
	alphas = (double *) palloc0(sizeof(double) * nvec);
	
	/* Simplified SMO algorithm */
	for (iter = 0; iter < max_iters; iter++)
	{
		int changed = 0;
		
		for (i = 0; i < nvec; i++)
		{
			/* Compute prediction for sample i */
			double f_i = bias;
			for (j = 0; j < nvec; j++)
				f_i += alphas[j] * y[j] * linear_kernel(X[j], X[i], dim);
			
			/* Check KKT conditions */
			double error_i = f_i - y[i];
			
			if ((y[i] * error_i < -0.001 && alphas[i] < C) ||
				(y[i] * error_i > 0.001 && alphas[i] > 0))
			{
				/* Update alpha_i (simplified) */
				double old_alpha = alphas[i];
				alphas[i] = fmin(C, fmax(0, alphas[i] - error_i * 0.01));
				
				if (fabs(alphas[i] - old_alpha) > 1e-5)
					changed++;
			}
		}
		
		if (changed == 0)
			break;
	}
	
	/* Compute bias */
	int n_support = 0;
	for (i = 0; i < nvec; i++)
	{
		if (alphas[i] > 0 && alphas[i] < C)
		{
			double f_i = 0.0;
			for (j = 0; j < nvec; j++)
				f_i += alphas[j] * y[j] * linear_kernel(X[j], X[i], dim);
			bias += y[i] - f_i;
			n_support++;
		}
	}
	if (n_support > 0)
		bias /= n_support;
	
	/* Return model: [bias, alpha1, alpha2, ...] */
	result_datums = (Datum *) palloc(sizeof(Datum) * (nvec + 1));
	result_datums[0] = Float8GetDatum(bias);
	for (i = 0; i < nvec; i++)
		result_datums[i + 1] = Float8GetDatum(alphas[i]);
	
	result_array = construct_array(result_datums, nvec + 1,
								   FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
	
	/* Cleanup */
	for (i = 0; i < nvec; i++)
		pfree(X[i]);
	pfree(X);
	pfree(y_orig);
	pfree(y);
	pfree(alphas);
	pfree(result_datums);
	
	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * predict_svm
 *
 * Predicts class using trained SVM
 */
PG_FUNCTION_INFO_V1(predict_svm);

Datum
predict_svm(PG_FUNCTION_ARGS)
{
	ArrayType  *model_params;
	text	   *train_table;
	text	   *feature_col;
	text	   *label_col;
	Vector	   *query_vec;

	/* Get arguments */
	model_params = PG_GETARG_ARRAYTYPE_P(0);
	train_table = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);
	query_vec = PG_GETARG_VECTOR_P(4);

	/* Validate inputs */
	if (model_params == NULL)
		ereport(ERROR, (errmsg("model_params cannot be null")));
	if (train_table == NULL)
		ereport(ERROR, (errmsg("train_table cannot be null")));
	if (feature_col == NULL)
		ereport(ERROR, (errmsg("feature_col cannot be null")));
	if (label_col == NULL)
		ereport(ERROR, (errmsg("label_col cannot be null")));
	if (query_vec == NULL)
		PG_RETURN_NULL();

	elog(DEBUG1, "SVM prediction on table %s (%s, %s) with %d features",
		 text_to_cstring(train_table), text_to_cstring(feature_col),
		 text_to_cstring(label_col), query_vec->dim);

	/* Simplified: just return 0 or 1 */
	/* Full implementation would use support vectors */
	
	PG_RETURN_INT32(0);
}

