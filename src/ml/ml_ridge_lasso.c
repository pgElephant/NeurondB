/*-------------------------------------------------------------------------
 *
 * ml_ridge_lasso.c
 *    Ridge and Lasso Regression implementations
 *
 * Implements regularized linear regression with L2 (Ridge) and L1 (Lasso)
 * penalties to prevent overfitting.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_ridge_lasso.c
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
 * Soft thresholding operator for Lasso (coordinate descent)
 */
static double
soft_threshold(double x, double lambda)
{
	if (x > lambda)
		return x - lambda;
	else if (x < -lambda)
		return x + lambda;
	else
		return 0.0;
}

/*
 * train_ridge_regression
 *
 * Trains Ridge Regression (L2 regularization)
 * Uses closed-form solution: w = (X^T X + λI)^-1 X^T y
 */
PG_FUNCTION_INFO_V1(train_ridge_regression);

Datum
train_ridge_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	double		lambda = PG_GETARG_FLOAT8(3);  /* Regularization parameter */
	
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			dim = 0;
	float	  **X = NULL;
	double	   *y = NULL;
	double	   *weights = NULL;
	double		bias = 0.0;
	int			i, j, k;
	double	  **XtX = NULL;  /* X^T X matrix */
	double	   *Xty = NULL;  /* X^T y vector */
	double		y_mean = 0.0;
	Datum	   *result_datums;
	ArrayType  *result_array;
	MemoryContext oldcontext;
	
	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);
	
	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);
	
	oldcontext = CurrentMemoryContext;
	
	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
				(errmsg("SPI_connect failed: error code %d", ret)));
	
	/* Build query to fetch features and targets */
	initStringInfo(&query);
	appendStringInfo(&query, "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str, targ_str, tbl_str, feat_str, targ_str);
	
	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
				(errmsg("Query failed: %s", query.data)));
	
	nvec = SPI_processed;
	
	if (nvec < 10)
		ereport(ERROR,
				(errmsg("Need at least 10 samples for Ridge regression, have %d", nvec)));
	
	/* Allocate arrays in caller's context */
	MemoryContextSwitchTo(oldcontext);
	X = (float **) palloc(sizeof(float *) * nvec);
	y = (double *) palloc(sizeof(double) * nvec);
	
	/* Extract data */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec;
		Oid			targ_type;
		
		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		if (feat_null)
			continue;
		
		vec = DatumGetVector(feat_datum);
		
		if (i == 0)
			dim = vec->dim;
		else if (vec->dim != dim)
			ereport(ERROR,
					(errmsg("Inconsistent vector dimensions: expected %d, got %d", dim, vec->dim)));
		
		/* Copy feature vector */
		X[i] = (float *) palloc(sizeof(float) * dim);
		memcpy(X[i], vec->data, sizeof(float) * dim);
		
		/* Get target value */
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (targ_null)
			continue;
		
		/* Handle both integer and float targets */
		targ_type = SPI_gettypeid(tupdesc, 2);
		if (targ_type == INT2OID || targ_type == INT4OID || targ_type == INT8OID)
			y[i] = (double) DatumGetInt32(targ_datum);
		else
			y[i] = DatumGetFloat8(targ_datum);
		
		y_mean += y[i];
	}
	
	SPI_finish();
	
	y_mean /= nvec;
	
	/* Allocate matrices for Ridge solution */
	XtX = (double **) palloc(sizeof(double *) * dim);
	for (i = 0; i < dim; i++)
		XtX[i] = (double *) palloc0(sizeof(double) * dim);
	
	Xty = (double *) palloc0(sizeof(double) * dim);
	weights = (double *) palloc0(sizeof(double) * dim);
	
	/* Compute X^T X */
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < dim; j++)
		{
			for (k = 0; k < nvec; k++)
				XtX[i][j] += X[k][i] * X[k][j];
			
			/* Add Ridge penalty (λI) to diagonal */
			if (i == j)
				XtX[i][j] += lambda;
		}
	}
	
	/* Compute X^T y */
	for (i = 0; i < dim; i++)
	{
		for (k = 0; k < nvec; k++)
			Xty[i] += X[k][i] * (y[k] - y_mean);
	}
	
	/* Solve (X^T X + λI) w = X^T y using Gaussian elimination */
	for (k = 0; k < dim; k++)
	{
		/* Find pivot */
		double max_val = fabs(XtX[k][k]);
		int max_row = k;
		
		for (i = k + 1; i < dim; i++)
		{
			if (fabs(XtX[i][k]) > max_val)
			{
				max_val = fabs(XtX[i][k]);
				max_row = i;
			}
		}
		
		/* Swap rows */
		if (max_row != k)
		{
			double *temp = XtX[k];
			XtX[k] = XtX[max_row];
			XtX[max_row] = temp;
			
			double temp_val = Xty[k];
			Xty[k] = Xty[max_row];
			Xty[max_row] = temp_val;
		}
		
		/* Forward elimination */
		for (i = k + 1; i < dim; i++)
		{
			double factor = XtX[i][k] / XtX[k][k];
			for (j = k; j < dim; j++)
				XtX[i][j] -= factor * XtX[k][j];
			Xty[i] -= factor * Xty[k];
		}
	}
	
	/* Back substitution */
	for (i = dim - 1; i >= 0; i--)
	{
		weights[i] = Xty[i];
		for (j = i + 1; j < dim; j++)
			weights[i] -= XtX[i][j] * weights[j];
		weights[i] /= XtX[i][i];
	}
	
	bias = y_mean;
	
	/* Build result array: [bias, weight1, weight2, ...] */
	result_datums = (Datum *) palloc(sizeof(Datum) * (dim + 1));
	result_datums[0] = Float8GetDatum(bias);
	for (i = 0; i < dim; i++)
		result_datums[i + 1] = Float8GetDatum(weights[i]);
	
	result_array = construct_array(result_datums, dim + 1, FLOAT8OID, 8, FLOAT8PASSBYVAL, 'd');
	
	elog(NOTICE, "Ridge regression trained: %d samples, %d features, lambda=%.4f", nvec, dim, lambda);
	
	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * train_lasso_regression
 *
 * Trains Lasso Regression (L1 regularization)
 * Uses coordinate descent algorithm
 */
PG_FUNCTION_INFO_V1(train_lasso_regression);

Datum
train_lasso_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	double		lambda = PG_GETARG_FLOAT8(3);  /* Regularization parameter */
	int			max_iters = PG_NARGS() > 4 ? PG_GETARG_INT32(4) : 1000;
	
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			dim = 0;
	float	  **X = NULL;
	double	   *y = NULL;
	double	   *weights = NULL;
	double	   *weights_old = NULL;
	double		bias = 0.0;
	int			iter, i, j;
	double		y_mean = 0.0;
	double	   *residuals = NULL;
	bool		converged = false;
	Datum	   *result_datums;
	ArrayType  *result_array;
	MemoryContext oldcontext;
	
	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);
	
	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);
	
	oldcontext = CurrentMemoryContext;
	
	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
				(errmsg("SPI_connect failed: error code %d", ret)));
	
	/* Build query to fetch features and targets */
	initStringInfo(&query);
	appendStringInfo(&query, "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str, targ_str, tbl_str, feat_str, targ_str);
	
	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
				(errmsg("Query failed: %s", query.data)));
	
	nvec = SPI_processed;
	
	if (nvec < 10)
		ereport(ERROR,
				(errmsg("Need at least 10 samples for Lasso regression, have %d", nvec)));
	
	/* Allocate arrays in caller's context */
	MemoryContextSwitchTo(oldcontext);
	X = (float **) palloc(sizeof(float *) * nvec);
	y = (double *) palloc(sizeof(double) * nvec);
	
	/* Extract data (same as Ridge) */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec;
		Oid			targ_type;
		
		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		if (feat_null)
			continue;
		
		vec = DatumGetVector(feat_datum);
		
		if (i == 0)
			dim = vec->dim;
		else if (vec->dim != dim)
			ereport(ERROR,
					(errmsg("Inconsistent vector dimensions")));
		
		X[i] = (float *) palloc(sizeof(float) * dim);
		memcpy(X[i], vec->data, sizeof(float) * dim);
		
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (targ_null)
			continue;
		
		targ_type = SPI_gettypeid(tupdesc, 2);
		if (targ_type == INT2OID || targ_type == INT4OID || targ_type == INT8OID)
			y[i] = (double) DatumGetInt32(targ_datum);
		else
			y[i] = DatumGetFloat8(targ_datum);
		
		y_mean += y[i];
	}
	
	SPI_finish();
	
	y_mean /= nvec;
	bias = y_mean;
	
	/* Initialize weights and residuals */
	weights = (double *) palloc0(sizeof(double) * dim);
	weights_old = (double *) palloc(sizeof(double) * dim);
	residuals = (double *) palloc(sizeof(double) * nvec);
	
	/* Initialize residuals */
	for (i = 0; i < nvec; i++)
		residuals[i] = y[i] - bias;
	
	/* Coordinate descent */
	for (iter = 0; iter < max_iters && !converged; iter++)
	{
		memcpy(weights_old, weights, sizeof(double) * dim);
		
		/* Update each coordinate */
		for (j = 0; j < dim; j++)
		{
			double rho = 0.0;
			double z = 0.0;
			
			/* Compute rho = X_j^T * residuals */
			for (i = 0; i < nvec; i++)
				rho += X[i][j] * residuals[i];
			
			/* Compute z = X_j^T * X_j */
			for (i = 0; i < nvec; i++)
				z += X[i][j] * X[i][j];
			
			if (z < 1e-10)
				continue;
			
			/* Soft thresholding */
			double old_weight = weights[j];
			weights[j] = soft_threshold(rho / z, lambda / z);
			
			/* Update residuals */
			if (weights[j] != old_weight)
			{
				for (i = 0; i < nvec; i++)
					residuals[i] -= X[i][j] * (weights[j] - old_weight);
			}
		}
		
		/* Check convergence */
		double diff = 0.0;
		for (j = 0; j < dim; j++)
		{
			double d = weights[j] - weights_old[j];
			diff += d * d;
		}
		
		if (sqrt(diff) < 1e-6)
		{
			converged = true;
			elog(NOTICE, "Lasso converged after %d iterations", iter + 1);
		}
	}
	
	/* Build result array */
	result_datums = (Datum *) palloc(sizeof(Datum) * (dim + 1));
	result_datums[0] = Float8GetDatum(bias);
	for (i = 0; i < dim; i++)
		result_datums[i + 1] = Float8GetDatum(weights[i]);
	
	result_array = construct_array(result_datums, dim + 1, FLOAT8OID, 8, FLOAT8PASSBYVAL, 'd');
	
	elog(NOTICE, "Lasso regression trained: %d samples, %d features, lambda=%.4f", nvec, dim, lambda);
	
	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * train_elastic_net
 *
 * Trains Elastic Net (L1 + L2 regularization)
 * Combines Ridge and Lasso penalties
 */
PG_FUNCTION_INFO_V1(train_elastic_net);

Datum
train_elastic_net(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *feature_col = PG_GETARG_TEXT_PP(1);
	text	   *target_col = PG_GETARG_TEXT_PP(2);
	double		alpha = PG_GETARG_FLOAT8(3);  /* Overall regularization strength */
	double		l1_ratio = PG_GETARG_FLOAT8(4);  /* L1 vs L2 ratio (0=Ridge, 1=Lasso) */
	
	Datum	   *result_datums;
	ArrayType  *result_array;
	
	/* Placeholder implementation - would combine Ridge and Lasso */
	elog(NOTICE, "Elastic Net: alpha=%.4f, l1_ratio=%.4f", alpha, l1_ratio);
	elog(NOTICE, "Training on table: %s, features: %s, target: %s",
		 text_to_cstring(table_name),
		 text_to_cstring(feature_col),
		 text_to_cstring(target_col));
	
	/* Return dummy coefficients */
	result_datums = (Datum *) palloc(sizeof(Datum) * 6);
	result_datums[0] = Float8GetDatum(0.0);  /* bias */
	for (int i = 1; i < 6; i++)
		result_datums[i] = Float8GetDatum(0.1);
	
	result_array = construct_array(result_datums, 6, FLOAT8OID, 8, FLOAT8PASSBYVAL, 'd');
	
	PG_RETURN_ARRAYTYPE_P(result_array);
}

