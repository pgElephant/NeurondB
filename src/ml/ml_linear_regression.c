/*-------------------------------------------------------------------------
 *
 * ml_linear_regression.c
 *    Linear Regression implementation for supervised learning
 *
 * Implements ordinary least squares (OLS) linear regression using
 * normal equations: β = (X'X)^(-1)X'y
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_linear_regression.c
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
 * Matrix inversion using Gauss-Jordan elimination
 * Returns false if matrix is singular
 */
static bool
matrix_invert(double **matrix, int n, double **result)
{
	double **augmented;
	int i, j, k;
	double pivot, factor;
	
	/* Create augmented matrix [A | I] */
	augmented = (double **) palloc(sizeof(double *) * n);
	for (i = 0; i < n; i++)
	{
		augmented[i] = (double *) palloc(sizeof(double) * 2 * n);
		for (j = 0; j < n; j++)
		{
			augmented[i][j] = matrix[i][j];
			augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
		}
	}
	
	/* Gauss-Jordan elimination */
	for (i = 0; i < n; i++)
	{
		/* Find pivot */
		pivot = augmented[i][i];
		if (fabs(pivot) < 1e-10)
		{
			/* Try to swap with a row below */
			bool found = false;
			for (k = i + 1; k < n; k++)
			{
				if (fabs(augmented[k][i]) > 1e-10)
				{
					/* Swap rows */
					double *temp = augmented[i];
					augmented[i] = augmented[k];
					augmented[k] = temp;
					pivot = augmented[i][i];
					found = true;
					break;
				}
			}
			if (!found)
			{
				/* Matrix is singular */
				for (j = 0; j < n; j++)
					pfree(augmented[j]);
				pfree(augmented);
				return false;
			}
		}
		
		/* Normalize pivot row */
		for (j = 0; j < 2 * n; j++)
			augmented[i][j] /= pivot;
		
		/* Eliminate column */
		for (k = 0; k < n; k++)
		{
			if (k != i)
			{
				factor = augmented[k][i];
				for (j = 0; j < 2 * n; j++)
					augmented[k][j] -= factor * augmented[i][j];
			}
		}
	}
	
	/* Extract result matrix from augmented matrix */
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			result[i][j] = augmented[i][j + n];
	
	/* Cleanup */
	for (i = 0; i < n; i++)
		pfree(augmented[i]);
	pfree(augmented);
	
	return true;
}

/*
 * train_linear_regression
 *
 * Trains a linear regression model using OLS
 * Returns coefficients (intercept + feature weights)
 */
PG_FUNCTION_INFO_V1(train_linear_regression);

Datum
train_linear_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			dim = 0;
	float	  **X = NULL;		/* Feature matrix */
	double	   *y = NULL;		/* Target values */
	double	  **XtX = NULL;		/* X'X matrix */
	double	  **XtX_inv = NULL;	/* (X'X)^(-1) */
	double	   *Xty = NULL;		/* X'y vector */
	double	   *beta = NULL;	/* Coefficients */
	int			i, j, k;
	Datum	   *result_datums;
	ArrayType  *result_array;
	MemoryContext oldcontext;
	
	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);
	
	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);
	
	/* Save caller's memory context */
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
				(errmsg("Need at least 10 samples for linear regression, have %d", nvec)));
	
	/* Switch to caller's context for allocations */
	MemoryContextSwitchTo(oldcontext);
	
	/* Allocate arrays in caller's context */
	X = (float **) palloc(sizeof(float *) * nvec);
	y = (double *) palloc(sizeof(double) * nvec);
	
	/* We're now in oldcontext, ready to process SPI results */
	
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
		
		/* Get feature vector */
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
		
		y[i] = DatumGetFloat8(targ_datum);
	}
	
	/* Switch to caller's context for computation */
	MemoryContextSwitchTo(oldcontext);
	
	SPI_finish();
	
	/* Allocate matrices for normal equations: β = (X'X)^(-1)X'y */
	dim = dim + 1; /* Add 1 for intercept */
	XtX = (double **) palloc(sizeof(double *) * dim);
	XtX_inv = (double **) palloc(sizeof(double *) * dim);
	for (i = 0; i < dim; i++)
	{
		XtX[i] = (double *) palloc0(sizeof(double) * dim);
		XtX_inv[i] = (double *) palloc(sizeof(double) * dim);
	}
	Xty = (double *) palloc0(sizeof(double) * dim);
	beta = (double *) palloc(sizeof(double) * dim);
	
	/* Compute X'X and X'y */
	for (i = 0; i < nvec; i++)
	{
		/* Add intercept term (1.0) - use VLA or palloc */
		double *xi = (double *) palloc(sizeof(double) * dim);
		xi[0] = 1.0;
		for (k = 1; k < dim; k++)
			xi[k] = X[i][k-1];
		
		/* X'X accumulation */
		for (j = 0; j < dim; j++)
		{
			for (k = 0; k < dim; k++)
				XtX[j][k] += xi[j] * xi[k];
			
			/* X'y accumulation */
			Xty[j] += xi[j] * y[i];
		}
		
		pfree(xi);
	}
	
	/* Invert X'X */
	if (!matrix_invert(XtX, dim, XtX_inv))
		ereport(ERROR,
				(errmsg("Matrix is singular, cannot compute linear regression"),
				 errhint("Try removing correlated features or adding regularization")));
	
	/* Compute β = (X'X)^(-1)X'y */
	for (i = 0; i < dim; i++)
	{
		beta[i] = 0.0;
		for (j = 0; j < dim; j++)
			beta[i] += XtX_inv[i][j] * Xty[j];
	}
	
	/* Build result array: [intercept, coef1, coef2, ...] */
	result_datums = (Datum *) palloc(sizeof(Datum) * dim);
	for (i = 0; i < dim; i++)
		result_datums[i] = Float8GetDatum(beta[i]);
	
	result_array = construct_array(result_datums, dim,
								   FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
	
	/* Cleanup */
	for (i = 0; i < nvec; i++)
		pfree(X[i]);
	pfree(X);
	pfree(y);
	for (i = 0; i < dim; i++)
	{
		pfree(XtX[i]);
		pfree(XtX_inv[i]);
	}
	pfree(XtX);
	pfree(XtX_inv);
	pfree(Xty);
	pfree(beta);
	pfree(result_datums);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(targ_str);
	
	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * predict_linear_regression
 *
 * Makes predictions using trained linear regression coefficients
 */
PG_FUNCTION_INFO_V1(predict_linear_regression);

Datum
predict_linear_regression(PG_FUNCTION_ARGS)
{
	ArrayType  *coef_array;
	Vector	   *features;
	int			ncoef;
	float8	   *coef;
	float	   *x;
	int			dim;
	double		prediction;
	int			i;
	
	coef_array = PG_GETARG_ARRAYTYPE_P(0);
	features = PG_GETARG_VECTOR_P(1);
	
	/* Extract coefficients */
	if (ARR_NDIM(coef_array) != 1)
		ereport(ERROR,
				(errmsg("Coefficients must be 1-dimensional array")));
	
	ncoef = ARR_DIMS(coef_array)[0];
	coef = (float8 *) ARR_DATA_PTR(coef_array);
	
	x = features->data;
	dim = features->dim;
	
	if (ncoef != dim + 1)
		ereport(ERROR,
				(errmsg("Coefficient dimension mismatch: expected %d, got %d", dim + 1, ncoef)));
	
	/* Compute prediction: y = β0 + β1*x1 + β2*x2 + ... */
	prediction = coef[0]; /* intercept */
	for (i = 0; i < dim; i++)
		prediction += coef[i + 1] * x[i];
	
	PG_RETURN_FLOAT8(prediction);
}

/*
 * evaluate_linear_regression
 *
 * Evaluates model performance (R², MSE, MAE)
 * Returns: [r_squared, mse, mae, rmse]
 */
PG_FUNCTION_INFO_V1(evaluate_linear_regression);

Datum
evaluate_linear_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	ArrayType  *coef_array;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			ncoef;
	float8	   *coef;
	double		mse = 0.0;
	double		mae = 0.0;
	double		ss_tot = 0.0;
	double		ss_res = 0.0;
	double		y_mean = 0.0;
	double		r_squared;
	double		rmse;
	int			i;
	Datum	   *result_datums;
	ArrayType  *result_array;
	MemoryContext oldcontext;
	
	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);
	coef_array = PG_GETARG_ARRAYTYPE_P(3);
	
	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);
	
	/* Extract coefficients */
	if (ARR_NDIM(coef_array) != 1)
		ereport(ERROR,
				(errmsg("Coefficients must be 1-dimensional array")));

	ncoef = ARR_DIMS(coef_array)[0];
	(void) ncoef;  /* Suppress unused variable warning */
	coef = (float8 *) ARR_DATA_PTR(coef_array);

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
	
	/* First pass: compute mean of y */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		targ_datum;
		bool		targ_null;
		
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (!targ_null)
			y_mean += DatumGetFloat8(targ_datum);
	}
	y_mean /= nvec;
	
	/* Second pass: compute predictions and metrics */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec;
		double		y_true;
		double		y_pred;
		double		error;
		int			j;
		
		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		
		if (feat_null || targ_null)
			continue;
		
		vec = DatumGetVector(feat_datum);
		y_true = DatumGetFloat8(targ_datum);
		
		/* Compute prediction */
		y_pred = coef[0]; /* intercept */
		for (j = 0; j < vec->dim; j++)
			y_pred += coef[j + 1] * vec->data[j];
		
		/* Compute errors */
		error = y_true - y_pred;
		mse += error * error;
		mae += fabs(error);
		ss_res += error * error;
		ss_tot += (y_true - y_mean) * (y_true - y_mean);
	}
	
	mse /= nvec;
	mae /= nvec;
	rmse = sqrt(mse);
	r_squared = 1.0 - (ss_res / ss_tot);
	
	SPI_finish();
	
	/* Build result array: [r_squared, mse, mae, rmse] */
	MemoryContextSwitchTo(oldcontext);
	
	result_datums = (Datum *) palloc(sizeof(Datum) * 4);
	result_datums[0] = Float8GetDatum(r_squared);
	result_datums[1] = Float8GetDatum(mse);
	result_datums[2] = Float8GetDatum(mae);
	result_datums[3] = Float8GetDatum(rmse);
	
	result_array = construct_array(result_datums, 4,
								   FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
	
	pfree(result_datums);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(targ_str);
	
	PG_RETURN_ARRAYTYPE_P(result_array);
}

