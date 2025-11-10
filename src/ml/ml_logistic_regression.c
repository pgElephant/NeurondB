/*-------------------------------------------------------------------------
 *
 * ml_logistic_regression.c
 *    Logistic Regression implementation for binary classification
 *
 * Implements logistic regression using gradient descent with L2 regularization.
 * Supports binary classification with sigmoid activation.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_logistic_regression.c
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
 * Sigmoid function: 1 / (1 + exp(-z))
 */
static inline double
sigmoid(double z)
{
	if (z > 500)
		return 1.0;
	if (z < -500)
		return 0.0;
	return 1.0 / (1.0 + exp(-z));
}

/*
 * Compute log loss (binary cross-entropy)
 */
static double
compute_log_loss(double *y_true, double *y_pred, int n)
{
	double loss = 0.0;
	int i;
	
	for (i = 0; i < n; i++)
	{
		double pred = y_pred[i];
		/* Clip predictions to avoid log(0) */
		pred = fmax(1e-15, fmin(1.0 - 1e-15, pred));
		
		if (y_true[i] > 0.5)
			loss -= log(pred);
		else
			loss -= log(1.0 - pred);
	}
	
	return loss / n;
}

/*
 * train_logistic_regression
 *
 * Trains a logistic regression model using gradient descent
 * Returns coefficients (intercept + feature weights)
 */
PG_FUNCTION_INFO_V1(train_logistic_regression);

Datum
train_logistic_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	int			max_iters = PG_GETARG_INT32(3);
	double		learning_rate = PG_GETARG_FLOAT8(4);
	double		lambda = PG_GETARG_FLOAT8(5);  /* L2 regularization */
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
	int			iter, i, j;
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
				(errmsg("Need at least 10 samples for logistic regression, have %d", nvec)));
	
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
		
		/* Try to get as integer first, then fallback to float */
		{
			Oid targ_type = SPI_gettypeid(tupdesc, 2);
			if (targ_type == INT2OID || targ_type == INT4OID || targ_type == INT8OID)
			{
				y[i] = (double) DatumGetInt32(targ_datum);
			}
			else
			{
				y[i] = DatumGetFloat8(targ_datum);
			}
		}
		
		/* Validate binary target */
		if (y[i] != 0.0 && y[i] != 1.0)
			ereport(ERROR,
					(errmsg("Logistic regression requires binary target values (0 or 1), got %f", y[i])));

	}
	
	SPI_finish();
	
	/* Initialize weights and bias */
	weights = (double *) palloc0(sizeof(double) * dim);
	
	/* Gradient descent */
	for (iter = 0; iter < max_iters; iter++)
	{
		double *predictions = (double *) palloc(sizeof(double) * nvec);
		double *grad_w = (double *) palloc0(sizeof(double) * dim);
		double grad_b = 0.0;
		
		/* Forward pass: compute predictions */
		for (i = 0; i < nvec; i++)
		{
			double z = bias;
			for (j = 0; j < dim; j++)
				z += weights[j] * X[i][j];
			predictions[i] = sigmoid(z);
		}
		
		/* Backward pass: compute gradients */
		for (i = 0; i < nvec; i++)
		{
			double error = predictions[i] - y[i];
			grad_b += error;
			for (j = 0; j < dim; j++)
				grad_w[j] += error * X[i][j];
		}
		
		/* Average gradients and add L2 regularization */
		grad_b /= nvec;
		for (j = 0; j < dim; j++)
		{
			grad_w[j] = grad_w[j] / nvec + lambda * weights[j];
		}
		
		/* Update weights and bias */
		bias -= learning_rate * grad_b;
		for (j = 0; j < dim; j++)
			weights[j] -= learning_rate * grad_w[j];
		
		pfree(predictions);
		pfree(grad_w);
		
		/* Log progress every 100 iterations */
		if (iter % 100 == 0)
		{
			double	   *preds;
			double		loss;

			preds = (double *) palloc(sizeof(double) * nvec);
			for (i = 0; i < nvec; i++)
			{
				double z = bias;
				for (j = 0; j < dim; j++)
					z += weights[j] * X[i][j];
				preds[i] = sigmoid(z);
			}
			loss = compute_log_loss(y, preds, nvec);
			elog(NOTICE, "Iteration %d: loss = %f", iter, loss);
			pfree(preds);
		}
	}
	
	/* Build result array: [bias, weight1, weight2, ...] */
	result_datums = (Datum *) palloc(sizeof(Datum) * (dim + 1));
	result_datums[0] = Float8GetDatum(bias);
	for (i = 0; i < dim; i++)
		result_datums[i + 1] = Float8GetDatum(weights[i]);
	
	result_array = construct_array(result_datums, dim + 1,
								   FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
	
	/* Cleanup */
	for (i = 0; i < nvec; i++)
		pfree(X[i]);
	pfree(X);
	pfree(y);
	pfree(weights);
	pfree(result_datums);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(targ_str);
	
	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * predict_logistic_regression
 *
 * Predicts probability for logistic regression
 */
PG_FUNCTION_INFO_V1(predict_logistic_regression);

Datum
predict_logistic_regression(PG_FUNCTION_ARGS)
{
	ArrayType  *coef_array;
	Vector	   *features;
	int			ncoef;
	float8	   *coef;
	float	   *x;
	int			dim;
	double		z;
	double		probability;
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
	
	/* Compute z = bias + w1*x1 + w2*x2 + ... */
	z = coef[0]; /* bias */
	for (i = 0; i < dim; i++)
		z += coef[i + 1] * x[i];
	
	/* Apply sigmoid */
	probability = sigmoid(z);
	
	PG_RETURN_FLOAT8(probability);
}

/*
 * evaluate_logistic_regression
 *
 * Evaluates model performance (accuracy, precision, recall, F1, AUC)
 * Returns: [accuracy, precision, recall, f1_score, log_loss]
 */
PG_FUNCTION_INFO_V1(evaluate_logistic_regression);

Datum
evaluate_logistic_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	ArrayType  *coef_array;
	double		threshold = PG_GETARG_FLOAT8(4);
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			ncoef;
	float8	   *coef;
	int			tp = 0, tn = 0, fp = 0, fn = 0;
	double		log_loss = 0.0;
	double		accuracy, precision, recall, f1_score;
	int			i, j;
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
	
	/* Compute predictions and metrics */
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
		double		z;
		double		probability;
		int			y_pred;
		
		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		
		if (feat_null || targ_null)
			continue;
		
		vec = DatumGetVector(feat_datum);
		y_true = DatumGetFloat8(targ_datum);
		
		/* Compute probability */
		z = coef[0]; /* bias */
		for (j = 0; j < vec->dim; j++)
			z += coef[j + 1] * vec->data[j];
		probability = sigmoid(z);
		
		/* Apply threshold */
		y_pred = (probability >= threshold) ? 1 : 0;
		
		/* Update confusion matrix */
		if (y_true > 0.5)
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
		
		/* Update log loss */
		probability = fmax(1e-15, fmin(1.0 - 1e-15, probability));
		if (y_true > 0.5)
			log_loss -= log(probability);
		else
			log_loss -= log(1.0 - probability);
	}
	
	log_loss /= nvec;
	
	SPI_finish();
	
	/* Compute metrics */
	accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
	precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
	recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
	f1_score = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
	
	/* Build result array: [accuracy, precision, recall, f1_score, log_loss] */
	MemoryContextSwitchTo(oldcontext);
	
	result_datums = (Datum *) palloc(sizeof(Datum) * 5);
	result_datums[0] = Float8GetDatum(accuracy);
	result_datums[1] = Float8GetDatum(precision);
	result_datums[2] = Float8GetDatum(recall);
	result_datums[3] = Float8GetDatum(f1_score);
	result_datums[4] = Float8GetDatum(log_loss);
	
	result_array = construct_array(result_datums, 5,
								   FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
	
	pfree(result_datums);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(targ_str);
	
	PG_RETURN_ARRAYTYPE_P(result_array);
}

