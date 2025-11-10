/*-------------------------------------------------------------------------
 *
 * ml_naive_bayes.c
 *    Naive Bayes classifier implementation
 *
 * Implements Gaussian Naive Bayes for continuous features.
 * Assumes features follow Gaussian distribution within each class.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_naive_bayes.c
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Gaussian Naive Bayes model parameters
 */
typedef struct
{
	double	   *class_priors;		/* P(class) */
	double	  **means;				/* Mean for each feature per class */
	double	  **variances;			/* Variance for each feature per class */
	int			n_classes;
	int			n_features;
} GaussianNBModel;

/*
 * Gaussian probability density function
 */
static double
gaussian_pdf(double x, double mean, double variance)
{
	double exponent;
	
	if (variance < 1e-9)
		variance = 1e-9;  /* Avoid division by zero */
	
	exponent = -0.5 * pow((x - mean), 2) / variance;
	return (1.0 / sqrt(2.0 * M_PI * variance)) * exp(exponent);
}

/*
 * train_naive_bayes_classifier
 *
 * Trains a Gaussian Naive Bayes classifier
 * Returns model parameters as JSONB
 */
PG_FUNCTION_INFO_V1(train_naive_bayes_classifier);

Datum
train_naive_bayes_classifier(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *label_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			dim = 0;
	float	  **X = NULL;
	double	   *y = NULL;
	GaussianNBModel model;
	int			i, j, class;
	int			n_params;
	int			idx;
	int		   *class_counts;
	double	 ***class_samples;		/* [class][sample_idx][feature] */
	int		   *class_sizes;
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
		
		y[i] = DatumGetFloat8(label_datum);
	}
	
	SPI_finish();
	
	/* Initialize model for binary classification */
	model.n_classes = 2;
	model.n_features = dim;
	model.class_priors = (double *) palloc0(sizeof(double) * 2);
	model.means = (double **) palloc(sizeof(double *) * 2);
	model.variances = (double **) palloc(sizeof(double *) * 2);
	
	class_counts = (int *) palloc0(sizeof(int) * 2);
	class_sizes = (int *) palloc0(sizeof(int) * 2);
	class_samples = (double ***) palloc(sizeof(double **) * 2);
	
	/* Count samples per class */
	for (i = 0; i < nvec; i++)
	{
		class = (int) y[i];
		if (class >= 0 && class < 2)
			class_counts[class]++;
	}
	
	/* Allocate class sample arrays */
	for (class = 0; class < 2; class++)
	{
		class_samples[class] = (double **) palloc(sizeof(double *) * class_counts[class]);
		model.means[class] = (double *) palloc0(sizeof(double) * dim);
		model.variances[class] = (double *) palloc0(sizeof(double) * dim);
	}
	
	/* Group samples by class */
	for (i = 0; i < nvec; i++)
	{
		class = (int) y[i];
		if (class >= 0 && class < 2)
		{
			class_samples[class][class_sizes[class]] = (double *) palloc(sizeof(double) * dim);
			for (j = 0; j < dim; j++)
				class_samples[class][class_sizes[class]][j] = X[i][j];
			class_sizes[class]++;
		}
	}
	
	/* Compute class priors, means, and variances */
	for (class = 0; class < 2; class++)
	{
		model.class_priors[class] = (double) class_sizes[class] / nvec;
		
		/* Compute means */
		for (j = 0; j < dim; j++)
		{
			double sum = 0.0;
			for (i = 0; i < class_sizes[class]; i++)
				sum += class_samples[class][i][j];
			model.means[class][j] = sum / class_sizes[class];
		}
		
		/* Compute variances */
		for (j = 0; j < dim; j++)
		{
			double sum_sq = 0.0;
			for (i = 0; i < class_sizes[class]; i++)
			{
				double diff = class_samples[class][i][j] - model.means[class][j];
				sum_sq += diff * diff;
			}
			model.variances[class][j] = sum_sq / class_sizes[class];
		}
	}
	
	/* Serialize model parameters to array */
	/* Format: [n_classes, n_features, prior0, prior1, mean0_0, mean0_1, ..., var0_0, var0_1, ...] */
	n_params = 2 + 2 + (2 * dim) + (2 * dim);  /* metadata + priors + means + variances */
	result_datums = (Datum *) palloc(sizeof(Datum) * n_params);
	
	result_datums[0] = Float8GetDatum(model.n_classes);
	result_datums[1] = Float8GetDatum(model.n_features);
	result_datums[2] = Float8GetDatum(model.class_priors[0]);
	result_datums[3] = Float8GetDatum(model.class_priors[1]);
	
	idx = 4;
	for (class = 0; class < 2; class++)
		for (j = 0; j < dim; j++)
			result_datums[idx++] = Float8GetDatum(model.means[class][j]);
	
	for (class = 0; class < 2; class++)
		for (j = 0; j < dim; j++)
			result_datums[idx++] = Float8GetDatum(model.variances[class][j]);
	
	result_array = construct_array(result_datums, n_params,
								   FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
	
	/* Cleanup */
	for (i = 0; i < nvec; i++)
		pfree(X[i]);
	pfree(X);
	pfree(y);
	for (class = 0; class < 2; class++)
	{
		for (i = 0; i < class_sizes[class]; i++)
			pfree(class_samples[class][i]);
		pfree(class_samples[class]);
		pfree(model.means[class]);
		pfree(model.variances[class]);
	}
	pfree(class_samples);
	pfree(class_counts);
	pfree(class_sizes);
	pfree(model.class_priors);
	pfree(model.means);
	pfree(model.variances);
	pfree(result_datums);
	
	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * predict_naive_bayes
 *
 * Predicts class using trained Naive Bayes model
 */
PG_FUNCTION_INFO_V1(predict_naive_bayes);

Datum
predict_naive_bayes(PG_FUNCTION_ARGS)
{
	ArrayType  *model_params;
	Vector	   *features;
	float8	   *params;
	int			n_params;
	int			n_classes;
	int			n_features;
	double	   *class_priors;
	double	  **means;
	double	  **variances;
	double		log_probs[2] = {0.0, 0.0};
	int			predicted_class;
	int			i, j, idx;
	
	model_params = PG_GETARG_ARRAYTYPE_P(0);
	features = PG_GETARG_VECTOR_P(1);
	
	if (ARR_NDIM(model_params) != 1)
		ereport(ERROR,
				(errmsg("Model parameters must be 1-dimensional array")));
	
	n_params = ARR_DIMS(model_params)[0];
	(void) n_params;  /* Suppress unused variable warning */
	params = (float8 *) ARR_DATA_PTR(model_params);
	
	n_classes = (int) params[0];
	n_features = (int) params[1];
	
	if (features->dim != n_features)
		ereport(ERROR,
				(errmsg("Feature dimension mismatch: expected %d, got %d", n_features, features->dim)));
	
	/* Extract model parameters */
	class_priors = (double *) palloc(sizeof(double) * n_classes);
	means = (double **) palloc(sizeof(double *) * n_classes);
	variances = (double **) palloc(sizeof(double *) * n_classes);
	
	class_priors[0] = params[2];
	class_priors[1] = params[3];
	
	idx = 4;
	for (i = 0; i < n_classes; i++)
	{
		means[i] = (double *) palloc(sizeof(double) * n_features);
		for (j = 0; j < n_features; j++)
			means[i][j] = params[idx++];
	}
	
	for (i = 0; i < n_classes; i++)
	{
		variances[i] = (double *) palloc(sizeof(double) * n_features);
		for (j = 0; j < n_features; j++)
			variances[i][j] = params[idx++];
	}
	
	/* Compute log probabilities for each class */
	for (i = 0; i < n_classes; i++)
	{
		log_probs[i] = log(class_priors[i]);
		
		for (j = 0; j < n_features; j++)
		{
			double pdf = gaussian_pdf(features->data[j], means[i][j], variances[i][j]);
			log_probs[i] += log(pdf + 1e-10);  /* Add small constant to avoid log(0) */
		}
	}
	
	/* Return class with highest log probability */
	predicted_class = (log_probs[1] > log_probs[0]) ? 1 : 0;
	
	/* Cleanup */
	pfree(class_priors);
	for (i = 0; i < n_classes; i++)
	{
		pfree(means[i]);
		pfree(variances[i]);
	}
	pfree(means);
	pfree(variances);
	
	PG_RETURN_INT32(predicted_class);
}

