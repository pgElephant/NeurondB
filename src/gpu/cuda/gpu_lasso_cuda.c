/*-------------------------------------------------------------------------
 *
 * gpu_lasso_cuda.c
 *    CUDA backend bridge for Lasso Regression training and prediction.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_lasso_cuda.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#ifdef NDB_GPU_CUDA

#include <float.h>
#include <math.h>
#include <string.h>

#include "neurondb_cuda_runtime.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"

#include "ml_lasso_regression_internal.h"
#include "neurondb_cuda_lasso.h"

/*
 * Soft thresholding operator for Lasso
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

int
ndb_cuda_lasso_pack_model(const LassoModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	size_t payload_bytes;
	bytea *blob;
	char *base;
	NdbCudaLassoModelHeader *hdr;
	float *coef_dest;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid Lasso model for CUDA pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaLassoModelHeader)
		+ sizeof(float) * (size_t)model->n_features;

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaLassoModelHeader *)base;
	hdr->feature_dim = model->n_features;
	hdr->n_samples = model->n_samples;
	hdr->intercept = (float)model->intercept;
	hdr->lambda = model->lambda;
	hdr->max_iters = model->max_iters;
	hdr->r_squared = model->r_squared;
	hdr->mse = model->mse;
	hdr->mae = model->mae;

	coef_dest = (float *)(base + sizeof(NdbCudaLassoModelHeader));
	if (model->coefficients != NULL)
	{
		int i;

		for (i = 0; i < model->n_features; i++)
			coef_dest[i] = (float)model->coefficients[i];
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"lasso\","
			"\"storage\":\"gpu\","
			"\"n_features\":%d,"
			"\"n_samples\":%d,"
			"\"lambda\":%.6f,"
			"\"max_iters\":%d,"
			"\"r_squared\":%.6f,"
			"\"mse\":%.6f,"
			"\"mae\":%.6f}",
			model->n_features,
			model->n_samples,
			model->lambda,
			model->max_iters,
			model->r_squared,
			model->mse,
			model->mae);

		metrics_json = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(buf.data)));
		pfree(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

int
ndb_cuda_lasso_train(const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	double lambda = 0.01;  /* Default regularization */
	int max_iters = 1000;  /* Default iterations */
	double *weights = NULL;
	double *weights_old = NULL;
	double *residuals = NULL;
	double y_mean = 0.0;
	bytea *payload = NULL;
	Jsonb *metrics_json = NULL;
	int iter, i, j;
	bool converged = false;
	int rc = -1;

	if (errstr)
		*errstr = NULL;

	if (features == NULL || targets == NULL || n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for CUDA Lasso train");
		return -1;
	}

	/* Extract hyperparameters from JSON */
	if (hyperparams != NULL)
	{
		char *hyperparams_text = DatumGetCString(
			DirectFunctionCall1(jsonb_out, JsonbPGetDatum(hyperparams)));
		/* Simple extraction - can be enhanced with proper JSON parsing */
		/* For now, use defaults */
		pfree(hyperparams_text);
	}

	/* Compute mean of targets */
	for (i = 0; i < n_samples; i++)
		y_mean += targets[i];
	y_mean /= n_samples;

	/* Initialize weights and residuals */
	weights = (double *)palloc0(sizeof(double) * feature_dim);
	weights_old = (double *)palloc(sizeof(double) * feature_dim);
	residuals = (double *)palloc(sizeof(double) * n_samples);

	/* Initialize residuals */
	for (i = 0; i < n_samples; i++)
		residuals[i] = targets[i] - y_mean;

	/* Coordinate descent */
	for (iter = 0; iter < max_iters && !converged; iter++)
	{
		double diff;

		memcpy(weights_old, weights, sizeof(double) * feature_dim);

		/* Update each coordinate */
		for (j = 0; j < feature_dim; j++)
		{
			double rho = 0.0;
			double z = 0.0;
			double old_weight;
			const float *feature_col_j;

			/* Compute rho = X_j^T * residuals */
			for (i = 0; i < n_samples; i++)
			{
				feature_col_j = features + (i * feature_dim + j);
				rho += (*feature_col_j) * residuals[i];
			}

			/* Compute z = X_j^T * X_j */
			for (i = 0; i < n_samples; i++)
			{
				feature_col_j = features + (i * feature_dim + j);
				z += (*feature_col_j) * (*feature_col_j);
			}

			if (z < 1e-10)
				continue;

			/* Soft thresholding */
			old_weight = weights[j];
			weights[j] = soft_threshold(rho / z, lambda / z);

			/* Update residuals */
			if (weights[j] != old_weight)
			{
				double weight_diff = weights[j] - old_weight;
				for (i = 0; i < n_samples; i++)
				{
					feature_col_j = features + (i * feature_dim + j);
					residuals[i] -= (*feature_col_j) * weight_diff;
				}
			}
		}

		/* Check convergence */
		diff = 0.0;
		for (j = 0; j < feature_dim; j++)
		{
			double d = weights[j] - weights_old[j];
			diff += d * d;
		}

		if (sqrt(diff) < 1e-6)
			converged = true;
	}

	/* Build model */
	{
		LassoModel model;
		double ss_tot = 0.0;
		double ss_res = 0.0;
		double mse = 0.0;
		double mae = 0.0;

		model.n_features = feature_dim;
		model.n_samples = n_samples;
		model.intercept = y_mean;
		model.lambda = lambda;
		model.max_iters = max_iters;
		model.coefficients = (double *)palloc(sizeof(double) * feature_dim);
		for (i = 0; i < feature_dim; i++)
			model.coefficients[i] = weights[i];

		/* Compute metrics */
		for (i = 0; i < n_samples; i++)
		{
			const float *row = features + (i * feature_dim);
			double y_pred = model.intercept;
			double error;
			int j;

			for (j = 0; j < feature_dim; j++)
				y_pred += model.coefficients[j] * row[j];

			error = targets[i] - y_pred;
			mse += error * error;
			mae += fabs(error);
			ss_res += error * error;
			ss_tot += (targets[i] - y_mean) * (targets[i] - y_mean);
		}

		mse /= n_samples;
		mae /= n_samples;
		model.r_squared = (ss_tot > 0.0) ? (1.0 - (ss_res / ss_tot)) : 0.0;
		model.mse = mse;
		model.mae = mae;

		/* Pack model */
		rc = ndb_cuda_lasso_pack_model(&model, &payload, &metrics_json, errstr);

		pfree(model.coefficients);
	}

	/* Cleanup */
	pfree(weights);
	pfree(weights_old);
	pfree(residuals);

	if (rc == 0 && payload != NULL)
	{
		*model_data = payload;
		if (metrics != NULL)
			*metrics = metrics_json;
		return 0;
	}

	if (payload != NULL)
		pfree(payload);
	if (metrics_json != NULL)
		pfree(metrics_json);

	return -1;
}

int
ndb_cuda_lasso_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	double *prediction_out,
	char **errstr)
{
	const NdbCudaLassoModelHeader *hdr;
	const float *coefficients;
	const bytea *detoasted;
	double prediction;
	int i;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || prediction_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for CUDA Lasso predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted = (const bytea *)PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t expected_size = sizeof(NdbCudaLassoModelHeader) + sizeof(float) * (size_t)feature_dim;
		size_t actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < expected_size)
		{
			if (errstr)
				*errstr = psprintf("model data too small: expected %zu bytes, got %zu",
					expected_size, actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaLassoModelHeader *)VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model has %d, input has %d",
				hdr->feature_dim, feature_dim);
		return -1;
	}

	coefficients = (const float *)((const char *)hdr + sizeof(NdbCudaLassoModelHeader));

	prediction = hdr->intercept;
	for (i = 0; i < feature_dim; i++)
		prediction += coefficients[i] * input[i];

	*prediction_out = prediction;
	return 0;
}

#endif /* NDB_GPU_CUDA */

