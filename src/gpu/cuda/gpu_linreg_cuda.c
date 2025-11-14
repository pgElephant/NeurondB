/*-------------------------------------------------------------------------
 *
 * gpu_linreg_cuda.c
 *    CUDA backend bridge for Linear Regression training and prediction.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_linreg_cuda.c
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

#include "ml_linear_regression_internal.h"
#include "neurondb_cuda_linreg.h"

/* Forward declaration for kernel */
extern void ndb_cuda_linreg_compute_xtx_kernel(const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	int dim_with_intercept,
	double *XtX,
	double *Xty);

int
ndb_cuda_linreg_pack_model(const LinRegModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	size_t payload_bytes;
	bytea *blob;
	char *base;
	NdbCudaLinRegModelHeader *hdr;
	float *coef_dest;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid LinReg model for CUDA pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaLinRegModelHeader)
		+ sizeof(float) * (size_t)model->n_features;

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaLinRegModelHeader *)base;
	hdr->feature_dim = model->n_features;
	hdr->n_samples = model->n_samples;
	hdr->intercept = (float)model->intercept;
	hdr->r_squared = model->r_squared;
	hdr->mse = model->mse;
	hdr->mae = model->mae;

	coef_dest = (float *)(base + sizeof(NdbCudaLinRegModelHeader));
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
			"{\"algorithm\":\"linear_regression\","
			"\"storage\":\"gpu\","
			"\"n_features\":%d,"
			"\"n_samples\":%d,"
			"\"r_squared\":%.6f,"
			"\"mse\":%.6f,"
			"\"mae\":%.6f}",
			model->n_features,
			model->n_samples,
			model->r_squared,
			model->mse,
			model->mae);

		metrics_json = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(buf.data)));
		pfree(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

int
ndb_cuda_linreg_train(const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	float *d_features = NULL;
	double *d_targets = NULL;
	double *d_XtX = NULL;
	double *d_Xty = NULL;
	double *d_XtX_inv __attribute__((unused)) = NULL;
	double *d_beta __attribute__((unused)) = NULL;
	double *h_XtX = NULL;
	double *h_Xty = NULL;
	double *h_XtX_inv = NULL;
	double *h_beta = NULL;
	bytea *payload = NULL;
	Jsonb *metrics_json = NULL;
	cudaError_t status __attribute__((unused)) = cudaSuccess;
	size_t feature_bytes __attribute__((unused));
	size_t target_bytes __attribute__((unused));
	size_t XtX_bytes;
	size_t Xty_bytes;
	size_t beta_bytes;
	int dim_with_intercept;
	int i, j, k;
	int rc = -1;

	if (errstr)
		*errstr = NULL;

	if (features == NULL || targets == NULL || n_samples <= 0
		|| feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for CUDA "
					  "LinReg train");
		return -1;
	}

	dim_with_intercept = feature_dim + 1;

	/* Allocate host memory for matrices */
	XtX_bytes = sizeof(double) * (size_t)dim_with_intercept
		* (size_t)dim_with_intercept;
	Xty_bytes = sizeof(double) * (size_t)dim_with_intercept;
	beta_bytes = sizeof(double) * (size_t)dim_with_intercept;

	h_XtX = (double *)palloc0(XtX_bytes);
	h_Xty = (double *)palloc0(Xty_bytes);
	h_XtX_inv = (double *)palloc(XtX_bytes);
	h_beta = (double *)palloc(beta_bytes);

	/* Compute X'X and X'y on GPU */
	{
#ifdef __CUDACC__
		int threads_per_block = 256;
		int blocks = (n_samples + threads_per_block - 1) / threads_per_block;
#endif
		cudaError_t err;

		/* Allocate device memory */
		feature_bytes = sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
		target_bytes = sizeof(double) * (size_t)n_samples;
		XtX_bytes = sizeof(double) * (size_t)dim_with_intercept * (size_t)dim_with_intercept;
		Xty_bytes = sizeof(double) * (size_t)dim_with_intercept;

		err = cudaMalloc((void **)&d_features, feature_bytes);
		if (err != cudaSuccess)
		{
			if (errstr)
				*errstr = pstrdup("CUDA LinReg train: failed to allocate device memory for features");
			goto cpu_fallback;
		}

		err = cudaMalloc((void **)&d_targets, target_bytes);
		if (err != cudaSuccess)
		{
			cudaFree(d_features);
			if (errstr)
				*errstr = pstrdup("CUDA LinReg train: failed to allocate device memory for targets");
			goto cpu_fallback;
		}

		err = cudaMalloc((void **)&d_XtX, XtX_bytes);
		if (err != cudaSuccess)
		{
			cudaFree(d_features);
			cudaFree(d_targets);
			if (errstr)
				*errstr = pstrdup("CUDA LinReg train: failed to allocate device memory for XtX");
			goto cpu_fallback;
		}

		err = cudaMalloc((void **)&d_Xty, Xty_bytes);
		if (err != cudaSuccess)
		{
			cudaFree(d_features);
			cudaFree(d_targets);
			cudaFree(d_XtX);
			if (errstr)
				*errstr = pstrdup("CUDA LinReg train: failed to allocate device memory for Xty");
			goto cpu_fallback;
		}

		/* Initialize XtX and Xty to zero */
		err = cudaMemset(d_XtX, 0, XtX_bytes);
		if (err != cudaSuccess)
			goto gpu_cleanup;
		err = cudaMemset(d_Xty, 0, Xty_bytes);
		if (err != cudaSuccess)
			goto gpu_cleanup;

		/* Copy data to device */
		err = cudaMemcpy(d_features, features, feature_bytes, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
			goto gpu_cleanup;
		err = cudaMemcpy(d_targets, targets, target_bytes, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
			goto gpu_cleanup;

		/* Launch kernel */
#ifdef __CUDACC__
		ndb_cuda_linreg_compute_xtx_kernel<<<blocks, threads_per_block>>>(
			d_features, d_targets, n_samples, feature_dim, dim_with_intercept,
			d_XtX, d_Xty);

		err = cudaGetLastError();
		if (err != cudaSuccess)
			goto gpu_cleanup;
#else
		/* Kernel launch not available when compiling with gcc, fall back to CPU */
		if (d_features)
			cudaFree(d_features);
		if (d_targets)
			cudaFree(d_targets);
		if (d_XtX)
			cudaFree(d_XtX);
		if (d_Xty)
			cudaFree(d_Xty);
		goto cpu_fallback;
#endif

#ifdef __CUDACC__
		/* Wait for kernel to complete */
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
			goto gpu_cleanup;

		/* Copy results back to host */
		err = cudaMemcpy(h_XtX, d_XtX, XtX_bytes, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
			goto gpu_cleanup;
		err = cudaMemcpy(h_Xty, d_Xty, Xty_bytes, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
			goto gpu_cleanup;

		cudaFree(d_features);
		cudaFree(d_targets);
		cudaFree(d_XtX);
		cudaFree(d_Xty);
#endif

gpu_cleanup:
		if (d_features)
			cudaFree(d_features);
		if (d_targets)
			cudaFree(d_targets);
		if (d_XtX)
			cudaFree(d_XtX);
		if (d_Xty)
			cudaFree(d_Xty);

cpu_fallback:
		/* Fallback to CPU computation */
		for (i = 0; i < n_samples; i++)
		{
			const float *row = features + (i * feature_dim);
			double *xi = (double *)palloc(sizeof(double) * dim_with_intercept);
			
			xi[0] = 1.0; /* intercept */
			for (k = 1; k < dim_with_intercept; k++)
				xi[k] = row[k-1];
			
			/* X'X accumulation */
			for (j = 0; j < dim_with_intercept; j++)
			{
				for (k = 0; k < dim_with_intercept; k++)
					h_XtX[j * dim_with_intercept + k] += xi[j] * xi[k];
				
				/* X'y accumulation */
				h_Xty[j] += xi[j] * targets[i];
			}
			
			pfree(xi);
		}
	}

	/* Invert X'X using Gauss-Jordan elimination */
	{
		double **augmented;
		int row, col, k_local;
		double pivot, factor;
		bool invert_success = true;

		/* Create augmented matrix [A | I] */
		augmented = (double **)palloc(
			sizeof(double *) * dim_with_intercept);
		for (row = 0; row < dim_with_intercept; row++)
		{
			augmented[row] = (double *)palloc(
				sizeof(double) * 2 * dim_with_intercept);
			for (col = 0; col < dim_with_intercept; col++)
			{
				augmented[row][col] =
					h_XtX[row * dim_with_intercept + col];
				augmented[row][col + dim_with_intercept] =
					(row == col) ? 1.0 : 0.0;
			}
		}

		/* Gauss-Jordan elimination */
		for (row = 0; row < dim_with_intercept; row++)
		{
			pivot = augmented[row][row];
			if (fabs(pivot) < 1e-10)
			{
				bool found = false;
				for (k_local = row + 1; k_local < dim_with_intercept; k_local++)
				{
					if (fabs(augmented[k_local][row]) > 1e-10)
					{
						double *temp = augmented[row];
						augmented[row] = augmented[k_local];
						augmented[k_local] = temp;
						pivot = augmented[row][row];
						found = true;
						break;
					}
				}
				if (!found)
				{
					invert_success = false;
					break;
				}
			}

			for (col = 0; col < 2 * dim_with_intercept; col++)
				augmented[row][col] /= pivot;

			for (k_local = 0; k_local < dim_with_intercept; k_local++)
			{
				if (k_local != row)
				{
					factor = augmented[k_local][row];
					for (col = 0;
						col < 2 * dim_with_intercept;
						col++)
						augmented[k_local][col] -= factor
							* augmented[row][col];
				}
			}
		}

		if (invert_success)
		{
			for (row = 0; row < dim_with_intercept; row++)
				for (col = 0; col < dim_with_intercept; col++)
					h_XtX_inv[row * dim_with_intercept
						+ col] = augmented[row][col
						+ dim_with_intercept];
		}

		for (row = 0; row < dim_with_intercept; row++)
			pfree(augmented[row]);
		pfree(augmented);

		if (!invert_success)
		{
			pfree(h_XtX);
			pfree(h_Xty);
			pfree(h_XtX_inv);
			pfree(h_beta);
			if (errstr)
				*errstr = pstrdup("Matrix is singular, cannot "
						  "compute linear regression");
			return -1;
		}
	}

	/* Compute β = (X'X)^(-1)X'y */
	for (i = 0; i < dim_with_intercept; i++)
	{
		h_beta[i] = 0.0;
		for (j = 0; j < dim_with_intercept; j++)
			h_beta[i] += h_XtX_inv[i * dim_with_intercept + j]
				* h_Xty[j];
	}

	/* Build model */
	{
		LinRegModel model;
		double y_mean = 0.0;
		double ss_tot = 0.0;
		double ss_res = 0.0;
		double mse = 0.0;
		double mae = 0.0;

		model.n_features = feature_dim;
		model.n_samples = n_samples;
		model.intercept = h_beta[0];
		model.coefficients =
			(double *)palloc(sizeof(double) * feature_dim);
		for (i = 0; i < feature_dim; i++)
			model.coefficients[i] = h_beta[i + 1];

		/* Compute metrics */
		for (i = 0; i < n_samples; i++)
			y_mean += targets[i];
		y_mean /= n_samples;

		for (i = 0; i < n_samples; i++)
		{
		const float *row = features + (i * feature_dim);
		double y_pred = model.intercept;
		double error;
		int j_local;

	for (j_local = 0; j_local < feature_dim; j_local++)
		y_pred += model.coefficients[j_local] * row[j_local];

	error = targets[i] - y_pred;
	mse += error * error;
	mae += fabs(error);
	ss_res += error * error;
	ss_tot += (targets[i] - y_mean) * (targets[i] - y_mean);
		}

		mse /= n_samples;
		mae /= n_samples;
		model.r_squared = 1.0 - (ss_res / ss_tot);
		model.mse = mse;
		model.mae = mae;

		/* Pack model */
		rc = ndb_cuda_linreg_pack_model(
			&model, &payload, &metrics_json, errstr);

		pfree(model.coefficients);
	}

	/* Cleanup */
	pfree(h_XtX);
	pfree(h_Xty);
	pfree(h_XtX_inv);
	pfree(h_beta);

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
ndb_cuda_linreg_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	double *prediction_out,
	char **errstr)
{
	const NdbCudaLinRegModelHeader *hdr;
	const float *coefficients;
	const bytea *detoasted;
	double prediction;
	int i;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || prediction_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup(
				"invalid parameters for CUDA LinReg predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted =
		(const bytea *)PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t expected_size = sizeof(NdbCudaLinRegModelHeader)
			+ sizeof(float) * (size_t)feature_dim;
		size_t actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < expected_size)
		{
			if (errstr)
				*errstr =
					psprintf("model data too small: "
						 "expected %zu bytes, got %zu",
						expected_size,
						actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaLinRegModelHeader *)VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model "
					   "has %d, input has %d",
				hdr->feature_dim,
				feature_dim);
		return -1;
	}

	coefficients = (const float *)((const char *)hdr
		+ sizeof(NdbCudaLinRegModelHeader));

	prediction = hdr->intercept;
	for (i = 0; i < feature_dim; i++)
		prediction += coefficients[i] * input[i];

	*prediction_out = prediction;
	return 0;
}

#endif /* NDB_GPU_CUDA */
