/*-------------------------------------------------------------------------
 *
 * gpu_lr_cuda.c
 *	  CUDA backend bridge for Logistic Regression training and prediction.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *	  src/gpu/cuda/gpu_lr_cuda.c
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

#include "ml_logistic_regression_internal.h"
#include "neurondb_cuda_lr.h"

int
ndb_cuda_lr_pack_model(const LRModel *model,
					  bytea **model_data,
					  Jsonb **metrics,
					  char **errstr)
{
	size_t			 payload_bytes;
	bytea			*blob;
	char			*base;
	NdbCudaLrModelHeader *hdr;
	float			*weights_dest;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid LR model for CUDA pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaLrModelHeader) +
					sizeof(float) * (size_t) model->n_features;

	blob = (bytea *) palloc(VARHDRSZ + payload_bytes);
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaLrModelHeader *) base;
	hdr->feature_dim	= model->n_features;
	hdr->n_samples		= model->n_samples;
	hdr->max_iters		= model->max_iters;
	hdr->learning_rate	= model->learning_rate;
	hdr->lambda			= model->lambda;
	hdr->bias			= model->bias;

	weights_dest = (float *) (base + sizeof(NdbCudaLrModelHeader));
	if (model->weights != NULL)
	{
		int i;

		for (i = 0; i < model->n_features; i++)
			weights_dest[i] = (float) model->weights[i];
	}

	if (metrics != NULL)
	{
		StringInfoData	buf;
		Jsonb		   *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"logistic_regression\","
						 "\"storage\":\"gpu\","
						 "\"n_features\":%d,"
						 "\"n_samples\":%d,"
						 "\"max_iters\":%d,"
						 "\"learning_rate\":%.6f,"
						 "\"lambda\":%.6f,"
						 "\"final_loss\":%.6f,"
						 "\"accuracy\":%.6f}",
						 model->n_features,
						 model->n_samples,
						 model->max_iters,
						 model->learning_rate,
						 model->lambda,
						 model->final_loss,
						 model->accuracy);

		metrics_json = DatumGetJsonbP(DirectFunctionCall1(
												jsonb_in,
												CStringGetDatum(buf.data)));
		pfree(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

int
ndb_cuda_lr_train(const float *features,
				  const double *labels,
				  int n_samples,
				  int feature_dim,
				  const Jsonb *hyperparams,
				  bytea **model_data,
				  Jsonb **metrics,
				  char **errstr)
{
	const int		default_max_iters = 1000;
	const double	default_learning_rate = 0.01;
	const double	default_lambda = 0.001;
	int				max_iters = default_max_iters;
	double			learning_rate = default_learning_rate;
	double			lambda = default_lambda;
	float		   *d_features = NULL;
	double		   *d_labels = NULL;
	double		   *d_predictions = NULL;
	double		   *d_z = NULL;
	double		   *weights = NULL;
	double		   *grad_weights = NULL;
	double			bias = 0.0;
	double			grad_bias = 0.0;
	bytea		   *payload = NULL;
	Jsonb		   *metrics_json = NULL;
	cudaError_t		status = cudaSuccess;
	size_t			feature_bytes;
	size_t			label_bytes;
	size_t			pred_bytes;
	size_t			z_bytes;
	size_t			weight_bytes;
	size_t			weight_bytes_gpu;
	size_t			free_mem;
	size_t			total_mem;
	float		   *d_weights = NULL;
	int				iter;
	int				i;
	int				rc = -1;

	if (errstr)
		*errstr = NULL;

	elog(DEBUG1,
		 "ndb_cuda_lr_train: entry: model_data=%p, features=%p, "
		 "labels=%p, n_samples=%d, feature_dim=%d",
		 model_data,
		 features,
		 labels,
		 n_samples,
		 feature_dim);

	if (model_data == NULL ||
		features == NULL ||
		labels == NULL ||
		n_samples <= 0 ||
		feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup(
				"invalid input parameters for CUDA LR train");
		elog(DEBUG1, "ndb_cuda_lr_train: invalid parameters detected");
		return -1;
	}

	elog(DEBUG1,
		 "ndb_cuda_lr_train: starting training: n_samples=%d, "
		 "feature_dim=%d",
		 n_samples,
		 feature_dim);

	/* Extract hyperparameters from JSONB */
	if (hyperparams != NULL)
	{
		Datum	max_iters_datum;
		Datum	lr_datum;
		Datum	lambda_datum;
		Datum	numeric_datum;
		Numeric	num;

		max_iters_datum = DirectFunctionCall2(jsonb_object_field,
											  JsonbPGetDatum(hyperparams),
											  CStringGetTextDatum("max_iters"));
		if (DatumGetPointer(max_iters_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(jsonb_numeric, max_iters_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				max_iters = DatumGetInt32(DirectFunctionCall1(
							numeric_int4, NumericGetDatum(num)));
				if (max_iters <= 0)
					max_iters = default_max_iters;
			}
		}

		lr_datum = DirectFunctionCall2(jsonb_object_field,
									   JsonbPGetDatum(hyperparams),
									   CStringGetTextDatum("learning_rate"));
		if (DatumGetPointer(lr_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(jsonb_numeric, lr_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				learning_rate = DatumGetFloat8(
					DirectFunctionCall1(numeric_float8, NumericGetDatum(num)));
				if (learning_rate <= 0.0)
					learning_rate = default_learning_rate;
			}
		}

		lambda_datum = DirectFunctionCall2(jsonb_object_field,
										   JsonbPGetDatum(hyperparams),
										   CStringGetTextDatum("lambda"));
		if (DatumGetPointer(lambda_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(jsonb_numeric, lambda_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				lambda = DatumGetFloat8(
					DirectFunctionCall1(numeric_float8, NumericGetDatum(num)));
				if (lambda < 0.0)
					lambda = default_lambda;
			}
		}
	}

	/* Allocate host memory */
	weights = (double *) palloc0(sizeof(double) * (size_t) feature_dim);
	grad_weights = (double *) palloc(sizeof(double) * (size_t) feature_dim);

	/* Allocate GPU memory */
	feature_bytes = sizeof(float) * (size_t) n_samples * (size_t) feature_dim;
	label_bytes = sizeof(double) * (size_t) n_samples;
	pred_bytes = sizeof(double) * (size_t) n_samples;
	z_bytes = sizeof(double) * (size_t) n_samples;
	weight_bytes = sizeof(double) * (size_t) feature_dim;
	weight_bytes_gpu = sizeof(float) * (size_t) feature_dim;

	elog(DEBUG1,
		 "ndb_cuda_lr_train: allocating GPU memory: feature_bytes=%zu "
		 "(%.2f MB), label_bytes=%zu",
		 feature_bytes,
		 feature_bytes / (1024.0 * 1024.0),
		 label_bytes);

	/* Check available GPU memory before allocation */
	cudaMemGetInfo(&free_mem, &total_mem);
	elog(DEBUG1,
		 "ndb_cuda_lr_train: GPU memory: free=%.2f MB, total=%.2f MB",
		 free_mem / (1024.0 * 1024.0),
		 total_mem / (1024.0 * 1024.0));

	status = cudaMalloc((void **) &d_features, feature_bytes);
	if (status != cudaSuccess)
	{
		elog(DEBUG1,
			 "ndb_cuda_lr_train: cudaMalloc d_features failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}
	elog(DEBUG1, "ndb_cuda_lr_train: d_features allocated: %p", d_features);

	status = cudaMalloc((void **) &d_labels, label_bytes);
	if (status != cudaSuccess)
	{
		elog(DEBUG1,
			 "ndb_cuda_lr_train: cudaMalloc d_labels failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}
	status = cudaMalloc((void **) &d_predictions, pred_bytes);
	if (status != cudaSuccess)
	{
		elog(DEBUG1,
			 "ndb_cuda_lr_train: cudaMalloc d_predictions failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}
	status = cudaMalloc((void **) &d_z, z_bytes);
	if (status != cudaSuccess)
	{
		elog(DEBUG1,
			 "ndb_cuda_lr_train: cudaMalloc d_z failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	elog(DEBUG1,
		 "ndb_cuda_lr_train: all GPU memory allocated successfully");

	/* Copy data to GPU */
	status = cudaMemcpy(d_features, features, feature_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto gpu_fail;
	status = cudaMemcpy(d_labels, labels, label_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto gpu_fail;

	/* Allocate GPU memory for weights (needed for forward pass) */
	status = cudaMalloc((void **) &d_weights, weight_bytes_gpu);
	if (status != cudaSuccess)
	{
		elog(DEBUG1,
			 "ndb_cuda_lr_train: cudaMalloc d_weights failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	/* Gradient descent */
	for (iter = 0; iter < max_iters; iter++)
	{
		/* Copy weights to GPU (convert double to float) */
		{
			float  *h_weights;
			int		j;

			h_weights = (float *) palloc(sizeof(float) * (size_t) feature_dim);
			for (j = 0; j < feature_dim; j++)
				h_weights[j] = (float) weights[j];
			status = cudaMemcpy(d_weights,
								h_weights,
								weight_bytes_gpu,
								cudaMemcpyHostToDevice);
			pfree(h_weights);
			if (status != cudaSuccess)
			{
				elog(DEBUG1,
					 "ndb_cuda_lr_train: cudaMemcpy weights failed: %s",
					 cudaGetErrorString(status));
				goto gpu_fail;
			}
		}

		/* Forward pass: compute z = X * w + b directly on GPU */
		{
			int ret;

			/* Use the GPU wrapper function that accepts pre-allocated GPU memory */
			ret = ndb_cuda_lr_forward_pass_gpu(d_features,
											   d_weights,
											   (float) bias,
											   n_samples,
											   feature_dim,
											   d_z);
			if (ret != 0)
			{
				elog(DEBUG1,
					 "ndb_cuda_lr_train: forward_pass_gpu failed at iter %d",
					 iter);
				if (errstr && *errstr == NULL)
					*errstr = psprintf("forward_pass_gpu failed at iter %d",
									   iter);
				goto gpu_fail;
			}

			/* Synchronize to ensure kernel completes */
			status = cudaDeviceSynchronize();
			if (status != cudaSuccess)
			{
				elog(DEBUG1,
					 "ndb_cuda_lr_train: cudaDeviceSynchronize failed at iter %d: %s",
					 iter,
					 cudaGetErrorString(status));
				if (errstr && *errstr == NULL)
					*errstr = psprintf("CUDA sync failed: %s",
									   cudaGetErrorString(status));
				goto gpu_fail;
			}

			elog(DEBUG1, "ndb_cuda_lr_train: forward_pass succeeded at iter %d", iter);
		}

		/* Apply sigmoid */
		/* Note: ndb_cuda_lr_sigmoid expects host pointers and handles GPU allocation internally */
		/* We need to copy d_z to host first, then call sigmoid, then copy back */
		{
			double	   *host_z;
			cudaError_t z_status;

			host_z = (double *) palloc(sizeof(double) * (size_t) n_samples);
			z_status = cudaMemcpy(
					host_z, d_z, z_bytes, cudaMemcpyDeviceToHost);
			if (z_status != cudaSuccess)
			{
				pfree(host_z);
				elog(DEBUG1,
					 "ndb_cuda_lr_train: cudaMemcpy d_z failed: %s",
					 cudaGetErrorString(z_status));
				goto gpu_fail;
			}

			if (ndb_cuda_lr_sigmoid(host_z, n_samples, host_z) != 0)
			{
				pfree(host_z);
				elog(DEBUG1,
					 "ndb_cuda_lr_train: sigmoid failed at iter %d",
					 iter);
				goto gpu_fail;
			}

			z_status = cudaMemcpy(d_predictions,
								  host_z,
								  pred_bytes,
								  cudaMemcpyHostToDevice);
			pfree(host_z);
			if (z_status != cudaSuccess)
			{
				elog(DEBUG1,
					 "ndb_cuda_lr_train: cudaMemcpy predictions failed: %s",
					 cudaGetErrorString(z_status));
				goto gpu_fail;
			}
		}

		/* Copy predictions back to host for gradient computation */
		{
			double *host_predictions;

			host_predictions = (double *) palloc(sizeof(double) * (size_t) n_samples);

			status = cudaMemcpy(host_predictions,
							   d_predictions,
							   pred_bytes,
							   cudaMemcpyDeviceToHost);
			if (status != cudaSuccess)
			{
				pfree(host_predictions);
				goto gpu_fail;
			}

			/* Compute gradients */
			/* Note: ndb_cuda_lr_compute_gradients expects host pointers and handles GPU allocation internally */
			if (ndb_cuda_lr_compute_gradients(features,
											  labels,
											  host_predictions,
											  n_samples,
											  feature_dim,
											  grad_weights,
											  &grad_bias) != 0)
			{
				pfree(host_predictions);
				elog(DEBUG1,
					 "ndb_cuda_lr_train: compute_gradients failed at iter %d",
					 iter);
				goto gpu_fail;
			}

			pfree(host_predictions);
		}

		/* Average gradients and add L2 regularization */
		grad_bias /= (double) n_samples;
		for (i = 0; i < feature_dim; i++)
		{
			grad_weights[i] = grad_weights[i] / (double) n_samples +
							  lambda * weights[i];
		}

		/* Update weights and bias */
		bias -= learning_rate * grad_bias;
		for (i = 0; i < feature_dim; i++)
			weights[i] -= learning_rate * grad_weights[i];

		/* Reset gradients for next iteration */
		memset(grad_weights, 0, weight_bytes);
		grad_bias = 0.0;
	}

	/* Build model payload */
	{
		size_t		payload_bytes;
		char	   *base;
		NdbCudaLrModelHeader *hdr;
		float	   *weights_dest;

		payload_bytes = sizeof(NdbCudaLrModelHeader) +
						sizeof(float) * (size_t) feature_dim;
		payload = (bytea *) palloc(VARHDRSZ + payload_bytes);
		SET_VARSIZE(payload, VARHDRSZ + payload_bytes);
		base = VARDATA(payload);

		hdr = (NdbCudaLrModelHeader *) base;
		hdr->feature_dim	= feature_dim;
		hdr->n_samples		= n_samples;
		hdr->max_iters		= max_iters;
		hdr->learning_rate	= learning_rate;
		hdr->lambda			= lambda;
		hdr->bias			= bias;

		weights_dest = (float *) (base + sizeof(NdbCudaLrModelHeader));
		for (i = 0; i < feature_dim; i++)
			weights_dest[i] = (float) weights[i];
	}

	/* Compute final loss for metrics and build metrics JSON */
	/* Use GPU predictions that were already computed in the last iteration */
	{
		double	final_loss = 0.0;
		double	accuracy = 0.0;
		double *host_preds = NULL;
		int		correct = 0;

		/* Copy final predictions from GPU (they were computed in the last iteration) */
		host_preds = (double *) palloc(sizeof(double) * (size_t) n_samples);
		status = cudaMemcpy(host_preds,
							d_predictions,
							pred_bytes,
							cudaMemcpyDeviceToHost);
		if (status == cudaSuccess)
		{
			int idx;
			/* Predictions are already sigmoided from the last iteration */
			for (idx = 0; idx < n_samples; idx++)
			{
				double pred = fmax(1e-15,
								   fmin(1.0 - 1e-15, host_preds[idx]));

				if (labels[idx] > 0.5)
					final_loss -= log(pred);
				else
					final_loss -= log(1.0 - pred);

				if ((pred >= 0.5 && labels[idx] > 0.5) ||
					(pred <  0.5 && labels[idx] <= 0.5))
					correct++;
			}
			final_loss /= (double) n_samples;
			accuracy = (correct > 0 && n_samples > 0) ?
								((double) correct / (double) n_samples) :
								0.0;
		}
		else
		{
			elog(WARNING,
				 "ndb_cuda_lr_train: failed to copy predictions for metrics, using defaults");
			final_loss = 0.0;
			accuracy = 0.0;
		}

		/* Build metrics JSON - always build, even if forward_pass failed */
		{
			StringInfoData buf;

			initStringInfo(&buf);
			appendStringInfo(&buf,
							 "{\"algorithm\":\"logistic_regression\","
							 "\"storage\":\"gpu\","
							 "\"n_features\":%d,"
							 "\"n_samples\":%d,"
							 "\"max_iters\":%d,"
							 "\"learning_rate\":%.6f,"
							 "\"lambda\":%.6f,"
							 "\"final_loss\":%.6f,"
							 "\"accuracy\":%.6f}",
							 feature_dim,
							 n_samples,
							 max_iters,
							 learning_rate,
							 lambda,
							 final_loss,
							 accuracy);

			metrics_json = DatumGetJsonbP(DirectFunctionCall1(
												jsonb_in,
												CStringGetDatum(buf.data)));
			pfree(buf.data);
			if (metrics_json == NULL)
			{
				elog(ERROR,
					 "ndb_cuda_lr_train: failed to create metrics_json from JSON string");
			}
			elog(DEBUG1,
				 "ndb_cuda_lr_train: created metrics_json: %p",
				 (void *) metrics_json);
		}

		if (host_preds != NULL)
			pfree(host_preds);
	}

	*model_data = payload;

	/* Always set metrics - build default if metrics_json is NULL */
	if (metrics != NULL)
	{
		if (metrics_json == NULL)
		{
			elog(WARNING,
				 "ndb_cuda_lr_train: metrics_json is NULL, building default metrics");
			/* Build default metrics JSON */
			{
				StringInfoData buf;

				initStringInfo(&buf);
				appendStringInfo(&buf,
								 "{\"algorithm\":\"logistic_regression\","
								 "\"storage\":\"gpu\","
								 "\"n_features\":%d,"
								 "\"n_samples\":%d,"
								 "\"max_iters\":%d,"
								 "\"learning_rate\":%.6f,"
								 "\"lambda\":%.6f,"
								 "\"final_loss\":0.0,"
								 "\"accuracy\":0.0}",
								 feature_dim,
								 n_samples,
								 max_iters,
								 learning_rate,
								 lambda);
				metrics_json = DatumGetJsonbP(
						DirectFunctionCall1(jsonb_in,
										   CStringGetDatum(buf.data)));
				pfree(buf.data);
			}
		}
		*metrics = metrics_json;
		elog(DEBUG1,
			 "ndb_cuda_lr_train: setting *metrics = %p (metrics_json=%p)",
			 (void *) *metrics,
			 (void *) metrics_json);
	}
	else
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: metrics output parameter is NULL!");
	}
	rc = 0;

cleanup:
	if (weights)
		pfree(weights);
	if (grad_weights)
		pfree(grad_weights);

	if (d_features)
		cudaFree(d_features);
	if (d_labels)
		cudaFree(d_labels);
	if (d_predictions)
		cudaFree(d_predictions);
	if (d_z)
		cudaFree(d_z);

	return rc;

gpu_fail:
	if (errstr)
		*errstr = pstrdup(cudaGetErrorString(status));
	rc = -1;
	goto cleanup;
}

int
ndb_cuda_lr_predict(const bytea *model_data,
					const float *input,
					int feature_dim,
					double *probability_out,
					char **errstr)
{
	const NdbCudaLrModelHeader *hdr;
	const float *weights;
	const bytea *detoasted;
	double		z;
	int			i;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || probability_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for CUDA LR predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted = (const bytea *) PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t	expected_size = sizeof(NdbCudaLrModelHeader) +
								sizeof(float) * (size_t) feature_dim;
		size_t	actual_size = VARSIZE(detoasted) - VARHDRSZ;

		elog(DEBUG1,
			 "ndb_cuda_lr_predict: payload size check: expected=%zu, actual=%zu, feature_dim=%d",
			 expected_size,
			 actual_size,
			 feature_dim);

		if (actual_size < expected_size)
		{
			if (errstr)
				*errstr = psprintf("model data too small: expected %zu bytes, got %zu",
								   expected_size, actual_size);
			elog(DEBUG1, "ndb_cuda_lr_predict: %s", *errstr);
			return -1;
		}
	}

	hdr = (const NdbCudaLrModelHeader *) VARDATA(detoasted);
	elog(DEBUG1,
		 "ndb_cuda_lr_predict: header feature_dim=%d, input feature_dim=%d",
		 hdr->feature_dim,
		 feature_dim);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model has %d, input has %d",
							   hdr->feature_dim,
							   feature_dim);
		elog(DEBUG1, "ndb_cuda_lr_predict: %s", *errstr);
		return -1;
	}

	weights = (const float *) ((const char *) hdr +
							   sizeof(NdbCudaLrModelHeader));

	z = hdr->bias;
	for (i = 0; i < feature_dim; i++)
		z += weights[i] * input[i];

	*probability_out = 1.0 / (1.0 + exp(-z));
	return 0;
}

/* Kernel wrapper functions are implemented in gpu_lr_kernels.cu */

#else	/* !NDB_GPU_CUDA */

int
ndb_cuda_lr_pack_model(const LRModel *model,
					  bytea **model_data,
					  Jsonb **metrics,
					  char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA support not compiled");
	(void) model;
	(void) model_data;
	(void) metrics;
	return -1;
}

int
ndb_cuda_lr_train(const float *features,
				  const double *labels,
				  int n_samples,
				  int feature_dim,
				  const Jsonb *hyperparams,
				  bytea **model_data,
				  Jsonb **metrics,
				  char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA support not compiled");
	(void) features;
	(void) labels;
	(void) n_samples;
	(void) feature_dim;
	(void) hyperparams;
	(void) model_data;
	(void) metrics;
	return -1;
}

int
ndb_cuda_lr_predict(const bytea *model_data,
					const float *input,
					int feature_dim,
					double *probability_out,
					char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA support not compiled");
	(void) model_data;
	(void) input;
	(void) feature_dim;
	(void) probability_out;
	return -1;
}

/* These functions are implemented in gpu_lr_kernels.cu */

#endif	/* NDB_GPU_CUDA */
