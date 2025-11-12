/*-------------------------------------------------------------------------
 *
 * gpu_svm_cuda.c
 *    CUDA backend bridge for Support Vector Machine training and prediction.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_svm_cuda.c
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

#include "ml_svm_internal.h"
#include "neurondb_cuda_svm.h"

int
ndb_cuda_svm_pack_model(const SVMModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	size_t payload_bytes;
	bytea *blob;
	char *base;
	NdbCudaSvmModelHeader *hdr;
	float *alphas_dest;
	float *sv_dest;
	int32 *indices_dest;
	int i, j;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid SVM model for CUDA pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaSvmModelHeader)
		+ sizeof(float) * (size_t)model->n_support_vectors
		+ sizeof(float) * (size_t)model->n_support_vectors * (size_t)model->n_features
		+ sizeof(int32) * (size_t)model->n_support_vectors;

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaSvmModelHeader *)base;
	hdr->feature_dim = model->n_features;
	hdr->n_samples = model->n_samples;
	hdr->n_support_vectors = model->n_support_vectors;
	hdr->bias = (float)model->bias;
	hdr->C = model->C;
	hdr->max_iters = model->max_iters;

	alphas_dest = (float *)(base + sizeof(NdbCudaSvmModelHeader));
	sv_dest = alphas_dest + model->n_support_vectors;
	indices_dest = (int32 *)(sv_dest + model->n_support_vectors * model->n_features);

	if (model->alphas != NULL && model->support_vectors != NULL && model->support_vector_indices != NULL)
	{
		for (i = 0; i < model->n_support_vectors; i++)
		{
			alphas_dest[i] = (float)model->alphas[i];
			indices_dest[i] = model->support_vector_indices[i];
			for (j = 0; j < model->n_features; j++)
				sv_dest[i * model->n_features + j] = model->support_vectors[i * model->n_features + j];
		}
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"svm\","
			"\"storage\":\"gpu\","
			"\"n_features\":%d,"
			"\"n_samples\":%d,"
			"\"n_support_vectors\":%d,"
			"\"C\":%.6f,"
			"\"max_iters\":%d}",
			model->n_features,
			model->n_samples,
			model->n_support_vectors,
			model->C,
			model->max_iters);

		metrics_json = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(buf.data)));
		pfree(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

int
ndb_cuda_svm_train(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	/* Placeholder - full GPU SMO implementation can be added later */
	/* For now, return -1 to fall back to CPU */
	if (errstr)
		*errstr = pstrdup("GPU SVM training not yet implemented");
	return -1;
}

int
ndb_cuda_svm_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	int *class_out,
	double *confidence_out,
	char **errstr)
{
	const NdbCudaSvmModelHeader *hdr;
	const float *alphas;
	const float *support_vectors;
	const int32 *indices;
	const bytea *detoasted;
	double prediction;
	int i, j;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || class_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for CUDA SVM predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted = (const bytea *)PG_DETOAST_DATUM(PointerGetDatum(model_data));
	
	hdr = (const NdbCudaSvmModelHeader *)VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model has %d, input has %d", 
				hdr->feature_dim, feature_dim);
		return -1;
	}

	alphas = (const float *)((const char *)hdr + sizeof(NdbCudaSvmModelHeader));
	support_vectors = alphas + hdr->n_support_vectors;
	indices = (const int32 *)(support_vectors + hdr->n_support_vectors * hdr->feature_dim);

	/* Compute prediction: f(x) = Σ(alpha_i * y_i * K(x_i, x)) + bias */
	prediction = hdr->bias;
	for (i = 0; i < hdr->n_support_vectors; i++)
	{
		double kernel_val = 0.0;
		const float *sv = support_vectors + (i * feature_dim);
		
		/* Linear kernel: K(x_i, x) = x_i · x */
		for (j = 0; j < feature_dim; j++)
			kernel_val += sv[j] * input[j];
		
		/* Note: y_i is stored implicitly via label sign in training */
		/* For now, assume positive label for all support vectors */
		prediction += alphas[i] * kernel_val;
	}

	*class_out = (prediction >= 0.0) ? 1 : 0;
	if (confidence_out != NULL)
		*confidence_out = fabs(prediction);

	return 0;
}

#endif /* NDB_GPU_CUDA */

