/*-------------------------------------------------------------------------
 *
 * neurondb_cuda_lr.h
 *    CUDA-specific data structures and API for Logistic Regression
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_cuda_lr.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_CUDA_LR_H
#define NEURONDB_CUDA_LR_H

#ifndef __CUDACC__
#include "postgres.h"
#include "utils/jsonb.h"
#include "ml_logistic_regression_internal.h"
#else
struct varlena;
typedef struct varlena bytea;
struct Jsonb;
struct LRModel;
#endif

typedef struct NdbCudaLrModelHeader
{
	int feature_dim;
	int n_samples;
	int max_iters;
	double learning_rate;
	double lambda;
	double bias;
} NdbCudaLrModelHeader;

#ifdef __cplusplus
extern "C" {
#endif

extern int ndb_cuda_lr_pack_model(const struct LRModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_cuda_lr_train(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_cuda_lr_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	double *probability_out,
	char **errstr);

extern int ndb_cuda_lr_forward_pass(const float *features,
	const double *weights,
	double bias,
	int n_samples,
	int feature_dim,
	double *outputs);

extern int ndb_cuda_lr_forward_pass_gpu(const float *d_features,
	const float *d_weights,
	float bias,
	int n_samples,
	int feature_dim,
	double *d_outputs);

extern int ndb_cuda_lr_compute_gradients(const float *features,
	const double *labels,
	const double *predictions,
	int n_samples,
	int feature_dim,
	double *grad_weights,
	double *grad_bias);

extern int ndb_cuda_lr_sigmoid(const double *inputs, int n, double *outputs);

#ifdef __cplusplus
}
#endif

#endif /* NEURONDB_CUDA_LR_H */
