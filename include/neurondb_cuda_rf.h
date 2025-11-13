/*-------------------------------------------------------------------------
 *
 * neurondb_cuda_rf.h
 *	  CUDA-backed Random Forest data structures and API surface.
 *
 * Defines host-side representations for GPU random forest models and the
 * entry points used by the CUDA backend. Implementations live under
 * src/gpu/cuda.
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_CUDA_RF_H
#define NEURONDB_CUDA_RF_H

#ifndef __CUDACC__
#include "postgres.h"
#include "utils/jsonb.h"
#else
struct varlena;
typedef struct varlena bytea;
struct Jsonb;
#endif

struct RFModel;

typedef struct NdbCudaRfNode
{
	int feature_idx;
	float threshold;
	int left_child;
	int right_child;
	float value;
} NdbCudaRfNode;

typedef struct NdbCudaRfModelHeader
{
	int tree_count;
	int feature_dim;
	int class_count;
	int sample_count;
	int majority_class;
	double majority_fraction;
} NdbCudaRfModelHeader;

typedef struct NdbCudaRfTreeHeader
{
	int node_count;
	int nodes_start;
	int root_index;
} NdbCudaRfTreeHeader;

#ifdef __cplusplus
extern "C" {
#endif

extern int ndb_cuda_rf_pack_model(const struct RFModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_cuda_rf_train(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	int class_count,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_cuda_rf_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	int *class_out,
	char **errstr);

extern int ndb_cuda_rf_infer(const NdbCudaRfNode *nodes,
	const NdbCudaRfTreeHeader *trees,
	int tree_count,
	const float *input,
	int feature_dim,
	int class_count,
	int *votes);

extern int ndb_cuda_rf_histogram(const int *labels,
	int n_samples,
	int class_count,
	int *counts_out);

extern int ndb_cuda_rf_launch_feature_stats(const float *features,
	int n_samples,
	int feature_dim,
	int feature_idx,
	double *sum_dev,
	double *sumsq_dev);

extern int ndb_cuda_rf_launch_split_counts(const float *features,
	const int *labels,
	int n_samples,
	int feature_dim,
	int feature_idx,
	float threshold,
	int class_count,
	int *left_counts_dev,
	int *right_counts_dev);

#ifdef __cplusplus
}
#endif

#endif /* NEURONDB_CUDA_RF_H */
