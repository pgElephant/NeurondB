/*-------------------------------------------------------------------------
 *
 * gpu_rf_rocm.c
 *    ROCm backend bridge for Random Forest training and prediction.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_rf_rocm.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#ifdef NDB_GPU_HIP

#include <float.h>
#include <math.h>
#include <string.h>

#include <hip/hip_runtime.h>
#include "neurondb_rocm_launchers.h"
#include "common/pg_prng.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "miscadmin.h"
#include "common/hashfn.h"
#include "storage/ipc.h"

#include "ml_random_forest_internal.h"
#include "ml_random_forest_shared.h"
#include "neurondb_cuda_rf.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"

/* Forward declarations for kernel launchers */
extern int launch_rf_predict_batch_kernel_hip(const NdbCudaRfNode *d_nodes,
	const NdbCudaRfTreeHeader *d_trees,
	int tree_count,
	const float *d_features,
	int n_samples,
	int feature_dim,
	int class_count,
	int *d_votes);
extern int ndb_rocm_rf_infer(const NdbCudaRfNode *nodes,
	const NdbCudaRfTreeHeader *trees,
	int tree_count,
	const float *input,
	int feature_dim,
	int class_count,
	int *votes);

static void
rf_copy_tree_nodes_rocm(const GTree *tree, NdbCudaRfNode *dest,
	int *node_offset, int *max_feat_idx)
{
	const GTreeNode *src_nodes;
	int count;
	int i;
	int max_idx = -1;

	if (tree == NULL || dest == NULL || node_offset == NULL)
		return;

	src_nodes = gtree_nodes(tree);
	count = tree->count;
	for (i = 0; i < count; i++)
	{
		const GTreeNode *src = &src_nodes[i];
		NdbCudaRfNode *dst = &dest[*node_offset + i];

		dst->feature_idx = src->feature_idx;
		dst->threshold = (float)src->threshold;
		if (src->is_leaf)
		{
			dst->left_child = -1;
			dst->right_child = -1;
		} else
		{
			dst->left_child = src->left;
			dst->right_child = src->right;
		}
		dst->value = (float)src->value;

		/* Track maximum feature index */
		if (src->feature_idx >= 0 && src->feature_idx > max_idx)
			max_idx = src->feature_idx;
	}
	*node_offset += count;
	if (max_feat_idx != NULL)
		*max_feat_idx = max_idx;
}

int
ndb_rocm_rf_pack_model(const struct RFModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	int tree_count = 0;
	int total_nodes = 0;
	int i;
	size_t header_bytes;
	size_t nodes_bytes;
	size_t payload_bytes;
	bytea *blob;
	char *base;
	NdbCudaRfModelHeader *model_hdr;
	NdbCudaRfTreeHeader *tree_hdrs;
	NdbCudaRfNode *nodes;
	int node_cursor = 0;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid RF model for ROCm pack");
		return -1;
	}

	if (model->tree_count > 0 && model->trees != NULL)
	{
		tree_count = model->tree_count;
		for (i = 0; i < model->tree_count; i++)
		{
			const GTree *tree = model->trees[i];

			if (tree != NULL)
				total_nodes += tree->count;
		}
	} else if (model->tree != NULL)
	{
		tree_count = 1;
		total_nodes = model->tree->count;
	} else
	{
		if (errstr)
			*errstr = pstrdup("random_forest model has no trees");
		return -1;
	}

	if (tree_count <= 0 || total_nodes <= 0)
	{
		if (errstr)
			*errstr = pstrdup("random_forest model empty");
		return -1;
	}

	header_bytes = sizeof(NdbCudaRfModelHeader)
		+ (sizeof(NdbCudaRfTreeHeader) * tree_count);
	nodes_bytes = sizeof(NdbCudaRfNode) * total_nodes;
	payload_bytes = header_bytes + nodes_bytes;

	/* Defensive check for integer overflow and MaxAllocSize */
	if (payload_bytes > MaxAllocSize || header_bytes > MaxAllocSize ||
		nodes_bytes > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("ROCm RF pack: payload size exceeds MaxAllocSize");
		return -1;
	}
	if (VARHDRSZ + payload_bytes > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("ROCm RF pack: total size exceeds MaxAllocSize");
		return -1;
	}

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
	if (blob == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ROCm RF pack: palloc failed");
		return -1;
	}
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	model_hdr = (NdbCudaRfModelHeader *)base;
	model_hdr->tree_count = tree_count;
	model_hdr->feature_dim = model->n_features;
	model_hdr->class_count = model->n_classes;
	model_hdr->sample_count = model->n_samples;
	model_hdr->majority_class = (int)rint(model->majority_value);
	model_hdr->majority_fraction = model->majority_fraction;

	tree_hdrs =
		(NdbCudaRfTreeHeader *)(base + sizeof(NdbCudaRfModelHeader));
	nodes = (NdbCudaRfNode *)(base + header_bytes);

	node_cursor = 0;
	for (i = 0; i < tree_count; i++)
	{
		const GTree *tree =
			(model->tree_count > 0 && model->trees != NULL)
			? model->trees[i]
			: model->tree;
		int node_count = tree ? tree->count : 0;

		tree_hdrs[i].node_count = node_count;
		tree_hdrs[i].nodes_start = node_cursor;
		tree_hdrs[i].root_index = tree ? tree->root : 0;
		tree_hdrs[i].max_feature_index = -1;

		if (node_count > 0)
			rf_copy_tree_nodes_rocm(tree, nodes, &node_cursor,
				&tree_hdrs[i].max_feature_index);
	}

	*model_data = blob;

	if (metrics != NULL)
	{
		RFMetricsSpec spec;

		memset(&spec, 0, sizeof(spec));
		spec.storage = "gpu";
		spec.algorithm = "random_forest";
		spec.tree_count = tree_count;
		spec.majority_class = model_hdr->majority_class;
		spec.majority_fraction = model_hdr->majority_fraction;
		spec.gini = model->gini_impurity;
		spec.oob_accuracy = model->oob_accuracy;
		*metrics = rf_build_metrics_json(&spec);
	}

	return 0;
}

int
ndb_rocm_rf_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	int *class_out,
	char **errstr)
{
	const NdbCudaRfModelHeader *model_hdr;
	const NdbCudaRfTreeHeader *tree_hdrs;
	const NdbCudaRfNode *nodes;
	int *votes = NULL;
	int i;
	int best_class = 0;
	int max_votes = 0;
	int rc = -1;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || class_out == NULL
		|| feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for ROCm RF predict");
		return -1;
	}

	if (VARSIZE(model_data) - VARHDRSZ < (int)sizeof(NdbCudaRfModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("ROCm RF model data too small");
		return -1;
	}

	model_hdr = (const NdbCudaRfModelHeader *)VARDATA(model_data);
	if (model_hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = pstrdup("feature dimension mismatch in ROCm RF predict");
		return -1;
	}

	tree_hdrs = (const NdbCudaRfTreeHeader *)((const char *)model_hdr
		+ sizeof(NdbCudaRfModelHeader));
	nodes = (const NdbCudaRfNode *)((const char *)tree_hdrs
		+ sizeof(NdbCudaRfTreeHeader) * model_hdr->tree_count);

	votes = (int *)palloc0(sizeof(int) * model_hdr->class_count);
	if (votes == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ROCm RF predict: failed to allocate votes");
		return -1;
	}

	rc = ndb_rocm_rf_infer(nodes,
		tree_hdrs,
		model_hdr->tree_count,
		input,
		feature_dim,
		model_hdr->class_count,
		votes);

	if (rc != 0)
	{
		if (errstr)
			*errstr = pstrdup("ROCm RF inference failed");
		NDB_SAFE_PFREE_AND_NULL(votes);
		return -1;
	}

	/* Find class with most votes */
	for (i = 0; i < model_hdr->class_count; i++)
	{
		if (votes[i] > max_votes)
		{
			max_votes = votes[i];
			best_class = i;
		}
	}

	/* Fallback to majority class if no votes */
	if (max_votes == 0)
		best_class = model_hdr->majority_class;

	*class_out = best_class;
	NDB_SAFE_PFREE_AND_NULL(votes);
	return 0;
}

int
ndb_rocm_rf_train(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	int class_count,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	/* ROCm RF training - fallback to CPU for now */
	/* TODO: Implement full GPU training */
	if (errstr)
		*errstr = pstrdup("ROCm RF training not yet implemented, use CPU");
	return -1;
}

#endif /* NDB_GPU_HIP */

