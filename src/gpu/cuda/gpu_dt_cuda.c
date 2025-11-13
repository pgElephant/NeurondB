/*-------------------------------------------------------------------------
 *
 * gpu_dt_cuda.c
 *    CUDA backend bridge for Decision Tree training and prediction.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_dt_cuda.c
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

#include "ml_decision_tree_internal.h"
#include "neurondb_cuda_dt.h"

/*
 * Helper function to serialize tree nodes recursively
 */
static int
dt_serialize_node_recursive(const DTNode *node,
	NdbCudaDtNode *dest,
	int *node_idx,
	int *max_idx)
{
	int current_idx;
	int left_idx = -1;
	int right_idx = -1;

	if (node == NULL || dest == NULL || node_idx == NULL)
		return -1;

	current_idx = *node_idx;
	if (current_idx >= *max_idx)
		return -1;

	dest[current_idx].is_leaf = node->is_leaf;
	dest[current_idx].value = (float)node->leaf_value;
	dest[current_idx].feature_idx = node->feature_idx;
	dest[current_idx].threshold = node->threshold;
	dest[current_idx].left_child = -1;
	dest[current_idx].right_child = -1;

	if (!node->is_leaf)
	{
		(*node_idx)++;
		left_idx = *node_idx;
		if (dt_serialize_node_recursive(
			    node->left, dest, node_idx, max_idx)
			< 0)
			return -1;

		(*node_idx)++;
		right_idx = *node_idx;
		if (dt_serialize_node_recursive(
			    node->right, dest, node_idx, max_idx)
			< 0)
			return -1;

		dest[current_idx].left_child = left_idx;
		dest[current_idx].right_child = right_idx;
	}

	return 0;
}

/*
 * Count nodes in tree recursively
 */
static int
dt_count_nodes(const DTNode *node)
{
	if (node == NULL)
		return 0;
	if (node->is_leaf)
		return 1;
	return 1 + dt_count_nodes(node->left) + dt_count_nodes(node->right);
}

int
ndb_cuda_dt_pack_model(const DTModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	int node_count = 0;
	size_t header_bytes;
	size_t nodes_bytes;
	size_t payload_bytes;
	bytea *blob;
	char *base;
	NdbCudaDtModelHeader *hdr;
	NdbCudaDtNode *nodes;
	int node_idx = 0;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid DT model for CUDA pack");
		return -1;
	}

	if (model->root == NULL)
	{
		if (errstr)
			*errstr =
				pstrdup("decision_tree model has no root node");
		return -1;
	}

	node_count = dt_count_nodes(model->root);
	if (node_count <= 0)
	{
		if (errstr)
			*errstr = pstrdup("decision_tree model empty");
		return -1;
	}

	header_bytes = sizeof(NdbCudaDtModelHeader);
	nodes_bytes = sizeof(NdbCudaDtNode) * (size_t)node_count;
	payload_bytes = header_bytes + nodes_bytes;

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaDtModelHeader *)base;
	hdr->feature_dim = model->n_features;
	hdr->n_samples = model->n_samples;
	hdr->max_depth = model->max_depth;
	hdr->min_samples_split = model->min_samples_split;
	hdr->node_count = node_count;

	nodes = (NdbCudaDtNode *)(base + sizeof(NdbCudaDtModelHeader));
	if (dt_serialize_node_recursive(
		    model->root, nodes, &node_idx, &node_count)
		< 0)
	{
		pfree(blob);
		if (errstr)
			*errstr = pstrdup(
				"failed to serialize decision tree nodes");
		return -1;
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"decision_tree\","
			"\"storage\":\"gpu\","
			"\"n_features\":%d,"
			"\"n_samples\":%d,"
			"\"max_depth\":%d,"
			"\"min_samples_split\":%d,"
			"\"node_count\":%d}",
			model->n_features,
			model->n_samples,
			model->max_depth,
			model->min_samples_split,
			node_count);

		metrics_json = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(buf.data)));
		pfree(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

int
ndb_cuda_dt_train(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	/* Decision Tree training on GPU is complex - for now, fall back to CPU */
	/* This can be enhanced later with GPU-accelerated split finding */
	if (errstr)
		*errstr = pstrdup("Decision Tree GPU training not yet "
				  "implemented, use CPU training");
	return -1;
}

int
ndb_cuda_dt_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	double *prediction_out,
	char **errstr)
{
	const NdbCudaDtModelHeader *hdr;
	const NdbCudaDtNode *nodes;
	const bytea *detoasted;
	int node_idx = 0;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || prediction_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup(
				"invalid parameters for CUDA DT predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted =
		(const bytea *)PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t min_size = sizeof(NdbCudaDtModelHeader);
		size_t actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < min_size)
		{
			if (errstr)
				*errstr = psprintf(
					"model data too small: expected at "
					"least %zu bytes, got %zu",
					min_size,
					actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaDtModelHeader *)VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model "
					   "has %d, input has %d",
				hdr->feature_dim,
				feature_dim);
		return -1;
	}

	nodes = (const NdbCudaDtNode *)((const char *)hdr
		+ sizeof(NdbCudaDtModelHeader));

	/* Traverse tree to make prediction */
	while (node_idx >= 0 && node_idx < hdr->node_count)
	{
		const NdbCudaDtNode *node = &nodes[node_idx];

		if (node->is_leaf)
		{
			*prediction_out = (double)node->value;
			return 0;
		}

		if (input[node->feature_idx] <= node->threshold)
			node_idx = node->left_child;
		else
			node_idx = node->right_child;
	}

	if (errstr)
		*errstr = pstrdup(
			"tree traversal failed - invalid tree structure");
	return -1;
}

#endif /* NDB_GPU_CUDA */
