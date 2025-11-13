/*-------------------------------------------------------------------------
 *
 * gpu_rf_cuda.c
 *	  CUDA backend bridge for Random Forest training and prediction.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_rf_cuda.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#ifdef NDB_GPU_CUDA

#include <float.h>
#include <math.h>
#include <string.h>

#include "neurondb_cuda_runtime.h"
#include "common/pg_prng.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"

#include "ml_random_forest_internal.h"
#include "ml_random_forest_shared.h"
#include "neurondb_cuda_rf.h"

static void
rf_copy_tree_nodes(const GTree *tree, NdbCudaRfNode *dest, int *node_offset)
{
	const GTreeNode *src_nodes;
	int count;
	int i;

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
	}
	*node_offset += count;
}

static void
rf_fill_single_node_tree(NdbCudaRfNode *node, int majority_class)
{
	node->feature_idx = -1;
	node->threshold = 0.0f;
	node->left_child = -1;
	node->right_child = -1;
	node->value = (float)majority_class;
}

int
ndb_cuda_rf_pack_model(const RFModel *model,
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
			*errstr = pstrdup("invalid RF model for CUDA pack");
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

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
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

		if (node_count > 0)
			rf_copy_tree_nodes(tree, nodes, &node_cursor);
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
ndb_cuda_rf_train(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	int class_count,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	const int default_n_trees = 32;
	int n_trees = default_n_trees;
	int *label_ints = NULL;
	int *class_counts = NULL;
	int *best_left_counts = NULL;
	int *best_right_counts = NULL;
	int *tmp_left_counts = NULL;
	int *tmp_right_counts = NULL;
	bytea *payload = NULL;
	Jsonb *metrics_json = NULL;
	float *d_features = NULL;
	int *d_labels = NULL;
	double *d_feature_sum = NULL;
	double *d_feature_sumsq = NULL;
	int *d_left_counts = NULL;
	int *d_right_counts = NULL;
	pg_prng_state rng;
	bool seeded = false;
	cudaError_t status = cudaSuccess;
	double gini_accumulator = 0.0;
	size_t feature_bytes;
	size_t label_bytes;
	size_t class_bytes;
	int i;
	int j;
	int rc = -1;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || labels == NULL || n_samples <= 0
		|| feature_dim <= 0 || class_count <= 0)
	{
		if (errstr)
			*errstr = pstrdup(
				"invalid input parameters for CUDA RF train");
		return -1;
	}

	(void)hyperparams;
	if (n_trees <= 0)
		n_trees = default_n_trees;
	if (class_count > 4096)
		class_count = 4096;

	label_ints = (int *)palloc(sizeof(int) * n_samples);
	class_counts = (int *)palloc0(sizeof(int) * class_count);
	tmp_left_counts = (int *)palloc(sizeof(int) * class_count);
	tmp_right_counts = (int *)palloc(sizeof(int) * class_count);
	best_left_counts = (int *)palloc(sizeof(int) * class_count);
	best_right_counts = (int *)palloc(sizeof(int) * class_count);
	for (i = 0; i < n_samples; i++)
	{
		double val = labels[i];

		label_ints[i] = (int)rint(val);
		if (label_ints[i] < 0 || label_ints[i] >= class_count)
			label_ints[i] = 0;
	}

	if (ndb_cuda_rf_histogram(
		    label_ints, n_samples, class_count, class_counts)
		!= 0)
	{
		memset(class_counts, 0, sizeof(int) * class_count);
		for (i = 0; i < n_samples; i++)
			class_counts[label_ints[i]]++;
	}

	{
		NdbCudaRfModelHeader model_hdr;
		NdbCudaRfTreeHeader *tree_hdrs;
		NdbCudaRfNode *nodes;
		int total_nodes = n_trees * 3;
		size_t header_bytes;
		size_t payload_bytes;
		char *base;
		int majority_class = 0;
		int best_count = class_counts[0];
		double majority_fraction;

		feature_bytes =
			sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
		label_bytes = sizeof(int) * (size_t)n_samples;
		class_bytes = sizeof(int) * (size_t)class_count;

		status = cudaMalloc((void **)&d_features, feature_bytes);
		if (status != cudaSuccess)
			goto gpu_fail;
		status = cudaMalloc((void **)&d_labels, label_bytes);
		if (status != cudaSuccess)
			goto gpu_fail;
		status = cudaMalloc((void **)&d_feature_sum, sizeof(double));
		if (status != cudaSuccess)
			goto gpu_fail;
		status = cudaMalloc((void **)&d_feature_sumsq, sizeof(double));
		if (status != cudaSuccess)
			goto gpu_fail;
		status = cudaMalloc((void **)&d_left_counts, class_bytes);
		if (status != cudaSuccess)
			goto gpu_fail;
		status = cudaMalloc((void **)&d_right_counts, class_bytes);
		if (status != cudaSuccess)
			goto gpu_fail;

		status = cudaMemcpy(d_features,
			features,
			feature_bytes,
			cudaMemcpyHostToDevice);
		if (status != cudaSuccess)
			goto gpu_fail;
		status = cudaMemcpy(d_labels,
			label_ints,
			label_bytes,
			cudaMemcpyHostToDevice);
		if (status != cudaSuccess)
			goto gpu_fail;

		if (!seeded)
		{
			if (!pg_prng_strong_seed(&rng))
				pg_prng_seed(&rng,
					(uint64)n_samples
						^ (uint64)feature_dim);
			seeded = true;
		}

		for (i = 1; i < class_count; i++)
		{
			if (class_counts[i] > best_count)
			{
				best_count = class_counts[i];
				majority_class = i;
			}
		}
		majority_fraction = (n_samples > 0)
			? ((double)best_count / (double)n_samples)
			: 0.0;

		header_bytes = sizeof(NdbCudaRfModelHeader)
			+ sizeof(NdbCudaRfTreeHeader) * n_trees;
		payload_bytes =
			header_bytes + sizeof(NdbCudaRfNode) * total_nodes;
		payload = (bytea *)palloc(VARHDRSZ + payload_bytes);
		SET_VARSIZE(payload, VARHDRSZ + payload_bytes);
		base = VARDATA(payload);

		model_hdr.tree_count = n_trees;
		model_hdr.feature_dim = feature_dim;
		model_hdr.class_count = class_count;
		model_hdr.sample_count = n_samples;
		model_hdr.majority_class = majority_class;
		model_hdr.majority_fraction = majority_fraction;

		memcpy(base, &model_hdr, sizeof(model_hdr));
		tree_hdrs = (NdbCudaRfTreeHeader *)(base + sizeof(model_hdr));
		nodes = (NdbCudaRfNode *)(base + header_bytes);

		for (i = 0; i < n_trees; i++)
		{
			double best_gini = DBL_MAX;
			float best_threshold = 0.0f;
			int best_feature = -1;
			int left_majority = majority_class;
			int right_majority = majority_class;
			int left_total = 0;
			int right_total = 0;
			int node_offset = i * 3;
			double noise = pg_prng_double(&rng) - 0.5;

			memset(best_left_counts, 0, class_bytes);
			memset(best_right_counts, 0, class_bytes);

			for (j = 0; j < feature_dim; j++)
			{
				double sum_host = 0.0;
				double sumsq_host = 0.0;
				double variance;
				float threshold;

				if (ndb_cuda_rf_launch_feature_stats(d_features,
					    n_samples,
					    feature_dim,
					    j,
					    d_feature_sum,
					    d_feature_sumsq)
					!= 0)
					continue;

				status = cudaMemcpy(&sum_host,
					d_feature_sum,
					sizeof(double),
					cudaMemcpyDeviceToHost);
				if (status != cudaSuccess)
					goto gpu_fail;
				status = cudaMemcpy(&sumsq_host,
					d_feature_sumsq,
					sizeof(double),
					cudaMemcpyDeviceToHost);
				if (status != cudaSuccess)
					goto gpu_fail;

				if (sum_host == 0.0 && sumsq_host == 0.0)
					continue;

				threshold =
					(float)(sum_host / (double)n_samples);
				variance = (sumsq_host / (double)n_samples)
					- ((double)threshold
						* (double)threshold);
				if (variance < 0.0)
					variance = 0.0;
				if (variance > 0.0)
					threshold += (float)(noise
						* sqrt(variance) * 0.25);

				if (ndb_cuda_rf_launch_split_counts(d_features,
					    d_labels,
					    n_samples,
					    feature_dim,
					    j,
					    threshold,
					    class_count,
					    d_left_counts,
					    d_right_counts)
					!= 0)
					continue;

				status = cudaMemcpy(tmp_left_counts,
					d_left_counts,
					class_bytes,
					cudaMemcpyDeviceToHost);
				if (status != cudaSuccess)
					goto gpu_fail;
				status = cudaMemcpy(tmp_right_counts,
					d_right_counts,
					class_bytes,
					cudaMemcpyDeviceToHost);
				if (status != cudaSuccess)
					goto gpu_fail;

				{
					double gini =
						rf_split_gini(tmp_left_counts,
							tmp_right_counts,
							class_count,
							&left_total,
							&right_total,
							NULL,
							NULL);

					if (gini < best_gini && gini >= 0.0)
					{
						best_gini = gini;
						best_feature = j;
						best_threshold = threshold;
						memcpy(best_left_counts,
							tmp_left_counts,
							class_bytes);
						memcpy(best_right_counts,
							tmp_right_counts,
							class_bytes);
					}
				}
			}

			if (best_feature < 0)
			{
				tree_hdrs[i].node_count = 1;
				tree_hdrs[i].nodes_start = node_offset;
				tree_hdrs[i].root_index = 0;
				rf_fill_single_node_tree(
					&nodes[node_offset], majority_class);
				continue;
			}

			left_total = 0;
			right_total = 0;
			for (j = 0; j < class_count; j++)
			{
				if (best_left_counts[j] > left_total)
				{
					left_total = best_left_counts[j];
					left_majority = j;
				}
				if (best_right_counts[j] > right_total)
				{
					right_total = best_right_counts[j];
					right_majority = j;
				}
			}

			tree_hdrs[i].node_count = 3;
			tree_hdrs[i].nodes_start = node_offset;
			tree_hdrs[i].root_index = 0;

			nodes[node_offset].feature_idx = best_feature;
			nodes[node_offset].threshold = best_threshold;
			nodes[node_offset].left_child = 1;
			nodes[node_offset].right_child = 2;
			nodes[node_offset].value = (float)majority_class;

			rf_fill_single_node_tree(
				&nodes[node_offset + 1], left_majority);
			rf_fill_single_node_tree(
				&nodes[node_offset + 2], right_majority);

			if (best_gini > 0.0 && best_gini < DBL_MAX / 4.0)
				gini_accumulator += best_gini;
		}

		{
			RFMetricsSpec spec;

			memset(&spec, 0, sizeof(spec));
			spec.storage = "gpu";
			spec.algorithm = "random_forest";
			spec.tree_count = n_trees;
			spec.majority_class = majority_class;
			spec.majority_fraction = majority_fraction;
			spec.gini = (gini_accumulator > 0.0)
				? (gini_accumulator / (double)n_trees)
				: 0.0;
			spec.oob_accuracy = 0.0;
			metrics_json = rf_build_metrics_json(&spec);
		}
	}

	*model_data = payload;
	if (metrics != NULL)
	{
		*metrics = metrics_json;
		metrics_json = NULL;
	}
	rc = 0;

gpu_cleanup:
	if (d_features)
		cudaFree(d_features);
	if (d_labels)
		cudaFree(d_labels);
	if (d_feature_sum)
		cudaFree(d_feature_sum);
	if (d_feature_sumsq)
		cudaFree(d_feature_sumsq);
	if (d_left_counts)
		cudaFree(d_left_counts);
	if (d_right_counts)
		cudaFree(d_right_counts);
	if (label_ints)
		pfree(label_ints);
	if (class_counts)
		pfree(class_counts);
	if (tmp_left_counts)
		pfree(tmp_left_counts);
	if (tmp_right_counts)
		pfree(tmp_right_counts);
	if (best_left_counts)
		pfree(best_left_counts);
	if (best_right_counts)
		pfree(best_right_counts);
	if (metrics_json)
		pfree(metrics_json);

	return rc;

gpu_fail:
	if (errstr != NULL)
	{
		if (status != cudaSuccess)
			*errstr = pstrdup(cudaGetErrorString(status));
		else
			*errstr = pstrdup("cuda random_forest training failed");
	}
	rc = -1;
	goto gpu_cleanup;
}

int
ndb_cuda_rf_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	int *class_out,
	char **errstr)
{
	const char *base;
	const NdbCudaRfModelHeader *model_hdr;
	const NdbCudaRfTreeHeader *tree_hdrs;
	const NdbCudaRfNode *nodes_base;
	size_t header_bytes;
	int *votes = NULL;
	int rc;
	int i;
	int best_class;
	int best_votes;
	int effective_dim;

	if (errstr)
		*errstr = NULL;
	if (class_out == NULL || model_data == NULL || input == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid inputs for CUDA RF predict");
		return -1;
	}

	base = VARDATA(model_data);
	model_hdr = (const NdbCudaRfModelHeader *)base;
	if (model_hdr->class_count <= 0 || model_hdr->tree_count <= 0)
	{
		if (errstr)
			*errstr = pstrdup("CUDA RF model missing metadata");
		return -1;
	}

	effective_dim = model_hdr->feature_dim > 0 ? model_hdr->feature_dim
						   : feature_dim;
	if (effective_dim <= 0)
		effective_dim = feature_dim;

	header_bytes = sizeof(NdbCudaRfModelHeader)
		+ sizeof(NdbCudaRfTreeHeader) * model_hdr->tree_count;
	tree_hdrs = (const NdbCudaRfTreeHeader *)(base
		+ sizeof(NdbCudaRfModelHeader));
	nodes_base = (const NdbCudaRfNode *)(base + header_bytes);

	votes = (int *)palloc0(sizeof(int) * model_hdr->class_count);
	rc = ndb_cuda_rf_infer(nodes_base,
		tree_hdrs,
		model_hdr->tree_count,
		input,
		effective_dim,
		model_hdr->class_count,
		votes);
	if (rc == 0)
	{
		best_class = model_hdr->majority_class;
		best_votes = -1;
		for (i = 0; i < model_hdr->class_count; i++)
		{
			if (votes[i] > best_votes)
			{
				best_votes = votes[i];
				best_class = i;
			}
		}
		*class_out = best_class;
	} else
	{
		if (errstr != NULL && *errstr == NULL)
			*errstr = pstrdup("CUDA RF inference failed");
		*class_out = model_hdr->majority_class;
	}

	if (votes)
		pfree(votes);

	return 0;
}

#endif /* NDB_GPU_CUDA */
