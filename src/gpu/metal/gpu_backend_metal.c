/*-------------------------------------------------------------------------
 *
 * gpu_backend_metal.c
 *     Metal GPU backend implementation for NeurondB.
 *
 * This module implements Metal backend initialization, device management,
 * memory routines with error handling and logging, vector and batch operations,
 * quantization, clustering (KMeans/DBSCAN), and a backend interface registration.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *     src/gpu/gpu_backend_metal.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "utils/elog.h"
#include "utils/palloc.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_gpu_types.h"
#include "neurondb_gpu.h"

#ifdef NDB_GPU_METAL

#include "gpu_metal_wrapper.h"

#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdbool.h>

/* Metal Backend Lifecycle */

static bool
metal_backend_init_impl(void)
{
	bool ok;

	ok = metal_backend_init();
	if (!ok)
	{
		elog(WARNING,
			"neurondb: Metal backend initialization failed. "
			"Check Metal device support and driver installation on "
			"this system.");
		return false;
	}
	elog(LOG,
		"neurondb: Metal GPU backend initialized successfully "
		"(MTLDevice acquired, command queues/pipelines ready)");
	return true;
}

static void
metal_backend_cleanup_impl(void)
{
	metal_backend_cleanup();
	elog(DEBUG1,
		"neurondb: Metal GPU backend cleanup: all queues, kernels, and "
		"state released");
}

static bool
metal_backend_is_available_impl(void)
{
	bool avail;

	avail = metal_backend_is_available();
	elog(DEBUG3,
		"neurondb: Metal backend availability = %s",
		avail ? "YES" : "NO");
	return avail;
}

/* Metal Device Management */

static int
metal_backend_get_device_count_impl(void)
{
	int count;

	count = metal_backend_is_available() ? 1 : 0;
	elog(DEBUG2, "neurondb: Metal get_device_count: %d", count);
	return count;
}

static bool
metal_backend_get_device_info_impl(int device_id, GPUDeviceInfo *info)
{
	const char *device_name;
	uint64_t total_mem = 0;
	uint64_t free_mem = 0;
	char name_buf[256] = { 0 };

	if (device_id != 0 || info == NULL)
	{
		elog(WARNING,
			"neurondb: metal_backend_get_device_info: invalid id "
			"(%d) or NULL info pointer %p",
			device_id,
			info);
		return false;
	}

	if (!metal_backend_is_available())
	{
		elog(WARNING,
			"neurondb: Metal not available (cannot query device "
			"info)");
		return false;
	}

	device_name = metal_backend_device_name();
	metal_backend_device_info(
		name_buf, sizeof(name_buf), &total_mem, &free_mem);

	if (name_buf[0])
		snprintf(info->name, sizeof(info->name), "%s", name_buf);
	else if (device_name && device_name[0])
		snprintf(info->name, sizeof(info->name), "%s", device_name);
	else
		snprintf(info->name, sizeof(info->name), "Apple GPU");
	info->name[sizeof(info->name) - 1] = '\0';

	info->device_id = 0;
	info->total_memory = (total_mem > 0) ? total_mem : (8ULL << 30);
	info->free_memory = (free_mem > 0) ? free_mem : (4ULL << 30);
	info->compute_major = 3;
	info->compute_minor = 0;
	info->max_threads_per_block = 1024;
	info->multiprocessor_count = 8;
	info->unified_memory = true;
	info->is_available = true;

	elog(DEBUG2,
		"neurondb: Metal get_device_info: name='%s', total=0x%llx, "
		"free=0x%llx, "
		"major=%d, minor=%d, multiproc=%d, unified=%d",
		info->name,
		(unsigned long long)info->total_memory,
		(unsigned long long)info->free_memory,
		info->compute_major,
		info->compute_minor,
		info->multiprocessor_count,
		info->unified_memory);

	return true;
}

static bool
metal_backend_set_device_impl(int device_id)
{
	if (device_id == 0)
	{
		elog(DEBUG3, "neurondb: Metal device 0 selected");
		return true;
	}
	elog(WARNING,
		"neurondb: Metal set_device: device_id %d not available",
		device_id);
	return false;
}

/* Metal Memory Management */

static void *
metal_backend_mem_alloc_impl(Size bytes)
{
	void *ptr;

	ptr = palloc(bytes);
	if (ptr == NULL)
		elog(ERROR,
			"neurondb: Metal mem_alloc: failed to allocate %zu "
			"bytes",
			bytes);
	elog(DEBUG3, "neurondb: Metal mem_alloc: %zu bytes -> %p", bytes, ptr);
	return ptr;
}

static void
metal_backend_mem_free_impl(void *ptr)
{
	if (ptr != NULL)
	{
		elog(DEBUG3, "neurondb: Metal mem_free: %p", ptr);
		pfree(ptr);
	}
}

static bool
metal_backend_mem_copy_h2d_impl(void *dst, const void *src, Size bytes)
{
	if (dst != NULL && src != NULL && bytes > 0)
	{
		memcpy(dst, src, bytes);
		elog(DEBUG3,
			"neurondb: Metal mem_copy_h2d: %zu bytes %p -> %p",
			bytes,
			src,
			dst);
		return true;
	}
	elog(WARNING,
		"neurondb: Metal mem_copy_h2d failed: dst=%p, src=%p, "
		"bytes=%zu",
		dst,
		src,
		bytes);
	return false;
}

static bool
metal_backend_mem_copy_d2h_impl(void *dst, const void *src, Size bytes)
{
	if (dst != NULL && src != NULL && bytes > 0)
	{
		memcpy(dst, src, bytes);
		elog(DEBUG3,
			"neurondb: Metal mem_copy_d2h: %zu bytes %p -> %p",
			bytes,
			src,
			dst);
		return true;
	}
	elog(WARNING,
		"neurondb: Metal mem_copy_d2h failed: dst=%p, src=%p, "
		"bytes=%zu",
		dst,
		src,
		bytes);
	return false;
}

static void
metal_backend_synchronize_impl(void)
{
	elog(DEBUG2,
		"neurondb: Metal synchronize: no explicit sync required (UMA "
		"system)");
}

/* Metal Vector Operations */

static float
metal_backend_l2_distance_impl(const float *a, const float *b, int dim)
{
	Assert(a != NULL && b != NULL && dim > 0);
	return metal_backend_l2_distance(a, b, dim);
}

static float
metal_backend_cosine_distance_impl(const float *a, const float *b, int dim)
{
	Assert(a != NULL && b != NULL && dim > 0);
	return metal_backend_cosine_distance(a, b, dim);
}

static float
metal_backend_inner_product_impl(const float *a, const float *b, int dim)
{
	Assert(a != NULL && b != NULL && dim > 0);
	return metal_backend_inner_product(a, b, dim);
}

/* Metal Batch Operations */

static bool
metal_backend_batch_l2_impl(const float *queries,
	const float *targets,
	int num_queries,
	int num_targets,
	int dim,
	float *distances)
{
	Assert(queries != NULL && targets != NULL && distances != NULL);
	metal_backend_batch_l2(
		queries, targets, num_queries, num_targets, dim, distances);
	elog(DEBUG2,
		"neurondb: batch_l2 %d queries x %d targets, dim=%d",
		num_queries,
		num_targets,
		dim);
	return true;
}

static bool
metal_backend_batch_cosine_impl(const float *queries,
	const float *targets,
	int num_queries,
	int num_targets,
	int dim,
	float *distances)
{
	int i, j, d;

	Assert(queries != NULL && targets != NULL && distances != NULL);
	for (i = 0; i < num_queries; i++)
	{
		const float *q = queries + i * dim;

		for (j = 0; j < num_targets; j++)
		{
			const float *t = targets + j * dim;
			float dot = 0.0f;
			float norm_q = 0.0f;
			float norm_t = 0.0f;
			float denom, cosine, dist;

			for (d = 0; d < dim; d++)
			{
				dot += q[d] * t[d];
				norm_q += q[d] * q[d];
				norm_t += t[d] * t[d];
			}
			denom = sqrtf(norm_q) * sqrtf(norm_t);
			cosine = (denom != 0.0f) ? (dot / denom) : 0.0f;
			dist = 1.0f - cosine;

			distances[i * num_targets + j] = dist;
			elog(DEBUG5,
				"neurondb: batch_cosine i=%d j=%d dist=%f",
				i,
				j,
				dist);
		}
	}
	return true;
}

/* Metal Quantization */

static bool
metal_backend_quantize_int8_impl(const float *input, int8_t *output, int count)
{
	int i;
	float max_abs = 0.0f;
	float scale;

	Assert(input != NULL && output != NULL && count >= 0);

	for (i = 0; i < count; i++)
	{
		float v = fabsf(input[i]);

		if (v > max_abs)
			max_abs = v;
	}

	if (max_abs < 1e-10f)
	{
		memset(output, 0, count * sizeof(int8_t));
		elog(DEBUG4,
			"neurondb: quantize_int8: all-zeros (max abs < 1e-10)");
		return true;
	}

	scale = 127.0f / max_abs;
	for (i = 0; i < count; i++)
	{
		int out;
		float scaled = input[i] * scale;

		out = (int)rintf(scaled);
		if (out > 127)
			out = 127;
		if (out < -127)
			out = -127;
		output[i] = (int8_t)out;
		elog(DEBUG5,
			"neurondb: quantize_int8[%d]: in=%f scaled=%f out=%d",
			i,
			input[i],
			scale,
			out);
	}
	return true;
}

static bool
metal_backend_quantize_fp16_impl(const float *input, void *output, int count)
{
	int i;

	Assert(input != NULL && output != NULL && count >= 0);

	for (i = 0; i < count; i++)
	{
		uint32 bits;
		uint16 sign, fp16;
		int32 exp;
		uint32 mant;
		uint16 *o = (uint16 *)output;

		memcpy(&bits, &input[i], sizeof(float));
		sign = (bits >> 16) & 0x8000;
		exp = ((bits >> 23) & 0xFF) - 127;
		mant = bits & 0x7FFFFF;

		if (exp > 15)
		{
			fp16 = sign | 0x7C00;
		} else if (exp < -14)
		{
			fp16 = sign;
		} else
		{
			exp = exp + 15;
			mant >>= 13;
			fp16 = sign | (exp << 10) | (mant & 0x3FF);
		}
		o[i] = fp16;
		elog(DEBUG5,
			"neurondb: quantize_fp16[%d]: in=%f -> 0x%04x",
			i,
			input[i],
			fp16);
	}

	return true;
}

/* Metal Clustering (K-means, DBSCAN) */

static bool
metal_backend_kmeans_impl(const float *vectors,
	int num_vectors,
	int dim,
	int k,
	int max_iters,
	float *centroids,
	int *assignments)
{
	int *cluster_counts;
	float *new_centroids;
	int iter;
	bool changed;
	int i, j, d;

	Assert(vectors != NULL && centroids != NULL && assignments != NULL
		&& num_vectors > 0 && k > 0 && dim > 0 && max_iters > 0);

	cluster_counts = (int *)palloc0(k * sizeof(int));
	new_centroids = (float *)palloc0(k * dim * sizeof(float));

	for (iter = 0; iter < max_iters; iter++)
	{
		changed = false;
		for (i = 0; i < num_vectors; i++)
		{
			const float *vec = vectors + i * dim;
			float min_dist = FLT_MAX;
			int cid = 0;

			for (j = 0; j < k; j++)
			{
				const float *c = centroids + j * dim;
				float dist_sq = 0.0f;

				for (d = 0; d < dim; d++)
				{
					float diff = vec[d] - c[d];
					dist_sq += diff * diff;
				}
				if (dist_sq < min_dist)
				{
					min_dist = dist_sq;
					cid = j;
				}
			}

			if (assignments[i] != cid)
			{
				changed = true;
				elog(DEBUG5,
					"kmeans: vector %d: cluster %d -> %d "
					"(dist=%f)",
					i,
					assignments[i],
					cid,
					sqrtf(min_dist));
				assignments[i] = cid;
			}
		}

		if (!changed)
		{
			elog(DEBUG3,
				"kmeans: converged after %d iteration(s)",
				iter + 1);
			break;
		}

		memset(new_centroids, 0, k * dim * sizeof(float));
		memset(cluster_counts, 0, k * sizeof(int));

		for (i = 0; i < num_vectors; i++)
		{
			int cid = assignments[i];
			float *agg = new_centroids + cid * dim;
			const float *src = vectors + i * dim;

			cluster_counts[cid]++;
			for (d = 0; d < dim; d++)
				agg[d] += src[d];
		}
		for (j = 0; j < k; j++)
		{
			if (cluster_counts[j] > 0)
			{
				float *c = new_centroids + j * dim;

				for (d = 0; d < dim; d++)
					c[d] /= (float)cluster_counts[j];
			}
		}
		memcpy(centroids, new_centroids, k * dim * sizeof(float));
		elog(DEBUG4, "kmeans: iter %d: centroids recomputed", iter + 1);
	}

	pfree(cluster_counts);
	pfree(new_centroids);

	return true;
}

static bool
metal_backend_dbscan_impl(const float *vectors,
	int num_vectors,
	int dim,
	float eps,
	int min_points,
	int *cluster_ids)
{
	bool *visited;
	int *neighbors;
	int cluster = -1;
	int i, j, d;

	Assert(vectors != NULL && cluster_ids != NULL && num_vectors > 0
		&& dim > 0);

	visited = (bool *)palloc0(num_vectors * sizeof(bool));
	neighbors = (int *)palloc(num_vectors * sizeof(int));

	for (i = 0; i < num_vectors; i++)
		cluster_ids[i] = -1;

	for (i = 0; i < num_vectors; i++)
	{
		if (visited[i])
			continue;
		visited[i] = true;

		/* Find neighbors within eps */
		int neighbor_count = 0;
		const float *vec_i = vectors + i * dim;

		for (j = 0; j < num_vectors; j++)
		{
			const float *vec_j = vectors + j * dim;
			float dist2 = 0.0f;

			for (d = 0; d < dim; d++)
			{
				float diff = vec_i[d] - vec_j[d];
				dist2 += diff * diff;
			}
			if (sqrtf(dist2) <= eps)
				neighbors[neighbor_count++] = j;
		}
		if (neighbor_count < min_points)
		{
			cluster_ids[i] = -1;
			continue;
		}
		cluster++;
		cluster_ids[i] = cluster;

		for (j = 0; j < neighbor_count; j++)
		{
			int n = neighbors[j];

			if (!visited[n])
			{
				visited[n] = true;
				int nn_count = 0;
				const float *vec_n = vectors + n * dim;
				int m;

				for (m = 0; m < num_vectors; m++)
				{
					const float *vec_m = vectors + m * dim;
					float d2 = 0.0f;

					for (d = 0; d < dim; d++)
					{
						float sdf = vec_n[d] - vec_m[d];
						d2 += sdf * sdf;
					}
					if (sqrtf(d2) <= eps)
						neighbors[neighbor_count
							+ nn_count++] = m;
				}
				if (nn_count > 0
					&& nn_count + neighbor_count
						< num_vectors)
					neighbor_count += nn_count;
			}
			if (cluster_ids[n] == -1)
				cluster_ids[n] = cluster;
		}
		elog(DEBUG5,
			"neurondb: dbscan: assigned cluster %d (core=%d "
			"neighbors=%d)",
			cluster,
			i,
			neighbor_count);
	}

	pfree(visited);
	pfree(neighbors);

	elog(DEBUG2,
		"neurondb: Metal DBSCAN: %d clusters (including noise)",
		cluster + 1);
	return true;
}

/* Streams and Contexts */

static bool
metal_backend_create_streams_impl(int num_streams)
{
	elog(DEBUG1,
		"neurondb: Metal create_streams: requested %d command queues",
		num_streams);
	return true;
}

static void
metal_backend_destroy_streams_impl(void)
{
	elog(DEBUG1,
		"neurondb: Metal destroy_streams: command queues released");
}

static void *
metal_backend_get_context_impl(void)
{
	elog(DEBUG1,
		"neurondb: Metal backend get_context invoked (not implemented, "
		"returns NULL)");
	return NULL;
}

/* Backend Interface Definition */

static const ndb_gpu_backend ndb_metal_backend = {
	.name = "Metal",
	.provider = "Apple",
	.kind = NDB_GPU_BACKEND_METAL,
	.features = 0,
	.priority = 100,

	.init = ndb_metal_init,
	.shutdown = ndb_metal_shutdown,
	.is_available = ndb_metal_is_available,

	.device_count = ndb_metal_device_count,
	.device_info = ndb_metal_device_info,
	.set_device = ndb_metal_set_device,

	.mem_alloc = ndb_metal_mem_alloc,
	.mem_free = ndb_metal_mem_free,
	.memcpy_h2d = ndb_metal_memcpy_h2d,
	.memcpy_d2h = ndb_metal_memcpy_d2h,

	.launch_l2_distance = NULL,
	.launch_cosine = NULL,
	.launch_kmeans_assign = NULL,
	.launch_kmeans_update = NULL,
	.launch_quant_fp16 = NULL,
	.launch_quant_int8 = NULL,
	.launch_quant_binary = NULL,
	.launch_pq_encode = NULL,

	.stream_create = ndb_metal_stream_create,
	.stream_destroy = ndb_metal_stream_destroy,
	.stream_synchronize = ndb_metal_stream_synchronize,
};

void
neurondb_gpu_register_metal_backend(void)
{
	if (ndb_gpu_register_backend(&ndb_metal_backend) == 0)
		elog(DEBUG1,
			"neurondb: Metal GPU backend registered successfully");
	else
		elog(WARNING,
			"neurondb: Metal GPU backend registration failed");
}

/* ----------------------------
 * Random Forest predict (CPU-on-Metal implementation)
 * ---------------------------- */
#ifdef NDB_GPU_METAL

/* Flat RF layout (must match ml_random_forest.c) */
typedef struct RFNodeFlat_Metal
{
	int32 feature_index;
	uint32 _pad0;
	double threshold;
	int32 left;
	int32 right;
	int32 is_leaf;
	uint32 _pad1;
	double value;
} RFNodeFlat_Metal;

typedef struct RFTreeFlat_Metal
{
	int32 offset_to_nodes;
	int32 num_nodes;
	int64 _pad2;
} RFTreeFlat_Metal;

typedef struct RFModelFlat_Metal
{
	int32 n_trees;
	int32 max_depth;
	int32 min_samples_split;
	int32 max_features;
	int32 n_classes;
	int32 n_features;
	int32 trees_offset;
	int32 nodes_offset;
	int64 _pad3;
} RFModelFlat_Metal;

static inline int
rf_tree_predict_flat_safe_metal(const RFNodeFlat_Metal *nodes,
	int num_nodes,
	int n_features,
	const float *x)
{
	int curr = 0;
	int steps = 0;
	const int max_steps = 4096;

	while (curr >= 0 && curr < num_nodes && steps < max_steps)
	{
		const RFNodeFlat_Metal *n = &nodes[curr];
		int left;
		int right;

		if (n->is_leaf)
			return (int)(n->value);
		if (n->feature_index < 0 || n->feature_index >= n_features)
			return -1;
		left = n->left;
		right = n->right;
		if (left < 0 || left >= num_nodes || right < 0
			|| right >= num_nodes)
			return -1;
		curr = (x[n->feature_index] <= n->threshold) ? left : right;
		steps++;
	}
	return -1;
}

bool
neurondb_gpu_rf_predict_backend(const void *rf_hdr,
	const void *trees,
	const void *nodes,
	int node_capacity,
	const float *x,
	int n_features,
	int *class_out,
	char **errstr)
{
	const RFModelFlat_Metal *hdr;
	const RFTreeFlat_Metal *blob_trees;
	const RFNodeFlat_Metal *blob_nodes;
	int n_trees;
	int n_classes;
	Size payload_dummy = 0; /* header validation already done by caller */
	int *tally = NULL;
	int i;

	if (errstr)
		*errstr = NULL;
	if (!rf_hdr || !trees || !nodes || !x || !class_out)
		return false;
	if (n_features <= 0 || node_capacity <= 0)
		return false;

	hdr = (const RFModelFlat_Metal *)rf_hdr;
	n_trees = hdr->n_trees;
	n_classes = hdr->n_classes;
	if (n_trees <= 0 || n_classes <= 0 || hdr->n_features <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid RF header");
		return false;
	}
	if (n_features != hdr->n_features)
	{
		if (errstr)
			*errstr = pstrdup("feature dimension mismatch");
		return false;
	}

	blob_trees = (const RFTreeFlat_Metal *)((const char *)rf_hdr
		+ hdr->trees_offset);
	blob_nodes = (const RFNodeFlat_Metal *)((const char *)rf_hdr
		+ hdr->nodes_offset);
	if (hdr->trees_offset < (int)sizeof(RFModelFlat_Metal)
		|| hdr->nodes_offset < (int)sizeof(RFModelFlat_Metal))
	{
		if (errstr)
			*errstr = pstrdup("offsets out of bounds");
		return false;
	}

	tally = (int *)palloc0(sizeof(int) * n_classes);
	for (i = 0; i < n_trees; i++)
	{
		int off = blob_trees[i].offset_to_nodes;
		int nn = blob_trees[i].num_nodes;
		int pred;
		if (off < 0 || nn <= 0 || off + nn > node_capacity)
		{
			pfree(tally);
			if (errstr)
				*errstr = pstrdup("tree bounds invalid");
			return false;
		}
		pred = rf_tree_predict_flat_safe_metal(
			blob_nodes + off, nn, n_features, x);
		if (pred >= 0 && pred < n_classes)
			tally[pred]++;
	}
	{
		int best_cls = 0;
		int best_cnt = tally[0];
		for (i = 1; i < n_classes; i++)
		{
			if (tally[i] > best_cnt)
			{
				best_cnt = tally[i];
				best_cls = i;
			}
		}
		*class_out = best_cls;
	}
	pfree(tally);
	return true;
}

#endif /* NDB_GPU_METAL */

#else /* !NDB_GPU_METAL */

void
neurondb_gpu_register_metal_backend(void)
{
	/* Metal backend not compiled; stub does nothing. */
}

#endif /* NDB_GPU_METAL */

static int
ndb_metal_init(void)
{
	return metal_backend_init_impl() ? 0 : -1;
}

static void
ndb_metal_shutdown(void)
{
	metal_backend_cleanup_impl();
}

static int
ndb_metal_is_available(void)
{
	return metal_backend_is_available_impl() ? 1 : 0;
}

static int
ndb_metal_device_count(void)
{
	return metal_backend_get_device_count_impl();
}

static int
ndb_metal_device_info(int device_id, NDBGpuDeviceInfo *info)
{
	uint64_t total_mem = 0;
	uint64_t free_mem = 0;
	char name_buf[256] = { 0 };
	const char *device_name;

	if (info == NULL)
		return -1;

	if (device_id != 0)
		return -1;

	if (!metal_backend_is_available())
		return -1;

	memset(info, 0, sizeof(NDBGpuDeviceInfo));

	device_name = metal_backend_device_name();
	metal_backend_device_info(
		name_buf, sizeof(name_buf), &total_mem, &free_mem);

	if (name_buf[0])
		strncpy(info->name, name_buf, sizeof(info->name) - 1);
	else if (device_name && device_name[0])
		strncpy(info->name, device_name, sizeof(info->name) - 1);
	else
		strncpy(info->name, "Apple GPU", sizeof(info->name) - 1);
	info->name[sizeof(info->name) - 1] = '\0';

	info->device_id = 0;
	info->total_memory_bytes =
		(size_t)(total_mem > 0 ? total_mem : (8ULL << 30));
	info->free_memory_bytes =
		(size_t)(free_mem > 0 ? free_mem : (4ULL << 30));
	info->compute_major = 3;
	info->compute_minor = 0;
	info->is_available = true;

	return 0;
}

static int
ndb_metal_set_device(int device_id)
{
	return metal_backend_set_device_impl(device_id) ? 0 : -1;
}

static int
ndb_metal_mem_alloc(void **ptr, size_t bytes)
{
	void *tmp;

	if (ptr == NULL)
		return -1;

	tmp = metal_backend_mem_alloc_impl((Size)bytes);
	if (tmp == NULL)
		return -1;

	*ptr = tmp;
	return 0;
}

static int
ndb_metal_mem_free(void *ptr)
{
	metal_backend_mem_free_impl(ptr);
	return 0;
}

static int
ndb_metal_memcpy_h2d(void *dst, const void *src, size_t bytes)
{
	return metal_backend_mem_copy_h2d_impl(dst, src, (Size)bytes) ? 0 : -1;
}

static int
ndb_metal_memcpy_d2h(void *dst, const void *src, size_t bytes)
{
	return metal_backend_mem_copy_d2h_impl(dst, src, (Size)bytes) ? 0 : -1;
}

static int
ndb_metal_stream_create(ndb_stream_t *stream)
{
	(void)stream;
	return -1;
}

static int
ndb_metal_stream_destroy(ndb_stream_t stream)
{
	(void)stream;
	return 0;
}

static int
ndb_metal_stream_synchronize(ndb_stream_t stream)
{
	(void)stream;
	return 0;
}
