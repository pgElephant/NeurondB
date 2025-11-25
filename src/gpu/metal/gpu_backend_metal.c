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
#include "utils/jsonb.h"
#include "utils/builtins.h"
#include "fmgr.h"
#include "utils/numeric.h"
#include "lib/stringinfo.h"
#include "utils/memutils.h"
#include "miscadmin.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_gpu_types.h"
#include "neurondb_gpu.h"
#include "ml_gpu_random_forest.h"
#include "ml_gpu_logistic_regression.h"
#include "ml_gpu_linear_regression.h"
#include "ml_gpu_svm.h"
#include "ml_random_forest_internal.h"
#include "ml_random_forest_shared.h"
#include "ml_logistic_regression_internal.h"
#include "common/pg_prng.h"
#include "ml_linear_regression_internal.h"
#include "ml_svm_internal.h"
#include "ml_decision_tree_internal.h"
#include "ml_ridge_regression_internal.h"
#include "ml_lasso_regression_internal.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#ifdef HAVE_ONNX_RUNTIME
#include "neurondb_onnx.h"
#include "neurondb_llm.h"
#endif

#ifdef NDB_GPU_METAL

#include "gpu_metal_wrapper.h"

#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdbool.h>

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

/* Forward declarations for all backend functions */
static int ndb_metal_init(void);
static void ndb_metal_shutdown(void);
static int ndb_metal_is_available(void);
static int ndb_metal_device_count(void);
static int ndb_metal_device_info(int device_id, NDBGpuDeviceInfo *info);
static int ndb_metal_set_device(int device_id);
static int ndb_metal_mem_alloc(void **ptr, size_t bytes);
static int ndb_metal_mem_free(void *ptr);
static int ndb_metal_memcpy_h2d(void *dst, const void *src, size_t bytes);
static int ndb_metal_memcpy_d2h(void *dst, const void *src, size_t bytes);
static int ndb_metal_launch_l2_distance(const float *A, const float *B, float *out, int n, int d, ndb_stream_t stream);
static int ndb_metal_launch_cosine(const float *A, const float *B, float *out, int n, int d, ndb_stream_t stream);
static int ndb_metal_launch_kmeans_assign(const float *X, const float *C, int *idx, int n, int d, int k, ndb_stream_t stream);
static int ndb_metal_launch_kmeans_update(const float *X, const int *idx, float *C, int n, int d, int k, ndb_stream_t stream);
static int ndb_metal_launch_quant_fp16(const float *in, void *out, int n, ndb_stream_t stream);
static int ndb_metal_launch_quant_int8(const float *in, int8_t *out, int n, float scale, ndb_stream_t stream);
static int ndb_metal_launch_quant_int4(const float *in, unsigned char *out, int n, float scale, ndb_stream_t stream);
static int ndb_metal_launch_quant_fp8_e4m3(const float *in, unsigned char *out, int n, ndb_stream_t stream);
static int ndb_metal_launch_quant_fp8_e5m2(const float *in, unsigned char *out, int n, ndb_stream_t stream);
static int ndb_metal_launch_quant_binary(const float *in, uint8_t *out, int n, ndb_stream_t stream);
static int ndb_metal_launch_pq_encode(const float *X, const float *codebooks, uint8_t *codes, int n, int d, int m, int ks, ndb_stream_t stream);
static int ndb_metal_rf_train(const float *features, const double *labels, int n_samples, int feature_dim, int class_count, const Jsonb *hyperparams, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_rf_predict(const bytea *model_data, const float *input, int feature_dim, int *class_out, char **errstr);
static int ndb_metal_rf_pack(const struct RFModel *model, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_lr_train(const float *features, const double *labels, int n_samples, int feature_dim, const Jsonb *hyperparams, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_lr_predict(const bytea *model_data, const float *input, int feature_dim, double *probability_out, char **errstr);
static int ndb_metal_lr_pack(const struct LRModel *model, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_linreg_train(const float *features, const double *targets, int n_samples, int feature_dim, const Jsonb *hyperparams, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_linreg_predict(const bytea *model_data, const float *input, int feature_dim, double *prediction_out, char **errstr);
static int ndb_metal_linreg_pack(const struct LinRegModel *model, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_svm_train(const float *features, const double *labels, int n_samples, int feature_dim, const Jsonb *hyperparams, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_svm_predict(const bytea *model_data, const float *input, int feature_dim, int *class_out, double *confidence_out, char **errstr);
static int ndb_metal_svm_pack(const struct SVMModel *model, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_dt_train(const float *features, const double *labels, int n_samples, int feature_dim, const Jsonb *hyperparams, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_dt_predict(const bytea *model_data, const float *input, int feature_dim, double *prediction_out, char **errstr);
static int ndb_metal_dt_pack(const struct DTModel *model, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_ridge_train(const float *features, const double *targets, int n_samples, int feature_dim, const Jsonb *hyperparams, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_ridge_predict(const bytea *model_data, const float *input, int feature_dim, double *prediction_out, char **errstr);
static int ndb_metal_ridge_pack(const struct RidgeModel *model, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_lasso_train(const float *features, const double *targets, int n_samples, int feature_dim, const Jsonb *hyperparams, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_lasso_predict(const bytea *model_data, const float *input, int feature_dim, double *prediction_out, char **errstr);
static int ndb_metal_lasso_pack(const struct LassoModel *model, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_nb_train(const float *features, const double *labels, int n_samples, int feature_dim, int class_count, const Jsonb *hyperparams, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_nb_predict(const bytea *model_data, const float *input, int feature_dim, int *class_out, double *probability_out, char **errstr);
static int ndb_metal_nb_pack(const struct GaussianNBModel *model, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_gmm_train(const float *features, int n_samples, int feature_dim, int n_components, const Jsonb *hyperparams, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_gmm_predict(const bytea *model_data, const float *input, int feature_dim, int *cluster_out, double *probability_out, char **errstr);
static int ndb_metal_gmm_pack(const struct GMMModel *model, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_knn_train(const float *features, const double *labels, int n_samples, int feature_dim, int k, int task_type, const Jsonb *hyperparams, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_knn_predict(const bytea *model_data, const float *input, int feature_dim, double *prediction_out, char **errstr);
static int ndb_metal_knn_pack(const struct KNNModel *model, bytea **model_data, Jsonb **metrics, char **errstr);
static int ndb_metal_hf_embed(const char *model_name, const char *text, float **vec_out, int *dim_out, char **errstr);
static int ndb_metal_hf_complete(const char *model_name, const char *prompt, const char *params_json, char **text_out, char **errstr);
static int ndb_metal_hf_rerank(const char *model_name, const char *query, const char **docs, int ndocs, float **scores_out, char **errstr);
static int ndb_metal_stream_create(ndb_stream_t *stream);
static int ndb_metal_stream_destroy(ndb_stream_t stream);
static int ndb_metal_stream_synchronize(ndb_stream_t stream);

/* Weak symbol for RF prediction - forward declaration */
bool neurondb_gpu_rf_predict_backend(const void *rf_hdr,
	const void *trees,
	const void *nodes,
	int node_capacity,
	const float *x,
	int n_features,
	int *class_out,
	char **errstr);

/* Metal Backend Lifecycle */

static bool
metal_backend_init_impl(void)
{
	bool ok;
	const char *device_name;

	/* Expanded diagnostics for initialization path */
	const char *arch = NULL;
	const char *deploy = NULL;
	arch =
	#if defined(__aarch64__)
		"arm64";
	#elif defined(__x86_64__)
		"x86_64";
	#else
		"unknown";
	#endif
	deploy = getenv("MACOSX_DEPLOYMENT_TARGET");
	if (!deploy)
		deploy = "(unset)";
	elog(LOG,
		"neurondb: Attempting Metal backend initialization (arch=%s, DEPLOYMENT_TARGET=%s)",
		arch,
		deploy);
	elog(DEBUG1, "neurondb: Metal pre-check available=%s name='%s'", metal_backend_is_available() ? "YES" : "NO", metal_backend_device_name());
	
	/* Try to initialize Metal */
	ok = metal_backend_init();
	if (!ok)
	{
		elog(DEBUG1, "neurondb: metal_backend_init() returned false; re-checking availability");
		/* Initialization failed - try to get diagnostic info */
		device_name = metal_backend_device_name();
		elog(DEBUG1,
			"neurondb: Metal device_name after failure='%s' available=%s", device_name ? device_name : "(null)", metal_backend_is_available() ? "YES" : "NO");
		
		/* Check if Metal is at least detectable (is_available checks this) */
		if (metal_backend_is_available())
		{
			elog(WARNING,
				"neurondb: Metal backend initialization failed "
				"but device is detectable. "
				"This may indicate a Metal framework issue in PostgreSQL context "
				"(device creation or command queue creation failed). "
				"Falling back to CPU.");
		} else
		{
			elog(WARNING,
				"neurondb: Metal backend initialization failed "
				"(MTLCreateSystemDefaultDevice() returned nil). "
				"Metal is not available in this process context. "
				"Falling back to CPU.");
		}
		return false;
	}
	
	device_name = metal_backend_device_name();
	elog(DEBUG1, "neurondb: Metal init succeeded device_name='%s'", device_name ? device_name : "(null)");
	{
		char caps[512];
		metal_backend_get_capabilities(caps, sizeof(caps));
		elog(DEBUG1, "neurondb: Metal capabilities: %s", caps);
	}
	elog(LOG,
		"neurondb: Metal GPU backend initialized successfully "
		"(device: %s, MTLDevice acquired, command queues/pipelines ready)",
		device_name ? device_name : "unknown");
	return true;
}

static void
metal_backend_cleanup_impl(void)
{
	metal_backend_cleanup();
	elog(LOG,
		"neurondb: Metal GPU backend cleanup: all queues, kernels, and "
		"state released");
}

static bool
metal_backend_is_available_impl(void)
{
	bool avail;

	avail = metal_backend_is_available();
	elog(DEBUG1,
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
	return count;
}

static bool __attribute__((unused))
metal_backend_get_device_info_impl(int device_id, GPUDeviceInfo *info)
{
	const char *device_name;
	uint64_t total_mem = 0;
	uint64_t free_mem = 0;
	char name_buf[256] = { 0 };

	if (device_id != 0 || info == NULL)
	{
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: metal_backend_get_device_info: invalid id "
					"(%d) or NULL info pointer %p",
					device_id,
					info)));
		return false;
	}

	if (!metal_backend_is_available())
	{
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: Metal not available (cannot query device "
					"info)")));
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
	info->total_memory_mb = (total_mem > 0) ? (int64)(total_mem / (1024 * 1024)) : (8ULL << 20);
	info->free_memory_mb = (free_mem > 0) ? (int64)(free_mem / (1024 * 1024)) : (4ULL << 20);
	info->compute_major = 3;
	info->compute_minor = 0;
	info->is_available = true;

	elog(DEBUG1,
		"neurondb: Metal get_device_info: name='%s', total=%lld MB, "
		"free=%lld MB, "
		"major=%d, minor=%d",
		info->name,
		(long long)info->total_memory_mb,
		(long long)info->free_memory_mb,
		info->compute_major,
		info->compute_minor);

	return true;
}

static bool
metal_backend_set_device_impl(int device_id)
{
	if (device_id == 0)
	{
		return true;
	}
	ereport(ERROR,
		(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			errmsg("neurondb: Metal set_device: device_id %d not available",
				device_id)));
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
	return ptr;
}

static void
metal_backend_mem_free_impl(void *ptr)
{
	NDB_SAFE_PFREE_AND_NULL(ptr);
}

static bool
metal_backend_mem_copy_h2d_impl(void *dst, const void *src, Size bytes)
{
	if (dst != NULL && src != NULL && bytes > 0)
	{
		memcpy(dst, src, bytes);
		elog(DEBUG2,
			"neurondb: Metal mem_copy_h2d: %zu bytes %p -> %p",
			bytes,
			src,
			dst);
		return true;
	}
	ereport(ERROR,
		(errcode(ERRCODE_INTERNAL_ERROR),
			errmsg("neurondb: Metal mem_copy_h2d failed: dst=%p, src=%p, "
				"bytes=%zu",
				dst,
				src,
				bytes)));
	return false;
}

static bool
metal_backend_mem_copy_d2h_impl(void *dst, const void *src, Size bytes)
{
	if (dst != NULL && src != NULL && bytes > 0)
	{
		memcpy(dst, src, bytes);
		elog(DEBUG2,
			"neurondb: Metal mem_copy_d2h: %zu bytes %p -> %p",
			bytes,
			src,
			dst);
		return true;
	}
	ereport(ERROR,
		(errcode(ERRCODE_INTERNAL_ERROR),
			errmsg("neurondb: Metal mem_copy_d2h failed: dst=%p, src=%p, "
				"bytes=%zu",
				dst,
				src,
				bytes)));
	return false;
}

static void __attribute__((unused))
metal_backend_synchronize_impl(void)
{
	elog(DEBUG2,
		"neurondb: Metal synchronize: no explicit sync required (UMA "
		"system)");
}

/* Metal Vector Operations */

static float __attribute__((unused))
metal_backend_l2_distance_impl(const float *a, const float *b, int dim)
{
	Assert(a != NULL && b != NULL && dim > 0);
	return metal_backend_l2_distance(a, b, dim);
}

static float __attribute__((unused))
metal_backend_cosine_distance_impl(const float *a, const float *b, int dim)
{
	Assert(a != NULL && b != NULL && dim > 0);
	return metal_backend_cosine_distance(a, b, dim);
}

static float __attribute__((unused))
metal_backend_inner_product_impl(const float *a, const float *b, int dim)
{
	Assert(a != NULL && b != NULL && dim > 0);
	return metal_backend_inner_product(a, b, dim);
}

/* Metal Batch Operations */

static bool __attribute__((unused))
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

static bool __attribute__((unused))
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
			elog(DEBUG3,
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
		elog(DEBUG2,
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
		elog(DEBUG3,
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
		elog(DEBUG3,
			"neurondb: quantize_fp16[%d]: in=%f -> 0x%04x",
			i,
			input[i],
			fp16);
	}

	return true;
}

/* Metal Clustering (K-means, DBSCAN) */

static bool __attribute__((unused))
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
				elog(DEBUG3,
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
			elog(DEBUG1,
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
	}

	NDB_SAFE_PFREE_AND_NULL(cluster_counts);
	NDB_SAFE_PFREE_AND_NULL(new_centroids);

	return true;
}

static bool __attribute__((unused))
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
	int i;

	Assert(vectors != NULL && cluster_ids != NULL && num_vectors > 0
		&& dim > 0);

	visited = (bool *)palloc0(num_vectors * sizeof(bool));
	neighbors = (int *)palloc(num_vectors * sizeof(int));

	for (i = 0; i < num_vectors; i++)
		cluster_ids[i] = -1;

	for (i = 0; i < num_vectors; i++)
	{
		int neighbor_count;
		const float *vec_i;
		int j;
		int n;
		int nn_count;
		const float *vec_n;
		int m;

		if (visited[i])
			continue;
		visited[i] = true;

		/* Find neighbors within eps */
		neighbor_count = 0;
		vec_i = vectors + i * dim;

		for (j = 0; j < num_vectors; j++)
		{
			const float *vec_j;
			float dist2;
			int d;

			vec_j = vectors + j * dim;
			dist2 = 0.0f;

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
			n = neighbors[j];

			if (!visited[n])
			{
				const float *vec_m;
				float d2;

				visited[n] = true;
				nn_count = 0;
				vec_n = vectors + n * dim;

				for (m = 0; m < num_vectors; m++)
				{
					int d;

					vec_m = vectors + m * dim;
					d2 = 0.0f;

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
		elog(DEBUG2,
			"neurondb: dbscan: assigned cluster %d (core=%d "
			"neighbors=%d)",
			cluster,
			i,
			neighbor_count);
	}

	NDB_SAFE_PFREE_AND_NULL(visited);
	NDB_SAFE_PFREE_AND_NULL(neighbors);

	elog(DEBUG1,
		"neurondb: Metal DBSCAN: %d clusters (including noise)",
		cluster + 1);
	return true;
}

/* Streams and Contexts */

static bool __attribute__((unused))
metal_backend_create_streams_impl(int num_streams)
{
	elog(DEBUG1,
		"neurondb: Metal create_streams: requested %d command queues",
		num_streams);
	return true;
}

static void __attribute__((unused))
metal_backend_destroy_streams_impl(void)
{
	elog(DEBUG1,
		"neurondb: Metal destroy_streams: command queues released");
}

/* Metal Launcher Functions */

static int
ndb_metal_launch_l2_distance(const float *A,
			     const float *B,
			     float *out,
			     int n,
			     int d,
			     ndb_stream_t stream)
{
	int i;

	(void) stream;	/* Metal uses unified memory, no explicit stream needed */

	if (A == NULL || B == NULL || out == NULL || n <= 0 || d <= 0)
		return -1;

	/* Use Metal's Accelerate framework for GPU-accelerated L2 distance */
	for (i = 0; i < n; i++)
	{
		const float *a = A + i * d;
		const float *b = B + i * d;
		out[i] = metal_backend_l2_distance(a, b, d);
	}

	return 0;
}

static int
ndb_metal_launch_cosine(const float *A,
			const float *B,
			float *out,
			int n,
			int d,
			ndb_stream_t stream)
{
	int i;

	(void) stream;

	if (A == NULL || B == NULL || out == NULL || n <= 0 || d <= 0)
		return -1;

	/* Use Metal's Accelerate framework for GPU-accelerated cosine distance */
	for (i = 0; i < n; i++)
	{
		const float *a = A + i * d;
		const float *b = B + i * d;
		out[i] = metal_backend_cosine_distance(a, b, d);
	}

	return 0;
}

static int
ndb_metal_launch_kmeans_assign(const float *vectors,
			       const float *centroids,
			       int *assignments,
			       int num_vectors,
			       int dim,
			       int k,
			       ndb_stream_t stream)
{
	int i, c;
	float min_dist;
	int best_c;

	(void) stream;

	if (vectors == NULL || centroids == NULL || assignments == NULL ||
	    num_vectors <= 0 || dim <= 0 || k <= 0)
		return -1;

	/* Assign each vector to nearest centroid using Metal-accelerated distance */
	for (i = 0; i < num_vectors; i++)
	{
		const float *vec = vectors + i * dim;
		min_dist = FLT_MAX;
		best_c = 0;

		for (c = 0; c < k; c++)
		{
			const float *cent = centroids + c * dim;
			float dist = metal_backend_l2_distance(vec, cent, dim);

			/* Defensive: Handle invalid distances */
			if (!isfinite(dist) || dist < 0.0f)
				dist = FLT_MAX;

			if (dist < min_dist)
			{
				min_dist = dist;
				best_c = c;
			}
		}

		assignments[i] = best_c;
	}

	return 0;
}

static int
ndb_metal_launch_kmeans_update(const float *vectors,
				const int *assignments,
				float *centroids,
				int num_vectors,
				int dim,
				int k,
				ndb_stream_t stream)
{
	int *counts;
	int i, j, c;

	(void) stream;

	if (vectors == NULL || assignments == NULL || centroids == NULL ||
	    num_vectors <= 0 || dim <= 0 || k <= 0)
		return -1;

	counts = (int *) palloc0(k * sizeof(int));
	memset(centroids, 0, k * dim * sizeof(float));

	/* Sum vectors by cluster */
	for (i = 0; i < num_vectors; i++)
	{
		c = assignments[i];
		if (c >= 0 && c < k)
		{
			counts[c]++;
			for (j = 0; j < dim; j++)
				centroids[c * dim + j] += vectors[i * dim + j];
		}
	}

	/* Average to get new centroids */
	for (c = 0; c < k; c++)
	{
		if (counts[c] > 0)
		{
			for (j = 0; j < dim; j++)
			{
				float val = centroids[c * dim + j] / (float) counts[c];
				/* Defensive: Ensure finite values */
				if (isfinite(val))
					centroids[c * dim + j] = val;
				else
					centroids[c * dim + j] = 0.0f;
			}
		}
		/* Note: Empty clusters (counts[c] == 0) keep zero centroids */
	}

	NDB_SAFE_PFREE_AND_NULL(counts);
	return 0;
}

static int
ndb_metal_launch_quant_fp16(const float *input,
			    void *output,
			    int count,
			    ndb_stream_t stream)
{
	(void) stream;

	if (input == NULL || output == NULL || count <= 0)
		return -1;

	return metal_backend_quantize_fp16_impl(input, output, count) ? 0 : -1;
}

static int
ndb_metal_launch_quant_int8(const float *input,
			    int8_t *output,
			    int count,
			    float scale,
			    ndb_stream_t stream)
{
	(void) stream;
	(void) scale;	/* Metal implementation uses its own scale calculation */

	if (input == NULL || output == NULL || count <= 0)
		return -1;

	return metal_backend_quantize_int8_impl(input, output, count) ? 0 : -1;
}

static int
ndb_metal_launch_quant_binary(const float *input,
			       uint8_t *output,
			       int count,
			       ndb_stream_t stream)
{
	int i;

	(void) stream;

	if (input == NULL || output == NULL || count <= 0)
		return -1;

	/* Binary quantization: 1 if >= 0, 0 otherwise */
	/* Defensive: Handle NaN/Inf by treating as 0 */
	for (i = 0; i < count; i++)
	{
		float val = input[i];
		if (!isfinite(val))
			output[i] = 0;
		else
			output[i] = (val >= 0.0f) ? 1 : 0;
	}

	return 0;
}

static int
ndb_metal_launch_quant_int4(const float *input,
	unsigned char *output,
	int count,
	float scale,
	ndb_stream_t stream)
{
	(void)stream;

	if (input == NULL || output == NULL || count <= 0)
		return -1;

	/* Use common GPU quantization function */
	neurondb_gpu_quantize_int4(input, output, count);
	return 0;
}

static int
ndb_metal_launch_quant_fp8_e4m3(const float *input,
	unsigned char *output,
	int count,
	ndb_stream_t stream)
{
	(void)stream;

	if (input == NULL || output == NULL || count <= 0)
		return -1;

	/* Use common GPU quantization function */
	neurondb_gpu_quantize_fp8_e4m3(input, output, count);
	return 0;
}

static int
ndb_metal_launch_quant_fp8_e5m2(const float *input,
	unsigned char *output,
	int count,
	ndb_stream_t stream)
{
	(void)stream;

	if (input == NULL || output == NULL || count <= 0)
		return -1;

	/* Use common GPU quantization function */
	neurondb_gpu_quantize_fp8_e5m2(input, output, count);
	return 0;
}

static int
ndb_metal_launch_pq_encode(const float *vectors,
			   const float *codebooks,
			   uint8_t *codes,
			   int nvec,
			   int dim,
			   int m,
			   int ks,
			   ndb_stream_t stream)
{
	int i, j, subdim;
	int subvec_idx;
	float min_dist;
	int best_code;

	(void) stream;

	if (vectors == NULL || codebooks == NULL || codes == NULL ||
	    nvec <= 0 || dim <= 0 || m <= 0 || ks <= 0)
		return -1;

	/* Defensive: Validate dimension divisibility */
	if (dim % m != 0)
	{
		elog(WARNING, "Metal PQ encode: dimension %d must be divisible by m %d", dim, m);
		return -1;
	}

	subdim = dim / m;

	/* Product quantization encoding */
	for (i = 0; i < nvec; i++)
	{
		const float *vec = vectors + i * dim;

		for (j = 0; j < m; j++)
		{
			const float *subvec = vec + j * subdim;
			const float *codebook = codebooks + j * ks * subdim;
			min_dist = FLT_MAX;
			best_code = 0;

			/* Find nearest codeword in codebook */
			for (subvec_idx = 0; subvec_idx < ks; subvec_idx++)
			{
				const float *codeword = codebook + subvec_idx * subdim;
				float dist = metal_backend_l2_distance(subvec, codeword, subdim);

				/* Defensive: Handle invalid distances */
				if (!isfinite(dist) || dist < 0.0f)
					dist = FLT_MAX;

				if (dist < min_dist)
				{
					min_dist = dist;
					best_code = subvec_idx;
				}
			}

			/* Defensive: Ensure best_code is within valid range */
			if (best_code < 0 || best_code >= ks)
				best_code = 0;

			codes[i * m + j] = (uint8_t) best_code;
		}
	}

	return 0;
}

/* Metal ML Training Functions (CPU fallback using existing implementations) */

/* Metal RF model structures (matches CUDA format) - defined early for use in training */
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
	int max_feature_index;
} NdbCudaRfTreeHeader;

/* Metal LR model header (matches CUDA format) */
typedef struct NdbCudaLrModelHeader
{
	int feature_dim;
	int n_samples;
	int max_iters;
	double learning_rate;
	double lambda;
	double bias;
} NdbCudaLrModelHeader;

/* Helper: Fill a single-node tree (leaf) */
static void
rf_fill_single_node_tree_metal(NdbCudaRfNode *node, int majority_class)
{
	node->feature_idx = -1;
	node->threshold = 0.0f;
	node->left_child = -1;
	node->right_child = -1;
	node->value = (float)majority_class;
}

/* Helper: Compute feature statistics (mean, variance) */
static void
rf_compute_feature_stats_metal(const float *features,
	int n_samples,
	int feature_dim,
	int feature_idx,
	double *sum_out,
	double *sumsq_out)
{
	double sum = 0.0;
	double sumsq = 0.0;
	int i;

	for (i = 0; i < n_samples; i++)
	{
		float val = features[i * feature_dim + feature_idx];
		if (isfinite(val))
		{
			sum += (double)val;
			sumsq += (double)val * (double)val;
		}
	}

	*sum_out = sum;
	*sumsq_out = sumsq;
}

/* Helper: Compute split counts for left/right branches */
static void
rf_compute_split_counts_metal(const float *features,
	const int *labels,
	int n_samples,
	int feature_dim,
	int feature_idx,
	float threshold,
	int class_count,
	int *left_counts,
	int *right_counts)
{
	int i;

	memset(left_counts, 0, sizeof(int) * class_count);
	memset(right_counts, 0, sizeof(int) * class_count);

	for (i = 0; i < n_samples; i++)
	{
		float val = features[i * feature_dim + feature_idx];
		int label = labels[i];

		if (label < 0 || label >= class_count)
			continue;

		if (isfinite(val) && (double)val <= (double)threshold)
			left_counts[label]++;
		else
			right_counts[label]++;
	}
}

static int
ndb_metal_rf_train(const float *features,
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
	pg_prng_state rng;
	bool seeded = false;
	double gini_accumulator = 0.0;
	size_t class_bytes;
	int i;
	int j;
	int rc = -1;
	NdbCudaRfModelHeader model_hdr;
	NdbCudaRfTreeHeader *tree_hdrs;
	NdbCudaRfNode *nodes;
	int total_nodes;
	size_t header_bytes;
	size_t payload_bytes;
	char *base;
	int majority_class = 0;
	int best_count = 0;
	double majority_fraction = 0.0;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || labels == NULL || n_samples <= 0
		|| feature_dim <= 0 || class_count <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for Metal RF train");
		return -1;
	}

	(void)hyperparams;
	if (n_trees <= 0)
		n_trees = default_n_trees;
	if (class_count > 4096)
	{
		if (errstr)
			*errstr = pstrdup("Metal RF training: class_count exceeds maximum of 4096");
		return -1;
	}

	class_bytes = sizeof(int) * (size_t)class_count;
	if (class_bytes > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("Metal RF train: allocation size exceeds MaxAllocSize");
		return -1;
	}

	label_ints = (int *)palloc(sizeof(int) * (size_t)n_samples);
	class_counts = (int *)palloc0(class_bytes);
	tmp_left_counts = (int *)palloc(class_bytes);
	tmp_right_counts = (int *)palloc(class_bytes);
	best_left_counts = (int *)palloc(class_bytes);
	best_right_counts = (int *)palloc(class_bytes);

	if (label_ints == NULL || class_counts == NULL ||
		tmp_left_counts == NULL || tmp_right_counts == NULL ||
		best_left_counts == NULL || best_right_counts == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal RF train: palloc failed");
		goto cleanup;
	}

	for (i = 0; i < n_samples; i++)
	{
		double val = labels[i];
		label_ints[i] = (int)rint(val);
		if (label_ints[i] < 0 || label_ints[i] >= class_count)
			label_ints[i] = 0;
	}

	for (i = 0; i < n_samples; i++)
		class_counts[label_ints[i]]++;

	if (!seeded)
	{
		if (!pg_prng_strong_seed(&rng))
			pg_prng_seed(&rng, (uint64)n_samples ^ (uint64)feature_dim);
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

	total_nodes = n_trees * 3;
	header_bytes = sizeof(NdbCudaRfModelHeader)
		+ sizeof(NdbCudaRfTreeHeader) * n_trees;
	payload_bytes = header_bytes + sizeof(NdbCudaRfNode) * total_nodes;

	if (payload_bytes > MaxAllocSize || VARHDRSZ + payload_bytes > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("Metal RF train: payload size exceeds MaxAllocSize");
		goto cleanup;
	}

	payload = (bytea *)palloc(VARHDRSZ + payload_bytes);
	if (payload == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal RF train: palloc failed for payload");
		goto cleanup;
	}
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

			rf_compute_feature_stats_metal(features, n_samples,
				feature_dim, j, &sum_host, &sumsq_host);

			if (sum_host == 0.0 && sumsq_host == 0.0)
				continue;

			threshold = (float)(sum_host / (double)n_samples);
			variance = (sumsq_host / (double)n_samples)
				- ((double)threshold * (double)threshold);
			if (variance < 0.0)
				variance = 0.0;
			if (variance > 0.0)
				threshold += (float)(noise * sqrt(variance) * 0.25);

			rf_compute_split_counts_metal(features, label_ints,
				n_samples, feature_dim, j, threshold,
				class_count, tmp_left_counts, tmp_right_counts);

			{
				double gini = rf_split_gini(tmp_left_counts,
					tmp_right_counts, class_count,
					&left_total, &right_total, NULL, NULL);

				if (gini < best_gini && gini >= 0.0)
				{
					best_gini = gini;
					best_feature = j;
					best_threshold = threshold;
					memcpy(best_left_counts, tmp_left_counts, class_bytes);
					memcpy(best_right_counts, tmp_right_counts, class_bytes);
				}
			}
		}

		if (best_feature < 0)
		{
			tree_hdrs[i].node_count = 1;
			tree_hdrs[i].nodes_start = node_offset;
			tree_hdrs[i].root_index = 0;
			tree_hdrs[i].max_feature_index = -1;
			rf_fill_single_node_tree_metal(&nodes[node_offset], majority_class);
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
		tree_hdrs[i].max_feature_index = best_feature;

		nodes[node_offset].feature_idx = best_feature;
		nodes[node_offset].threshold = best_threshold;
		nodes[node_offset].left_child = 1;
		nodes[node_offset].right_child = 2;
		nodes[node_offset].value = (float)majority_class;

		rf_fill_single_node_tree_metal(&nodes[node_offset + 1], left_majority);
		rf_fill_single_node_tree_metal(&nodes[node_offset + 2], right_majority);

		if (best_gini > 0.0 && best_gini < DBL_MAX / 4.0)
			gini_accumulator += best_gini;
	}

	{
		RFMetricsSpec spec;

		memset(&spec, 0, sizeof(spec));
		spec.storage = "metal";
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

	*model_data = payload;
	if (metrics != NULL)
	{
		*metrics = metrics_json;
		metrics_json = NULL;
	}
	rc = 0;

cleanup:
	if (label_ints != NULL)
		NDB_SAFE_PFREE_AND_NULL(label_ints);
	if (class_counts != NULL)
		NDB_SAFE_PFREE_AND_NULL(class_counts);
	if (tmp_left_counts != NULL)
		NDB_SAFE_PFREE_AND_NULL(tmp_left_counts);
	if (tmp_right_counts != NULL)
		NDB_SAFE_PFREE_AND_NULL(tmp_right_counts);
	if (best_left_counts != NULL)
		NDB_SAFE_PFREE_AND_NULL(best_left_counts);
	if (best_right_counts != NULL)
		NDB_SAFE_PFREE_AND_NULL(best_right_counts);

	return rc;
}

static int
ndb_metal_rf_predict(const bytea *model_data,
		    const float *input,
		    int feature_dim,
		    int *class_out,
		    char **errstr)
{
	/* Use existing Metal RF prediction implementation */
	/* The neurondb_gpu_rf_predict_backend weak symbol is implemented in this file */
	extern bool neurondb_gpu_rf_predict_backend(const void *, const void *, const void *, int, const float *, int, int *, char **);
	const char *base;
	const void *rf_hdr;
	const void *trees;
	const void *nodes;
	int node_capacity;
	bool ok;

	if (!model_data || !input || !class_out || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for Metal RF prediction");
		return -1;
	}

	/* Parse bytea to extract model components (same format as CUDA) */
	base = VARDATA_ANY(model_data);
	rf_hdr = (const void *)base;
	trees = (const void *)(base + sizeof(int32) * 7);  /* Skip header fields */
	nodes = (const void *)(base + sizeof(int32) * 7 + sizeof(int32) * 2);  /* Skip trees */

	/* Estimate node capacity from bytea size */
	node_capacity = (VARSIZE_ANY_EXHDR(model_data) - sizeof(int32) * 9) / sizeof(RFNodeFlat_Metal);

	/* Call weak symbol - Metal backend provides this implementation */
	ok = neurondb_gpu_rf_predict_backend(rf_hdr, trees, nodes, node_capacity, input, feature_dim, class_out, errstr);
	return ok ? 0 : -1;
}

/* Helper: Copy tree nodes from GTree to flat Metal format */
static void
rf_copy_tree_nodes_metal(const GTree *tree, NdbCudaRfNode *dest,
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
		}
		else
		{
			dst->left_child = src->left;
			dst->right_child = src->right;
		}
		dst->value = (float)src->value;

		if (src->feature_idx >= 0 && src->feature_idx > max_idx)
			max_idx = src->feature_idx;
	}
	*node_offset += count;
	if (max_feat_idx != NULL)
		*max_feat_idx = max_idx;
}

static int
ndb_metal_rf_pack(const struct RFModel *model,
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
	RFMetricsSpec spec;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid RF model for Metal pack");
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
	}
	else if (model->tree != NULL)
	{
		tree_count = 1;
		total_nodes = model->tree->count;
	}
	else
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

	if (payload_bytes > MaxAllocSize || header_bytes > MaxAllocSize ||
		nodes_bytes > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("Metal RF pack: payload size exceeds MaxAllocSize");
		return -1;
	}
	if (VARHDRSZ + payload_bytes > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("Metal RF pack: total size exceeds MaxAllocSize");
		return -1;
	}

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
	if (blob == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal RF pack: palloc failed");
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
			rf_copy_tree_nodes_metal(tree, nodes, &node_cursor,
				&tree_hdrs[i].max_feature_index);
	}

	*model_data = blob;

	if (metrics != NULL)
	{
		memset(&spec, 0, sizeof(spec));
		spec.storage = "metal";
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

static int
ndb_metal_lr_train(const float *features,
		   const double *labels,
		   int n_samples,
		   int feature_dim,
		   const Jsonb *hyperparams,
		   bytea **model_data,
		   Jsonb **metrics,
		   char **errstr)
{
	const int default_max_iters = 1000;
	const double default_learning_rate = 0.01;
	const double default_lambda = 0.001;
	int max_iters = default_max_iters;
	double learning_rate = default_learning_rate;
	double lambda = default_lambda;
	double *weights = NULL;
	double *grad_weights = NULL;
	double *predictions = NULL;
	double bias = 0.0;
	double grad_bias = 0.0;
	Jsonb *metrics_json = NULL;
	size_t weight_bytes;
	size_t pred_bytes;
	int iter;
	int i;
	int j;
	int rc = -1;
	double final_loss = 0.0;
	int correct = 0;
	double accuracy = 0.0;

	if (errstr)
		*errstr = NULL;

	if (model_data == NULL || features == NULL || labels == NULL ||
		n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for Metal LR train");
		return -1;
	}

	(void)hyperparams;

	weight_bytes = sizeof(double) * (size_t)feature_dim;
	pred_bytes = sizeof(double) * (size_t)n_samples;

	if (weight_bytes > MaxAllocSize || pred_bytes > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("Metal LR train: allocation size exceeds MaxAllocSize");
		return -1;
	}

	weights = (double *)palloc0(weight_bytes);
	grad_weights = (double *)palloc(weight_bytes);
	predictions = (double *)palloc(pred_bytes);

	if (weights == NULL || grad_weights == NULL || predictions == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal LR train: palloc failed");
		goto cleanup;
	}

	for (iter = 0; iter < max_iters; iter++)
	{
		double loss = 0.0;

		for (i = 0; i < n_samples; i++)
		{
			double z = bias;
			const float *feat = features + (i * feature_dim);

			for (j = 0; j < feature_dim; j++)
				z += weights[j] * feat[j];

			predictions[i] = 1.0 / (1.0 + exp(-z));
		}

		grad_bias = 0.0;
		for (i = 0; i < feature_dim; i++)
			grad_weights[i] = 0.0;

		for (i = 0; i < n_samples; i++)
		{
			double error = predictions[i] - labels[i];
			const float *feat = features + (i * feature_dim);

			grad_bias += error;
			for (j = 0; j < feature_dim; j++)
				grad_weights[j] += error * feat[j];

			if (labels[i] > 0.5)
				loss -= log(fmax(1e-15, predictions[i]));
			else
				loss -= log(fmax(1e-15, 1.0 - predictions[i]));
		}

		grad_bias /= (double)n_samples;
		for (i = 0; i < feature_dim; i++)
		{
			double avg_grad = grad_weights[i] / (double)n_samples;
			double reg_term = lambda * weights[i];
			grad_weights[i] = avg_grad + reg_term;
		}

		bias -= learning_rate * grad_bias;
		for (i = 0; i < feature_dim; i++)
			weights[i] -= learning_rate * grad_weights[i];

		final_loss = loss / (double)n_samples;
	}

	for (i = 0; i < n_samples; i++)
	{
		if ((predictions[i] >= 0.5 && labels[i] > 0.5) ||
			(predictions[i] < 0.5 && labels[i] <= 0.5))
			correct++;
	}
	accuracy = (double)correct / (double)n_samples;

	{
		LRModel model;
		size_t payload_bytes;
		bytea *blob;
		char *base;
		NdbCudaLrModelHeader *hdr;
		float *weights_dest;

		memset(&model, 0, sizeof(model));
		model.n_features = feature_dim;
		model.n_samples = n_samples;
		model.max_iters = max_iters;
		model.learning_rate = learning_rate;
		model.lambda = lambda;
		model.bias = bias;
		model.weights = weights;
		model.final_loss = final_loss;
		model.accuracy = accuracy;

		payload_bytes = sizeof(NdbCudaLrModelHeader) +
			sizeof(float) * (size_t)feature_dim;

		if (VARHDRSZ + payload_bytes > MaxAllocSize)
		{
			if (errstr)
				*errstr = pstrdup("Metal LR train: model size exceeds MaxAllocSize");
			goto cleanup;
		}

		blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
		if (blob == NULL)
		{
			if (errstr)
				*errstr = pstrdup("Metal LR train: palloc failed for model");
			goto cleanup;
		}
		SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
		base = VARDATA(blob);

		hdr = (NdbCudaLrModelHeader *)base;
		hdr->feature_dim = feature_dim;
		hdr->n_samples = n_samples;
		hdr->max_iters = max_iters;
		hdr->learning_rate = learning_rate;
		hdr->lambda = lambda;
		hdr->bias = bias;

		weights_dest = (float *)(base + sizeof(NdbCudaLrModelHeader));
		for (i = 0; i < feature_dim; i++)
			weights_dest[i] = (float)weights[i];

		*model_data = blob;

		if (metrics != NULL)
		{
			StringInfoData buf;

			initStringInfo(&buf);
			appendStringInfo(&buf,
				"{\"algorithm\":\"logistic_regression\","
				"\"storage\":\"metal\","
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
			NDB_SAFE_PFREE_AND_NULL(buf.data);
			*metrics = metrics_json;
		}

		rc = 0;
	}

cleanup:
	if (weights != NULL)
		NDB_SAFE_PFREE_AND_NULL(weights);
	if (grad_weights != NULL)
		NDB_SAFE_PFREE_AND_NULL(grad_weights);
	if (predictions != NULL)
		NDB_SAFE_PFREE_AND_NULL(predictions);

	return rc;
}

static int
ndb_metal_lr_predict(const bytea *model_data,
		    const float *input,
		    int feature_dim,
		    double *probability_out,
		    char **errstr)
{
	const NdbCudaLrModelHeader *hdr;
	const float *weights;
	const bytea *detoasted;
	double z;
	int i;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || probability_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for Metal LR predict");
		return -1;
	}

	detoasted = (const bytea *)PG_DETOAST_DATUM(PointerGetDatum(model_data));

	{
		size_t expected_size = sizeof(NdbCudaLrModelHeader) +
			sizeof(float) * (size_t)feature_dim;
		size_t actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < expected_size)
		{
			if (errstr)
				*errstr = psprintf("model data too small: expected %zu bytes, got %zu",
					expected_size, actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaLrModelHeader *)VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model has %d, input has %d",
				hdr->feature_dim, feature_dim);
		return -1;
	}

	weights = (const float *)((const char *)hdr +
		sizeof(NdbCudaLrModelHeader));

	z = hdr->bias;
	for (i = 0; i < feature_dim; i++)
		z += weights[i] * input[i];

	*probability_out = 1.0 / (1.0 + exp(-z));
	return 0;
}

static int
ndb_metal_lr_pack(const struct LRModel *model,
		  bytea **model_data,
		  Jsonb **metrics,
		  char **errstr)
{
	size_t payload_bytes;
	bytea *blob;
	char *base;
	NdbCudaLrModelHeader *hdr;
	float *weights_dest;
	int i;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid LR model for Metal pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaLrModelHeader) +
		sizeof(float) * (size_t)model->n_features;

	if (VARHDRSZ + payload_bytes > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("Metal LR pack: total size exceeds MaxAllocSize");
		return -1;
	}

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
	if (blob == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal LR pack: palloc failed");
		return -1;
	}
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaLrModelHeader *)base;
	hdr->feature_dim = model->n_features;
	hdr->n_samples = model->n_samples;
	hdr->max_iters = model->max_iters;
	hdr->learning_rate = model->learning_rate;
	hdr->lambda = model->lambda;
	hdr->bias = model->bias;

	weights_dest = (float *)(base + sizeof(NdbCudaLrModelHeader));
	if (model->weights != NULL)
	{
		for (i = 0; i < model->n_features; i++)
			weights_dest[i] = (float)model->weights[i];
	}
	else
	{
		memset(weights_dest, 0, sizeof(float) * model->n_features);
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"logistic_regression\","
			"\"storage\":\"metal\","
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
		NDB_SAFE_PFREE_AND_NULL(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

/* Metal Linear Regression model header (matches CUDA) */
typedef struct NdbCudaLinRegModelHeader
{
	int32 feature_dim;
	int32 n_samples;
	float intercept;
	float *coefficients;  /* Array of feature_dim floats */
	double r_squared;
	double mse;
	double mae;
} NdbCudaLinRegModelHeader;

static int
ndb_metal_linreg_pack(const struct LinRegModel *model,
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
			*errstr = pstrdup("invalid LinReg model for Metal pack");
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
			"\"storage\":\"metal\","
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

		metrics_json = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(buf.data)));
		NDB_SAFE_PFREE_AND_NULL(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

static int
ndb_metal_linreg_train(const float *features,
		      const double *targets,
		      int n_samples,
		      int feature_dim,
		      const Jsonb *hyperparams,
		      bytea **model_data,
		      Jsonb **metrics,
		      char **errstr)
{
	double *h_XtX = NULL;
	double *h_Xty = NULL;
	double *h_XtX_inv = NULL;
	double *h_beta = NULL;
	bytea *payload = NULL;
	Jsonb *metrics_json = NULL;
	size_t XtX_bytes;
	size_t Xty_bytes;
	size_t beta_bytes;
	int dim_with_intercept;
	int i, j, k;
	int rc = -1;

	if (errstr)
		*errstr = NULL;

	if (features == NULL || targets == NULL || n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for Metal LinReg train");
		return -1;
	}

	dim_with_intercept = feature_dim + 1;

	/* Allocate host memory for matrices */
	XtX_bytes = sizeof(double) * (size_t)dim_with_intercept * (size_t)dim_with_intercept;
	Xty_bytes = sizeof(double) * (size_t)dim_with_intercept;
	beta_bytes = sizeof(double) * (size_t)dim_with_intercept;

	h_XtX = (double *)palloc0(XtX_bytes);
	h_Xty = (double *)palloc0(Xty_bytes);
	h_XtX_inv = (double *)palloc(XtX_bytes);
	h_beta = (double *)palloc(beta_bytes);

	/* Compute X'X and X'y (CPU-based) */
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
		
		NDB_SAFE_PFREE_AND_NULL(xi);
	}

	/* Invert X'X using Gauss-Jordan elimination */
	{
		double **augmented;
		int row, col, k;
		double pivot, factor;
		bool invert_success = true;
		
		/* Create augmented matrix [A | I] */
		augmented = (double **)palloc(sizeof(double *) * dim_with_intercept);
		for (row = 0; row < dim_with_intercept; row++)
		{
			augmented[row] = (double *)palloc(sizeof(double) * 2 * dim_with_intercept);
			for (col = 0; col < dim_with_intercept; col++)
			{
				augmented[row][col] = h_XtX[row * dim_with_intercept + col];
				augmented[row][col + dim_with_intercept] = (row == col) ? 1.0 : 0.0;
			}
		}
		
		/* Gauss-Jordan elimination */
		for (row = 0; row < dim_with_intercept; row++)
		{
			pivot = augmented[row][row];
			if (fabs(pivot) < 1e-10)
			{
				bool found = false;
				for (k = row + 1; k < dim_with_intercept; k++)
				{
					if (fabs(augmented[k][row]) > 1e-10)
					{
						double *temp = augmented[row];
						augmented[row] = augmented[k];
						augmented[k] = temp;
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
			
			for (k = 0; k < dim_with_intercept; k++)
			{
				if (k != row)
				{
					factor = augmented[k][row];
					for (col = 0; col < 2 * dim_with_intercept; col++)
						augmented[k][col] -= factor * augmented[row][col];
				}
			}
		}
		
		if (invert_success)
		{
			for (row = 0; row < dim_with_intercept; row++)
				for (col = 0; col < dim_with_intercept; col++)
					h_XtX_inv[row * dim_with_intercept + col] = augmented[row][col + dim_with_intercept];
		}
		
		for (row = 0; row < dim_with_intercept; row++)
			NDB_SAFE_PFREE_AND_NULL(augmented[row]);
		NDB_SAFE_PFREE_AND_NULL(augmented);
		
		if (!invert_success)
		{
			NDB_SAFE_PFREE_AND_NULL(h_XtX);
			NDB_SAFE_PFREE_AND_NULL(h_Xty);
			NDB_SAFE_PFREE_AND_NULL(h_XtX_inv);
			NDB_SAFE_PFREE_AND_NULL(h_beta);
			if (errstr)
				*errstr = pstrdup("Matrix is singular, cannot compute linear regression");
			return -1;
		}
	}

	/* Compute β = (X'X)^(-1)X'y */
	for (i = 0; i < dim_with_intercept; i++)
	{
		h_beta[i] = 0.0;
		for (j = 0; j < dim_with_intercept; j++)
			h_beta[i] += h_XtX_inv[i * dim_with_intercept + j] * h_Xty[j];
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
		model.coefficients = (double *)palloc(sizeof(double) * feature_dim);
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
		model.r_squared = 1.0 - (ss_res / ss_tot);
		model.mse = mse;
		model.mae = mae;

		/* Pack model */
		rc = ndb_metal_linreg_pack(&model, &payload, &metrics_json, errstr);
		
		NDB_SAFE_PFREE_AND_NULL(model.coefficients);
	}

	/* Cleanup */
	NDB_SAFE_PFREE_AND_NULL(h_XtX);
	NDB_SAFE_PFREE_AND_NULL(h_Xty);
	NDB_SAFE_PFREE_AND_NULL(h_XtX_inv);
	NDB_SAFE_PFREE_AND_NULL(h_beta);

	if (rc == 0 && payload != NULL)
	{
		*model_data = payload;
		if (metrics != NULL)
			*metrics = metrics_json;
		return 0;
	}

	if (payload != NULL)
		NDB_SAFE_PFREE_AND_NULL(payload);
	if (metrics_json != NULL)
		NDB_SAFE_PFREE_AND_NULL(metrics_json);

	return -1;
}

static int
ndb_metal_linreg_predict(const bytea *model_data,
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
			*errstr = pstrdup("invalid parameters for Metal LinReg predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted = (const bytea *)PG_DETOAST_DATUM(PointerGetDatum(model_data));
	
	/* Validate bytea size */
	{
		size_t expected_size = sizeof(NdbCudaLinRegModelHeader) + sizeof(float) * (size_t)feature_dim;
		size_t actual_size = VARSIZE(detoasted) - VARHDRSZ;
		
		if (actual_size < expected_size)
		{
			if (errstr)
				*errstr = psprintf("model data too small: expected %zu bytes, got %zu", 
					expected_size, actual_size);
			return -1;
		}
	}
	
	hdr = (const NdbCudaLinRegModelHeader *)VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model has %d, input has %d", 
				hdr->feature_dim, feature_dim);
		return -1;
	}

	coefficients = (const float *)((const char *)hdr + sizeof(NdbCudaLinRegModelHeader));

	/* Compute prediction: y = intercept + Σ(coef_i * x_i) */
	prediction = (double)hdr->intercept;
	for (i = 0; i < feature_dim; i++)
		prediction += (double)coefficients[i] * (double)input[i];

	*prediction_out = prediction;
	return 0;
}

/* Metal SVM model header (matches CUDA) */
typedef struct NdbCudaSvmModelHeader
{
	int32 feature_dim;
	int32 n_samples;
	int32 n_support_vectors;
	float bias;
	float *alphas;  /* Array of n_support_vectors floats */
	float *support_vectors;  /* Array of n_support_vectors * feature_dim floats */
	int32 *support_vector_indices;  /* Array of n_support_vectors ints */
	double C;
	int32 max_iters;
} NdbCudaSvmModelHeader;

/*
 * Helper: Compute linear kernel K(x, y) = x · y
 */
static float
svm_linear_kernel_metal(const float *x, const float *y, int feature_dim)
{
	float result = 0.0f;
	int i;

	for (i = 0; i < feature_dim; i++)
		result += x[i] * y[i];

	return result;
}

static int
ndb_metal_svm_pack(const struct SVMModel *model,
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
			*errstr = pstrdup("invalid SVM model for Metal pack");
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
			"\"storage\":\"metal\","
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
		NDB_SAFE_PFREE_AND_NULL(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

static int
ndb_metal_svm_train(const float *features,
		   const double *labels,
		   int n_samples,
		   int feature_dim,
		   const Jsonb *hyperparams,
		   bytea **model_data,
		   Jsonb **metrics,
		   char **errstr)
{
	double C = 1.0;
	int max_iters = 1000;
	float *alphas = NULL;
	float *errors = NULL;
	float *kernel_matrix = NULL;
	float bias = 0.0f;
	int actual_max_iters;
	int sample_limit;
	int iter;
	int num_changed = 0;
	int examine_all = 1;
	double eps = 1e-3;
	int sv_count = 0;
	SVMModel model;
	int i, j;
	int rc = -1;

	if (errstr)
		*errstr = NULL;

	/* Comprehensive input validation */
	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal SVM train: features array is NULL");
		return -1;
	}
	if (labels == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal SVM train: labels array is NULL");
		return -1;
	}
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal SVM train: model_data output pointer is NULL");
		return -1;
	}
	if (n_samples <= 0 || n_samples > 100000)
	{
		if (errstr)
			*errstr = pstrdup("Metal SVM train: n_samples must be between 1 and 100000");
		return -1;
	}
	if (feature_dim <= 0 || feature_dim > 10000)
	{
		if (errstr)
			*errstr = pstrdup("Metal SVM train: feature_dim must be between 1 and 10000");
		return -1;
	}

	/* Extract hyperparameters */
	if (hyperparams != NULL)
	{
		Datum C_datum;
		Datum max_iters_datum;
		Datum numeric_datum;
		Numeric num;

		C_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("C"));
		if (DatumGetPointer(C_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, C_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				C = DatumGetFloat8(
					DirectFunctionCall1(numeric_float8,
						NumericGetDatum(num)));
				if (C <= 0.0)
					C = 1.0;
				if (C > 1000.0)
					C = 1000.0;
			}
		}

		max_iters_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("max_iters"));
		if (DatumGetPointer(max_iters_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, max_iters_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				max_iters = DatumGetInt32(
					DirectFunctionCall1(numeric_int4,
						NumericGetDatum(num)));
				if (max_iters <= 0)
					max_iters = 1000;
				if (max_iters > 100000)
					max_iters = 100000;
			}
		}
	}

	/* Limit iterations and samples for large datasets */
	actual_max_iters = (max_iters > 1000 && n_samples > 1000) ? 1000 : max_iters;
	sample_limit = (n_samples > 5000) ? 5000 : n_samples;

	/* Validate input data for NaN/Inf */
	for (i = 0; i < n_samples && i < sample_limit; i++)
	{
		if (!isfinite(labels[i]))
		{
			if (errstr)
				*errstr = pstrdup("Metal SVM train: non-finite value in labels array");
			return -1;
		}
		for (j = 0; j < feature_dim; j++)
		{
			if (!isfinite(features[i * feature_dim + j]))
			{
				if (errstr)
					*errstr = pstrdup("Metal SVM train: non-finite value in features array");
				return -1;
			}
		}
	}

	/* Allocate memory */
	alphas = (float *)palloc0(sizeof(float) * (size_t)sample_limit);
	errors = (float *)palloc(sizeof(float) * (size_t)sample_limit);
	kernel_matrix = (float *)palloc(sizeof(float) * (size_t)sample_limit * (size_t)sample_limit);

	if (alphas == NULL || errors == NULL || kernel_matrix == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal SVM train: failed to allocate memory");
		if (alphas)
			NDB_SAFE_PFREE_AND_NULL(alphas);
		if (errors)
			NDB_SAFE_PFREE_AND_NULL(errors);
		if (kernel_matrix)
			NDB_SAFE_PFREE_AND_NULL(kernel_matrix);
		return -1;
	}

	/* Pre-compute kernel matrix (CPU-based) */
	for (i = 0; i < sample_limit; i++)
	{
		for (j = 0; j < sample_limit; j++)
		{
			kernel_matrix[i * sample_limit + j] = svm_linear_kernel_metal(
				features + (i * feature_dim),
				features + (j * feature_dim),
				feature_dim);
		}
	}

	/* Initialize errors: E_i = f(x_i) - y_i, where f(x_i) = 0 initially */
	for (i = 0; i < sample_limit; i++)
		errors[i] = -(float)labels[i];

	/* Simplified SMO: iterate until convergence or max iterations */
	for (iter = 0; iter < actual_max_iters; iter++)
	{
		num_changed = 0;

		/* Simplified update: adjust alphas based on errors */
		for (i = 0; i < sample_limit; i++)
		{
			float error_i = errors[i];
			float label_i = (float)labels[i];
			float alpha_i = alphas[i];
			float eta;
			float L = 0.0f;
			float H = (float)C;
			float new_alpha_i;
			float delta_alpha;

			/* Compute kernel for self (simplified) */
			eta = 2.0f * kernel_matrix[i * sample_limit + i] - kernel_matrix[i * sample_limit + i];

			if (eta >= 0.0f)
				continue;

			/* Update alpha */
			new_alpha_i = alpha_i - label_i * error_i / eta;

			/* Clip to bounds */
			if (new_alpha_i < L)
				new_alpha_i = L;
			if (new_alpha_i > H)
				new_alpha_i = H;

			if (fabsf(new_alpha_i - alpha_i) < (float)eps)
				continue;

			delta_alpha = new_alpha_i - alpha_i;
			alphas[i] = new_alpha_i;

			/* Update errors (CPU-based) */
			for (j = 0; j < sample_limit; j++)
			{
				float k_val = kernel_matrix[i * sample_limit + j];
				errors[j] -= delta_alpha * label_i * k_val;
			}

			num_changed++;
		}

		/* Update bias (simplified) */
		if (num_changed > 0)
		{
			float bias_sum = 0.0f;
			int bias_count = 0;
			for (i = 0; i < sample_limit; i++)
			{
				if (alphas[i] > (float)eps && alphas[i] < ((float)C - (float)eps))
				{
					float pred = 0.0f;
					for (j = 0; j < sample_limit; j++)
					{
						if (alphas[j] > (float)eps)
						{
							float k_val = kernel_matrix[j * sample_limit + i];
							pred += alphas[j] * (float)labels[j] * k_val;
						}
					}
					bias_sum += (float)labels[i] - pred;
					bias_count++;
				}
			}
			if (bias_count > 0)
				bias = bias_sum / (float)bias_count;
		}

		if (examine_all)
			examine_all = 0;
		else if (num_changed == 0)
			examine_all = 1;

		if (num_changed == 0 && !examine_all)
			break;
	}

	/* Count support vectors */
	sv_count = 0;
	for (i = 0; i < sample_limit; i++)
	{
		if (alphas[i] > (float)eps)
			sv_count++;
	}

	/* Handle case when no support vectors found */
	if (sv_count == 0)
		sv_count = 1;

	/* Build SVMModel */
	memset(&model, 0, sizeof(model));
	model.n_features = feature_dim;
	model.n_samples = n_samples;
	model.n_support_vectors = sv_count;
	model.bias = (double)bias;
	model.C = C;
	model.max_iters = actual_max_iters;

	/* Allocate support vectors and alphas */
	model.alphas = (double *)palloc(sizeof(double) * (size_t)sv_count);
	model.support_vectors = (float *)palloc(sizeof(float) * (size_t)sv_count * (size_t)feature_dim);
	model.support_vector_indices = (int *)palloc(sizeof(int) * (size_t)sv_count);

	if (model.alphas == NULL || model.support_vectors == NULL || model.support_vector_indices == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal SVM train: failed to allocate support vectors");
		if (model.alphas)
			NDB_SAFE_PFREE_AND_NULL(model.alphas);
		if (model.support_vectors)
			NDB_SAFE_PFREE_AND_NULL(model.support_vectors);
		if (model.support_vector_indices)
			NDB_SAFE_PFREE_AND_NULL(model.support_vector_indices);
		NDB_SAFE_PFREE_AND_NULL(alphas);
		NDB_SAFE_PFREE_AND_NULL(errors);
		NDB_SAFE_PFREE_AND_NULL(kernel_matrix);
		return -1;
	}

	/* Copy support vectors */
	{
		int sv_idx = 0;
		for (i = 0; i < sample_limit && sv_idx < sv_count; i++)
		{
			if (alphas[i] > (float)eps || (sv_count == 1 && sv_idx == 0))
			{
				model.alphas[sv_idx] = (double)alphas[i];
				model.support_vector_indices[sv_idx] = i;
				memcpy(model.support_vectors + sv_idx * feature_dim,
					features + i * feature_dim,
					sizeof(float) * feature_dim);
				sv_idx++;
			}
		}
		if (sv_idx == 0)
		{
			/* Fallback: use first sample */
			model.alphas[0] = 1.0;
			model.support_vector_indices[0] = 0;
			memcpy(model.support_vectors, features, sizeof(float) * feature_dim);
		}
	}

	/* Pack model */
	if (ndb_metal_svm_pack(&model, model_data, metrics, errstr) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("Metal SVM train: model packing failed");
		if (model.alphas)
			NDB_SAFE_PFREE_AND_NULL(model.alphas);
		if (model.support_vectors)
			NDB_SAFE_PFREE_AND_NULL(model.support_vectors);
		if (model.support_vector_indices)
			NDB_SAFE_PFREE_AND_NULL(model.support_vector_indices);
		NDB_SAFE_PFREE_AND_NULL(alphas);
		NDB_SAFE_PFREE_AND_NULL(errors);
		NDB_SAFE_PFREE_AND_NULL(kernel_matrix);
		return -1;
	}

	/* Cleanup */
	if (model.alphas)
		NDB_SAFE_PFREE_AND_NULL(model.alphas);
	if (model.support_vectors)
		NDB_SAFE_PFREE_AND_NULL(model.support_vectors);
	if (model.support_vector_indices)
		NDB_SAFE_PFREE_AND_NULL(model.support_vector_indices);
	NDB_SAFE_PFREE_AND_NULL(alphas);
	NDB_SAFE_PFREE_AND_NULL(errors);
	NDB_SAFE_PFREE_AND_NULL(kernel_matrix);

	rc = 0;
	return rc;
}

static int
ndb_metal_svm_predict(const bytea *model_data,
		      const float *input,
		      int feature_dim,
		      int *class_out,
		      double *confidence_out,
		      char **errstr)
{
	const NdbCudaSvmModelHeader *hdr;
	const float *alphas;
	const float *support_vectors;
	const bytea *detoasted;
	double prediction;
	int i, j;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || class_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for Metal SVM predict");
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
	/* Note: indices are stored after support_vectors but not used in linear kernel prediction */

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

/* Metal Decision Tree model structures (matches CUDA) */
typedef struct NdbCudaDtNode
{
	int		feature_idx;
	float	threshold;
	int		left_child;
	int		right_child;
	float	value;
	bool	is_leaf;
} NdbCudaDtNode;

typedef struct NdbCudaDtModelHeader
{
	int32 feature_dim;
	int32 n_samples;
	int32 max_depth;
	int32 min_samples_split;
	int32 node_count;
} NdbCudaDtModelHeader;

/*
 * Helper: Compute Gini impurity from class counts
 */
static double
dt_compute_gini_metal(const int *class_counts, int class_count, int total)
{
	double gini = 1.0;
	int i;

	if (total <= 0)
		return 0.0;

	for (i = 0; i < class_count; i++)
	{
		double p = (double)class_counts[i] / (double)total;
		gini -= p * p;
	}

	return gini;
}

/*
 * Helper: Compute variance from sum and sum of squares
 */
static double
dt_compute_variance_metal(double sum, double sumsq, int count)
{
	double mean;
	double variance;

	if (count <= 0)
		return 0.0;

	mean = sum / (double)count;
	variance = (sumsq / (double)count) - (mean * mean);

	return (variance > 0.0) ? variance : 0.0;
}

/*
 * Helper: Free tree recursively
 */
static void
dt_free_tree_metal(DTNode *node)
{
	if (node == NULL)
		return;
	if (!node->is_leaf)
	{
		dt_free_tree_metal(node->left);
		dt_free_tree_metal(node->right);
	}
	NDB_SAFE_PFREE_AND_NULL(node);
}

/*
 * Helper: Count nodes in tree recursively
 */
static int
dt_count_nodes_metal(const DTNode *node)
{
	if (node == NULL)
		return 0;
	if (node->is_leaf)
		return 1;
	return 1 + dt_count_nodes_metal(node->left) + dt_count_nodes_metal(node->right);
}

/*
 * Helper: Serialize tree nodes recursively
 */
static int
dt_serialize_node_recursive_metal(const DTNode *node, NdbCudaDtNode *dest, int *node_idx, int *max_idx)
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
		if (dt_serialize_node_recursive_metal(node->left, dest, node_idx, max_idx) < 0)
			return -1;

		(*node_idx)++;
		right_idx = *node_idx;
		if (dt_serialize_node_recursive_metal(node->right, dest, node_idx, max_idx) < 0)
			return -1;

		dest[current_idx].left_child = left_idx;
		dest[current_idx].right_child = right_idx;
	}

	return 0;
}

/*
 * Helper: Build tree node recursively (CPU-based, similar to CUDA)
 */
static DTNode *
dt_build_tree_metal(const float *features,
	const double *labels,
	const int *indices,
	int n_samples,
	int feature_dim,
	int max_depth,
	int min_samples_split,
	bool is_classification,
	int class_count,
	char **errstr)
{
	DTNode *node;
	int i;
	int best_feature = -1;
	float best_threshold = 0.0f;
	double best_gain = -DBL_MAX;
	int *left_indices = NULL;
	int *right_indices = NULL;
	int left_count = 0;
	int right_count = 0;

	if (errstr)
		*errstr = NULL;

	/* Allocate node */
	node = (DTNode *)palloc0(sizeof(DTNode));
	if (node == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal DT train: failed to allocate node");
		return NULL;
	}

	/* Stopping criteria */
	if (max_depth <= 0 || n_samples < min_samples_split)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			/* Count classes for majority vote */
			int *class_counts;
			int majority;
			int label;
			int i;

			class_counts = (int *)palloc0(sizeof(int) * class_count);
			for (i = 0; i < n_samples; i++)
			{
				label = (int)labels[indices[i]];
				if (label >= 0 && label < class_count)
					class_counts[label]++;
			}
			majority = 0;
			for (i = 1; i < class_count; i++)
			{
				if (class_counts[i] > class_counts[majority])
					majority = i;
			}
			node->leaf_value = (double)majority;
			NDB_SAFE_PFREE_AND_NULL(class_counts);
		}
		else
		{
			/* Compute mean for regression */
			double sum = 0.0;
			for (i = 0; i < n_samples; i++)
				sum += labels[indices[i]];
			node->leaf_value = (n_samples > 0) ? (sum / (double)n_samples) : 0.0;
		}
		return node;
	}

	/* Allocate working arrays */
	left_indices = (int *)palloc(sizeof(int) * n_samples);
	right_indices = (int *)palloc(sizeof(int) * n_samples);
	if (left_indices == NULL || right_indices == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal DT train: failed to allocate index arrays");
		if (left_indices)
			NDB_SAFE_PFREE_AND_NULL(left_indices);
		if (right_indices)
			NDB_SAFE_PFREE_AND_NULL(right_indices);
		NDB_SAFE_PFREE_AND_NULL(node);
		return NULL;
	}

	/* Find best split (CPU-based) */
	{
		int feat;
		float min_val;
		float max_val;
		float val;
		int thresh_idx;
		float threshold;
		double gain;
		double left_imp;
		double right_imp;
		int left_total;
		int right_total;

		for (feat = 0; feat < feature_dim; feat++)
		{
			min_val = FLT_MAX;
			max_val = -FLT_MAX;

			/* Compute feature range */
			for (i = 0; i < n_samples; i++)
			{
				val = features[indices[i] * feature_dim + feat];
				if (isfinite(val))
				{
					if (val < min_val)
						min_val = val;
					if (val > max_val)
						max_val = val;
				}
			}

			if (min_val == max_val || !isfinite(min_val) || !isfinite(max_val))
				continue;

			/* Try candidate thresholds (10 uniformly spaced) */
			for (thresh_idx = 1; thresh_idx < 10; thresh_idx++)
			{
				threshold = min_val + (max_val - min_val) * (float)thresh_idx / 10.0f;
				gain = 0.0;
				left_imp = 0.0;
				right_imp = 0.0;
				left_total = 0;
				right_total = 0;

				if (is_classification)
				{
					/* Count classes for left and right splits */
					int *left_counts;
					int *right_counts;
					int label;

					left_counts = (int *)palloc0(sizeof(int) * class_count);
					right_counts = (int *)palloc0(sizeof(int) * class_count);

					for (i = 0; i < n_samples; i++)
					{
						val = features[indices[i] * feature_dim + feat];
						label = (int)labels[indices[i]];
						if (label >= 0 && label < class_count)
						{
							if (isfinite(val) && val <= threshold)
								left_counts[label]++;
							else
								right_counts[label]++;
						}
					}

					/* Compute totals */
					for (i = 0; i < class_count; i++)
					{
						left_total += left_counts[i];
						right_total += right_counts[i];
					}

					if (left_total <= 0 || right_total <= 0)
					{
						NDB_SAFE_PFREE_AND_NULL(left_counts);
						NDB_SAFE_PFREE_AND_NULL(right_counts);
						continue;
					}

					/* Compute parent impurity from all samples */
					{
						int *parent_counts;
						double parent_imp;
						int label;
						int i;

						parent_counts = (int *)palloc0(sizeof(int) * class_count);
						for (i = 0; i < n_samples; i++)
						{
							label = (int)labels[indices[i]];
							if (label >= 0 && label < class_count)
								parent_counts[label]++;
						}
						parent_imp = dt_compute_gini_metal(parent_counts, class_count, n_samples);
						NDB_SAFE_PFREE_AND_NULL(parent_counts);

						/* Compute Gini impurity */
						left_imp = dt_compute_gini_metal(left_counts, class_count, left_total);
						right_imp = dt_compute_gini_metal(right_counts, class_count, right_total);

						/* Information gain */
						gain = parent_imp - (((double)left_total / (double)n_samples) * left_imp +
							((double)right_total / (double)n_samples) * right_imp);
					}

					NDB_SAFE_PFREE_AND_NULL(left_counts);
					NDB_SAFE_PFREE_AND_NULL(right_counts);
				}
				else
				{
					/* Compute statistics for regression */
					double left_sum;
					double left_sumsq;
					int left_count_reg;
					double right_sum;
					double right_sumsq;
					int right_count_reg;
					double label_val;

					left_sum = 0.0;
					left_sumsq = 0.0;
					left_count_reg = 0;
					right_sum = 0.0;
					right_sumsq = 0.0;
					right_count_reg = 0;

					for (i = 0; i < n_samples; i++)
					{
						val = features[indices[i] * feature_dim + feat];
						label_val = labels[indices[i]];

						if (!isfinite(val) || !isfinite(label_val))
							continue;

						if (val <= threshold)
						{
							left_sum += label_val;
							left_sumsq += label_val * label_val;
							left_count_reg++;
						}
						else
						{
							right_sum += label_val;
							right_sumsq += label_val * label_val;
							right_count_reg++;
						}
					}

					if (left_count_reg <= 0 || right_count_reg <= 0)
						continue;

					/* Compute variance */
					{
						double left_var;
						double right_var;
						double parent_var;

						left_var = dt_compute_variance_metal(left_sum, left_sumsq, left_count_reg);
						right_var = dt_compute_variance_metal(right_sum, right_sumsq, right_count_reg);

						/* Variance reduction */
						parent_var = dt_compute_variance_metal(left_sum + right_sum,
							left_sumsq + right_sumsq, left_count_reg + right_count_reg);
						gain = parent_var - (((double)left_count_reg / (double)n_samples) * left_var +
							((double)right_count_reg / (double)n_samples) * right_var);
					}
					left_total = left_count_reg;
					right_total = right_count_reg;
				}

				/* Update best split if this is better */
				if (gain > best_gain && isfinite(gain))
				{
					best_gain = gain;
					best_feature = feat;
					best_threshold = threshold;
				}
			}
		}
	}

	/* If no good split found, make leaf */
	if (best_feature < 0 || best_gain <= 0.0)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			int *class_counts;
			int majority;
			int label;
			int i;

			class_counts = (int *)palloc0(sizeof(int) * class_count);
			for (i = 0; i < n_samples; i++)
			{
				label = (int)labels[indices[i]];
				if (label >= 0 && label < class_count)
					class_counts[label]++;
			}
			majority = 0;
			for (i = 1; i < class_count; i++)
			{
				if (class_counts[i] > class_counts[majority])
					majority = i;
			}
			node->leaf_value = (double)majority;
			NDB_SAFE_PFREE_AND_NULL(class_counts);
		}
		else
		{
			double sum = 0.0;
			for (i = 0; i < n_samples; i++)
				sum += labels[indices[i]];
			node->leaf_value = (n_samples > 0) ? (sum / (double)n_samples) : 0.0;
		}
		NDB_SAFE_PFREE_AND_NULL(left_indices);
		NDB_SAFE_PFREE_AND_NULL(right_indices);
		return node;
	}

	/* Partition indices based on best split */
	{
		float val;

		for (i = 0; i < n_samples; i++)
		{
			val = features[indices[i] * feature_dim + best_feature];
			if (isfinite(val) && val <= best_threshold)
				left_indices[left_count++] = indices[i];
			else
				right_indices[right_count++] = indices[i];
		}
	}

	/* Validate split */
	if (left_count <= 0 || right_count <= 0)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			int *class_counts;
			int majority;
			int label;
			int i;

			class_counts = (int *)palloc0(sizeof(int) * class_count);
			for (i = 0; i < n_samples; i++)
			{
				label = (int)labels[indices[i]];
				if (label >= 0 && label < class_count)
					class_counts[label]++;
			}
			majority = 0;
			for (i = 1; i < class_count; i++)
			{
				if (class_counts[i] > class_counts[majority])
					majority = i;
			}
			node->leaf_value = (double)majority;
			NDB_SAFE_PFREE_AND_NULL(class_counts);
		}
		else
		{
			double sum = 0.0;
			for (i = 0; i < n_samples; i++)
				sum += labels[indices[i]];
			node->leaf_value = (n_samples > 0) ? (sum / (double)n_samples) : 0.0;
		}
		NDB_SAFE_PFREE_AND_NULL(left_indices);
		NDB_SAFE_PFREE_AND_NULL(right_indices);
		return node;
	}

	/* Build left and right subtrees recursively */
	node->is_leaf = false;
	node->feature_idx = best_feature;
	node->threshold = best_threshold;

	node->left = dt_build_tree_metal(features, labels, left_indices, left_count,
		feature_dim, max_depth - 1, min_samples_split, is_classification,
		class_count, errstr);
	if (node->left == NULL)
	{
		NDB_SAFE_PFREE_AND_NULL(left_indices);
		NDB_SAFE_PFREE_AND_NULL(right_indices);
		NDB_SAFE_PFREE_AND_NULL(node);
		return NULL;
	}

	node->right = dt_build_tree_metal(features, labels, right_indices, right_count,
		feature_dim, max_depth - 1, min_samples_split, is_classification,
		class_count, errstr);
	if (node->right == NULL)
	{
		dt_free_tree_metal(node->left);
		NDB_SAFE_PFREE_AND_NULL(left_indices);
		NDB_SAFE_PFREE_AND_NULL(right_indices);
		NDB_SAFE_PFREE_AND_NULL(node);
		return NULL;
	}

	NDB_SAFE_PFREE_AND_NULL(left_indices);
	NDB_SAFE_PFREE_AND_NULL(right_indices);
	return node;
}

static int
ndb_metal_dt_train(const float *features,
		  const double *labels,
		  int n_samples,
		  int feature_dim,
		  const Jsonb *hyperparams,
		  bytea **model_data,
		  Jsonb **metrics,
		  char **errstr)
{
	int max_depth = 10;
	int min_samples_split = 2;
	bool is_classification = true;
	int class_count = 2;
	DTModel *model = NULL;
	DTNode *root = NULL;
	int *indices = NULL;
	int i;
	int rc = -1;

	if (errstr)
		*errstr = NULL;

	/* Comprehensive input validation */
	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal DT train: features array is NULL");
		return -1;
	}
	if (labels == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal DT train: labels array is NULL");
		return -1;
	}
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal DT train: model_data output pointer is NULL");
		return -1;
	}
	if (n_samples <= 0 || n_samples > 100000000)
	{
		if (errstr)
			*errstr = pstrdup("Metal DT train: n_samples must be between 1 and 100000000");
		return -1;
	}
	if (feature_dim <= 0 || feature_dim > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("Metal DT train: feature_dim must be between 1 and 1000000");
		return -1;
	}

	/* Extract hyperparameters */
	if (hyperparams != NULL)
	{
		Datum max_depth_datum;
		Datum min_samples_split_datum;
		Datum is_classification_datum;
		Datum class_count_datum;
		Datum numeric_datum;
		Numeric num;

		max_depth_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("max_depth"));
		if (DatumGetPointer(max_depth_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, max_depth_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				max_depth = DatumGetInt32(
					DirectFunctionCall1(numeric_int4,
						NumericGetDatum(num)));
				if (max_depth <= 0)
					max_depth = 10;
				if (max_depth > 100)
					max_depth = 100;
			}
		}

		min_samples_split_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("min_samples_split"));
		if (DatumGetPointer(min_samples_split_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, min_samples_split_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				min_samples_split = DatumGetInt32(
					DirectFunctionCall1(numeric_int4,
						NumericGetDatum(num)));
				if (min_samples_split < 2)
					min_samples_split = 2;
			}
		}

		is_classification_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("is_classification"));
		if (DatumGetPointer(is_classification_datum) != NULL)
		{
			bool val = DatumGetBool(
				DirectFunctionCall1(jsonb_bool, is_classification_datum));
			is_classification = val;
		}

		class_count_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("class_count"));
		if (DatumGetPointer(class_count_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, class_count_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				class_count = DatumGetInt32(
					DirectFunctionCall1(numeric_int4,
						NumericGetDatum(num)));
				if (class_count < 2)
					class_count = 2;
				if (class_count > 1000)
					class_count = 1000;
			}
		}
	}

	/* Validate input data for NaN/Inf */
	for (i = 0; i < n_samples; i++)
	{
		if (!isfinite(labels[i]))
		{
			if (errstr)
				*errstr = pstrdup("Metal DT train: non-finite value in labels array");
			return -1;
		}
		{
			int j;

			for (j = 0; j < feature_dim; j++)
			{
				if (!isfinite(features[i * feature_dim + j]))
				{
					if (errstr)
						*errstr = pstrdup("Metal DT train: non-finite value in features array");
					return -1;
				}
			}
		}
	}

	/* Create index array */
	indices = (int *)palloc(sizeof(int) * n_samples);
	if (indices == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal DT train: failed to allocate indices array");
		return -1;
	}
	for (i = 0; i < n_samples; i++)
		indices[i] = i;

	/* Build tree using CPU-based algorithm */
	root = dt_build_tree_metal(features, labels, indices, n_samples, feature_dim,
		max_depth, min_samples_split, is_classification, class_count, errstr);
	if (root == NULL)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("Metal DT train: failed to build tree");
		NDB_SAFE_PFREE_AND_NULL(indices);
		return -1;
	}

	/* Create model structure */
	model = (DTModel *)palloc0(sizeof(DTModel));
	if (model == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Metal DT train: failed to allocate model");
		dt_free_tree_metal(root);
		NDB_SAFE_PFREE_AND_NULL(indices);
		return -1;
	}

	model->model_id = 0;  /* Will be set by catalog */
	model->n_features = feature_dim;
	model->n_samples = n_samples;
	model->max_depth = max_depth;
	model->min_samples_split = min_samples_split;
	model->root = root;

	/* Pack model */
	if (ndb_metal_dt_pack(model, model_data, metrics, errstr) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("Metal DT train: model packing failed");
		dt_free_tree_metal(root);
		NDB_SAFE_PFREE_AND_NULL(model);
		NDB_SAFE_PFREE_AND_NULL(indices);
		return -1;
	}

	NDB_SAFE_PFREE_AND_NULL(indices);
	NDB_SAFE_PFREE_AND_NULL(model);  /* Note: root is now owned by packed model */

	rc = 0;
	return rc;
}

static int
ndb_metal_dt_pack(const struct DTModel *model,
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
			*errstr = pstrdup("invalid DT model for Metal pack");
		return -1;
	}

	if (model->root == NULL)
	{
		if (errstr)
			*errstr = pstrdup("decision_tree model has no root node");
		return -1;
	}

	node_count = dt_count_nodes_metal(model->root);
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
	if (dt_serialize_node_recursive_metal(model->root, nodes, &node_idx, &node_count) < 0)
	{
		NDB_SAFE_PFREE_AND_NULL(blob);
		if (errstr)
			*errstr = pstrdup("failed to serialize decision tree nodes");
		return -1;
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"decision_tree\","
			"\"storage\":\"metal\","
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

		metrics_json = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(buf.data)));
		NDB_SAFE_PFREE_AND_NULL(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

static int
ndb_metal_dt_predict(const bytea *model_data,
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
			*errstr = pstrdup("invalid parameters for Metal DT predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted = (const bytea *)PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t min_size = sizeof(NdbCudaDtModelHeader);
		size_t actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < min_size)
		{
			if (errstr)
				*errstr = psprintf("model data too small: expected at least %zu bytes, got %zu",
					min_size, actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaDtModelHeader *)VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model has %d, input has %d",
				hdr->feature_dim, feature_dim);
		return -1;
	}

	nodes = (const NdbCudaDtNode *)((const char *)hdr + sizeof(NdbCudaDtModelHeader));

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
		*errstr = pstrdup("tree traversal failed - invalid tree structure");
	return -1;
}

/* Metal Ridge Regression model header (matches CUDA) */
typedef struct NdbCudaRidgeModelHeader
{
	int32 feature_dim;
	int32 n_samples;
	float intercept;
	float *coefficients;  /* Array of feature_dim floats */
	double lambda;
	double r_squared;
	double mse;
	double mae;
} NdbCudaRidgeModelHeader;

static int
ndb_metal_ridge_pack(const struct RidgeModel *model,
		     bytea **model_data,
		     Jsonb **metrics,
		     char **errstr)
{
	size_t payload_bytes;
	bytea *blob;
	char *base;
	NdbCudaRidgeModelHeader *hdr;
	float *coef_dest;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid Ridge model for Metal pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaRidgeModelHeader)
		+ sizeof(float) * (size_t)model->n_features;

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaRidgeModelHeader *)base;
	hdr->feature_dim = model->n_features;
	hdr->n_samples = model->n_samples;
	hdr->intercept = (float)model->intercept;
	hdr->lambda = model->lambda;
	hdr->r_squared = model->r_squared;
	hdr->mse = model->mse;
	hdr->mae = model->mae;

	coef_dest = (float *)(base + sizeof(NdbCudaRidgeModelHeader));
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
			"{\"algorithm\":\"ridge\","
			"\"storage\":\"metal\","
			"\"n_features\":%d,"
			"\"n_samples\":%d,"
			"\"lambda\":%.6f,"
			"\"r_squared\":%.6f,"
			"\"mse\":%.6f,"
			"\"mae\":%.6f}",
			model->n_features,
			model->n_samples,
			model->lambda,
			model->r_squared,
			model->mse,
			model->mae);

		metrics_json = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(buf.data)));
		NDB_SAFE_PFREE_AND_NULL(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

static int
ndb_metal_ridge_train(const float *features,
		     const double *targets,
		     int n_samples,
		     int feature_dim,
		     const Jsonb *hyperparams,
		     bytea **model_data,
		     Jsonb **metrics,
		     char **errstr)
{
	double lambda = 0.01;  /* Default regularization */
	double *h_XtX = NULL;
	double *h_Xty = NULL;
	double *h_XtX_inv = NULL;
	double *h_beta = NULL;
	bytea *payload = NULL;
	Jsonb *metrics_json = NULL;
	size_t XtX_bytes;
	size_t Xty_bytes;
	size_t beta_bytes;
	int dim_with_intercept;
	int i, j, k;
	int rc = -1;

	if (errstr)
		*errstr = NULL;

	if (features == NULL || targets == NULL || n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for Metal Ridge train");
		return -1;
	}

	/* Extract lambda from hyperparameters */
	if (hyperparams != NULL)
	{
		Datum lambda_datum;
		Datum numeric_datum;
		Numeric num;

		lambda_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("lambda"));
		if (DatumGetPointer(lambda_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, lambda_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				lambda = DatumGetFloat8(
					DirectFunctionCall1(numeric_float8,
						NumericGetDatum(num)));
				if (lambda < 0.0)
					lambda = 0.01;
				if (lambda > 1000.0)
					lambda = 1000.0;
			}
		}
	}

	dim_with_intercept = feature_dim + 1;

	/* Allocate host memory for matrices */
	XtX_bytes = sizeof(double) * (size_t)dim_with_intercept * (size_t)dim_with_intercept;
	Xty_bytes = sizeof(double) * (size_t)dim_with_intercept;
	beta_bytes = sizeof(double) * (size_t)dim_with_intercept;

	h_XtX = (double *)palloc0(XtX_bytes);
	h_Xty = (double *)palloc0(Xty_bytes);
	h_XtX_inv = (double *)palloc(XtX_bytes);
	h_beta = (double *)palloc(beta_bytes);

	/* Compute X'X and X'y */
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

		NDB_SAFE_PFREE_AND_NULL(xi);
	}

	/* Add Ridge penalty (λI) to diagonal (excluding intercept) */
	for (i = 1; i < dim_with_intercept; i++)
		h_XtX[i * dim_with_intercept + i] += lambda;

	/* Invert X'X + λI using Gauss-Jordan elimination */
	{
		double **augmented;
		int row, col, k;
		double pivot, factor;
		bool invert_success = true;

		/* Create augmented matrix [A | I] */
		augmented = (double **)palloc(sizeof(double *) * dim_with_intercept);
		for (row = 0; row < dim_with_intercept; row++)
		{
			augmented[row] = (double *)palloc(sizeof(double) * 2 * dim_with_intercept);
			for (col = 0; col < dim_with_intercept; col++)
			{
				augmented[row][col] = h_XtX[row * dim_with_intercept + col];
				augmented[row][col + dim_with_intercept] = (row == col) ? 1.0 : 0.0;
			}
		}

		/* Gauss-Jordan elimination */
		for (row = 0; row < dim_with_intercept; row++)
		{
			pivot = augmented[row][row];
			if (fabs(pivot) < 1e-10)
			{
				bool found = false;
				for (k = row + 1; k < dim_with_intercept; k++)
				{
					if (fabs(augmented[k][row]) > 1e-10)
					{
						double *temp = augmented[row];
						augmented[row] = augmented[k];
						augmented[k] = temp;
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

			for (k = 0; k < dim_with_intercept; k++)
			{
				if (k != row)
				{
					factor = augmented[k][row];
					for (col = 0; col < 2 * dim_with_intercept; col++)
						augmented[k][col] -= factor * augmented[row][col];
				}
			}
		}

		if (invert_success)
		{
			for (row = 0; row < dim_with_intercept; row++)
				for (col = 0; col < dim_with_intercept; col++)
					h_XtX_inv[row * dim_with_intercept + col] = augmented[row][col + dim_with_intercept];
		}

		for (row = 0; row < dim_with_intercept; row++)
			NDB_SAFE_PFREE_AND_NULL(augmented[row]);
		NDB_SAFE_PFREE_AND_NULL(augmented);

		if (!invert_success)
		{
			NDB_SAFE_PFREE_AND_NULL(h_XtX);
			NDB_SAFE_PFREE_AND_NULL(h_Xty);
			NDB_SAFE_PFREE_AND_NULL(h_XtX_inv);
			NDB_SAFE_PFREE_AND_NULL(h_beta);
			if (errstr)
				*errstr = pstrdup("Matrix is singular, cannot compute Ridge regression");
			return -1;
		}
	}

	/* Compute β = (X'X + λI)^(-1)X'y */
	for (i = 0; i < dim_with_intercept; i++)
	{
		h_beta[i] = 0.0;
		for (j = 0; j < dim_with_intercept; j++)
			h_beta[i] += h_XtX_inv[i * dim_with_intercept + j] * h_Xty[j];
	}

	/* Build model */
	{
		RidgeModel model;
		double y_mean = 0.0;
		double ss_tot = 0.0;
		double ss_res = 0.0;
		double mse = 0.0;
		double mae = 0.0;

		model.n_features = feature_dim;
		model.n_samples = n_samples;
		model.intercept = h_beta[0];
		model.lambda = lambda;
		model.coefficients = (double *)palloc(sizeof(double) * feature_dim);
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
		rc = ndb_metal_ridge_pack(&model, &payload, &metrics_json, errstr);

		NDB_SAFE_PFREE_AND_NULL(model.coefficients);
	}

	/* Cleanup */
	NDB_SAFE_PFREE_AND_NULL(h_XtX);
	NDB_SAFE_PFREE_AND_NULL(h_Xty);
	NDB_SAFE_PFREE_AND_NULL(h_XtX_inv);
	NDB_SAFE_PFREE_AND_NULL(h_beta);

	if (rc == 0 && payload != NULL)
	{
		*model_data = payload;
		if (metrics != NULL)
			*metrics = metrics_json;
		return 0;
	}

	if (payload != NULL)
		NDB_SAFE_PFREE_AND_NULL(payload);
	if (metrics_json != NULL)
		NDB_SAFE_PFREE_AND_NULL(metrics_json);

	return -1;
}

static int
ndb_metal_ridge_predict(const bytea *model_data,
		       const float *input,
		       int feature_dim,
		       double *prediction_out,
		       char **errstr)
{
	const NdbCudaRidgeModelHeader *hdr;
	const float *coefficients;
	const bytea *detoasted;
	double prediction;
	int i;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || prediction_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for Metal Ridge predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted = (const bytea *)PG_DETOAST_DATUM(PointerGetDatum(model_data));
	
	hdr = (const NdbCudaRidgeModelHeader *)VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model has %d, input has %d", 
				hdr->feature_dim, feature_dim);
		return -1;
	}

	coefficients = (const float *)((const char *)hdr + sizeof(NdbCudaRidgeModelHeader));

	/* Compute prediction: y = intercept + Σ(coef_i * x_i) */
	prediction = (double)hdr->intercept;
	for (i = 0; i < feature_dim; i++)
		prediction += (double)coefficients[i] * (double)input[i];

	*prediction_out = prediction;
	return 0;
}

/* Metal Lasso Regression model header (matches CUDA) */
typedef struct NdbCudaLassoModelHeader
{
	int32 feature_dim;
	int32 n_samples;
	float intercept;
	float *coefficients;  /* Array of feature_dim floats */
	double lambda;
	int32 max_iters;
	double r_squared;
	double mse;
	double mae;
} NdbCudaLassoModelHeader;

/*
 * Soft thresholding operator for Lasso
 */
static double
soft_threshold_metal(double x, double lambda)
{
	if (x > lambda)
		return x - lambda;
	else if (x < -lambda)
		return x + lambda;
	else
		return 0.0;
}

static int
ndb_metal_lasso_pack(const struct LassoModel *model,
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
			*errstr = pstrdup("invalid Lasso model for Metal pack");
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
			"\"storage\":\"metal\","
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
		NDB_SAFE_PFREE_AND_NULL(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

static int
ndb_metal_lasso_train(const float *features,
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
			*errstr = pstrdup("invalid input parameters for Metal Lasso train");
		return -1;
	}

	/* Extract hyperparameters from JSON */
	if (hyperparams != NULL)
	{
		Datum lambda_datum;
		Datum max_iters_datum;
		Datum numeric_datum;
		Numeric num;

		lambda_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("lambda"));
		if (DatumGetPointer(lambda_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, lambda_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				lambda = DatumGetFloat8(
					DirectFunctionCall1(numeric_float8,
						NumericGetDatum(num)));
				if (lambda < 0.0)
					lambda = 0.01;
				if (lambda > 1000.0)
					lambda = 1000.0;
			}
		}

		max_iters_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("max_iters"));
		if (DatumGetPointer(max_iters_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, max_iters_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				max_iters = DatumGetInt32(
					DirectFunctionCall1(numeric_int4,
						NumericGetDatum(num)));
				if (max_iters <= 0)
					max_iters = 1000;
				if (max_iters > 100000)
					max_iters = 100000;
			}
		}
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
			weights[j] = soft_threshold_metal(rho / z, lambda / z);

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
		rc = ndb_metal_lasso_pack(&model, &payload, &metrics_json, errstr);

		NDB_SAFE_PFREE_AND_NULL(model.coefficients);
	}

	/* Cleanup */
	NDB_SAFE_PFREE_AND_NULL(weights);
	NDB_SAFE_PFREE_AND_NULL(weights_old);
	NDB_SAFE_PFREE_AND_NULL(residuals);

	if (rc == 0 && payload != NULL)
	{
		*model_data = payload;
		if (metrics != NULL)
			*metrics = metrics_json;
		return 0;
	}

	if (payload != NULL)
		NDB_SAFE_PFREE_AND_NULL(payload);
	if (metrics_json != NULL)
		NDB_SAFE_PFREE_AND_NULL(metrics_json);

	return -1;
}

static int
ndb_metal_lasso_predict(const bytea *model_data,
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
			*errstr = pstrdup("invalid parameters for Metal Lasso predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted = (const bytea *)PG_DETOAST_DATUM(PointerGetDatum(model_data));
	
	hdr = (const NdbCudaLassoModelHeader *)VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model has %d, input has %d", 
				hdr->feature_dim, feature_dim);
		return -1;
	}

	coefficients = (const float *)((const char *)hdr + sizeof(NdbCudaLassoModelHeader));

	/* Compute prediction: y = intercept + Σ(coef_i * x_i) */
	prediction = (double)hdr->intercept;
	for (i = 0; i < feature_dim; i++)
		prediction += (double)coefficients[i] * (double)input[i];

	*prediction_out = prediction;
	return 0;
}

/* Metal Naive Bayes Functions */

/* Gaussian Naive Bayes model structure (matches ml_naive_bayes.c) */
typedef struct GaussianNBModel
{
	double	   *class_priors;		/* P(class) */
	double	  **means;				/* Mean for each feature per class */
	double	  **variances;			/* Variance for each feature per class */
	int			n_classes;
	int			n_features;
} GaussianNBModel;

/* Metal-specific Naive Bayes model header (matches CUDA) */
typedef struct NdbCudaNbModelHeader
{
	int32 n_classes;
	int32 n_features;
	int32 n_samples;
} NdbCudaNbModelHeader;

static int
ndb_metal_nb_pack(const struct GaussianNBModel *model,
		  bytea **model_data,
		  Jsonb **metrics,
		  char **errstr)
{
	size_t payload_bytes;
	bytea *blob;
	char *base;
	NdbCudaNbModelHeader *hdr;
	double *priors_dest;
	double *means_dest;
	double *variances_dest;
	int i, j;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid NB model for Metal pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaNbModelHeader)
		+ sizeof(double) * (size_t)model->n_classes  /* priors */
		+ sizeof(double) * (size_t)model->n_classes * (size_t)model->n_features  /* means */
		+ sizeof(double) * (size_t)model->n_classes * (size_t)model->n_features;  /* variances */

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaNbModelHeader *)base;
	hdr->n_classes = model->n_classes;
	hdr->n_features = model->n_features;
	hdr->n_samples = 0;  /* Not stored in model */

	priors_dest = (double *)(base + sizeof(NdbCudaNbModelHeader));
	means_dest = (double *)(base + sizeof(NdbCudaNbModelHeader) + sizeof(double) * (size_t)model->n_classes);
	variances_dest = (double *)(base + sizeof(NdbCudaNbModelHeader) + sizeof(double) * (size_t)model->n_classes + sizeof(double) * (size_t)model->n_classes * (size_t)model->n_features);

	if (model->class_priors != NULL)
	{
		for (i = 0; i < model->n_classes; i++)
			priors_dest[i] = model->class_priors[i];
	}

	if (model->means != NULL)
	{
		for (i = 0; i < model->n_classes; i++)
		{
			if (model->means[i] != NULL)
			{
				for (j = 0; j < model->n_features; j++)
					means_dest[i * model->n_features + j] = model->means[i][j];
			}
		}
	}

	if (model->variances != NULL)
	{
		for (i = 0; i < model->n_classes; i++)
		{
			if (model->variances[i] != NULL)
			{
				for (j = 0; j < model->n_features; j++)
					variances_dest[i * model->n_features + j] = model->variances[i][j];
			}
		}
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"naive_bayes\","
			"\"storage\":\"metal\","
			"\"n_classes\":%d,"
			"\"n_features\":%d}",
			model->n_classes,
			model->n_features);

		metrics_json = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(buf.data)));
		NDB_SAFE_PFREE_AND_NULL(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

static int
ndb_metal_nb_train(const float *features,
		   const double *labels,
		   int n_samples,
		   int feature_dim,
		   int class_count,
		   const Jsonb *hyperparams,
		   bytea **model_data,
		   Jsonb **metrics,
		   char **errstr)
{
	int *class_counts = NULL;
	double *class_priors = NULL;
	double *means = NULL;
	double *variances = NULL;
	struct GaussianNBModel model;
	bytea *blob = NULL;
	Jsonb *metrics_json = NULL;
	int i, j, class;
	int rc = -1;

	if (errstr)
		*errstr = NULL;

	if (features == NULL || labels == NULL || model_data == NULL
		|| n_samples <= 0 || feature_dim <= 0 || class_count <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for Metal NB train");
		return -1;
	}

	elog(DEBUG1,
		"neurondb: Metal NB train: n_samples=%d, feature_dim=%d, class_count=%d",
		n_samples, feature_dim, class_count);

	/* Allocate host memory */
	class_counts = (int *)palloc0(sizeof(int) * class_count);
	class_priors = (double *)palloc(sizeof(double) * class_count);
	means = (double *)palloc0(sizeof(double) * (size_t)class_count * (size_t)feature_dim);
	variances = (double *)palloc0(sizeof(double) * (size_t)class_count * (size_t)feature_dim);

	/* Step 1: Count samples per class */
	for (i = 0; i < n_samples; i++)
	{
		class = (int)labels[i];
		if (class >= 0 && class < class_count)
			class_counts[class]++;
	}

	/* Step 2: Compute class priors */
	for (i = 0; i < class_count; i++)
	{
		if (class_counts[i] > 0)
			class_priors[i] = (double)class_counts[i] / n_samples;
		else
			class_priors[i] = 1e-10;  /* Avoid log(0) */
	}

	/* Step 3: Compute means */
	for (i = 0; i < n_samples; i++)
	{
		class = (int)labels[i];
		if (class >= 0 && class < class_count && class_counts[class] > 0)
		{
			for (j = 0; j < feature_dim; j++)
				means[class * feature_dim + j] += (double)features[i * feature_dim + j];
		}
	}

	for (i = 0; i < class_count; i++)
	{
		if (class_counts[i] > 0)
		{
			for (j = 0; j < feature_dim; j++)
				means[i * feature_dim + j] /= class_counts[i];
		}
	}

	/* Step 4: Compute variances */
	for (i = 0; i < n_samples; i++)
	{
		class = (int)labels[i];
		if (class >= 0 && class < class_count && class_counts[class] > 0)
		{
			for (j = 0; j < feature_dim; j++)
			{
				double diff = (double)features[i * feature_dim + j] - means[class * feature_dim + j];
				variances[class * feature_dim + j] += diff * diff;
			}
		}
	}

	for (i = 0; i < class_count; i++)
	{
		if (class_counts[i] > 0)
		{
			for (j = 0; j < feature_dim; j++)
				variances[i * feature_dim + j] /= class_counts[i];
		}
	}

	/* Regularize variances to avoid division by zero */
	for (i = 0; i < class_count * feature_dim; i++)
	{
		if (variances[i] < 1e-9)
			variances[i] = 1e-9;
	}

	/* Build model structure for packing */
	model.n_classes = class_count;
	model.n_features = feature_dim;
	model.class_priors = class_priors;
	model.means = (double **)palloc(sizeof(double *) * class_count);
	model.variances = (double **)palloc(sizeof(double *) * class_count);

	for (i = 0; i < class_count; i++)
	{
		model.means[i] = means + i * feature_dim;
		model.variances[i] = variances + i * feature_dim;
	}

	/* Pack model */
	if (ndb_metal_nb_pack(&model, &blob, &metrics_json, errstr) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("Metal NB model packing failed");
		goto cleanup;
	}

	*model_data = blob;
	if (metrics != NULL)
		*metrics = metrics_json;

	rc = 0;

cleanup:
	/* Free model structure arrays (not the data they point to) */
	if (model.means != NULL)
		NDB_SAFE_PFREE_AND_NULL(model.means);
	if (model.variances != NULL)
		NDB_SAFE_PFREE_AND_NULL(model.variances);

	/* Free host memory */
	if (class_counts != NULL)
		NDB_SAFE_PFREE_AND_NULL(class_counts);
	if (class_priors != NULL)
		NDB_SAFE_PFREE_AND_NULL(class_priors);
	if (means != NULL)
		NDB_SAFE_PFREE_AND_NULL(means);
	if (variances != NULL)
		NDB_SAFE_PFREE_AND_NULL(variances);

	return rc;
}

static int
ndb_metal_nb_predict(const bytea *model_data,
		    const float *input,
		    int feature_dim,
		    int *class_out,
		    double *probability_out,
		    char **errstr)
{
	const char *base;
	NdbCudaNbModelHeader *hdr;
	const double *priors;
	const double *means;
	const double *variances;
	double *class_log_probs;
	double max_log_prob;
	int best_class;
	int i, j;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || class_out == NULL || probability_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for Metal NB prediction");
		return -1;
	}

	base = VARDATA_ANY(model_data);
	hdr = (NdbCudaNbModelHeader *)base;

	if (feature_dim != hdr->n_features)
	{
		if (errstr)
			*errstr = pstrdup("feature dimension mismatch");
		return -1;
	}

	priors = (const double *)(base + sizeof(NdbCudaNbModelHeader));
	means = (const double *)(base + sizeof(NdbCudaNbModelHeader) + sizeof(double) * (size_t)hdr->n_classes);
	variances = (const double *)(base + sizeof(NdbCudaNbModelHeader) + sizeof(double) * (size_t)hdr->n_classes + sizeof(double) * (size_t)hdr->n_classes * (size_t)hdr->n_features);

	class_log_probs = (double *)palloc(sizeof(double) * hdr->n_classes);

	/* Compute log probability for each class */
	for (i = 0; i < hdr->n_classes; i++)
	{
		double log_prob = log(priors[i] + 1e-10);  /* Add small epsilon to avoid log(0) */

		for (j = 0; j < feature_dim; j++)
		{
			double diff = (double)input[j] - means[i * feature_dim + j];
			double var = variances[i * feature_dim + j] + 1e-9;  /* Regularization */
			double log_pdf = -0.5 * log(2.0 * M_PI * var) - 0.5 * (diff * diff) / var;
			log_prob += log_pdf;
		}

		class_log_probs[i] = log_prob;
	}

	/* Find class with maximum log probability */
	max_log_prob = class_log_probs[0];
	best_class = 0;
	for (i = 1; i < hdr->n_classes; i++)
	{
		if (class_log_probs[i] > max_log_prob)
		{
			max_log_prob = class_log_probs[i];
			best_class = i;
		}
	}

	/* Convert log probabilities to probabilities (normalize) */
	{
		double max_log = max_log_prob;
		double sum = 0.0;
		int k;

		for (k = 0; k < hdr->n_classes; k++)
			sum += exp(class_log_probs[k] - max_log);

		*probability_out = exp(class_log_probs[best_class] - max_log) / sum;
	}

	*class_out = best_class;
	NDB_SAFE_PFREE_AND_NULL(class_log_probs);

	return 0;
}

/* Metal Gaussian Mixture Model Functions */

/* GMM model structure (matches ml_gmm.c) */
typedef struct GMMModel
{
	int k;  /* Number of components */
	int dim;  /* Feature dimension */
	double *mixing_coeffs;  /* Mixing coefficients */
	double **means;  /* Mean vectors */
	double **variances;  /* Variance vectors */
} GMMModel;

/* Metal-specific GMM model header (matches CUDA) */
typedef struct NdbCudaGmmModelHeader
{
	int32 n_components;
	int32 n_features;
	int32 n_samples;
	int32 max_iters;
	double tolerance;
} NdbCudaGmmModelHeader;

#define GMM_MIN_PROB 1e-10
#define GMM_EPSILON 1e-9

static int
ndb_metal_gmm_pack(const struct GMMModel *model,
		   bytea **model_data,
		   Jsonb **metrics,
		   char **errstr)
{
	size_t payload_bytes;
	bytea *blob;
	char *base;
	NdbCudaGmmModelHeader *hdr;
	double *mixing_dest;
	double *means_dest;
	double *variances_dest;
	int i, j;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid GMM model for Metal pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaGmmModelHeader)
		+ sizeof(double) * (size_t)model->k  /* mixing_coeffs */
		+ sizeof(double) * (size_t)model->k * (size_t)model->dim  /* means */
		+ sizeof(double) * (size_t)model->k * (size_t)model->dim;  /* variances */

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaGmmModelHeader *)base;
	hdr->n_components = model->k;
	hdr->n_features = model->dim;
	hdr->n_samples = 0;  /* Not stored in model */
	hdr->max_iters = 100;
	hdr->tolerance = 1e-6;

	mixing_dest = (double *)(base + sizeof(NdbCudaGmmModelHeader));
	means_dest = (double *)(base + sizeof(NdbCudaGmmModelHeader) + sizeof(double) * (size_t)model->k);
	variances_dest = (double *)(base + sizeof(NdbCudaGmmModelHeader) + sizeof(double) * (size_t)model->k + sizeof(double) * (size_t)model->k * (size_t)model->dim);

	if (model->mixing_coeffs != NULL)
	{
		for (i = 0; i < model->k; i++)
			mixing_dest[i] = model->mixing_coeffs[i];
	}

	if (model->means != NULL)
	{
		for (i = 0; i < model->k; i++)
		{
			if (model->means[i] != NULL)
			{
				for (j = 0; j < model->dim; j++)
					means_dest[i * model->dim + j] = model->means[i][j];
			}
		}
	}

	if (model->variances != NULL)
	{
		for (i = 0; i < model->k; i++)
		{
			if (model->variances[i] != NULL)
			{
				for (j = 0; j < model->dim; j++)
					variances_dest[i * model->dim + j] = model->variances[i][j];
			}
		}
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"gmm\","
			"\"storage\":\"metal\","
			"\"n_components\":%d,"
			"\"n_features\":%d}",
			model->k,
			model->dim);

		metrics_json = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(buf.data)));
		NDB_SAFE_PFREE_AND_NULL(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

static int
ndb_metal_gmm_train(const float *features,
		    int n_samples,
		    int feature_dim,
		    int n_components,
		    const Jsonb *hyperparams,
		    bytea **model_data,
		    Jsonb **metrics,
		    char **errstr)
{
	int max_iters = 100;
	double tolerance = 1e-6;
	double *mixing_coeffs = NULL;
	double *means = NULL;
	double *variances = NULL;
	double **means_2d = NULL;
	double **variances_2d = NULL;
	double *responsibilities = NULL;
	double log_likelihood = 0.0;
	double prev_log_likelihood = -DBL_MAX;
	struct GMMModel model;
	bytea *blob = NULL;
	Jsonb *metrics_json = NULL;
	int iter;
	int i, k, d;
	int rc = -1;

	if (errstr)
		*errstr = NULL;

	if (features == NULL || model_data == NULL
		|| n_samples <= 0 || feature_dim <= 0 || n_components <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for Metal GMM train");
		return -1;
	}

	elog(DEBUG1,
		"neurondb: Metal GMM train: n_samples=%d, feature_dim=%d, n_components=%d",
		n_samples, feature_dim, n_components);

	/* Extract hyperparameters from JSONB */
	if (hyperparams != NULL)
	{
		Datum max_iters_datum;
		Datum tolerance_datum;
		Datum numeric_datum;
		Numeric num;

		max_iters_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("max_iters"));
		if (DatumGetPointer(max_iters_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, max_iters_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				max_iters = DatumGetInt32(
					DirectFunctionCall1(numeric_int4,
						NumericGetDatum(num)));
				if (max_iters <= 0)
					max_iters = 100;
			}
		}

		tolerance_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("tolerance"));
		if (DatumGetPointer(tolerance_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, tolerance_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				tolerance = DatumGetFloat8(
					DirectFunctionCall1(numeric_float8,
						NumericGetDatum(num)));
				if (tolerance <= 0.0)
					tolerance = 1e-6;
			}
		}
	}

	/* Allocate host memory */
	mixing_coeffs = (double *)palloc(sizeof(double) * n_components);
	means = (double *)palloc0(sizeof(double) * (size_t)n_components * (size_t)feature_dim);
	variances = (double *)palloc0(sizeof(double) * (size_t)n_components * (size_t)feature_dim);
	means_2d = (double **)palloc(sizeof(double *) * n_components);
	variances_2d = (double **)palloc(sizeof(double *) * n_components);
	responsibilities = (double *)palloc(sizeof(double) * (size_t)n_samples * (size_t)n_components);

	/* Initialize means with random data points (K-means++ style) */
	for (k = 0; k < n_components; k++)
	{
		int idx = (k * n_samples) / n_components;  /* Spread initial means */
		means_2d[k] = means + k * feature_dim;
		variances_2d[k] = variances + k * feature_dim;

		for (d = 0; d < feature_dim; d++)
			means[k * feature_dim + d] = (double)features[idx * feature_dim + d];

		/* Initialize variances to 1.0 */
		for (d = 0; d < feature_dim; d++)
			variances[k * feature_dim + d] = 1.0;

		/* Equal mixing coefficients initially */
		mixing_coeffs[k] = 1.0 / n_components;
	}

	/* EM algorithm */
	for (iter = 0; iter < max_iters; iter++)
	{
		/* E-step: Compute responsibilities */
		for (i = 0; i < n_samples; i++)
		{
			double sum = 0.0;
			const float *x = features + i * feature_dim;

			for (k = 0; k < n_components; k++)
			{
				double log_likelihood = 0.0;
				double log_det = 0.0;

				for (d = 0; d < feature_dim; d++)
				{
					double diff = (double)x[d] - means[k * feature_dim + d];
					double var = variances[k * feature_dim + d] + GMM_EPSILON;
					log_likelihood -= 0.5 * (diff * diff) / var;
					log_det += log(var);
				}

				log_likelihood -= 0.5 * (feature_dim * log(2.0 * M_PI) + log_det);
				responsibilities[i * n_components + k] = mixing_coeffs[k] * exp(log_likelihood);
				sum += responsibilities[i * n_components + k];
			}

			/* Normalize responsibilities */
			if (sum > GMM_MIN_PROB)
			{
				for (k = 0; k < n_components; k++)
					responsibilities[i * n_components + k] /= sum;
			}
		}

		/* Compute log-likelihood for convergence check */
		log_likelihood = 0.0;
		for (i = 0; i < n_samples; i++)
		{
			double sum = 0.0;
			for (k = 0; k < n_components; k++)
				sum += responsibilities[i * n_components + k];
			if (sum > GMM_MIN_PROB)
				log_likelihood += log(sum);
		}
		log_likelihood /= n_samples;

		/* Check convergence */
		if (fabs(log_likelihood - prev_log_likelihood) < tolerance)
		{
			break;
		}
		prev_log_likelihood = log_likelihood;

		/* M-step: Update parameters */
		/* Compute N_k (sum of responsibilities per component) */
		for (k = 0; k < n_components; k++)
		{
			double Nk = 0.0;
			for (i = 0; i < n_samples; i++)
				Nk += responsibilities[i * n_components + k];

			/* Update mixing coefficients */
			if (Nk > GMM_MIN_PROB)
				mixing_coeffs[k] = Nk / n_samples;
			else
				mixing_coeffs[k] = 1.0 / n_components;

			/* Update means */
			for (d = 0; d < feature_dim; d++)
				means[k * feature_dim + d] = 0.0;

			for (i = 0; i < n_samples; i++)
			{
				const float *x = features + i * feature_dim;
				for (d = 0; d < feature_dim; d++)
					means[k * feature_dim + d] += responsibilities[i * n_components + k] * (double)x[d];
			}

			if (Nk > GMM_MIN_PROB)
			{
				for (d = 0; d < feature_dim; d++)
					means[k * feature_dim + d] /= Nk;
			}

			/* Update variances */
			for (d = 0; d < feature_dim; d++)
				variances[k * feature_dim + d] = 0.0;

			for (i = 0; i < n_samples; i++)
			{
				const float *x = features + i * feature_dim;
				for (d = 0; d < feature_dim; d++)
				{
					double diff = (double)x[d] - means[k * feature_dim + d];
					variances[k * feature_dim + d] += responsibilities[i * n_components + k] * diff * diff;
				}
			}

			if (Nk > GMM_MIN_PROB)
			{
				for (d = 0; d < feature_dim; d++)
					variances[k * feature_dim + d] /= Nk;
			}
		}

		/* Regularize variances */
		for (k = 0; k < n_components; k++)
		{
			for (d = 0; d < feature_dim; d++)
			{
				if (variances[k * feature_dim + d] < GMM_EPSILON)
					variances[k * feature_dim + d] = GMM_EPSILON;
			}
		}
	}

	/* Build model structure for packing */
	model.k = n_components;
	model.dim = feature_dim;
	model.mixing_coeffs = mixing_coeffs;
	model.means = means_2d;
	model.variances = variances_2d;

	/* Pack model */
	if (ndb_metal_gmm_pack(&model, &blob, &metrics_json, errstr) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("Metal GMM model packing failed");
		goto cleanup;
	}

	*model_data = blob;
	if (metrics != NULL)
		*metrics = metrics_json;

	rc = 0;

cleanup:
	/* Free model structure arrays (not the data they point to) */
	if (means_2d != NULL)
		NDB_SAFE_PFREE_AND_NULL(means_2d);
	if (variances_2d != NULL)
		NDB_SAFE_PFREE_AND_NULL(variances_2d);

	/* Free host memory */
	if (mixing_coeffs != NULL)
		NDB_SAFE_PFREE_AND_NULL(mixing_coeffs);
	if (means != NULL)
		NDB_SAFE_PFREE_AND_NULL(means);
	if (variances != NULL)
		NDB_SAFE_PFREE_AND_NULL(variances);
	if (responsibilities != NULL)
		NDB_SAFE_PFREE_AND_NULL(responsibilities);

	return rc;
}

static int
ndb_metal_gmm_predict(const bytea *model_data,
		     const float *input,
		     int feature_dim,
		     int *cluster_out,
		     double *probability_out,
		     char **errstr)
{
	const char *base;
	NdbCudaGmmModelHeader *hdr;
	const double *mixing;
	const double *means;
	const double *variances;
	double *component_probs;
	double max_prob;
	int best_component;
	int i, j;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || cluster_out == NULL || probability_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for Metal GMM prediction");
		return -1;
	}

	base = VARDATA_ANY(model_data);
	hdr = (NdbCudaGmmModelHeader *)base;

	if (feature_dim != hdr->n_features)
	{
		if (errstr)
			*errstr = pstrdup("feature dimension mismatch");
		return -1;
	}

	mixing = (const double *)(base + sizeof(NdbCudaGmmModelHeader));
	means = (const double *)(base + sizeof(NdbCudaGmmModelHeader) + sizeof(double) * (size_t)hdr->n_components);
	variances = (const double *)(base + sizeof(NdbCudaGmmModelHeader) + sizeof(double) * (size_t)hdr->n_components + sizeof(double) * (size_t)hdr->n_components * (size_t)hdr->n_features);

	component_probs = (double *)palloc(sizeof(double) * hdr->n_components);

	/* Compute probability for each component */
	for (i = 0; i < hdr->n_components; i++)
	{
		double log_likelihood = 0.0;
		double log_det = 0.0;

		for (j = 0; j < feature_dim; j++)
		{
			double diff = (double)input[j] - means[i * feature_dim + j];
			double var = variances[i * feature_dim + j] + GMM_EPSILON;
			log_likelihood -= 0.5 * (diff * diff) / var;
			log_det += log(var);
		}

		log_likelihood -= 0.5 * (feature_dim * log(2.0 * M_PI) + log_det);
		component_probs[i] = mixing[i] * exp(log_likelihood);
	}

	/* Find component with maximum probability */
	max_prob = component_probs[0];
	best_component = 0;
	for (i = 1; i < hdr->n_components; i++)
	{
		if (component_probs[i] > max_prob)
		{
			max_prob = component_probs[i];
			best_component = i;
		}
	}

	/* Normalize probabilities */
	{
		double sum = 0.0;
		int k;

		for (k = 0; k < hdr->n_components; k++)
			sum += component_probs[k];

		if (sum > 0.0)
		{
			for (k = 0; k < hdr->n_components; k++)
				component_probs[k] /= sum;
		}
	}

	*cluster_out = best_component;
	*probability_out = component_probs[best_component];
	NDB_SAFE_PFREE_AND_NULL(component_probs);

	return 0;
}

/* Metal K-Nearest Neighbors Functions */

/* KNN model structure (matches ml_knn.c) */
typedef struct KNNModel
{
	int n_samples;
	int n_features;
	int k;
	int task_type;  /* 0=classification, 1=regression */
	float *features;  /* Training features [n_samples * n_features] */
	double *labels;    /* Training labels [n_samples] */
} KNNModel;

/* Metal-specific KNN model header (matches CUDA) */
typedef struct NdbCudaKnnModelHeader
{
	int32 n_samples;
	int32 n_features;
	int32 k;
	int32 task_type;  /* 0=classification, 1=regression */
} NdbCudaKnnModelHeader;

static int
ndb_metal_knn_pack(const struct KNNModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	size_t payload_bytes;
	bytea *blob;
	char *base;
	NdbCudaKnnModelHeader *hdr;
	float *features_dest;
	double *labels_dest;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model for Metal pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaKnnModelHeader)
		+ sizeof(float) * (size_t)model->n_samples * (size_t)model->n_features
		+ sizeof(double) * (size_t)model->n_samples;

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaKnnModelHeader *)base;
	hdr->n_samples = model->n_samples;
	hdr->n_features = model->n_features;
	hdr->k = model->k;
	hdr->task_type = model->task_type;

	features_dest = (float *)(base + sizeof(NdbCudaKnnModelHeader));
	labels_dest = (double *)(base + sizeof(NdbCudaKnnModelHeader) + sizeof(float) * (size_t)model->n_samples * (size_t)model->n_features);

	if (model->features != NULL)
	{
		memcpy(features_dest, model->features, sizeof(float) * (size_t)model->n_samples * (size_t)model->n_features);
	}

	if (model->labels != NULL)
	{
		memcpy(labels_dest, model->labels, sizeof(double) * (size_t)model->n_samples);
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"knn\","
			"\"storage\":\"metal\","
			"\"n_samples\":%d,"
			"\"n_features\":%d,"
			"\"k\":%d,"
			"\"task_type\":%d}",
			model->n_samples,
			model->n_features,
			model->k,
			model->task_type);

		metrics_json = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(buf.data)));
		NDB_SAFE_PFREE_AND_NULL(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

static int
ndb_metal_knn_train(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	int k,
	int task_type,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	struct KNNModel model;
	bytea *blob = NULL;
	Jsonb *metrics_json = NULL;
	float *features_copy = NULL;
	double *labels_copy = NULL;
	int rc = -1;

	if (errstr)
		*errstr = NULL;

	if (features == NULL || labels == NULL || model_data == NULL
		|| n_samples <= 0 || feature_dim <= 0 || k <= 0 || k > n_samples)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for Metal KNN train");
		return -1;
	}

	elog(DEBUG1,
		"neurondb: Metal KNN train: n_samples=%d, feature_dim=%d, k=%d, task_type=%d",
		n_samples, feature_dim, k, task_type);

	/* Extract hyperparameters if provided */
	if (hyperparams != NULL)
	{
		Datum k_datum;
		Datum task_type_datum;
		Datum numeric_datum;
		Numeric num;

		k_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("k"));
		if (DatumGetPointer(k_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, k_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				k = DatumGetInt32(
					DirectFunctionCall1(numeric_int4,
						NumericGetDatum(num)));
				if (k <= 0 || k > n_samples)
					k = (n_samples < 10) ? n_samples : 10;
			}
		}

		task_type_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("task_type"));
		if (DatumGetPointer(task_type_datum) != NULL)
		{
			int extracted_task;

			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, task_type_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				extracted_task = DatumGetInt32(
					DirectFunctionCall1(numeric_int4,
						NumericGetDatum(num)));
				if (extracted_task == 0 || extracted_task == 1)
					task_type = extracted_task;
			}
		}
	}

	/* Copy training data (KNN is a lazy learner - just stores data) */
	features_copy = (float *)palloc(sizeof(float) * (size_t)n_samples * (size_t)feature_dim);
	labels_copy = (double *)palloc(sizeof(double) * (size_t)n_samples);

	memcpy(features_copy, features, sizeof(float) * (size_t)n_samples * (size_t)feature_dim);
	memcpy(labels_copy, labels, sizeof(double) * (size_t)n_samples);

	/* Build model structure */
	model.n_samples = n_samples;
	model.n_features = feature_dim;
	model.k = k;
	model.task_type = task_type;
	model.features = features_copy;
	model.labels = labels_copy;

	/* Pack model */
	if (ndb_metal_knn_pack(&model, &blob, &metrics_json, errstr) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("Metal KNN model packing failed");
		goto cleanup;
	}

	*model_data = blob;
	if (metrics != NULL)
		*metrics = metrics_json;

	rc = 0;

cleanup:
	/* Note: features_copy and labels_copy are now owned by the packed model */
	/* They will be freed when the model is freed */

	return rc;
}

static int
ndb_metal_knn_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	double *prediction_out,
	char **errstr)
{
	const char *base;
	NdbCudaKnnModelHeader *hdr;
	const float *training_features;
	const double *training_labels;
	float *distances = NULL;
	int *top_k_indices = NULL;
	int i, j;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || prediction_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for Metal KNN prediction");
		return -1;
	}

	base = VARDATA_ANY(model_data);
	hdr = (NdbCudaKnnModelHeader *)base;

	if (feature_dim != hdr->n_features)
	{
		if (errstr)
			*errstr = pstrdup("feature dimension mismatch");
		return -1;
	}

	training_features = (const float *)(base + sizeof(NdbCudaKnnModelHeader));
	training_labels = (const double *)(base + sizeof(NdbCudaKnnModelHeader) + sizeof(float) * (size_t)hdr->n_samples * (size_t)hdr->n_features);

	/* Allocate distance array */
	distances = (float *)palloc(sizeof(float) * hdr->n_samples);
	top_k_indices = (int *)palloc(sizeof(int) * hdr->k);

	/* Step 1: Compute distances using Metal-accelerated L2 distance */
	for (i = 0; i < hdr->n_samples; i++)
	{
		const float *training_vec = training_features + i * hdr->n_features;
		distances[i] = metal_backend_l2_distance(input, training_vec, hdr->n_features);
	}

	/* Step 2: Find top-k indices (simple selection sort) */
	for (i = 0; i < hdr->k && i < hdr->n_samples; i++)
	{
		int min_idx = i;
		float min_dist = distances[i];

		for (j = i + 1; j < hdr->n_samples; j++)
		{
			if (distances[j] < min_dist)
			{
				min_dist = distances[j];
				min_idx = j;
			}
		}

		/* Swap */
		if (min_idx != i)
		{
			float temp_dist = distances[i];
			distances[i] = distances[min_idx];
			distances[min_idx] = temp_dist;
		}

		top_k_indices[i] = min_idx;
	}

	/* Step 3: Compute prediction */
	if (hdr->task_type == 0)
	{
		/* Classification: majority vote */
		int class_votes[2] = {0, 0};
		for (i = 0; i < hdr->k && i < hdr->n_samples; i++)
		{
			int label = (int)training_labels[top_k_indices[i]];
			if (label >= 0 && label < 2)
				class_votes[label]++;
		}
		*prediction_out = (class_votes[0] > class_votes[1]) ? 0.0 : 1.0;
	}
	else
	{
		/* Regression: average */
		double sum = 0.0;
		for (i = 0; i < hdr->k && i < hdr->n_samples; i++)
			sum += training_labels[top_k_indices[i]];
		*prediction_out = sum / hdr->k;
	}

	NDB_SAFE_PFREE_AND_NULL(distances);
	NDB_SAFE_PFREE_AND_NULL(top_k_indices);
	return 0;
}

/*
 * ndb_metal_hf_embed
 *	  Generate text embeddings using Metal-accelerated ONNX Runtime with CoreML provider.
 *
 * This implementation uses ONNX Runtime with CoreML execution provider for GPU acceleration
 * on Apple Silicon devices.
 */
static int
ndb_metal_hf_embed(const char *model_name,
	const char *text,
	float **vec_out,
	int *dim_out,
	char **errstr)
{
#ifdef HAVE_ONNX_RUNTIME
	ONNXModelSession *session = NULL;
	int32 *token_ids = NULL;
	int32 token_length = 0;
	ONNXTensor *input_tensor = NULL;
	ONNXTensor *output_tensor = NULL;
	float *input_data = NULL;
	float *embedding = NULL;
	int64 input_shape[2];
	int embed_dim = 0;
	int i;
	int j;
	float sum;
	int rc = -1;

	if (errstr)
		*errstr = NULL;

	/* Validate input parameters */
	if (model_name == NULL || text == NULL || vec_out == NULL
		|| dim_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Invalid parameters for Metal HF embed");
		return -1;
	}

	/* Validate non-empty strings */
	if (model_name[0] == '\0')
	{
		if (errstr)
			*errstr = pstrdup("Model name cannot be empty");
		return -1;
	}
	if (text[0] == '\0')
	{
		if (errstr)
			*errstr = pstrdup("Input text cannot be empty");
		return -1;
	}

	/* Check ONNX Runtime availability */
	if (!neurondb_onnx_available())
	{
		if (errstr)
			*errstr = pstrdup("ONNX Runtime not available for Metal embeddings");
		return -1;
	}

	/* Initialize outputs */
	*vec_out = NULL;
	*dim_out = 0;

	PG_TRY();
	{
		/* Load or get cached model (will use CoreML provider on Metal) */
		session = neurondb_onnx_get_or_load_model(
			model_name, ONNX_MODEL_EMBEDDING);
		if (session == NULL)
		{
			if (errstr)
				*errstr = pstrdup("Failed to load ONNX model for Metal embedding");
			rc = -1;
			goto cleanup;
		}

		/* Tokenize input text */
		token_ids = neurondb_tokenize_with_model(
			text, 128, &token_length, model_name);
		if (token_ids == NULL || token_length <= 0)
		{
			if (errstr)
				*errstr = pstrdup("Tokenization failed for Metal embedding");
			rc = -1;
			goto cleanup;
		}

		/* Convert token IDs to float array for ONNX */
		input_data = (float *)palloc(token_length * sizeof(float));
		for (i = 0; i < token_length; i++)
			input_data[i] = (float)token_ids[i];

		/* Create input tensor */
		input_shape[0] = 1;
		input_shape[1] = token_length;
		input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 2);
		if (input_tensor == NULL)
		{
			if (errstr)
				*errstr = pstrdup("Failed to create input tensor for Metal embedding");
			rc = -1;
			goto cleanup;
		}

		/* Run inference (uses CoreML provider on Metal) */
		output_tensor = neurondb_onnx_run_inference(session, input_tensor);
		if (output_tensor == NULL)
		{
			if (errstr)
				*errstr = pstrdup("ONNX inference failed for Metal embedding");
			rc = -1;
			goto cleanup;
		}

		/* Calculate embedding dimension */
		embed_dim = output_tensor->size / token_length;
		if (embed_dim <= 0)
		{
			if (errstr)
				*errstr = pstrdup("Invalid embedding dimension from Metal inference");
			rc = -1;
			goto cleanup;
		}

		/* Allocate output embedding */
		embedding = (float *)palloc(embed_dim * sizeof(float));

		/* Pool embeddings (mean pooling across sequence dimension) */
		/* Defensive: Handle zero token_length */
		if (token_length > 0)
		{
			for (i = 0; i < embed_dim; i++)
			{
				sum = 0.0f;
				for (j = 0; j < token_length; j++)
				{
					float val = output_tensor->data[j * embed_dim + i];
					/* Defensive: Handle NaN/Inf in embeddings */
					if (isfinite(val))
						sum += val;
				}
				embedding[i] = sum / token_length;
			}
		}
		else
		{
			/* Zero-length input: return zero embedding */
			for (i = 0; i < embed_dim; i++)
				embedding[i] = 0.0f;
		}

		/* Cleanup */
		neurondb_onnx_free_tensor(input_tensor);
		neurondb_onnx_free_tensor(output_tensor);
		NDB_SAFE_PFREE_AND_NULL(input_data);
		NDB_SAFE_PFREE_AND_NULL(token_ids);

		/* Set outputs */
		*vec_out = embedding;
		*dim_out = embed_dim;
		rc = 0;
	}
	PG_CATCH();
	{
		/* Cleanup on error */
		FlushErrorState();
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("Exception during Metal embedding inference");
		rc = -1;
	}
	PG_END_TRY();

cleanup:
	if (rc != 0)
	{
		if (input_tensor)
			neurondb_onnx_free_tensor(input_tensor);
		if (output_tensor)
			neurondb_onnx_free_tensor(output_tensor);
		if (input_data)
			NDB_SAFE_PFREE_AND_NULL(input_data);
		if (token_ids)
			NDB_SAFE_PFREE_AND_NULL(token_ids);
		if (embedding)
			NDB_SAFE_PFREE_AND_NULL(embedding);
	}

	return rc;
#else
	/* ONNX Runtime not available */
	if (errstr)
		*errstr = pstrdup("ONNX Runtime not compiled in - Metal embeddings require ONNX Runtime");
	return -1;
#endif
}

/*
 * ndb_metal_hf_complete
 *	  Generate text completion using Metal-accelerated ONNX Runtime with CoreML provider.
 *
 * This implementation uses ONNX Runtime with CoreML execution provider for GPU acceleration
 * on Apple Silicon devices.
 */
static int
ndb_metal_hf_complete(const char *model_name,
	const char *prompt,
	const char *params_json,
	char **text_out,
	char **errstr)
{
#ifdef HAVE_ONNX_RUNTIME
	/* Defensive: Validate input parameters */
	if (model_name == NULL || prompt == NULL || text_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Invalid parameters for Metal HF complete");
		return -1;
	}

	/* Validate non-empty strings */
	if (model_name[0] == '\0')
	{
		if (errstr)
			*errstr = pstrdup("Model name cannot be empty");
		return -1;
	}
	if (prompt[0] == '\0')
	{
		if (errstr)
			*errstr = pstrdup("Prompt cannot be empty");
		return -1;
	}

	/* Use ONNX Runtime with CoreML provider (automatically uses Metal on Apple Silicon) */
	return ndb_onnx_hf_complete(model_name, prompt, params_json, text_out, errstr);
#else
	/* ONNX Runtime not available */
	if (errstr)
		*errstr = pstrdup("ONNX Runtime not compiled in - Metal completion requires ONNX Runtime");
	return -1;
#endif
}

/*
 * ndb_metal_hf_rerank
 *	  Rerank documents using Metal-accelerated ONNX Runtime with CoreML provider.
 *
 * This implementation uses ONNX Runtime with CoreML execution provider for GPU acceleration
 * on Apple Silicon devices.
 */
static int
ndb_metal_hf_rerank(const char *model_name,
	const char *query,
	const char **docs,
	int ndocs,
	float **scores_out,
	char **errstr)
{
#ifdef HAVE_ONNX_RUNTIME
	/* Defensive: Validate input parameters */
	if (model_name == NULL || query == NULL || docs == NULL || scores_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Invalid parameters for Metal HF rerank");
		return -1;
	}

	if (ndocs <= 0)
	{
		if (errstr)
			*errstr = pstrdup("Invalid number of documents for Metal HF rerank");
		return -1;
	}

	/* Validate non-empty strings */
	if (model_name[0] == '\0')
	{
		if (errstr)
			*errstr = pstrdup("Model name cannot be empty");
		return -1;
	}
	if (query[0] == '\0')
	{
		if (errstr)
			*errstr = pstrdup("Query cannot be empty");
		return -1;
	}

	/* Validate documents array */
	{
		int i;
		for (i = 0; i < ndocs; i++)
		{
			if (docs[i] == NULL)
			{
				if (errstr)
					*errstr = psprintf("Document %d is NULL", i);
				return -1;
			}
		}
	}

	/* Use ONNX Runtime with CoreML provider (automatically uses Metal on Apple Silicon) */
	return ndb_onnx_hf_rerank(model_name, query, docs, ndocs, scores_out, errstr);
#else
	/* ONNX Runtime not available */
	if (errstr)
		*errstr = pstrdup("ONNX Runtime not compiled in - Metal reranking requires ONNX Runtime");
	return -1;
#endif
}

/* Backend Interface Definition */

static const ndb_gpu_backend ndb_metal_backend = {
	.name = "Metal",
	.provider = "Apple",
	.kind = NDB_GPU_BACKEND_METAL,
	.features = NDB_GPU_FEATURE_DISTANCE | NDB_GPU_FEATURE_QUANTIZE | NDB_GPU_FEATURE_CLUSTERING,
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

	.launch_l2_distance = ndb_metal_launch_l2_distance,
	.launch_cosine = ndb_metal_launch_cosine,
	.launch_kmeans_assign = ndb_metal_launch_kmeans_assign,
	.launch_kmeans_update = ndb_metal_launch_kmeans_update,
	.launch_quant_fp16 = ndb_metal_launch_quant_fp16,
	.launch_quant_int8 = ndb_metal_launch_quant_int8,
	.launch_quant_int4 = ndb_metal_launch_quant_int4,
	.launch_quant_fp8_e4m3 = ndb_metal_launch_quant_fp8_e4m3,
	.launch_quant_fp8_e5m2 = ndb_metal_launch_quant_fp8_e5m2,
	.launch_quant_binary = ndb_metal_launch_quant_binary,
	.launch_pq_encode = ndb_metal_launch_pq_encode,

	.rf_train = ndb_metal_rf_train,
	.rf_predict = ndb_metal_rf_predict,
	.rf_pack = ndb_metal_rf_pack,

	.lr_train = ndb_metal_lr_train,
	.lr_predict = ndb_metal_lr_predict,
	.lr_pack = ndb_metal_lr_pack,

	.linreg_train = ndb_metal_linreg_train,
	.linreg_predict = ndb_metal_linreg_predict,
	.linreg_pack = ndb_metal_linreg_pack,

	.svm_train = ndb_metal_svm_train,
	.svm_predict = ndb_metal_svm_predict,
	.svm_pack = ndb_metal_svm_pack,

	.dt_train = ndb_metal_dt_train,
	.dt_predict = ndb_metal_dt_predict,
	.dt_pack = ndb_metal_dt_pack,

	.ridge_train = ndb_metal_ridge_train,
	.ridge_predict = ndb_metal_ridge_predict,
	.ridge_pack = ndb_metal_ridge_pack,

	.lasso_train = ndb_metal_lasso_train,
	.lasso_predict = ndb_metal_lasso_predict,
	.lasso_pack = ndb_metal_lasso_pack,

	.nb_train = ndb_metal_nb_train,
	.nb_predict = ndb_metal_nb_predict,
	.nb_pack = ndb_metal_nb_pack,

	.gmm_train = ndb_metal_gmm_train,
	.gmm_predict = ndb_metal_gmm_predict,
	.gmm_pack = ndb_metal_gmm_pack,

	.knn_train = ndb_metal_knn_train,
	.knn_predict = ndb_metal_knn_predict,
	.knn_pack = ndb_metal_knn_pack,

	.hf_embed = ndb_metal_hf_embed,
#ifdef HAVE_ONNX_RUNTIME
	.hf_image_embed = ndb_onnx_hf_image_embed, /* Use ONNX Runtime with CoreML provider */
	.hf_multimodal_embed = ndb_onnx_hf_multimodal_embed, /* Use ONNX Runtime with CoreML provider */
#else
	.hf_image_embed = NULL,
	.hf_multimodal_embed = NULL,
#endif
	.hf_complete = ndb_metal_hf_complete,
	.hf_rerank = ndb_metal_hf_rerank,

	.stream_create = ndb_metal_stream_create,
	.stream_destroy = ndb_metal_stream_destroy,
	.stream_synchronize = ndb_metal_stream_synchronize,
};

void
neurondb_gpu_register_metal_backend(void)
{
	if (ndb_gpu_register_backend(&ndb_metal_backend) == 0)
		elog(LOG,
			"neurondb: Metal GPU backend registered successfully");
	else
		elog(WARNING,
			"neurondb: Metal GPU backend registration failed");
}

/* ----------------------------
 * Random Forest predict (CPU-on-Metal implementation)
 * ---------------------------- */

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

/* Weak symbol for RF prediction backend - can be overridden */
__attribute__((weak))
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
			NDB_SAFE_PFREE_AND_NULL(tally);
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
	NDB_SAFE_PFREE_AND_NULL(tally);
	return true;
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

	/* Check if Metal is available - this will try to create a device if not initialized */
	if (!metal_backend_is_available())
	{
		/* If Metal can't be initialized, return error */
		return -1;
	}
	
	/* Try to initialize Metal if device name is "None" (not initialized yet) */
	const char *device_name_check;
	device_name_check = metal_backend_device_name();
	if (device_name_check && strcmp(device_name_check, "None") == 0)
	{
		/* Metal is available but not initialized - try to initialize now */
		if (!metal_backend_init())
		{
			/* Initialization failed - return partial info */
			memset(info, 0, sizeof(NDBGpuDeviceInfo));
			info->device_id = 0;
			strncpy(info->name, "Apple GPU (init failed)", sizeof(info->name) - 1);
			info->name[sizeof(info->name) - 1] = '\0';
			info->is_available = false;
			return 0;
		}
	}

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
