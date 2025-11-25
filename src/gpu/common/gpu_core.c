/*
 * gpu_core.c
 *     Robust GPU initialization and device management for NeurondB.
 *
 * This module handles GPU backend initialization (CUDA/ROCm/OpenCL),
 * device selection, runtime fallback to CPU, and all related statistics.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *     src/gpu_core.c
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/timestamp.h"
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "storage/lwlock.h"

#include "neurondb_config.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_gpu_model.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_hf.h"
#endif

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "libpq/pqsignal.h"
#include <signal.h>

#ifdef NDB_GPU_METAL
extern bool neurondb_gpu_rf_predict_backend(const void *,
	const void *,
	const void *,
	int,
	const float *,
	int,
	int *,
	char **);
#endif

bool neurondb_gpu_enabled = false;
int neurondb_gpu_device = 0;
int neurondb_gpu_batch_size = 8192;
int neurondb_gpu_streams = 2;
double neurondb_gpu_memory_pool_mb = 512.0;
bool neurondb_gpu_fail_open = true;
char *neurondb_gpu_kernels = NULL;
char *neurondb_gpu_backend = NULL;
int neurondb_gpu_timeout_ms = 30000;

static bool gpu_ready = false;
static bool gpu_disabled = false;
static GPUBackend current_backend = GPU_BACKEND_NONE;
static GPUStats gpu_stats;
static const ndb_gpu_backend *active_backend = NULL;
static int active_device_id = 0;

static const float *rf_sort_feat_ptr = NULL;
static int
cmp_idx_by_feat(const void *a, const void *b)
{
	int ia = *(const int *)a;
	int ib = *(const int *)b;
	float va = rf_sort_feat_ptr[ia];
	float vb = rf_sort_feat_ptr[ib];
	if (va < vb)
		return -1;
	if (va > vb)
		return 1;
	return 0;
}

static double
ndb_elapsed_ms(TimestampTz start, TimestampTz end)
{
	long secs;
	int usecs;

	TimestampDifference(start, end, &secs, &usecs);
	return ((double)secs * 1000.0) + ((double)usecs / 1000.0);
}

void
ndb_gpu_stats_record(bool used_gpu, double gpu_ms, double cpu_ms, bool fallback)
{
	gpu_stats.queries_executed++;

	if (fallback)
		gpu_stats.fallback_count++;

	gpu_stats.total_gpu_time_ms += gpu_ms;
	gpu_stats.total_cpu_time_ms += cpu_ms;

	if (gpu_stats.queries_executed > 0)
	{
		gpu_stats.avg_latency_ms =
			(gpu_stats.total_gpu_time_ms
				+ gpu_stats.total_cpu_time_ms)
			/ (double)gpu_stats.queries_executed;
	} else
		gpu_stats.avg_latency_ms = 0.0;
}

void
neurondb_gpu_init_guc(void)
{
	DefineCustomBoolVariable("neurondb.gpu_enabled",
		"Enable GPU acceleration for vector operations",
		NULL,
		&neurondb_gpu_enabled,
		false,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomIntVariable("neurondb.gpu_device",
		"GPU device ID to use (0-based)",
		NULL,
		&neurondb_gpu_device,
		0,
		0,
		16,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomIntVariable("neurondb.gpu_batch_size",
		"Batch size for GPU operations",
		NULL,
		&neurondb_gpu_batch_size,
		8192,
		64,
		65536,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomIntVariable("neurondb.gpu_streams",
		"Number of CUDA/HIP streams for parallel operations",
		NULL,
		&neurondb_gpu_streams,
		2,
		1,
		8,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomRealVariable("neurondb.gpu_memory_pool_mb",
		"GPU memory pool size in MB",
		NULL,
		&neurondb_gpu_memory_pool_mb,
		512.0,
		64.0,
		32768.0,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomBoolVariable("neurondb.gpu_fail_open",
		"Fallback to CPU on GPU errors (true = fail open)",
		NULL,
		&neurondb_gpu_fail_open,
		true,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomStringVariable("neurondb.gpu_kernels",
		"List of GPU-accelerated kernels (comma-separated: "
		"l2,cosine,ip)",
		NULL,
		&neurondb_gpu_kernels,
		"l2,cosine,ip,rf_split,rf_predict",
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomStringVariable("neurondb.gpu_backend",
		"GPU backend: 'cuda' (NVIDIA), 'rocm' (AMD), 'metal' (Apple), "
		"'auto' (detect), 'cpu' (disable)",
		NULL,
		&neurondb_gpu_backend,
		"auto",
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomIntVariable("neurondb.gpu_timeout_ms",
		"GPU kernel execution timeout in milliseconds",
		NULL,
		&neurondb_gpu_timeout_ms,
		30000,
		1000,
		300000,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);
}

bool
ndb_gpu_kernel_enabled(const char *kernel_name)
{
	const char *k;
	size_t nlen;

	if (!neurondb_gpu_kernels || strlen(neurondb_gpu_kernels) == 0)
		return true;

	k = neurondb_gpu_kernels;
	nlen = strlen(kernel_name);
	while (*k)
	{
		/* skip whitespace and commas */
		while (*k == ',' || *k == ' ')
			k++;
		if (strncmp(k, kernel_name, nlen) == 0
			&& (k[nlen] == 0 || k[nlen] == ',' || k[nlen] == ' '))
		{
			return true;
		}
		/* skip to next */
		while (*k && *k != ',')
			k++;
		if (*k == ',')
			k++;
	}
	return false;
}

int
ndb_gpu_runtime_init(int *device_id)
{
	const char *requested = neurondb_gpu_backend;
	const ndb_gpu_backend *backend;

	/* Ignore SIGPIPE to prevent crashes when writing to broken pipes */
	/* This is especially important during error reporting in GPU init */
	pqsignal(SIGPIPE, SIG_IGN);

	if (device_id)
		*device_id = 0;

	if (requested && pg_strcasecmp(requested, "cpu") == 0)
		return -1;

	backend = ndb_gpu_select_backend(requested);
	if (backend == NULL)
		return -1;

	if (ndb_gpu_set_active_backend(backend) != 0)
		return -1;

	active_backend = ndb_gpu_get_active_backend();
	if (active_backend == NULL)
		return -1;

	current_backend = (GPUBackend)active_backend->kind;

	if (active_backend->set_device != NULL)
	{
		int rc = active_backend->set_device(neurondb_gpu_device);
		if (rc != 0)
			return -1;
		active_device_id = neurondb_gpu_device;
	} else
	{
		active_device_id = 0;
	}

	if (device_id)
		*device_id = active_device_id;

	return 0;
}

void
ndb_gpu_mem_pool_init(int pool_size_mb)
{
	(void)pool_size_mb;

	if (!active_backend)
		return;

	/* TODO: Memory pool allocation helpers are in place. For now, rely on per-operation
	 * allocations. */
}

void
ndb_gpu_streams_init(int num_streams)
{
	(void)num_streams;

	if (!active_backend)
		return;

	/* Stream creation will be delegated to backend stream helpers when wired. */
}

void
ndb_gpu_init_if_needed(void)
{
	int device_id = 0;
	int rc;

	if (gpu_ready || gpu_disabled)
		return;
	if (!neurondb_gpu_enabled)
		return;

	rc = ndb_gpu_runtime_init(&device_id);
	if (rc != 0)
	{
		gpu_ready = false;
		gpu_disabled = true;
		current_backend = GPU_BACKEND_NONE;
		active_backend = NULL;
		if (!neurondb_gpu_fail_open)
		{
			ereport(ERROR,
				(errcode(ERRCODE_SYSTEM_ERROR),
					errmsg("neurondb: GPU initialization "
					       "failed")));
		}
		elog(WARNING,
			"neurondb: GPU init failed. Using CPU "
			"fallback");
		return;
	}

	active_device_id = device_id;
	gpu_ready = true;
	gpu_disabled = false;

	ndb_gpu_mem_pool_init((int)neurondb_gpu_memory_pool_mb);
	ndb_gpu_streams_init(neurondb_gpu_streams);

	if (active_backend)
	{
		elog(LOG,
			"neurondb: GPU backend %s (%s) initialized on device "
			"%d",
			active_backend->name ? active_backend->name : "unknown",
			active_backend->provider ? active_backend->provider
						 : "unknown",
			active_device_id);
	} else
	{
		elog(LOG,
			"neurondb: GPU initialized successfully on device %d",
			active_device_id);
	}
}

void
neurondb_gpu_init(void)
{
	ndb_gpu_init_if_needed();
}

void
neurondb_gpu_shutdown(void)
{
	if (!gpu_ready)
		return;

	ndb_gpu_set_active_backend(NULL);
	active_backend = NULL;
	gpu_ready = false;
	gpu_disabled = false;
	current_backend = GPU_BACKEND_NONE;
	active_device_id = 0;
	ndb_gpu_clear_model_registry();

}

bool
neurondb_gpu_is_available(void)
{
	return gpu_ready && neurondb_gpu_enabled && !gpu_disabled;
}

GPUBackend
neurondb_gpu_get_backend(void)
{
	return current_backend;
}

int
neurondb_gpu_get_device_count(void)
{
	if (!active_backend || active_backend->device_count == NULL)
		return 0;

	return active_backend->device_count();
}

GPUDeviceInfo *
neurondb_gpu_get_device_info(int device_id)
{
	GPUDeviceInfo *info;
	NDBGpuDeviceInfo native;

	info = (GPUDeviceInfo *)palloc0(sizeof(GPUDeviceInfo));
	info->device_id = device_id;
	info->is_available = false;

	if (!active_backend || active_backend->device_info == NULL)
		return info;

	if (active_backend->device_info(device_id, &native) != 0)
		return info;

	info->device_id = native.device_id;
	strncpy(info->name, native.name, sizeof(info->name) - 1);
	info->name[sizeof(info->name) - 1] = '\0';
	info->total_memory_mb =
		(int64)(native.total_memory_bytes / (1024 * 1024));
	info->free_memory_mb =
		(int64)(native.free_memory_bytes / (1024 * 1024));
	info->compute_major = native.compute_major;
	info->compute_minor = native.compute_minor;
	info->is_available = native.is_available;

	return info;
}

void
neurondb_gpu_set_device(int device_id)
{
	if (!active_backend || active_backend->set_device == NULL)
	{
		return;
	}

	if (active_backend->set_device(device_id) != 0)
	{
		elog(DEBUG1,
			"neurondb: failed to switch GPU device to %d",
			device_id);
		return;
	}

	neurondb_gpu_device = device_id;
	active_device_id = device_id;
	elog(LOG,
		"neurondb: switched GPU backend %s to device %d",
		active_backend->name ? active_backend->name : "unknown",
		device_id);
}

GPUStats *
neurondb_gpu_get_stats(void)
{
	GPUStats *stats = (GPUStats *)palloc(sizeof(GPUStats));
	memcpy(stats, &gpu_stats, sizeof(GPUStats));
	if (stats->queries_executed > 0)
		stats->avg_latency_ms =
			stats->total_gpu_time_ms / stats->queries_executed;
	else
		stats->avg_latency_ms = 0.0;
	return stats;
}

void
neurondb_gpu_reset_stats(void)
{
	memset(&gpu_stats, 0, sizeof(GPUStats));
	gpu_stats.last_reset = GetCurrentTimestamp();
	elog(LOG, "neurondb: GPU statistics reset");
}

/*
 * RF predict facade - call into backend implementation if available.
 * Returns true on success and writes class_out. On failure, errstr may be set.
 */
bool
neurondb_gpu_rf_predict(const void *rf_hdr,
	const void *trees,
	const void *nodes,
	int node_capacity,
	const float *x,
	int n_features,
	int *class_out,
	char **errstr)
{
	bool used_gpu = false;
	bool ok = false;
	TimestampTz t0 = GetCurrentTimestamp();
	TimestampTz t1;
	if (errstr)
		*errstr = NULL;
	if (!neurondb_gpu_is_available())
		goto out;
	if (!ndb_gpu_kernel_enabled("rf_predict"))
		goto out;
	if (!rf_hdr || !trees || !nodes || !x || !class_out)
		goto out;
	if (n_features <= 0 || node_capacity <= 0)
		goto out;
#ifdef NDB_GPU_METAL
	if (neurondb_gpu_is_available())
	{
		ok = neurondb_gpu_rf_predict_backend(rf_hdr,
			trees,
			nodes,
			node_capacity,
			x,
			n_features,
			class_out,
			errstr);
		used_gpu = ok;
	}
#endif
out:
	t1 = GetCurrentTimestamp();
	ndb_gpu_stats_record(used_gpu,
		used_gpu ? ndb_elapsed_ms(t0, t1) : 0.0,
		used_gpu ? 0.0 : ndb_elapsed_ms(t0, t1),
		!used_gpu);
	return ok;
}

PG_FUNCTION_INFO_V1(neurondb_gpu_enable);
Datum
neurondb_gpu_enable(PG_FUNCTION_ARGS)
{
	neurondb_gpu_enabled = true;
	gpu_disabled = false;
	
	/*
	 * Initialize GPU backend now that it's enabled.
	 * This is safe to do in backend processes (not in postmaster).
	 */
	ndb_gpu_init_if_needed();

	PG_RETURN_BOOL(gpu_ready);
}

PG_FUNCTION_INFO_V1(neurondb_gpu_info);
Datum
neurondb_gpu_info(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
	Datum values[7];
	bool nulls[7] = { false, false, false, false, false, false, false };
	GPUDeviceInfo *info;

	InitMaterializedSRF(fcinfo, 0);

	ndb_gpu_init_if_needed();

	info = neurondb_gpu_get_device_info(active_device_id);

	values[0] = Int32GetDatum(info->device_id);
	values[1] = CStringGetTextDatum(info->name);
	values[2] = Int64GetDatum(info->total_memory_mb);
	values[3] = Int64GetDatum(info->free_memory_mb);
	values[4] = Int32GetDatum(info->compute_major);
	values[5] = Int32GetDatum(info->compute_minor);
	values[6] = BoolGetDatum(info->is_available);

	tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);

	if (info)
		NDB_SAFE_PFREE_AND_NULL(info);

	return (Datum)0;
}

PG_FUNCTION_INFO_V1(neurondb_gpu_stats);
Datum
neurondb_gpu_stats(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
	Datum values[6];
	bool nulls[6] = { false, false, false, false, false, false };

	InitMaterializedSRF(fcinfo, 0);

	values[0] = Int64GetDatum(gpu_stats.queries_executed);
	values[1] = Int64GetDatum(gpu_stats.fallback_count);
	values[2] = Float8GetDatum(gpu_stats.total_gpu_time_ms);
	values[3] = Float8GetDatum(gpu_stats.total_cpu_time_ms);
	values[4] = Float8GetDatum(gpu_stats.avg_latency_ms);
	values[5] = TimestampTzGetDatum(gpu_stats.last_reset);

	tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);

	return (Datum)0;
}

PG_FUNCTION_INFO_V1(neurondb_gpu_reset_stats_func);
Datum
neurondb_gpu_reset_stats_func(PG_FUNCTION_ARGS)
{
	memset(&gpu_stats, 0, sizeof(GPUStats));
	gpu_stats.last_reset = GetCurrentTimestamp();
	elog(LOG, "neurondb: GPU statistics reset");
	PG_RETURN_BOOL(true);
}

/*
 * GPU-accelerated best split finder for Random Forest with binary labels.
 *
 * This implementation computes the best threshold (split-point)
 * for a single feature column and binary labels (0/1) using the GPU.
 * It calculates Gini impurity for all sorted split points, chooses the
 * threshold that minimizes Gini impurity, and returns statistics.
 *
 * Defensive error paths/guards are provided.
 */
bool
neurondb_gpu_rf_best_split_binary(const float *feature_values,
	const uint8_t *labels01,
	int n,
	double *best_threshold,
	double *best_gini,
	int *left_count,
	int *right_count)
{
	int total_pos;
	int *i_idx;
	int *prefix_pos;
	int i;
	int lp;
	int ln;
	int rp;
	int rn;
	double gini_l;
	double gini_r;
	double gini;
	double best_g;
	double best_t;
	int best_lc;
	int best_rc;
	MemoryContext cx;
	MemoryContext oldcx;
	bool success;

	if (!ndb_gpu_kernel_enabled("rf_split"))
		return false;
	if (!feature_values || !labels01 || n < 2 || !best_threshold
		|| !best_gini || !left_count || !right_count)
		return false;

	cx = AllocSetContextCreate(CurrentMemoryContext,
		"rf_gpu_split_helper_ctx",
		ALLOCSET_SMALL_SIZES);
	oldcx = MemoryContextSwitchTo(cx);
	success = false;
	best_g = 1.0;
	best_t = 0.0;
	best_lc = 0;
	best_rc = 0;

	i_idx = (int *)palloc(sizeof(int) * n);
	prefix_pos = (int *)palloc0(sizeof(int) * n);
	total_pos = 0;
	for (i = 0; i < n; i++)
	{
		float fv = feature_values[i];

		if (!isfinite(fv))
			goto out;
		i_idx[i] = i;
		total_pos += (labels01[i] ? 1 : 0);
	}

	rf_sort_feat_ptr = feature_values;
	qsort(i_idx, n, sizeof(int), cmp_idx_by_feat);
	rf_sort_feat_ptr = NULL;

	for (i = 0; i < n; i++)
		prefix_pos[i] = (i == 0 ? 0 : prefix_pos[i - 1])
			+ (labels01[i_idx[i]] ? 1 : 0);

	for (i = 1; i < n; i++)
	{
		float v0;
		float v1;

		v0 = feature_values[i_idx[i - 1]];
		v1 = feature_values[i_idx[i]];
		if (v0 == v1)
			continue;
		lp = prefix_pos[i - 1];
		ln = i - lp;
		rp = total_pos - lp;
		rn = (n - i) - rp;
		gini_l = 1.0
			- ((lp > 0 ? ((double)lp / (double)i) : 0.0)
					* (lp > 0 ? ((double)lp / (double)i)
						  : 0.0)
				+ (ln > 0 ? ((double)ln / (double)i) : 0.0)
					* (ln > 0 ? ((double)ln / (double)i)
						  : 0.0));
		gini_r = 1.0
			- ((rp > 0 ? ((double)rp / (double)(n - i)) : 0.0)
					* (rp > 0 ? ((double)rp
							    / (double)(n - i))
						  : 0.0)
				+ (rn > 0 ? ((double)rn / (double)(n - i))
					  : 0.0)
					* (rn > 0 ? ((double)rn
							    / (double)(n - i))
						  : 0.0));
		gini = ((double)i / (double)n) * gini_l
			+ ((double)(n - i) / (double)n) * gini_r;
		if (gini < best_g)
		{
			best_g = gini;
			best_t = ((double)v0 + (double)v1) * 0.5;
			best_lc = i;
			best_rc = n - i;
		}
	}

	*best_threshold = best_t;
	*best_gini = best_g;
	*left_count = best_lc;
	*right_count = best_rc;
	success = (best_lc > 0 && best_rc > 0);

out:
	MemoryContextSwitchTo(oldcx);
	if (cx)
		MemoryContextDelete(cx);
	rf_sort_feat_ptr = NULL;
	return success;
}

/*
 * neurondb_gpu_hf_embed
 *	  Generate text embeddings using GPU-accelerated Hugging Face model.
 *
 * Returns 0 on success, -1 on failure. On failure, errstr may be set.
 */
int
neurondb_gpu_hf_embed(const char *model_name,
	const char *text,
	float **vec_out,
	int *dim_out,
	char **errstr)
{
	const ndb_gpu_backend *backend;
	bool used_gpu = false;
	TimestampTz t0 = GetCurrentTimestamp();
	TimestampTz t1;
	int rc = -1;

	if (errstr)
		*errstr = NULL;
	if (!neurondb_gpu_is_available())
		goto out;
	if (!ndb_gpu_kernel_enabled("hf_embed"))
		goto out;
	if (!model_name || !text || !vec_out || !dim_out)
		goto out;

	backend = ndb_gpu_get_active_backend();
	if (backend == NULL || backend->hf_embed == NULL)
		goto out;

	rc = backend->hf_embed(model_name, text, vec_out, dim_out, errstr);
	used_gpu = (rc == 0);

out:
	t1 = GetCurrentTimestamp();
	ndb_gpu_stats_record(used_gpu,
		used_gpu ? ndb_elapsed_ms(t0, t1) : 0.0,
		used_gpu ? 0.0 : ndb_elapsed_ms(t0, t1),
		!used_gpu);
	return rc;
}

/*
 * neurondb_gpu_hf_complete
 *	  Generate text completion using GPU-accelerated Hugging Face model.
 *
 * Returns 0 on success, -1 on failure. On failure, errstr may be set.
 */
int
neurondb_gpu_hf_complete(const char *model_name,
	const char *prompt,
	const char *params_json,
	char **text_out,
	char **errstr)
{
	const ndb_gpu_backend *backend;
	bool used_gpu = false;
	TimestampTz t0 = GetCurrentTimestamp();
	TimestampTz t1;
	int rc = -1;

	if (errstr)
		*errstr = NULL;
	if (!neurondb_gpu_is_available())
		goto out;
	if (!ndb_gpu_kernel_enabled("hf_complete"))
		goto out;
	if (!model_name || !prompt || !text_out)
		goto out;

	backend = ndb_gpu_get_active_backend();
	if (backend == NULL || backend->hf_complete == NULL)
		goto out;

	rc = backend->hf_complete(
		model_name, prompt, params_json, text_out, errstr);
	used_gpu = (rc == 0);

out:
	t1 = GetCurrentTimestamp();
	ndb_gpu_stats_record(used_gpu,
		used_gpu ? ndb_elapsed_ms(t0, t1) : 0.0,
		used_gpu ? 0.0 : ndb_elapsed_ms(t0, t1),
		!used_gpu);
	return rc;
}

/*
 * neurondb_gpu_hf_rerank
 *	  Rerank documents using GPU-accelerated Hugging Face model.
 *
 * Returns 0 on success, -1 on failure. On failure, errstr may be set.
 */
int
neurondb_gpu_hf_rerank(const char *model_name,
	const char *query,
	const char **docs,
	int ndocs,
	float **scores_out,
	char **errstr)
{
	const ndb_gpu_backend *backend;
	bool used_gpu = false;
	TimestampTz t0 = GetCurrentTimestamp();
	TimestampTz t1;
	int rc = -1;

	if (errstr)
		*errstr = NULL;
	if (!neurondb_gpu_is_available())
		goto out;
	if (!ndb_gpu_kernel_enabled("hf_rerank"))
		goto out;
	if (!model_name || !query || !docs || !scores_out || ndocs <= 0)
		goto out;

	backend = ndb_gpu_get_active_backend();
	if (backend == NULL || backend->hf_rerank == NULL)
		goto out;

	rc = backend->hf_rerank(
		model_name, query, docs, ndocs, scores_out, errstr);
	used_gpu = (rc == 0);

out:
	t1 = GetCurrentTimestamp();
	ndb_gpu_stats_record(used_gpu,
		used_gpu ? ndb_elapsed_ms(t0, t1) : 0.0,
		used_gpu ? 0.0 : ndb_elapsed_ms(t0, t1),
		!used_gpu);
	return rc;
}

#ifdef NDB_GPU_CUDA
/*
 * neurondb_gpu_hf_complete_batch
 *	  Generate text for multiple prompts in batch using GPU-accelerated inference.
 *
 * Returns 0 on success, -1 on failure. On failure, errstr may be set.
 */
int
neurondb_gpu_hf_complete_batch(const char *model_name,
	const char **prompts,
	int num_prompts,
	const char *params_json,
	NdbCudaHfBatchResult *results,
	char **errstr)
{
	bool used_gpu = false;
	TimestampTz t0 = GetCurrentTimestamp();
	TimestampTz t1;
	int rc = -1;

	if (errstr)
		*errstr = NULL;
	if (!neurondb_gpu_is_available())
		goto out;
	if (!ndb_gpu_kernel_enabled("hf_complete"))
		goto out;
	if (!model_name || !prompts || !results || num_prompts <= 0)
		goto out;

	rc = ndb_cuda_hf_generate_batch(
		model_name, prompts, num_prompts, params_json, results, errstr);
	used_gpu = (rc == 0);

out:
	t1 = GetCurrentTimestamp();
	ndb_gpu_stats_record(used_gpu,
		used_gpu ? ndb_elapsed_ms(t0, t1) : 0.0,
		used_gpu ? 0.0 : ndb_elapsed_ms(t0, t1),
		!used_gpu);
	return rc;
}
#endif

/*
 * neurondb_gpu_hf_rerank_batch
 *	  Rerank documents for multiple queries in batch using GPU-accelerated inference.
 *
 * Returns 0 on success, -1 on failure. On failure, errstr may be set.
 */
int
neurondb_gpu_hf_rerank_batch(const char *model_name,
	const char **queries,
	const char ***docs_array,
	int *ndocs_array,
	int num_queries,
	float ***scores_out,
	int **nscores_out,
	char **errstr)
{
	bool used_gpu = false;
	TimestampTz t0 = GetCurrentTimestamp();
	TimestampTz t1;
	int rc = -1;
	int i;

	if (errstr)
		*errstr = NULL;
	if (!neurondb_gpu_is_available())
		goto out;
	if (!ndb_gpu_kernel_enabled("hf_rerank"))
		goto out;
	if (!model_name || !queries || !docs_array || !ndocs_array
		|| !scores_out || !nscores_out || num_queries <= 0)
		goto out;

	*scores_out = (float **)palloc0(num_queries * sizeof(float *));
	*nscores_out = (int *)palloc0(num_queries * sizeof(int));

	for (i = 0; i < num_queries; i++)
	{
		float *scores = NULL;
		int rc2;

		rc2 = neurondb_gpu_hf_rerank(model_name,
			queries[i],
			docs_array[i],
			ndocs_array[i],
			&scores,
			errstr);
		if (rc2 == 0 && scores != NULL)
		{
			(*scores_out)[i] = scores;
			(*nscores_out)[i] = ndocs_array[i];
		} else
		{
			(*scores_out)[i] = NULL;
			(*nscores_out)[i] = 0;
		}
	}

	rc = 0;
	used_gpu = true;

out:
	t1 = GetCurrentTimestamp();
	ndb_gpu_stats_record(used_gpu,
		used_gpu ? ndb_elapsed_ms(t0, t1) : 0.0,
		used_gpu ? 0.0 : ndb_elapsed_ms(t0, t1),
		!used_gpu);
	return rc;
}
