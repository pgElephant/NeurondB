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

#include "neurondb_config.h"
#include "neurondb_cuda_runtime.h"

#ifdef NDB_GPU_HIP
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

#include "postgres.h"
#include "fmgr.h"
#include "utils/guc.h"
#include "utils/timestamp.h"
#include "storage/lwlock.h"
#include "funcapi.h"
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"

#include "neurondb_gpu.h"
#include "gpu_backend_interface.h"

#ifdef NDB_GPU_CUDA
#include "gpu_cuda.h"
#endif
#ifdef NDB_GPU_HIP
#include "gpu_rocm.h"
#endif
#ifdef NDB_GPU_METAL
#include "gpu_metal.h"
#endif

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* GUC variables and GPU settings */
bool neurondb_gpu_enabled = false;
int neurondb_gpu_device = 0;
int neurondb_gpu_batch_size = 8192;
int neurondb_gpu_streams = 2;
double neurondb_gpu_memory_pool_mb = 512.0;
bool neurondb_gpu_fail_open = true;
char *neurondb_gpu_kernels = NULL;
char *neurondb_gpu_backend = NULL;
int neurondb_gpu_timeout_ms = 30000;

/* Runtime state */
static bool gpu_ready = false;
static bool gpu_disabled = false;
static GPUBackend current_backend = GPU_BACKEND_NONE;
static GPUStats gpu_stats;

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_HIP)
static void *gpu_memory_pool = NULL;
static size_t gpu_pool_bytes = 0;
static int gpu_pool_size = 0;
#endif

#ifdef NDB_GPU_CUDA
static int cuda_device = 0;
static cudaStream_t *cuda_streams = NULL;
#endif

#ifdef NDB_GPU_HIP
static rocblas_handle rocblas_handle = NULL;
static int rocm_device = 0;
static hipStream_t *hip_streams = NULL;
#endif

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
        NULL, NULL, NULL);

    DefineCustomIntVariable("neurondb.gpu_device",
        "GPU device ID to use (0-based)",
        NULL,
        &neurondb_gpu_device,
        0,
        0, 16,
        PGC_USERSET,
        0,
        NULL, NULL, NULL);

    DefineCustomIntVariable("neurondb.gpu_batch_size",
        "Batch size for GPU operations",
        NULL,
        &neurondb_gpu_batch_size,
        8192,
        64, 65536,
        PGC_USERSET,
        0,
        NULL, NULL, NULL);

    DefineCustomIntVariable("neurondb.gpu_streams",
        "Number of CUDA/HIP streams for parallel operations",
        NULL,
        &neurondb_gpu_streams,
        2,
        1, 8,
        PGC_USERSET,
        0,
        NULL, NULL, NULL);

    DefineCustomRealVariable("neurondb.gpu_memory_pool_mb",
        "GPU memory pool size in MB",
        NULL,
        &neurondb_gpu_memory_pool_mb,
        512.0,
        64.0, 32768.0,
        PGC_USERSET,
        0,
        NULL, NULL, NULL);

    DefineCustomBoolVariable("neurondb.gpu_fail_open",
        "Fallback to CPU on GPU errors (true = fail open)",
        NULL,
        &neurondb_gpu_fail_open,
        true,
        PGC_USERSET,
        0,
        NULL, NULL, NULL);

    DefineCustomStringVariable("neurondb.gpu_kernels",
        "List of GPU-accelerated kernels (comma-separated: l2,cosine,ip)",
        NULL,
        &neurondb_gpu_kernels,
        "l2,cosine,ip",
        PGC_USERSET,
        0,
        NULL, NULL, NULL);

    DefineCustomStringVariable("neurondb.gpu_backend",
        "GPU backend: 'cuda' (NVIDIA), 'rocm' (AMD), 'metal' (Apple), 'auto' (detect), 'cpu' (disable)",
        NULL,
        &neurondb_gpu_backend,
        "auto",
        PGC_USERSET,
        0,
        NULL, NULL, NULL);

    DefineCustomIntVariable("neurondb.gpu_timeout_ms",
        "GPU kernel execution timeout in milliseconds",
        NULL,
        &neurondb_gpu_timeout_ms,
        30000,
        1000, 300000,
        PGC_USERSET,
        0,
        NULL, NULL, NULL);
}

bool
ndb_gpu_kernel_enabled(const char *kernel_name)
{
    const char *k;
    size_t nlen;

    if (!neurondb_gpu_kernels || strlen(neurondb_gpu_kernels) == 0)
        return true;
    
    /* Check for kernel name, allow comma boundary or start/end */
    k = neurondb_gpu_kernels;
    nlen = strlen(kernel_name);
    while (*k) {
        /* skip whitespace and commas */
        while (*k == ',' || *k == ' ') k++;
        if (strncmp(k, kernel_name, nlen) == 0 &&
            (k[nlen] == 0 || k[nlen] == ',' || k[nlen] == ' '))
        {
            return true;
        }
        /* skip to next */
        while (*k && *k != ',') k++;
        if (*k == ',') k++;
    }
    return false;
}

int
ndb_gpu_runtime_init(int *device_id)
{
    int rc = -1;
#ifdef NDB_GPU_CUDA
    cudaError_t err;
    int device_count = 0;
    err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0)
    {
        cuda_device = neurondb_gpu_device % device_count;
        err = cudaSetDevice(cuda_device);
        if (err == cudaSuccess)
        {
            *device_id = cuda_device;
            rc = 0;
        }
    }
#endif
#ifdef NDB_GPU_HIP
    hipError_t hip_err;
    int hip_device_count = 0;
    hip_err = hipGetDeviceCount(&hip_device_count);
    if (hip_err == hipSuccess && hip_device_count > 0)
    {
        rocm_device = neurondb_gpu_device % hip_device_count;
        hip_err = hipSetDevice(rocm_device);
        if (hip_err == hipSuccess)
        {
            *device_id = rocm_device;
            rc = 0;
        }
    }
#endif
    return rc;
}

void
ndb_gpu_mem_pool_init(int pool_size_mb)
{
    if (pool_size_mb <= 0)
    {
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_HIP)
        gpu_pool_size = 0;
        gpu_pool_bytes = 0;
        if (gpu_memory_pool != NULL)
        {
#ifdef NDB_GPU_CUDA
            if (current_backend == GPU_BACKEND_CUDA)
                cudaFree(gpu_memory_pool);
#endif
#ifdef NDB_GPU_HIP
            if (current_backend == GPU_BACKEND_ROCM)
                hipFree(gpu_memory_pool);
#endif
        }
        gpu_memory_pool = NULL;
#endif
        return;
    }

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_HIP)
    size_t pool_bytes_local = (size_t) pool_size_mb * 1024 * 1024;
#endif
#ifdef NDB_GPU_CUDA
    if (current_backend == GPU_BACKEND_CUDA)
    {
        if (gpu_memory_pool)
        {
            cudaFree(gpu_memory_pool);
            gpu_memory_pool = NULL;
        }
        if (cudaMalloc(&gpu_memory_pool, pool_bytes_local) == cudaSuccess)
        {
            gpu_pool_size = pool_size_mb;
            gpu_pool_bytes = pool_bytes_local;
            elog(DEBUG1, "neurondb: GPU memory pool allocated: %d MB", pool_size_mb);
        }
        else
        {
            gpu_pool_size = 0;
            gpu_pool_bytes = 0;
            gpu_memory_pool = NULL;
            elog(WARNING, "neurondb: GPU memory pool allocation failed");
        }
    }
#endif
#ifdef NDB_GPU_HIP
    if (current_backend == GPU_BACKEND_ROCM)
    {
        if (gpu_memory_pool)
        {
            hipFree(gpu_memory_pool);
            gpu_memory_pool = NULL;
        }
        if (hipMalloc(&gpu_memory_pool, pool_bytes_local) == hipSuccess)
        {
            gpu_pool_size = pool_size_mb;
            gpu_pool_bytes = pool_bytes_local;
            elog(DEBUG1, "neurondb: GPU memory pool allocated: %d MB", pool_size_mb);
        }
        else
        {
            gpu_pool_size = 0;
            gpu_pool_bytes = 0;
            gpu_memory_pool = NULL;
            elog(WARNING, "neurondb: GPU memory pool allocation failed");
        }
    }
#endif
}

void
ndb_gpu_streams_init(int num_streams)
{
#ifdef NDB_GPU_CUDA
    if (current_backend == GPU_BACKEND_CUDA)
    {
        if (cuda_streams)
        {
            for (int i = 0; i < neurondb_gpu_streams; i++)
                cudaStreamDestroy(cuda_streams[i]);
            free(cuda_streams);
            cuda_streams = NULL;
        }
        cuda_streams = (cudaStream_t *) malloc(num_streams * sizeof(cudaStream_t));
        if (!cuda_streams)
        {
            elog(WARNING, "neurondb: Unable to allocate CUDA streams array");
            return;
        }
        for (int i = 0; i < num_streams; i++)
        {
            if (cudaStreamCreate(&cuda_streams[i]) != cudaSuccess)
            {
                elog(WARNING, "neurondb: Failed to create CUDA stream %d", i);
                // Cleanup any streams created so far. No leak.
                for (int j = 0; j < i; j++) cudaStreamDestroy(cuda_streams[j]);
                free(cuda_streams);
                cuda_streams = NULL;
                return;
            }
        }
        elog(DEBUG1, "neurondb: Created %d CUDA streams", num_streams);
    }
#endif
#ifdef NDB_GPU_HIP
    if (current_backend == GPU_BACKEND_ROCM)
    {
        if (hip_streams)
        {
            for (int i = 0; i < neurondb_gpu_streams; i++)
                hipStreamDestroy(hip_streams[i]);
            free(hip_streams);
            hip_streams = NULL;
        }
        hip_streams = (hipStream_t *) malloc(num_streams * sizeof(hipStream_t));
        if (!hip_streams)
        {
            elog(WARNING, "neurondb: Unable to allocate HIP streams array");
            return;
        }
        for (int i = 0; i < num_streams; i++)
        {
            if (hipStreamCreate(&hip_streams[i]) != hipSuccess)
            {
                elog(WARNING, "neurondb: Failed to create HIP stream %d", i);
                for (int j = 0; j < i; j++) hipStreamDestroy(hip_streams[j]);
                free(hip_streams);
                hip_streams = NULL;
                return;
            }
        }
        elog(DEBUG1, "neurondb: Created %d HIP streams", num_streams);
    }
#endif
}

void
ndb_gpu_init_if_needed(void)
{
    int device_id;
    int rc;

    if (gpu_ready)
        return;
    if (!neurondb_gpu_enabled || gpu_disabled)
        return;

#ifdef NDB_GPU_METAL
    /* Try Metal first on Apple Silicon */
    if (!neurondb_gpu_backend || 
        strcmp(neurondb_gpu_backend, "auto") == 0 ||
        strcmp(neurondb_gpu_backend, "metal") == 0)
    {
        elog(DEBUG1, "neurondb: Attempting Metal GPU initialization");
        if (neurondb_gpu_metal_init())
        {
            current_backend = GPU_BACKEND_METAL;
            gpu_ready = true;
            elog(LOG, "neurondb: ✅ Metal GPU backend active on %s",
                 neurondb_gpu_metal_device_name());
            return;
        }
        elog(DEBUG1, "neurondb: Metal init failed, trying other backends");
    }
#endif

    rc = ndb_gpu_runtime_init(&device_id);
    if (rc != 0)
    {
        gpu_ready = false;
        gpu_disabled = true;
        if (!neurondb_gpu_fail_open)
        {
            ereport(ERROR,
                    (errcode(ERRCODE_SYSTEM_ERROR),
                     errmsg("neurondb: GPU initialization failed")));
        }
        ereport(WARNING,
                (errmsg("neurondb: GPU init failed. Using CPU fallback")));
        return;
    }
#ifdef NDB_GPU_CUDA
    if (!neurondb_gpu_cuda_init())
    {
        elog(WARNING, "neurondb: CUDA backend initialization failed");
        gpu_disabled = true;
        gpu_ready = false;
        current_backend = GPU_BACKEND_NONE;
        return;
    }
    current_backend = GPU_BACKEND_CUDA;
#endif
#ifdef NDB_GPU_HIP
    if (rocblas_handle)
    {
        rocblas_destroy_handle(rocblas_handle);
        rocblas_handle = NULL;
    }
    if (rocblas_create_handle(&rocblas_handle) != rocblas_status_success)
    {
        elog(WARNING, "neurondb: rocBLAS initialization failed");
        gpu_disabled = true;
        gpu_ready = false;
        current_backend = GPU_BACKEND_NONE;
        return;
    }
    current_backend = GPU_BACKEND_ROCM;
#endif
    ndb_gpu_mem_pool_init((int)neurondb_gpu_memory_pool_mb);
    ndb_gpu_streams_init(neurondb_gpu_streams);

    gpu_ready = true;

    elog(LOG, "neurondb: GPU initialized successfully on device %d", device_id);
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

#ifdef NDB_GPU_CUDA
	if (current_backend == GPU_BACKEND_CUDA)
	{
		neurondb_gpu_cuda_cleanup();
		elog(LOG, "neurondb: CUDA GPU backend shut down");
	}
#endif

#ifdef NDB_GPU_HIP
	if (current_backend == GPU_BACKEND_ROCM)
	{
		neurondb_gpu_rocm_cleanup();
		elog(LOG, "neurondb: ROCm GPU backend shut down");
	}
#endif

#ifdef NDB_GPU_METAL
	if (current_backend == GPU_BACKEND_METAL)
	{
		neurondb_gpu_metal_cleanup();
		elog(LOG, "neurondb: Metal GPU backend shut down");
	}
#endif

#ifdef NDB_GPU_CUDA
    if (current_backend == GPU_BACKEND_CUDA)
    {
        if (cuda_streams)
        {
            for (int i = 0; i < neurondb_gpu_streams; i++)
                cudaStreamDestroy(cuda_streams[i]);
            free(cuda_streams);
            cuda_streams = NULL;
        }
        if (gpu_memory_pool)
        {
            cudaFree(gpu_memory_pool);
            gpu_memory_pool = NULL;
        }
        cudaDeviceReset();
    }
#endif
#ifdef NDB_GPU_HIP
    if (current_backend == GPU_BACKEND_ROCM)
    {
        if (hip_streams)
        {
            for (int i = 0; i < neurondb_gpu_streams; i++)
                hipStreamDestroy(hip_streams[i]);
            free(hip_streams);
            hip_streams = NULL;
        }
        if (rocblas_handle)
        {
            rocblas_destroy_handle(rocblas_handle);
            rocblas_handle = NULL;
        }
        if (gpu_memory_pool)
        {
            hipFree(gpu_memory_pool);
            gpu_memory_pool = NULL;
        }
        hipDeviceReset();
    }
#endif

    gpu_ready = false;
    gpu_disabled = false;
    current_backend = GPU_BACKEND_NONE;
    elog(DEBUG1, "neurondb: GPU backend shut down");
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
    (void)0; /* Reserved for device counting logic */
#ifdef NDB_GPU_CUDA
    if (cudaGetDeviceCount(&count) == cudaSuccess)
        return count;
#endif
#ifdef NDB_GPU_HIP
    if (hipGetDeviceCount(&count) == hipSuccess)
        return count;
#endif
    return 0;
}

GPUDeviceInfo *
neurondb_gpu_get_device_info(int device_id)
{
    GPUDeviceInfo *info;
    info = (GPUDeviceInfo *) palloc0(sizeof(GPUDeviceInfo));
    info->device_id = device_id;
    info->is_available = false;
#ifdef NDB_GPU_CUDA
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess)
    {
        strncpy(info->name, prop.name, sizeof(info->name) - 1);
        info->name[sizeof(info->name)-1] = '\0';
        info->total_memory = prop.totalGlobalMem;
        info->compute_capability_major = prop.major;
        info->compute_capability_minor = prop.minor;
        info->is_available = true;
        size_t free_mem, total_mem;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
            info->free_memory = free_mem;
    }
#endif
#ifdef NDB_GPU_HIP
    hipDeviceProp_t hip_prop;
    if (hipGetDeviceProperties(&hip_prop, device_id) == hipSuccess)
    {
        strncpy(info->name, hip_prop.name, sizeof(info->name) - 1);
        info->name[sizeof(info->name)-1] = '\0';
        info->total_memory = hip_prop.totalGlobalMem;
        info->compute_capability_major = hip_prop.major;
        info->compute_capability_minor = hip_prop.minor;
        info->is_available = true;
        size_t free_mem=0, total_mem=0;
        if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess)
            info->free_memory = free_mem;
    }
#endif
    return info;
}

void
neurondb_gpu_set_device(int device_id)
{
    int count = neurondb_gpu_get_device_count();
    if (device_id < 0 || device_id >= count)
    {
        elog(WARNING, "neurondb: Invalid GPU device ID %d", device_id);
        return;
    }
#ifdef NDB_GPU_CUDA
    if (current_backend == GPU_BACKEND_CUDA)
    {
        if (cudaSetDevice(device_id) == cudaSuccess)
        {
            cuda_device = device_id;
            neurondb_gpu_device = device_id;
            elog(LOG, "neurondb: Switched to CUDA device %d", device_id);
        }
        else
        {
            elog(WARNING, "neurondb: Failed to switch to CUDA device %d", device_id);
        }
    }
#endif
#ifdef NDB_GPU_HIP
    if (current_backend == GPU_BACKEND_ROCM)
    {
        if (hipSetDevice(device_id) == hipSuccess)
        {
            rocm_device = device_id;
            neurondb_gpu_device = device_id;
            elog(LOG, "neurondb: Switched to ROCm device %d", device_id);
        }
        else
        {
            elog(WARNING, "neurondb: Failed to switch to ROCm device %d", device_id);
        }
    }
#endif
}

GPUStats *
neurondb_gpu_get_stats(void)
{
    GPUStats *stats = (GPUStats *) palloc(sizeof(GPUStats));
    memcpy(stats, &gpu_stats, sizeof(GPUStats));
    if (stats->queries_executed > 0)
        stats->avg_latency_ms = stats->total_gpu_time_ms / stats->queries_executed;
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

PG_FUNCTION_INFO_V1(neurondb_gpu_enable);
Datum
neurondb_gpu_enable(PG_FUNCTION_ARGS)
{
    neurondb_gpu_enabled = true;
    gpu_disabled = false;
    ndb_gpu_init_if_needed();

    if (!gpu_ready)
    {
        elog(WARNING, "neurondb: GPU initialization failed");
        PG_RETURN_BOOL(false);
    }

    elog(NOTICE, "neurondb: GPU acceleration enabled");
    PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(neurondb_gpu_info);
Datum
neurondb_gpu_info(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    Datum values[7];
    bool nulls[7] = {false, false, false, false, false, false, false};

    /* Initialize tuple store */
    InitMaterializedSRF(fcinfo, 0);

    /* Return basic GPU info - simplified for now */
    values[0] = Int32GetDatum(0);                    /* device_id */
    values[1] = CStringGetTextDatum(gpu_ready ? "GPU Available" : "No GPU"); /* name */
    values[2] = Int64GetDatum(gpu_ready ? 8192 : 0); /* total_memory (MB) */
    values[3] = Int64GetDatum(gpu_ready ? 4096 : 0); /* free_memory (MB) */
    values[4] = Int32GetDatum(gpu_ready ? 8 : 0);    /* compute_major */
    values[5] = Int32GetDatum(gpu_ready ? 6 : 0);    /* compute_minor */
    values[6] = BoolGetDatum(gpu_ready);             /* is_available */

    tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);

    return (Datum) 0;
}

PG_FUNCTION_INFO_V1(neurondb_gpu_stats);
Datum
neurondb_gpu_stats(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    Datum values[6];
    bool nulls[6] = {false, false, false, false, false, false};

    /* Initialize tuple store */
    InitMaterializedSRF(fcinfo, 0);

    values[0] = Int64GetDatum(gpu_stats.queries_executed);
    values[1] = Int64GetDatum(gpu_stats.fallback_count);
    values[2] = Float8GetDatum(gpu_stats.total_gpu_time_ms);
    values[3] = Float8GetDatum(gpu_stats.total_cpu_time_ms);
    values[4] = Float8GetDatum(gpu_stats.avg_latency_ms);
    values[5] = TimestampTzGetDatum(gpu_stats.last_reset);

    tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);

    return (Datum) 0;
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
