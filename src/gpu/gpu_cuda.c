/*
 * gpu_cuda.c
 *     NVIDIA CUDA GPU backend for NeurondB
 *
 * Implements CUDA-accelerated vector operations for NVIDIA GPUs.
 * Supports CUDA 11.0+ with compute capability 6.0+.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/elog.h"

#include "neurondb_config.h"
#include "neurondb_gpu.h"

#ifdef NDB_GPU_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

/* CUDA resources */
static bool cuda_initialized = false;
static int cuda_device_id = 0;
static cublasHandle_t cublas_handle = NULL;
static cudaStream_t cuda_stream = NULL;
static uint64_t total_cuda_ops = 0;

/*
 * Initialize CUDA backend
 */
bool
neurondb_gpu_cuda_init(void)
{
	cudaError_t cuda_err;
	cublasStatus_t cublas_status;
	int device_count = 0;
	
	if (cuda_initialized)
		return true;
	
	/* Check for CUDA devices */
	cuda_err = cudaGetDeviceCount(&device_count);
	if (cuda_err != cudaSuccess || device_count == 0)
	{
		elog(WARNING, "neurondb: No CUDA devices found: %s", 
			 cudaGetErrorString(cuda_err));
		return false;
	}
	
	/* Set device */
	cuda_err = cudaSetDevice(cuda_device_id);
	if (cuda_err != cudaSuccess)
	{
		elog(WARNING, "neurondb: Failed to set CUDA device %d: %s",
			 cuda_device_id, cudaGetErrorString(cuda_err));
		return false;
	}
	
	/* Get device properties */
	cudaDeviceProp prop;
	cuda_err = cudaGetDeviceProperties(&prop, cuda_device_id);
	if (cuda_err == cudaSuccess)
	{
		elog(LOG, "neurondb: ✅ CUDA GPU: %s", prop.name);
		elog(LOG, "neurondb: ✅ Compute capability: %d.%d", 
			 prop.major, prop.minor);
		elog(LOG, "neurondb: ✅ GPU memory: %.2f GB",
			 prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
		elog(LOG, "neurondb: ✅ CUDA cores: %d multiprocessors",
			 prop.multiProcessorCount);
	}
	
	/* Create cuBLAS handle */
	cublas_status = cublasCreate(&cublas_handle);
	if (cublas_status != CUBLAS_STATUS_SUCCESS)
	{
		elog(WARNING, "neurondb: Failed to create cuBLAS handle: %d",
			 cublas_status);
		return false;
	}
	
	/* Create CUDA stream */
	cuda_err = cudaStreamCreate(&cuda_stream);
	if (cuda_err != cudaSuccess)
	{
		cublasDestroy(cublas_handle);
		elog(WARNING, "neurondb: Failed to create CUDA stream: %s",
			 cudaGetErrorString(cuda_err));
		return false;
	}
	
	/* Set stream for cuBLAS */
	cublasSetStream(cublas_handle, cuda_stream);
	
	cuda_initialized = true;
	total_cuda_ops = 0;
	
	elog(LOG, "neurondb: ✅ CUDA backend initialized successfully");
	
	return true;
}

/*
 * Cleanup CUDA resources
 */
void
neurondb_gpu_cuda_cleanup(void)
{
	if (!cuda_initialized)
		return;
	
	if (cuda_stream)
	{
		cudaStreamDestroy(cuda_stream);
		cuda_stream = NULL;
	}
	
	if (cublas_handle)
	{
		cublasDestroy(cublas_handle);
		cublas_handle = NULL;
	}
	
	cuda_initialized = false;
	
	if (total_cuda_ops > 0)
		elog(LOG, "neurondb: CUDA cleanup: %llu GPU operations performed",
			 (unsigned long long)total_cuda_ops);
}

/*
 * Check if CUDA is available
 */
bool
neurondb_gpu_cuda_is_available(void)
{
	return cuda_initialized;
}

/*
 * Get CUDA device name
 */
const char *
neurondb_gpu_cuda_device_name(void)
{
	static char device_name[256] = "None";
	cudaDeviceProp prop;
	
	if (!cuda_initialized)
		return device_name;
	
	if (cudaGetDeviceProperties(&prop, cuda_device_id) == cudaSuccess)
		snprintf(device_name, sizeof(device_name), "%s", prop.name);
	
	return device_name;
}

/*
 * L2 distance using CUDA
 */
float
neurondb_gpu_cuda_l2_distance(const float *a, const float *b, int dim)
{
	cudaError_t cuda_err;
	float *d_a = NULL, *d_b = NULL, *d_diff = NULL;
	float result = 0.0f;
	float alpha = -1.0f;
	size_t size = dim * sizeof(float);
	
	if (!cuda_initialized || dim < 64)
		return -1.0f;
	
	/* Allocate device memory */
	cuda_err = cudaMalloc(&d_a, size);
	if (cuda_err != cudaSuccess) return -1.0f;
	
	cuda_err = cudaMalloc(&d_b, size);
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_a);
		return -1.0f;
	}
	
	cuda_err = cudaMalloc(&d_diff, size);
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_a);
		cudaFree(d_b);
		return -1.0f;
	}
	
	/* Copy data to device */
	cudaMemcpyAsync(d_a, a, size, cudaMemcpyHostToDevice, cuda_stream);
	cudaMemcpyAsync(d_b, b, size, cudaMemcpyHostToDevice, cuda_stream);
	cudaMemcpyAsync(d_diff, a, size, cudaMemcpyHostToDevice, cuda_stream);
	
	/* Compute difference: d_diff = d_a - d_b */
	cublasSaxpy(cublas_handle, dim, &alpha, d_b, 1, d_diff, 1);
	
	/* Compute dot product: result = d_diff · d_diff */
	cublasSnrm2(cublas_handle, dim, d_diff, 1, &result);
	
	/* Wait for completion */
	cudaStreamSynchronize(cuda_stream);
	
	/* Free device memory */
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_diff);
	
	total_cuda_ops++;
	return result;
}

/*
 * Cosine distance using CUDA
 */
float
neurondb_gpu_cuda_cosine_distance(const float *a, const float *b, int dim)
{
	float *d_a = NULL, *d_b = NULL;
	float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
	float similarity, result;
	size_t size = dim * sizeof(float);
	
	if (!cuda_initialized || dim < 64)
		return -1.0f;
	
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	
	if (!d_a || !d_b)
	{
		if (d_a) cudaFree(d_a);
		if (d_b) cudaFree(d_b);
		return -1.0f;
	}
	
	cudaMemcpyAsync(d_a, a, size, cudaMemcpyHostToDevice, cuda_stream);
	cudaMemcpyAsync(d_b, b, size, cudaMemcpyHostToDevice, cuda_stream);
	
	/* Compute dot product and norms using cuBLAS */
	cublasSdot(cublas_handle, dim, d_a, 1, d_b, 1, &dot);
	cublasSnrm2(cublas_handle, dim, d_a, 1, &norm_a);
	cublasSnrm2(cublas_handle, dim, d_b, 1, &norm_b);
	
	cudaStreamSynchronize(cuda_stream);
	
	cudaFree(d_a);
	cudaFree(d_b);
	
	similarity = dot / (norm_a * norm_b);
	result = 1.0f - similarity;
	
	total_cuda_ops++;
	return result;
}

/*
 * Inner product using CUDA
 */
float
neurondb_gpu_cuda_inner_product(const float *a, const float *b, int dim)
{
	float *d_a = NULL, *d_b = NULL;
	float result = 0.0f;
	size_t size = dim * sizeof(float);
	
	if (!cuda_initialized || dim < 64)
		return -1.0f;
	
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	
	if (!d_a || !d_b)
	{
		if (d_a) cudaFree(d_a);
		if (d_b) cudaFree(d_b);
		return -1.0f;
	}
	
	cudaMemcpyAsync(d_a, a, size, cudaMemcpyHostToDevice, cuda_stream);
	cudaMemcpyAsync(d_b, b, size, cudaMemcpyHostToDevice, cuda_stream);
	
	cublasSdot(cublas_handle, dim, d_a, 1, d_b, 1, &result);
	
	cudaStreamSynchronize(cuda_stream);
	
	cudaFree(d_a);
	cudaFree(d_b);
	
	total_cuda_ops++;
	return -result;
}

/*
 * Batch L2 distance using CUDA
 */
void
neurondb_gpu_cuda_batch_l2(const float *queries, const float *targets,
							int num_queries, int num_targets, int dim,
							float *distances)
{
	/* Implementation uses CUDA kernel for parallel computation */
	/* For now, fall back to individual calculations */
	for (int i = 0; i < num_queries; i++)
	{
		for (int j = 0; j < num_targets; j++)
		{
			float dist = neurondb_gpu_cuda_l2_distance(
				queries + i * dim,
				targets + j * dim,
				dim
			);
			if (dist >= 0.0f)
				distances[i * num_targets + j] = dist;
		}
	}
}

/* Statistics */
uint64_t neurondb_gpu_cuda_get_operations_count(void) { return total_cuda_ops; }

#else /* !NDB_GPU_CUDA */

/* Stub functions when CUDA is not available */
bool neurondb_gpu_cuda_init(void) { return false; }
void neurondb_gpu_cuda_cleanup(void) { }
bool neurondb_gpu_cuda_is_available(void) { return false; }
const char *neurondb_gpu_cuda_device_name(void) { return "Not compiled"; }
float neurondb_gpu_cuda_l2_distance(const float *a, const float *b, int dim) { (void)a; (void)b; (void)dim; return -1.0f; }
float neurondb_gpu_cuda_cosine_distance(const float *a, const float *b, int dim) { (void)a; (void)b; (void)dim; return -1.0f; }
float neurondb_gpu_cuda_inner_product(const float *a, const float *b, int dim) { (void)a; (void)b; (void)dim; return -1.0f; }
void neurondb_gpu_cuda_batch_l2(const float *q, const float *t, int nq, int nt, int d, float *dist) { (void)q; (void)t; (void)nq; (void)nt; (void)d; (void)dist; }
uint64_t neurondb_gpu_cuda_get_operations_count(void) { return 0; }

#endif /* NDB_GPU_CUDA */

