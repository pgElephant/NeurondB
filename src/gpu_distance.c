/*-------------------------------------------------------------------------
 *
 * gpu_distance.c
 *		GPU-accelerated distance operations for NeurondB
 *
 * Implements L2, cosine, and inner product distance metrics using
 * CUDA cuBLAS or ROCm rocBLAS for high-performance vector operations.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/gpu_distance.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/timestamp.h"
#include "neurondb_gpu.h"
#include <stdio.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
extern cublasHandle_t cublas_handle;
extern int cuda_device;
#endif

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
extern rocblas_handle rocblas_handle;
extern int rocm_device;
#endif

#ifdef HAVE_CUDA
/* CUDA kernel for vector subtraction: out = a - b */
__global__ void vec_subtract_kernel(const float *a, const float *b, float *out, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		out[idx] = a[idx] - b[idx];
}
#endif

#ifdef HAVE_ROCM
/* HIP kernel for vector subtraction: out = a - b */
__global__ void hip_vec_subtract_kernel(const float *a, const float *b, float *out, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		out[idx] = a[idx] - b[idx];
}
#endif

/*
 * GPU L2 distance (Euclidean)
 */
float
neurondb_gpu_l2_distance(const float *vec1, const float *vec2, int dim)
{
	float result = 0.0f;

	if (!neurondb_gpu_is_available())
		return -1.0f;  /* Signal CPU fallback */

#ifdef HAVE_CUDA
	if (neurondb_gpu_get_backend() == GPU_BACKEND_CUDA && cublas_handle)
	{
		float *d_vec1 = NULL, *d_vec2 = NULL, *d_diff = NULL;
		float h_result = 0.0f;
		size_t size = dim * sizeof(float);

		cudaError_t cudaStatus;
		cudaStatus = cudaMalloc((void**)&d_vec1, size);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;
		cudaStatus = cudaMalloc((void**)&d_vec2, size);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;
		cudaStatus = cudaMalloc((void**)&d_diff, size);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;

		cudaStatus = cudaMemcpy(d_vec1, vec1, size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;
		cudaStatus = cudaMemcpy(d_vec2, vec2, size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;

		// Launch kernel to compute diff = vec1 - vec2
		int block = 256;
		int grid = (dim + block - 1) / block;
		vec_subtract_kernel<<<grid, block>>>(d_vec1, d_vec2, d_diff, dim);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;

		// Use cublas for L2 norm of difference
		cublasStatus_t cublasStatus;
		cublasStatus = cublasSnrm2(cublas_handle, dim, d_diff, 1, &h_result);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS) goto cleanup_cuda;

		result = h_result;

	cleanup_cuda:
		if (d_vec1) cudaFree(d_vec1);
		if (d_vec2) cudaFree(d_vec2);
		if (d_diff) cudaFree(d_diff);
		return result;
	}
#endif

#ifdef HAVE_ROCM
	if (neurondb_gpu_get_backend() == GPU_BACKEND_ROCM && rocblas_handle)
	{
		float *d_vec1 = NULL, *d_vec2 = NULL, *d_diff = NULL;
		float h_result = 0.0f;
		size_t size = dim * sizeof(float);

		hipError_t hipStatus;
		hipStatus = hipMalloc((void**)&d_vec1, size);
		if (hipStatus != hipSuccess) goto cleanup_hip;
		hipStatus = hipMalloc((void**)&d_vec2, size);
		if (hipStatus != hipSuccess) goto cleanup_hip;
		hipStatus = hipMalloc((void**)&d_diff, size);
		if (hipStatus != hipSuccess) goto cleanup_hip;

		hipStatus = hipMemcpy(d_vec1, vec1, size, hipMemcpyHostToDevice);
		if (hipStatus != hipSuccess) goto cleanup_hip;
		hipStatus = hipMemcpy(d_vec2, vec2, size, hipMemcpyHostToDevice);
		if (hipStatus != hipSuccess) goto cleanup_hip;

		// Launch kernel to compute diff = vec1 - vec2
		int block = 256;
		int grid = (dim + block - 1) / block;
		hipLaunchKernelGGL(hip_vec_subtract_kernel, grid, block, 0, 0, d_vec1, d_vec2, d_diff, dim);
		hipStatus = hipDeviceSynchronize();
		if (hipStatus != hipSuccess) goto cleanup_hip;

		rocblas_status rstatus;
		rstatus = rocblas_snrm2(rocblas_handle, dim, d_diff, 1, &h_result);
		if (rstatus != rocblas_status_success) goto cleanup_hip;

		result = h_result;

	cleanup_hip:
		if (d_vec1) hipFree(d_vec1);
		if (d_vec2) hipFree(d_vec2);
		if (d_diff) hipFree(d_diff);
		return result;
	}
#endif

	return -1.0f;  /* CPU fallback */
}

/*
 * GPU cosine distance
 */
float
neurondb_gpu_cosine_distance(const float *vec1, const float *vec2, int dim)
{
	float result = 0.0f;

	if (!neurondb_gpu_is_available())
		return -1.0f;

#ifdef HAVE_CUDA
	if (neurondb_gpu_get_backend() == GPU_BACKEND_CUDA && cublas_handle)
	{
		float *d_vec1 = NULL, *d_vec2 = NULL;
		size_t size = dim * sizeof(float);
		float h_dot = 0.0f, h_norm1 = 0.0f, h_norm2 = 0.0f;

		cudaError_t cudaStatus;
		cudaStatus = cudaMalloc((void**)&d_vec1, size);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;
		cudaStatus = cudaMalloc((void**)&d_vec2, size);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;

		cudaStatus = cudaMemcpy(d_vec1, vec1, size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;
		cudaStatus = cudaMemcpy(d_vec2, vec2, size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;

		// Compute dot product
		cublasStatus_t cublasStatus;
		cublasStatus = cublasSdot(cublas_handle, dim, d_vec1, 1, d_vec2, 1, &h_dot);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS) goto cleanup_cuda;

		// Compute norms
		cublasStatus = cublasSnrm2(cublas_handle, dim, d_vec1, 1, &h_norm1);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS) goto cleanup_cuda;
		cublasStatus = cublasSnrm2(cublas_handle, dim, d_vec2, 1, &h_norm2);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS) goto cleanup_cuda;

		// Calculate cosine distance
		if (h_norm1 > 0 && h_norm2 > 0)
			result = 1.0f - (h_dot / (h_norm1 * h_norm2));
		else
			result = 1.0f;

	cleanup_cuda:
		if (d_vec1) cudaFree(d_vec1);
		if (d_vec2) cudaFree(d_vec2);
		return result;
	}
#endif

#ifdef HAVE_ROCM
	if (neurondb_gpu_get_backend() == GPU_BACKEND_ROCM && rocblas_handle)
	{
		float *d_vec1 = NULL, *d_vec2 = NULL;
		size_t size = dim * sizeof(float);
		float h_dot = 0.0f, h_norm1 = 0.0f, h_norm2 = 0.0f;

		hipError_t hipStatus;
		hipStatus = hipMalloc((void**)&d_vec1, size);
		if (hipStatus != hipSuccess) goto cleanup_hip;
		hipStatus = hipMalloc((void**)&d_vec2, size);
		if (hipStatus != hipSuccess) goto cleanup_hip;

		hipStatus = hipMemcpy(d_vec1, vec1, size, hipMemcpyHostToDevice);
		if (hipStatus != hipSuccess) goto cleanup_hip;
		hipStatus = hipMemcpy(d_vec2, vec2, size, hipMemcpyHostToDevice);
		if (hipStatus != hipSuccess) goto cleanup_hip;

		rocblas_status rstatus;
		rstatus = rocblas_sdot(rocblas_handle, dim, d_vec1, 1, d_vec2, 1, &h_dot);
		if (rstatus != rocblas_status_success) goto cleanup_hip;
		rstatus = rocblas_snrm2(rocblas_handle, dim, d_vec1, 1, &h_norm1);
		if (rstatus != rocblas_status_success) goto cleanup_hip;
		rstatus = rocblas_snrm2(rocblas_handle, dim, d_vec2, 1, &h_norm2);
		if (rstatus != rocblas_status_success) goto cleanup_hip;

		if (h_norm1 > 0 && h_norm2 > 0)
			result = 1.0f - (h_dot / (h_norm1 * h_norm2));
		else
			result = 1.0f;

	cleanup_hip:
		if (d_vec1) hipFree(d_vec1);
		if (d_vec2) hipFree(d_vec2);
		return result;
	}
#endif

	return -1.0f;  /* CPU fallback */
}

/*
 * GPU inner product
 */
float
neurondb_gpu_inner_product(const float *vec1, const float *vec2, int dim)
{
	float result = 0.0f;

	if (!neurondb_gpu_is_available())
		return -1.0f;

#ifdef HAVE_CUDA
	if (neurondb_gpu_get_backend() == GPU_BACKEND_CUDA && cublas_handle)
	{
		float *d_vec1 = NULL, *d_vec2 = NULL;
		size_t size = dim * sizeof(float);
		float temp_result = 0.0f;

		cudaError_t cudaStatus;
		cudaStatus = cudaMalloc((void**)&d_vec1, size);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;
		cudaStatus = cudaMalloc((void**)&d_vec2, size);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;

		cudaStatus = cudaMemcpy(d_vec1, vec1, size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;
		cudaStatus = cudaMemcpy(d_vec2, vec2, size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) goto cleanup_cuda;

		// Dot product via cublas
		cublasStatus_t cublasStatus;
		cublasStatus = cublasSdot(cublas_handle, dim, d_vec1, 1, d_vec2, 1, &temp_result);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS) goto cleanup_cuda;

		result = temp_result;

	cleanup_cuda:
		if (d_vec1) cudaFree(d_vec1);
		if (d_vec2) cudaFree(d_vec2);
		return result;
	}
#endif

#ifdef HAVE_ROCM
	if (neurondb_gpu_get_backend() == GPU_BACKEND_ROCM && rocblas_handle)
	{
		float *d_vec1 = NULL, *d_vec2 = NULL;
		size_t size = dim * sizeof(float);
		float temp_result = 0.0f;

		hipError_t hipStatus;
		hipStatus = hipMalloc((void**)&d_vec1, size);
		if (hipStatus != hipSuccess) goto cleanup_hip;
		hipStatus = hipMalloc((void**)&d_vec2, size);
		if (hipStatus != hipSuccess) goto cleanup_hip;

		hipStatus = hipMemcpy(d_vec1, vec1, size, hipMemcpyHostToDevice);
		if (hipStatus != hipSuccess) goto cleanup_hip;
		hipStatus = hipMemcpy(d_vec2, vec2, size, hipMemcpyHostToDevice);
		if (hipStatus != hipSuccess) goto cleanup_hip;

		rocblas_status rstatus;
		rstatus = rocblas_sdot(rocblas_handle, dim, d_vec1, 1, d_vec2, 1, &temp_result);
		if (rstatus != rocblas_status_success) goto cleanup_hip;

		result = temp_result;

	cleanup_hip:
		if (d_vec1) hipFree(d_vec1);
		if (d_vec2) hipFree(d_vec2);
		return result;
	}
#endif

	return -1.0f;  /* CPU fallback */
}
