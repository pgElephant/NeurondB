/*-------------------------------------------------------------------------
 *
 * gpu_batch.c
 *		GPU-accelerated batch distance operations
 *
 * Computes distance matrices between query vectors and database vectors
 * efficiently using GPU batch operations (cuBLAS/rocBLAS matrix operations).
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/gpu_batch.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"

#include "neurondb_config.h"
#include "neurondb_gpu.h"

#ifdef NDB_GPU_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern cublasHandle_t cublas_handle;
#endif

#ifdef NDB_GPU_HIP
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

extern rocblas_handle rocblas_handle;
#endif

/*
 * GPU batch L2 distance
 * Computes distances between num_queries query vectors and num_vectors database vectors
 */
void
neurondb_gpu_batch_l2_distance(const float *queries, const float *vectors,
							   float *results, int num_queries, int num_vectors, int dim)
{
	if (!neurondb_gpu_is_available())
		return;  /* CPU fallback handled by caller */

#ifdef NDB_GPU_CUDA
	if (neurondb_gpu_get_backend() == GPU_BACKEND_CUDA && cublas_handle)
	{
		float *d_queries, *d_vectors, *d_results, *d_squared_queries, *d_squared_vectors;
		size_t queries_size = num_queries * dim * sizeof(float);
		size_t vectors_size = num_vectors * dim * sizeof(float);
		size_t results_size = num_queries * num_vectors * sizeof(float);
		
		cudaMalloc(&d_queries, queries_size);
		cudaMalloc(&d_vectors, vectors_size);
		cudaMalloc(&d_results, results_size);
		cudaMalloc(&d_squared_queries, num_queries * sizeof(float));
		cudaMalloc(&d_squared_vectors, num_vectors * sizeof(float));
		
		/* Copy to GPU */
		cudaMemcpy(d_queries, queries, queries_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_vectors, vectors, vectors_size, cudaMemcpyHostToDevice);
		
		/* Compute ||q||^2 for each query vector */
		/* This would require a custom kernel or multiple calls to snrm2 */
		/* For simplicity, we use GEMM: results = queries * vectors^T */
		/* Then adjust: L2^2 = ||q||^2 + ||v||^2 - 2*q·v */
		
		/* Matrix multiplication: C = queries * vectors^T */
		/* This gives us dot products between all query-vector pairs */
		const float alpha = -2.0f, beta = 0.0f;
		cublasSgemm(cublas_handle,
				   CUBLAS_OP_N, CUBLAS_OP_T,
				   num_queries, num_vectors, dim,
				   &alpha,
				   d_queries, num_queries,
				   d_vectors, num_vectors,
				   &beta,
				   d_results, num_queries);
		
		/* TODO: Add ||q||^2 and ||v||^2 vectors, then sqrt */
		/* For now, this computes -2*q·v which can be adjusted by caller */
		
		/* Copy results back */
		cudaMemcpy(results, d_results, results_size, cudaMemcpyDeviceToHost);
		
		cudaFree(d_queries);
		cudaFree(d_vectors);
		cudaFree(d_results);
		cudaFree(d_squared_queries);
		cudaFree(d_squared_vectors);
		
		return;
	}
#endif

#ifdef NDB_GPU_HIP
	if (neurondb_gpu_get_backend() == GPU_BACKEND_ROCM && rocblas_handle)
	{
		float *d_queries, *d_vectors, *d_results;
		size_t queries_size = num_queries * dim * sizeof(float);
		size_t vectors_size = num_vectors * dim * sizeof(float);
		size_t results_size = num_queries * num_vectors * sizeof(float);
		
		hipMalloc(&d_queries, queries_size);
		hipMalloc(&d_vectors, vectors_size);
		hipMalloc(&d_results, results_size);
		
		hipMemcpy(d_queries, queries, queries_size, hipMemcpyHostToDevice);
		hipMemcpy(d_vectors, vectors, vectors_size, hipMemcpyHostToDevice);
		
		const float alpha = -2.0f, beta = 0.0f;
		rocblas_sgemm(rocblas_handle,
					  rocblas_operation_none, rocblas_operation_transpose,
					  num_queries, num_vectors, dim,
					  &alpha,
					  d_queries, num_queries,
					  d_vectors, num_vectors,
					  &beta,
					  d_results, num_queries);
		
		hipMemcpy(results, d_results, results_size, hipMemcpyDeviceToHost);
		
		hipFree(d_queries);
		hipFree(d_vectors);
		hipFree(d_results);
		
		return;
	}
#endif

	/* CPU fallback handled by caller */
}

/*
 * GPU batch cosine distance
 */
void
neurondb_gpu_batch_cosine_distance(const float *queries, const float *vectors,
								   float *results, int num_queries, int num_vectors, int dim)
{
	if (!neurondb_gpu_is_available())
		return;

#ifdef NDB_GPU_CUDA
	if (neurondb_gpu_get_backend() == GPU_BACKEND_CUDA && cublas_handle)
	{
		float *d_queries, *d_vectors, *d_results, *d_norms_q, *d_norms_v;
		size_t queries_size = num_queries * dim * sizeof(float);
		size_t vectors_size = num_vectors * dim * sizeof(float);
		size_t results_size = num_queries * num_vectors * sizeof(float);
		
		cudaMalloc(&d_queries, queries_size);
		cudaMalloc(&d_vectors, vectors_size);
		cudaMalloc(&d_results, results_size);
		cudaMalloc(&d_norms_q, num_queries * sizeof(float));
		cudaMalloc(&d_norms_v, num_vectors * sizeof(float));
		
		cudaMemcpy(d_queries, queries, queries_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_vectors, vectors, vectors_size, cudaMemcpyHostToDevice);
		
		/* Normalize query vectors */
		/* Compute norms and divide */
		for (int i = 0; i < num_queries; i++)
		{
			cublasSnrm2(cublas_handle, dim, d_queries + i * dim, 1, d_norms_q + i);
			float norm;
			cudaMemcpy(&norm, d_norms_q + i, sizeof(float), cudaMemcpyDeviceToHost);
			if (norm > 0)
			{
				float inv_norm = 1.0f / norm;
				cublasSscal(cublas_handle, dim, &inv_norm, d_queries + i * dim, 1);
			}
		}
		
		/* Normalize vector database */
		for (int i = 0; i < num_vectors; i++)
		{
			cublasSnrm2(cublas_handle, dim, d_vectors + i * dim, 1, d_norms_v + i);
			float norm;
			cudaMemcpy(&norm, d_norms_v + i, sizeof(float), cudaMemcpyDeviceToHost);
			if (norm > 0)
			{
				float inv_norm = 1.0f / norm;
				cublasSscal(cublas_handle, dim, &inv_norm, d_vectors + i * dim, 1);
			}
		}
		
		/* Compute cosine similarity: queries * vectors^T */
		const float alpha = 1.0f, beta = 0.0f;
		cublasSgemm(cublas_handle,
				   CUBLAS_OP_N, CUBLAS_OP_T,
				   num_queries, num_vectors, dim,
				   &alpha,
				   d_queries, num_queries,
				   d_vectors, num_vectors,
				   &beta,
				   d_results, num_queries);
		
		/* Convert similarity to distance: 1 - similarity */
		/* Would need a custom kernel for element-wise subtraction */
		
		cudaMemcpy(results, d_results, results_size, cudaMemcpyDeviceToHost);
		
		/* Convert to distance on CPU (1 - similarity) */
		for (int i = 0; i < num_queries * num_vectors; i++)
			results[i] = 1.0f - results[i];
		
		cudaFree(d_queries);
		cudaFree(d_vectors);
		cudaFree(d_results);
		cudaFree(d_norms_q);
		cudaFree(d_norms_v);
		
		return;
	}
#endif

#ifdef NDB_GPU_HIP
	if (neurondb_gpu_get_backend() == GPU_BACKEND_ROCM && rocblas_handle)
	{
		/* Similar implementation for ROCm */
		float *d_queries, *d_vectors, *d_results;
		size_t queries_size = num_queries * dim * sizeof(float);
		size_t vectors_size = num_vectors * dim * sizeof(float);
		size_t results_size = num_queries * num_vectors * sizeof(float);
		
		hipMalloc(&d_queries, queries_size);
		hipMalloc(&d_vectors, vectors_size);
		hipMalloc(&d_results, results_size);
		
		hipMemcpy(d_queries, queries, queries_size, hipMemcpyHostToDevice);
		hipMemcpy(d_vectors, vectors, vectors_size, hipMemcpyHostToDevice);
		
		const float alpha = 1.0f, beta = 0.0f;
		rocblas_sgemm(rocblas_handle,
					  rocblas_operation_none, rocblas_operation_transpose,
					  num_queries, num_vectors, dim,
					  &alpha,
					  d_queries, num_queries,
					  d_vectors, num_vectors,
					  &beta,
					  d_results, num_queries);
		
		hipMemcpy(results, d_results, results_size, hipMemcpyDeviceToHost);
		
		/* Convert to distance */
		for (int i = 0; i < num_queries * num_vectors; i++)
			results[i] = 1.0f - results[i];
		
		hipFree(d_queries);
		hipFree(d_vectors);
		hipFree(d_results);
		
		return;
	}
#endif

	/* CPU fallback handled by caller */
}

