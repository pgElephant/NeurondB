/*
 * gpu_cuda_kernels.cu
 *	  CUDA kernels for GPU-accelerated vector operations
 *
 * IDENTIFICATION
 *	  src/gpu/gpu_cuda_kernels.cu
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

/*
 * CUDA kernel for L2 distance calculation
 * Computes Euclidean distance between two vectors
 */
__global__ void
cuda_l2_distance_kernel(const float *a, const float *b, float *result, int dim)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (tid == 0)
	{
		float sum = 0.0f;
		for (int i = 0; i < dim; i++)
		{
			float diff = a[i] - b[i];
			sum += diff * diff;
		}
		*result = sqrtf(sum);
	}
}

/*
 * CUDA kernel for batch L2 distance calculation
 * Computes distances between queries and targets in parallel
 */
__global__ void
cuda_batch_l2_kernel(const float *queries, const float *targets,
					 float *distances, int nq, int nt, int dim)
{
	int query_idx = blockIdx.y;
	int target_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (query_idx < nq && target_idx < nt)
	{
		float sum = 0.0f;
		const float *q = &queries[query_idx * dim];
		const float *t = &targets[target_idx * dim];
		
		for (int i = 0; i < dim; i++)
		{
			float diff = q[i] - t[i];
			sum += diff * diff;
		}
		
		distances[query_idx * nt + target_idx] = sqrtf(sum);
	}
}

/*
 * CUDA kernel for cosine distance calculation
 */
__global__ void
cuda_cosine_distance_kernel(const float *a, const float *b, float *result, int dim)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (tid == 0)
	{
		float dot = 0.0f;
		float norm_a = 0.0f;
		float norm_b = 0.0f;
		
		for (int i = 0; i < dim; i++)
		{
			dot += a[i] * b[i];
			norm_a += a[i] * a[i];
			norm_b += b[i] * b[i];
		}
		
		norm_a = sqrtf(norm_a);
		norm_b = sqrtf(norm_b);
		
		if (norm_a > 0.0f && norm_b > 0.0f)
			*result = 1.0f - (dot / (norm_a * norm_b));
		else
			*result = 1.0f;
	}
}

/*
 * CUDA kernel for inner product calculation
 */
__global__ void
cuda_inner_product_kernel(const float *a, const float *b, float *result, int dim)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (tid == 0)
	{
		float sum = 0.0f;
		for (int i = 0; i < dim; i++)
		{
			sum += a[i] * b[i];
		}
		*result = sum;
	}
}

/*
 * Host function to launch L2 distance kernel
 */
extern "C" float
cuda_compute_l2_distance(const float *a, const float *b, int dim)
{
	float *d_a, *d_b, *d_result;
	float h_result;
	
	/* Allocate device memory */
	cudaMalloc(&d_a, dim * sizeof(float));
	cudaMalloc(&d_b, dim * sizeof(float));
	cudaMalloc(&d_result, sizeof(float));
	
	/* Copy data to device */
	cudaMemcpy(d_a, a, dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, dim * sizeof(float), cudaMemcpyHostToDevice);
	
	/* Launch kernel */
	cuda_l2_distance_kernel<<<1, 1>>>(d_a, d_b, d_result, dim);
	
	/* Copy result back */
	cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
	
	/* Free device memory */
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
	
	return h_result;
}

/*
 * Host function to launch batch L2 distance kernel
 */
extern "C" void
cuda_compute_batch_l2(const float *queries, const float *targets,
					  float *distances, int nq, int nt, int dim)
{
	float *d_queries, *d_targets, *d_distances;
	
	/* Allocate device memory */
	cudaMalloc(&d_queries, nq * dim * sizeof(float));
	cudaMalloc(&d_targets, nt * dim * sizeof(float));
	cudaMalloc(&d_distances, nq * nt * sizeof(float));
	
	/* Copy data to device */
	cudaMemcpy(d_queries, queries, nq * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_targets, targets, nt * dim * sizeof(float), cudaMemcpyHostToDevice);
	
	/* Configure kernel launch */
	dim3 block(256);
	dim3 grid((nt + block.x - 1) / block.x, nq);
	
	/* Launch kernel */
	cuda_batch_l2_kernel<<<grid, block>>>(d_queries, d_targets, d_distances, nq, nt, dim);
	
	/* Copy result back */
	cudaMemcpy(distances, d_distances, nq * nt * sizeof(float), cudaMemcpyDeviceToHost);
	
	/* Free device memory */
	cudaFree(d_queries);
	cudaFree(d_targets);
	cudaFree(d_distances);
}

/*
 * Host function to launch cosine distance kernel
 */
extern "C" float
cuda_compute_cosine_distance(const float *a, const float *b, int dim)
{
	float *d_a, *d_b, *d_result;
	float h_result;
	
	/* Allocate device memory */
	cudaMalloc(&d_a, dim * sizeof(float));
	cudaMalloc(&d_b, dim * sizeof(float));
	cudaMalloc(&d_result, sizeof(float));
	
	/* Copy data to device */
	cudaMemcpy(d_a, a, dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, dim * sizeof(float), cudaMemcpyHostToDevice);
	
	/* Launch kernel */
	cuda_cosine_distance_kernel<<<1, 1>>>(d_a, d_b, d_result, dim);
	
	/* Copy result back */
	cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
	
	/* Free device memory */
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
	
	return h_result;
}

/*
 * Host function to launch inner product kernel
 */
extern "C" float
cuda_compute_inner_product(const float *a, const float *b, int dim)
{
	float *d_a, *d_b, *d_result;
	float h_result;
	
	/* Allocate device memory */
	cudaMalloc(&d_a, dim * sizeof(float));
	cudaMalloc(&d_b, dim * sizeof(float));
	cudaMalloc(&d_result, sizeof(float));
	
	/* Copy data to device */
	cudaMemcpy(d_a, a, dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, dim * sizeof(float), cudaMemcpyHostToDevice);
	
	/* Launch kernel */
	cuda_inner_product_kernel<<<1, 1>>>(d_a, d_b, d_result, dim);
	
	/* Copy result back */
	cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
	
	/* Free device memory */
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
	
	return h_result;
}

/*
 * Check if CUDA is available
 */
extern "C" int
cuda_check_available(void)
{
	int deviceCount = 0;
	cudaError_t error = cudaGetDeviceCount(&deviceCount);
	
	if (error != cudaSuccess || deviceCount == 0)
		return 0;
	
	return 1;
}

/*
 * Get CUDA device name
 */
extern "C" void
cuda_get_device_name(char *name, int maxlen)
{
	cudaDeviceProp prop;
	int device = 0;
	
	if (cudaGetDeviceProperties(&prop, device) == cudaSuccess)
	{
		snprintf(name, maxlen, "%s", prop.name);
	}
	else
	{
		snprintf(name, maxlen, "CUDA Device (Unknown)");
	}
}

