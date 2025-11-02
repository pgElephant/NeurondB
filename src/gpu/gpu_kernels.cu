/*-------------------------------------------------------------------------
 *
 * gpu_kernels.cu
 *		CUDA/HIP kernels for NeurondB GPU acceleration
 *
 * Implements high-performance distance computation, quantization, and
 * batch operations using CUDA or HIP runtime.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/gpu_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#ifdef NDB_GPU_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#endif

#ifdef NDB_GPU_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

/*
 * L2 distance kernel
 * Computes squared L2 distance between query and each row in matrix
 */
extern "C" __global__
void l2_distance_kernel(const float * __restrict__ query,
                        const float * __restrict__ matrix,
                        float * __restrict__ out,
                        int rows,
                        int dim)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < rows)
	{
		float acc = 0.0f;
		const float *vec = matrix + row * dim;
		
		for (int d = 0; d < dim; d++)
		{
			float diff = query[d] - vec[d];
			acc += diff * diff;
		}
		
		out[row] = sqrtf(acc);
	}
}

/*
 * Cosine distance kernel
 * Computes cosine distance = 1 - (dot_product / (norm_a * norm_b))
 */
extern "C" __global__
void cosine_distance_kernel(const float * __restrict__ query,
                            const float * __restrict__ matrix,
                            float * __restrict__ out,
                            int rows,
                            int dim)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < rows)
	{
		float dot_product = 0.0f;
		float norm_query = 0.0f;
		float norm_vec = 0.0f;
		const float *vec = matrix + row * dim;
		
		for (int d = 0; d < dim; d++)
		{
			float q = query[d];
			float v = vec[d];
			dot_product += q * v;
			norm_query += q * q;
			norm_vec += v * v;
		}
		
		norm_query = sqrtf(norm_query);
		norm_vec = sqrtf(norm_vec);
		
		float cosine_sim = 0.0f;
		if (norm_query > 1e-10f && norm_vec > 1e-10f)
			cosine_sim = dot_product / (norm_query * norm_vec);
		
		out[row] = 1.0f - cosine_sim;
	}
}

/*
 * Inner product kernel
 */
extern "C" __global__
void inner_product_kernel(const float * __restrict__ query,
                          const float * __restrict__ matrix,
                          float * __restrict__ out,
                          int rows,
                          int dim)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < rows)
	{
		float acc = 0.0f;
		const float *vec = matrix + row * dim;
		
		for (int d = 0; d < dim; d++)
			acc += query[d] * vec[d];
		
		out[row] = -acc;  /* Negative for max distance = min inner product */
	}
}

/*
 * FP16 quantization kernel
 */
extern "C" __global__
void quantize_fp16_kernel(const float * __restrict__ input,
                          __half * __restrict__ output,
                          int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < count)
		output[idx] = __float2half(input[idx]);
}

/*
 * INT8 quantization kernel
 */
extern "C" __global__
void quantize_int8_kernel(const float * __restrict__ input,
                          signed char * __restrict__ output,
                          int count,
                          float scale)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < count)
	{
		float val = input[idx] * scale;
		val = fmaxf(-128.0f, fminf(127.0f, val));
		output[idx] = (signed char) val;
	}
}

/*
 * Binary quantization kernel
 */
extern "C" __global__
void quantize_binary_kernel(const float * __restrict__ input,
                            unsigned char * __restrict__ output,
                            int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int byte_count = (count + 7) / 8;
	
	if (idx < byte_count)
	{
		unsigned char byte = 0;
		int start = idx * 8;
		
		for (int i = 0; i < 8 && (start + i) < count; i++)
		{
			if (input[start + i] > 0.0f)
				byte |= (1 << i);
		}
		
		output[idx] = byte;
	}
}

/*
 * KMeans assignment kernel
 * Assigns each vector to nearest centroid
 */
extern "C" __global__
void kmeans_assign_kernel(const float * __restrict__ vectors,
                          const float * __restrict__ centroids,
                          int * __restrict__ assignments,
                          int num_vectors,
                          int num_centroids,
                          int dim)
{
	int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (vec_idx < num_vectors)
	{
		const float *vec = vectors + vec_idx * dim;
		float min_dist = INFINITY;
		int best_centroid = 0;
		
		for (int c = 0; c < num_centroids; c++)
		{
			const float *centroid = centroids + c * dim;
			float dist = 0.0f;
			
			for (int d = 0; d < dim; d++)
			{
				float diff = vec[d] - centroid[d];
				dist += diff * diff;
			}
			
			if (dist < min_dist)
			{
				min_dist = dist;
				best_centroid = c;
			}
		}
		
		assignments[vec_idx] = best_centroid;
	}
}

/*
 * KMeans update kernel
 * Updates centroids based on assignments
 */
extern "C" __global__
void kmeans_update_kernel(const float * __restrict__ vectors,
                          const int * __restrict__ assignments,
                          float * __restrict__ centroids,
                          int * __restrict__ counts,
                          int num_vectors,
                          int num_centroids,
                          int dim)
{
	int c = blockIdx.x;  /* One block per centroid */
	int d = threadIdx.x; /* One thread per dimension */
	
	if (c < num_centroids && d < dim)
	{
		float sum = 0.0f;
		int count = 0;
		
		for (int v = 0; v < num_vectors; v++)
		{
			if (assignments[v] == c)
			{
				sum += vectors[v * dim + d];
				if (d == 0)
					count++;
			}
		}
		
		if (count > 0)
		{
			centroids[c * dim + d] = sum / count;
			if (d == 0)
				counts[c] = count;
		}
	}
}

