/*-------------------------------------------------------------------------
 *
 * gpu_kernels.cu
 *    CUDA/HIP kernels for NeurondB GPU acceleration
 *
 *  This file implements GPU-accelerated fundamental computations such as
 *  vector distances (L2, cosine, inner product), quantization operations
 *  (float32 → float16, int8, and binary), and KMeans clustering steps,
 *  adhering strictly to PostgreSQL code style and robustness conventions.
 *  Importantly, this file is written according to PostgreSQL C style,
 *  maximizing clarity, predictability, and maintainability.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/gpu_kernels.cu
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

#include <float.h>
#include <math.h>
#include "neurondb_cuda_launchers.h"

/*
 * Device kernel to compute L2 (Euclidean) distance between query and each vector row.
 */
extern "C" __global__
void
l2_distance_kernel(const float * __restrict__ query,
				   const float * __restrict__ matrix,
				   float * __restrict__ out,
				   int rows,
				   int dim)
{
	int			row;
	const float *vec;
	double		acc;
	int			d;

	row = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (row < rows)
	{
		vec = matrix + (row * dim);
		acc = 0.0;
		for (d = 0; d < dim; d++)
		{
			float	diff;

			diff = query[d] - vec[d];
			acc += (double) diff * (double) diff;
		}
		out[row] = sqrtf((float) acc);
	}
}

/*
 * Device kernel to compute cosine distance:
 * d(query, vec) = 1 - dot(query, vec) / (||query|| * ||vec||)
 */
extern "C" __global__
void
cosine_distance_kernel(const float * __restrict__ query,
					  const float * __restrict__ matrix,
					  float * __restrict__ out,
					  int rows,
					  int dim)
{
	int			row;
	const float *vec;
	double		dot, n_q, n_v;
	int			d;
	float		cosine_sim;

	row = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (row < rows)
	{
		vec = matrix + (row * dim);
		dot = 0.0;
		n_q = 0.0;
		n_v = 0.0;

		/* Loop unrolling in strict C style (4-way) */
		d = 0;
		for (; d <= dim - 4; d += 4)
		{
			float q0 = query[d],     v0 = vec[d];
			float q1 = query[d+1],   v1 = vec[d+1];
			float q2 = query[d+2],   v2 = vec[d+2];
			float q3 = query[d+3],   v3 = vec[d+3];

			dot +=  (double)q0 * (double)v0 +
					(double)q1 * (double)v1 +
					(double)q2 * (double)v2 +
					(double)q3 * (double)v3;

			n_q +=  (double)q0 * (double)q0 +
					(double)q1 * (double)q1 +
					(double)q2 * (double)q2 +
					(double)q3 * (double)q3;

			n_v +=  (double)v0 * (double)v0 +
					(double)v1 * (double)v1 +
					(double)v2 * (double)v2 +
					(double)v3 * (double)v3;
		}
		for (; d < dim; d++)
		{
			float q = query[d];
			float v = vec[d];

			dot += (double)q * (double)v;
			n_q += (double)q * (double)q;
			n_v += (double)v * (double)v;
		}

		/* Convert to norms (magnitude) */
		n_q = sqrt(n_q);
		n_v = sqrt(n_v);

		cosine_sim = 0.0f;
		if (n_q > 1e-10 && n_v > 1e-10)
			cosine_sim = (float) (dot / (n_q * n_v));

		/* Clamp to [-1,1] to avoid floating point error propagation */
		if (cosine_sim < -1.0f)
			cosine_sim = -1.0f;
		else if (cosine_sim > 1.0f)
			cosine_sim = 1.0f;

		out[row] = 1.0f - cosine_sim;
	}
}

/*
 * Device kernel to compute negative inner product for ANN distance.
 */
extern "C" __global__
void
inner_product_kernel(const float * __restrict__ query,
					 const float * __restrict__ matrix,
					 float * __restrict__ out,
					 int rows,
					 int dim)
{
	int			row;
	const float *vec;
	double		acc;
	int			d;

	row = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (row < rows)
	{
		vec = matrix + (row * dim);
		acc = 0.0;
		d = 0;
		for (; d <= dim - 4; d += 4)
		{
			acc += (double)query[d]   * (double)vec[d]   +
				   (double)query[d+1] * (double)vec[d+1] +
				   (double)query[d+2] * (double)vec[d+2] +
				   (double)query[d+3] * (double)vec[d+3];
		}
		for (; d < dim; d++)
			acc += (double)query[d] * (double)vec[d];

		out[row] = (float) (-acc);
	}
}

/*
 * Device kernel for quantizing float32 to float16.
 */
extern "C" __global__
void
quantize_fp16_kernel(const float * __restrict__ input,
					 __half * __restrict__ output,
					 int count)
{
	int		idx;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < count)
	{
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_HIP)
		output[idx] = __float2half(input[idx]);
#else
		/* Should not occur; fallback for compilation safety. */
		union
		{
			float		f;
			unsigned	u32;
		} u;
		u.f = input[idx];
		output[idx] = (__half) (u.u32 >> 16);
#endif
	}
}

/*
 * Device kernel for quantizing float32 to int8, with scaling.
 * Explicitly clamps to int8_t range.
 */
extern "C" __global__
void
quantize_int8_kernel(const float * __restrict__ input,
					 signed char * __restrict__ output,
					 int count,
					 float scale)
{
	int		idx;
	float	val;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < count)
	{
		val = input[idx] * scale;

		if (val > 127.0f)
			val = 127.0f;
		else if (val < -128.0f)
			val = -128.0f;

		output[idx] = (signed char) __float2int_rn(val);
	}
}

/*
 * Device kernel for binary quantization:
 * Packs groups of 8 input floats to one output byte.
 */
extern "C" __global__
void
quantize_binary_kernel(const float * __restrict__ input,
					   unsigned char * __restrict__ output,
					   int count)
{
	int				idx;
	int				byte_count;
	unsigned char	byte;
	int				start, in_idx, i;

	idx = blockIdx.x * blockDim.x + threadIdx.x;
	byte_count = (count + 7) / 8;

	if (idx < byte_count)
	{
		byte = 0;
		start = idx * 8;
#pragma unroll
		for (i = 0; i < 8; i++)
		{
			in_idx = start + i;
			if (in_idx < count && input[in_idx] > 0.0f)
				byte |= (1 << i);
		}
		output[idx] = byte;
	}
}

#if defined(NDB_GPU_CUDA)

#define NDB_KERNEL_LAUNCH_PARAMS	\
	int block = 256;				\
	int grid = (count + block - 1) / block

/*
 * Host launch wrapper for quantize_fp16_kernel (float32 → fp16).
 */
extern "C" cudaError_t
launch_quantize_fp32_to_fp16(const float *input,
							 void *output,
							 int count,
							 cudaStream_t stream)
{
	if (count <= 0 || input == NULL || output == NULL)
		return cudaSuccess;

	NDB_KERNEL_LAUNCH_PARAMS;
	quantize_fp16_kernel<<<grid, block, 0, stream>>>(
		input, reinterpret_cast<__half *>(output), count);

	return cudaGetLastError();
}

/*
 * Host launch wrapper for quantize_int8_kernel (float32 → int8).
 */
extern "C" cudaError_t
launch_quantize_fp32_to_int8(const float *input,
							 signed char *output,
							 int count,
							 float scale,
							 cudaStream_t stream)
{
	if (count <= 0 || input == NULL || output == NULL)
		return cudaSuccess;

	NDB_KERNEL_LAUNCH_PARAMS;
	quantize_int8_kernel<<<grid, block, 0, stream>>>(
		input, output, count, scale);

	return cudaGetLastError();
}

/*
 * Host launch wrapper for quantize_binary_kernel (float32 → binary).
 */
extern "C" cudaError_t
launch_quantize_fp32_to_binary(const float *input,
							   unsigned char *output,
							   int count,
							   cudaStream_t stream)
{
	int block;
	int grid;

	if (count <= 0 || input == NULL || output == NULL)
		return cudaSuccess;

	block = 256;
	grid = ((count + 7) / 8 + block - 1) / block;
	quantize_binary_kernel<<<grid, block, 0, stream>>>(
		input, output, count);

	return cudaGetLastError();
}

#undef NDB_KERNEL_LAUNCH_PARAMS

#endif /* NDB_GPU_CUDA */
