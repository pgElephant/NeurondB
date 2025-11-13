/*-------------------------------------------------------------------------
 *
 * gpu_svm_kernels.cu
 *    CUDA kernels for Support Vector Machine operations
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_svm_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#include "neurondb_cuda_runtime.h"

#ifdef NDB_GPU_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

/*
 * CUDA kernel for linear kernel computation: K(x, y) = x · y
 */
__global__ void
ndb_cuda_svm_linear_kernel_kernel(const float *x,
	const float *y,
	int feature_dim,
	float *result)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0)
	{
		float dot = 0.0f;
		int i;

		for (i = 0; i < feature_dim; i++)
			dot += x[i] * y[i];

		*result = dot;
	}
}

/*
 * CUDA kernel for SVM prediction: f(x) = Σ(alpha_i * y_i * K(x_i, x)) + bias
 */
__global__ void
ndb_cuda_svm_predict_kernel(const float *input,
	const float *support_vectors,
	const float *alphas,
	const double *labels,
	float bias,
	int n_support_vectors,
	int feature_dim,
	float *prediction)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0)
	{
		float result = bias;
		int i, j;

		for (i = 0; i < n_support_vectors; i++)
		{
			float kernel_val = 0.0f;
			const float *sv = support_vectors + (i * feature_dim);

			/* Linear kernel: K(x_i, x) = x_i · x */
			for (j = 0; j < feature_dim; j++)
				kernel_val += sv[j] * input[j];

			result += alphas[i] * (float)labels[i] * kernel_val;
		}

		*prediction = result;
	}
}

#endif /* NDB_GPU_CUDA */
