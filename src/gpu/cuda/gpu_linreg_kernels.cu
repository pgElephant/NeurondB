/*-------------------------------------------------------------------------
 *
 * gpu_linreg_kernels.cu
 *    CUDA kernels for Linear Regression operations
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_linreg_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#include "neurondb_cuda_runtime.h"

#ifdef NDB_GPU_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

/*
 * CUDA kernel to compute X'X and X'y for normal equations
 * Each thread computes contributions for one sample, uses atomic operations for accumulation
 */
__global__ void
ndb_cuda_linreg_compute_xtx_kernel(const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	int dim_with_intercept,
	double *XtX,
	double *Xty)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= n_samples)
		return;
	
	int i, j;
	double xi[128];  /* Max 128 features + intercept */
	double yi = targets[idx];
	
	/* Build xi vector with intercept */
	xi[0] = 1.0;
	for (i = 1; i < dim_with_intercept && i < 128; i++)
	{
		if (i - 1 < feature_dim)
			xi[i] = (double)features[idx * feature_dim + (i - 1)];
		else
			xi[i] = 0.0;
	}
	
	/* Accumulate X'X and X'y using atomic operations */
	for (j = 0; j < dim_with_intercept && j < 128; j++)
	{
		/* X'y accumulation */
		atomicAdd((unsigned long long *)&Xty[j], __double_as_longlong(xi[j] * yi));
		
		/* X'X accumulation */
		for (i = 0; i < dim_with_intercept && i < 128; i++)
		{
			atomicAdd((unsigned long long *)&XtX[j * dim_with_intercept + i],
				__double_as_longlong(xi[j] * xi[i]));
		}
	}
}

/*
 * CUDA kernel for prediction: y = intercept + coef1*x1 + coef2*x2 + ...
 */
__global__ void
ndb_cuda_linreg_predict_kernel(const float *input,
	const float *coefficients,
	float intercept,
	int feature_dim,
	float *prediction)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0)
	{
		float result = intercept;
		int i;

		for (i = 0; i < feature_dim; i++)
			result += coefficients[i] * input[i];

		*prediction = result;
	}
}

#endif /* NDB_GPU_CUDA */
