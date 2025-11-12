/*
 * gpu_lr_kernels.cu
 *    CUDA kernels for Logistic Regression training and inference.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "neurondb_cuda_lr.h"

__global__ static void
ndb_cuda_lr_forward_pass_kernel(const float *features,
	const float *weights,
	float bias,
	int n_samples,
	int feature_dim,
	double *outputs)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_samples)
		return;

	if (idx == 0)
		printf("[CUDA kernel] forward_pass: n_samples=%d, feature_dim=%d, bias=%f\n", 
			n_samples, feature_dim, bias);

	const float *row = features + idx * feature_dim;
	double z = (double)bias;

	for (int j = 0; j < feature_dim; j++)
		z += (double)weights[j] * (double)row[j];

	outputs[idx] = z;

	if (idx == 0)
		printf("[CUDA kernel] forward_pass: first output z[0]=%f\n", z);
}

__global__ static void
ndb_cuda_lr_sigmoid_kernel(const double *inputs,
	int n,
	double *outputs)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	double z = inputs[idx];

	if (z > 500.0)
		outputs[idx] = 1.0;
	else if (z < -500.0)
		outputs[idx] = 0.0;
	else
		outputs[idx] = 1.0 / (1.0 + exp(-z));
}

__global__ static void
ndb_cuda_lr_compute_gradients_kernel(const float *features,
	const double *labels,
	const double *predictions,
	int n_samples,
	int feature_dim,
	double *grad_weights,
	double *grad_bias)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_samples)
		return;

	double error = predictions[idx] - labels[idx];
	const float *row = features + idx * feature_dim;

	atomicAdd(grad_bias, error);

	for (int j = 0; j < feature_dim; j++)
		atomicAdd(&grad_weights[j], error * (double)row[j]);
}

extern "C" int
ndb_cuda_lr_forward_pass(const float *features,
	const double *weights,
	double bias,
	int n_samples,
	int feature_dim,
	double *outputs)
{
	float *d_features = NULL;
	float *d_weights = NULL;
	double *d_outputs = NULL;
	size_t feature_bytes;
	size_t weight_bytes;
	size_t output_bytes;
	cudaError_t status;
	int threads = 256;
	int blocks;

	if (features == NULL || weights == NULL || outputs == NULL
		|| n_samples <= 0 || feature_dim <= 0)
	{
		/* Use elog via PostgreSQL's error reporting - but this is CUDA code, so we can't */
		/* Just return -1 and let caller report */
		return -1;
	}

	feature_bytes = sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
	weight_bytes = sizeof(float) * (size_t)feature_dim;
	output_bytes = sizeof(double) * (size_t)n_samples;

	/* Clear any previous CUDA errors */
	cudaGetLastError();

	status = cudaMalloc((void **)&d_features, feature_bytes);
	if (status != cudaSuccess)
	{
		printf("[CUDA host] ndb_cuda_lr_forward_pass: cudaMalloc d_features failed: %s\n", 
			cudaGetErrorString(status));
		return -1;
	}
	printf("[CUDA host] ndb_cuda_lr_forward_pass: d_features allocated: %p\n", d_features);

	status = cudaMalloc((void **)&d_weights, weight_bytes);
	if (status != cudaSuccess)
	{
		printf("[CUDA host] ndb_cuda_lr_forward_pass: cudaMalloc d_weights failed: %s\n", 
			cudaGetErrorString(status));
		cudaFree(d_features);
		return -1;
	}
	printf("[CUDA host] ndb_cuda_lr_forward_pass: d_weights allocated: %p\n", d_weights);

	status = cudaMalloc((void **)&d_outputs, output_bytes);
	if (status != cudaSuccess)
	{
		printf("[CUDA host] ndb_cuda_lr_forward_pass: cudaMalloc d_outputs failed: %s\n", 
			cudaGetErrorString(status));
		cudaFree(d_weights);
		cudaFree(d_features);
		return -1;
	}
	printf("[CUDA host] ndb_cuda_lr_forward_pass: d_outputs allocated: %p\n", d_outputs);

	printf("[CUDA host] ndb_cuda_lr_forward_pass: copying features to GPU\n");
	status = cudaMemcpy(d_features,
		features,
		feature_bytes,
		cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		printf("[CUDA host] ndb_cuda_lr_forward_pass: cudaMemcpy features failed: %s\n", 
			cudaGetErrorString(status));
		goto cleanup_outputs;
	}
	printf("[CUDA host] ndb_cuda_lr_forward_pass: features copied to GPU\n");

	{
		float *h_weights = (float *)malloc(weight_bytes);
		int i;

		for (i = 0; i < feature_dim; i++)
			h_weights[i] = (float)weights[i];
		status = cudaMemcpy(d_weights,
			h_weights,
			weight_bytes,
			cudaMemcpyHostToDevice);
		free(h_weights);
		if (status != cudaSuccess)
			goto cleanup_outputs;
	}

	blocks = (n_samples + threads - 1) / threads;
	printf("[CUDA host] ndb_cuda_lr_forward_pass: launching kernel: blocks=%d, threads=%d\n", 
		blocks, threads);
	ndb_cuda_lr_forward_pass_kernel<<<blocks, threads>>>(
		d_features, d_weights, (float)bias, n_samples, feature_dim, d_outputs);

	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		printf("[CUDA host] ndb_cuda_lr_forward_pass: kernel launch error: %s\n", 
			cudaGetErrorString(status));
		goto cleanup_outputs;
	}
	printf("[CUDA host] ndb_cuda_lr_forward_pass: kernel launched successfully\n");

	printf("[CUDA host] ndb_cuda_lr_forward_pass: copying outputs from GPU\n");
	status = cudaMemcpy(outputs,
		d_outputs,
		output_bytes,
		cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		printf("[CUDA host] ndb_cuda_lr_forward_pass: cudaMemcpy outputs failed: %s\n", 
			cudaGetErrorString(status));
		goto cleanup_outputs;
	}
	printf("[CUDA host] ndb_cuda_lr_forward_pass: outputs copied from GPU, first value: %f\n", 
		outputs[0]);

cleanup_outputs:
	if (d_outputs != NULL)
	{
		cudaFree(d_outputs);
		printf("[CUDA host] ndb_cuda_lr_forward_pass: freed d_outputs\n");
	}
	if (d_weights != NULL)
	{
		cudaFree(d_weights);
		printf("[CUDA host] ndb_cuda_lr_forward_pass: freed d_weights\n");
	}
	if (d_features != NULL)
	{
		cudaFree(d_features);
		printf("[CUDA host] ndb_cuda_lr_forward_pass: freed d_features\n");
	}

	if (status == cudaSuccess)
	{
		printf("[CUDA host] ndb_cuda_lr_forward_pass: success, returning 0\n");
		return 0;
	}
	else
	{
		printf("[CUDA host] ndb_cuda_lr_forward_pass: failure, returning -1\n");
		return -1;
	}
}

/* Host-side wrapper for launching forward pass kernel with pre-allocated GPU memory */
extern "C" int
ndb_cuda_lr_forward_pass_gpu(const float *d_features,
	const float *d_weights,
	float bias,
	int n_samples,
	int feature_dim,
	double *d_outputs)
{
	cudaError_t status;
	int threads = 256;
	int blocks;

	if (d_features == NULL || d_weights == NULL || d_outputs == NULL
		|| n_samples <= 0 || feature_dim <= 0)
		return -1;

	/* Clear any previous CUDA errors */
	cudaGetLastError();

	blocks = (n_samples + threads - 1) / threads;
	printf("[CUDA host] ndb_cuda_lr_forward_pass_gpu: launching kernel: blocks=%d, threads=%d\n", 
		blocks, threads);

	ndb_cuda_lr_forward_pass_kernel<<<blocks, threads>>>(
		d_features, d_weights, bias, n_samples, feature_dim, d_outputs);

	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		printf("[CUDA host] ndb_cuda_lr_forward_pass_gpu: kernel launch error: %s\n", 
			cudaGetErrorString(status));
		return -1;
	}

	printf("[CUDA host] ndb_cuda_lr_forward_pass_gpu: kernel launched successfully\n");
	return 0;
}

extern "C" int
ndb_cuda_lr_sigmoid(const double *inputs,
	int n,
	double *outputs)
{
	double *d_inputs = NULL;
	double *d_outputs = NULL;
	size_t input_bytes;
	size_t output_bytes;
	cudaError_t status;
	int threads = 256;
	int blocks;

	if (inputs == NULL || outputs == NULL || n <= 0)
		return -1;

	input_bytes = sizeof(double) * (size_t)n;
	output_bytes = sizeof(double) * (size_t)n;

	status = cudaMalloc((void **)&d_inputs, input_bytes);
	if (status != cudaSuccess)
		return -1;
	status = cudaMalloc((void **)&d_outputs, output_bytes);
	if (status != cudaSuccess)
		goto cleanup_inputs;

	status = cudaMemcpy(d_inputs,
		inputs,
		input_bytes,
		cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto cleanup_outputs;

	blocks = (n + threads - 1) / threads;
	ndb_cuda_lr_sigmoid_kernel<<<blocks, threads>>>(d_inputs, n, d_outputs);

	status = cudaGetLastError();
	if (status != cudaSuccess)
		goto cleanup_outputs;

	status = cudaMemcpy(outputs,
		d_outputs,
		output_bytes,
		cudaMemcpyDeviceToHost);

cleanup_outputs:
	cudaFree(d_outputs);
cleanup_inputs:
	cudaFree(d_inputs);

	return (status == cudaSuccess) ? 0 : -1;
}

extern "C" int
ndb_cuda_lr_compute_gradients(const float *features,
	const double *labels,
	const double *predictions,
	int n_samples,
	int feature_dim,
	double *grad_weights,
	double *grad_bias)
{
	float *d_features = NULL;
	double *d_labels = NULL;
	double *d_predictions = NULL;
	double *d_grad_weights = NULL;
	double *d_grad_bias = NULL;
	size_t feature_bytes;
	size_t label_bytes;
	size_t pred_bytes;
	size_t grad_weight_bytes;
	size_t grad_bias_bytes;
	cudaError_t status;
	int threads = 256;
	int blocks;

	if (features == NULL || labels == NULL || predictions == NULL
		|| grad_weights == NULL || grad_bias == NULL
		|| n_samples <= 0 || feature_dim <= 0)
		return -1;

	feature_bytes = sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
	label_bytes = sizeof(double) * (size_t)n_samples;
	pred_bytes = sizeof(double) * (size_t)n_samples;
	grad_weight_bytes = sizeof(double) * (size_t)feature_dim;
	grad_bias_bytes = sizeof(double);

	memset(grad_weights, 0, grad_weight_bytes);
	*grad_bias = 0.0;

	status = cudaMalloc((void **)&d_features, feature_bytes);
	if (status != cudaSuccess)
		return -1;
	status = cudaMalloc((void **)&d_labels, label_bytes);
	if (status != cudaSuccess)
		goto cleanup_features;
	status = cudaMalloc((void **)&d_predictions, pred_bytes);
	if (status != cudaSuccess)
		goto cleanup_labels;
	status = cudaMalloc((void **)&d_grad_weights, grad_weight_bytes);
	if (status != cudaSuccess)
		goto cleanup_predictions;
	status = cudaMalloc((void **)&d_grad_bias, grad_bias_bytes);
	if (status != cudaSuccess)
		goto cleanup_grad_weights;

	status = cudaMemcpy(d_features,
		features,
		feature_bytes,
		cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto cleanup_grad_bias;
	status = cudaMemcpy(d_labels,
		labels,
		label_bytes,
		cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto cleanup_grad_bias;
	status = cudaMemcpy(d_predictions,
		predictions,
		pred_bytes,
		cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto cleanup_grad_bias;

	status = cudaMemset(d_grad_weights, 0, grad_weight_bytes);
	if (status != cudaSuccess)
		goto cleanup_grad_bias;
	status = cudaMemset(d_grad_bias, 0, grad_bias_bytes);
	if (status != cudaSuccess)
		goto cleanup_grad_bias;

	blocks = (n_samples + threads - 1) / threads;
	ndb_cuda_lr_compute_gradients_kernel<<<blocks, threads>>>(
		d_features,
		d_labels,
		d_predictions,
		n_samples,
		feature_dim,
		d_grad_weights,
		d_grad_bias);

	status = cudaGetLastError();
	if (status != cudaSuccess)
		goto cleanup_grad_bias;

	status = cudaMemcpy(grad_weights,
		d_grad_weights,
		grad_weight_bytes,
		cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto cleanup_grad_bias;
	status = cudaMemcpy(grad_bias,
		d_grad_bias,
		grad_bias_bytes,
		cudaMemcpyDeviceToHost);

cleanup_grad_bias:
	cudaFree(d_grad_bias);
cleanup_grad_weights:
	cudaFree(d_grad_weights);
cleanup_predictions:
	cudaFree(d_predictions);
cleanup_labels:
	cudaFree(d_labels);
cleanup_features:
	cudaFree(d_features);

	return (status == cudaSuccess) ? 0 : -1;
}

