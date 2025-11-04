/*
 * gpu_cuda.h
 *     NVIDIA CUDA GPU backend header
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 */

#ifndef NEURONDB_GPU_CUDA_H
#define NEURONDB_GPU_CUDA_H

#include <stdbool.h>
#include <stdint.h>

/* CUDA initialization and cleanup */
extern bool neurondb_gpu_cuda_init(void);
extern void neurondb_gpu_cuda_cleanup(void);
extern bool neurondb_gpu_cuda_is_available(void);
extern const char *neurondb_gpu_cuda_device_name(void);

/* CUDA distance operations */
extern float neurondb_gpu_cuda_l2_distance(const float *a, const float *b, int dim);
extern float neurondb_gpu_cuda_cosine_distance(const float *a, const float *b, int dim);
extern float neurondb_gpu_cuda_inner_product(const float *a, const float *b, int dim);

/* CUDA batch operations */
extern void neurondb_gpu_cuda_batch_l2(const float *queries, const float *targets,
									   int num_queries, int num_targets, int dim,
									   float *distances);

/* Statistics */
extern uint64_t neurondb_gpu_cuda_get_operations_count(void);

#endif /* NEURONDB_GPU_CUDA_H */

