/*-------------------------------------------------------------------------
 *
 * neurondb_gpu.h
 *		GPU acceleration support for NeurondB vector operations
 *
 * This module provides optional GPU acceleration for compute-intensive
 * operations using CUDA (NVIDIA) or ROCm (AMD). GPU support is optional
 * and falls back to CPU automatically if GPU is unavailable.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  include/neurondb_gpu.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_GPU_H
#define NEURONDB_GPU_H

#include "postgres.h"

/* GPU backend types */
typedef enum GPUBackend
{
	GPU_BACKEND_NONE = 0,
	GPU_BACKEND_CUDA = 1,
	GPU_BACKEND_ROCM = 2,
	GPU_BACKEND_OPENCL = 3
} GPUBackend;

/* GPU device info */
typedef struct GPUDeviceInfo
{
	int device_id;
	char name[256];
	size_t total_memory;
	size_t free_memory;
	int compute_capability_major;
	int compute_capability_minor;
	bool is_available;
} GPUDeviceInfo;

/* GPU statistics */
typedef struct GPUStats
{
	int64 queries_executed;
	int64 fallback_count;
	double total_gpu_time_ms;
	double total_cpu_time_ms;
	double avg_latency_ms;
	TimestampTz last_reset;
} GPUStats;

/* GPU configuration GUCs */
extern bool neurondb_gpu_enabled;
extern int neurondb_gpu_device;
extern int neurondb_gpu_batch_size;
extern int neurondb_gpu_streams;
extern double neurondb_gpu_memory_pool_mb;
extern bool neurondb_gpu_fail_open;
extern char *neurondb_gpu_kernels;
extern int neurondb_gpu_timeout_ms;

/* GPU initialization and cleanup */
extern void neurondb_gpu_init(void);
extern void neurondb_gpu_shutdown(void);
extern bool neurondb_gpu_is_available(void);
extern GPUBackend neurondb_gpu_get_backend(void);

/* GPU device management */
extern int neurondb_gpu_get_device_count(void);
extern GPUDeviceInfo *neurondb_gpu_get_device_info(int device_id);
extern void neurondb_gpu_set_device(int device_id);

/* GPU distance operations */
extern float neurondb_gpu_l2_distance(const float *vec1, const float *vec2, int dim);
extern float neurondb_gpu_cosine_distance(const float *vec1, const float *vec2, int dim);
extern float neurondb_gpu_inner_product(const float *vec1, const float *vec2, int dim);

/* GPU batch distance operations */
extern void neurondb_gpu_batch_l2_distance(const float *queries, const float *vectors,
										   float *results, int num_queries, int num_vectors, int dim);
extern void neurondb_gpu_batch_cosine_distance(const float *queries, const float *vectors,
											    float *results, int num_queries, int num_vectors, int dim);

/* GPU ANN search */
extern void neurondb_gpu_hnsw_search(const float *query, int dim, int k,
									 int *result_ids, float *result_distances);

/* GPU quantization */
extern void neurondb_gpu_quantize_fp16(const float *input, void *output, int count);
extern void neurondb_gpu_quantize_int8(const float *input, int8 *output, int count);
extern void neurondb_gpu_quantize_binary(const float *input, uint8 *output, int count);

/* GPU clustering */
extern void neurondb_gpu_kmeans(const float *vectors, int num_vectors, int dim,
								int k, int max_iters, float *centroids, int *assignments);
extern void neurondb_gpu_dbscan(const float *vectors, int num_vectors, int dim,
								float eps, int min_pts, int *labels);

/* GPU dimensionality reduction */
extern void neurondb_gpu_pca(const float *vectors, int num_vectors, int dim,
							 int target_dim, float *output);

/* GPU embedding inference */
extern void neurondb_gpu_onnx_inference(void *model_handle, const float *input,
										int input_size, float *output, int output_size);

/* GPU statistics and monitoring */
extern GPUStats *neurondb_gpu_get_stats(void);
extern void neurondb_gpu_reset_stats(void);

/* SQL-callable functions */
extern Datum neurondb_gpu_enable(PG_FUNCTION_ARGS);
extern Datum neurondb_gpu_info(PG_FUNCTION_ARGS);
extern Datum neurondb_gpu_stats(PG_FUNCTION_ARGS);
extern Datum neurondb_gpu_stats_reset(PG_FUNCTION_ARGS);

/* GPU distance function overrides */
extern Datum vector_l2_distance_gpu(PG_FUNCTION_ARGS);
extern Datum vector_cosine_distance_gpu(PG_FUNCTION_ARGS);
extern Datum vector_inner_product_gpu(PG_FUNCTION_ARGS);

/* GPU ANN search functions */
extern Datum hnsw_knn_search_gpu(PG_FUNCTION_ARGS);
extern Datum ivf_knn_search_gpu(PG_FUNCTION_ARGS);

/* GPU quantization functions */
extern Datum vector_to_int8_gpu(PG_FUNCTION_ARGS);
extern Datum vector_to_fp16_gpu(PG_FUNCTION_ARGS);
extern Datum vector_to_binary_gpu(PG_FUNCTION_ARGS);

/* GPU clustering functions */
extern Datum cluster_kmeans_gpu(PG_FUNCTION_ARGS);

/* Internal GPU runtime functions */
extern int ndb_gpu_runtime_init(int *device_id);
extern void ndb_gpu_mem_pool_init(int pool_size_mb);
extern void ndb_gpu_streams_init(int num_streams);
extern bool ndb_gpu_kernel_enabled(const char *kernel_name);
extern void ndb_gpu_init_if_needed(void);

#endif /* NEURONDB_GPU_H */

