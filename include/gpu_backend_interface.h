/*-------------------------------------------------------------------------
 *
 * gpu_backend_interface.h
 *     Modular GPU backend interface for NeurondB
 *
 * This defines a unified interface that all GPU backends must implement,
 * enabling clean separation and easy addition of new GPU types.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *     include/gpu_backend_interface.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPU_BACKEND_INTERFACE_H
#define GPU_BACKEND_INTERFACE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Forward declarations */
typedef struct GPUBackendInterface GPUBackendInterface;
typedef struct GPUBackendContext GPUBackendContext;

/*
 * GPU Backend Types
 */
typedef enum GPUBackendType
{
    GPU_BACKEND_TYPE_NONE = 0,
    GPU_BACKEND_TYPE_CUDA,      /* NVIDIA CUDA */
    GPU_BACKEND_TYPE_ROCM,      /* AMD ROCm/HIP */
    GPU_BACKEND_TYPE_METAL,     /* Apple Metal */
    GPU_BACKEND_TYPE_OPENCL,    /* OpenCL (future) */
    GPU_BACKEND_TYPE_VULKAN,    /* Vulkan Compute (future) */
    GPU_BACKEND_TYPE_SYCL       /* Intel oneAPI/SYCL (future) */
} GPUBackendType;

/*
 * GPU Device Information
 */
typedef struct GPUDeviceInfo
{
    int         device_id;
    char        name[256];
    uint64_t    total_memory;      /* Total memory in bytes */
    uint64_t    free_memory;       /* Free memory in bytes */
    int         compute_major;
    int         compute_minor;
    int         max_threads_per_block;
    int         multiprocessor_count;
    bool        unified_memory;
    bool        is_available;
} GPUDeviceInfo;

/*
 * GPU Backend Interface - All backends must implement these functions
 * 
 * This provides a clean abstraction layer for different GPU vendors.
 * Each backend (CUDA, ROCm, Metal, etc.) provides its own implementation.
 */
struct GPUBackendInterface
{
    /* Backend identification */
    GPUBackendType  type;
    const char     *name;            /* Human-readable name */
    const char     *version;         /* Backend version string */
    
    /* === Lifecycle Management === */
    
    /* Initialize backend - called once at startup */
    bool (*init)(void);
    
    /* Cleanup backend - called at shutdown */
    void (*cleanup)(void);
    
    /* Check if backend is available on this system */
    bool (*is_available)(void);
    
    /* === Device Management === */
    
    /* Get number of available devices */
    int (*get_device_count)(void);
    
    /* Get device information */
    bool (*get_device_info)(int device_id, GPUDeviceInfo *info);
    
    /* Set active device */
    bool (*set_device)(int device_id);
    
    /* === Memory Management === */
    
    /* Allocate device memory */
    void* (*mem_alloc)(size_t bytes);
    
    /* Free device memory */
    void (*mem_free)(void *ptr);
    
    /* Copy host to device */
    bool (*mem_copy_h2d)(void *dst, const void *src, size_t bytes);
    
    /* Copy device to host */
    bool (*mem_copy_d2h)(void *dst, const void *src, size_t bytes);
    
    /* Synchronize device */
    void (*synchronize)(void);
    
    /* === Vector Operations === */
    
    /* L2 distance between two vectors */
    float (*l2_distance)(const float *a, const float *b, int dim);
    
    /* Cosine distance between two vectors */
    float (*cosine_distance)(const float *a, const float *b, int dim);
    
    /* Inner product between two vectors */
    float (*inner_product)(const float *a, const float *b, int dim);
    
    /* === Batch Operations === */
    
    /* Batch L2 distance computation */
    bool (*batch_l2)(const float *queries, const float *targets,
                     int num_queries, int num_targets, int dim,
                     float *distances);
    
    /* Batch cosine distance computation */
    bool (*batch_cosine)(const float *queries, const float *targets,
                         int num_queries, int num_targets, int dim,
                         float *distances);
    
    /* === Quantization === */
    
    /* Quantize to int8 */
    bool (*quantize_int8)(const float *input, int8_t *output, int count);
    
    /* Quantize to fp16 */
    bool (*quantize_fp16)(const float *input, void *output, int count);
    
    /* === Clustering === */
    
    /* K-means clustering */
    bool (*kmeans)(const float *vectors, int num_vectors, int dim,
                   int k, int max_iters, float *centroids, int *assignments);
    
    /* DBSCAN clustering */
    bool (*dbscan)(const float *vectors, int num_vectors, int dim,
                   float eps, int min_points, int *cluster_ids);
    
    /* === Advanced Features (optional) === */
    
    /* Create compute streams/queues (may be NULL) */
    bool (*create_streams)(int num_streams);
    
    /* Destroy compute streams/queues (may be NULL) */
    void (*destroy_streams)(void);
    
    /* Get backend-specific context (opaque) */
    void* (*get_context)(void);
};

/* Maximum number of GPU backends that can be registered */
#define MAX_GPU_BACKENDS 8

/*
 * GPU Backend Registration
 * 
 * Each backend implementation should call this during module initialization
 * to register itself with the GPU core system.
 */
extern bool gpu_backend_register(const GPUBackendInterface *backend);

/*
 * Get registered backends
 */
extern const GPUBackendInterface* gpu_backend_get(GPUBackendType type);
extern int gpu_backend_get_all(const GPUBackendInterface **backends, int max_count);

/*
 * Backend auto-detection and selection
 */
extern const GPUBackendInterface* gpu_backend_select_best(void);
extern const GPUBackendInterface* gpu_backend_select_by_name(const char *name);

/*
 * Active backend management
 */
extern bool gpu_backend_set_active(const GPUBackendInterface *backend);
extern const GPUBackendInterface* gpu_backend_get_active(void);

/*
 * List all registered backends (for debugging)
 */
extern void gpu_backend_list(void);

#endif /* GPU_BACKEND_INTERFACE_H */

