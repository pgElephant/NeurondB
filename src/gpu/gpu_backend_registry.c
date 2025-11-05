/*-------------------------------------------------------------------------
 *
 * gpu_backend_registry.c
 *     GPU backend registration and selection system
 *
 * This module manages the registry of available GPU backends and provides
 * automatic backend selection based on system capabilities.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *     src/gpu/gpu_backend_registry.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "utils/elog.h"
#include "gpu_backend_interface.h"

#include <string.h>

/* Backend registry */
typedef struct {
    const GPUBackendInterface *backends[MAX_GPU_BACKENDS];
    int num_backends;
} GPUBackendRegistry;

static GPUBackendRegistry registry = {
    .backends = {NULL},
    .num_backends = 0
};

/* Current active backend */
static const GPUBackendInterface *active_backend = NULL;

/*
 * Register a GPU backend
 */
bool
gpu_backend_register(const GPUBackendInterface *backend)
{
    int i;
    
    if (!backend)
    {
        elog(WARNING, "gpu_backend_register: NULL backend");
        return false;
    }
    
    if (registry.num_backends >= MAX_GPU_BACKENDS)
    {
        elog(WARNING, "gpu_backend_register: registry full");
        return false;
    }
    
    /* Check for duplicates */
    for (i = 0; i < registry.num_backends; i++)
    {
        if (registry.backends[i]->type == backend->type)
        {
            elog(WARNING, "gpu_backend_register: backend type %d already registered", 
                 backend->type);
            return false;
        }
    }
    
    /* Add to registry */
    registry.backends[registry.num_backends++] = backend;
    
    elog(DEBUG1, "neurondb: GPU backend registered: %s (version %s)",
         backend->name, backend->version ? backend->version : "unknown");
    
    return true;
}

/*
 * Get a specific backend by type
 */
const GPUBackendInterface*
gpu_backend_get(GPUBackendType type)
{
    int i;
    
    for (i = 0; i < registry.num_backends; i++)
    {
        if (registry.backends[i]->type == type)
            return registry.backends[i];
    }
    
    return NULL;
}

/*
 * Get all registered backends
 */
int
gpu_backend_get_all(const GPUBackendInterface **backends, int max_count)
{
    int i;
    int count = registry.num_backends < max_count ? registry.num_backends : max_count;
    
    for (i = 0; i < count; i++)
    {
        backends[i] = registry.backends[i];
    }
    
    return count;
}

/*
 * Select the best available backend
 * 
 * Priority order:
 * 1. Metal (on Apple Silicon)
 * 2. CUDA (on NVIDIA)
 * 3. ROCm (on AMD)
 * 4. OpenCL (fallback)
 * 5. Vulkan (fallback)
 */
const GPUBackendInterface*
gpu_backend_select_best(void)
{
    int i;
    const GPUBackendInterface *best = NULL;
    int best_priority = -1;
    
    /* Priority map */
    static const struct {
        GPUBackendType type;
        int priority;
    } priority_map[] = {
        {GPU_BACKEND_TYPE_METAL, 100},
        {GPU_BACKEND_TYPE_CUDA, 90},
        {GPU_BACKEND_TYPE_ROCM, 80},
        {GPU_BACKEND_TYPE_OPENCL, 50},
        {GPU_BACKEND_TYPE_VULKAN, 40},
        {GPU_BACKEND_TYPE_SYCL, 30}
    };
    
    for (i = 0; i < registry.num_backends; i++)
    {
        const GPUBackendInterface *backend = registry.backends[i];
        int j;
        int priority = 0;
        
        /* Check if backend is available */
        if (!backend->is_available || !backend->is_available())
            continue;
        
        /* Find priority */
        for (j = 0; j < (int) (sizeof(priority_map) / sizeof(priority_map[0])); j++)
        {
            if (priority_map[j].type == backend->type)
            {
                priority = priority_map[j].priority;
                break;
            }
        }
        
        if (priority > best_priority)
        {
            best = backend;
            best_priority = priority;
        }
    }
    
    if (best)
    {
        elog(LOG, "neurondb: Auto-selected GPU backend: %s", best->name);
    }
    else
    {
        elog(DEBUG1, "neurondb: No GPU backends available");
    }
    
    return best;
}

/*
 * Select backend by name
 */
const GPUBackendInterface*
gpu_backend_select_by_name(const char *name)
{
    int i;
    
    if (!name || !*name || strcmp(name, "auto") == 0)
        return gpu_backend_select_best();
    
    for (i = 0; i < registry.num_backends; i++)
    {
        const GPUBackendInterface *backend = registry.backends[i];
        
        if (strcasecmp(backend->name, name) == 0)
        {
            if (backend->is_available && backend->is_available())
            {
                elog(LOG, "neurondb: Selected GPU backend: %s", backend->name);
                return backend;
            }
            else
            {
                elog(WARNING, "neurondb: GPU backend '%s' not available", name);
                return NULL;
            }
        }
    }
    
    elog(WARNING, "neurondb: GPU backend '%s' not found", name);
    return NULL;
}

/*
 * Set active backend
 */
bool
gpu_backend_set_active(const GPUBackendInterface *backend)
{
    if (active_backend && active_backend->cleanup)
    {
        active_backend->cleanup();
    }
    
    active_backend = backend;
    
    if (backend && backend->init)
    {
        if (!backend->init())
        {
            active_backend = NULL;
            return false;
        }
    }
    
    return true;
}

/*
 * Get active backend
 */
const GPUBackendInterface*
gpu_backend_get_active(void)
{
    return active_backend;
}

/*
 * List all registered backends (for debugging)
 */
void
gpu_backend_list(void)
{
    int i;
    
    elog(LOG, "neurondb: GPU backends registered: %d", registry.num_backends);
    
    for (i = 0; i < registry.num_backends; i++)
    {
        const GPUBackendInterface *backend = registry.backends[i];
        bool available = backend->is_available ? backend->is_available() : false;
        
        elog(LOG, "  [%d] %s (v%s) - %s",
             i + 1,
             backend->name,
             backend->version ? backend->version : "unknown",
             available ? "available" : "unavailable");
    }
}

