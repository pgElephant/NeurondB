/*-------------------------------------------------------------------------
 *
 * gpu_distance.c
 *		GPU-accelerated distance operations for NeurondB
 *
 * Implements L2, cosine, and inner product distance metrics using
 * CUDA cuBLAS or ROCm rocBLAS for high-performance vector operations.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/gpu_distance.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/timestamp.h"

#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"

#include <math.h>
#include <string.h>

/*
 * GPU L2 distance (Euclidean)
 */
float
neurondb_gpu_l2_distance(const float *vec1, const float *vec2, int dim)
{
    const ndb_gpu_backend *backend;
    float result = -1.0f;

    if (!neurondb_gpu_is_available())
        return -1.0f;

    backend = ndb_gpu_get_active_backend();
    if (!backend || !backend->launch_l2_distance)
    {
        elog(DEBUG1, "neurondb: GPU l2 distance not implemented for backend %s; using CPU fallback",
             backend && backend->name ? backend->name : "unknown");
        return -1.0f;
    }

    if (backend->launch_l2_distance(vec1, vec2, &result, 1, dim, NULL) != 0)
        return -1.0f;

    return result;
}

/*
 * GPU cosine distance
 */
float
neurondb_gpu_cosine_distance(const float *vec1, const float *vec2, int dim)
{
    const ndb_gpu_backend *backend;
    float result = -1.0f;

    if (!neurondb_gpu_is_available())
        return -1.0f;

    backend = ndb_gpu_get_active_backend();
    if (!backend || !backend->launch_cosine)
    {
        elog(DEBUG1, "neurondb: GPU cosine distance not implemented for backend %s; using CPU fallback",
             backend && backend->name ? backend->name : "unknown");
        return -1.0f;
    }

    if (backend->launch_cosine(vec1, vec2, &result, 1, dim, NULL) != 0)
        return -1.0f;

    return result;
}

/*
 * GPU inner product
 */
float
neurondb_gpu_inner_product(const float *vec1, const float *vec2, int dim)
{
    (void) vec1;
    (void) vec2;
    (void) dim;

    return -1.0f;
}
