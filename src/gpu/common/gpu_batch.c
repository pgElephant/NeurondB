/*-------------------------------------------------------------------------
 *
 * gpu_batch.c
 *		GPU-accelerated batch distance operations
 *
 * Computes distance matrices between query vectors and database vectors
 * efficiently using GPU batch operations (cuBLAS/rocBLAS matrix operations).
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/gpu_batch.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"

#include "neurondb_config.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"

void
neurondb_gpu_batch_l2_distance(const float *queries, const float *vectors,
                               float *results, int num_queries, int num_vectors, int dim)
{
    (void) queries;
    (void) vectors;
    (void) results;
    (void) num_queries;
    (void) num_vectors;
    (void) dim;

    if (!neurondb_gpu_is_available())
        return;

    elog(DEBUG1, "neurondb: GPU batch L2 distance not implemented for current backend; using CPU fallback");
}

void
neurondb_gpu_batch_cosine_distance(const float *queries, const float *vectors,
                                   float *results, int num_queries, int num_vectors, int dim)
{
    (void) queries;
    (void) vectors;
    (void) results;
    (void) num_queries;
    (void) num_vectors;
    (void) dim;

    if (!neurondb_gpu_is_available())
        return;

    elog(DEBUG1, "neurondb: GPU batch cosine distance not implemented for current backend; using CPU fallback");
}

