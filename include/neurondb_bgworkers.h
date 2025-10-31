/*-------------------------------------------------------------------------
 *
 * neurondb_bgworkers.h
 *		Background worker declarations for NeurondB
 *
 * This header defines the three background workers:
 *   - neuranq: Queue executor for async jobs
 *   - neuranmon: Auto-tuner and monitoring
 *   - neurandefrag: Index maintenance
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  include/neurondb_bgworkers.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_BGWORKERS_H
#define NEURONDB_BGWORKERS_H

#include "storage/lwlock.h"

/*
 * Background worker: neuranq (Queue Executor)
 */
extern void neuranq_init_guc(void);
extern Size neuranq_shmem_size(void);
extern void neuranq_shmem_init(void);
extern void neuranq_main(Datum main_arg) pg_attribute_noreturn();

/*
 * Background worker: neuranmon (Auto-Tuner)
 */
extern void neuranmon_init_guc(void);
extern Size neuranmon_shmem_size(void);
extern void neuranmon_shmem_init(void);
extern void neuranmon_main(Datum main_arg) pg_attribute_noreturn();

/*
 * Background worker: neurandefrag (Index Maintenance)
 */
extern void neurandefrag_init_guc(void);
extern Size neurandefrag_shmem_size(void);
extern void neurandefrag_shmem_init(void);
extern void neurandefrag_main(Datum main_arg) pg_attribute_noreturn();

/*
 * Manual execution functions for operators
 */
Datum neuranq_run_once(PG_FUNCTION_ARGS);
Datum neuranmon_sample(PG_FUNCTION_ARGS);
Datum neurandefrag_run(PG_FUNCTION_ARGS);

/* Worker watchdog view - to be implemented separately if needed */

#endif   /* NEURONDB_BGWORKERS_H */

