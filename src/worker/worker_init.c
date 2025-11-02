/*-------------------------------------------------------------------------
 *
 * bgworker_init.c
 *		Background worker initialization and registration
 *
 * This module handles registration of all NeurondB background workers
 * via shared_preload_libraries mechanism.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/bgworker_init.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "utils/guc.h"

/* Forward declarations from background worker modules */
extern void neuranq_main(Datum main_arg) pg_attribute_noreturn();
extern void neuranq_init_guc(void);
extern Size neuranq_shmem_size(void);
extern void neuranq_shmem_init(void);

extern void neuranmon_main(Datum main_arg) pg_attribute_noreturn();
extern void neuranmon_init_guc(void);
extern Size neuranmon_shmem_size(void);
extern void neuranmon_shmem_init(void);

extern void neurandefrag_main(Datum main_arg) pg_attribute_noreturn();
extern void neurandefrag_init_guc(void);
extern Size neurandefrag_shmem_size(void);
extern void neurandefrag_shmem_init(void);

/* GPU module declarations */
extern void neurondb_gpu_init_guc(void);
extern void neurondb_gpu_init(void);
extern void neurondb_llm_init_guc(void);
extern Size neurondb_llm_shmem_size(void);
extern void neurondb_llm_shmem_init(void);
extern void neuranllm_main(Datum main_arg) pg_attribute_noreturn();

/* Prometheus and cache declarations */
extern Size entrypoint_cache_shmem_size(void);
extern void entrypoint_cache_shmem_init(void);

/* Module initialization */
void _PG_init(void);
void _PG_fini(void);

/* Shared memory hooks */
#if PG_VERSION_NUM >= 150000
static shmem_request_hook_type prev_shmem_request_hook = NULL;
static void neurondb_shmem_request(void);
#endif
static shmem_startup_hook_type prev_shmem_startup_hook = NULL;
static void neurondb_shmem_startup(void);

/*
 * Module load callback
 */
void
_PG_init(void)
{
	BackgroundWorker worker;

	if (!process_shared_preload_libraries_in_progress)
	{
		elog(WARNING, "neurondb: background workers require shared_preload_libraries");
		return;
	}

	elog(LOG, "neurondb: initializing background workers");

	/* Initialize GUC variables for all workers */
	neuranq_init_guc();
	neuranmon_init_guc();
	neurandefrag_init_guc();
	
    /* Initialize GPU and LLM GUC variables */
	neurondb_gpu_init_guc();
    neurondb_llm_init_guc();

    /* Install shared memory request hook or request directly (older PG) */
#if PG_VERSION_NUM >= 150000
    prev_shmem_request_hook = shmem_request_hook;
    shmem_request_hook = neurondb_shmem_request;
#else
    /* Request shared memory and LWLocks directly for older versions */
    RequestAddinShmemSpace(neuranq_shmem_size() +
                           neuranmon_shmem_size() +
                           neurandefrag_shmem_size() +
                           neurondb_llm_shmem_size() +
                           entrypoint_cache_shmem_size() +
                           8192); /* prometheus metrics */
    RequestNamedLWLockTranche("neurondb_queue", 1);
    RequestNamedLWLockTranche("neurondb_tuner", 1);
    RequestNamedLWLockTranche("neurondb_defrag", 1);
    RequestNamedLWLockTranche("neurondb_llm", 1);
    RequestNamedLWLockTranche("neurondb_prometheus", 1);
    RequestNamedLWLockTranche("neurondb_entrypoint_cache", 1);
#endif

	/* Install shared memory startup hook */
	prev_shmem_startup_hook = shmem_startup_hook;
	shmem_startup_hook = neurondb_shmem_startup;

	/*
	 * Register background worker: neuranq (Queue Executor)
	 */
	memset(&worker, 0, sizeof(worker));
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS |
					   BGWORKER_BACKEND_DATABASE_CONNECTION;
	worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
	snprintf(worker.bgw_library_name, BGW_MAXLEN, "neurondb");
	snprintf(worker.bgw_function_name, BGW_MAXLEN, "neuranq_main");
	snprintf(worker.bgw_name, BGW_MAXLEN, "neurondb: queue worker");
	snprintf(worker.bgw_type, BGW_MAXLEN, "neurondb_queue");
	worker.bgw_restart_time = BGW_DEFAULT_RESTART_INTERVAL;
	worker.bgw_notify_pid = 0;
	worker.bgw_main_arg = (Datum) 0;

	RegisterBackgroundWorker(&worker);

	elog(LOG, "neurondb: registered neuranq background worker");

	/*
	 * Register background worker: neuranmon (Auto-Tuner)
	 */
	memset(&worker, 0, sizeof(worker));
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS |
					   BGWORKER_BACKEND_DATABASE_CONNECTION;
	worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
	snprintf(worker.bgw_library_name, BGW_MAXLEN, "neurondb");
	snprintf(worker.bgw_function_name, BGW_MAXLEN, "neuranmon_main");
	snprintf(worker.bgw_name, BGW_MAXLEN, "neurondb: tuner worker");
	snprintf(worker.bgw_type, BGW_MAXLEN, "neurondb_tuner");
	worker.bgw_restart_time = BGW_DEFAULT_RESTART_INTERVAL;
	worker.bgw_notify_pid = 0;
	worker.bgw_main_arg = (Datum) 0;

	RegisterBackgroundWorker(&worker);

	elog(LOG, "neurondb: registered neuranmon background worker");

	/*
	 * Register background worker: neurandefrag (Index Maintenance)
	 */
	memset(&worker, 0, sizeof(worker));
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS |
					   BGWORKER_BACKEND_DATABASE_CONNECTION;
	worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
	snprintf(worker.bgw_library_name, BGW_MAXLEN, "neurondb");
	snprintf(worker.bgw_function_name, BGW_MAXLEN, "neurandefrag_main");
	snprintf(worker.bgw_name, BGW_MAXLEN, "neurondb: defrag worker");
	snprintf(worker.bgw_type, BGW_MAXLEN, "neurondb_defrag");
	worker.bgw_restart_time = BGW_DEFAULT_RESTART_INTERVAL;
	worker.bgw_notify_pid = 0;
	worker.bgw_main_arg = (Datum) 0;

	RegisterBackgroundWorker(&worker);

    elog(LOG, "neurondb: registered neurandefrag background worker");
    
    /* Register background worker: neuranllm (LLM jobs) */
    memset(&worker, 0, sizeof(worker));
    worker.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
    worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
    snprintf(worker.bgw_library_name, BGW_MAXLEN, "neurondb");
    snprintf(worker.bgw_function_name, BGW_MAXLEN, "neuranllm_main");
    snprintf(worker.bgw_name, BGW_MAXLEN, "neurondb: llm worker");
    snprintf(worker.bgw_type, BGW_MAXLEN, "neurondb_llm");
    worker.bgw_restart_time = BGW_DEFAULT_RESTART_INTERVAL;
    worker.bgw_notify_pid = 0;
    worker.bgw_main_arg = (Datum) 0;
    RegisterBackgroundWorker(&worker);
    elog(LOG, "neurondb: registered neuranllm background worker");
	elog(LOG, "neurondb: all background workers registered successfully");
}

/*
 * Module unload callback
 */
void
_PG_fini(void)
{
    /* Restore previous hooks */
#if PG_VERSION_NUM >= 150000
    shmem_request_hook = prev_shmem_request_hook;
#endif
	shmem_startup_hook = prev_shmem_startup_hook;

	elog(LOG, "neurondb: background workers shutting down");
}

#if PG_VERSION_NUM >= 150000
/*
 * Shared memory request hook (PG15+)
 */
static void
neurondb_shmem_request(void)
{
    /* Call previous hook if any */
    if (prev_shmem_request_hook)
        prev_shmem_request_hook();

    /* Request shared memory */
    RequestAddinShmemSpace(neuranq_shmem_size() +
                           neuranmon_shmem_size() +
                           neurandefrag_shmem_size() +
                           neurondb_llm_shmem_size() +
                           entrypoint_cache_shmem_size() +
                           8192); /* prometheus metrics */

    /* Request LWLocks */
    RequestNamedLWLockTranche("neurondb_queue", 1);
    RequestNamedLWLockTranche("neurondb_tuner", 1);
    RequestNamedLWLockTranche("neurondb_defrag", 1);
    RequestNamedLWLockTranche("neurondb_llm", 1);
    RequestNamedLWLockTranche("neurondb_prometheus", 1);
    RequestNamedLWLockTranche("neurondb_entrypoint_cache", 1);
}
#endif

/*
 * Shared memory startup hook
 */
static void
neurondb_shmem_startup(void)
{
	/* Call previous hook if any */
	if (prev_shmem_startup_hook)
		prev_shmem_startup_hook();

	/* Initialize shared memory for all workers */
	neuranq_shmem_init();
	neuranmon_shmem_init();
	neurandefrag_shmem_init();
    neurondb_llm_shmem_init();
    entrypoint_cache_shmem_init();

	elog(LOG, "neurondb: shared memory initialized for all workers");
}

