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

/* Module initialization */
void _PG_init(void);
void _PG_fini(void);

/* Shared memory hooks */
static shmem_request_hook_type prev_shmem_request_hook = NULL;
static void neurondb_shmem_request(void);
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

	/* Install shared memory request hook */
	prev_shmem_request_hook = shmem_request_hook;
	shmem_request_hook = neurondb_shmem_request;

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
	elog(LOG, "neurondb: all background workers registered successfully");
}

/*
 * Module unload callback
 */
void
_PG_fini(void)
{
	/* Restore previous hooks */
	shmem_request_hook = prev_shmem_request_hook;
	shmem_startup_hook = prev_shmem_startup_hook;

	elog(LOG, "neurondb: background workers shutting down");
}

/*
 * Shared memory request hook
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
						   neurandefrag_shmem_size());

	/* Request LWLocks */
	RequestNamedLWLockTranche("neurondb_queue", 1);
	RequestNamedLWLockTranche("neurondb_tuner", 1);
	RequestNamedLWLockTranche("neurondb_defrag", 1);
}

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

	elog(LOG, "neurondb: shared memory initialized for all workers");
}

