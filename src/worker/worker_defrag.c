/*
 *-------------------------------------------------------------------------
 *
 * worker_defrag.c
 *    Background worker for NeuronDB: neurandefrag - Index maintenance.
 *
 * Handles periodic HNSW index compaction, edge cleanup, level rebalancing,
 * tombstone pruning, and maintenance window scheduling for optimal performance.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/worker/worker_defrag.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb_compat.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "pgstat.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/lwlock.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "executor/spi.h"
#include "libpq/pqsignal.h"
#include "utils/guc.h"
#include "utils/timestamp.h"
#include "utils/builtins.h"
#include "utils/snapmgr.h"
#include "utils/memutils.h"
#include "lib/stringinfo.h"
#include "catalog/pg_type.h"
#include "access/xact.h"
#include "commands/vacuum.h"
#include <setjmp.h>
#include <signal.h>
#include <errno.h>

#include "neurondb_bgworkers.h"

/* Defrag configuration parameters (GUCs; see also neurandefrag_init_guc) */
static int      neurandefrag_naptime = 300000;
static int      neurandefrag_compact_threshold = 10000;
static double   neurandefrag_fragmentation_threshold = 0.3;
static bool     neurandefrag_enabled = true;
static char    *neurandefrag_maintenance_window = "02:00-04:00";

/* Shared state structure for defrag worker, lives in shmem */
typedef struct NeurandefragSharedState
{
	LWLock       *lock;
	int64         compactions_done;
	int64         edges_cleaned;
	int64         tombstones_pruned;
	int64         rebalances_done;
	TimestampTz   last_heartbeat;
	TimestampTz   last_full_rebuild;
	pid_t         worker_pid;
	bool          in_maintenance_window;
} NeurandefragSharedState;

/* Pointer to shared memory state */
static NeurandefragSharedState *neurandefrag_state = NULL;

/* Exported worker entrypoint */
PGDLLEXPORT void neurandefrag_main(Datum main_arg);

static volatile sig_atomic_t got_sigterm = 0;
static volatile sig_atomic_t got_sighup = 0;
static jmp_buf segv_jmp_buf;
static volatile sig_atomic_t segv_recursed = 0;
static void neurandefrag_on_exit(int code, Datum arg);

PG_FUNCTION_INFO_V1(neurandefrag_run);

Datum
neurandefrag_run(PG_FUNCTION_ARGS)
{
	elog(NOTICE, "neurondb: neurandefrag_run invoked (stub implementation)");
	PG_RETURN_BOOL(false);
}

/* SIGTERM: request orderly shutdown */
static void __attribute__((unused))
neurandefrag_sigterm(SIGNAL_ARGS)
{
	int save_errno = errno;
	got_sigterm = 1;
	if (MyLatch)
		SetLatch(MyLatch);
	errno = save_errno;
}

/* SIGHUP: reload config */
static void __attribute__((unused))
neurandefrag_sighup(SIGNAL_ARGS)
{
	int save_errno = errno;
	got_sighup = 1;
	if (MyLatch)
		SetLatch(MyLatch);
	errno = save_errno;
}

/* SIGSEGV handler: try to recover from crash in worker logic */
static void __attribute__((unused))
neurandefrag_segv_handler(int signum)
{
	if (segv_recursed)
	{
		/* Already handling a SEGV: give up and terminate */
		signal(signum, SIG_DFL);
		raise(signum);
		return;
	}
	segv_recursed = 1;
	elog(LOG, "neurondb: neurandefrag worker caught SIGSEGV, recovering via longjmp");
	longjmp(segv_jmp_buf, 1);
}

/* Register all tunable parameters (GUCs) for the defrag worker. */
void
neurandefrag_init_guc(void)
{
	DefineCustomIntVariable("neurondb.neurandefrag_naptime",
							"Duration between maintenance cycles (ms)",
							NULL,
							&neurandefrag_naptime,
							300000, 60000, 3600000,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomIntVariable("neurondb.neurandefrag_compact_threshold",
							"Edge count threshold for compaction trigger",
							NULL,
							&neurandefrag_compact_threshold,
							10000, 1000, 1000000,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomRealVariable("neurondb.neurandefrag_fragmentation_threshold",
							 "Fragmentation ratio necessary to trigger a full rebuild",
							 NULL,
							 &neurandefrag_fragmentation_threshold,
							 0.3, 0.1, 0.9,
							 PGC_SIGHUP,
							 0,
							 NULL, NULL, NULL);

	DefineCustomStringVariable("neurondb.neurandefrag_maintenance_window",
							   "Maintenance window in HH:MM-HH:MM format",
							   NULL,
							   &neurandefrag_maintenance_window,
							   "02:00-04:00",
							   PGC_SIGHUP,
							   0,
							   NULL, NULL, NULL);

	DefineCustomBoolVariable("neurondb.neurandefrag_enabled",
							 "Enable/disable the Neurandefrag background worker",
							 NULL,
							 &neurandefrag_enabled,
							 true,
							 PGC_SIGHUP,
							 0,
							 NULL, NULL, NULL);
}

/* Shared memory size required for the worker */
Size
neurandefrag_shmem_size(void)
{
	return MAXALIGN(sizeof(NeurandefragSharedState));
}

/* Initialize or attach to shared memory state for defrag worker */
void
neurandefrag_shmem_init(void)
{
	bool found;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);

	neurandefrag_state = (NeurandefragSharedState *) ShmemInitStruct(
		"NeuronDB Defrag Worker State",
		neurandefrag_shmem_size(),
		&found
	);

	if (!found)
	{
		neurandefrag_state->lock = &(GetNamedLWLockTranche("neurondb_defrag"))->lock;
		neurandefrag_state->compactions_done = 0;
		neurandefrag_state->edges_cleaned = 0;
		neurandefrag_state->tombstones_pruned = 0;
		neurandefrag_state->rebalances_done = 0;
		neurandefrag_state->last_heartbeat = GetCurrentTimestamp();
		neurandefrag_state->last_full_rebuild = 0;
		neurandefrag_state->worker_pid = 0;
		neurandefrag_state->in_maintenance_window = false;
	}

	LWLockRelease(AddinShmemInitLock);
}

static void
neurandefrag_on_exit(int code, Datum arg)
{
	if (neurandefrag_state == NULL)
		return;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	if (neurandefrag_state != NULL)
	{
		neurandefrag_state->worker_pid = 0;
		neurandefrag_state->in_maintenance_window = false;
	}
	LWLockRelease(AddinShmemInitLock);
}

void
neurandefrag_main(Datum main_arg)
{
	sigjmp_buf	local_sigjmp_buf;
	MemoryContext	worker_memctx;
	MemoryContext	oldctx;
	bool		found;
	int			wait_events;

	BackgroundWorkerUnblockSignals();
	pqsignal(SIGTERM, neurandefrag_sigterm);
	pqsignal(SIGHUP, neurandefrag_sighup);
	pqsignal(SIGSEGV, neurandefrag_segv_handler);

	CurrentResourceOwner = ResourceOwnerCreate(NULL, "neurandefrag worker");

	/* Attach to shmem state */
	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	neurandefrag_state = (NeurandefragSharedState *) ShmemInitStruct(
		"NeuronDB Defrag Worker State",
		neurandefrag_shmem_size(),
		&found
	);
	LWLockRelease(AddinShmemInitLock);

	if (!found || neurandefrag_state == NULL)
	{
		ereport(ERROR, (errmsg("neurondb: unable to attach defrag shared state")));
		proc_exit(1);
	}

	/* Set up longjmp exception handler */
	if (sigsetjmp(local_sigjmp_buf, 1) != 0)
	{
		/* Error occurred, reset and cleanup */
		FlushErrorState();
		MemoryContextSwitchTo(TopMemoryContext);
		if (worker_memctx)
			MemoryContextReset(worker_memctx);
		ereport(LOG, (errmsg("neurondb: neurandefrag worker error, restarting loop")));
		pg_usleep(1000000L); /* sleep 1s before retry */
	}
	PG_exception_stack = &local_sigjmp_buf;

	worker_memctx = AllocSetContextCreate(TopMemoryContext,
		"neurandefrag worker context",
		ALLOCSET_DEFAULT_SIZES);

	ereport(LOG, (errmsg("neurondb: neurandefrag worker started (PID %d)", MyProcPid)));

	on_shmem_exit(neurandefrag_on_exit, (Datum) 0);

	while (!got_sigterm)
	{
		long timeout_ms = Max(neurandefrag_naptime, 1000);
		wait_events = WaitLatch(MyLatch,
							 WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
							 timeout_ms,
							 PG_WAIT_EXTENSION);
		ResetLatch(MyLatch);
		CHECK_FOR_INTERRUPTS();

		if (wait_events & WL_POSTMASTER_DEATH)
			proc_exit(1);

		if (got_sigterm)
			break;

		if (got_sighup)
		{
			got_sighup = 0;
			ProcessConfigFile(PGC_SIGHUP);
			ereport(LOG, (errmsg("neurondb: neurandefrag worker processed SIGHUP")));
		}

		if (!neurandefrag_enabled)
			continue;

		oldctx = MemoryContextSwitchTo(worker_memctx);

		/* Heartbeat */
		LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
		neurandefrag_state->last_heartbeat = GetCurrentTimestamp();
		LWLockRelease(AddinShmemInitLock);

		/* TODO: Implement main defragmentation logic here:
		 *  - scan indexes for tombstones
		 *  - compact orphaned pages/edges
		 *  - rebalance structures if needed
		 *  - update shared state counters/statistics
		 */
		ereport(DEBUG1, (errmsg("neurondb: neurandefrag worker heartbeat (PID %d)", MyProcPid)));

		MemoryContextSwitchTo(oldctx);
		MemoryContextReset(worker_memctx);
	}

	ereport(LOG, (errmsg("neurondb: neurandefrag worker shutting down (PID %d)", MyProcPid)));
	proc_exit(0);
}
