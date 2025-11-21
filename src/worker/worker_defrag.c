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
static int neurandefrag_naptime = 300000;
static int neurandefrag_compact_threshold = 10000;
static double neurandefrag_fragmentation_threshold = 0.3;
static bool neurandefrag_enabled = true;
static char *neurandefrag_maintenance_window = "02:00-04:00";

/* Shared state structure for defrag worker, lives in shmem */
typedef struct NeurandefragSharedState
{
	LWLock *lock;
	int64 compactions_done;
	int64 edges_cleaned;
	int64 tombstones_pruned;
	int64 rebalances_done;
	TimestampTz last_heartbeat;
	TimestampTz last_full_rebuild;
	pid_t worker_pid;
	bool in_maintenance_window;
} NeurandefragSharedState;

/* Pointer to shared memory state */
static NeurandefragSharedState *neurandefrag_state = NULL;

/* Exported worker entrypoint */
PGDLLEXPORT void neurandefrag_main(Datum main_arg);

/* Defragmentation functions */
static void scan_and_compact_indexes(void);
static void prune_tombstones(void);
static void rebalance_structures(void);
static void update_statistics(void);
static bool is_in_maintenance_window(void);
static int64 count_index_fragmentation(Oid index_oid);

static volatile sig_atomic_t got_sigterm = 0;
static volatile sig_atomic_t got_sighup = 0;
static jmp_buf segv_jmp_buf;
static volatile sig_atomic_t segv_recursed = 0;
static void neurandefrag_on_exit(int code, Datum arg);

PG_FUNCTION_INFO_V1(neurandefrag_run);

Datum
neurandefrag_run(PG_FUNCTION_ARGS)
{
	bool success = false;

	PG_TRY();
	{
		StartTransactionCommand();
		PushActiveSnapshot(GetTransactionSnapshot());

		scan_and_compact_indexes();
		prune_tombstones();
		rebalance_structures();
		update_statistics();

		PopActiveSnapshot();
		CommitTransactionCommand();
		success = true;
	}
	PG_CATCH();
	{
		if (IsTransactionState())
			AbortCurrentTransaction();
		EmitErrorReport();
		FlushErrorState();
		success = false;
	}
	PG_END_TRY();

	PG_RETURN_BOOL(success);
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
	elog(LOG,
		"neurondb: neurandefrag worker caught SIGSEGV, recovering via "
		"longjmp");
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
		300000,
		60000,
		3600000,
		PGC_SIGHUP,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomIntVariable("neurondb.neurandefrag_compact_threshold",
		"Edge count threshold for compaction trigger",
		NULL,
		&neurandefrag_compact_threshold,
		10000,
		1000,
		1000000,
		PGC_SIGHUP,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomRealVariable(
		"neurondb.neurandefrag_fragmentation_threshold",
		"Fragmentation ratio necessary to trigger a full rebuild",
		NULL,
		&neurandefrag_fragmentation_threshold,
		0.3,
		0.1,
		0.9,
		PGC_SIGHUP,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomStringVariable("neurondb.neurandefrag_maintenance_window",
		"Maintenance window in HH:MM-HH:MM format",
		NULL,
		&neurandefrag_maintenance_window,
		"02:00-04:00",
		PGC_SIGHUP,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomBoolVariable("neurondb.neurandefrag_enabled",
		"Enable/disable the Neurandefrag background worker",
		NULL,
		&neurandefrag_enabled,
		true,
		PGC_SIGHUP,
		0,
		NULL,
		NULL,
		NULL);
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

	neurandefrag_state = (NeurandefragSharedState *)ShmemInitStruct(
		"NeuronDB Defrag Worker State",
		neurandefrag_shmem_size(),
		&found);

	if (!found)
	{
		neurandefrag_state->lock =
			&(GetNamedLWLockTranche("neurondb_defrag"))->lock;
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
	sigjmp_buf local_sigjmp_buf;
	MemoryContext worker_memctx;
	MemoryContext oldctx;
	bool found;
	int wait_events;

	BackgroundWorkerUnblockSignals();
	pqsignal(SIGTERM, neurandefrag_sigterm);
	pqsignal(SIGHUP, neurandefrag_sighup);
	pqsignal(SIGSEGV, neurandefrag_segv_handler);

	CurrentResourceOwner = ResourceOwnerCreate(NULL, "neurandefrag worker");

	/* Attach to shmem state */
	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	neurandefrag_state = (NeurandefragSharedState *)ShmemInitStruct(
		"NeuronDB Defrag Worker State",
		neurandefrag_shmem_size(),
		&found);
	LWLockRelease(AddinShmemInitLock);

	if (!found || neurandefrag_state == NULL)
	{
		ereport(ERROR,
			(errmsg("neurondb: unable to attach defrag shared "
				"state")));
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
		ereport(LOG,
			(errmsg("neurondb: neurandefrag worker error, "
				"restarting loop")));
		pg_usleep(1000000L); /* sleep 1s before retry */
	}
	PG_exception_stack = &local_sigjmp_buf;

	worker_memctx = AllocSetContextCreate(TopMemoryContext,
		"neurandefrag worker context",
		ALLOCSET_DEFAULT_SIZES);

	ereport(LOG,
		(errmsg("neurondb: neurandefrag worker started (PID %d)",
			MyProcPid)));

	on_shmem_exit(neurandefrag_on_exit, (Datum)0);

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
			ereport(LOG,
				(errmsg("neurondb: neurandefrag worker "
					"processed SIGHUP")));
		}

		if (!neurandefrag_enabled)
			continue;

		oldctx = MemoryContextSwitchTo(worker_memctx);

		/* Heartbeat */
		LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
		neurandefrag_state->last_heartbeat = GetCurrentTimestamp();
		LWLockRelease(AddinShmemInitLock);

		/* Perform defragmentation tasks */
		PG_TRY();
		{
			StartTransactionCommand();
			PushActiveSnapshot(GetTransactionSnapshot());

			scan_and_compact_indexes();
			prune_tombstones();
			rebalance_structures();
			update_statistics();

			PopActiveSnapshot();
			CommitTransactionCommand();
		}
		PG_CATCH();
		{
			if (IsTransactionState())
				AbortCurrentTransaction();
			EmitErrorReport();
			FlushErrorState();
			ereport(LOG,
				(errmsg("neurondb: defrag worker error, continuing")));
		}
		PG_END_TRY();

		ereport(DEBUG1,
			(errmsg("neurondb: neurandefrag worker heartbeat (PID "
				"%d)",
				MyProcPid)));

		MemoryContextSwitchTo(oldctx);
		MemoryContextReset(worker_memctx);
	}

	ereport(LOG,
		(errmsg("neurondb: neurandefrag worker shutting down (PID %d)",
			MyProcPid)));
	proc_exit(0);
}

/*
 * Scan indexes and compact orphaned pages/edges
 */
static void
scan_and_compact_indexes(void)
{
	StringInfoData sql;
	int ret;
	int64 edges_cleaned = 0;
	int64 compactions_done = 0;

	if (SPI_connect() != SPI_OK_CONNECT)
	{
		elog(DEBUG1, "neurondb: defrag worker SPI_connect failed");
		return;
	}

	PG_TRY();
	{
		/* Find all HNSW and IVF indexes */
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT i.oid, i.relname, c.relname as table_name, "
			"       n.nspname as schema_name "
			"FROM pg_index idx "
			"JOIN pg_class i ON idx.indexrelid = i.oid "
			"JOIN pg_class c ON idx.indrelid = c.oid "
			"JOIN pg_namespace n ON i.relnamespace = n.oid "
			"JOIN pg_am a ON i.relam = a.oid "
			"WHERE a.amname IN ('hnsw', 'ivfflat') "
			"  AND i.relkind = 'i' "
			"  AND n.nspname NOT IN ('pg_catalog', 'information_schema') "
			"ORDER BY i.relname");

		ret = SPI_execute(sql.data, true, 0);

		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			int i;

			for (i = 0; i < (int)SPI_processed; i++)
			{
				bool isnull;
				Oid index_oid;
				char *index_name;
				char *table_name;
				char *schema_name;
				int64 fragmentation;

				index_oid = DatumGetObjectId(
					SPI_getbinval(SPI_tuptable->vals[i],
						SPI_tuptable->tupdesc,
						1,
						&isnull));
				index_name = SPI_getvalue(SPI_tuptable->vals[i],
					SPI_tuptable->tupdesc,
					2);
				table_name = SPI_getvalue(SPI_tuptable->vals[i],
					SPI_tuptable->tupdesc,
					3);
				schema_name = SPI_getvalue(SPI_tuptable->vals[i],
					SPI_tuptable->tupdesc,
					4);

				if (isnull || !index_name || !table_name
					|| !schema_name)
					continue;

				/* Check fragmentation level */
				fragmentation = count_index_fragmentation(index_oid);

				/* If fragmentation exceeds threshold, perform VACUUM */
				if (fragmentation > neurandefrag_compact_threshold)
				{
					resetStringInfo(&sql);
					appendStringInfo(&sql,
						"VACUUM ANALYZE %s.%s",
						schema_name,
						index_name);

					ret = SPI_execute(sql.data, false, 0);
					if (ret == SPI_OK_UTILITY)
					{
						compactions_done++;
						edges_cleaned += fragmentation;
						ereport(DEBUG1,
							(errmsg("neurondb: compacted index %s.%s "
								"(fragmentation: " NDB_INT64_FMT
								")",
								schema_name,
								index_name,
								NDB_INT64_CAST(fragmentation))));
					}
				}

				pfree(index_name);
				pfree(table_name);
				pfree(schema_name);
			}
		}

		pfree(sql.data);
	}
	PG_CATCH();
	{
		EmitErrorReport();
		FlushErrorState();
		elog(LOG,
			"neurondb: error in scan_and_compact_indexes, continuing");
	}
	PG_END_TRY();

	SPI_finish();

	/* Update shared state */
	if (compactions_done > 0 || edges_cleaned > 0)
	{
		LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
		neurandefrag_state->compactions_done += compactions_done;
		neurandefrag_state->edges_cleaned += edges_cleaned;
		LWLockRelease(AddinShmemInitLock);
	}
}

/*
 * Prune tombstones (deleted entries) from indexes
 */
static void
prune_tombstones(void)
{
	StringInfoData sql;
	int ret;
	int64 tombstones_pruned = 0;

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	PG_TRY();
	{
		/* Check if index statistics table exists */
		ret = SPI_execute(
			"SELECT 1 FROM pg_tables WHERE schemaname = 'neurondb' "
			"AND tablename = 'neurondb_index_stats'",
			true,
			0);
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			SPI_finish();
			return;
		}

		/* Find indexes with high tombstone counts */
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT index_oid, index_name, schema_name, "
			"       tombstone_count "
			"FROM neurondb.neurondb_index_stats "
			"WHERE tombstone_count > 100 "
			"ORDER BY tombstone_count DESC "
			"LIMIT 10");

		ret = SPI_execute(sql.data, true, 0);

		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			int i;

			for (i = 0; i < (int)SPI_processed; i++)
			{
				bool isnull;
				char *index_name;
				char *schema_name;
				int64 tombstone_count;

				(void) DatumGetObjectId(
					SPI_getbinval(SPI_tuptable->vals[i],
						SPI_tuptable->tupdesc,
						1,
						&isnull));
				index_name = SPI_getvalue(SPI_tuptable->vals[i],
					SPI_tuptable->tupdesc,
					2);
				schema_name = SPI_getvalue(SPI_tuptable->vals[i],
					SPI_tuptable->tupdesc,
					3);
				tombstone_count = DatumGetInt64(
					SPI_getbinval(SPI_tuptable->vals[i],
						SPI_tuptable->tupdesc,
						4,
						&isnull));

				if (isnull || !index_name || !schema_name)
					continue;

				/* Perform VACUUM to remove tombstones */
				resetStringInfo(&sql);
				appendStringInfo(&sql,
					"VACUUM %s.%s",
					schema_name,
					index_name);

				ret = SPI_execute(sql.data, false, 0);
				if (ret == SPI_OK_UTILITY)
				{
					tombstones_pruned += tombstone_count;
					ereport(DEBUG1,
						(errmsg("neurondb: pruned " NDB_INT64_FMT
							" tombstones from %s.%s",
							NDB_INT64_CAST(tombstone_count),
							schema_name,
							index_name)));
				}

				pfree(index_name);
				pfree(schema_name);
			}
		}

		pfree(sql.data);
	}
	PG_CATCH();
	{
		EmitErrorReport();
		FlushErrorState();
		elog(LOG,
			"neurondb: error in prune_tombstones, continuing");
	}
	PG_END_TRY();

	SPI_finish();

	/* Update shared state */
	if (tombstones_pruned > 0)
	{
		LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
		neurandefrag_state->tombstones_pruned += tombstones_pruned;
		LWLockRelease(AddinShmemInitLock);
	}
}

/*
 * Rebalance index structures if needed
 */
static void
rebalance_structures(void)
{
	StringInfoData sql;
	int ret;
	int64 rebalances_done = 0;
	double fragmentation_ratio;

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	PG_TRY();
	{
		/* Check if we're in maintenance window */
		if (!is_in_maintenance_window())
		{
			SPI_finish();
			return;
		}

		/* Find indexes with high fragmentation ratio */
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT i.oid, i.relname, n.nspname, "
			"       pg_relation_size(i.oid) as index_size, "
			"       pg_total_relation_size(c.oid) as table_size "
			"FROM pg_index idx "
			"JOIN pg_class i ON idx.indexrelid = i.oid "
			"JOIN pg_class c ON idx.indrelid = c.oid "
			"JOIN pg_namespace n ON i.relnamespace = n.oid "
			"JOIN pg_am a ON i.relam = a.oid "
			"WHERE a.amname IN ('hnsw', 'ivfflat') "
			"  AND i.relkind = 'i' "
			"  AND n.nspname NOT IN ('pg_catalog', 'information_schema') "
			"  AND pg_relation_size(i.oid) > 1048576 "
			"ORDER BY pg_relation_size(i.oid) DESC "
			"LIMIT 5");

		ret = SPI_execute(sql.data, true, 0);

		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			int i;

			for (i = 0; i < (int)SPI_processed; i++)
			{
				bool isnull;
				char *index_name;
				char *schema_name;
				int64 index_size;
				int64 table_size;

				index_name = SPI_getvalue(SPI_tuptable->vals[i],
					SPI_tuptable->tupdesc,
					2);
				schema_name = SPI_getvalue(SPI_tuptable->vals[i],
					SPI_tuptable->tupdesc,
					3);
				index_size = DatumGetInt64(
					SPI_getbinval(SPI_tuptable->vals[i],
						SPI_tuptable->tupdesc,
						4,
						&isnull));
				table_size = DatumGetInt64(
					SPI_getbinval(SPI_tuptable->vals[i],
						SPI_tuptable->tupdesc,
						5,
						&isnull));

				if (!index_name || !schema_name || table_size == 0)
					continue;

				/* Calculate fragmentation ratio */
				fragmentation_ratio =
					(double)index_size / (double)table_size;

				/* If fragmentation exceeds threshold, consider rebuild */
				if (fragmentation_ratio
					> neurandefrag_fragmentation_threshold)
				{
					/* Perform REINDEX in maintenance window */
					resetStringInfo(&sql);
					appendStringInfo(&sql,
						"REINDEX INDEX CONCURRENTLY %s.%s",
						schema_name,
						index_name);

					ret = SPI_execute(sql.data, false, 0);
					if (ret == SPI_OK_UTILITY)
					{
						rebalances_done++;
						ereport(LOG,
							(errmsg("neurondb: rebalanced index %s.%s "
								"(fragmentation: %.2f%%)",
								schema_name,
								index_name,
								fragmentation_ratio * 100.0)));
					}
				}

				pfree(index_name);
				pfree(schema_name);
			}
		}

		pfree(sql.data);
	}
	PG_CATCH();
	{
		EmitErrorReport();
		FlushErrorState();
		elog(LOG,
			"neurondb: error in rebalance_structures, continuing");
	}
	PG_END_TRY();

	SPI_finish();

	/* Update shared state */
	if (rebalances_done > 0)
	{
		LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
		neurandefrag_state->rebalances_done += rebalances_done;
		if (rebalances_done > 0)
			neurandefrag_state->last_full_rebuild =
				GetCurrentTimestamp();
		LWLockRelease(AddinShmemInitLock);
	}
}

/*
 * Update defragmentation statistics
 */
static void
update_statistics(void)
{
	StringInfoData sql;
	int ret;

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	PG_TRY();
	{
		/* Check if statistics table exists */
		ret = SPI_execute(
			"SELECT 1 FROM pg_tables WHERE schemaname = 'neurondb' "
			"AND tablename = 'neurondb_index_stats'",
			true,
			0);
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			SPI_finish();
			return;
		}

		/* Update index statistics */
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"INSERT INTO neurondb.neurondb_index_stats "
			"(index_oid, index_name, schema_name, "
			" fragmentation_ratio, last_maintenance) "
			"SELECT i.oid, i.relname, n.nspname, "
			"       CASE WHEN pg_total_relation_size(c.oid) > 0 "
			"            THEN pg_relation_size(i.oid)::float / "
			"                 pg_total_relation_size(c.oid)::float "
			"            ELSE 0.0 END, "
			"       now() "
			"FROM pg_index idx "
			"JOIN pg_class i ON idx.indexrelid = i.oid "
			"JOIN pg_class c ON idx.indrelid = c.oid "
			"JOIN pg_namespace n ON i.relnamespace = n.oid "
			"JOIN pg_am a ON i.relam = a.oid "
			"WHERE a.amname IN ('hnsw', 'ivfflat') "
			"  AND i.relkind = 'i' "
			"  AND n.nspname NOT IN ('pg_catalog', 'information_schema') "
			"ON CONFLICT (index_oid) DO UPDATE SET "
			"  fragmentation_ratio = EXCLUDED.fragmentation_ratio, "
			"  last_maintenance = EXCLUDED.last_maintenance");

		SPI_execute(sql.data, false, 0);

		pfree(sql.data);
	}
	PG_CATCH();
	{
		EmitErrorReport();
		FlushErrorState();
		elog(LOG,
			"neurondb: error in update_statistics, continuing");
	}
	PG_END_TRY();

	SPI_finish();
}

/*
 * Check if current time is within maintenance window
 */
static bool
is_in_maintenance_window(void)
{
	bool in_window = false;
	TimestampTz now;
	int hour;
	int minute;
	char window_start[6];
	char window_end[6];
	int start_hour, start_min, end_hour, end_min;

	if (!neurandefrag_maintenance_window)
		return true;

	now = GetCurrentTimestamp();
	hour = (int)((now / 3600000000LL) % 24);
	minute = (int)((now / 60000000LL) % 60);

	/* Parse maintenance window (format: "HH:MM-HH:MM") */
	if (sscanf(neurandefrag_maintenance_window,
		"%5[^-]-%5s",
		window_start,
		window_end) != 2)
		return true;

	if (sscanf(window_start, "%d:%d", &start_hour, &start_min) != 2)
		return true;
	if (sscanf(window_end, "%d:%d", &end_hour, &end_min) != 2)
		return true;

	/* Check if current time is within window */
	if (start_hour < end_hour)
	{
		in_window = (hour > start_hour
			|| (hour == start_hour && minute >= start_min))
			&& (hour < end_hour
				|| (hour == end_hour && minute <= end_min));
	} else
	{
		/* Window spans midnight */
		in_window = (hour > start_hour
			|| (hour == start_hour && minute >= start_min))
			|| (hour < end_hour
				|| (hour == end_hour && minute <= end_min));
	}

	/* Update shared state */
	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	neurandefrag_state->in_maintenance_window = in_window;
	LWLockRelease(AddinShmemInitLock);

	return in_window;
}

/*
 * Count index fragmentation (orphaned edges/pages)
 */
static int64
count_index_fragmentation(Oid index_oid)
{
	StringInfoData sql;
	int ret;
	int64 fragmentation = 0;

	if (SPI_connect() != SPI_OK_CONNECT)
		return 0;

	PG_TRY();
	{
		/* Estimate fragmentation based on index bloat */
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT pg_relation_size(%u) - "
			"       pg_relation_size(%u) * "
			"       (SELECT COALESCE(reltuples, 0)::float / "
			"               NULLIF(relpages, 0)::float "
			"        FROM pg_class WHERE oid = %u) / "
			"       (SELECT COALESCE(reltuples, 0)::float / "
			"               NULLIF(relpages, 0)::float "
			"        FROM pg_class WHERE oid = %u "
			"        ORDER BY reltuples DESC LIMIT 1) "
			"       AS fragmentation",
			index_oid,
			index_oid,
			index_oid,
			index_oid);

		ret = SPI_execute(sql.data, true, 1);

		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool isnull;
			Datum frag_datum;

			frag_datum = SPI_getbinval(SPI_tuptable->vals[0],
				SPI_tuptable->tupdesc,
				1,
				&isnull);
			if (!isnull)
				fragmentation = DatumGetInt64(frag_datum);
		}

		pfree(sql.data);
	}
	PG_CATCH();
	{
		FlushErrorState();
	}
	PG_END_TRY();

	SPI_finish();

	/* Fallback: use index size as proxy for fragmentation via SPI */
	if (fragmentation <= 0)
	{
		if (SPI_connect() == SPI_OK_CONNECT)
		{
			PG_TRY();
			{
				resetStringInfo(&sql);
				appendStringInfo(&sql,
					"SELECT pg_relation_size(%u)",
					index_oid);
				ret = SPI_execute(sql.data, true, 1);
				if (ret == SPI_OK_SELECT && SPI_processed > 0)
				{
					bool isnull;
					Datum size_datum;

					size_datum = SPI_getbinval(SPI_tuptable->vals[0],
						SPI_tuptable->tupdesc,
						1,
						&isnull);
					if (!isnull)
						fragmentation = DatumGetInt64(size_datum) / 1024;
				}
				pfree(sql.data);
			}
			PG_CATCH();
			{
				FlushErrorState();
			}
			PG_END_TRY();
			SPI_finish();
		}
	}

	return Max(fragmentation, 0);
}
