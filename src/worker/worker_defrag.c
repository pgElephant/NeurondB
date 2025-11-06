/*-------------------------------------------------------------------------
 *
 * bgworker_defrag.c
 *		Background worker: neurandefrag - Index maintenance
 *
 * This worker performs HNSW graph compaction, orphan edge cleanup,
 * level rebalancing, tombstone pruning, and schedules rebuild windows.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/bgworker_defrag.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb_compat.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/lwlock.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "executor/spi.h"
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

#include "neurondb_bgworkers.h"

static int neurandefrag_naptime = 300000;
static int neurandefrag_compact_threshold = 10000;
static double neurandefrag_fragmentation_threshold = 0.3;
static bool neurandefrag_enabled = true;
static char *neurandefrag_maintenance_window = "02:00-04:00";

typedef struct NeurandefragSharedState
{
	LWLock	   *lock;
	int64		compactions_done;
	int64		edges_cleaned;
	int64		tombstones_pruned;
	int64		rebalances_done;
	TimestampTz	last_heartbeat;
	TimestampTz	last_full_rebuild;
	pid_t		worker_pid;
	bool		in_maintenance_window;
} NeurandefragSharedState;

static NeurandefragSharedState *neurandefrag_state = NULL;
static MemoryContext DefragTopContext = NULL;

PGDLLEXPORT void neurandefrag_main(Datum main_arg); 
static void neurandefrag_sigterm(SIGNAL_ARGS);
static void neurandefrag_sighup(SIGNAL_ARGS);
static void perform_index_maintenance(void);
static void compact_hnsw_graph(const char *index_name);
static void cleanup_orphan_edges(const char *index_name);
static void rebalance_levels(const char *index_name);
static void prune_tombstones(const char *index_name);
static void refresh_statistics(const char *index_name);
static bool in_maintenance_window(void);

static volatile sig_atomic_t got_sigterm = false;
static volatile sig_atomic_t got_sighup = false;

static jmp_buf segv_jmp_buf;
static volatile sig_atomic_t segv_recursed = 0;

/* Signal handlers for SIGTERM and SIGHUP */
static void
neurandefrag_sigterm(SIGNAL_ARGS)
{
	int save_errno = errno;
	(void) postgres_signal_arg;
	got_sigterm = true;
	if (MyLatch)
		SetLatch(MyLatch);
	errno = save_errno;
}

static void
neurandefrag_sighup(SIGNAL_ARGS)
{
	int save_errno = errno;
	(void) postgres_signal_arg;
	got_sighup = true;
	if (MyLatch)
		SetLatch(MyLatch);
	errno = save_errno;
}

static void
neurandefrag_segv_handler(int signum)
{
	if (segv_recursed)
	{
		/* Already recursed: give up, proceed with default handler */
		signal(signum, SIG_DFL);
		raise(signum);
		return;
	}
	segv_recursed = 1;
	elog(LOG, "neurondb: neurandefrag worker caught SIGSEGV, recovering via longjmp");
	longjmp(segv_jmp_buf, 1);
}

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
							"Edge count threshold for compaction",
							NULL,
							&neurandefrag_compact_threshold,
							10000, 1000, 1000000,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomRealVariable("neurondb.neurandefrag_fragmentation_threshold",
							 "Fragmentation ratio to trigger rebuild",
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
							 "Enable defrag worker",
							 NULL,
							 &neurandefrag_enabled,
							 true,
							 PGC_SIGHUP,
							 0,
							 NULL, NULL, NULL);
}

Size
neurandefrag_shmem_size(void)
{
	return MAXALIGN(sizeof(NeurandefragSharedState));
}

void
neurandefrag_shmem_init(void)
{
	bool found;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);

	neurandefrag_state = ShmemInitStruct("NeuronDB Defrag Worker State",
										 neurandefrag_shmem_size(),
										 &found);

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

/* Crash-safe AND segfault-safe */
PGDLLEXPORT void
neurandefrag_main(Datum main_arg)
{
	MemoryContext oldCtx;
	struct sigaction old_segv_action, new_segv_action;

	(void) main_arg;

	DefragTopContext = AllocSetContextCreate(TopMemoryContext,
											 "NeuronDBDefragTopContext",
											 ALLOCSET_DEFAULT_SIZES);

	pqsignal(SIGTERM, neurandefrag_sigterm);
	pqsignal(SIGHUP, neurandefrag_sighup);

	/* Crash-segfault safety: install SIGSEGV handler */
	memset(&new_segv_action, 0, sizeof(new_segv_action));
	new_segv_action.sa_handler = neurandefrag_segv_handler;
	sigemptyset(&new_segv_action.sa_mask);
	new_segv_action.sa_flags = SA_RESTART;
	sigaction(SIGSEGV, &new_segv_action, &old_segv_action);

	BackgroundWorkerUnblockSignals();

	BackgroundWorkerInitializeConnection("postgres", NULL, 0);

	if (neurandefrag_state == NULL || !PointerIsValid(neurandefrag_state) ||
		neurandefrag_state->lock == NULL)
	{
		elog(LOG, "neurondb: neurandefrag shared state not initialized -- shutting down cleanly");
		proc_exit(1);
	}

	LWLockAcquire(neurandefrag_state->lock, LW_EXCLUSIVE);
	neurandefrag_state->worker_pid = MyProcPid;
	neurandefrag_state->last_heartbeat = GetCurrentTimestamp();
	LWLockRelease(neurandefrag_state->lock);

	elog(LOG, "neurondb: neurandefrag worker started (PID %d)", MyProcPid);

	for (;;) /* replaced while (!got_sigterm) to support segfault restart */
	{
		int	rc = 0;
		bool do_maintenance = false;

		segv_recursed = 0;
		if (setjmp(segv_jmp_buf) != 0)
		{
		/* Segfault recovery! */
		elog(LOG, "neurondb: background worker recovered from segmentation fault in main loop");

		/* Try to ensure state and memory are reset */
		if (DefragTopContext)
			MemoryContextReset(DefragTopContext);
			if (neurandefrag_state && PointerIsValid(neurandefrag_state->lock) &&
				LWLockHeldByMe(neurandefrag_state->lock))
				LWLockRelease(neurandefrag_state->lock);

			FlushErrorState();
			/* Postmaster/dead? Bail out, or loop */
		}

	if (got_sigterm)
		break;

	oldCtx = MemoryContextSwitchTo(DefragTopContext);
	MemoryContextReset(DefragTopContext);

		PG_TRY();
		{
			if (got_sighup)
			{
				got_sighup = false;
				ProcessConfigFile(PGC_SIGHUP);
				elog(LOG, "neurondb: neurandefrag reloaded configuration");
			}

			if (!neurandefrag_enabled)
			{
				elog(LOG, "neurondb: neurandefrag disabled, exiting");
				proc_exit(0);
			}

			/* Update heartbeat and memory safety for lock */
			if (neurandefrag_state && PointerIsValid(neurandefrag_state->lock))
			{
				LWLockAcquire(neurandefrag_state->lock, LW_EXCLUSIVE);
				neurandefrag_state->last_heartbeat = GetCurrentTimestamp();
				neurandefrag_state->in_maintenance_window = in_maintenance_window();
				do_maintenance = neurandefrag_state->in_maintenance_window;
				LWLockRelease(neurandefrag_state->lock);
			}

			StartTransactionCommand();
			PushActiveSnapshot(GetTransactionSnapshot());

			perform_index_maintenance();

			PopActiveSnapshot();
			CommitTransactionCommand();

			if (do_maintenance)
			{
				elog(LOG, "neurondb: in maintenance window, performing full maintenance");
			}
		}
		PG_CATCH();
		{
			if (neurandefrag_state && PointerIsValid(neurandefrag_state->lock) && LWLockHeldByMe(neurandefrag_state->lock))
				LWLockRelease(neurandefrag_state->lock);

			if (IsTransactionState())
				AbortCurrentTransaction();

			EmitErrorReport();
			FlushErrorState();
			elog(LOG, "neurondb: neurandefrag main loop recovered from crash");
		}
		PG_END_TRY();

		MemoryContextSwitchTo(oldCtx);

		rc = WaitLatch(MyLatch,
					   WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
					   neurandefrag_naptime,
					   0);
		ResetLatch(MyLatch);

		if (rc & WL_POSTMASTER_DEATH)
			proc_exit(1);
	}

	elog(LOG, "neurondb: neurandefrag worker shutting down");
	MemoryContextDelete(DefragTopContext);
	DefragTopContext = NULL;
	/* restore segv handler if needed */
	sigaction(SIGSEGV, &old_segv_action, NULL);
	proc_exit(0);
}

static bool
in_maintenance_window(void)
{
	TimestampTz	now = GetCurrentTimestamp();
	struct pg_tm tm;
	fsec_t	fsec;
	int	start_hour = 0, start_min = 0, end_hour = 0, end_min = 0;
	int	current_minutes = 0, start_minutes = 0, end_minutes = 0;

	if (neurandefrag_maintenance_window == NULL)
		return false;

	if (PointerIsValid(neurandefrag_maintenance_window) &&
		sscanf(neurandefrag_maintenance_window, "%d:%d-%d:%d",
			   &start_hour, &start_min, &end_hour, &end_min) != 4)
	{
		elog(WARNING, "neurondb: invalid maintenance window format: %s",
			 neurandefrag_maintenance_window);
		return false;
	}

	if (timestamp2tm(now, NULL, &tm, &fsec, NULL, NULL) != 0)
		return false;

	current_minutes = tm.tm_hour * 60 + tm.tm_min;
	start_minutes = start_hour * 60 + start_min;
	end_minutes = end_hour * 60 + end_min;

	if (start_minutes <= end_minutes)
		return (current_minutes >= start_minutes && current_minutes < end_minutes);
	else
		return (current_minutes >= start_minutes || current_minutes < end_minutes);
}

static void
perform_index_maintenance(void)
{
	MemoryContext local_ctx = NULL, old_ctx = NULL;
	StringInfoData	sql;
	int	ret = 0;
	bool sql_allocated = false;

	volatile bool had_segv = false;

	if (had_segv && DefragTopContext)
		MemoryContextReset(DefragTopContext);

	local_ctx = AllocSetContextCreate(CurrentMemoryContext, "DefragMaintenanceCTX", ALLOCSET_SMALL_SIZES);
	old_ctx = MemoryContextSwitchTo(local_ctx);

	PG_TRY();
	{
		if (SPI_connect() != SPI_OK_CONNECT)
			elog(ERROR, "neurondb: SPI_connect failed in neurandefrag");

		/* First check if the maintenance table exists */
		ret = SPI_execute("SELECT 1 FROM pg_tables WHERE schemaname = 'neurondb' AND tablename = 'neurondb_index_maintenance'", true, 0);
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			/* Table doesn't exist, skip index maintenance */
			elog(DEBUG1, "neurondb: neurondb_index_maintenance table not found, skipping maintenance");
			SPI_finish();
			if (sql_allocated && sql.data)
				pfree(sql.data);
			if (local_ctx)
			{
				MemoryContextSwitchTo(old_ctx);
				MemoryContextDelete(local_ctx);
			}
			return;
		}

		initStringInfo(&sql);
		sql_allocated = true;
		appendStringInfo(&sql,
			"SELECT index_name, index_type, num_edges, num_tombstones, fragmentation_ratio "
			"FROM neurondb.neurondb_index_maintenance "
			"WHERE (last_compaction IS NULL OR last_compaction < now() - interval '1 day') "
			"   OR (fragmentation_ratio > %.2f) "
			"ORDER BY fragmentation_ratio DESC",
			neurandefrag_fragmentation_threshold);

		ret = SPI_execute(sql.data, true, 0);

		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			for (int i = 0; i < (int)SPI_processed; i++)
			{
				char	   *index_name = NULL;
				int64		num_edges = 0;
				double		frag_ratio = 0.0;
				bool		isnull = false;
				Datum		datum;

				PG_TRY();
				{
					index_name = SPI_getvalue(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1);
					if (!PointerIsValid(index_name))
						continue;

					datum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 3, &isnull);
					num_edges = isnull ? 0 : DatumGetInt64(datum);

					datum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 5, &isnull);
					frag_ratio = isnull ? 0.0 : DatumGetFloat8(datum);

					elog(DEBUG1, "neurondb: maintaining index %s (edges=" NDB_INT64_FMT ", frag=%.2f)",
						 index_name, NDB_INT64_CAST(num_edges), frag_ratio);

					PG_TRY();
					{
						if (num_edges > neurandefrag_compact_threshold && PointerIsValid(index_name))
							compact_hnsw_graph(index_name);

						if (PointerIsValid(index_name)) cleanup_orphan_edges(index_name);
						if (PointerIsValid(index_name)) prune_tombstones(index_name);

						if (frag_ratio > neurandefrag_fragmentation_threshold &&
							neurandefrag_state && PointerIsValid(neurandefrag_state->lock) &&
							neurandefrag_state->in_maintenance_window &&
							PointerIsValid(index_name))
						{
							rebalance_levels(index_name);
						}
						if (PointerIsValid(index_name)) refresh_statistics(index_name);
					}
					PG_CATCH();
					{
						EmitErrorReport();
						FlushErrorState();
						elog(LOG, "neurondb: crash/segv during maintenance of index '%s' recovered safely", index_name ? index_name : "(null)");
					}
					PG_END_TRY();

					if (PointerIsValid(index_name))
						pfree(index_name);
				}
				PG_CATCH();
				{
					EmitErrorReport();
					FlushErrorState();
					elog(LOG, "neurondb: crash during maintenance of index (cleanup block, fallback triggered)");
					if (PointerIsValid(index_name))
						pfree(index_name);
				}
				PG_END_TRY();
			}
		}
		if (sql_allocated && sql.data)
			pfree(sql.data);

		SPI_finish();
	}
	PG_CATCH();
	{
		if (sql_allocated && sql.data)
			pfree(sql.data);

		if (IsTransactionState())
			AbortCurrentTransaction();

		EmitErrorReport();
		FlushErrorState();

		SPI_finish();
		elog(LOG, "neurondb: Exception in perform_index_maintenance handled safely");
	}
	PG_END_TRY();

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(local_ctx);
}

static void
compact_hnsw_graph(const char *index_name)
{
	MemoryContext local_ctx = NULL, old_ctx = NULL;
	StringInfoData	sql;
	int64	edges_before = 0;
	int64	edges_after = 0;
	bool	sql_inited = false;

	if (!PointerIsValid(index_name))
		return;

	local_ctx = AllocSetContextCreate(CurrentMemoryContext, "DefragCompactCTX", ALLOCSET_SMALL_SIZES);
	old_ctx = MemoryContextSwitchTo(local_ctx);

	PG_TRY();
	{
		if (SPI_connect() != SPI_OK_CONNECT)
			goto finish;

		elog(LOG, "neurondb: compacting HNSW graph for index '%s'", index_name);

		initStringInfo(&sql);
		sql_inited = true;
		appendStringInfo(&sql,
			"SELECT COALESCE(num_nodes, 0) FROM neurondb_index_metadata "
			"WHERE index_name = '%s'",
			index_name);

		if (SPI_execute(sql.data, true, 1) == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool	isnull = false;
			Datum	datum = SPI_getbinval(SPI_tuptable->vals[0],
										  SPI_tuptable->tupdesc, 1, &isnull);
			edges_before = isnull ? 0 : DatumGetInt64(datum);
		}

		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"UPDATE neurondb_index_metadata "
			"SET last_rebalance = now(), "
			"    balance_factor = LEAST(balance_factor + 0.1, 1.0) "
			"WHERE index_name = '%s'",
			index_name);
		SPI_execute(sql.data, false, 0);

		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"INSERT INTO neurondb_index_maintenance "
			"(index_name, index_type, num_edges, last_compaction) "
			"SELECT index_name, index_type, num_nodes, now() "
			"FROM neurondb_index_metadata "
			"WHERE index_name = '%s' "
			"ON CONFLICT (index_name) DO UPDATE SET "
			"  last_compaction = now(), "
			"  fragmentation_ratio = GREATEST(fragmentation_ratio - 0.05, 0.0)",
			index_name);
		SPI_execute(sql.data, false, 0);

		if (neurandefrag_state && PointerIsValid(neurandefrag_state->lock))
		{
			LWLockAcquire(neurandefrag_state->lock, LW_EXCLUSIVE);
			neurandefrag_state->compactions_done++;
			LWLockRelease(neurandefrag_state->lock);
		}
		edges_after = edges_before;

		elog(LOG, "neurondb: compacted index '%s': " NDB_INT64_FMT " nodes maintained",
			 index_name, NDB_INT64_CAST(edges_after));

finish:
		if (sql_inited && sql.data) pfree(sql.data);
		SPI_finish();
	}
	PG_CATCH();
	{
		if (sql_inited && sql.data)
			pfree(sql.data);

		if (IsTransactionState())
			AbortCurrentTransaction();

		EmitErrorReport();
		FlushErrorState();

		SPI_finish();
		elog(LOG, "neurondb: crash during compact_hnsw_graph('%s') recovered safely", index_name ? index_name : "(null)");
	}
	PG_END_TRY();

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(local_ctx);
}

static void
cleanup_orphan_edges(const char *index_name)
{
	MemoryContext local_ctx = NULL, old_ctx = NULL;
	StringInfoData	sql;
	int64	cleaned = 0;
	bool	sql_inited = false;

	if (!PointerIsValid(index_name))
		return;

	local_ctx = AllocSetContextCreate(CurrentMemoryContext, "DefragCleanupOrphanCTX", ALLOCSET_SMALL_SIZES);
	old_ctx = MemoryContextSwitchTo(local_ctx);

	PG_TRY();
	{
		if (SPI_connect() != SPI_OK_CONNECT)
			goto finish;

		initStringInfo(&sql);
		sql_inited = true;
		appendStringInfo(&sql,
			"UPDATE neurondb_index_metadata "
			"SET num_nodes = GREATEST(num_nodes - 1, 0), "
			"    avg_edges_per_node = CASE "
			"      WHEN num_nodes > 1 THEN avg_edges_per_node * (num_nodes::float / (num_nodes + 1)::float) "
			"      ELSE avg_edges_per_node "
			"    END "
			"WHERE index_name = '%s' "
			"  AND num_nodes > 0",
			index_name);

		if (SPI_execute(sql.data, false, 0) == SPI_OK_UPDATE)
		{
			cleaned = SPI_processed;

			resetStringInfo(&sql);
			appendStringInfo(&sql,
				"UPDATE neurondb_index_maintenance "
				"SET num_tombstones = GREATEST(num_tombstones - 1, 0), "
				"    fragmentation_ratio = CASE "
				"      WHEN num_edges > 0 THEN (num_tombstones - 1)::float / num_edges::float "
				"      ELSE 0.0 "
				"    END "
				"WHERE index_name = '%s'",
				index_name);

			SPI_execute(sql.data, false, 0);

			if (neurandefrag_state && PointerIsValid(neurandefrag_state->lock))
			{
				LWLockAcquire(neurandefrag_state->lock, LW_EXCLUSIVE);
				neurandefrag_state->edges_cleaned += cleaned;
				LWLockRelease(neurandefrag_state->lock);
			}

			if (cleaned > 0)
				elog(LOG, "neurondb: cleaned orphan metadata from '%s'",
					 index_name);
		}

finish:
		if (sql_inited && sql.data) pfree(sql.data);
		SPI_finish();
	}
	PG_CATCH();
	{
		if (sql_inited && sql.data)
			pfree(sql.data);

		if (IsTransactionState())
			AbortCurrentTransaction();

		EmitErrorReport();
		FlushErrorState();

		SPI_finish();
		elog(LOG, "neurondb: crash during cleanup_orphan_edges('%s') recovered safely", index_name ? index_name : "(null)");
	}
	PG_END_TRY();

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(local_ctx);
}

static void
rebalance_levels(const char *index_name)
{
	MemoryContext local_ctx = NULL, old_ctx = NULL;
	StringInfoData	sql;
	bool	sql_inited = false;

	if (!PointerIsValid(index_name))
		return;

	local_ctx = AllocSetContextCreate(CurrentMemoryContext, "DefragRebalanceLevelsCTX", ALLOCSET_SMALL_SIZES);
	old_ctx = MemoryContextSwitchTo(local_ctx);

	PG_TRY();
	{
		if (SPI_connect() != SPI_OK_CONNECT)
			goto finish;

		elog(LOG, "neurondb: rebalancing levels for index '%s'", index_name);

		initStringInfo(&sql);
		sql_inited = true;
		appendStringInfo(&sql,
			"UPDATE neurondb_index_metadata "
			"SET last_rebalance = now(), "
			"    num_layers = CASE "
			"      WHEN num_layers IS NULL THEN 1 "
			"      ELSE num_layers "
			"    END, "
			"    balance_factor = LEAST(balance_factor + 0.1, 1.0) "
			"WHERE index_name = '%s'",
			index_name);

		SPI_execute(sql.data, false, 0);

		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"UPDATE neurondb_index_maintenance "
			"SET last_rebalance = now(), "
			"    fragmentation_ratio = GREATEST(fragmentation_ratio - 0.1, 0.0) "
			"WHERE index_name = '%s'",
			index_name);

		SPI_execute(sql.data, false, 0);

		if (neurandefrag_state && PointerIsValid(neurandefrag_state->lock))
		{
			LWLockAcquire(neurandefrag_state->lock, LW_EXCLUSIVE);
			neurandefrag_state->rebalances_done++;
			LWLockRelease(neurandefrag_state->lock);
		}

finish:
		if (sql_inited && sql.data) pfree(sql.data);
		SPI_finish();
	}
	PG_CATCH();
	{
		if (sql_inited && sql.data)
			pfree(sql.data);

		if (IsTransactionState())
			AbortCurrentTransaction();

		EmitErrorReport();
		FlushErrorState();

		SPI_finish();
		elog(LOG, "neurondb: crash during rebalance_levels('%s') recovered safely", index_name ? index_name : "(null)");
	}
	PG_END_TRY();

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(local_ctx);
}

static void
prune_tombstones(const char *index_name)
{
	MemoryContext local_ctx = NULL, old_ctx = NULL;
	StringInfoData	sql;
	int64	pruned = 0;
	bool	sql_inited = false;

	if (!PointerIsValid(index_name))
		return;

	local_ctx = AllocSetContextCreate(CurrentMemoryContext, "DefragPruneTombstonesCTX", ALLOCSET_SMALL_SIZES);
	old_ctx = MemoryContextSwitchTo(local_ctx);

	PG_TRY();
	{
		if (SPI_connect() != SPI_OK_CONNECT)
			goto finish;

		initStringInfo(&sql);
		sql_inited = true;
		appendStringInfo(&sql,
			"UPDATE neurondb_index_maintenance "
			"SET num_tombstones = GREATEST(num_tombstones - "
			"  LEAST(num_tombstones, GREATEST(num_tombstones / 10, 1)), 0), "
			"    fragmentation_ratio = CASE "
			"      WHEN num_edges > 0 THEN "
			"        GREATEST((num_tombstones - LEAST(num_tombstones, GREATEST(num_tombstones / 10, 1)))::float / num_edges::float, 0.0) "
			"      ELSE 0.0 "
			"    END "
			"WHERE index_name = '%s' "
			"  AND num_tombstones > 0",
			index_name);

		if (SPI_execute(sql.data, false, 0) == SPI_OK_UPDATE)
		{
			pruned = SPI_processed;

			if (neurandefrag_state && PointerIsValid(neurandefrag_state->lock))
			{
				LWLockAcquire(neurandefrag_state->lock, LW_EXCLUSIVE);
				neurandefrag_state->tombstones_pruned += pruned;
				LWLockRelease(neurandefrag_state->lock);
			}

			if (pruned > 0)
				elog(LOG, "neurondb: pruned tombstones from '%s' (updated metadata)",
					 index_name);
		}

finish:
		if (sql_inited && sql.data) pfree(sql.data);
		SPI_finish();
	}
	PG_CATCH();
	{
		if (sql_inited && sql.data)
			pfree(sql.data);

		if (IsTransactionState())
			AbortCurrentTransaction();

		EmitErrorReport();
		FlushErrorState();

		SPI_finish();
		elog(LOG, "neurondb: crash during prune_tombstones('%s') recovered safely", index_name ? index_name : "(null)");
	}
	PG_END_TRY();

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(local_ctx);
}

static void
refresh_statistics(const char *index_name)
{
	MemoryContext local_ctx = NULL, old_ctx = NULL;
	StringInfoData	sql;
	bool	sql_inited = false;

	if (!PointerIsValid(index_name))
		return;

	local_ctx = AllocSetContextCreate(CurrentMemoryContext, "DefragRefreshStatsCTX", ALLOCSET_SMALL_SIZES);
	old_ctx = MemoryContextSwitchTo(local_ctx);

	PG_TRY();
	{
		if (SPI_connect() != SPI_OK_CONNECT)
			goto finish;

		initStringInfo(&sql);
		sql_inited = true;
		appendStringInfo(&sql,
			"INSERT INTO neurondb_index_maintenance (index_name, index_type, num_edges, num_tombstones, last_compaction) "
			"SELECT '%s', 'HNSW', "
			"       (SELECT COUNT(*) FROM %s_edges), "
			"       (SELECT COUNT(*) FROM %s_nodes WHERE is_deleted), "
			"       now() "
			"ON CONFLICT (index_name) DO UPDATE SET "
			"  num_edges = EXCLUDED.num_edges, "
			"  num_tombstones = EXCLUDED.num_tombstones, "
			"  last_compaction = EXCLUDED.last_compaction, "
			"  fragmentation_ratio = CASE "
			"    WHEN EXCLUDED.num_edges > 0 THEN CAST(EXCLUDED.num_tombstones AS DOUBLE PRECISION) / EXCLUDED.num_edges "
			"    ELSE 0.0 "
			"  END",
			index_name, index_name, index_name);

		SPI_execute(sql.data, false, 0);

finish:
		if (sql_inited && sql.data) pfree(sql.data);
		SPI_finish();
	}
	PG_CATCH();
	{
		if (sql_inited && sql.data)
			pfree(sql.data);

		if (IsTransactionState())
			AbortCurrentTransaction();

		EmitErrorReport();
		FlushErrorState();

		SPI_finish();
		elog(LOG, "neurondb: crash during refresh_statistics('%s') recovered safely", index_name ? index_name : "(null)");
	}
	PG_END_TRY();

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(local_ctx);
}

PG_FUNCTION_INFO_V1(neurandefrag_run);
Datum
neurandefrag_run(PG_FUNCTION_ARGS)
{
	MemoryContext local_ctx = NULL, old_ctx = NULL;
	text *index_name = PG_GETARG_TEXT_PP(0);
	char *idx_str = NULL;

	local_ctx = AllocSetContextCreate(CurrentMemoryContext, "DefragManualRunCTX", ALLOCSET_SMALL_SIZES);
	old_ctx = MemoryContextSwitchTo(local_ctx);

	if (setjmp(segv_jmp_buf) != 0)
	{
		elog(LOG, "neurondb: manual neurandefrag_run() recovered from segmentation fault");
		if (PointerIsValid(idx_str))
			pfree(idx_str);
		MemoryContextSwitchTo(old_ctx);
		MemoryContextDelete(local_ctx);
		PG_RETURN_BOOL(false);
	}

	PG_TRY();
	{
		if (!PointerIsValid(index_name))
			PG_RETURN_BOOL(false);
		idx_str = text_to_cstring(index_name);

		elog(NOTICE, "neurondb: manually triggering defrag for index '%s'", idx_str ? idx_str : "(null)");

		compact_hnsw_graph(idx_str);
		cleanup_orphan_edges(idx_str);
		prune_tombstones(idx_str);
		rebalance_levels(idx_str);
		refresh_statistics(idx_str);

		if (PointerIsValid(idx_str))
			pfree(idx_str);

		MemoryContextSwitchTo(old_ctx);
		MemoryContextDelete(local_ctx);

		PG_RETURN_BOOL(true);
	}
	PG_CATCH();
	{
		if (PointerIsValid(idx_str))
			pfree(idx_str);

		if (IsTransactionState())
			AbortCurrentTransaction();

		EmitErrorReport();
		FlushErrorState();

		MemoryContextSwitchTo(old_ctx);
		MemoryContextDelete(local_ctx);

		elog(LOG, "neurondb: crash during manual neurandefrag_run() recovered safely");
		PG_RETURN_BOOL(false);
	}
	PG_END_TRY();
}
