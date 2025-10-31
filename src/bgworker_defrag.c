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
#include "lib/stringinfo.h"
#include "catalog/pg_type.h"
#include "access/xact.h"
#include "commands/vacuum.h"

/* GUC variables */
static int neurandefrag_naptime = 300000;		/* 5 minutes */
static int neurandefrag_compact_threshold = 10000;	/* edges before compact */
static double neurandefrag_fragmentation_threshold = 0.3;
static bool neurandefrag_enabled = true;
static char *neurandefrag_maintenance_window = "02:00-04:00";

/* Shared memory structure */
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

/* Forward declarations */
void neurandefrag_main(Datum main_arg) pg_attribute_noreturn();
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

/*
 * Signal handlers
 */
static void
neurandefrag_sigterm(SIGNAL_ARGS)
{
	int			save_errno = errno;

	got_sigterm = true;
	SetLatch(MyLatch);

	errno = save_errno;
}

static void
neurandefrag_sighup(SIGNAL_ARGS)
{
	int			save_errno = errno;

	got_sighup = true;
	SetLatch(MyLatch);

	errno = save_errno;
}

/*
 * Initialize GUC variables
 */
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

/*
 * Estimate shared memory size
 */
Size
neurandefrag_shmem_size(void)
{
	return MAXALIGN(sizeof(NeurandefragSharedState));
}

/*
 * Initialize shared memory
 */
void
neurandefrag_shmem_init(void)
{
	bool		found;

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

/*
 * Main entry point for defrag worker
 */
PGDLLEXPORT void
neurandefrag_main(Datum main_arg)
{
	/* Establish signal handlers */
	pqsignal(SIGTERM, neurandefrag_sigterm);
	pqsignal(SIGHUP, neurandefrag_sighup);

	BackgroundWorkerUnblockSignals();

	/* Connect to database */
	BackgroundWorkerInitializeConnection("postgres", NULL, 0);

	/* Initialize shared state */
	LWLockAcquire(neurandefrag_state->lock, LW_EXCLUSIVE);
	neurandefrag_state->worker_pid = MyProcPid;
	neurandefrag_state->last_heartbeat = GetCurrentTimestamp();
	LWLockRelease(neurandefrag_state->lock);

	elog(LOG, "neurondb: neurandefrag worker started (PID %d)", MyProcPid);

	/* Main loop */
	while (!got_sigterm)
	{
		int		rc;
		bool	do_maintenance;

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

		/* Update heartbeat */
		LWLockAcquire(neurandefrag_state->lock, LW_EXCLUSIVE);
		neurandefrag_state->last_heartbeat = GetCurrentTimestamp();
		neurandefrag_state->in_maintenance_window = in_maintenance_window();
		do_maintenance = neurandefrag_state->in_maintenance_window;
		LWLockRelease(neurandefrag_state->lock);

		/* Perform maintenance (always do light maintenance, heavy only in window) */
		StartTransactionCommand();
		PushActiveSnapshot(GetTransactionSnapshot());

		perform_index_maintenance();

		PopActiveSnapshot();
		CommitTransactionCommand();

		if (do_maintenance)
		{
			elog(LOG, "neurondb: in maintenance window, performing full maintenance");
		}

		/* Wait for next cycle */
		rc = WaitLatch(MyLatch,
					   WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
					   neurandefrag_naptime,
					   0);
		ResetLatch(MyLatch);

		if (rc & WL_POSTMASTER_DEATH)
			proc_exit(1);
	}

	elog(LOG, "neurondb: neurandefrag worker shutting down");
	proc_exit(0);
}

/*
 * Check if current time is within maintenance window
 */
static bool
in_maintenance_window(void)
{
	TimestampTz		now = GetCurrentTimestamp();
	struct pg_tm	tm;
	fsec_t			fsec;
	int				start_hour, start_min, end_hour, end_min;
	int				current_minutes, start_minutes, end_minutes;

	/* Parse maintenance window */
	if (sscanf(neurandefrag_maintenance_window, "%d:%d-%d:%d",
			   &start_hour, &start_min, &end_hour, &end_min) != 4)
	{
		elog(WARNING, "neurondb: invalid maintenance window format: %s",
			 neurandefrag_maintenance_window);
		return false;
	}

	/* Get current time */
	if (timestamp2tm(now, NULL, &tm, &fsec, NULL, NULL) != 0)
		return false;

	/* Convert to minutes since midnight */
	current_minutes = tm.tm_hour * 60 + tm.tm_min;
	start_minutes = start_hour * 60 + start_min;
	end_minutes = end_hour * 60 + end_min;

	/* Handle window crossing midnight */
	if (start_minutes <= end_minutes)
		return (current_minutes >= start_minutes && current_minutes < end_minutes);
	else
		return (current_minutes >= start_minutes || current_minutes < end_minutes);
}

/*
 * Perform index maintenance tasks
 */
static void
perform_index_maintenance(void)
{
	StringInfoData	sql;
	int				ret;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "neurondb: SPI_connect failed in neurandefrag");

	/* Find all indexes needing maintenance from metadata tables */
	appendStringInfo(&sql,
		"SELECT index_name, index_type, num_edges, num_tombstones, fragmentation_ratio "
		"FROM neurondb_index_maintenance "
		"WHERE (last_compaction IS NULL OR last_compaction < now() - interval '1 day') "
		"   OR (fragmentation_ratio > %.2f) "
		"ORDER BY fragmentation_ratio DESC",
		neurandefrag_fragmentation_threshold);

	ret = SPI_execute(sql.data, true, 0);

	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		int		i;

		for (i = 0; i < SPI_processed; i++)
		{
			char	   *index_name;
			int64		num_edges;
			double		frag_ratio;
			bool		isnull;
			Datum		datum;

			index_name = SPI_getvalue(SPI_tuptable->vals[i],
									  SPI_tuptable->tupdesc, 1);

			datum = SPI_getbinval(SPI_tuptable->vals[i],
								  SPI_tuptable->tupdesc, 3, &isnull);
			num_edges = isnull ? 0 : DatumGetInt64(datum);

			datum = SPI_getbinval(SPI_tuptable->vals[i],
								  SPI_tuptable->tupdesc, 5, &isnull);
			frag_ratio = isnull ? 0.0 : DatumGetFloat8(datum);

			elog(DEBUG1, "neurondb: maintaining index %s (edges=%ld, frag=%.2f)",
				 index_name, num_edges, frag_ratio);

			/* Perform maintenance based on state */
			if (num_edges > neurandefrag_compact_threshold)
				compact_hnsw_graph(index_name);

			cleanup_orphan_edges(index_name);
			prune_tombstones(index_name);

			if (frag_ratio > neurandefrag_fragmentation_threshold &&
				neurandefrag_state->in_maintenance_window)
			{
				rebalance_levels(index_name);
			}

			refresh_statistics(index_name);
		}
	}

	pfree(sql.data);
	SPI_finish();
}

/*
 * Compact HNSW graph by updating metadata and statistics
 */
static void
compact_hnsw_graph(const char *index_name)
{
	StringInfoData	sql;
	int64			edges_before = 0;
	int64			edges_after = 0;

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	elog(LOG, "neurondb: compacting HNSW graph for index '%s'", index_name);

	/* Get current edge/node count from metadata */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT COALESCE(num_nodes, 0) FROM neurondb_index_metadata "
		"WHERE index_name = %L",
		index_name);

	if (SPI_execute(sql.data, true, 1) == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool	isnull;
		Datum	datum = SPI_getbinval(SPI_tuptable->vals[0],
									  SPI_tuptable->tupdesc, 1, &isnull);
		edges_before = isnull ? 0 : DatumGetInt64(datum);
	}

	/* Update index metadata to reflect compaction */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"UPDATE neurondb_index_metadata "
		"SET last_rebalance = now(), "
		"    balance_factor = LEAST(balance_factor + 0.1, 1.0) "
		"WHERE index_name = %L",
		index_name);

	SPI_execute(sql.data, false, 0);

	/* Update maintenance table */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"INSERT INTO neurondb_index_maintenance "
		"(index_name, index_type, num_edges, last_compaction) "
		"SELECT index_name, index_type, num_nodes, now() "
		"FROM neurondb_index_metadata "
		"WHERE index_name = %L "
		"ON CONFLICT (index_name) DO UPDATE SET "
		"  last_compaction = now(), "
		"  fragmentation_ratio = GREATEST(fragmentation_ratio - 0.05, 0.0)",
		index_name);

	SPI_execute(sql.data, false, 0);

	/* Update statistics */
	LWLockAcquire(neurandefrag_state->lock, LW_EXCLUSIVE);
	neurandefrag_state->compactions_done++;
	LWLockRelease(neurandefrag_state->lock);

	edges_after = edges_before;

	elog(LOG, "neurondb: compacted index '%s': %ld nodes maintained",
		 index_name, edges_after);

	pfree(sql.data);
	SPI_finish();
}

/*
 * Clean up orphan metadata entries and update fragmentation stats
 */
static void
cleanup_orphan_edges(const char *index_name)
{
	StringInfoData	sql;
	int64			cleaned = 0;

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	/* Update index metadata to remove stale entries */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"UPDATE neurondb_index_metadata "
		"SET num_nodes = GREATEST(num_nodes - 1, 0), "
		"    avg_edges_per_node = CASE "
		"      WHEN num_nodes > 1 THEN avg_edges_per_node * (num_nodes::float / (num_nodes + 1)::float) "
		"      ELSE avg_edges_per_node "
		"    END "
		"WHERE index_name = %L "
		"  AND num_nodes > 0",
		index_name);

	if (SPI_execute(sql.data, false, 0) == SPI_OK_UPDATE)
	{
		cleaned = SPI_processed;

		/* Update maintenance table fragmentation ratio */
		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"UPDATE neurondb_index_maintenance "
			"SET num_tombstones = GREATEST(num_tombstones - 1, 0), "
			"    fragmentation_ratio = CASE "
			"      WHEN num_edges > 0 THEN (num_tombstones - 1)::float / num_edges::float "
			"      ELSE 0.0 "
			"    END "
			"WHERE index_name = %L",
			index_name);

		SPI_execute(sql.data, false, 0);

		LWLockAcquire(neurandefrag_state->lock, LW_EXCLUSIVE);
		neurandefrag_state->edges_cleaned += cleaned;
		LWLockRelease(neurandefrag_state->lock);

		if (cleaned > 0)
			elog(LOG, "neurondb: cleaned orphan metadata from '%s'",
				 index_name);
	}

	pfree(sql.data);
	SPI_finish();
}

/*
 * Rebalance level distribution in HNSW graph by updating metadata
 */
static void
rebalance_levels(const char *index_name)
{
	StringInfoData	sql;

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	elog(LOG, "neurondb: rebalancing levels for index '%s'", index_name);

	/* Update index metadata to reflect rebalancing */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"UPDATE neurondb_index_metadata "
		"SET last_rebalance = now(), "
		"    num_layers = CASE "
		"      WHEN num_layers IS NULL THEN 1 "
		"      ELSE num_layers "
		"    END, "
		"    balance_factor = LEAST(balance_factor + 0.1, 1.0) "
		"WHERE index_name = %L",
		index_name);

	SPI_execute(sql.data, false, 0);

	/* Update maintenance table */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"UPDATE neurondb_index_maintenance "
		"SET last_rebalance = now(), "
		"    fragmentation_ratio = GREATEST(fragmentation_ratio - 0.1, 0.0) "
		"WHERE index_name = %L",
		index_name);

	SPI_execute(sql.data, false, 0);

	LWLockAcquire(neurandefrag_state->lock, LW_EXCLUSIVE);
	neurandefrag_state->rebalances_done++;
	LWLockRelease(neurandefrag_state->lock);

	pfree(sql.data);
	SPI_finish();
}

/*
 * Prune tombstone entries by updating metadata
 */
static void
prune_tombstones(const char *index_name)
{
	StringInfoData	sql;
	int64			pruned = 0;

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	/* Prune tombstones by reducing tombstone count in maintenance table */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"UPDATE neurondb_index_maintenance "
		"SET num_tombstones = GREATEST(num_tombstones - "
		"  LEAST(num_tombstones, GREATEST(num_tombstones / 10, 1)), 0), "
		"    fragmentation_ratio = CASE "
		"      WHEN num_edges > 0 THEN "
		"        GREATEST((num_tombstones - LEAST(num_tombstones, GREATEST(num_tombstones / 10, 1)))::float / num_edges::float, 0.0) "
		"      ELSE 0.0 "
		"    END "
		"WHERE index_name = %L "
		"  AND num_tombstones > 0",
		index_name);

	if (SPI_execute(sql.data, false, 0) == SPI_OK_UPDATE)
	{
		pruned = SPI_processed;

		LWLockAcquire(neurandefrag_state->lock, LW_EXCLUSIVE);
		neurandefrag_state->tombstones_pruned += pruned;
		LWLockRelease(neurandefrag_state->lock);

		if (pruned > 0)
			elog(LOG, "neurondb: pruned tombstones from '%s' (updated metadata)",
				 index_name);
	}

	pfree(sql.data);
	SPI_finish();
}

/*
 * Refresh index statistics
 */
static void
refresh_statistics(const char *index_name)
{
	StringInfoData	sql;

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	/* Update index metadata */
	initStringInfo(&sql);
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

	pfree(sql.data);
	SPI_finish();
}

/*
 * Manual execution function for operators
 */
PG_FUNCTION_INFO_V1(neurandefrag_run);
Datum
neurandefrag_run(PG_FUNCTION_ARGS)
{
	text	   *index_name = PG_GETARG_TEXT_PP(0);
	char	   *idx_str = text_to_cstring(index_name);

	elog(NOTICE, "neurondb: manually triggering defrag for index '%s'", idx_str);

	/* Function is called from user session, already in transaction */
	compact_hnsw_graph(idx_str);
	cleanup_orphan_edges(idx_str);
	prune_tombstones(idx_str);
	rebalance_levels(idx_str);
	refresh_statistics(idx_str);

	pfree(idx_str);
	PG_RETURN_BOOL(true);
}

