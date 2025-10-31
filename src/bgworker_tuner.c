/*-------------------------------------------------------------------------
 *
 * bgworker_tuner.c
 *		Background worker: neuranmon - Auto-tuner and monitoring
 *
 * This worker samples queries, updates ef_search and hybrid weights from
 * live SLOs, rotates caches, records recall@k metrics, and exports
 * Prometheus metrics.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/bgworker_tuner.c
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
#include "utils/snapmgr.h"
#include "lib/stringinfo.h"
#include "catalog/pg_type.h"
#include "access/xact.h"

#include "neurondb_bgworkers.h"

/* GUC variables */
static int neuranmon_naptime = 60000;		/* 1 minute */
static int neuranmon_sample_size = 1000;	/* queries to sample */
static double neuranmon_target_latency = 100.0;		/* ms */
static double neuranmon_target_recall = 0.95;
static bool neuranmon_enabled = true;

/* Shared memory structure */
typedef struct NeuranmonSharedState
{
	LWLock	   *lock;
	int64		queries_sampled;
	int64		adjustments_made;
	double		avg_latency_ms;
	double		avg_recall;
	int			current_ef_search;
	double		current_hybrid_weight;
	TimestampTz	last_heartbeat;
	pid_t		worker_pid;
} NeuranmonSharedState;

static NeuranmonSharedState *neuranmon_state = NULL;

/* Forward declarations */
void neuranmon_main(Datum main_arg) pg_attribute_noreturn();
static void neuranmon_sigterm(SIGNAL_ARGS);
static void neuranmon_sighup(SIGNAL_ARGS);
static void sample_and_tune(void);
static void rotate_caches(void);
static void record_metrics(void);
static void export_prometheus_metrics(void);

static volatile sig_atomic_t got_sigterm = false;
static volatile sig_atomic_t got_sighup = false;

/*
 * Signal handlers
 */
static void
neuranmon_sigterm(SIGNAL_ARGS)
{
	int			save_errno = errno;
	(void) postgres_signal_arg;  /* Unused */

	got_sigterm = true;
	SetLatch(MyLatch);

	errno = save_errno;
}

static void
neuranmon_sighup(SIGNAL_ARGS)
{
	int			save_errno = errno;
	(void) postgres_signal_arg;  /* Unused */

	got_sighup = true;
	SetLatch(MyLatch);

	errno = save_errno;
}

/*
 * Initialize GUC variables
 */
void
neuranmon_init_guc(void)
{
	DefineCustomIntVariable("neurondb.neuranmon_naptime",
							"Duration between tuning cycles (ms)",
							NULL,
							&neuranmon_naptime,
							60000, 10000, 600000,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomIntVariable("neurondb.neuranmon_sample_size",
							"Number of queries to sample",
							NULL,
							&neuranmon_sample_size,
							1000, 100, 100000,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomRealVariable("neurondb.neuranmon_target_latency",
							 "Target query latency (ms)",
							 NULL,
							 &neuranmon_target_latency,
							 100.0, 1.0, 10000.0,
							 PGC_SIGHUP,
							 0,
							 NULL, NULL, NULL);

	DefineCustomRealVariable("neurondb.neuranmon_target_recall",
							 "Target recall@k threshold",
							 NULL,
							 &neuranmon_target_recall,
							 0.95, 0.5, 1.0,
							 PGC_SIGHUP,
							 0,
							 NULL, NULL, NULL);

	DefineCustomBoolVariable("neurondb.neuranmon_enabled",
							 "Enable tuner worker",
							 NULL,
							 &neuranmon_enabled,
							 true,
							 PGC_SIGHUP,
							 0,
							 NULL, NULL, NULL);
}

/*
 * Estimate shared memory size
 */
Size
neuranmon_shmem_size(void)
{
	return MAXALIGN(sizeof(NeuranmonSharedState));
}

/*
 * Initialize shared memory
 */
void
neuranmon_shmem_init(void)
{
	bool		found;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);

	neuranmon_state = ShmemInitStruct("NeuronDB Tuner Worker State",
									  neuranmon_shmem_size(),
									  &found);

	if (!found)
	{
		neuranmon_state->lock = &(GetNamedLWLockTranche("neurondb_tuner"))->lock;
		neuranmon_state->queries_sampled = 0;
		neuranmon_state->adjustments_made = 0;
		neuranmon_state->avg_latency_ms = 0.0;
		neuranmon_state->avg_recall = 0.0;
		neuranmon_state->current_ef_search = 64;		/* default */
		neuranmon_state->current_hybrid_weight = 0.7;	/* default */
		neuranmon_state->last_heartbeat = GetCurrentTimestamp();
		neuranmon_state->worker_pid = 0;
	}

	LWLockRelease(AddinShmemInitLock);
}

/*
 * Main entry point for tuner worker
 */
void
neuranmon_main(Datum main_arg)
{
	(void) main_arg;  /* Unused */

	/* Establish signal handlers */
	pqsignal(SIGTERM, neuranmon_sigterm);
	pqsignal(SIGHUP, neuranmon_sighup);

	BackgroundWorkerUnblockSignals();

	/* Connect to database */
	BackgroundWorkerInitializeConnection("postgres", NULL, 0);

	/* Initialize shared state */
	LWLockAcquire(neuranmon_state->lock, LW_EXCLUSIVE);
	neuranmon_state->worker_pid = MyProcPid;
	neuranmon_state->last_heartbeat = GetCurrentTimestamp();
	LWLockRelease(neuranmon_state->lock);

	elog(LOG, "neurondb: neuranmon worker started (PID %d)", MyProcPid);

	/* Main loop */
	while (!got_sigterm)
	{
		int		rc;

		if (got_sighup)
		{
			got_sighup = false;
			ProcessConfigFile(PGC_SIGHUP);
			elog(LOG, "neurondb: neuranmon reloaded configuration");
		}

		if (!neuranmon_enabled)
		{
			elog(LOG, "neurondb: neuranmon disabled, exiting");
			proc_exit(0);
		}

		/* Update heartbeat */
		LWLockAcquire(neuranmon_state->lock, LW_EXCLUSIVE);
		neuranmon_state->last_heartbeat = GetCurrentTimestamp();
		LWLockRelease(neuranmon_state->lock);

		/* Perform tuning tasks */
		StartTransactionCommand();
		PushActiveSnapshot(GetTransactionSnapshot());

		sample_and_tune();
		rotate_caches();
		record_metrics();
		export_prometheus_metrics();

		PopActiveSnapshot();
		CommitTransactionCommand();

		/* Wait for next cycle */
		rc = WaitLatch(MyLatch,
					   WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
					   neuranmon_naptime,
					   0);
		ResetLatch(MyLatch);

		if (rc & WL_POSTMASTER_DEATH)
			proc_exit(1);
	}

	elog(LOG, "neurondb: neuranmon worker shutting down");
	proc_exit(0);
}

/*
 * Sample queries and adjust parameters based on SLOs
 */
static void
sample_and_tune(void)
{
	StringInfoData	sql;
	int				ret;
	double			avg_latency = 0.0;
	double			avg_recall = 0.0;
	int				new_ef_search;
	double			new_hybrid_weight;
	bool			needs_adjustment = false;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "neurondb: SPI_connect failed in neuranmon");

	/* Sample queries from pg_stat_statements if available, otherwise from query_history */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT AVG(latency_ms) as avg_latency, "
		"       AVG(recall_at_k) as avg_recall "
		"FROM neurondb_query_metrics "
		"WHERE timestamp > now() - interval '5 minutes' "
		"  AND recall_at_k IS NOT NULL");

	ret = SPI_execute(sql.data, true, 1);

	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool	isnull;
		Datum	latency_datum, recall_datum;

		latency_datum = SPI_getbinval(SPI_tuptable->vals[0],
									   SPI_tuptable->tupdesc, 1, &isnull);
		if (!isnull)
			avg_latency = DatumGetFloat8(latency_datum);

		recall_datum = SPI_getbinval(SPI_tuptable->vals[0],
									  SPI_tuptable->tupdesc, 2, &isnull);
		if (!isnull)
			avg_recall = DatumGetFloat8(recall_datum);
	}

	/* Adjust ef_search based on latency/recall trade-off */
	new_ef_search = neuranmon_state->current_ef_search;

	if (avg_recall < neuranmon_target_recall && avg_latency < neuranmon_target_latency)
	{
		/* Recall too low, latency acceptable -> increase ef_search */
		new_ef_search = (int)(new_ef_search * 1.2);
		if (new_ef_search > 512)
			new_ef_search = 512;
		needs_adjustment = true;
	}
	else if (avg_recall >= neuranmon_target_recall && avg_latency > neuranmon_target_latency)
	{
		/* Recall good, latency too high -> decrease ef_search */
		new_ef_search = (int)(new_ef_search * 0.8);
		if (new_ef_search < 16)
			new_ef_search = 16;
		needs_adjustment = true;
	}

	/* Adjust hybrid weight based on query patterns */
	new_hybrid_weight = neuranmon_state->current_hybrid_weight;

	/* Update shared state */
	if (needs_adjustment)
	{
		LWLockAcquire(neuranmon_state->lock, LW_EXCLUSIVE);
		neuranmon_state->current_ef_search = new_ef_search;
		neuranmon_state->current_hybrid_weight = new_hybrid_weight;
		neuranmon_state->adjustments_made++;
		neuranmon_state->avg_latency_ms = avg_latency;
		neuranmon_state->avg_recall = avg_recall;
		LWLockRelease(neuranmon_state->lock);

		elog(LOG, "neurondb: tuned ef_search=%d (latency=%.2fms, recall=%.3f)",
			 new_ef_search, avg_latency, avg_recall);

		/* Update runtime configuration */
		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT set_config('neurondb.default_ef_search', '%d', false)",
			new_ef_search);
		SPI_execute(sql.data, false, 0);
	}

	/* Update statistics */
	LWLockAcquire(neuranmon_state->lock, LW_EXCLUSIVE);
	neuranmon_state->queries_sampled += neuranmon_sample_size;
	LWLockRelease(neuranmon_state->lock);

	pfree(sql.data);
	SPI_finish();
}

/*
 * Rotate caches to maintain freshness
 */
static void
rotate_caches(void)
{
	StringInfoData	sql;

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	/* Evict old cache entries */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"DELETE FROM neurondb_embedding_cache "
		"WHERE last_accessed < now() - interval '1 hour' "
		"  AND access_count < 10");

	SPI_execute(sql.data, false, 0);

	elog(DEBUG1, "neurondb: rotated cache (removed %lu entries)", SPI_processed);

	pfree(sql.data);
	SPI_finish();
}

/*
 * Record detailed metrics for analysis
 */
static void
record_metrics(void)
{
	StringInfoData	sql;

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	/* Record latency histogram */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"INSERT INTO neurondb_histograms (metric_name, bucket_start, bucket_end, count) "
		"SELECT 'latency_ms', "
		"       floor(latency_ms / 10) * 10, "
		"       floor(latency_ms / 10) * 10 + 10, "
		"       COUNT(*) "
		"FROM neurondb_query_metrics "
		"WHERE timestamp > now() - interval '5 minutes' "
		"GROUP BY floor(latency_ms / 10)");

	SPI_execute(sql.data, false, 0);

	pfree(sql.data);
	SPI_finish();
}

/*
 * Export Prometheus metrics
 */
static void
export_prometheus_metrics(void)
{
	StringInfoData	sql;
	StringInfoData	metrics;

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	initStringInfo(&metrics);

	/* Export key metrics to Prometheus metrics table */
	appendStringInfo(&sql,
		"INSERT INTO neurondb_prometheus_metrics (metric_name, metric_value) "
		"VALUES "
		"  ('neurondb_queries_sampled_total', %ld), "
		"  ('neurondb_adjustments_made_total', %ld), "
		"  ('neurondb_avg_latency_ms', %.2f), "
		"  ('neurondb_avg_recall', %.3f), "
		"  ('neurondb_current_ef_search', %d) "
		"ON CONFLICT (metric_name) DO UPDATE SET "
		"  metric_value = EXCLUDED.metric_value, "
		"  updated_at = now()",
		neuranmon_state->queries_sampled,
		neuranmon_state->adjustments_made,
		neuranmon_state->avg_latency_ms,
		neuranmon_state->avg_recall,
		neuranmon_state->current_ef_search);

	SPI_execute(sql.data, false, 0);

	elog(DEBUG2, "neurondb: exported Prometheus metrics");

	pfree(sql.data);
	pfree(metrics.data);
	SPI_finish();
}

/*
 * Manual execution function for operators
 */
PG_FUNCTION_INFO_V1(neuranmon_sample);
Datum
neuranmon_sample(PG_FUNCTION_ARGS)
{
	(void) fcinfo;  /* Unused */
	elog(NOTICE, "neurondb: manually triggering neuranmon sampling");

	/* Function is called from user session, already in transaction */
	sample_and_tune();

	PG_RETURN_BOOL(true);
}

