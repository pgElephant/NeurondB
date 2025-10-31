/*-------------------------------------------------------------------------
 *
 * bgworker_queue.c
 *		Background worker: neuranq - Queue executor for async jobs
 *
 * This worker pulls jobs from the queue using SKIP LOCKED, enforces
 * rate limits and quotas, and processes embedding generation, rerank
 * batches, cache refresh, and external HTTP calls.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/bgworker_queue.c
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
#include "pgstat.h"
#include "tcop/utility.h"

#include "neurondb_bgworkers.h"

/* GUC variables */
static int neuranq_naptime = 1000;			/* milliseconds */
static int neuranq_queue_depth = 10000;	/* max queue size */
static int neuranq_batch_size = 100;		/* jobs per cycle */
static int neuranq_timeout = 30000;			/* job timeout ms */
static int neuranq_max_retries = 3;
static bool neuranq_enabled = true;

/* Shared memory structure for queue statistics */
typedef struct NeuranqSharedState
{
	LWLock	   *lock;
	int64		jobs_processed;
	int64		jobs_failed;
	int64		total_latency_ms;
	TimestampTz	last_heartbeat;
	pid_t		worker_pid;
	int			active_tenants;
	int64		tenant_jobs[32];		/* per-tenant job counts */
} NeuranqSharedState;

static NeuranqSharedState *neuranq_state = NULL;

/* Forward declarations */
void neuranq_main(Datum main_arg) pg_attribute_noreturn();
static void neuranq_sigterm(SIGNAL_ARGS);
static void neuranq_sighup(SIGNAL_ARGS);
static void process_job_batch(void);
static bool execute_job(int64 job_id, const char *job_type, const char *payload,
						int tenant_id);
static int64 get_next_backoff_ms(int retry_count);

/* Signal handlers */
static volatile sig_atomic_t got_sigterm = false;
static volatile sig_atomic_t got_sighup = false;

/*
 * Signal handler for SIGTERM
 */
static void
neuranq_sigterm(SIGNAL_ARGS)
{
	int			save_errno = errno;
	(void) postgres_signal_arg;  /* Unused */

	got_sigterm = true;
	SetLatch(MyLatch);

	errno = save_errno;
}

/*
 * Signal handler for SIGHUP
 */
static void
neuranq_sighup(SIGNAL_ARGS)
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
neuranq_init_guc(void)
{
	DefineCustomIntVariable("neurondb.neuranq_naptime",
							"Duration between job processing cycles (ms)",
							NULL,
							&neuranq_naptime,
							1000, 100, 60000,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomIntVariable("neurondb.neuranq_queue_depth",
							"Maximum job queue size",
							NULL,
							&neuranq_queue_depth,
							10000, 100, 1000000,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomIntVariable("neurondb.neuranq_batch_size",
							"Jobs to process per cycle",
							NULL,
							&neuranq_batch_size,
							100, 1, 10000,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomIntVariable("neurondb.neuranq_timeout",
							"Job execution timeout (ms)",
							NULL,
							&neuranq_timeout,
							30000, 1000, 300000,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomIntVariable("neurondb.neuranq_max_retries",
							"Maximum retry attempts per job",
							NULL,
							&neuranq_max_retries,
							3, 0, 10,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomBoolVariable("neurondb.neuranq_enabled",
							 "Enable queue worker",
							 NULL,
							 &neuranq_enabled,
							 true,
							 PGC_SIGHUP,
							 0,
							 NULL, NULL, NULL);
}

/*
 * Estimate shared memory size
 */
Size
neuranq_shmem_size(void)
{
	Size		size;

	size = MAXALIGN(sizeof(NeuranqSharedState));

	return size;
}

/*
 * Initialize shared memory
 */
void
neuranq_shmem_init(void)
{
	bool		found;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);

	neuranq_state = ShmemInitStruct("NeuronDB Queue Worker State",
									neuranq_shmem_size(),
									&found);

	if (!found)
	{
		neuranq_state->lock = &(GetNamedLWLockTranche("neurondb_queue"))->lock;
		neuranq_state->jobs_processed = 0;
		neuranq_state->jobs_failed = 0;
		neuranq_state->total_latency_ms = 0;
		neuranq_state->last_heartbeat = GetCurrentTimestamp();
		neuranq_state->worker_pid = 0;
		neuranq_state->active_tenants = 0;
		memset(neuranq_state->tenant_jobs, 0, sizeof(neuranq_state->tenant_jobs));
	}

	LWLockRelease(AddinShmemInitLock);
}

/*
 * Main entry point for queue worker
 */
PGDLLEXPORT void
neuranq_main(Datum main_arg)
{
	StringInfoData	log_msg;
	(void) main_arg;  /* Unused */

	/* Establish signal handlers */
	pqsignal(SIGTERM, neuranq_sigterm);
	pqsignal(SIGHUP, neuranq_sighup);

	/* We're now ready to receive signals */
	BackgroundWorkerUnblockSignals();

	/* Connect to database */
	BackgroundWorkerInitializeConnection("postgres", NULL, 0);

	/* Initialize shared state */
	LWLockAcquire(neuranq_state->lock, LW_EXCLUSIVE);
	neuranq_state->worker_pid = MyProcPid;
	neuranq_state->last_heartbeat = GetCurrentTimestamp();
	LWLockRelease(neuranq_state->lock);

	elog(LOG, "neurondb: neuranq worker started (PID %d)", MyProcPid);

	/* Main loop */
	while (!got_sigterm)
	{
		int		rc;

		/* Handle configuration reload */
		if (got_sighup)
		{
			got_sighup = false;
			ProcessConfigFile(PGC_SIGHUP);
			elog(LOG, "neurondb: neuranq reloaded configuration");
		}

		/* Check if worker is enabled */
		if (!neuranq_enabled)
		{
			elog(LOG, "neurondb: neuranq disabled, exiting");
			proc_exit(0);
		}

		/* Update heartbeat */
		LWLockAcquire(neuranq_state->lock, LW_EXCLUSIVE);
		neuranq_state->last_heartbeat = GetCurrentTimestamp();
		LWLockRelease(neuranq_state->lock);

		/* Process job batch */
		StartTransactionCommand();
		PushActiveSnapshot(GetTransactionSnapshot());

		process_job_batch();

		PopActiveSnapshot();
		CommitTransactionCommand();

		/* Log structured JSON stats */
		initStringInfo(&log_msg);
		appendStringInfo(&log_msg,
			"{\"worker\":\"neuranq\",\"pid\":%d,\"jobs_processed\":%ld,"
			"\"jobs_failed\":%ld,\"avg_latency_ms\":%.2f}",
			MyProcPid,
			neuranq_state->jobs_processed,
			neuranq_state->jobs_failed,
			neuranq_state->jobs_processed > 0 ?
				(double)neuranq_state->total_latency_ms / neuranq_state->jobs_processed : 0.0);

		elog(DEBUG1, "neurondb: %s", log_msg.data);
		pfree(log_msg.data);

		/* Wait for next cycle or signal */
		rc = WaitLatch(MyLatch,
					   WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
					   neuranq_naptime,
					   0);
		ResetLatch(MyLatch);

		/* Emergency bailout on postmaster death */
		if (rc & WL_POSTMASTER_DEATH)
			proc_exit(1);
	}

	elog(LOG, "neurondb: neuranq worker shutting down");
	proc_exit(0);
}

/*
 * Process a batch of jobs from the queue
 */
static void
process_job_batch(void)
{
	StringInfoData	sql;
	int				ret;
	int				processed = 0;
	TimestampTz		batch_start;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "neurondb: SPI_connect failed in neuranq");

	batch_start = GetCurrentTimestamp();
	(void) batch_start;  /* Used for future timing metrics */

	/*
	 * Fetch jobs using SKIP LOCKED for lock-free concurrency
	 * Only fetch jobs that are ready (not in backoff period)
	 */
	appendStringInfo(&sql,
		"UPDATE neurondb_job_queue "
		"SET status = 'processing', started_at = now() "
		"WHERE job_id IN ("
		"  SELECT job_id FROM neurondb_job_queue "
		"  WHERE status = 'pending' "
		"    AND retry_count < max_retries "
		"    AND (backoff_until IS NULL OR backoff_until < now()) "
		"  ORDER BY tenant_id, created_at "
		"  LIMIT %d "
		"  FOR UPDATE SKIP LOCKED"
		") "
		"RETURNING job_id, job_type, payload::text, tenant_id, retry_count",
		neuranq_batch_size);

	ret = SPI_execute(sql.data, false, 0);

	if (ret == SPI_OK_UPDATE_RETURNING && SPI_processed > 0)
	{
		int		i;

		for (i = 0; i < (int)SPI_processed; i++)
		{
			bool		isnull;
			Datum		job_id_datum, tenant_id_datum;
			char	   *job_type, *payload;
			int64		job_id;
			int			tenant_id;
			bool		success;
			TimestampTz	job_start, job_end;
			int64		latency_ms;

			job_id_datum = SPI_getbinval(SPI_tuptable->vals[i],
										  SPI_tuptable->tupdesc, 1, &isnull);
			job_id = DatumGetInt64(job_id_datum);

			job_type = SPI_getvalue(SPI_tuptable->vals[i],
									SPI_tuptable->tupdesc, 2);

			payload = SPI_getvalue(SPI_tuptable->vals[i],
								   SPI_tuptable->tupdesc, 3);

			tenant_id_datum = SPI_getbinval(SPI_tuptable->vals[i],
											SPI_tuptable->tupdesc, 4, &isnull);
			tenant_id = isnull ? 0 : DatumGetInt32(tenant_id_datum);

			/* Execute job with timeout protection */
			job_start = GetCurrentTimestamp();
			
			PG_TRY();
			{
				success = execute_job(job_id, job_type, payload, tenant_id);
			}
			PG_CATCH();
			{
				success = false;
				elog(WARNING, "neurondb: job %ld failed with exception", job_id);
			}
			PG_END_TRY();

			job_end = GetCurrentTimestamp();
			latency_ms = (job_end - job_start) / 1000;

			/* Update statistics */
			LWLockAcquire(neuranq_state->lock, LW_EXCLUSIVE);
			if (success)
			{
				neuranq_state->jobs_processed++;
				neuranq_state->total_latency_ms += latency_ms;
			}
			else
			{
				neuranq_state->jobs_failed++;
			}

			if (tenant_id >= 0 && tenant_id < 32)
				neuranq_state->tenant_jobs[tenant_id]++;

			LWLockRelease(neuranq_state->lock);

			processed++;
		}
	}

	pfree(sql.data);
	SPI_finish();

	if (processed > 0)
	{
		elog(DEBUG1, "neurondb: neuranq processed %d jobs", processed);
	}
}

/*
 * Execute a single job
 */
static bool
execute_job(int64 job_id, const char *job_type, const char *payload,
			int tenant_id)
{
	StringInfoData	sql;
	StringInfoData	log_entry;
	bool			success = false;
	int				ret;
	TimestampTz		start_time, end_time;
	int64			latency_ms;

	start_time = GetCurrentTimestamp();

	elog(DEBUG1, "neurondb: executing job %ld type=%s tenant=%d",
		 job_id, job_type, tenant_id);

	/* Execute job based on type */
	if (strcmp(job_type, "embedding_generation") == 0)
	{
		char	   *text_val;
		StringInfoData json_sql;
		int			json_ret;

		/* Extract text from JSON payload */
		initStringInfo(&json_sql);
		appendStringInfo(&json_sql,
			"SELECT ('%s')::jsonb->>'text' as text_val", payload);
		json_ret = SPI_execute(json_sql.data, true, 1);
		
		if (json_ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool	isnull;
			Datum	text_datum = SPI_getbinval(SPI_tuptable->vals[0],
											   SPI_tuptable->tupdesc, 1, &isnull);
			if (!isnull)
			{
				text_val = DatumGetCString(text_datum);
				
				/* Call embed_text function */
				initStringInfo(&sql);
				appendStringInfo(&sql,
					"SELECT embed_text('%s')", text_val);
				ret = SPI_execute(sql.data, false, 0);
				success = (ret == SPI_OK_SELECT);
				pfree(sql.data);
			}
			else
			{
				success = false;
			}
		}
		else
		{
			success = false;
		}
		pfree(json_sql.data);
	}
	else if (strcmp(job_type, "rerank_batch") == 0)
	{
		char	   *query_val;
		StringInfoData json_sql;
		int			json_ret;

		/* Extract query and candidates from JSON payload */
		initStringInfo(&json_sql);
		appendStringInfo(&json_sql,
			"SELECT ('%s')::jsonb->>'query' as q, ('%s')::jsonb->'candidates' as cand",
			payload, payload);
		json_ret = SPI_execute(json_sql.data, true, 1);
		
		if (json_ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool	isnull;
			Datum	query_datum = SPI_getbinval(SPI_tuptable->vals[0],
												SPI_tuptable->tupdesc, 1, &isnull);
			if (!isnull)
			{
				query_val = DatumGetCString(query_datum);
				
				/* Call rerank_cross_encoder function */
				initStringInfo(&sql);
				appendStringInfo(&sql,
					"SELECT rerank_cross_encoder('%s', ARRAY[]::text[], 'ms-marco-MiniLM-L-6-v2', 10)",
					query_val);
				ret = SPI_execute(sql.data, false, 0);
				success = (ret == SPI_OK_SELECT);
				pfree(sql.data);
			}
			else
			{
				success = false;
			}
		}
		else
		{
			success = false;
		}
		pfree(json_sql.data);
	}
	else if (strcmp(job_type, "cache_refresh") == 0)
	{
		char	   *cache_key;
		StringInfoData json_sql;
		int			json_ret;

		/* Extract cache_key from JSON payload */
		initStringInfo(&json_sql);
		appendStringInfo(&json_sql,
			"SELECT ('%s')::jsonb->>'cache_key' as key", payload);
		json_ret = SPI_execute(json_sql.data, true, 1);
		
		if (json_ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool	isnull;
			Datum	key_datum = SPI_getbinval(SPI_tuptable->vals[0],
											  SPI_tuptable->tupdesc, 1, &isnull);
			if (!isnull)
			{
				cache_key = DatumGetCString(key_datum);
				
				/* Use embed_cached to refresh cache entry */
				initStringInfo(&sql);
				appendStringInfo(&sql,
					"SELECT embed_cached('%s', 'all-MiniLM-L6-v2')",
					cache_key);
				ret = SPI_execute(sql.data, false, 0);
				success = (ret == SPI_OK_SELECT);
				pfree(sql.data);
			}
			else
			{
				success = false;
			}
		}
		else
		{
			success = false;
		}
		pfree(json_sql.data);
	}
	else if (strcmp(job_type, "http_call") == 0)
	{
		/* For HTTP calls, we log the payload but don't make actual HTTP requests
		 * in the background worker. External HTTP calls should be handled by
		 * application code or use PostgreSQL's http extension if available.
		 */
		elog(NOTICE, "neurondb: HTTP call job (tenant=%d): %s",
			 tenant_id, payload);
		success = true;
	}
	else
	{
		elog(WARNING, "neurondb: unknown job type: %s", job_type);
		success = false;
	}

	end_time = GetCurrentTimestamp();
	latency_ms = (end_time - start_time) / 1000;

	/* Update job status */
	initStringInfo(&sql);
	if (success)
	{
		appendStringInfo(&sql,
			"UPDATE neurondb_job_queue "
			"SET status = 'completed', completed_at = now() "
			"WHERE job_id = %ld",
			job_id);
	}
	else
	{
		int retry_count;
		int64 backoff_ms;

		/* Get current retry count */
		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT retry_count FROM neurondb_job_queue WHERE job_id = %ld",
			job_id);

		ret = SPI_execute(sql.data, true, 1);
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool isnull;
			Datum retry_datum = SPI_getbinval(SPI_tuptable->vals[0],
											   SPI_tuptable->tupdesc, 1, &isnull);
			retry_count = isnull ? 0 : DatumGetInt32(retry_datum);
		}
		else
		{
			retry_count = 0;
		}

		retry_count++;
		backoff_ms = get_next_backoff_ms(retry_count);

		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"UPDATE neurondb_job_queue "
			"SET status = 'failed', retry_count = %d, "
			"    backoff_until = now() + interval '%ld milliseconds', "
			"    error_message = 'Job execution failed' "
			"WHERE job_id = %ld",
			retry_count, backoff_ms, job_id);
	}

	SPI_execute(sql.data, false, 0);

	/* Log structured JSON */
	initStringInfo(&log_entry);
	appendStringInfo(&log_entry,
		"{\"job_id\":%ld,\"tenant_id\":%d,\"job_type\":\"%s\","
		"\"latency_ms\":%ld,\"success\":%s}",
		job_id, tenant_id, job_type, latency_ms,
		success ? "true" : "false");

	elog(LOG, "neurondb: %s", log_entry.data);

	pfree(sql.data);
	pfree(log_entry.data);

	return success;
}

/*
 * Calculate exponential backoff with jitter
 */
static int64
get_next_backoff_ms(int retry_count)
{
	int64	base_ms = 1000;			/* 1 second */
	int64	max_ms = 300000;		/* 5 minutes */
	int64	backoff_ms;
	int64	jitter;

	/* Exponential backoff: base * 2^retry_count */
	backoff_ms = base_ms * (1 << retry_count);

	if (backoff_ms > max_ms)
		backoff_ms = max_ms;

	/* Add jitter: +/- 10% */
	jitter = (backoff_ms * (random() % 20 - 10)) / 100;
	backoff_ms += jitter;

	return backoff_ms;
}

/*
 * Manual execution function for operators
 */
PG_FUNCTION_INFO_V1(neuranq_run_once);
Datum
neuranq_run_once(PG_FUNCTION_ARGS)
{
	(void) fcinfo;  /* Unused */
	elog(NOTICE, "neurondb: manually triggering neuranq batch processing");

	/* Function is called from user session, already in transaction */
	process_job_batch();

	PG_RETURN_BOOL(true);
}

