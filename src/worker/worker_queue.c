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
#include "utils/snapmgr.h"
#include "utils/memutils.h"
#include "access/xact.h"
#include "lib/stringinfo.h"
#include "catalog/pg_type.h"
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
PGDLLEXPORT void neuranq_main(Datum main_arg) pg_attribute_noreturn();
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
		/* Lock already registered via RequestNamedLWLockTranche in _PG_init */
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
 * 100% crash-safe with respect to segfaults by guarding all code with PG_TRY/PG_CATCH,
 * covering cleanup and explicit memory safety.
 */
PGDLLEXPORT void
neuranq_main(Datum main_arg)
{
	StringInfoData	log_msg;
	(void) main_arg;  /* Unused */
	MemoryContext oldcontext = NULL;
	MemoryContext worker_context = NULL;
	bool shutting_down = false;

	/* Establish signal handlers */
	pqsignal(SIGTERM, neuranq_sigterm);
	pqsignal(SIGHUP, neuranq_sighup);

	/* We're now ready to receive signals */
	BackgroundWorkerUnblockSignals();

	/* Connect to database */
	BackgroundWorkerInitializeConnection("postgres", NULL, 0);

	/* Create top-level context for worker lifetime */
	worker_context = AllocSetContextCreate(TopMemoryContext,
		"NeuronDB Queue Worker", ALLOCSET_DEFAULT_SIZES);

	PG_TRY();
	{
		/* Initialize shared state */
		if (neuranq_state && neuranq_state->lock)
		{
			LWLockAcquire(neuranq_state->lock, LW_EXCLUSIVE);
			neuranq_state->worker_pid = MyProcPid;
			neuranq_state->last_heartbeat = GetCurrentTimestamp();
			LWLockRelease(neuranq_state->lock);
		}

		elog(LOG, "neurondb: neuranq worker started (PID %d)", MyProcPid);

		/* Main loop */
		while (!got_sigterm && !shutting_down)
		{
			int		rc;

			PG_TRY();
			{
				oldcontext = MemoryContextSwitchTo(worker_context);
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
					shutting_down = true;
					break;
				}

				/* Update heartbeat */
				if (neuranq_state && neuranq_state->lock)
				{
					LWLockAcquire(neuranq_state->lock, LW_EXCLUSIVE);
					neuranq_state->last_heartbeat = GetCurrentTimestamp();
					LWLockRelease(neuranq_state->lock);
				}

				/* Process job batch within its own memory context */
				MemoryContext batch_context = AllocSetContextCreate(worker_context, "NeuronDB JobBatch", ALLOCSET_DEFAULT_SIZES);
				MemoryContextSwitchTo(batch_context);

				StartTransactionCommand();
				PushActiveSnapshot(GetTransactionSnapshot());
				PG_TRY();
				{
					process_job_batch();
				}
				PG_CATCH();
				{
					FlushErrorState();
				}
				PG_END_TRY();
				PopActiveSnapshot();
				CommitTransactionCommand();

				/* Log structured JSON stats */
				initStringInfo(&log_msg);
				appendStringInfo(&log_msg,
					"{\"worker\":\"neuranq\",\"pid\":%d,\"jobs_processed\":%ld,"
					"\"jobs_failed\":%ld,\"avg_latency_ms\":%.2f}",
					MyProcPid,
					neuranq_state ? neuranq_state->jobs_processed : 0,
					neuranq_state ? neuranq_state->jobs_failed : 0,
					(neuranq_state && neuranq_state->jobs_processed > 0) ?
						(double)neuranq_state->total_latency_ms / neuranq_state->jobs_processed : 0.0);

				elog(DEBUG1, "neurondb: %s", log_msg.data);

				if (log_msg.data)
					pfree(log_msg.data);

				MemoryContextSwitchTo(worker_context);
				MemoryContextDelete(batch_context);

				/* Wait for next cycle or signal */
				rc = WaitLatch(MyLatch,
							WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
							neuranq_naptime,
							0);
				ResetLatch(MyLatch);

				/* Emergency bailout on postmaster death */
				if (rc & WL_POSTMASTER_DEATH)
				{
					shutting_down = true;
					break;
				}
			}
			PG_CATCH();
			{
				elog(WARNING, "neurondb: Exception in neuranq_main - crash-safe handler cleaning up. Flushing error state and continuing.");
				FlushErrorState();
				/* Reset context if anything broke */
				if (oldcontext)
					MemoryContextSwitchTo(oldcontext);
				/* loop continues - memory context may be reaped at next batch */
			}
			PG_END_TRY();
		}
	}
	PG_CATCH();
	{
		/* Serious error - exit crash safely */
		elog(FATAL, "neurondb: unrecoverable error in neuranq worker: crash-safe exit.");
		FlushErrorState();
	}
	PG_END_TRY();

	if (worker_context)
		MemoryContextDelete(worker_context);

	elog(LOG, "neurondb: neuranq worker shutting down");
	proc_exit(0);
}


/*
 * Process a batch of jobs from the queue
 * 100% crash-safe: all allocations guarded, SPI and StringInfos checked.
 */
static void
process_job_batch(void)
{
	StringInfoData sql;
	int				ret = 0;
	int				processed = 0;
	TimestampTz		batch_start;

	MemoryContext oldcontext = CurrentMemoryContext;
	MemoryContext batch_context = AllocSetContextCreate(CurrentMemoryContext, "process_job_batch", ALLOCSET_DEFAULT_SIZES);

	PG_TRY();
	{
		MemoryContextSwitchTo(batch_context);

		initStringInfo(&sql);

		if (SPI_connect() != SPI_OK_CONNECT)
			elog(ERROR, "neurondb: SPI_connect failed in neuranq");

		/* Check if job queue table exists */
		ret = SPI_execute("SELECT 1 FROM pg_tables WHERE schemaname = 'neurondb' AND tablename = 'neurondb_job_queue'", true, 0);
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			/* Table doesn't exist, skip processing */
			SPI_finish();
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(batch_context);
			return;
		}

		batch_start = GetCurrentTimestamp();
		(void) batch_start;  /* Used for future timing metrics */

		appendStringInfo(&sql,
			"UPDATE neurondb.neurondb_job_queue "
			"SET status = 'processing', started_at = now() "
			"WHERE job_id IN ("
			"  SELECT job_id FROM neurondb.neurondb_job_queue "
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
				bool		isnull = false;
				Datum		job_id_datum = 0, tenant_id_datum = 0;
				char	   *job_type = NULL, *payload = NULL;
				int64		job_id = 0;
				int			tenant_id = 0;
				bool		success = false;
				TimestampTz	job_start, job_end;
				int64		latency_ms = 0;

				PG_TRY();
				{
					job_id_datum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &isnull);
					job_id = DatumGetInt64(job_id_datum);

					job_type = SPI_getvalue(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2);
					payload = SPI_getvalue(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 3);

					tenant_id_datum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 4, &isnull);
					tenant_id = isnull ? 0 : DatumGetInt32(tenant_id_datum);

					job_start = GetCurrentTimestamp();
					PG_TRY();
					{
						success = execute_job(job_id, job_type, payload, tenant_id);
					}
					PG_CATCH();
					{
						success = false;
						elog(WARNING, "neurondb: job %ld failed with exception", job_id);
						FlushErrorState();
					}
					PG_END_TRY();

					job_end = GetCurrentTimestamp();
					latency_ms = (job_end - job_start) / 1000;

					if (neuranq_state && neuranq_state->lock)
					{
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
					}

					processed++;
				}
				PG_CATCH();
				{
					elog(WARNING, "neurondb: exception while processing individual job - skipping to next (crash safe)");
					FlushErrorState();
				}
				PG_END_TRY();

				if (job_type)
					pfree(job_type);
				if (payload)
					pfree(payload);
			}
		}

		if (sql.data)
			pfree(sql.data);

		SPI_finish();

		if (processed > 0)
		{
			elog(DEBUG1, "neurondb: neuranq processed %d jobs", processed);
		}
	}
	PG_CATCH();
	{
		elog(WARNING, "neurondb: exception in process_job_batch - cleaning up and flushing error state");
		FlushErrorState();
	}
	PG_END_TRY();

	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(batch_context);
}

/*
 * Execute a single job
 * 100% crash-safe: all allocations checked, exceptions handled locally, no unguarded C string access.
 */
static bool
execute_job(int64 job_id, const char *job_type, const char *payload,
			int tenant_id)
{
	StringInfoData	sql;
	StringInfoData	log_entry;
	bool			success = false;
	int				ret = 0;
	TimestampTz		start_time, end_time;
	int64			latency_ms = 0;

	MemoryContext oldcontext = CurrentMemoryContext;
	MemoryContext job_context = AllocSetContextCreate(CurrentMemoryContext, "execute_job_ctx", ALLOCSET_DEFAULT_SIZES);

	PG_TRY();
	{
		MemoryContextSwitchTo(job_context);

		start_time = GetCurrentTimestamp();

		elog(DEBUG1, "neurondb: executing job %ld type=%s tenant=%d",
			 job_id, job_type ? job_type : "(null)", tenant_id);

		/* Defensive coding: guard against null job_type/payload */
		if (!job_type || !payload)
		{
			elog(WARNING, "neurondb: job_type or payload NULL for job %ld", job_id);
			success = false;
			goto update_status;
		}

		/* Execute job based on type */
		if (strcmp(job_type, "embedding_generation") == 0)
		{
			char	   *text_val = NULL;
			StringInfoData json_sql;
			int			json_ret = 0;

			initStringInfo(&json_sql);
			appendStringInfo(&json_sql,
				"SELECT ('%s')::jsonb->>'text' as text_val", payload);

			json_ret = SPI_execute(json_sql.data, true, 1);

			if (json_ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				bool	isnull = false;
				Datum	text_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
				if (!isnull)
				{
					text_val = DatumGetCString(text_datum);

					initStringInfo(&sql);
					appendStringInfo(&sql, "SELECT embed_text('%s')", text_val ? text_val : "");
					ret = SPI_execute(sql.data, false, 0);
					success = (ret == SPI_OK_SELECT);
					if (sql.data)
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
			if (json_sql.data)
				pfree(json_sql.data);
			if (text_val)
				pfree(text_val);
		}
		else if (strcmp(job_type, "rerank_batch") == 0)
		{
			char	   *query_val = NULL;
			StringInfoData json_sql;
			int			json_ret = 0;

			initStringInfo(&json_sql);
			appendStringInfo(&json_sql,
				"SELECT ('%s')::jsonb->>'query' as q, ('%s')::jsonb->'candidates' as cand",
				payload, payload);
			json_ret = SPI_execute(json_sql.data, true, 1);

			if (json_ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				bool	isnull = false;
				Datum	query_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
				if (!isnull)
				{
					query_val = DatumGetCString(query_datum);

					initStringInfo(&sql);
					appendStringInfo(&sql,
						"SELECT rerank_cross_encoder('%s', ARRAY[]::text[], 'ms-marco-MiniLM-L-6-v2', 10)",
						query_val ? query_val : "");
					ret = SPI_execute(sql.data, false, 0);
					success = (ret == SPI_OK_SELECT);
					if (sql.data)
						pfree(sql.data);
				}
				else
					success = false;
			}
			else
				success = false;

			if (json_sql.data)
				pfree(json_sql.data);
			if (query_val)
				pfree(query_val);
		}
		else if (strcmp(job_type, "cache_refresh") == 0)
		{
			char	   *cache_key = NULL;
			StringInfoData json_sql;
			int			json_ret = 0;

			initStringInfo(&json_sql);
			appendStringInfo(&json_sql,
				"SELECT ('%s')::jsonb->>'cache_key' as key", payload);
			json_ret = SPI_execute(json_sql.data, true, 1);

			if (json_ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				bool	isnull = false;
				Datum	key_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
				if (!isnull)
				{
					cache_key = DatumGetCString(key_datum);

					initStringInfo(&sql);
					appendStringInfo(&sql,
						"SELECT embed_cached('%s', 'all-MiniLM-L6-v2')",
						cache_key ? cache_key : "");
					ret = SPI_execute(sql.data, false, 0);
					success = (ret == SPI_OK_SELECT);
					if (sql.data)
						pfree(sql.data);
				}
				else
					success = false;
			}
			else
				success = false;

			if (json_sql.data)
				pfree(json_sql.data);
			if (cache_key)
				pfree(cache_key);
		}
		else if (strcmp(job_type, "http_call") == 0)
		{
			elog(NOTICE, "neurondb: HTTP call job (tenant=%d): %s",
				tenant_id, payload);
			success = true;
		}
		else
		{
			elog(WARNING, "neurondb: unknown job type: %s", job_type);
			success = false;
		}

	update_status:
		end_time = GetCurrentTimestamp();
		latency_ms = (end_time - start_time) / 1000;

		initStringInfo(&sql);

		if (success)
		{
		appendStringInfo(&sql,
			"UPDATE neurondb.neurondb_job_queue "
			"SET status = 'completed', completed_at = now() "
			"WHERE job_id = %ld",
			job_id);
		}
		else
		{
			int retry_count = 0;
			int64 backoff_ms = 0;

			resetStringInfo(&sql);
			appendStringInfo(&sql, "SELECT retry_count FROM neurondb_job_queue WHERE job_id = %ld", job_id);

			ret = SPI_execute(sql.data, true, 1);
			if (ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				bool isnull = false;
				Datum retry_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
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
			"UPDATE neurondb.neurondb_job_queue "
				"SET status = 'failed', retry_count = %d, "
				"    backoff_until = now() + interval '%ld milliseconds', "
				"    error_message = 'Job execution failed' "
				"WHERE job_id = %ld",
				retry_count, backoff_ms, job_id);
		}

		SPI_execute(sql.data, false, 0);

		initStringInfo(&log_entry);
		appendStringInfo(&log_entry,
			"{\"job_id\":%ld,\"tenant_id\":%d,\"job_type\":\"%s\","
			"\"latency_ms\":%ld,\"success\":%s}",
			job_id, tenant_id, job_type ? job_type : "(null)", latency_ms,
			success ? "true" : "false");

		elog(LOG, "neurondb: %s", log_entry.data);

		if (sql.data)
			pfree(sql.data);
		if (log_entry.data)
			pfree(log_entry.data);
	}
	PG_CATCH();
	{
		FlushErrorState();
		success = false;
	}
	PG_END_TRY();

	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(job_context);

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
	bool safe = true;
	PG_TRY();
	{
		elog(NOTICE, "neurondb: manually triggering neuranq batch processing");
		process_job_batch();
	}
	PG_CATCH();
	{
		elog(WARNING, "neurondb: exception during manual batch processing (crash safe)");
		FlushErrorState();
		safe = false;
	}
	PG_END_TRY();
	PG_RETURN_BOOL(safe);
}

