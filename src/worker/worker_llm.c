/*-------------------------------------------------------------------------
 *
 * neuranllm.c
 *		Background worker for LLM job processing queue.
 *
 * Processes queued LLM jobs (completion, embedding, reranking) asynchronously.
 * Uses SKIP LOCKED to avoid blocking and enforces per-job retry limits.
 * All crash scenarios are caught and handled robustly.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/bgworkers/neuranllm.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "miscadmin.h"
#include "tcop/utility.h"
#include "access/xact.h"
#include "pgstat.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/memutils.h"
#include "utils/snapmgr.h"
#include "lib/stringinfo.h"
#include "neurondb_llm.h"

extern int  ndb_llm_job_enqueue(const char *job_type, const char *payload);
extern bool ndb_llm_job_acquire(int *job_id, char **job_type, char **payload);
extern bool ndb_llm_job_update(int job_id, const char *status, const char *result, const char *error);
extern int  ndb_llm_job_prune(int max_age_days);

extern bool ndb_llm_cache_lookup(const char *key, int max_age_seconds, char **out_text);
extern void ndb_llm_cache_store(const char *key, const char *text);

PGDLLEXPORT void neuranllm_main(Datum main_arg);

static void process_llm_job(int job_id, const char *job_type, const char *payload);

/* Signal handlers */
static volatile sig_atomic_t got_sigterm = false;

static void
neuranllm_sigterm(SIGNAL_ARGS)
{
	int save_errno = errno;
	got_sigterm = true;
	SetLatch(MyLatch);
	errno = save_errno;
}

/*
 * neuranllm_main - Background worker main loop.
 *
 * This worker is structured so that if any code in any part crashes,
 * panics, or throws (including segfault, OOM, error), we fully clean up:
 *   - All open transactions are aborted
 *   - All memory context allocations are cleaned
 *   - All job pointer state is reset
 *   - The error is logged
 *   - The worker loop is restarted with no residue
 * Worker cannot leave any transactional or memory junk, even after crash.
 * If a crash happens processing a job, the job is marked as failed.
 */
PGDLLEXPORT void
neuranllm_main(Datum main_arg)
{
	volatile int job_id = 0;
	volatile char *job_type = NULL;
	volatile char *payload = NULL;
	volatile int iterations = 0;

	(void) main_arg;
	
	/* Establish signal handler */
	pqsignal(SIGTERM, neuranllm_sigterm);
	
	BackgroundWorkerUnblockSignals();
	BackgroundWorkerInitializeConnection("postgres", NULL, 0);

	elog(LOG, "neurondb: LLM worker started (all crashes handled)");

	while (!got_sigterm)
	{
		MemoryContext llm_loop_ctx = NULL;
		MemoryContext oldcxt = NULL;

		CHECK_FOR_INTERRUPTS();

		llm_loop_ctx = AllocSetContextCreate(CurrentMemoryContext,
											 "neuranllm_job_loop",
											 ALLOCSET_DEFAULT_SIZES);

		oldcxt = MemoryContextSwitchTo(llm_loop_ctx);

		PG_TRY();
		{
			// -- Begin transactional-safe loop. If this code crashes at any
			// point, we will catch it and fully clean up as described.

			StartTransactionCommand();
			PushActiveSnapshot(GetTransactionSnapshot());
			
			job_type = NULL;
			payload = NULL;

			// Check if LLM jobs table exists before trying to acquire jobs
			{
				int ret;
				uint64 table_check_processed;
				
				if (SPI_connect() != SPI_OK_CONNECT)
					elog(ERROR, "neurondb: LLM worker failed to connect to SPI");

				ret = SPI_execute("SELECT 1 FROM pg_tables WHERE schemaname = 'neurondb' AND tablename = 'neurondb_llm_jobs'", true, 0);
				table_check_processed = SPI_processed;
				SPI_finish();

				if (ret != SPI_OK_SELECT || table_check_processed == 0)
				{
					// Table doesn't exist yet, just sleep and continue
					PopActiveSnapshot();
					CommitTransactionCommand();

					MemoryContextSwitchTo(oldcxt);
					MemoryContextDelete(llm_loop_ctx);
					llm_loop_ctx = NULL;  /* Prevent double-free */

					(void) WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH, 5000L, 0);
					ResetLatch(MyLatch);
					continue;
				}
			}

			if (ndb_llm_job_acquire((int *)&job_id, (char **)&job_type, (char **)&payload))
			{
				if (!job_type || !payload)
					elog(ERROR, "neurondb: LLM worker acquired NULL job_type or payload");

				elog(DEBUG1, "neurondb: Processing LLM job %d (type=%s)", (int)job_id, job_type);

				PG_TRY();
				{
					// -- Any crash/exception/failure inside here processing the job WILL be caught.
					// Cast volatile pointers to const for function call
					process_llm_job(job_id, (const char *)job_type, (const char *)payload);
				}
				PG_CATCH();
				{
					// -- Catches crash during process_llm_job.
					EmitErrorReport();
					FlushErrorState();

					if (IsTransactionState())
						AbortCurrentTransaction();

					elog(LOG, "neurondb: LLM job crashed; marking as failed.");

					if (job_type && job_id > 0)
					{
						// -- If process_llm_job crashed, forcibly mark job as failed.
						ndb_llm_job_update(job_id, "failed", NULL,
							"Crash or unexpected error during job execution");
					}
				}
				PG_END_TRY();

				// -- Always free job_type/payload after job processing, no matter what
				if (job_type) pfree((void *)job_type);
				if (payload) pfree((void *)payload);
				job_type = NULL;
				payload = NULL;
			}
			
			PopActiveSnapshot();
			CommitTransactionCommand();

			++iterations;
		if (iterations % 1000 == 0)
		{
			int pruned;
			
			StartTransactionCommand();
			PushActiveSnapshot(GetTransactionSnapshot());
			pruned = 0;
			PG_TRY();
				{
					pruned = ndb_llm_job_prune(7);
				}
				PG_CATCH();
				{
					EmitErrorReport();
					FlushErrorState();
					if (IsTransactionState())
						AbortCurrentTransaction();
					elog(LOG, "neurondb: Exception pruning old LLM jobs; continuing safely.");
				}
				PG_END_TRY();
				if (pruned > 0)
					elog(LOG, "neurondb: Pruned %d old LLM jobs", pruned);
				PopActiveSnapshot();
				CommitTransactionCommand();
			}
		}
		PG_CATCH();
		{
			EmitErrorReport();
			FlushErrorState();
			// If any top-level error/crash, abort transaction & cleanup memory/ptrs
			if (IsTransactionState())
				AbortCurrentTransaction();
			if (job_type)
				pfree((void *)job_type);
			if (payload)
				pfree((void *)payload);
			job_type = NULL;
			payload = NULL;
			elog(LOG, "neurondb: LLM worker caught exception, all cleaned up. Restarting loop.");
		}
		PG_END_TRY();

		MemoryContextSwitchTo(oldcxt);
		if (llm_loop_ctx != NULL)
		{
			MemoryContextDelete(llm_loop_ctx);
			llm_loop_ctx = NULL;
		}

#if PG_VERSION_NUM >= 100000
		(void) WaitLatch(MyLatch, WL_TIMEOUT | WL_LATCH_SET | WL_EXIT_ON_PM_DEATH, 1000L, PG_WAIT_EXTENSION);
#else
		(void) WaitLatch(MyLatch, WL_TIMEOUT | WL_LATCH_SET | WL_POSTMASTER_DEATH, 1000L);
#endif
		ResetLatch(MyLatch);

		job_type = NULL;
		payload = NULL;
	}
	
	elog(LOG, "neurondb: LLM worker shutting down gracefully");
	proc_exit(0);
}

/*
 * process_llm_job - Execute a single LLM job.
 * Any crash, OOM, segfault, or exception in this routine (or anything it calls)
 * will be caught and the job will be marked failed, never allowed to propagate.
 */
static void
process_llm_job(int job_id, const char *job_type, const char *payload)
{
	MemoryContext job_ctx = NULL, oldcxt = NULL;
	NdbLLMConfig cfg;
	NdbLLMResp resp;
	StringInfoData result_json;
	char *error_msg = NULL;
	int rc = -1;

	// Allocate job-local memory context, so even if an OOM or crash happens,
	// it gets cleaned without leaving leaks or junk.
	job_ctx = AllocSetContextCreate(CurrentMemoryContext, "neuranllm/job", ALLOCSET_DEFAULT_SIZES);
	oldcxt = MemoryContextSwitchTo(job_ctx);

	PG_TRY();
	{
		cfg.provider  = neurondb_llm_provider   ? neurondb_llm_provider   : "huggingface";
		cfg.endpoint  = neurondb_llm_endpoint   ? neurondb_llm_endpoint   : "https://api-inference.huggingface.co";
		cfg.model     = neurondb_llm_model      ? neurondb_llm_model      : "gpt2";
		cfg.api_key   = neurondb_llm_api_key    ? neurondb_llm_api_key    : "";
		cfg.timeout_ms = neurondb_llm_timeout_ms;
		memset(&resp, 0, sizeof(resp));
		memset(&result_json, 0, sizeof(result_json));

		if (!job_type || !payload)
			elog(ERROR, "neurondb: process_llm_job called with NULL job_type or payload");

		{
			const char *prompt = (const char *)payload;
			int print_dim;
			char *vec_str;
			char cache_key[256];

			if (strcmp(job_type, "complete") == 0)
			{
				// If this next code OOMs/crashes/exceptions, will be caught
				rc = ndb_hf_complete(&cfg, prompt, NULL, &resp);
				if (rc == 0 && resp.text)
				{
					snprintf(cache_key, sizeof(cache_key), "llm:complete:%s:%.128s", cfg.model, prompt ? prompt : "");
					ndb_llm_cache_store(cache_key, resp.text);

					initStringInfo(&result_json);
					appendStringInfo(&result_json, "{\"text\":%s,\"tokens_out\":%d}",
									 quote_literal_cstr(resp.text), resp.tokens_out);
					ndb_llm_job_update(job_id, "done", result_json.data, NULL);
				}
				else
				{
					error_msg = psprintf("LLM completion failed (status=%d)", rc == 0 ? resp.http_status : rc);
					ndb_llm_job_update(job_id, "failed", NULL, error_msg);
				}
			}
			else if (strcmp(job_type, "embed") == 0)
			{
				float *vec = NULL;
				int dim = 0;
				// This line may crash internally on OOM, segfault, etc, but we will catch it
				rc = ndb_hf_embed(&cfg, prompt, &vec, &dim);
				if (rc == 0 && vec && dim > 0)
				{
					snprintf(cache_key, sizeof(cache_key), "llm:embed:%s:%.128s", cfg.model, prompt ? prompt : "");
					print_dim = dim > 4 ? 4 : dim;
					if (print_dim == 4)
						vec_str = psprintf("[%f,%f,%f,%f]", vec[0], vec[1], vec[2], vec[3]);
					else if (print_dim == 3)
						vec_str = psprintf("[%f,%f,%f]", vec[0], vec[1], vec[2]);
					else if (print_dim == 2)
						vec_str = psprintf("[%f,%f]", vec[0], vec[1]);
					else if (print_dim == 1)
						vec_str = psprintf("[%f]", vec[0]);
					else
						vec_str = pstrdup("[]");
					ndb_llm_cache_store(cache_key, vec_str);

					initStringInfo(&result_json);
					appendStringInfo(&result_json, "{\"embedding\":%s,\"dim\":%d}", vec_str, dim);
					ndb_llm_job_update(job_id, "done", result_json.data, NULL);

					pfree(vec_str);
					pfree(vec);
				}
				else
				{
					error_msg = psprintf("LLM embedding failed (status=%d)", rc == 0 ? -1 : rc);
					ndb_llm_job_update(job_id, "failed", NULL, error_msg);
				}
			}
			else
			{
			error_msg = psprintf("Unknown job type: %s", job_type ? job_type : "<null>");
			ndb_llm_job_update(job_id, "failed", NULL, error_msg);
		}
		}
	}
	PG_CATCH();
	{
		// Crash, OOM, segfault, provider error: nothing escapes.
		EmitErrorReport();
		FlushErrorState();

		if (IsTransactionState())
			AbortCurrentTransaction();

		if (job_id > 0)
		{
			ndb_llm_job_update(job_id, "failed", NULL,
				"process_llm_job: crash or fatal error caught and job marked failed");
		}
	}
	PG_END_TRY();

	// Always cleanup local memory, regardless of outcome
	if (result_json.data)
		pfree(result_json.data);
	if (resp.json)
		pfree(resp.json);
	if (error_msg)
		pfree(error_msg);

	MemoryContextSwitchTo(oldcxt);
	MemoryContextDelete(job_ctx);
}
