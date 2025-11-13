#include "postgres.h"
#include "fmgr.h"
#include "utils/guc.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "executor/spi.h"
#include "access/xact.h"
#include "utils/array.h"
#include "funcapi.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "utils/timestamp.h"
#include "lib/stringinfo.h"
#include "catalog/pg_type.h"
#include "openssl/sha.h"
#include "neurondb.h"
#include "neurondb_llm.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_onnx.h"
#include <ctype.h>

/* ---- GUCs ---- */
char *neurondb_llm_provider = NULL;
char *neurondb_llm_model = NULL;
char *neurondb_llm_endpoint = NULL;
char *neurondb_llm_api_key = NULL;
int neurondb_llm_timeout_ms = 30000;
int neurondb_llm_cache_ttl = 600;
int neurondb_llm_rate_limiter_qps = 5;
bool neurondb_llm_fail_open = true;

/* ---- Shared rate-limiter (Token Bucket) ---- */
typedef struct NdbLLMRateLimiter
{
	double tokens;
	TimestampTz last_refill;
	int capacity; /* tokens per second */
} NdbLLMRateLimiter;

static NdbLLMRateLimiter *llm_rl = NULL;
static LWLock *llm_rl_lock = NULL;

/* Returns the size required for shared memory allocation for the rate limiter. */
Size
neurondb_llm_shmem_size(void)
{
	return MAXALIGN(sizeof(NdbLLMRateLimiter));
}

/* Initializes the shared memory for the LLM rate limiter and sets up the lock. */
void
neurondb_llm_shmem_init(void)
{
	bool found = false;
	llm_rl = (NdbLLMRateLimiter *)ShmemInitStruct(
		"neurondb_llm_rate", sizeof(NdbLLMRateLimiter), &found);
	llm_rl_lock = &(GetNamedLWLockTranche("neurondb_llm")[0].lock);
	if (!found)
	{
		/* First time initialization. */
		llm_rl->tokens = 0.0;
		llm_rl->last_refill = GetCurrentTimestamp();
		llm_rl->capacity = neurondb_llm_rate_limiter_qps > 0
			? neurondb_llm_rate_limiter_qps
			: 5;
	}
}

/* Attempts to acquire a token from the rate limiter; returns true if allowed. */
static bool
llm_acquire_token(void)
{
	TimestampTz now = GetCurrentTimestamp();
	long secs;
	int usecs;
	double elapsed;
	int cap;

	if (llm_rl == NULL || llm_rl_lock == NULL)
		return true; /* No limiter initialized, always allow. */

	LWLockAcquire(llm_rl_lock, LW_EXCLUSIVE);

	cap = neurondb_llm_rate_limiter_qps > 0 ? neurondb_llm_rate_limiter_qps
						: llm_rl->capacity;

	TimestampDifference(llm_rl->last_refill, now, &secs, &usecs);
	elapsed = (double)secs + ((double)usecs / 1000000.0);

	if (elapsed > 0.0)
	{
		llm_rl->tokens =
			Min((double)cap, llm_rl->tokens + elapsed * cap);
		llm_rl->last_refill = now;
		llm_rl->capacity = cap;
	}
	if (llm_rl->tokens >= 1.0)
	{
		llm_rl->tokens -= 1.0;
		LWLockRelease(llm_rl_lock);
		return true;
	}
	LWLockRelease(llm_rl_lock);
	return false;
}

/* Helper: Calculate latency in milliseconds */
static double
ndb_elapsed_ms(TimestampTz start, TimestampTz end)
{
	long secs;
	int usecs;

	TimestampDifference(start, end, &secs, &usecs);
	return ((double)secs * 1000.0) + ((double)usecs / 1000.0);
}

/* Defines the GUC (config) variables. */
void
neurondb_llm_init_guc(void)
{
	DefineCustomStringVariable("neurondb.llm_provider",
		"LLM provider",
		NULL,
		&neurondb_llm_provider,
		"huggingface",
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomStringVariable("neurondb.llm_model",
		"Default LLM model id",
		NULL,
		&neurondb_llm_model,
		"sentence-transformers/all-MiniLM-L6-v2",
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomStringVariable("neurondb.llm_endpoint",
		"LLM endpoint base URL",
		NULL,
		&neurondb_llm_endpoint,
		"https://api-inference.huggingface.co",
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomStringVariable("neurondb.llm_api_key",
		"LLM API key (set via ALTER SYSTEM or env)",
		NULL,
		&neurondb_llm_api_key,
		"",
		PGC_SUSET,
		GUC_SUPERUSER_ONLY,
		NULL,
		NULL,
		NULL);

	DefineCustomIntVariable("neurondb.llm_timeout_ms",
		"HTTP timeout (ms)",
		NULL,
		&neurondb_llm_timeout_ms,
		30000,
		1000,
		600000,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomIntVariable("neurondb.llm_cache_ttl",
		"Cache TTL seconds",
		NULL,
		&neurondb_llm_cache_ttl,
		600,
		0,
		86400,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomIntVariable("neurondb.llm_rate_limiter_qps",
		"Rate limiter QPS",
		NULL,
		&neurondb_llm_rate_limiter_qps,
		5,
		1,
		10000,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomBoolVariable("neurondb.llm_fail_open",
		"Fail open on provider errors",
		NULL,
		&neurondb_llm_fail_open,
		true,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);
}

/* Computes a unique cache key for LLM requests by hashing input params. */
static void
compute_cache_key(StringInfo dst,
	const char *provider,
	const char *model,
	const char *endpoint,
	const char *payload)
{
	unsigned char hash[SHA256_DIGEST_LENGTH];
	StringInfoData src;
	int i;

	initStringInfo(&src);
	appendStringInfo(&src,
		"%s|%s|%s|%s",
		provider ? provider : "",
		model ? model : "",
		endpoint ? endpoint : "",
		payload ? payload : "");
	SHA256((unsigned char *)src.data, src.len, hash);

	initStringInfo(dst);
	for (i = 0; i < SHA256_DIGEST_LENGTH; i++)
		appendStringInfo(&dst[0], "%02x", hash[i]);
}

/* Looks up cache entry for a given key; if found, returns text via out pointer. */
static bool
cache_lookup_text(const char *key, char **text_out)
{
	bool hit = false;
	if (SPI_connect() != SPI_OK_CONNECT)
		return false;

	if (SPI_execute_with_args(
		    "SELECT value->>'text' FROM neurondb.neurondb_llm_cache WHERE key = "
		    "$1 AND now() - created_at < make_interval(secs => $2)",
		    2,
		    (Oid[]) { TEXTOID, INT4OID },
		    (Datum[]) { CStringGetTextDatum(key),
			    Int32GetDatum(neurondb_llm_cache_ttl) },
		    (char[]) { 'i', 'i' },
		    true,
		    0) == SPI_OK_SELECT
		&& SPI_processed == 1)
	{
		bool isnull;
		Datum d = SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc,
			1,
			&isnull);
		if (!isnull)
		{
			*text_out = TextDatumGetCString(d);
			hit = true;
		}
	}
	SPI_finish();
	return hit;
}

/* Stores a given value in the neurondb_llm_cache table, replacing on conflict. */
static void
cache_store_text(const char *key, const char *text)
{
	if (SPI_connect() != SPI_OK_CONNECT)
		return;
	Oid argtypes[2] = { TEXTOID, JSONBOID };
	Datum values[2];
	StringInfoData val;
	initStringInfo(&val);
	appendStringInfo(&val, "{\"text\":%s}", quote_literal_cstr(text));
	values[0] = CStringGetTextDatum(key);
	values[1] = CStringGetTextDatum(val.data);
	SPI_execute_with_args(
		"INSERT INTO neurondb.neurondb_llm_cache(key,value,created_at) "
		"VALUES($1,$2::jsonb,now()) "
		"ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, "
		"created_at=now()",
		2,
		argtypes,
		values,
		NULL,
		false,
		0);
	SPI_finish();
}

/* Record LLM statistics in neurondb_llm_stats table */
static void
record_llm_stats(const char *model_name,
	const char *operation,
	bool success,
	bool cache_hit,
	int64 latency_ms,
	int tokens_in,
	int tokens_out,
	const char *error_type)
{
	int ret;
	Oid argtypes[8];
	Datum values[8];
	char nulls[8];

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	/* Check if table exists */
	ret = SPI_execute("SELECT 1 FROM pg_tables WHERE schemaname = "
			  "'neurondb' AND tablename = 'neurondb_llm_stats'",
		true,
		0);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		/* Table doesn't exist, skip statistics recording */
		SPI_finish();
		return;
	}

	argtypes[0] = TEXTOID;
	argtypes[1] = TEXTOID;
	argtypes[2] = BOOLOID;
	argtypes[3] = BOOLOID;
	argtypes[4] = INT8OID;
	argtypes[5] = INT4OID;
	argtypes[6] = INT4OID;
	argtypes[7] = TEXTOID;

	values[0] = CStringGetTextDatum(model_name ? model_name : "unknown");
	values[1] = CStringGetTextDatum(operation ? operation : "unknown");
	nulls[0] = ' ';
	nulls[1] = ' ';

	if (success)
	{
		values[2] = BoolGetDatum(true);
		values[3] = BoolGetDatum(cache_hit);
		values[4] = Int64GetDatum(latency_ms);
		values[5] = Int32GetDatum(tokens_in);
		values[6] = Int32GetDatum(tokens_out);
		values[7] = (Datum)0;
		nulls[2] = ' ';
		nulls[3] = ' ';
		nulls[4] = ' ';
		nulls[5] = ' ';
		nulls[6] = ' ';
		nulls[7] = 'n';
	} else
	{
		values[2] = BoolGetDatum(false);
		values[3] = BoolGetDatum(false);
		values[4] = Int64GetDatum(latency_ms);
		values[5] = Int32GetDatum(tokens_in);
		values[6] = Int32GetDatum(tokens_out);
		values[7] = CStringGetTextDatum(
			error_type ? error_type : "unknown");
		nulls[2] = ' ';
		nulls[3] = ' ';
		nulls[4] = ' ';
		nulls[5] = ' ';
		nulls[6] = ' ';
		nulls[7] = ' ';
	}

	/* Update LLM statistics */
	SPI_execute_with_args(
		"INSERT INTO neurondb.neurondb_llm_stats (model_name, "
		"total_requests, successful_requests, failed_requests, "
		"cache_hits, total_latency_ms, total_tokens, total_tokens_in, "
		"total_tokens_out, last_request_at, last_updated) "
		"VALUES ($1, 1, "
		"  CASE WHEN $3 THEN 1 ELSE 0 END, "
		"  CASE WHEN $3 THEN 0 ELSE 1 END, "
		"  CASE WHEN $4 THEN 1 ELSE 0 END, "
		"  $5, $6 + $7, $6, $7, now(), now()) "
		"ON CONFLICT (model_name) DO UPDATE SET "
		"  total_requests = neurondb_llm_stats.total_requests + 1, "
		"  successful_requests = "
		"neurondb_llm_stats.successful_requests + CASE WHEN $3 THEN 1 "
		"ELSE 0 END, "
		"  failed_requests = neurondb_llm_stats.failed_requests + CASE "
		"WHEN $3 THEN 0 ELSE 1 END, "
		"  cache_hits = neurondb_llm_stats.cache_hits + CASE WHEN $4 "
		"THEN 1 ELSE 0 END, "
		"  total_latency_ms = neurondb_llm_stats.total_latency_ms + "
		"$5, "
		"  total_tokens = neurondb_llm_stats.total_tokens + $6 + $7, "
		"  total_tokens_in = neurondb_llm_stats.total_tokens_in + $6, "
		"  total_tokens_out = neurondb_llm_stats.total_tokens_out + "
		"$7, "
		"  last_request_at = now(), "
		"  last_updated = now()",
		8,
		argtypes,
		values,
		nulls,
		false,
		0);

	/* Record error in error tracking table if error occurred */
	if (!success && error_type)
	{
		Oid error_argtypes[4] = { TEXTOID, TEXTOID, TEXTOID, INT8OID };
		Datum error_values[4];
		char error_nulls[4] = { ' ', ' ', ' ', ' ' };

		error_values[0] = CStringGetTextDatum(
			model_name ? model_name : "unknown");
		error_values[1] =
			CStringGetTextDatum(operation ? operation : "unknown");
		error_values[2] = CStringGetTextDatum(error_type);
		error_values[3] = Int64GetDatum(latency_ms);

		/* Check if error tracking table exists */
		ret = SPI_execute(
			"SELECT 1 FROM pg_tables WHERE schemaname = 'neurondb' "
			"AND tablename = 'neurondb_llm_errors'",
			true,
			0);
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			SPI_execute_with_args(
				"INSERT INTO neurondb.neurondb_llm_errors "
				"(model_name, operation, error_type, "
				"latency_ms, error_timestamp) "
				"VALUES ($1, $2, $3, $4, now())",
				4,
				error_argtypes,
				error_values,
				error_nulls,
				false,
				0);
		}
	}

	/* Record latency histogram */
	ret = SPI_execute("SELECT 1 FROM pg_tables WHERE schemaname = "
			  "'neurondb' AND tablename = 'neurondb_histograms'",
		true,
		0);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		int bucket_start = ((int)(latency_ms / 10.0)) * 10;
		int bucket_end = bucket_start + 10;
		Oid hist_argtypes[4] = {
			TEXTOID, FLOAT4OID, FLOAT4OID, INT8OID
		};
		Datum hist_values[4];
		char hist_nulls[4] = { ' ', ' ', ' ', ' ' };
		StringInfoData metric_name;
		char *metric_name_str;

		initStringInfo(&metric_name);
		appendStringInfo(&metric_name,
			"llm_%s_%s_latency_ms",
			operation ? operation : "unknown",
			model_name ? model_name : "unknown");
		metric_name_str = pstrdup(metric_name.data);

		hist_values[0] = CStringGetTextDatum(metric_name_str);
		hist_values[1] = Float4GetDatum((float)bucket_start);
		hist_values[2] = Float4GetDatum((float)bucket_end);
		hist_values[3] = Int64GetDatum(1);

		SPI_execute_with_args(
			"INSERT INTO neurondb.neurondb_histograms "
			"(metric_name, bucket_start, bucket_end, count, "
			"last_updated) "
			"VALUES ($1, $2, $3, $4, now()) "
			"ON CONFLICT (metric_name, bucket_start) DO UPDATE SET "
			"  count = neurondb_histograms.count + $4, "
			"  last_updated = now()",
			4,
			hist_argtypes,
			hist_values,
			hist_nulls,
			false,
			0);

		pfree(metric_name.data);
		pfree(metric_name_str);
	}

	SPI_finish();
}

/* Fills an NdbLLMConfig struct from the GUCs */
static void
fill_cfg(NdbLLMConfig *cfg)
{
	cfg->provider =
		neurondb_llm_provider ? neurondb_llm_provider : "huggingface";
	cfg->endpoint = neurondb_llm_endpoint
		? neurondb_llm_endpoint
		: "https://api-inference.huggingface.co";
	cfg->model = neurondb_llm_model
		? neurondb_llm_model
		: "sentence-transformers/all-MiniLM-L6-v2";
	cfg->api_key = neurondb_llm_api_key;
	cfg->timeout_ms = neurondb_llm_timeout_ms;
	cfg->prefer_gpu = neurondb_gpu_enabled;
	cfg->require_gpu = false;
	if (cfg->provider != NULL
		&& (pg_strcasecmp(cfg->provider, "huggingface-local") == 0
			|| pg_strcasecmp(cfg->provider, "hf-local") == 0)
		&& !neurondb_llm_fail_open)
		cfg->require_gpu = true;
}

/* SQL: llm_complete(prompt text, params jsonb) RETURNS text
 *
 * Calls the configured LLM to complete a prompt, with input params as JSONB string.
 * Uses rate-limiter, in-memory/DB cache, and supports fail-open/fail-closed.
 */
PG_FUNCTION_INFO_V1(ndb_llm_complete);
Datum
ndb_llm_complete(PG_FUNCTION_ARGS)
{
	text *prompt_in = PG_GETARG_TEXT_PP(0);
	text *params_in = (PG_NARGS() > 1 && !PG_ARGISNULL(1))
		? PG_GETARG_TEXT_PP(1)
		: NULL;
	char *prompt = text_to_cstring(prompt_in);
	char *params = params_in ? text_to_cstring(params_in) : "{}";
	NdbLLMConfig cfg;
	NdbLLMResp resp = { 0 };
	NdbLLMCallOptions call_opts;
	StringInfoData keysrc, keyhex;
	char *cached = NULL;
	int rc;
	TimestampTz start_time;
	TimestampTz end_time;
	int64 latency_ms = 0;
	bool cache_hit = false;
	bool success = false;
	const char *error_type = NULL;
	int tokens_in = 0;
	int tokens_out = 0;

	start_time = GetCurrentTimestamp();
	fill_cfg(&cfg);
	call_opts.task = "complete";
	call_opts.prefer_gpu = cfg.prefer_gpu;
	call_opts.require_gpu = cfg.require_gpu;
	call_opts.fail_open = neurondb_llm_fail_open;

	/* Compose a cache key for this input */
	initStringInfo(&keysrc);
	appendStringInfo(&keysrc, "%s|%s|%s", prompt, params, cfg.model);
	compute_cache_key(
		&keyhex, cfg.provider, cfg.model, cfg.endpoint, keysrc.data);

	/* Check cache */
	if (cache_lookup_text(keyhex.data, &cached))
	{
		end_time = GetCurrentTimestamp();
		latency_ms = (int64)ndb_elapsed_ms(start_time, end_time);
		cache_hit = true;
		success = true;
		{
			/* Count tokens from prompt */
			int32 token_length;
			int32 *token_ids = neurondb_tokenize_with_model(
				prompt, 2048, &token_length, cfg.model);
			if (token_ids && token_length > 0)
				tokens_in = token_length;
			else
				tokens_in =
					0; /* Fallback: estimate from word count */
			if (token_ids)
				pfree(token_ids);

				/* Count tokens from cached response using tokenizer when available */
#ifdef HAVE_ONNX_RUNTIME
			PG_TRY();
			{
				int32 output_token_length;
				int32 *output_token_ids =
					neurondb_tokenize_with_model(cached,
						2048,
						&output_token_length,
						cfg.model);
				if (output_token_ids && output_token_length > 0)
				{
					tokens_out = output_token_length;
				} else
				{
					/* Fallback: estimate from word count */
					if (cached && strlen(cached) > 0)
					{
						const char *ptr = cached;
						int word_count = 0;
						int in_word = 0;

						while (*ptr)
						{
							if (!isspace((
								    unsigned char)*ptr))
							{
								if (!in_word)
								{
									word_count++;
									in_word =
										1;
								}
							} else
							{
								in_word = 0;
							}
							ptr++;
						}
						tokens_out = word_count > 0
							? word_count
							: 1;
					} else
					{
						tokens_out = 0;
					}
				}
				if (output_token_ids)
					pfree(output_token_ids);
			}
			PG_CATCH();
			{
				/* On error, use word count fallback */
				EmitErrorReport();
				FlushErrorState();

				if (cached && strlen(cached) > 0)
				{
					const char *ptr = cached;
					int word_count = 0;
					int in_word = 0;

					while (*ptr)
					{
						if (!isspace((
							    unsigned char)*ptr))
						{
							if (!in_word)
							{
								word_count++;
								in_word = 1;
							}
						} else
						{
							in_word = 0;
						}
						ptr++;
					}
					tokens_out =
						word_count > 0 ? word_count : 1;
				} else
				{
					tokens_out = 0;
				}
			}
			PG_END_TRY();
#else
			/* ONNX runtime not available, use word count fallback */
			if (cached && strlen(cached) > 0)
			{
				const char *ptr = cached;
				int word_count = 0;
				int in_word = 0;

				while (*ptr)
				{
					if (!isspace((unsigned char)*ptr))
					{
						if (!in_word)
						{
							word_count++;
							in_word = 1;
						}
					} else
					{
						in_word = 0;
					}
					ptr++;
				}
				tokens_out = word_count > 0 ? word_count : 1;
			} else
			{
				tokens_out = 0;
			}
#endif
		}
		record_llm_stats(cfg.model,
			"complete",
			success,
			cache_hit,
			latency_ms,
			tokens_in,
			tokens_out,
			NULL);
		PG_RETURN_TEXT_P(cstring_to_text(cached));
	}

	/* Shortcut: mock endpoint for test/dev */
	if (cfg.endpoint && strncmp(cfg.endpoint, "mock://", 7) == 0)
	{
		end_time = GetCurrentTimestamp();
		latency_ms = (int64)ndb_elapsed_ms(start_time, end_time);
		cache_hit = false;
		success = true;
		tokens_in = 0;
		tokens_out = 0;
		record_llm_stats(cfg.model,
			"complete",
			success,
			cache_hit,
			latency_ms,
			tokens_in,
			tokens_out,
			NULL);
		PG_RETURN_TEXT_P(cstring_to_text("mock-completion"));
	}

	/* Rate limit */
	if (!llm_acquire_token())
	{
		end_time = GetCurrentTimestamp();
		latency_ms = (int64)ndb_elapsed_ms(start_time, end_time);
		cache_hit = false;
		success = false;
		error_type = "rate_limit";
		if (neurondb_llm_fail_open)
		{
			ereport(WARNING,
				(errmsg("neurondb: LLM rate limited - fail "
					"open")));
			record_llm_stats(cfg.model,
				"complete",
				success,
				cache_hit,
				latency_ms,
				tokens_in,
				tokens_out,
				error_type);
			PG_RETURN_NULL();
		}
		record_llm_stats(cfg.model,
			"complete",
			success,
			cache_hit,
			latency_ms,
			tokens_in,
			tokens_out,
			error_type);
		ereport(ERROR, (errmsg("neurondb: LLM rate limited")));
	}

	/* Route to configured provider (remote Hugging Face or local runtime) */
	rc = ndb_llm_route_complete(&cfg, &call_opts, prompt, params, &resp);
	end_time = GetCurrentTimestamp();
	latency_ms = (int64)ndb_elapsed_ms(start_time, end_time);
	tokens_in = resp.tokens_in;
	tokens_out = resp.tokens_out;

	if (rc != 0)
	{
		cache_hit = false;
		success = false;
		error_type = "provider_error";
		if (neurondb_llm_fail_open)
		{
			record_llm_stats(cfg.model,
				"complete",
				success,
				cache_hit,
				latency_ms,
				tokens_in,
				tokens_out,
				error_type);
			PG_RETURN_NULL();
		}
		record_llm_stats(cfg.model,
			"complete",
			success,
			cache_hit,
			latency_ms,
			tokens_in,
			tokens_out,
			error_type);
		ereport(ERROR, (errmsg("neurondb: llm provider error")));
	}

	if (resp.text)
	{
		cache_store_text(keyhex.data, resp.text);
		cache_hit = false;
		success = true;
		error_type = NULL;
	} else
	{
		cache_hit = false;
		success = false;
		error_type = "empty_response";
	}

	record_llm_stats(cfg.model,
		"complete",
		success,
		cache_hit,
		latency_ms,
		tokens_in,
		tokens_out,
		error_type);
	PG_RETURN_TEXT_P(cstring_to_text(resp.text ? resp.text : ""));
}

/* SQL: llm_embed(txt text, model text default null) RETURNS vector
 *
 * Calls LLM endpoint to embed input text, with optional model override.
 */
PG_FUNCTION_INFO_V1(ndb_llm_embed);
Datum
ndb_llm_embed(PG_FUNCTION_ARGS)
{
	text *txt_in = PG_GETARG_TEXT_PP(0);
	char *txt = text_to_cstring(txt_in);
	NdbLLMConfig cfg;
	float *vec = NULL;
	int dim = 0;
	Vector *v;
	NdbLLMCallOptions call_opts;
	TimestampTz start_time;
	TimestampTz end_time;
	int64 latency_ms = 0;
	bool cache_hit = false;
	bool success = false;
	const char *error_type = NULL;
	int tokens_in = 0;
	int tokens_out = 0;

	start_time = GetCurrentTimestamp();
	fill_cfg(&cfg);
	if (PG_NARGS() > 1 && !PG_ARGISNULL(1))
		cfg.model = text_to_cstring(PG_GETARG_TEXT_PP(1));
	call_opts.task = "embed";
	call_opts.prefer_gpu = cfg.prefer_gpu;
	call_opts.require_gpu = cfg.require_gpu;
	call_opts.fail_open = neurondb_llm_fail_open;

	/* Shortcut: for dev/test, mock output. */
	if (cfg.endpoint && strncmp(cfg.endpoint, "mock://", 7) == 0)
	{
		end_time = GetCurrentTimestamp();
		latency_ms = (int64)ndb_elapsed_ms(start_time, end_time);
		dim = 4;
		v = new_vector(dim);
		v->data[0] = 1;
		v->data[1] = 2;
		v->data[2] = 3;
		v->data[3] = 4;
		cache_hit = false;
		success = true;
		tokens_in = 0;
		tokens_out = 0;
		record_llm_stats(cfg.model,
			"embed",
			success,
			cache_hit,
			latency_ms,
			tokens_in,
			tokens_out,
			NULL);
		PG_RETURN_VECTOR_P(v);
	}

	/* Rate limiting */
	if (!llm_acquire_token())
	{
		end_time = GetCurrentTimestamp();
		latency_ms = (int64)ndb_elapsed_ms(start_time, end_time);
		cache_hit = false;
		success = false;
		error_type = "rate_limit";
		if (neurondb_llm_fail_open)
		{
			ereport(WARNING,
				(errmsg("neurondb: LLM rate limited - fail "
					"open")));
			record_llm_stats(cfg.model,
				"embed",
				success,
				cache_hit,
				latency_ms,
				tokens_in,
				tokens_out,
				error_type);
			PG_RETURN_NULL();
		}
		record_llm_stats(cfg.model,
			"embed",
			success,
			cache_hit,
			latency_ms,
			tokens_in,
			tokens_out,
			error_type);
		ereport(ERROR, (errmsg("neurondb: LLM rate limited")));
	}

	/* Route embedding through configured provider */
	if (ndb_llm_route_embed(&cfg, &call_opts, txt, &vec, &dim) != 0
		|| dim <= 0)
	{
		end_time = GetCurrentTimestamp();
		latency_ms = (int64)ndb_elapsed_ms(start_time, end_time);
		cache_hit = false;
		success = false;
		error_type = "embed_failed";
		if (neurondb_llm_fail_open)
		{
			record_llm_stats(cfg.model,
				"embed",
				success,
				cache_hit,
				latency_ms,
				tokens_in,
				tokens_out,
				error_type);
			PG_RETURN_NULL();
		}
		record_llm_stats(cfg.model,
			"embed",
			success,
			cache_hit,
			latency_ms,
			tokens_in,
			tokens_out,
			error_type);
		ereport(ERROR, (errmsg("neurondb: embed failed")));
	}

	end_time = GetCurrentTimestamp();
	latency_ms = (int64)ndb_elapsed_ms(start_time, end_time);
	v = new_vector(dim);
	memcpy(v->data, vec, sizeof(float) * dim);
	cache_hit = false;
	success = true;
	{
		/* Count input tokens from text */
		int32 token_length;
		int32 *token_ids = neurondb_tokenize_with_model(
			txt, 2048, &token_length, cfg.model);
		if (token_ids && token_length > 0)
			tokens_in = token_length;
		else
			tokens_in = 0; /* Fallback: estimate from word count */
		if (token_ids)
			pfree(token_ids);
	}
	tokens_out = dim; /* Embedding dimension */
	record_llm_stats(cfg.model,
		"embed",
		success,
		cache_hit,
		latency_ms,
		tokens_in,
		tokens_out,
		NULL);
	PG_RETURN_VECTOR_P(v);
}

/* SQL: llm_rerank(query text, docs text[], model text default null, top_n int default 10)
 * RETURNS TABLE(idx int, score real)
 * 
 * This is a mock/test implementation only.
 */
PG_FUNCTION_INFO_V1(ndb_llm_rerank);
Datum
ndb_llm_rerank(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	typedef struct
	{
		int ndocs;
		int i;
	} Ctx;
	Ctx *c;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext old;
		Ctx *new_ctx;
		ArrayType *arr;
		TupleDesc tupdesc;

		funcctx = SRF_FIRSTCALL_INIT();
		old = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
		new_ctx = palloc0(sizeof(Ctx));
		arr = PG_GETARG_ARRAYTYPE_P(1);
		new_ctx->ndocs = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
		new_ctx->i = 0;
		funcctx->user_fctx = new_ctx;

		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(
			tupdesc, (AttrNumber)1, "idx", INT4OID, -1, 0);
		TupleDescInitEntry(
			tupdesc, (AttrNumber)2, "score", FLOAT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		MemoryContextSwitchTo(old);
	}
	funcctx = SRF_PERCALL_SETUP();
	c = (Ctx *)funcctx->user_fctx;

	if (c->i < c->ndocs)
	{
		Datum values[2];
		bool nulls[2] = { false, false };
		HeapTuple tup;

		values[0] = Int32GetDatum(c->i + 1);
		values[1] = Float4GetDatum(
			1.0f - ((float)c->i / (float)Max(1, c->ndocs)));
		tup = heap_form_tuple(funcctx->tuple_desc, values, nulls);
		c->i++;
		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tup));
	}
	SRF_RETURN_DONE(funcctx);
}

/* SQL: llm_enqueue(action text, payload jsonb) RETURNS bigint
 *
 * Enqueue an LLM job in the neurondb_llm_jobs table.
 */
PG_FUNCTION_INFO_V1(ndb_llm_enqueue);
Datum
ndb_llm_enqueue(PG_FUNCTION_ARGS)
{
	Oid argtypes[2] = { TEXTOID, JSONBOID };
	Datum values[2];
	bool isnull;
	Datum d;
	text *action = PG_GETARG_TEXT_PP(0);
	text *payload = PG_GETARG_TEXT_PP(1);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("SPI_connect failed")));

	values[0] = PointerGetDatum(action);
	values[1] = PointerGetDatum(payload);
	SPI_execute_with_args(
		"INSERT INTO "
		"neurondb_llm_jobs(action,payload,status,created_at) "
		"VALUES($1,$2::jsonb,'queued',now()) RETURNING id",
		2,
		argtypes,
		values,
		NULL,
		true,
		0);
	if (SPI_processed != 1)
		ereport(ERROR, (errmsg("enqueue failed")));

	d = SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
	SPI_finish();
	PG_RETURN_INT64(DatumGetInt64(d));
}

/*
 * neurondb_llm_gpu_available
 *	  Check if GPU is available for LLM operations
 */
PG_FUNCTION_INFO_V1(neurondb_llm_gpu_available);

Datum
neurondb_llm_gpu_available(PG_FUNCTION_ARGS)
{
	bool gpu_available = false;

	if (neurondb_gpu_is_available())
	{
		const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
		if (backend != NULL && backend->hf_embed != NULL)
			gpu_available = true;
	}

	PG_RETURN_BOOL(gpu_available);
}

/*
 * neurondb_llm_gpu_info
 *	  Get GPU information for LLM operations
 */
PG_FUNCTION_INFO_V1(neurondb_llm_gpu_info);

Datum
neurondb_llm_gpu_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	TupleDesc tupdesc;
	AttInMetadata *attinmeta;
	Datum values[6];
	bool nulls[6];
	HeapTuple tuple;
	MemoryContext oldcontext;

	if (SRF_IS_FIRSTCALL())
	{
		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					errmsg("function returning record "
					       "called in context "
					       "that cannot accept type "
					       "record")));

		attinmeta = TupleDescGetAttInMetadata(tupdesc);
		funcctx->attinmeta = attinmeta;
		funcctx->user_fctx = NULL;
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	attinmeta = funcctx->attinmeta;

	if (funcctx->call_cntr < 1)
	{
		const ndb_gpu_backend *backend;
		GPUDeviceInfo *device_info;
		bool gpu_available;
		char *backend_name = NULL;
		int device_id = 0;
		char *device_name = NULL;
		int64 total_memory_mb = 0;
		int64 free_memory_mb = 0;
		bool is_available = false;

		gpu_available = neurondb_gpu_is_available();
		backend = ndb_gpu_get_active_backend();

		if (gpu_available && backend != NULL)
		{
			backend_name = backend->name ? pstrdup(backend->name)
						     : pstrdup("unknown");
			device_id = neurondb_gpu_device;
			device_info = neurondb_gpu_get_device_info(device_id);
			if (device_info != NULL)
			{
				device_name = pstrdup(device_info->name);
				total_memory_mb = device_info->total_memory_mb;
				free_memory_mb = device_info->free_memory_mb;
				is_available = device_info->is_available;
			} else
			{
				device_name = pstrdup("unknown");
				is_available = false;
			}
		} else
		{
			backend_name = pstrdup("none");
			device_name = pstrdup("unavailable");
			is_available = false;
		}

		values[0] = CStringGetTextDatum(backend_name);
		values[1] = Int32GetDatum(device_id);
		values[2] = CStringGetTextDatum(device_name);
		values[3] = Int64GetDatum(total_memory_mb);
		values[4] = Int64GetDatum(free_memory_mb);
		values[5] = BoolGetDatum(is_available);

		nulls[0] = false;
		nulls[1] = false;
		nulls[2] = false;
		nulls[3] = false;
		nulls[4] = false;
		nulls[5] = false;

		tuple = heap_form_tuple(tupdesc, values, nulls);
		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	} else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * ndb_llm_complete_batch
 *	  Batch LLM completion with GPU acceleration support
 */
PG_FUNCTION_INFO_V1(ndb_llm_complete_batch);

Datum
ndb_llm_complete_batch(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	TupleDesc tupdesc;
	AttInMetadata *attinmeta;
	Datum values[5];
	bool nulls[5];
	HeapTuple tuple;
	MemoryContext oldcontext;
	typedef struct
	{
		int num_prompts;
		int current_idx;
		NdbLLMBatchResp *batch_resp;
		char **prompts;
		TupleDesc tupdesc;
	} BatchCompleteState;

	if (SRF_IS_FIRSTCALL())
	{
		ArrayType *prompts_arr;
		text *params_in;
		char *params;
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;
		NdbLLMBatchResp batch_resp;
		int i;
		int num_prompts;
		char **prompts;
		int rc;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					errmsg("function returning record "
					       "called in context "
					       "that cannot accept type "
					       "record")));

		attinmeta = TupleDescGetAttInMetadata(tupdesc);
		funcctx->attinmeta = attinmeta;

		/* Get prompts array */
		prompts_arr = PG_GETARG_ARRAYTYPE_P(0);
		num_prompts = ArrayGetNItems(
			ARR_NDIM(prompts_arr), ARR_DIMS(prompts_arr));
		if (num_prompts <= 0)
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("prompts array must not be "
					       "empty")));

		/* Get params */
		params_in = (PG_NARGS() > 1 && !PG_ARGISNULL(1))
			? PG_GETARG_TEXT_PP(1)
			: NULL;
		params = params_in ? text_to_cstring(params_in) : "{}";

		/* Extract prompts from array using deconstruct_array */
		{
			Datum *prompt_datums;
			bool *prompt_nulls;
			int nprompt_elems;

			deconstruct_array(prompts_arr,
				TEXTOID,
				-1,
				false,
				'i',
				&prompt_datums,
				&prompt_nulls,
				&nprompt_elems);

			if (nprompt_elems != num_prompts)
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("prompts array size mismatch: expected %d, got %d",
							num_prompts,
							nprompt_elems)));

			prompts = (char **)palloc0(num_prompts * sizeof(char *));
			for (i = 0; i < num_prompts; i++)
			{
				if (prompt_nulls[i])
					continue;
				prompts[i] = text_to_cstring(
					DatumGetTextP(prompt_datums[i]));
			}
		}

		/* Fill config and call options */
		fill_cfg(&cfg);
		call_opts.task = "complete";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = cfg.require_gpu;
		call_opts.fail_open = neurondb_llm_fail_open;

		/* Call batch completion router */
		elog(NOTICE, "neurondb: llm_runtime calling ndb_llm_route_complete_batch with %d prompts", num_prompts);
		memset(&batch_resp, 0, sizeof(NdbLLMBatchResp));
		rc = ndb_llm_route_complete_batch(&cfg,
			&call_opts,
			(const char **)prompts,
			num_prompts,
			params,
			&batch_resp);
		elog(NOTICE, "neurondb: llm_runtime got rc=%d from ndb_llm_route_complete_batch", rc);
		if (rc != 0)
		{
			/* Free prompts */
			for (i = 0; i < num_prompts; i++)
				if (prompts[i])
					pfree(prompts[i]);
			pfree(prompts);
			if (params)
				pfree(params);
			ereport(ERROR,
				(errcode(ERRCODE_EXTERNAL_ROUTINE_INVOCATION_EXCEPTION),
					errmsg("neurondb: batch completion "
					       "failed")));
		}

		/* Store state */
		{
			BatchCompleteState *state =
				(BatchCompleteState *)palloc0(
					sizeof(BatchCompleteState));
			state->num_prompts = num_prompts;
			state->current_idx = 0;
			state->batch_resp = (NdbLLMBatchResp *)palloc0(
				sizeof(NdbLLMBatchResp));
			/* Deep copy batch_resp structure */
			state->batch_resp->num_items = batch_resp.num_items;
			state->batch_resp->num_success = batch_resp.num_success;
			if (batch_resp.texts && batch_resp.num_items > 0)
			{
				int i;
				state->batch_resp->texts = (char **)palloc0(
					batch_resp.num_items * sizeof(char *));
				for (i = 0; i < batch_resp.num_items; i++)
				{
					if (batch_resp.texts[i])
						state->batch_resp->texts[i] =
							pstrdup(batch_resp.texts[i]);
				}
			}
			if (batch_resp.tokens_in && batch_resp.num_items > 0)
			{
				state->batch_resp->tokens_in = (int *)palloc(
					batch_resp.num_items * sizeof(int));
				memcpy(state->batch_resp->tokens_in,
					batch_resp.tokens_in,
					batch_resp.num_items * sizeof(int));
			}
			if (batch_resp.tokens_out && batch_resp.num_items > 0)
			{
				state->batch_resp->tokens_out = (int *)palloc(
					batch_resp.num_items * sizeof(int));
				memcpy(state->batch_resp->tokens_out,
					batch_resp.tokens_out,
					batch_resp.num_items * sizeof(int));
			}
			if (batch_resp.http_status && batch_resp.num_items > 0)
			{
				state->batch_resp->http_status = (int *)palloc(
					batch_resp.num_items * sizeof(int));
				memcpy(state->batch_resp->http_status,
					batch_resp.http_status,
					batch_resp.num_items * sizeof(int));
			}
			state->prompts = prompts;
			state->tupdesc = tupdesc;
			funcctx->user_fctx = state;
		}

		if (params)
			pfree(params);
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	attinmeta = funcctx->attinmeta;
	{
		BatchCompleteState *state =
			(BatchCompleteState *)funcctx->user_fctx;
		TupleDesc tupdesc;

		elog(NOTICE, "neurondb: llm_runtime SRF_PERCALL_SETUP: state=%p", (void *)state);
		if (!state || !state->batch_resp || !state->tupdesc)
		{
			elog(ERROR, "neurondb: batch completion state is invalid");
			SRF_RETURN_DONE(funcctx);
		}
		tupdesc = state->tupdesc;

		elog(NOTICE, "neurondb: llm_runtime SRF: current_idx=%d, num_items=%d", state->current_idx, state->batch_resp->num_items);
		if (state->current_idx < state->batch_resp->num_items)
		{
			int idx = state->current_idx;
			NdbLLMBatchResp *resp = state->batch_resp;

			if (idx < 0 || idx >= resp->num_items)
			{
				elog(ERROR, "neurondb: batch completion index %d out of bounds (0-%d)", idx, resp->num_items - 1);
				SRF_RETURN_DONE(funcctx);
			}

			elog(NOTICE, "neurondb: llm_runtime SRF processing idx=%d", idx);
			values[0] = Int32GetDatum(idx);
			if (resp->texts && idx < resp->num_items && resp->texts[idx])
			{
				elog(NOTICE, "neurondb: llm_runtime SRF text[%d]=%p, len=%zu", idx, (void *)resp->texts[idx], resp->texts[idx] ? strlen(resp->texts[idx]) : 0);
				values[1] =
					CStringGetTextDatum(resp->texts[idx]);
				elog(NOTICE, "neurondb: llm_runtime SRF accessing tokens_in[%d]", idx);
				values[2] = Int32GetDatum(resp->tokens_in
						&& idx < resp->num_items
						? resp->tokens_in[idx]
						: 0);
				elog(NOTICE, "neurondb: llm_runtime SRF accessing tokens_out[%d]", idx);
				values[3] = Int32GetDatum(resp->tokens_out
						&& idx < resp->num_items
						? resp->tokens_out[idx]
						: 0);
				elog(NOTICE, "neurondb: llm_runtime SRF accessing http_status[%d]", idx);
				values[4] = Int32GetDatum(resp->http_status
						&& idx < resp->num_items
						? resp->http_status[idx]
						: 200);
				elog(NOTICE, "neurondb: llm_runtime SRF creating tuple for idx=%d", idx);
				nulls[0] = false;
				nulls[1] = false;
				nulls[2] = false;
				nulls[3] = false;
				nulls[4] = false;
			} else
			{
				values[1] = (Datum)0;
				values[2] = Int32GetDatum(0);
				values[3] = Int32GetDatum(0);
				values[4] = Int32GetDatum(500);
				nulls[0] = false;
				nulls[1] = true;
				nulls[2] = false;
				nulls[3] = false;
				nulls[4] = false;
			}

			tuple = heap_form_tuple(tupdesc, values, nulls);
			elog(NOTICE, "neurondb: llm_runtime SRF tuple created for idx=%d, returning next", idx);
			state->current_idx++;
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		} else
		{
			/* Clean up */
			BatchCompleteState *state =
				(BatchCompleteState *)funcctx->user_fctx;
			int i;

			if (state->batch_resp)
			{
				if (state->batch_resp->texts)
				{
					for (i = 0; i
						< state->batch_resp->num_items;
						i++)
						if (state->batch_resp->texts[i])
							pfree(state->batch_resp->texts
									[i]);
					pfree(state->batch_resp->texts);
				}
				if (state->batch_resp->tokens_in)
					pfree(state->batch_resp->tokens_in);
				if (state->batch_resp->tokens_out)
					pfree(state->batch_resp->tokens_out);
				if (state->batch_resp->http_status)
					pfree(state->batch_resp->http_status);
				pfree(state->batch_resp);
			}
			if (state->prompts)
			{
				for (i = 0; i < state->num_prompts; i++)
					if (state->prompts[i])
						pfree(state->prompts[i]);
				pfree(state->prompts);
			}
			pfree(state);
			SRF_RETURN_DONE(funcctx);
		}
	}
}

/*
 * ndb_llm_rerank_batch
 *	  Batch LLM reranking with GPU acceleration support
 */
PG_FUNCTION_INFO_V1(ndb_llm_rerank_batch);

Datum
ndb_llm_rerank_batch(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	TupleDesc tupdesc;
	AttInMetadata *attinmeta;
	Datum values[3];
	bool nulls[3];
	HeapTuple tuple;
	MemoryContext oldcontext;
	typedef struct
	{
		int num_queries;
		int current_query_idx;
		int current_doc_idx;
		float **scores;
		int *nscores;
		int *ndocs_array;
	} BatchRerankState;

	if (SRF_IS_FIRSTCALL())
	{
		ArrayType *queries_arr;
		ArrayType *documents_arr;
		text *model_in;
		char *model;
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;
		int num_queries;
		char **queries;
		const char ***docs_array;
		int *ndocs_array;
		float **scores;
		int *nscores;
		int i;
		int j;
		int rc;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					errmsg("function returning record "
					       "called in context "
					       "that cannot accept type "
					       "record")));

		attinmeta = TupleDescGetAttInMetadata(tupdesc);
		funcctx->attinmeta = attinmeta;

		/* Get queries array */
		queries_arr = PG_GETARG_ARRAYTYPE_P(0);
		num_queries = ArrayGetNItems(
			ARR_NDIM(queries_arr), ARR_DIMS(queries_arr));
		if (num_queries <= 0)
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("queries array must not be "
					       "empty")));

		/* Get documents array (2D array or 1D array of arrays) */
		documents_arr = PG_GETARG_ARRAYTYPE_P(1);
		if (ARR_NDIM(documents_arr) == 0)
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("documents_array cannot be empty")));
		if (ARR_NDIM(documents_arr) > 2)
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("documents_array must be a 1D or 2D "
					       "array")));

		/* Get model */
		model_in = (PG_NARGS() > 2 && !PG_ARGISNULL(2))
			? PG_GETARG_TEXT_PP(2)
			: NULL;
		model = model_in ? text_to_cstring(model_in) : NULL;

		/* Extract queries from array using deconstruct_array */
		{
			Datum *query_datums;
			bool *query_nulls;
			int nquery_elems;

			deconstruct_array(queries_arr,
				TEXTOID,
				-1,
				false,
				'i',
				&query_datums,
				&query_nulls,
				&nquery_elems);

			if (nquery_elems != num_queries)
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("queries array size mismatch: expected %d, got %d",
							num_queries,
							nquery_elems)));

			queries = (char **)palloc0(num_queries * sizeof(char *));
			for (i = 0; i < num_queries; i++)
			{
				if (query_nulls[i])
					continue;
				queries[i] = text_to_cstring(
					DatumGetTextP(query_datums[i]));
			}
		}

		/* Extract documents from 2D array */
		ndocs_array = (int *)palloc0(num_queries * sizeof(int));
		docs_array =
			(const char ***)palloc0(num_queries * sizeof(char **));

		/* Extract documents from array (handles both 1D array of arrays and 2D array) */
		{
			Oid outer_elem_type;
			Oid inner_elem_type;
			int16 outer_elem_len;
			int16 inner_elem_len;
			bool outer_elem_byval;
			bool inner_elem_byval;
			char outer_elem_align;
			char inner_elem_align;
			int ndocs;
			char **docs;
			int indices[1];
			int indices_2d[2];

			if (ARR_NDIM(documents_arr) == 1)
			{
				/* Handle 1D array of arrays */
				int nrows = ARR_DIMS(documents_arr)[0];

				if (nrows != num_queries)
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("documents array size mismatch: expected %d rows, got %d",
								num_queries,
								nrows)));

				/* Get element type of outer array for array_ref */
				outer_elem_type = ARR_ELEMTYPE(documents_arr);
				get_typlenbyvalalign(outer_elem_type,
					&outer_elem_len,
					&outer_elem_byval,
					&outer_elem_align);

				for (i = 0; i < num_queries; i++)
				{
					bool isNull;
					Datum row_datum;
					ArrayType *doc_row;
					Datum *doc_datums;
					bool *doc_nulls;
					int ndoc_elems;

					indices[0] = i;

					/* Extract row i as a 1D array */
					row_datum = array_ref(documents_arr,
						1,
						indices,
						outer_elem_len,
						outer_elem_len,
						outer_elem_byval,
						outer_elem_align,
						&isNull);

					if (isNull)
						ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								errmsg("documents array row %d is NULL",
									i)));

					doc_row = DatumGetArrayTypeP(row_datum);

					/* Deconstruct the row array */
					deconstruct_array(doc_row,
						TEXTOID,
						-1,
						false,
						'i',
						&doc_datums,
						&doc_nulls,
						&ndoc_elems);

					ndocs_array[i] = ndoc_elems;
					docs = (char **)palloc0(ndoc_elems * sizeof(char *));
					for (j = 0; j < ndoc_elems; j++)
					{
						if (doc_nulls[j])
							continue;
						docs[j] = text_to_cstring(
							DatumGetTextP(doc_datums[j]));
					}
					docs_array[i] = (const char **)docs;
				}
			} else if (ARR_NDIM(documents_arr) == 2)
			{
				/* Handle true 2D array */
				/* For 2D arrays, ARR_ELEMTYPE returns the element type (text) */
				inner_elem_type = ARR_ELEMTYPE(documents_arr);
				
				/* Check if it's text or if we need to get element type of array type */
				if (inner_elem_type == TEXTOID)
				{
					/* Direct 2D text array */
					get_typlenbyvalalign(inner_elem_type,
						&inner_elem_len,
						&inner_elem_byval,
						&inner_elem_align);
				} else
				{
					/* Might be array of arrays - try to get inner element type */
					inner_elem_type = get_element_type(inner_elem_type);
					if (!OidIsValid(inner_elem_type) || inner_elem_type != TEXTOID)
					{
						/* Fall back to treating as 1D array of arrays */
						goto handle_as_1d;
					}
					get_typlenbyvalalign(inner_elem_type,
						&inner_elem_len,
						&inner_elem_byval,
						&inner_elem_align);
				}

				ndocs = ARR_DIMS(documents_arr)[1];

				for (i = 0; i < num_queries; i++)
				{
					ndocs_array[i] = ndocs;
					docs = (char **)palloc0(ndocs * sizeof(char *));
					for (j = 0; j < ndocs; j++)
					{
						bool isNull;
						Datum elem;

						indices_2d[0] = i;
						indices_2d[1] = j;

						/* Extract element [i][j] from 2D array */
						elem = array_ref(documents_arr,
							2,
							indices_2d,
							inner_elem_len,
							inner_elem_len,
							inner_elem_byval,
							inner_elem_align,
							&isNull);

						if (!isNull && DatumGetPointer(elem))
						{
							docs[j] = text_to_cstring(
								DatumGetTextP(elem));
						}
					}
					docs_array[i] = (const char **)docs;
				}
			} else
			{
				/* Unexpected dimension - treat as 1D array of arrays */
handle_as_1d:
				/* Handle as 1D array of arrays */
				int nrows = ARR_DIMS(documents_arr)[0];

				if (nrows != num_queries)
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("documents array size mismatch: expected %d rows, got %d",
								num_queries,
								nrows)));

				/* Get element type of outer array for array_ref */
				outer_elem_type = ARR_ELEMTYPE(documents_arr);
				get_typlenbyvalalign(outer_elem_type,
					&outer_elem_len,
					&outer_elem_byval,
					&outer_elem_align);

				for (i = 0; i < num_queries; i++)
				{
					bool isNull;
					Datum row_datum;
					ArrayType *doc_row;
					Datum *doc_datums;
					bool *doc_nulls;
					int ndoc_elems;

					indices[0] = i;

					/* Extract row i as a 1D array */
					row_datum = array_ref(documents_arr,
						1,
						indices,
						outer_elem_len,
						outer_elem_len,
						outer_elem_byval,
						outer_elem_align,
						&isNull);

					if (isNull)
						ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								errmsg("documents array row %d is NULL",
									i)));

					doc_row = DatumGetArrayTypeP(row_datum);

					/* Deconstruct the row array */
					deconstruct_array(doc_row,
						TEXTOID,
						-1,
						false,
						'i',
						&doc_datums,
						&doc_nulls,
						&ndoc_elems);

					ndocs_array[i] = ndoc_elems;
					docs = (char **)palloc0(ndoc_elems * sizeof(char *));
					for (j = 0; j < ndoc_elems; j++)
					{
						if (doc_nulls[j])
							continue;
						docs[j] = text_to_cstring(
							DatumGetTextP(doc_datums[j]));
					}
					docs_array[i] = (const char **)docs;
				}
			}
		}

		/* Fill config and call options */
		fill_cfg(&cfg);
		if (model)
			cfg.model = model;
		call_opts.task = "rerank";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = cfg.require_gpu;
		call_opts.fail_open = neurondb_llm_fail_open;

		/* Call batch reranking router */
		scores = NULL;
		nscores = NULL;
		rc = ndb_llm_route_rerank_batch(&cfg,
			&call_opts,
			(const char **)queries,
			docs_array,
			ndocs_array,
			num_queries,
			&scores,
			&nscores);
		if (rc != 0)
		{
			/* Free queries */
			for (i = 0; i < num_queries; i++)
				if (queries[i])
					pfree(queries[i]);
			pfree(queries);
			/* Free documents */
			for (i = 0; i < num_queries; i++)
			{
				if (docs_array[i])
				{
					for (j = 0; j < ndocs_array[i]; j++)
						if (docs_array[i][j])
							pfree((char *)docs_array
									[i][j]);
					pfree((char **)docs_array[i]);
				}
			}
			pfree(docs_array);
			pfree(ndocs_array);
			if (model)
				pfree(model);
			ereport(ERROR,
				(errcode(ERRCODE_EXTERNAL_ROUTINE_INVOCATION_EXCEPTION),
					errmsg("neurondb: batch reranking "
					       "failed")));
		}

		/* Store state */
		{
			BatchRerankState *state = (BatchRerankState *)palloc0(
				sizeof(BatchRerankState));
			state->num_queries = num_queries;
			state->current_query_idx = 0;
			state->current_doc_idx = 0;
			state->scores = scores;
			state->nscores = nscores;
			state->ndocs_array = ndocs_array;
			funcctx->user_fctx = state;
		}

		/* Free queries and documents arrays (scores are stored in state) */
		for (i = 0; i < num_queries; i++)
			if (queries[i])
				pfree(queries[i]);
		pfree(queries);
		for (i = 0; i < num_queries; i++)
		{
			if (docs_array[i])
			{
				for (j = 0; j < ndocs_array[i]; j++)
					if (docs_array[i][j])
						pfree((char *)docs_array[i][j]);
				pfree((char **)docs_array[i]);
			}
		}
		pfree(docs_array);
		if (model)
			pfree(model);
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	attinmeta = funcctx->attinmeta;
	{
		BatchRerankState *state =
			(BatchRerankState *)funcctx->user_fctx;

		/* Find next query-document pair */
		while (state->current_query_idx < state->num_queries)
		{
			int query_idx = state->current_query_idx;
			int doc_idx = state->current_doc_idx;

			if (doc_idx < state->ndocs_array[query_idx])
			{
				if (state->scores && state->scores[query_idx])
				{
					values[0] = Int32GetDatum(query_idx);
					values[1] = Int32GetDatum(doc_idx);
					values[2] = Float4GetDatum(
						state->scores[query_idx]
							     [doc_idx]);
					nulls[0] = false;
					nulls[1] = false;
					nulls[2] = false;
				} else
				{
					values[0] = Int32GetDatum(query_idx);
					values[1] = Int32GetDatum(doc_idx);
					values[2] = Float4GetDatum(0.0f);
					nulls[0] = false;
					nulls[1] = false;
					nulls[2] = false;
				}

				tuple = heap_form_tuple(tupdesc, values, nulls);
				state->current_doc_idx++;
				SRF_RETURN_NEXT(
					funcctx, HeapTupleGetDatum(tuple));
			} else
			{
				state->current_query_idx++;
				state->current_doc_idx = 0;
			}
		}

		/* Clean up */
		{
			int i;
			if (state->scores)
			{
				for (i = 0; i < state->num_queries; i++)
					if (state->scores[i])
						pfree(state->scores[i]);
				pfree(state->scores);
			}
			if (state->nscores)
				pfree(state->nscores);
			if (state->ndocs_array)
				pfree(state->ndocs_array);
			pfree(state);
		}
		SRF_RETURN_DONE(funcctx);
	}
}
