/*-------------------------------------------------------------------------
 * neurondb_llm.h
 *   Hugging Face / LLM provider integration for NeurondB
 *-------------------------------------------------------------------------*/

#ifndef NEURONDB_LLM_H
#define NEURONDB_LLM_H

#include "postgres.h"
#include "fmgr.h"

/* GUCs */
extern char *neurondb_llm_provider;      /* "huggingface" or "local" */
extern char *neurondb_llm_model;         /* default model id */
extern char *neurondb_llm_endpoint;      /* base URL */
extern char *neurondb_llm_api_key;       /* token (via ALTER SYSTEM/env) */
extern int   neurondb_llm_timeout_ms;    /* HTTP timeout */
extern int   neurondb_llm_cache_ttl;     /* cache TTL seconds */
extern int   neurondb_llm_rate_limiter_qps; /* QPS */
extern bool  neurondb_llm_fail_open;     /* fail-open policy */

typedef struct NdbLLMConfig
{
    const char *provider;
    const char *endpoint;
    const char *model;
    const char *api_key;
    int timeout_ms;
} NdbLLMConfig;

typedef struct NdbLLMResp
{
    char *text;        /* generation */
    char *json;        /* raw */
    int tokens_in;
    int tokens_out;
    int http_status;
} NdbLLMResp;

/* Provider calls */
int ndb_hf_complete(const NdbLLMConfig *cfg,
                    const char *prompt,
                    const char *params_json,
                    NdbLLMResp *out);

int ndb_hf_embed(const NdbLLMConfig *cfg,
                 const char *text,
                 float **vec_out,
                 int *dim_out);

int ndb_hf_rerank(const NdbLLMConfig *cfg,
                  const char *query,
                  const char **docs,
                  int ndocs,
                  float **scores_out);

/* SQL-callable */
extern Datum ndb_llm_complete(PG_FUNCTION_ARGS);
extern Datum ndb_llm_embed(PG_FUNCTION_ARGS);
extern Datum ndb_llm_rerank(PG_FUNCTION_ARGS);
extern Datum ndb_llm_enqueue(PG_FUNCTION_ARGS);

/* GUC init and shared memory */
void neurondb_llm_init_guc(void);
Size neurondb_llm_shmem_size(void);
void neurondb_llm_shmem_init(void);

/* Cache functions (from llm_cache.c) */
bool ndb_llm_cache_lookup(const char *key, int max_age_seconds, char **out_text);
void ndb_llm_cache_store(const char *key, const char *text);

/* Job functions (from llm_jobs.c) */
int  ndb_llm_job_enqueue(const char *job_type, const char *payload);
bool ndb_llm_job_acquire(int *job_id, char **job_type, char **payload);
bool ndb_llm_job_update(int job_id, const char *status, const char *result, const char *error);
int  ndb_llm_job_prune(int max_age_days);
int  ndb_llm_job_count_status(const char *status);
int  ndb_llm_job_retry_failed(int max_retries);
void ndb_llm_job_clear(void);

#endif /* NEURONDB_LLM_H */


