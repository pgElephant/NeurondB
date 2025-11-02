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
#include "openssl/sha.h"
#include "neurondb.h"
#include "neurondb_llm.h"

/* GUCs */
char *neurondb_llm_provider = NULL;
char *neurondb_llm_model = NULL;
char *neurondb_llm_endpoint = NULL;
char *neurondb_llm_api_key = NULL;
int   neurondb_llm_timeout_ms = 30000;
int   neurondb_llm_cache_ttl = 600;
int   neurondb_llm_rate_limiter_qps = 5;
bool  neurondb_llm_fail_open = true;

/* ------------------------
 *  Shared rate limiter (token bucket)
 * ------------------------
 */
typedef struct NdbLLMRateLimiter
{
    double      tokens;
    TimestampTz last_refill;
    int         capacity;   /* tokens per second */
} NdbLLMRateLimiter;

static NdbLLMRateLimiter *llm_rl = NULL;
static LWLock *llm_rl_lock = NULL;

Size
neurondb_llm_shmem_size(void)
{
    return MAXALIGN(sizeof(NdbLLMRateLimiter));
}

void
neurondb_llm_shmem_init(void)
{
    bool found;
    llm_rl = (NdbLLMRateLimiter *) ShmemInitStruct("neurondb_llm_rate", sizeof(NdbLLMRateLimiter), &found);
    llm_rl_lock = &(GetNamedLWLockTranche("neurondb_llm")[0].lock);
    if (!found)
    {
        llm_rl->tokens = 0.0;
        llm_rl->last_refill = GetCurrentTimestamp();
        llm_rl->capacity = neurondb_llm_rate_limiter_qps > 0 ? neurondb_llm_rate_limiter_qps : 5;
    }
}

static bool
llm_acquire_token(void)
{
    TimestampTz now = GetCurrentTimestamp();
    long secs; int usecs; double elapsed;
    int cap;
    if (llm_rl == NULL || llm_rl_lock == NULL)
        return true; /* no limiter initialized */

    LWLockAcquire(llm_rl_lock, LW_EXCLUSIVE);
    cap = neurondb_llm_rate_limiter_qps > 0 ? neurondb_llm_rate_limiter_qps : llm_rl->capacity;
    TimestampDifference(llm_rl->last_refill, now, &secs, &usecs);
    elapsed = (double) secs + ((double) usecs / 1000000.0);
    if (elapsed > 0.0)
    {
        llm_rl->tokens = Min((double) cap, llm_rl->tokens + elapsed * cap);
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

void
neurondb_llm_init_guc(void)
{
    DefineCustomStringVariable("neurondb.llm_provider",
                               "LLM provider",
                               NULL,
                               &neurondb_llm_provider,
                               "huggingface",
                               PGC_USERSET, 0, NULL, NULL, NULL);

    DefineCustomStringVariable("neurondb.llm_model",
                               "Default LLM model id",
                               NULL,
                               &neurondb_llm_model,
                               "sentence-transformers/all-MiniLM-L6-v2",
                               PGC_USERSET, 0, NULL, NULL, NULL);

    DefineCustomStringVariable("neurondb.llm_endpoint",
                               "LLM endpoint base URL",
                               NULL,
                               &neurondb_llm_endpoint,
                               "https://api-inference.huggingface.co",
                               PGC_USERSET, 0, NULL, NULL, NULL);

    DefineCustomStringVariable("neurondb.llm_api_key",
                               "LLM API key (set via ALTER SYSTEM or env)",
                               NULL,
                               &neurondb_llm_api_key,
                               "",
                               PGC_SUSET, GUC_SUPERUSER_ONLY, NULL, NULL, NULL);

    DefineCustomIntVariable("neurondb.llm_timeout_ms",
                            "HTTP timeout (ms)",
                            NULL,
                            &neurondb_llm_timeout_ms,
                            30000, 1000, 600000,
                            PGC_USERSET, 0, NULL, NULL, NULL);

    DefineCustomIntVariable("neurondb.llm_cache_ttl",
                            "Cache TTL seconds",
                            NULL,
                            &neurondb_llm_cache_ttl,
                            600, 0, 86400,
                            PGC_USERSET, 0, NULL, NULL, NULL);

    DefineCustomIntVariable("neurondb.llm_rate_limiter_qps",
                            "Rate limiter QPS",
                            NULL,
                            &neurondb_llm_rate_limiter_qps,
                            5, 1, 10000,
                            PGC_USERSET, 0, NULL, NULL, NULL);

    DefineCustomBoolVariable("neurondb.llm_fail_open",
                             "Fail open on provider errors",
                             NULL,
                             &neurondb_llm_fail_open,
                             true, PGC_USERSET, 0, NULL, NULL, NULL);
}

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
    appendStringInfo(&src, "%s|%s|%s|%s",
                     provider ? provider : "",
                     model ? model : "",
                     endpoint ? endpoint : "",
                     payload ? payload : "");
    SHA256((unsigned char *) src.data, src.len, hash);

    initStringInfo(dst);
    for (i = 0; i < SHA256_DIGEST_LENGTH; i++)
        appendStringInfo(&dst[0], "%02x", hash[i]);
}

static bool
cache_lookup_text(const char *key, char **text_out)
{
    bool hit = false;
    if (SPI_connect() != SPI_OK_CONNECT)
        return false;

    if (SPI_execute_with_args(
            "SELECT value->>'text' FROM neurondb_llm_cache WHERE key = $1 AND now() - created_at < make_interval(secs => $2)",
            2,
            (Oid[]){TEXTOID, INT4OID},
            (Datum[]){CStringGetTextDatum(key), Int32GetDatum(neurondb_llm_cache_ttl)},
            (char[]){'i','i'}, true, 0) == SPI_OK_SELECT && SPI_processed == 1)
    {
        bool isnull;
        Datum d = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
        if (!isnull)
        {
            *text_out = TextDatumGetCString(d);
            hit = true;
        }
    }
    SPI_finish();
    return hit;
}

static void
cache_store_text(const char *key, const char *text)
{
    if (SPI_connect() != SPI_OK_CONNECT)
        return;
    Oid argtypes[2] = {TEXTOID, JSONBOID};
    Datum values[2];
    StringInfoData val;
    initStringInfo(&val);
    appendStringInfo(&val, "{\"text\":%s}", quote_literal_cstr(text));
    values[0] = CStringGetTextDatum(key);
    values[1] = CStringGetTextDatum(val.data);
    SPI_execute_with_args("INSERT INTO neurondb_llm_cache(key,value,created_at) VALUES($1,$2::jsonb,now()) ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, created_at=now()",
                          2, argtypes, values, NULL, false, 0);
    SPI_finish();
}

static void
fill_cfg(NdbLLMConfig *cfg)
{
    cfg->provider = neurondb_llm_provider ? neurondb_llm_provider : "huggingface";
    cfg->endpoint = neurondb_llm_endpoint ? neurondb_llm_endpoint : "https://api-inference.huggingface.co";
    cfg->model = neurondb_llm_model ? neurondb_llm_model : "sentence-transformers/all-MiniLM-L6-v2";
    cfg->api_key = neurondb_llm_api_key;
    cfg->timeout_ms = neurondb_llm_timeout_ms;
}

/* SQL: llm_complete(prompt text, params jsonb) RETURNS text */
PG_FUNCTION_INFO_V1(ndb_llm_complete);
Datum
ndb_llm_complete(PG_FUNCTION_ARGS)
{
    text *prompt_in = PG_GETARG_TEXT_PP(0);
    text *params_in = (PG_NARGS() > 1 && !PG_ARGISNULL(1)) ? PG_GETARG_TEXT_PP(1) : NULL;
    char *prompt = text_to_cstring(prompt_in);
    char *params = params_in ? text_to_cstring(params_in) : "{}";
    NdbLLMConfig cfg; NdbLLMResp resp = {0};
    StringInfoData keysrc, keyhex;
    char *cached = NULL;
    int rc;

    fill_cfg(&cfg);
    initStringInfo(&keysrc);
    appendStringInfo(&keysrc, "%s|%s|%s", prompt, params, cfg.model);
    compute_cache_key(&keyhex, cfg.provider, cfg.model, cfg.endpoint, keysrc.data);

    if (cache_lookup_text(keyhex.data, &cached))
        PG_RETURN_TEXT_P(cstring_to_text(cached));

    if (cfg.endpoint && strncmp(cfg.endpoint, "mock://", 7) == 0)
    {
        PG_RETURN_TEXT_P(cstring_to_text("mock-completion"));
    }

    if (!llm_acquire_token())
    {
        if (neurondb_llm_fail_open)
        {
            ereport(WARNING, (errmsg("neurondb: LLM rate limited - fail open")));
            PG_RETURN_NULL();
        }
        ereport(ERROR, (errmsg("neurondb: LLM rate limited")));
    }

    rc = ndb_hf_complete(&cfg, prompt, params, &resp);
    if (rc != 0)
    {
        if (neurondb_llm_fail_open)
            PG_RETURN_NULL();
        ereport(ERROR, (errmsg("neurondb: llm provider error")));
    }
    if (resp.text)
        cache_store_text(keyhex.data, resp.text);
    PG_RETURN_TEXT_P(cstring_to_text(resp.text ? resp.text : ""));
}

/* SQL: llm_embed(txt text, model text default null) RETURNS vector */
PG_FUNCTION_INFO_V1(ndb_llm_embed);
Datum
ndb_llm_embed(PG_FUNCTION_ARGS)
{
    text *txt_in = PG_GETARG_TEXT_PP(0);
    char *txt = text_to_cstring(txt_in);
    NdbLLMConfig cfg; float *vec = NULL; int dim = 0; Vector *v;
    fill_cfg(&cfg);
    if (PG_NARGS() > 1 && !PG_ARGISNULL(1)) cfg.model = text_to_cstring(PG_GETARG_TEXT_PP(1));

    if (cfg.endpoint && strncmp(cfg.endpoint, "mock://", 7) == 0)
    {
        dim = 4;
        v = new_vector(dim);
        v->data[0] = 1; v->data[1] = 2; v->data[2] = 3; v->data[3] = 4;
        PG_RETURN_VECTOR_P(v);
    }
    if (!llm_acquire_token())
    {
        if (neurondb_llm_fail_open)
        {
            ereport(WARNING, (errmsg("neurondb: LLM rate limited - fail open")));
            PG_RETURN_NULL();
        }
        ereport(ERROR, (errmsg("neurondb: LLM rate limited")));
    }
    if (ndb_hf_embed(&cfg, txt, &vec, &dim) != 0 || dim <= 0)
    {
        if (neurondb_llm_fail_open)
            PG_RETURN_NULL();
        ereport(ERROR, (errmsg("neurondb: embed failed")));
    }
    v = new_vector(dim);
    memcpy(v->data, vec, sizeof(float) * dim);
    PG_RETURN_VECTOR_P(v);
}

/* SQL: llm_rerank(query text, docs text[], model text default null, top_n int default 10)
 * RETURNS TABLE(idx int, score real)
 */
PG_FUNCTION_INFO_V1(ndb_llm_rerank);
Datum
ndb_llm_rerank(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    typedef struct { int ndocs; int i; } Ctx;
    Ctx *c;
    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext old;
        Ctx *c;
        ArrayType *arr;
        TupleDesc tupdesc;

        funcctx = SRF_FIRSTCALL_INIT();
        old = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        c = palloc0(sizeof(Ctx));
        arr = PG_GETARG_ARRAYTYPE_P(1);
        c->ndocs = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
        c->i = 0;
        funcctx->user_fctx = c;
        tupdesc = CreateTemplateTupleDesc(2);
        TupleDescInitEntry(tupdesc, (AttrNumber) 1, "idx", INT4OID, -1, 0);
        TupleDescInitEntry(tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        MemoryContextSwitchTo(old);
    }
    funcctx = SRF_PERCALL_SETUP();
    c = (Ctx *) funcctx->user_fctx;
    if (c->i < c->ndocs)
    {
        Datum values[2]; bool nulls[2] = {false,false}; HeapTuple tup;
        values[0] = Int32GetDatum(c->i + 1);
        values[1] = Float4GetDatum(1.0f - ((float) c->i / (float) Max(1,c->ndocs)));
        tup = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        c->i++;
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tup));
    }
    SRF_RETURN_DONE(funcctx);
}

PG_FUNCTION_INFO_V1(ndb_llm_enqueue);
Datum
ndb_llm_enqueue(PG_FUNCTION_ARGS)
{
    Oid argtypes[2] = {TEXTOID, JSONBOID};
    Datum values[2];
    bool isnull;
    Datum d;
    text *action = PG_GETARG_TEXT_PP(0);
    text *payload = PG_GETARG_TEXT_PP(1);
    if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR, (errmsg("SPI_connect failed")));
    values[0] = PointerGetDatum(action);
    values[1] = PointerGetDatum(payload);
    SPI_execute_with_args("INSERT INTO neurondb_llm_jobs(action,payload,status,created_at) VALUES($1,$2::jsonb,'queued',now()) RETURNING id",
                          2, argtypes, values, NULL, true, 0);
    if (SPI_processed != 1)
        ereport(ERROR, (errmsg("enqueue failed")));
    d = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
    SPI_finish();
    PG_RETURN_INT64(DatumGetInt64(d));
}


