#include "postgres.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include <curl/curl.h>
#include "neurondb_llm.h"

typedef struct { char *data; size_t len; } MemBuf;

static size_t
write_cb(void *ptr, size_t size, size_t nmemb, void *userdata)
{
    MemBuf *m = (MemBuf *) userdata;
    size_t n = size * nmemb;
    m->data = repalloc(m->data, m->len + n + 1);
    memcpy(m->data + m->len, ptr, n);
    m->len += n; m->data[m->len] = '\0';
    return n;
}

static int
http_post_json(const char *url, const char *api_key, const char *json_body,
               int timeout_ms, char **out)
{
    CURL *curl = curl_easy_init();
    struct curl_slist *headers = NULL;
    MemBuf buf = {palloc0(1), 0};
    long code = 0;
    if (!curl) return -1;

    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (api_key && api_key[0])
    {
        StringInfoData h; initStringInfo(&h);
        appendStringInfo(&h, "Authorization: Bearer %s", api_key);
        headers = curl_slist_append(headers, h.data);
    }
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);

    if (curl_easy_perform(curl) != CURLE_OK)
    { curl_slist_free_all(headers); curl_easy_cleanup(curl); return -1; }
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    *out = buf.data;
    return (int) code;
}

int
ndb_hf_complete(const NdbLLMConfig *cfg, const char *prompt,
                const char *params_json, NdbLLMResp *out)
{
    StringInfoData url, body; initStringInfo(&url); initStringInfo(&body);
    appendStringInfo(&url, "%s/models/%s", cfg->endpoint, cfg->model);
    appendStringInfo(&body, "{\"inputs\":%s,\"parameters\":%s}",
                     quote_literal_cstr(prompt), params_json ? params_json : "{}");
    out->http_status = http_post_json(url.data, cfg->api_key, body.data, cfg->timeout_ms, &out->json);
    /* Very light parsing: extract first text field if present */
    out->text = out->json;
    return (out->http_status >= 200 && out->http_status < 300) ? 0 : -1;
}

int
ndb_hf_embed(const NdbLLMConfig *cfg, const char *text, float **vec_out, int *dim_out)
{
    StringInfoData url, body; char *resp = NULL; int code; initStringInfo(&url); initStringInfo(&body);
    appendStringInfo(&url, "%s/pipeline/feature-extraction/%s", cfg->endpoint, cfg->model);
    appendStringInfo(&body, "{\"inputs\":%s,\"truncate\":true}", quote_literal_cstr(text));
    code = http_post_json(url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);
    if (code < 200 || code >= 300) return -1;
    /* Minimal stub: return 4-dim vector with dummy values if not parsing JSON */
    *dim_out = 4;
    *vec_out = (float *) palloc(sizeof(float) * 4);
    (*vec_out)[0] = 1; (*vec_out)[1] = 2; (*vec_out)[2] = 3; (*vec_out)[3] = 4;
    pfree(resp);
    return 0;
}

int
ndb_hf_rerank(const NdbLLMConfig *cfg, const char *query, const char **docs, int ndocs, float **scores_out)
{
    (void) cfg; (void) query; (void) docs;
    /* Minimal stub: descending scores */
    int i;
    *scores_out = (float *) palloc(sizeof(float) * ndocs);
    for (i = 0; i < ndocs; i++)
        (*scores_out)[i] = 1.0f - ((float) i / (float) Max(1,ndocs));
    return 0;
}


