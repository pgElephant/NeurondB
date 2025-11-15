#include "postgres.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include <curl/curl.h>
#include <stdlib.h>
#include "neurondb_llm.h"

/* Helper for dynamic memory buffer for curl writes */
typedef struct
{
	char *data;
	size_t len;
} MemBuf;

static size_t
write_cb(void *ptr, size_t size, size_t nmemb, void *userdata)
{
	MemBuf *m;
	size_t n;
	char *new_data;

	/* Defensive: Check NULL inputs */
	if (ptr == NULL || userdata == NULL)
		return 0;

	m = (MemBuf *)userdata;
	n = size * nmemb;

	/* Defensive: Check for overflow */
	if (n > SIZE_MAX - m->len - 1)
		return 0;

	new_data = repalloc(m->data, m->len + n + 1);
	if (new_data == NULL)
		return 0;

	m->data = new_data;
	memcpy(m->data + m->len, ptr, n);
	m->len += n;
	m->data[m->len] = '\0';
	return n;
}

/* HTTP POST with JSON body, outputs body and HTTP status code */
static int
http_post_json(const char *url,
	const char *api_key,
	const char *json_body,
	int timeout_ms,
	char **out)
{
	CURL *curl;
	struct curl_slist *headers = NULL;
	MemBuf buf;
	long code = 0;
	CURLcode res;

	/* Defensive: Check NULL inputs */
	if (url == NULL || out == NULL)
		return -1;

	buf.data = palloc0(1);
	if (buf.data == NULL)
		return -1;
	buf.len = 0;

	curl = curl_easy_init();
	if (!curl)
	{
		pfree(buf.data);
		return -1;
	}

	headers = curl_slist_append(headers, "Content-Type: application/json");
	if (api_key && api_key[0])
	{
		StringInfoData h;
		initStringInfo(&h);
		appendStringInfo(&h, "Authorization: Bearer %s", api_key);
		headers = curl_slist_append(headers, h.data);
	}
	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);
	curl_easy_setopt(curl, CURLOPT_USERAGENT, "neurondb-llm/1.0");

	res = curl_easy_perform(curl);
	if (res != CURLE_OK)
	{
		curl_slist_free_all(headers);
		curl_easy_cleanup(curl);
		if (buf.data)
			pfree(buf.data);
		return -1;
	}
	curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);

	*out = buf.data;
	return (int)code;
}

/* Extracts text field from HuggingFace inference API responses */
static char *
extract_hf_text(const char *json)
{
	/* The text generation output is a top-level list of { "generated_text": ... } objects.
     * Example: [{"generated_text":"result"}], so we parse it.
     * The response might also be { "error": ... }.
     */
	const char *key;
	char *p;
	char *q;
	size_t len;
	char *out;

	if (!json || json[0] == '\0')
		return NULL;
	if (strncmp(json, "{\"error\"", 8) == 0)
		return NULL;

	/* Find first "generated_text":"..." pattern */
	key = "\"generated_text\":";

	p = strstr(json, key);
	if (!p)
		return NULL;
	/* Find the first quote after the key */
	p = strchr(p + strlen(key), '"');
	if (!p)
		return NULL;
	p++;
	q = strchr(p, '"');
	if (!q)
		return NULL;
	len = q - p;
	out = (char *)palloc(len + 1);
	memcpy(out, p, len);
	out[len] = '\0';
	return out;
}

int
ndb_hf_complete(const NdbLLMConfig *cfg,
	const char *prompt,
	const char *params_json,
	NdbLLMResp *out)
{
	StringInfoData url, body;
	initStringInfo(&url);
	initStringInfo(&body);

	if (prompt == NULL)
	{
		pfree(url.data);
		pfree(body.data);
		return -1;
	}

	appendStringInfo(&url, "%s/models/%s", cfg->endpoint, cfg->model);
	appendStringInfo(&body,
		"{\"inputs\":%s,\"parameters\":%s}",
		quote_literal_cstr(prompt),
		params_json ? params_json : "{}");

	out->http_status = http_post_json(
		url.data, cfg->api_key, body.data, cfg->timeout_ms, &out->json);

	out->text = NULL;

	if (out->json && out->http_status >= 200 && out->http_status < 300)
	{
		/* Try to parse a "generated_text" value out */
		char *t = extract_hf_text(out->json);
		if (t)
			out->text = t;
		else
			out->text = pstrdup(out->json);
	} else if (out->json)
	{
		out->text = NULL;
	}

	return (out->http_status >= 200 && out->http_status < 300) ? 0 : -1;
}

/* Extracts a flat float vector from HF embedding API JSON response */
static bool
parse_hf_emb_vector(const char *json, float **vec_out, int *dim_out)
{
	/* Response is: [[float, float, ...]] */
	const char *p;
	float *vec = NULL;
	int n = 0;
	int cap = 32;
	char *endptr;
	double v;

	if (!json)
		return false;

	p = json;
	while (*p && *p != '[')
		p++;
	if (!*p)
		return false;
	p++;
	while (*p && isspace(*p))
		p++;
	/* Expect next char to be '[' */
	if (*p != '[')
		return false;
	p++;

	vec = (float *)palloc(sizeof(float) * cap);
	if (vec == NULL)
		return false;

	while (*p && *p != ']')
	{
		while (*p && (isspace(*p) || *p == ','))
			p++;
		endptr = NULL;
		v = strtod(p, &endptr);
		if (endptr == p)
			break;

		/* Defensive: Check for NaN/Inf */
		if (isnan(v) || isinf(v))
		{
			pfree(vec);
			return false;
		}

		if (n == cap)
		{
			/* Defensive: Check for overflow */
			if (cap > 1000000)
			{
				pfree(vec);
				return false;
			}
			cap *= 2;
			vec = repalloc(vec, sizeof(float) * cap);
			if (vec == NULL)
				return false;
		}
		vec[n++] = (float)v;
		p = endptr;
	}
	if (n > 0)
	{
		*vec_out = vec;
		*dim_out = n;
		return true;
	} else
	{
		pfree(vec);
		return false;
	}
}

int
ndb_hf_embed(const NdbLLMConfig *cfg,
	const char *text,
	float **vec_out,
	int *dim_out)
{
	StringInfoData url, body;
	char *resp = NULL;
	int code;
	bool ok;

	initStringInfo(&url);
	initStringInfo(&body);

	if (text == NULL)
	{
		pfree(url.data);
		pfree(body.data);
		return -1;
	}

	appendStringInfo(&url,
		"%s/pipeline/feature-extraction/%s",
		cfg->endpoint,
		cfg->model);
	appendStringInfo(&body,
		"{\"inputs\":%s,\"truncate\":true}",
		quote_literal_cstr(text));
	code = http_post_json(
		url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);
	if (code < 200 || code >= 300 || !resp)
	{
		if (resp)
			pfree(resp);
		return -1;
	}
	ok = parse_hf_emb_vector(resp, vec_out, dim_out);
	pfree(resp);
	if (!ok)
		return -1;
	return 0;
}

static bool
parse_hf_scores(const char *json, float **scores_out, int ndocs)
{
	/* The response is [{"scores":[float, float,...]}] or similar;
     * We will parse for the first float array in the string.
     */
	const char *scores_key = "\"scores\"";
	char *ps;
	float *scores;
	int n = 0;
	char *endptr;
	double v;

	if (!json)
		return false;
	ps = strstr(json, scores_key);
	if (!ps)
		return false;
	ps = strchr(ps, '[');
	if (!ps)
		return false;
	ps++;
	/* Defensive: Validate ndocs */
	if (ndocs <= 0 || ndocs > 1000000)
		return false;

	scores = palloc(sizeof(float) * ndocs);
	if (scores == NULL)
		return false;

	while (*ps && *ps != ']' && n < ndocs)
	{
		while (*ps && (isspace(*ps) || *ps == ','))
			ps++;
		endptr = NULL;
		v = strtod(ps, &endptr);
		if (endptr == ps)
			break;

		/* Defensive: Check for NaN/Inf */
		if (isnan(v) || isinf(v))
		{
			pfree(scores);
			return false;
		}

		scores[n++] = (float)v;
		ps = endptr;
	}
	if (n == ndocs)
	{
		*scores_out = scores;
		return true;
	}
	pfree(scores);
	return false;
}

int
ndb_hf_rerank(const NdbLLMConfig *cfg,
	const char *query,
	const char **docs,
	int ndocs,
	float **scores_out)
{
	StringInfoData url, body;
	StringInfoData docs_json;
	char *resp = NULL;
	int code;
	int i;
	bool ok;

	initStringInfo(&url);
	initStringInfo(&body);

	/* Validate inputs */
	if (query == NULL || docs == NULL || ndocs <= 0)
	{
		pfree(url.data);
		pfree(body.data);
		return -1;
	}

	/* Compose the docs JSON array */
	initStringInfo(&docs_json);
	appendStringInfoChar(&docs_json, '[');
	for (i = 0; i < ndocs; ++i)
	{
		if (i > 0)
			appendStringInfoChar(&docs_json, ',');
		if (docs[i] != NULL)
			appendStringInfoString(&docs_json, quote_literal_cstr(docs[i]));
		else
			appendStringInfoString(&docs_json, "null");
	}
	appendStringInfoChar(&docs_json, ']');

	/* URL: .../pipeline/text-classification/model */
	appendStringInfo(&url,
		"%s/pipeline/token-classification/%s",
		cfg->endpoint,
		cfg->model);
	/* Use models endpoint if above fails for reranking */
	appendStringInfo(&body,
		"{\"inputs\":{\"query\":%s,\"documents\":%s}}",
		quote_literal_cstr(query),
		docs_json.data);

	code = http_post_json(
		url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);

	if (code < 200 || code >= 300 || !resp)
	{
		if (resp)
			pfree(resp);
		return -1;
	}

	ok = parse_hf_scores(resp, scores_out, ndocs);
	pfree(resp);
	if (!ok)
		return -1;
	return 0;
}
