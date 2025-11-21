/*-------------------------------------------------------------------------
 *
 * openai_http.c
 *    OpenAI API integration for NeurondB
 *
 * Implements OpenAI API client for text completion, embeddings, and
 * related LLM operations. Provides stub implementations that are
 * compilable and can be extended with full JSON parsing.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/llm/openai_http.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include <curl/curl.h>
#include <stdlib.h>
#include <ctype.h>
#include "neurondb_llm.h"

/* Helper: Quote a C string for JSON (returns JSON string with quotes and escaping) */
static char *
ndb_quote_json_cstr(const char *str)
{
	StringInfoData buf;
	const char *p;
	char *result;

	if (str == NULL)
		return pstrdup("null");

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '"');

	for (p = str; *p; p++)
	{
		switch (*p)
		{
			case '"':
				appendStringInfoString(&buf, "\\\"");
				break;
			case '\\':
				appendStringInfoString(&buf, "\\\\");
				break;
			case '\b':
				appendStringInfoString(&buf, "\\b");
				break;
			case '\f':
				appendStringInfoString(&buf, "\\f");
				break;
			case '\n':
				appendStringInfoString(&buf, "\\n");
				break;
			case '\r':
				appendStringInfoString(&buf, "\\r");
				break;
			case '\t':
				appendStringInfoString(&buf, "\\t");
				break;
			default:
				if ((unsigned char) *p < 0x20)
				{
					appendStringInfo(&buf, "\\u%04x", (unsigned char) *p);
				} else
				{
					appendStringInfoChar(&buf, *p);
				}
				break;
		}
	}

	appendStringInfoChar(&buf, '"');
	result = pstrdup(buf.data);
	pfree(buf.data);
	return result;
}

/* Helper for dynamic memory buffer for curl writes */
typedef struct
{
	char *data;
	size_t len;
} MemBuf;

static size_t
write_cb(void *ptr, size_t size, size_t nmemb, void *userdata)
{
	MemBuf *m = (MemBuf *)userdata;
	size_t n = size * nmemb;
	m->data = repalloc(m->data, m->len + n + 1);
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
	CURL *curl = curl_easy_init();
	struct curl_slist *headers = NULL;
	MemBuf buf = { palloc0(1), 0 };
	long code = 0;
	CURLcode res;
	if (!curl)
		return -1;

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
	curl_easy_setopt(curl, CURLOPT_USERAGENT, "neurondb-openai/1.0");

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

/* Extracts text from OpenAI chat completion API response
 *
 * OpenAI response format:
 * {
 *   "id": "chatcmpl-...",
 *   "object": "chat.completion",
 *   "created": 1234567890,
 *   "model": "gpt-3.5-turbo",
 *   "choices": [{
 *     "index": 0,
 *     "message": {
 *       "role": "assistant",
 *       "content": "response text here"
 *     },
 *     "finish_reason": "stop"
 *   }],
 *   "usage": {
 *     "prompt_tokens": 10,
 *     "completion_tokens": 20,
 *     "total_tokens": 30
 *   }
 * }
 *
 * Stub implementation: returns placeholder text
 * TODO: Implement full JSON parsing
 */
static char *
extract_openai_text(const char *json)
{
	const char *key;
	const char *p;
	const char *q;
	size_t len;
	char *out;

	if (!json || json[0] == '\0')
		return NULL;
	if (strncmp(json, "{\"error\"", 8) == 0)
		return NULL;

	/* Look for "content" field in choices[0].message.content */
	key = "\"content\":";
	p = strstr(json, key);
	if (!p)
	{
		/* Fallback: return stub message */
		return pstrdup("OpenAI response (stub - JSON parsing not yet implemented)");
	}

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

/* Extracts token counts from OpenAI response
 * Returns true if found, false otherwise
 */
static bool
extract_openai_tokens(const char *json,
	int *tokens_in,
	int *tokens_out)
{
	const char *p;
	char *endptr;
	long val;

	if (!json || !tokens_in || !tokens_out)
		return false;

	/* Look for "prompt_tokens" */
	p = strstr(json, "\"prompt_tokens\":");
	if (p)
	{
		p = strchr(p, ':');
		if (p)
		{
			p++;
			while (*p && isspace((unsigned char)*p))
				p++;
			val = strtol(p, &endptr, 10);
			if (endptr != p)
				*tokens_in = (int)val;
		}
	}

	/* Look for "completion_tokens" */
	p = strstr(json, "\"completion_tokens\":");
	if (p)
	{
		p = strchr(p, ':');
		if (p)
		{
			p++;
			while (*p && isspace((unsigned char)*p))
				p++;
			val = strtol(p, &endptr, 10);
			if (endptr != p)
				*tokens_out = (int)val;
		}
	}

	return true;
}

/* Extracts embedding vector from OpenAI embedding API response
 *
 * OpenAI response format:
 * {
 *   "object": "list",
 *   "data": [{
 *     "object": "embedding",
 *     "index": 0,
 *     "embedding": [0.1, 0.2, 0.3, ...]
 *   }],
 *   "model": "text-embedding-ada-002",
 *   "usage": {
 *     "prompt_tokens": 8,
 *     "total_tokens": 8
 *   }
 * }
 *
 * Stub implementation: returns placeholder vector
 * TODO: Implement full JSON parsing
 */
static bool
parse_openai_emb_vector(const char *json, float **vec_out, int *dim_out)
{
	const char *p;
	float *vec = NULL;
	int n = 0;
	int cap = 32;
	char *endptr;
	double v;

	if (!json)
		return false;

	/* Look for "embedding":[ */
	p = strstr(json, "\"embedding\":");
	if (!p)
	{
		/* Stub: return zero vector */
		*dim_out = 1536; /* OpenAI text-embedding-ada-002 dimension */
		*vec_out = (float *)palloc(sizeof(float) * (*dim_out));
		for (int i = 0; i < *dim_out; i++)
			(*vec_out)[i] = 0.0f;
		return true;
	}

	/* Find the opening bracket */
	p = strchr(p, '[');
	if (!p)
		return false;
	p++;

	vec = (float *)palloc(sizeof(float) * cap);

	/* Parse array of floats */
	while (*p && *p != ']')
	{
		while (*p && (isspace(*p) || *p == ','))
			p++;
		if (*p == ']')
			break;
		endptr = NULL;
		v = strtod(p, &endptr);
		if (endptr == p)
			break;
		if (n == cap)
		{
			cap *= 2;
			vec = repalloc(vec, sizeof(float) * cap);
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
		/* Fallback: return zero vector */
		*dim_out = 1536;
		*vec_out = (float *)palloc(sizeof(float) * (*dim_out));
		for (int i = 0; i < *dim_out; i++)
			(*vec_out)[i] = 0.0f;
		return true;
	}
}

/* OpenAI chat completion API
 *
 * Endpoint: POST /v1/chat/completions
 * Request body:
 * {
 *   "model": "gpt-3.5-turbo",
 *   "messages": [{"role": "user", "content": "prompt"}],
 *   "temperature": 0.7,
 *   "max_tokens": 100
 * }
 */
int
ndb_openai_complete(const NdbLLMConfig *cfg,
	const char *prompt,
	const char *params_json,
	NdbLLMResp *out)
{
	StringInfoData url, body;
	char *model_quoted;
	char *prompt_quoted;
	initStringInfo(&url);
	initStringInfo(&body);

	if (prompt == NULL || out == NULL)
	{
		pfree(url.data);
		pfree(body.data);
		return -1;
	}

	/* Build OpenAI chat completion endpoint */
	if (cfg->endpoint)
	{
		appendStringInfo(&url, "%s/v1/chat/completions", cfg->endpoint);
	} else
	{
		appendStringInfo(&url, "https://api.openai.com/v1/chat/completions");
	}

	/* Quote model and prompt for JSON */
	model_quoted = ndb_quote_json_cstr(cfg->model ? cfg->model : "gpt-3.5-turbo");
	prompt_quoted = ndb_quote_json_cstr(prompt);

	/* Build request body */
	appendStringInfo(&body,
		"{\"model\":%s,\"messages\":[{\"role\":\"user\",\"content\":%s}]",
		model_quoted,
		prompt_quoted);

	/* Append params_json if provided (temperature, max_tokens, etc.) */
	if (params_json && params_json[0] != '\0' && strcmp(params_json, "{}") != 0)
	{
		/* Merge params into body */
		appendStringInfoChar(&body, ',');
		/* Remove outer braces from params_json and append */
		const char *p = params_json;
		while (*p && (*p == '{' || isspace((unsigned char)*p)))
			p++;
		const char *end = params_json + strlen(params_json) - 1;
		while (end > p && (*end == '}' || isspace((unsigned char)*end)))
			end--;
		if (end > p)
		{
			size_t len = end - p + 1;
			appendStringInfo(&body, "%.*s", (int)len, p);
		}
	}

	appendStringInfoChar(&body, '}');

	pfree(model_quoted);
	pfree(prompt_quoted);

	/* Make HTTP request */
	out->http_status = http_post_json(
		url.data, cfg->api_key, body.data, cfg->timeout_ms, &out->json);

	out->text = NULL;
	out->tokens_in = 0;
	out->tokens_out = 0;

	if (out->json && out->http_status >= 200 && out->http_status < 300)
	{
		/* Extract response text */
		char *t = extract_openai_text(out->json);
		if (t)
			out->text = t;
		else
			out->text = pstrdup("OpenAI response (stub)");

		/* Extract token counts */
		extract_openai_tokens(out->json, &out->tokens_in, &out->tokens_out);
	} else if (out->json)
	{
		out->text = NULL;
	}

	pfree(url.data);
	pfree(body.data);

	return (out->http_status >= 200 && out->http_status < 300) ? 0 : -1;
}

/* OpenAI embedding API
 *
 * Endpoint: POST /v1/embeddings
 * Request body:
 * {
 *   "model": "text-embedding-ada-002",
 *   "input": "text to embed"
 * }
 */
int
ndb_openai_embed(const NdbLLMConfig *cfg,
	const char *text,
	float **vec_out,
	int *dim_out)
{
	StringInfoData url, body;
	char *model_quoted;
	char *text_quoted;
	char *json_resp = NULL;
	int http_status;

	initStringInfo(&url);
	initStringInfo(&body);

	if (text == NULL || vec_out == NULL || dim_out == NULL)
	{
		pfree(url.data);
		pfree(body.data);
		return -1;
	}

	/* Build OpenAI embedding endpoint */
	if (cfg->endpoint)
	{
		appendStringInfo(&url, "%s/v1/embeddings", cfg->endpoint);
	} else
	{
		appendStringInfo(&url, "https://api.openai.com/v1/embeddings");
	}

	/* Quote model and text for JSON */
	model_quoted = ndb_quote_json_cstr(
		cfg->model ? cfg->model : "text-embedding-ada-002");
	text_quoted = ndb_quote_json_cstr(text);

	/* Build request body */
	appendStringInfo(&body,
		"{\"model\":%s,\"input\":%s}",
		model_quoted,
		text_quoted);

	pfree(model_quoted);
	pfree(text_quoted);

	/* Make HTTP request */
	http_status = http_post_json(
		url.data, cfg->api_key, body.data, cfg->timeout_ms, &json_resp);

	if (http_status >= 200 && http_status < 300 && json_resp)
	{
		bool ok = parse_openai_emb_vector(json_resp, vec_out, dim_out);
		pfree(url.data);
		pfree(body.data);
		if (json_resp)
			pfree(json_resp);
		return ok ? 0 : -1;
	}

	pfree(url.data);
	pfree(body.data);
	if (json_resp)
		pfree(json_resp);
	return -1;
}

/* OpenAI batch embedding API
 *
 * Endpoint: POST /v1/embeddings
 * Request body:
 * {
 *   "model": "text-embedding-ada-002",
 *   "input": ["text1", "text2", ...]
 * }
 *
 * OpenAI supports batch embeddings in a single request
 */
int
ndb_openai_embed_batch(const NdbLLMConfig *cfg,
	const char **texts,
	int num_texts,
	float ***vecs_out,
	int **dims_out,
	int *num_success_out)
{
	StringInfoData url, body;
	char *model_quoted;
	char *json_resp = NULL;
	int http_status;
	float **vecs = NULL;
	int *dims = NULL;
	int success_count = 0;

	initStringInfo(&url);
	initStringInfo(&body);

	if (texts == NULL || num_texts <= 0 || vecs_out == NULL
		|| dims_out == NULL || num_success_out == NULL)
	{
		pfree(url.data);
		pfree(body.data);
		return -1;
	}

	/* Build OpenAI embedding endpoint */
	if (cfg->endpoint)
	{
		appendStringInfo(&url, "%s/v1/embeddings", cfg->endpoint);
	} else
	{
		appendStringInfo(&url, "https://api.openai.com/v1/embeddings");
	}

	/* Quote model for JSON */
	model_quoted = ndb_quote_json_cstr(
		cfg->model ? cfg->model : "text-embedding-ada-002");

	/* Build request body with array of inputs */
	appendStringInfo(&body, "{\"model\":%s,\"input\":[", model_quoted);

	for (int i = 0; i < num_texts; i++)
	{
		if (i > 0)
			appendStringInfoChar(&body, ',');
		char *text_quoted = ndb_quote_json_cstr(texts[i]);
		appendStringInfoString(&body, text_quoted);
		pfree(text_quoted);
	}

	appendStringInfoChar(&body, '}');
	appendStringInfoChar(&body, '}');

	pfree(model_quoted);

	/* Make HTTP request */
	http_status = http_post_json(
		url.data, cfg->api_key, body.data, cfg->timeout_ms, &json_resp);

	/* Allocate output arrays */
	vecs = (float **)palloc(sizeof(float *) * num_texts);
	dims = (int *)palloc(sizeof(int) * num_texts);

	if (http_status >= 200 && http_status < 300 && json_resp)
	{
		/* Parse batch response: array of embeddings */
		const char *p = json_resp;
		int vec_idx = 0;

		/* Find "data" array */
		p = strstr(p, "\"data\":");
		if (p)
		{
			p = strchr(p, '[');
			if (p)
			{
				p++;
				/* Parse each embedding in the array */
				while (*p && *p != ']' && vec_idx < num_texts)
				{
					/* Find "embedding":[ */
					const char *emb_start = strstr(p, "\"embedding\":");
					if (emb_start)
					{
						emb_start = strchr(emb_start, '[');
						if (emb_start)
						{
							emb_start++;
							/* Parse vector */
							float *vec = NULL;
							int vec_dim = 0;
							int vec_cap = 32;
							char *endptr;
							double v;

							vec = (float *)palloc(sizeof(float) * vec_cap);

							while (*emb_start && *emb_start != ']')
							{
								while (*emb_start
									&& (isspace((unsigned char)*emb_start)
										|| *emb_start == ','))
									emb_start++;
								if (*emb_start == ']')
									break;
								endptr = NULL;
								v = strtod(emb_start, &endptr);
								if (endptr == emb_start)
									break;
								if (vec_dim == vec_cap)
								{
									vec_cap *= 2;
									vec = repalloc(vec,
										sizeof(float) * vec_cap);
								}
								vec[vec_dim++] = (float)v;
								emb_start = endptr;
							}

							if (vec_dim > 0)
							{
								vecs[vec_idx] = vec;
								dims[vec_idx] = vec_dim;
								success_count++;
								vec_idx++;
							} else if (vec)
							{
								pfree(vec);
							}

							/* Move past this embedding object */
							p = strchr(emb_start, '}');
							if (p)
								p++;
						}
					} else
					{
						break;
					}
				}
			}
		}
	}

	/* Fill remaining slots with NULL/0 */
	for (int i = success_count; i < num_texts; i++)
	{
		vecs[i] = NULL;
		dims[i] = 0;
	}

	*vecs_out = vecs;
	*dims_out = dims;
	*num_success_out = success_count;

	pfree(url.data);
	pfree(body.data);
	if (json_resp)
		pfree(json_resp);

	return (success_count > 0) ? 0 : -1;
}

/* OpenAI image embedding
 *
 * Note: OpenAI doesn't have a direct image embedding API in v1.
 * Would require vision API or multimodal model (GPT-4 Vision).
 * This is a stub that reports not implemented.
 */
int
ndb_openai_image_embed(const NdbLLMConfig *cfg,
	const unsigned char *image_data,
	size_t image_size,
	float **vec_out,
	int *dim_out)
{
	ereport(ERROR,
		(errmsg("neurondb: OpenAI image embedding not yet implemented"),
			errdetail("OpenAI v1 API does not provide direct image embedding. "
				"Consider using GPT-4 Vision API or multimodal models.")));
	return -1;
}

/* OpenAI multimodal embedding
 *
 * Note: OpenAI multimodal embedding requires vision API or
 * GPT-4 Vision. This is a stub that reports not implemented.
 */
int
ndb_openai_multimodal_embed(const NdbLLMConfig *cfg,
	const char *text,
	const unsigned char *image_data,
	size_t image_size,
	float **vec_out,
	int *dim_out)
{
	ereport(ERROR,
		(errmsg("neurondb: OpenAI multimodal embedding not yet implemented"),
			errdetail("OpenAI multimodal embedding requires GPT-4 Vision API "
				"or similar multimodal models.")));
	return -1;
}

/* OpenAI reranking
 *
 * Note: OpenAI doesn't have a dedicated reranking API.
 * Could potentially use completion API with scoring prompt,
 * but this is not efficient. This is a stub that reports not implemented.
 */
int
ndb_openai_rerank(const NdbLLMConfig *cfg,
	const char *query,
	const char **docs,
	int ndocs,
	float **scores_out)
{
	ereport(ERROR,
		(errmsg("neurondb: OpenAI reranking not yet implemented"),
			errdetail("OpenAI does not provide a dedicated reranking API. "
				"Consider using completion API with custom scoring logic, "
				"or use a dedicated reranking model.")));
	return -1;
}

