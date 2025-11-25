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
#include "utils/jsonb.h"
#include "common/jsonapi.h"
#include "parser/parse_type.h"
#include "parser/parse_func.h"
#include "utils/lsyscache.h"
#include "catalog/pg_proc.h"
#include "nodes/makefuncs.h"
#include <curl/curl.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <errno.h>
#include <float.h>
#include "neurondb_llm.h"
#include "neurondb_safe_memory.h"
#include "neurondb_validation.h"

/* Forward declaration for image validation */
/* ImageMetadata and related functions are now defined in neurondb_llm.h */

/* Helper: Look up and call PostgreSQL's encode() function for base64 encoding */
static text *
ndb_encode_base64(bytea * data)
{
	List	   *funcname;
	Oid			argtypes[2];
	Oid			encode_oid;
	FmgrInfo	flinfo;
	Datum		result;

	/* Look up encode(bytea, text) function */
	funcname = list_make1(makeString("encode"));
	argtypes[0] = BYTEAOID;
	argtypes[1] = TEXTOID;
	encode_oid = LookupFuncName(funcname, 2, argtypes, false);

	if (!OidIsValid(encode_oid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_FUNCTION),
				 errmsg("encode function not found")));

	fmgr_info(encode_oid, &flinfo);
	result = FunctionCall2(&flinfo,
						   PointerGetDatum(data),
						   CStringGetDatum("base64"));

	return DatumGetTextP(result);
}

/* Helper: Quote a C string for JSON (returns JSON string with quotes and escaping) */
static char *
ndb_quote_json_cstr(const char *str)
{
	StringInfoData buf;
	const char *p;
	char	   *result;

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
				}
				else
				{
					appendStringInfoChar(&buf, *p);
				}
				break;
		}
	}

	appendStringInfoChar(&buf, '"');
	result = pstrdup(buf.data);
	NDB_SAFE_PFREE_AND_NULL(buf.data);
	buf.data = NULL;
	return result;
}

/* Helper for dynamic memory buffer for curl writes */
typedef struct
{
	char	   *data;
	size_t		len;
}			MemBuf;

static size_t
write_cb(void *ptr, size_t size, size_t nmemb, void *userdata)
{
	MemBuf	   *m = (MemBuf *) userdata;
	size_t		n = size * nmemb;

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
	CURL	   *curl = NULL;
	struct curl_slist *headers = NULL;
	MemBuf		buf = {palloc0(1), 0};
	long		code = 0;
	CURLcode	res = CURLE_OK;
	int			result = -1;
	char	   *auth_header_data = NULL;

	if (out == NULL)
		return -1;

	*out = NULL;

	PG_TRY();
	{
		curl = curl_easy_init();
		if (!curl)
		{
			NDB_SAFE_PFREE_AND_NULL(buf.data);
			buf.data = NULL;
			return -1;
		}

		headers = curl_slist_append(headers, "Content-Type: application/json");
		if (api_key && api_key[0])
		{
			StringInfoData h;

			initStringInfo(&h);
			appendStringInfo(&h, "Authorization: Bearer %s", api_key);
			auth_header_data = h.data;
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
			if (headers)
			{
				curl_slist_free_all(headers);
				headers = NULL;
			}
			if (curl)
			{
				curl_easy_cleanup(curl);
				curl = NULL;
			}
			if (buf.data)
			{
				NDB_SAFE_PFREE_AND_NULL(buf.data);
				buf.data = NULL;
			}
			if (auth_header_data)
			{
				NDB_SAFE_PFREE_AND_NULL(auth_header_data);
				auth_header_data = NULL;
			}
			return -1;
		}
		{
			CURLcode	getinfo_res = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);

			if (getinfo_res != CURLE_OK)
			{
				elog(WARNING,
					 "neurondb: http_post_json: failed to get HTTP response code: %s",
					 curl_easy_strerror(getinfo_res));
			}
		}
		if (headers)
		{
			curl_slist_free_all(headers);
			headers = NULL;
		}
		if (curl)
		{
			curl_easy_cleanup(curl);
			curl = NULL;
		}

		*out = buf.data;
		buf.data = NULL;		/* Ownership transferred to caller */
		if (auth_header_data)
		{
			NDB_SAFE_PFREE_AND_NULL(auth_header_data);
			auth_header_data = NULL;
		}
		result = (int) code;
	}
	PG_CATCH();
	{
		FlushErrorState();

		/* Cleanup all resources with NULL assignment */
		if (headers)
		{
			curl_slist_free_all(headers);
			headers = NULL;
		}
		if (curl)
		{
			curl_easy_cleanup(curl);
			curl = NULL;
		}
		if (buf.data)
		{
			NDB_SAFE_PFREE_AND_NULL(buf.data);
			buf.data = NULL;
		}
		if (auth_header_data)
		{
			NDB_SAFE_PFREE_AND_NULL(auth_header_data);
			auth_header_data = NULL;
		}
		if (out)
			*out = NULL;

		result = -1;
	}
	PG_END_TRY();

	return result;
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
 * Robust parsing with proper JSON string handling
 */
static char *
extract_openai_text(const char *json)
{
	const char *p;
	const char *q;
	size_t		len;
	char	   *result = NULL;
	char	   *unescaped = NULL;
	int			escape_next = 0;

	if (!json || json[0] == '\0')
		return NULL;

	/* Check for error response */
	if (strncmp(json, "{\"error\"", 8) == 0)
		return NULL;

	/*
	 * Look for "content" field - handle nested structure:
	 * choices[0].message.content
	 */
	/* First find "choices" array */
	p = strstr(json, "\"choices\"");
	if (!p)
		return NULL;

	/* Find opening bracket of choices array */
	p = strchr(p, '[');
	if (!p)
		return NULL;
	p++;

	/* Skip whitespace and find first object */
	while (*p && (isspace((unsigned char) *p) || *p == '{'))
	{
		if (*p == '{')
			break;
		p++;
	}
	if (*p != '{')
		return NULL;

	/* Now look for "message" within this choice object */
	p = strstr(p, "\"message\"");
	if (!p)
		return NULL;

	/* Find opening brace of message object */
	p = strchr(p, '{');
	if (!p)
		return NULL;
	p++;

	/* Look for "content" within message */
	p = strstr(p, "\"content\"");
	if (!p)
		return NULL;

	/* Find colon after "content" */
	p = strchr(p, ':');
	if (!p)
		return NULL;
	p++;

	/* Skip whitespace */
	while (*p && isspace((unsigned char) *p))
		p++;

	/* Check if value is a string (starts with quote) */
	if (*p != '"')
		return NULL;
	p++;						/* Skip opening quote */

	/* Extract string value, handling escaped characters */
	q = p;
	len = 0;
	while (*q)
	{
		if (escape_next)
		{
			escape_next = 0;
			len++;
			q++;
			continue;
		}
		if (*q == '\\')
		{
			escape_next = 1;
			len++;
			q++;
			continue;
		}
		if (*q == '"')
		{
			/* Found closing quote (not escaped) */
			break;
		}
		len++;
		q++;
	}

	if (len == 0)
		return NULL;

	/* Allocate and copy string */
	result = (char *) palloc(len + 1);
	unescaped = result;
	q = p;

	/* Copy and unescape */
	while (q < p + len)
	{
		if (*q == '\\' && q + 1 < p + len)
		{
			switch (q[1])
			{
				case 'n':
					*unescaped++ = '\n';
					q += 2;
					break;
				case 't':
					*unescaped++ = '\t';
					q += 2;
					break;
				case 'r':
					*unescaped++ = '\r';
					q += 2;
					break;
				case '\\':
					*unescaped++ = '\\';
					q += 2;
					break;
				case '"':
					*unescaped++ = '"';
					q += 2;
					break;
				case '/':
					*unescaped++ = '/';
					q += 2;
					break;
				case 'u':
					/* Unicode escape - for now, skip it or handle basic case */
					if (q + 5 < p + len && isxdigit((unsigned char) q[2]) &&
						isxdigit((unsigned char) q[3]) &&
						isxdigit((unsigned char) q[4]) &&
						isxdigit((unsigned char) q[5]))
					{
						/* Simple ASCII range unicode handling */
						unsigned int code = 0;

						sscanf(q + 2, "%4x", &code);
						if (code < 128)
						{
							*unescaped++ = (char) code;
						}
						else
						{
							/* Non-ASCII: keep as-is for now */
							*unescaped++ = '?';
						}
						q += 6;
					}
					else
					{
						*unescaped++ = *q++;
					}
					break;
				default:
					/* Unknown escape, keep both characters */
					*unescaped++ = *q++;
					*unescaped++ = *q++;
					break;
			}
		}
		else
		{
			*unescaped++ = *q++;
		}
	}
	*unescaped = '\0';

	return result;
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
	char	   *endptr;
	long		val;

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
			while (*p && isspace((unsigned char) *p))
				p++;
			val = strtol(p, &endptr, 10);
			if (endptr != p)
				*tokens_in = (int) val;
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
			while (*p && isspace((unsigned char) *p))
				p++;
			val = strtol(p, &endptr, 10);
			if (endptr != p)
				*tokens_out = (int) val;
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
 * Robust parsing of embedding array
 */
static bool
parse_openai_emb_vector(const char *json, float **vec_out, int *dim_out)
{
	const char *p;
	float	   *vec = NULL;
	int			n = 0;
	int			cap = 256;		/* Start with reasonable capacity */
	char	   *endptr;
	double		v;
	bool		in_array = false;

	if (!json || !vec_out || !dim_out)
		return false;

	/* Look for "data" array first, then "embedding" within it */
	p = strstr(json, "\"data\"");
	if (p)
	{
		/* Find opening bracket of data array */
		p = strchr(p, '[');
		if (p)
		{
			p++;
			/* Find first object in data array */
			p = strchr(p, '{');
			if (p)
			{
				/* Look for "embedding" within this object */
				p = strstr(p, "\"embedding\"");
			}
		}
	}

	/* If not found in data array, try direct search */
	if (!p || !strstr(p, "\"embedding\""))
	{
		p = strstr(json, "\"embedding\":");
	}

	if (!p)
	{
		/* No embedding found - return error */
		return false;
	}

	/* Find the opening bracket after "embedding": */
	p = strchr(p, '[');
	if (!p)
		return false;
	p++;
	in_array = true;

	/* Allocate initial vector */
	vec = (float *) palloc(sizeof(float) * cap);

	/* Parse array of floats */
	while (*p && in_array)
	{
		/* Skip whitespace and commas */
		while (*p && (isspace((unsigned char) *p) || *p == ','))
			p++;

		if (*p == ']')
		{
			in_array = false;
			break;
		}

		if (!*p)
			break;

		/* Parse float value */
		endptr = NULL;
		errno = 0;
		v = strtod(p, &endptr);

		/* Check for parse error */
		if (endptr == p || errno == ERANGE)
		{
			/* Invalid number, try to continue */
			while (*p && *p != ',' && *p != ']' && !isspace((unsigned char) *p))
				p++;
			continue;
		}

		/* Check for overflow */
		if (v > FLT_MAX || v < -FLT_MAX)
		{
			/* Value out of range, skip */
			p = endptr;
			continue;
		}

		/* Grow array if needed */
		if (n >= cap)
		{
			cap = cap * 2;
			vec = repalloc(vec, sizeof(float) * cap);
		}

		vec[n++] = (float) v;
		p = endptr;
	}

	if (n > 0)
	{
		/* Trim to actual size */
		if (n < cap)
		{
			vec = repalloc(vec, sizeof(float) * n);
		}
		*vec_out = vec;
		*dim_out = n;
		return true;
	}

	/* No valid values found */
	if (vec)
	{
		NDB_SAFE_PFREE_AND_NULL(vec);
		vec = NULL;
	}
	return false;
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
ndb_openai_complete(const NdbLLMConfig * cfg,
					const char *prompt,
					const char *params_json,
					NdbLLMResp * out)
{
	StringInfoData url,
				body;
	char	   *model_quoted;
	char	   *prompt_quoted;

	initStringInfo(&url);
	initStringInfo(&body);

	if (prompt == NULL || out == NULL)
	{
		NDB_SAFE_PFREE_AND_NULL(url.data);
		url.data = NULL;
		NDB_SAFE_PFREE_AND_NULL(body.data);
		body.data = NULL;
		return -1;
	}

	/* Build OpenAI chat completion endpoint */
	if (cfg->endpoint)
	{
		appendStringInfo(&url, "%s/v1/chat/completions", cfg->endpoint);
	}
	else
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
		const char *p;
		const char *end;

		/* Merge params into body */
		appendStringInfoChar(&body, ',');
		/* Remove outer braces from params_json and append */
		p = params_json;
		while (*p && (*p == '{' || isspace((unsigned char) *p)))
			p++;
		end = params_json + strlen(params_json) - 1;
		while (end > p && (*end == '}' || isspace((unsigned char) *end)))
			end--;
		if (end > p)
		{
			size_t		len = end - p + 1;

			appendStringInfo(&body, "%.*s", (int) len, p);
		}
	}

	appendStringInfoChar(&body, '}');

	NDB_SAFE_PFREE_AND_NULL(model_quoted);
	model_quoted = NULL;
	NDB_SAFE_PFREE_AND_NULL(prompt_quoted);
	prompt_quoted = NULL;

	/* Make HTTP request */
	out->http_status = http_post_json(
									  url.data, cfg->api_key, body.data, cfg->timeout_ms, &out->json);

	out->text = NULL;
	out->tokens_in = 0;
	out->tokens_out = 0;

	if (out->json && out->http_status >= 200 && out->http_status < 300)
	{
		/* Extract response text */
		char	   *t = extract_openai_text(out->json);

		if (t)
		{
			out->text = t;
		}
		else
		{
			/* Fallback: try to extract any text content from JSON using JSONB */
			Jsonb	   *jsonb = NULL;

			PG_TRY();
			{
				jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
														   CStringGetDatum(out->json)));
				if (jsonb)
				{
					/* Try to extract from choices[0].message.content */
					JsonbIterator *it = JsonbIteratorInit(&jsonb->root);
					JsonbValue	v;
					JsonbIteratorToken type;
					bool		found_content = false;

					while ((type = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
					{
						if (type == WJB_KEY && v.type == jbvString &&
							strncmp(v.val.string.val, "content", v.val.string.len) == 0)
						{
							type = JsonbIteratorNext(&it, &v, true);
							if (type == WJB_VALUE && v.type == jbvString)
							{
								out->text = pnstrdup(v.val.string.val,
													 v.val.string.len);
								found_content = true;
								break;
							}
						}
					}

					if (!found_content)
					{
						/* Last resort: extract any string value */
						it = JsonbIteratorInit(&jsonb->root);
						while ((type = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
						{
							if (type == WJB_VALUE && v.type == jbvString &&
								v.val.string.len > 0)
							{
								out->text = pnstrdup(v.val.string.val,
													 v.val.string.len);
								break;
							}
						}
					}
				}
			}
			PG_CATCH();
			{
				FlushErrorState();
			}
			PG_END_TRY();

			if (!out->text)
				out->text = pstrdup("OpenAI response (unable to parse)");
		}

		/* Extract token counts */
		extract_openai_tokens(out->json, &out->tokens_in, &out->tokens_out);
	}
	else if (out->json)
	{
		out->text = NULL;
	}

	NDB_SAFE_PFREE_AND_NULL(url.data);
	url.data = NULL;
	NDB_SAFE_PFREE_AND_NULL(body.data);
	body.data = NULL;

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
ndb_openai_embed(const NdbLLMConfig * cfg,
				 const char *text,
				 float **vec_out,
				 int *dim_out)
{
	StringInfoData url,
				body;
	char	   *model_quoted;
	char	   *text_quoted;
	char	   *json_resp = NULL;
	int			http_status;

	initStringInfo(&url);
	initStringInfo(&body);

	if (text == NULL || vec_out == NULL || dim_out == NULL)
	{
		NDB_SAFE_PFREE_AND_NULL(url.data);
		url.data = NULL;
		NDB_SAFE_PFREE_AND_NULL(body.data);
		body.data = NULL;
		return -1;
	}

	/* Build OpenAI embedding endpoint */
	if (cfg->endpoint)
	{
		appendStringInfo(&url, "%s/v1/embeddings", cfg->endpoint);
	}
	else
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

	NDB_SAFE_PFREE_AND_NULL(model_quoted);
	model_quoted = NULL;
	NDB_SAFE_PFREE_AND_NULL(text_quoted);
	text_quoted = NULL;

	/* Make HTTP request */
	http_status = http_post_json(
								 url.data, cfg->api_key, body.data, cfg->timeout_ms, &json_resp);

	if (http_status >= 200 && http_status < 300 && json_resp)
	{
		bool		ok = parse_openai_emb_vector(json_resp, vec_out, dim_out);

		NDB_SAFE_PFREE_AND_NULL(url.data);
		NDB_SAFE_PFREE_AND_NULL(body.data);
		if (json_resp)
			NDB_SAFE_PFREE_AND_NULL(json_resp);
		return ok ? 0 : -1;
	}

	NDB_SAFE_PFREE_AND_NULL(url.data);
	NDB_SAFE_PFREE_AND_NULL(body.data);
	if (json_resp)
		NDB_SAFE_PFREE_AND_NULL(json_resp);
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
ndb_openai_embed_batch(const NdbLLMConfig * cfg,
					   const char **texts,
					   int num_texts,
					   float ***vecs_out,
					   int **dims_out,
					   int *num_success_out)
{
	StringInfoData url,
				body;
	char	   *model_quoted;
	char	   *json_resp = NULL;
	int			http_status;
	float	  **vecs = NULL;
	int		   *dims = NULL;
	int			success_count = 0;

	initStringInfo(&url);
	initStringInfo(&body);

	if (texts == NULL || num_texts <= 0 || vecs_out == NULL
		|| dims_out == NULL || num_success_out == NULL)
	{
		NDB_SAFE_PFREE_AND_NULL(url.data);
		NDB_SAFE_PFREE_AND_NULL(body.data);
		return -1;
	}

	/* Build OpenAI embedding endpoint */
	if (cfg->endpoint)
	{
		appendStringInfo(&url, "%s/v1/embeddings", cfg->endpoint);
	}
	else
	{
		appendStringInfo(&url, "https://api.openai.com/v1/embeddings");
	}

	/* Quote model for JSON */
	model_quoted = ndb_quote_json_cstr(
									   cfg->model ? cfg->model : "text-embedding-ada-002");

	/* Build request body with array of inputs */
	appendStringInfo(&body, "{\"model\":%s,\"input\":[", model_quoted);

	{
		int			i;
		char	   *text_quoted;

		for (i = 0; i < num_texts; i++)
		{
			if (i > 0)
				appendStringInfoChar(&body, ',');
			text_quoted = ndb_quote_json_cstr(texts[i]);
			appendStringInfoString(&body, text_quoted);
			NDB_SAFE_PFREE_AND_NULL(text_quoted);
		}
	}

	appendStringInfoChar(&body, '}');
	appendStringInfoChar(&body, '}');

	NDB_SAFE_PFREE_AND_NULL(model_quoted);

	/* Make HTTP request */
	http_status = http_post_json(
								 url.data, cfg->api_key, body.data, cfg->timeout_ms, &json_resp);

	/* Allocate output arrays */
	vecs = (float **) palloc(sizeof(float *) * num_texts);
	dims = (int *) palloc(sizeof(int) * num_texts);

	if (http_status >= 200 && http_status < 300 && json_resp)
	{
		/* Parse batch response: array of embeddings */
		const char *p = json_resp;
		int			vec_idx = 0;

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
							float	   *vec;
							int			vec_dim;
							int			vec_cap;
							char	   *endptr;
							double		v;

							emb_start++;
							/* Parse vector */
							vec = NULL;
							vec_dim = 0;
							vec_cap = 32;

							vec = (float *) palloc(sizeof(float) * vec_cap);

							while (*emb_start && *emb_start != ']')
							{
								while (*emb_start
									   && (isspace((unsigned char) *emb_start)
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
								vec[vec_dim++] = (float) v;
								emb_start = endptr;
							}

							if (vec_dim > 0)
							{
								vecs[vec_idx] = vec;
								dims[vec_idx] = vec_dim;
								success_count++;
								vec_idx++;
							}
							else if (vec)
							{
								NDB_SAFE_PFREE_AND_NULL(vec);
							}

							/* Move past this embedding object */
							p = strchr(emb_start, '}');
							if (p)
								p++;
						}
					}
					else
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

	NDB_SAFE_PFREE_AND_NULL(url.data);
	NDB_SAFE_PFREE_AND_NULL(body.data);
	if (json_resp)
		NDB_SAFE_PFREE_AND_NULL(json_resp);

	return (success_count > 0) ? 0 : -1;
}

/* OpenAI Vision API - Image understanding (image-to-text)
 *
 * Uses GPT-4 Vision or other vision-capable models to analyze images.
 * Endpoint: POST /v1/chat/completions
 * Request body includes image as base64 in messages content.
 */
int
ndb_openai_vision_complete(const NdbLLMConfig * cfg,
						   const unsigned char *image_data,
						   size_t image_size,
						   const char *prompt,
						   const char *params_json,
						   NdbLLMResp * out)
{
	StringInfoData url,
				body;
	char	   *model_quoted = NULL;
	char	   *prompt_quoted = NULL;
	char	   *base64_data = NULL;
	text	   *encoded_text = NULL;
	bytea	   *image_bytea = NULL;
	ImageMetadata *img_meta;
	const char *vision_model;
	const char *vision_prompt;
	const char *mime_type;
	const char *p;
	const char *end;
	int			rc = -1;

	initStringInfo(&url);
	initStringInfo(&body);

	if (image_data == NULL || image_size == 0 || out == NULL)
	{
		NDB_SAFE_PFREE_AND_NULL(url.data);
		NDB_SAFE_PFREE_AND_NULL(body.data);
		return -1;
	}

	/* Validate image format and size */
	img_meta = ndb_validate_image(image_data, image_size, CurrentMemoryContext);
	if (!img_meta || !img_meta->is_valid)
	{
		const char *error_msg = img_meta && img_meta->error_msg
			? img_meta->error_msg
			: "Invalid image data";

		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: Invalid image: %s", error_msg)));
		if (img_meta)
		{
			if (img_meta->mime_type)
				NDB_SAFE_PFREE_AND_NULL(img_meta->mime_type);
			if (img_meta->error_msg)
				NDB_SAFE_PFREE_AND_NULL(img_meta->error_msg);
			NDB_SAFE_PFREE_AND_NULL(img_meta);
		}
		NDB_SAFE_PFREE_AND_NULL(url.data);
		NDB_SAFE_PFREE_AND_NULL(body.data);
		return -1;
	}

	/* Convert image data to bytea, then base64 encode */
	image_bytea = (bytea *) palloc(VARHDRSZ + image_size);
	SET_VARSIZE(image_bytea, VARHDRSZ + image_size);
	memcpy(VARDATA(image_bytea), image_data, image_size);

	encoded_text = ndb_encode_base64(image_bytea);
	base64_data = text_to_cstring(encoded_text);

	NDB_SAFE_PFREE_AND_NULL(image_bytea);
	NDB_SAFE_PFREE_AND_NULL(encoded_text);

	/* Build OpenAI chat completion endpoint for vision */
	if (cfg->endpoint)
	{
		appendStringInfo(&url, "%s/v1/chat/completions", cfg->endpoint);
	}
	else
	{
		appendStringInfo(&url, "https://api.openai.com/v1/chat/completions");
	}

	/* Use GPT-4 Vision or gpt-4o if model not specified */
	vision_model = cfg->model ? cfg->model : "gpt-4o";
	model_quoted = ndb_quote_json_cstr(vision_model);

	/* Use default prompt if not provided */
	vision_prompt = prompt ? prompt : "What's in this image? Describe it in detail.";
	prompt_quoted = ndb_quote_json_cstr(vision_prompt);

	/* Build request body with image in messages - use detected MIME type */
	mime_type = img_meta->mime_type ? img_meta->mime_type : "image/jpeg";
	appendStringInfo(&body,
					 "{\"model\":%s,\"messages\":[{\"role\":\"user\",\"content\":["
					 "{\"type\":\"text\",\"text\":%s},"
					 "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:%s;base64,%s\"}}"
					 "]}]",
					 model_quoted,
					 prompt_quoted,
					 mime_type,
					 base64_data);

	/* Append params_json if provided (temperature, max_tokens, etc.) */
	if (params_json && params_json[0] != '\0' && strcmp(params_json, "{}") != 0)
	{
		appendStringInfoChar(&body, ',');
		/* Remove outer braces from params_json and append */
		p = params_json;
		while (*p && (*p == '{' || isspace((unsigned char) *p)))
			p++;
		end = params_json + strlen(params_json) - 1;
		while (end > p && (*end == '}' || isspace((unsigned char) *end)))
			end--;
		if (end > p)
		{
			size_t		len = end - p + 1;

			appendStringInfo(&body, "%.*s", (int) len, p);
		}
	}

	appendStringInfoChar(&body, '}');

	NDB_SAFE_PFREE_AND_NULL(model_quoted);
	NDB_SAFE_PFREE_AND_NULL(prompt_quoted);
	NDB_SAFE_PFREE_AND_NULL(base64_data);

	/* Free image metadata */
	if (img_meta)
	{
		if (img_meta->mime_type)
			NDB_SAFE_PFREE_AND_NULL(img_meta->mime_type);
		if (img_meta->error_msg)
			NDB_SAFE_PFREE_AND_NULL(img_meta->error_msg);
		NDB_SAFE_PFREE_AND_NULL(img_meta);
	}

	/* Make HTTP request with retry logic */
	out->http_status = http_post_json(
									  url.data, cfg->api_key, body.data, cfg->timeout_ms, &out->json);

	out->text = NULL;
	out->tokens_in = 0;
	out->tokens_out = 0;

	if (out->json && out->http_status >= 200 && out->http_status < 300)
	{
		out->text = extract_openai_text(out->json);
		extract_openai_tokens(out->json, &out->tokens_in, &out->tokens_out);
		rc = 0;
	}

	NDB_SAFE_PFREE_AND_NULL(url.data);
	NDB_SAFE_PFREE_AND_NULL(body.data);

	return rc;
}

/* OpenAI image embedding
 *
 * OpenAI doesn't have a direct image embedding API, but we can use
 * GPT-4 Vision or text-embedding models. For now, we use a workaround
 * by converting image to base64 and using vision API for embedding.
 * Alternatively, use OpenAI's text-embedding-3 models with image descriptions.
 */
int
ndb_openai_image_embed(const NdbLLMConfig * cfg,
					   const unsigned char *image_data,
					   size_t image_size,
					   float **vec_out,
					   int *dim_out)
{
	/*
	 * OpenAI doesn't provide direct image embedding. Options: 1. Use GPT-4
	 * Vision to describe image, then embed description 2. Use third-party
	 * vision embedding models 3. Return error suggesting alternative approach
	 */
	ereport(ERROR,
			(errmsg("neurondb: OpenAI does not provide direct image embedding"),
			 errdetail("OpenAI v1 API does not provide direct image embedding. "
					   "Consider using: "
					   "1. neurondb_llm_image_analyze() to get image description, "
					   "then embed the description text, "
					   "2. Use HuggingFace CLIP models via neurondb_embed_image(), "
					   "3. Use OpenAI Vision API for image understanding.")));
	return -1;
}

/* OpenAI multimodal embedding
 *
 * OpenAI doesn't provide direct multimodal embedding. We can:
 * 1. Use GPT-4 Vision to analyze image+text, then embed the combined description
 * 2. Embed text and image description separately and concatenate
 * For now, we embed the text and suggest using vision API for image understanding.
 */
int
ndb_openai_multimodal_embed(const NdbLLMConfig * cfg,
							const char *text,
							const unsigned char *image_data,
							size_t image_size,
							float **vec_out,
							int *dim_out)
{
	/* Embed text only (image would need vision API analysis first) */
	if (text && strlen(text) > 0)
	{
		return ndb_openai_embed(cfg, text, vec_out, dim_out);
	}

	ereport(ERROR,
			(errmsg("neurondb: OpenAI multimodal embedding requires text input"),
			 errdetail("OpenAI does not provide direct multimodal embedding. "
					   "Use neurondb_llm_image_analyze() to get image description, "
					   "then combine with text and embed using neurondb_llm_embed().")));
	return -1;
}

/* OpenAI reranking
 *
 * Note: OpenAI doesn't have a dedicated reranking API.
 * Could potentially use completion API with scoring prompt,
 * but this is not efficient. This is a stub that reports not implemented.
 */
int
ndb_openai_rerank(const NdbLLMConfig * cfg,
				  const char *query,
				  const char **docs,
				  int ndocs,
				  float **scores_out)
{
	/*
	 * OpenAI doesn't have a dedicated reranking API, but we can use the chat
	 * completion API with a scoring prompt to achieve reranking. This is a
	 * workaround implementation.
	 */
	StringInfoData url;
	StringInfoData body;
	char	   *json_response = NULL;
	int			http_status;
	int			i;
	float	   *scores = NULL;

	if (!cfg || !cfg->api_key || !query || !docs || ndocs <= 0 || !scores_out)
		return -1;

	/* Allocate scores array */
	scores = (float *) palloc(sizeof(float) * ndocs);
	if (!scores)
		return -1;

	/* Use chat completion API with scoring prompt for each document */
	for (i = 0; i < ndocs; i++)
	{
		initStringInfo(&url);
		appendStringInfo(&url, "https://api.openai.com/v1/chat/completions");

		initStringInfo(&body);
		appendStringInfo(&body,
						 "{"
						 "\"model\":\"gpt-3.5-turbo\","
						 "\"messages\":["
						 "{\"role\":\"system\",\"content\":\"Rate relevance from 0.0 to 1.0.\"},"
						 "{\"role\":\"user\",\"content\":\"Query: %s\\nDocument: %s\\nRelevance score:\"}"
						 "],"
						 "\"max_tokens\":10,"
						 "\"temperature\":0"
						 "}",
						 query, docs[i] ? docs[i] : "");

		http_status = http_post_json(url.data, cfg->api_key, body.data,
									 cfg->timeout_ms > 0 ? cfg->timeout_ms : 30000, &json_response);

		NDB_SAFE_PFREE_AND_NULL(url.data);
		NDB_SAFE_PFREE_AND_NULL(body.data);

		if (http_status == 200 && json_response)
		{
			/* Extract score from response */
			char	   *text = extract_openai_text(json_response);

			if (text)
			{
				float		score = 0.5f;	/* Default */

				sscanf(text, "%f", &score);
				if (score < 0.0f)
					score = 0.0f;
				if (score > 1.0f)
					score = 1.0f;
				scores[i] = score;
				NDB_SAFE_PFREE_AND_NULL(text);
			}
			else
			{
				scores[i] = 0.5f;	/* Default if parsing fails */
			}
			NDB_SAFE_PFREE_AND_NULL(json_response);
			json_response = NULL;
		}
		else
		{
			scores[i] = 0.5f;	/* Default on error */
			if (json_response)
			{
				NDB_SAFE_PFREE_AND_NULL(json_response);
				json_response = NULL;
			}
		}
	}

	*scores_out = scores;
	return 0;
}
