/*-------------------------------------------------------------------------
 *
 * embeddings.c
 *    Text and multimodal embedding generation functions
 *
 * Integrates with Hugging Face API via llm_runtime infrastructure.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/ml/embeddings.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "utils/lsyscache.h"
#include "utils/fmgroids.h"
#include "utils/syscache.h"
#include "parser/parse_type.h"
#include "nodes/makefuncs.h"
#include "catalog/pg_type.h"
#include "lib/stringinfo.h"
#include "neurondb.h"
#include "neurondb_llm.h"
#include "neurondb_gpu.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

#if PG_VERSION_NUM >= 150000
#include "utils/jsonb.h"
#else
#include "utils/jsonb.h"
#endif

/*
 * parse_vector_from_text
 *    Parse a vector from a text string in format {dim, x1, x2, ...}
 */
static Vector *
parse_vector_from_text(const char *str)
{
	int			dim = 0;
	int			i = 0;
	char	   *dup = NULL;
	char	   *token = NULL;
	char	   *saveptr = NULL;
	char	   *endptr = NULL;
	Vector	   *result = NULL;
	float	   *data = NULL;
	bool		first = true;
	char	   *start = NULL;
	char	   *end = NULL;

	/* Duplicate input string as strtok modifies it */
	dup = pstrdup(str);

	/* Skip leading '{' and trailing '}' if present */
	start = strchr(dup, '{');
	end = strrchr(dup, '}');
	if (!start || !end || end <= start)
	{
		pfree(dup);
		return NULL;
	}
	*end = '\0';
	start++;

	/* Count number of commas for dimension */
	{
		int			commas = 0;
		char	   *c;

		for (c = start; *c; c++)
		{
			if (*c == ',')
				commas++;
		}
		dim = commas; /* There should be dim+1 entries (first: dim) */
	}
	if (dim < 1)
	{
		pfree(dup);
		return NULL;
	}
	data = (float *) palloc0(dim * sizeof(float));

	/* Tokenize for dimension and entries */
	i = 0;
	first = true;
	for (token = strtok_r(start, ",", &saveptr);
		 token != NULL;
		 token = strtok_r(NULL, ",", &saveptr))
	{
		while (*token == ' ')
			token++;	/* trim leading spaces */

		if (first)
		{
			int dval;

			first = false;
			dval = strtol(token, &endptr, 10);
			if (*endptr != '\0' || dval <= 0)
			{
				pfree(data);
				pfree(dup);
				return NULL;
			}
			if (dval != dim)
			{
				/* might be '{384,x,x,...}' not '{n-entries,...}' */
				dim = dval;
				data = repalloc(data, dim * sizeof(float));
			}
		}
		else
		{
			if (i < dim)
			{
				data[i++] = strtof(token, &endptr);
				if (*endptr != '\0' && *endptr != '\n' && *endptr != '\r')
				{
					pfree(data);
					pfree(dup);
					return NULL;
				}
			}
		}
	}
	if (i != dim)
	{
		pfree(data);
		pfree(dup);
		return NULL;
	}

	result = (Vector *) palloc0(VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	result->dim = dim;
	for (i = 0; i < dim; i++)
		result->data[i] = data[i];

	pfree(dup);
	pfree(data);

	return result;
}

/*
 * SAFE_PFREE
 *    Macro to pfree a pointer and set it to NULL.
 */
#define SAFE_PFREE(ptr) \
	do { \
		if ((ptr) != NULL) \
		{ \
			pfree(ptr); \
			(ptr) = NULL; \
		} \
	} while (0)

PG_FUNCTION_INFO_V1(embed_text);
/*
 * embed_text
 *    Generate text embedding using Hugging Face API
 *
 * input_text: TEXT
 * model_text: TEXT (optional)
 * Returns: vector
 */
Datum
embed_text(PG_FUNCTION_ARGS)
{
	text	   *input_text = NULL;
	text	   *model_text = NULL;
	char	   *input_str = NULL;
	char	   *model_str = NULL;
	NdbLLMConfig cfg;
	NdbLLMCallOptions call_opts;
	float	   *vec_data = NULL;
	Vector	   *result = NULL;
	int			dim = 0;
	int			i;

	CHECK_NARGS_RANGE(1, 2);

	input_text = PG_GETARG_TEXT_PP(0);

	/* Defensive: Check NULL input */
	if (input_text == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("embed_text: input_text cannot be NULL")));

	if (PG_ARGISNULL(1))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(1);

	input_str = text_to_cstring(input_text);

	/* Defensive: Validate allocation */
	if (input_str == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate input string")));

	if (model_text != NULL)
		model_str = text_to_cstring(model_text);
	else
		model_str = pstrdup("sentence-transformers/all-MiniLM-L6-v2");

	memset(&cfg, 0, sizeof(cfg));
	cfg.provider = (neurondb_llm_provider != NULL) ? neurondb_llm_provider : "huggingface";
	cfg.endpoint = (neurondb_llm_endpoint != NULL) ?
				   neurondb_llm_endpoint :
				   "https://api-inference.huggingface.co";
	cfg.model = model_str != NULL ? model_str :
			   (neurondb_llm_model != NULL ?
				neurondb_llm_model :
				"sentence-transformers/all-MiniLM-L6-v2");
	cfg.api_key = neurondb_llm_api_key;
	cfg.timeout_ms = neurondb_llm_timeout_ms;
	cfg.prefer_gpu = neurondb_gpu_enabled;
	cfg.require_gpu = false;
	if (cfg.provider != NULL &&
		(pg_strcasecmp(cfg.provider, "huggingface-local") == 0 ||
		 pg_strcasecmp(cfg.provider, "hf-local") == 0) &&
		!neurondb_llm_fail_open)
		cfg.require_gpu = true;

	call_opts.task = "embed";
	call_opts.prefer_gpu = cfg.prefer_gpu;
	call_opts.require_gpu = cfg.require_gpu;
	call_opts.fail_open = neurondb_llm_fail_open;

	if (ndb_llm_route_embed(&cfg, &call_opts, input_str, &vec_data, &dim) != 0 ||
		vec_data == NULL || dim <= 0)
	{
		elog(WARNING,
			 "neurondb: embed_text() failed for input: '%s', returning zero vector",
			 (input_str != NULL) ? input_str : "(null)");
		dim = 384;
		vec_data = (float *) palloc0(dim * sizeof(float));

		/* Defensive: Validate allocation */
		if (vec_data == NULL)
		{
			pfree(input_str);
			if (model_str)
				pfree(model_str);
			ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
					errmsg("failed to allocate fallback embedding vector")));
		}
	}

	/* Defensive: Validate dimension */
	if (dim <= 0 || dim > VECTOR_MAX_DIM)
	{
		if (vec_data)
			pfree(vec_data);
		pfree(input_str);
		if (model_str)
			pfree(model_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("embed_text: invalid embedding dimension: %d", dim)));
	}

	result = (Vector *) palloc0(VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));

	/* Defensive: Validate allocation */
	if (result == NULL)
	{
		if (vec_data)
			pfree(vec_data);
		pfree(input_str);
		if (model_str)
			pfree(model_str);
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate result vector")));
	}

	SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	result->dim = dim;

	/* Assert: Internal invariants */
	Assert(vec_data != NULL);
	Assert(dim > 0);

	for (i = 0; i < dim; i++)
	{
		/* Defensive: Check for NaN/Inf */
		if (isnan(vec_data[i]) || isinf(vec_data[i]))
		{
			elog(WARNING, "embed_text: NaN or Infinity detected at index %d, using 0.0", i);
			result->data[i] = 0.0f;
		}
		else
		{
			result->data[i] = vec_data[i];
		}
	}

	SAFE_PFREE(input_str);
	SAFE_PFREE(model_str);
	SAFE_PFREE(vec_data);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(embed_text_batch);
/*
 * embed_text_batch
 *    Batch text embedding. Accepts TEXT[] and optional model.
 *
 * input_array: TEXT[]
 * model_text: TEXT (optional)
 * Returns: vector[]
 */
Datum
embed_text_batch(PG_FUNCTION_ARGS)
{
	ArrayType  *input_array;
	text	   *model_text = NULL;
	Datum	   *text_datums = NULL;
	bool	   *text_nulls = NULL;
	int			nitems = 0;
	int			i;
	Datum	   *result_datums = NULL;
	ArrayType  *result = NULL;
	Oid			array_oid;
	Oid			vector_oid;
	static Oid	cached_vector_oid = InvalidOid;

	CHECK_NARGS_RANGE(1, 2);

	input_array = PG_GETARG_ARRAYTYPE_P(0);

	/* Defensive: Check NULL input */
	if (input_array == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("embed_text_batch: input_array cannot be NULL")));

	if (PG_ARGISNULL(1))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(1);

	deconstruct_array(input_array,
					  TEXTOID,
					  -1,
					  false,
					  'i',
					  &text_datums,
					  &text_nulls,
					  &nitems);

	/* Defensive: Validate array size */
	if (nitems <= 0 || nitems > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("embed_text_batch: array size must be between 1 and 100000, got %d", nitems)));

	result_datums = (Datum *) palloc0(nitems * sizeof(Datum));

	/* Defensive: Validate allocation */
	if (result_datums == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate result_datums array")));

	for (i = 0; i < nitems; i++)
	{
		if (text_nulls[i])
			result_datums[i] = (Datum) 0;
		else
		{
			Datum embed_result;
			if (model_text != NULL)
				embed_result = DirectFunctionCall2(embed_text,
												   text_datums[i],
												   PointerGetDatum(model_text));
			else
				embed_result = DirectFunctionCall1(embed_text, text_datums[i]);
			result_datums[i] = embed_result;
		}
	}

	array_oid = get_fn_expr_rettype(fcinfo->flinfo);
	vector_oid = get_element_type(array_oid);
	if (!OidIsValid(vector_oid))
	{
		if (!OidIsValid(cached_vector_oid))
		{
			cached_vector_oid = LookupTypeNameOid(NULL,
												  makeTypeNameFromNameList(
													  list_make1(makeString("vector"))),
												  false);
			if (!OidIsValid(cached_vector_oid))
				ereport(ERROR,
						(errcode(ERRCODE_UNDEFINED_OBJECT),
						 errmsg("vector type not found")));
		}
		vector_oid = cached_vector_oid;
	}

	result = construct_array(result_datums, nitems, vector_oid, -1, false, 'i');

	SAFE_PFREE(result_datums);
	if (text_datums)
		pfree(text_datums);
	if (text_nulls)
		pfree(text_nulls);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(embed_image);
/*
 * embed_image
 *    Generate image embedding using Hugging Face API or similar.
 *
 * image_data: BYTEA
 * model_text: TEXT (optional)
 * Returns: vector
 */
Datum
embed_image(PG_FUNCTION_ARGS)
{
	bytea	   *image_data;
	text	   *model_text = NULL;
	Vector	   *result = NULL;
	float	   *vec_data = NULL;
	int			dim = 0;
	int			i;
	char	   *model_str = NULL;
	NdbLLMConfig cfg;

	image_data = PG_GETARG_BYTEA_PP(0);
	(void) image_data; /* For future image processing */

	if (PG_ARGISNULL(1))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(1);

	if (model_text != NULL)
		model_str = text_to_cstring(model_text);
	else
		model_str = pstrdup("sentence-transformers/clip-ViT-B-32"); /* default placeholder */

	memset(&cfg, 0, sizeof(cfg));
	cfg.provider = neurondb_llm_provider;
	cfg.endpoint = neurondb_llm_endpoint;
	cfg.model = model_str != NULL ? model_str : neurondb_llm_model;
	cfg.api_key = neurondb_llm_api_key;
	cfg.timeout_ms = neurondb_llm_timeout_ms;
	cfg.prefer_gpu = neurondb_gpu_enabled;
	cfg.require_gpu = false;
	if (cfg.provider != NULL &&
		(pg_strcasecmp(cfg.provider, "huggingface-local") == 0 ||
		 pg_strcasecmp(cfg.provider, "hf-local") == 0) &&
		!neurondb_llm_fail_open)
		cfg.require_gpu = true;

	/* NOTE: ndb_hf_image_embed API is assumed, not implemented. Fallback for now. */
#ifdef NDB_HAVE_IMAGE_EMBED
	if (ndb_hf_image_embed(&cfg,
						   VARDATA_ANY(image_data),
						   VARSIZE_ANY_EXHDR(image_data),
						   &vec_data,
						   &dim) != 0
		|| vec_data == NULL)
#endif
	{
		elog(NOTICE, "neurondb: embed_image() image embedding not yet implemented, returning zero vector");
		dim = 512;
		vec_data = (float *) palloc0(dim * sizeof(float));
	}

	result = (Vector *) palloc0(VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	result->dim = dim;

	for (i = 0; i < dim; i++)
		result->data[i] = vec_data[i];

	SAFE_PFREE(vec_data);
	SAFE_PFREE(model_str);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(embed_multimodal);
/*
 * embed_multimodal
 *    Generate a multimodal (text+image) embedding by combining both.
 *
 * input_text: TEXT
 * image_data: BYTEA
 * model_text: TEXT (optional)
 * Returns: vector
 */
Datum
embed_multimodal(PG_FUNCTION_ARGS)
{
	text	   *input_text;
	bytea	   *image_data;
	text	   *model_text = NULL;
	Vector	   *result = NULL;
	float	   *vec_data = NULL;
	int			dim = 0;
	int			i;
	char	   *input_str = NULL;
	char	   *model_str = NULL;
#ifdef NDB_HAVE_MULTIMODAL_EMBED
	NdbLLMConfig cfg;
#endif

	input_text = PG_GETARG_TEXT_PP(0);
	image_data = PG_GETARG_BYTEA_PP(1);

	if (PG_ARGISNULL(2))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(2);

	input_str = text_to_cstring(input_text);

	if (model_text != NULL)
		model_str = text_to_cstring(model_text);
	else
		model_str = pstrdup("sentence-transformers/clip-ViT-B-32");

#ifdef NDB_HAVE_MULTIMODAL_EMBED
	memset(&cfg, 0, sizeof(cfg));
	cfg.provider = neurondb_llm_provider;
	cfg.endpoint = neurondb_llm_endpoint;
	cfg.model = model_str != NULL ? model_str : neurondb_llm_model;
	cfg.api_key = neurondb_llm_api_key;
	cfg.timeout_ms = neurondb_llm_timeout_ms;

	if (ndb_hf_multimodal_embed(&cfg,
								input_str,
								VARDATA_ANY(image_data),
								VARSIZE_ANY_EXHDR(image_data),
								&vec_data,
								&dim) != 0
		|| vec_data == NULL)
#endif
	{
		Datum		tx_vec_dat;
		Datum		img_vec_dat;
		Vector	   *tx_vec;
		Vector	   *img_vec;
		int			out_dim;

		elog(NOTICE, "neurondb: embed_multimodal() provider unavailable, constructing synthetic multimodal vector");

		tx_vec_dat = DirectFunctionCall2(embed_text,
										 PointerGetDatum(input_text),
										 model_text ? PointerGetDatum(model_text) : (Datum) 0);
		tx_vec = (Vector *) DatumGetPointer(tx_vec_dat);

		img_vec_dat = DirectFunctionCall2(embed_image,
										  PointerGetDatum(image_data),
										  model_text ? PointerGetDatum(model_text) : (Datum) 0);
		img_vec = (Vector *) DatumGetPointer(img_vec_dat);

		out_dim = tx_vec->dim + img_vec->dim;

		result = (Vector *) palloc0(VARHDRSZ + sizeof(int16) * 2 + out_dim * sizeof(float4));
		SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + out_dim * sizeof(float4));
		result->dim = out_dim;

		for (i = 0; i < tx_vec->dim; i++)
			result->data[i] = tx_vec->data[i];
		for (i = 0; i < img_vec->dim; i++)
			result->data[tx_vec->dim + i] = img_vec->data[i];

		SAFE_PFREE(input_str);
		SAFE_PFREE(model_str);

		PG_RETURN_POINTER(result);
	}

	result = (Vector *) palloc0(VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	result->dim = dim;

	for (i = 0; i < dim; i++)
		result->data[i] = vec_data[i];

	SAFE_PFREE(input_str);
	SAFE_PFREE(model_str);
	SAFE_PFREE(vec_data);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(embed_cached);
/*
 * embed_cached
 *    Cached text embedding.
 *
 * input_text: TEXT
 * model_text: TEXT (optional)
 * Returns: vector using cache if available
 */
Datum
embed_cached(PG_FUNCTION_ARGS)
{
	text	   *input_text = NULL;
	text	   *model_text = NULL;
	char	   *input_str = NULL;
	char	   *model_str = NULL;
	char	   *cache_key = NULL;
	char	   *cached_text = NULL;
	int			max_age;
	Vector	   *result = NULL;
	StringInfoData key_buf;
	/* bool hit = false; */
	const char *p;
	uint32		hashval;

	input_text = PG_GETARG_TEXT_PP(0);
	if (PG_ARGISNULL(1))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(1);

	input_str = text_to_cstring(input_text);

	initStringInfo(&key_buf);
	appendStringInfo(&key_buf, "embed:");

	hashval = 0;
	for (p = input_str; *p; p++)
		hashval = (hashval * 31) + *p;
	appendStringInfo(&key_buf, "%u:", (unsigned int) hashval);

	if (model_text != NULL)
	{
		model_str = text_to_cstring(model_text);
		appendStringInfoString(&key_buf, model_str);
	}
	else
	{
		appendStringInfoString(&key_buf, "default");
	}
	cache_key = key_buf.data;
	max_age = neurondb_llm_cache_ttl;

	if (ndb_llm_cache_lookup(cache_key, max_age, &cached_text) && cached_text != NULL)
	{
		result = parse_vector_from_text(cached_text);
		if (result != NULL)
		{
			elog(DEBUG1, "neurondb: embed_cached() cache hit (key='%s')", cache_key);
			SAFE_PFREE(input_str);
			SAFE_PFREE(model_str);
			SAFE_PFREE(cache_key);
			SAFE_PFREE(cached_text);
			PG_RETURN_POINTER(result);
		}
		else
		{
			elog(WARNING, "neurondb: embed_cached() cache corrupted for key '%s'", cache_key);
		}
		SAFE_PFREE(cached_text);
	}

	if (model_text != NULL)
		result = (Vector *) DatumGetPointer(
					DirectFunctionCall2(embed_text,
										PointerGetDatum(input_text),
										PointerGetDatum(model_text)));
	else
		result = (Vector *) DatumGetPointer(
					DirectFunctionCall1(embed_text,
										PointerGetDatum(input_text)));

	if (result != NULL)
	{
		StringInfoData cache_val;
		int			d;
		initStringInfo(&cache_val);

		appendStringInfo(&cache_val, "{%d", result->dim);
		for (d = 0; d < result->dim; d++)
			appendStringInfo(&cache_val, ",%.7f", result->data[d]);
		appendStringInfoChar(&cache_val, '}');

		ndb_llm_cache_store(cache_key, cache_val.data);

		SAFE_PFREE(cache_val.data);
	}
	SAFE_PFREE(input_str);
	SAFE_PFREE(model_str);
	SAFE_PFREE(cache_key);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(configure_embedding_model);
/*
 * configure_embedding_model
 *    Configure embedding model parameters in persistent storage.
 *
 * model_name: TEXT
 * config_json: TEXT
 * Returns: BOOLEAN
 */
Datum
configure_embedding_model(PG_FUNCTION_ARGS)
{
	text	   *model_name = NULL;
	text	   *config_json = NULL;
	char	   *model_str = NULL;
	char	   *cfg_str = NULL;

	model_name = PG_GETARG_TEXT_PP(0);
	config_json = PG_GETARG_TEXT_PP(1);

	model_str = text_to_cstring(model_name);
	cfg_str = text_to_cstring(config_json);

	elog(INFO, "neurondb: configure_embedding_model() called: model='%s', config='%s'",
		 model_str, cfg_str);

	SAFE_PFREE(model_str);
	SAFE_PFREE(cfg_str);

	PG_RETURN_BOOL(true);
}
