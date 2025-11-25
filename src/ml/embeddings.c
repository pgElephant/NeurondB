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
#include "utils/guc.h"
#include "parser/parse_type.h"
#include "nodes/makefuncs.h"
#include "catalog/pg_type.h"
#include "lib/stringinfo.h"
#include "executor/spi.h"
#include "access/tupdesc.h"
#include "funcapi.h"
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
 * quote_literal_cstr
 *    Quote a C string for SQL (returns SQL string literal with quotes and escaping)
 */
static char *
ndb_quote_json_cstr(const char *str)
{
	StringInfoData buf;
	const char *p;
	char	   *result;

	if (str == NULL)
		return pstrdup("NULL");

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '\'');

	for (p = str; *p; p++)
	{
		if (*p == '\'')
			appendStringInfoString(&buf, "''");
		else if (*p == '\\')
			appendStringInfoString(&buf, "\\\\");
		else
			appendStringInfoChar(&buf, *p);
	}

	appendStringInfoChar(&buf, '\'');
	result = pstrdup(buf.data);
	pfree(buf.data);
	return result;
}

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
		dim = commas;			/* There should be dim+1 entries (first: dim) */
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
			token++;			/* trim leading spaces */

		if (first)
		{
			int			dval;

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

/*
 * get_embedding_model_config_internal
 *    Retrieve stored configuration for a model from catalog table.
 *    Returns Jsonb* in caller's memory context, or NULL if not found.
 *    Caller must pfree the result.
 */
static Jsonb *
get_embedding_model_config_internal(const char *model_name)
{
	Jsonb	   *result = NULL;
	StringInfoData sql;
	int			spi_ret;
	Datum		datum;
	bool		isnull;
	MemoryContext oldcontext;

	if (model_name == NULL)
		return NULL;

	oldcontext = CurrentMemoryContext;

	if (SPI_connect() != SPI_OK_CONNECT)
		return NULL;

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT config_json FROM neurondb.neurondb_embedding_model_config "
					 "WHERE model_name = %s",
					 ndb_quote_json_cstr(model_name));

	spi_ret = SPI_execute(sql.data, true, 1);

	if (spi_ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		datum = SPI_getbinval(SPI_tuptable->vals[0],
							  SPI_tuptable->tupdesc,
							  1,
							  &isnull);
		if (!isnull)
		{
			Jsonb	   *temp_jsonb;

			temp_jsonb = DatumGetJsonbP(datum);
			MemoryContextSwitchTo(oldcontext);
			result = (Jsonb *) PG_DETOAST_DATUM_COPY((Datum) temp_jsonb);
		}
	}

	pfree(sql.data);
	SPI_finish();

	return result;
}

/*
 * apply_embedding_model_config
 *    Apply stored configuration to NdbLLMConfig structure.
 *    Merges stored config with GUC defaults.
 */
static void
apply_embedding_model_config(NdbLLMConfig * cfg, const char *model_name)
{
	Jsonb	   *config_jsonb = NULL;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	int			r;

	if (cfg == NULL || model_name == NULL)
		return;

	config_jsonb = get_embedding_model_config_internal(model_name);
	if (config_jsonb == NULL)
		return;

	it = JsonbIteratorInit(&config_jsonb->root);
	while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
	{
		if (r == WJB_KEY)
		{
			char	   *key = NULL;
			int			key_len = v.val.string.len;

			key = pnstrdup(v.val.string.val, key_len);
			r = JsonbIteratorNext(&it, &v, false);
			if (r == WJB_VALUE)
			{
				if (strcmp(key, "batch_size") == 0)
				{
					/* batch_size is handled by batch API, not config */
				}
				else if (strcmp(key, "normalize") == 0)
				{
					/* normalize is handled by embedding API, not config */
				}
				else if (strcmp(key, "device") == 0)
				{
					/* device is handled by prefer_gpu/require_gpu */
					if (v.type == jbvString)
					{
						char	   *device_str = NULL;

						device_str = pnstrdup(v.val.string.val, v.val.string.len);
						if (pg_strcasecmp(device_str, "gpu") == 0 ||
							pg_strcasecmp(device_str, "cuda") == 0)
						{
							cfg->prefer_gpu = true;
						}
						pfree(device_str);
					}
				}
				else if (strcmp(key, "timeout_ms") == 0)
				{
					if (v.type == jbvNumeric)
					{
						int32		timeout = DatumGetInt32(
															DirectFunctionCall1(numeric_int4,
																				NumericGetDatum(v.val.numeric)));

						if (timeout >= 100 && timeout <= 300000)
							cfg->timeout_ms = timeout;
					}
				}
			}
			pfree(key);
		}
	}

	if (config_jsonb)
		pfree(config_jsonb);
}

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

	input_text = PG_GETARG_TEXT_PP(0);
	if (PG_ARGISNULL(1))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(1);

	input_str = text_to_cstring(input_text);

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

	/* Apply stored model configuration if available */
	apply_embedding_model_config(&cfg, cfg.model);

	call_opts.task = "embed";
	call_opts.prefer_gpu = cfg.prefer_gpu;
	call_opts.require_gpu = cfg.require_gpu;
	call_opts.fail_open = neurondb_llm_fail_open;

	if (ndb_llm_route_embed(&cfg, &call_opts, input_str, &vec_data, &dim) != 0 ||
		vec_data == NULL || dim <= 0)
	{
		dim = 384;
		vec_data = (float *) palloc0(dim * sizeof(float));
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
	bool	   *result_nulls = NULL;
	ArrayType  *result = NULL;
	Oid			array_oid;
	Oid			vector_oid;
	static Oid cached_vector_oid = InvalidOid;

	input_array = PG_GETARG_ARRAYTYPE_P(0);
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

	result_datums = (Datum *) palloc0(nitems * sizeof(Datum));
	result_nulls = (bool *) palloc0(nitems * sizeof(bool));

	/* Use batch API for better performance */
	{
		char	  **text_cstrs = NULL;
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;
		float	  **vecs = NULL;
		int		   *dims = NULL;
		int			num_success = 0;
		int			rc;
		int			j;

		/* Convert text datums to C strings */
		text_cstrs = (char **) palloc(nitems * sizeof(char *));
		for (i = 0; i < nitems; i++)
		{
			if (text_nulls[i])
				text_cstrs[i] = NULL;
			else
				text_cstrs[i] = text_to_cstring((text *) DatumGetPointer(text_datums[i]));
		}

		/* Setup config */
		{
			char	   *model_str = NULL;

			memset(&cfg, 0, sizeof(cfg));
			cfg.provider = (neurondb_llm_provider != NULL) ? neurondb_llm_provider : "huggingface";
			cfg.endpoint = (neurondb_llm_endpoint != NULL) ?
				neurondb_llm_endpoint :
				"https://api-inference.huggingface.co";
			if (model_text != NULL)
			{
				model_str = text_to_cstring(model_text);
				cfg.model = model_str;
			}
			else
			{
				cfg.model = (neurondb_llm_model != NULL ?
							 neurondb_llm_model :
							 "sentence-transformers/all-MiniLM-L6-v2");
			}
			cfg.api_key = neurondb_llm_api_key;
			cfg.timeout_ms = neurondb_llm_timeout_ms;
			cfg.prefer_gpu = neurondb_gpu_enabled;
			cfg.require_gpu = false;

			/* Apply stored model configuration if available */
			apply_embedding_model_config(&cfg, cfg.model);

			/*
			 * Note: model_str is used in cfg.model, so we must keep it until
			 * after batch call
			 */
		}

		call_opts.task = "embed";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = cfg.require_gpu;
		call_opts.fail_open = neurondb_llm_fail_open;

		/* Call batch embedding API */
		rc = ndb_llm_route_embed_batch(&cfg, &call_opts,
									   (const char **) text_cstrs, nitems,
									   &vecs, &dims, &num_success);

		/* Convert results to Datum array */
		if (rc == NDB_LLM_ROUTE_SUCCESS && vecs != NULL && dims != NULL)
		{
			for (i = 0; i < nitems; i++)
			{
				if (text_nulls[i] || vecs[i] == NULL || dims[i] <= 0)
				{
					result_datums[i] = (Datum) 0;
					result_nulls[i] = true;
				}
				else
				{
					Vector	   *result_vec = NULL;

					result_vec = (Vector *) palloc0(VARHDRSZ + sizeof(int16) * 2 + dims[i] * sizeof(float4));
					SET_VARSIZE(result_vec, VARHDRSZ + sizeof(int16) * 2 + dims[i] * sizeof(float4));
					result_vec->dim = dims[i];
					for (j = 0; j < dims[i]; j++)
						result_vec->data[j] = vecs[i][j];
					result_datums[i] = PointerGetDatum(result_vec);
				}
			}

			/* Free batch results */
			for (i = 0; i < num_success; i++)
			{
				if (vecs[i])
					pfree(vecs[i]);
			}
			if (vecs)
				pfree(vecs);
			if (dims)
				pfree(dims);
		}
		else
		{
			/* Batch API failed, fall back to individual calls */
			for (i = 0; i < nitems; i++)
			{
				if (text_nulls[i])
				{
					result_datums[i] = (Datum) 0;
					result_nulls[i] = true;
				}
				else
				{
					Datum		embed_result;

					if (model_text != NULL)
						embed_result = DirectFunctionCall2(embed_text,
														   text_datums[i],
														   PointerGetDatum(model_text));
					else
						embed_result = DirectFunctionCall1(embed_text, text_datums[i]);
					result_datums[i] = embed_result;
				}
			}
		}

		/* Free C strings */
		for (i = 0; i < nitems; i++)
		{
			if (text_cstrs[i])
				pfree(text_cstrs[i]);
		}
		pfree(text_cstrs);

		/* Free model_str if allocated */
		if (model_text != NULL && cfg.model != NULL)
		{
			/* cfg.model points to model_str, safe to free */
			pfree((char *) cfg.model);
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

	{
		int16		typlen;
		bool		typbyval;
		char		typalign;
		int			dims[1];
		int			lbs[1];

		get_typlenbyvalalign(vector_oid, &typlen, &typbyval, &typalign);
		dims[0] = nitems;
		lbs[0] = 1;
		result = construct_md_array(result_datums, result_nulls, 1, dims, lbs, vector_oid, typlen, typbyval, typalign);
	}

	SAFE_PFREE(result_datums);
	SAFE_PFREE(result_nulls);
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

	/* Validate image data */
	if (image_data == NULL || VARSIZE_ANY_EXHDR(image_data) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("embed_image: image_data must not be NULL or empty")));

	if (PG_ARGISNULL(1))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(1);

	if (model_text != NULL)
		model_str = text_to_cstring(model_text);
	else
		model_str = pstrdup("sentence-transformers/clip-ViT-B-32");

	memset(&cfg, 0, sizeof(cfg));
	cfg.provider = (neurondb_llm_provider != NULL) ? neurondb_llm_provider : "huggingface";
	cfg.endpoint = (neurondb_llm_endpoint != NULL) ?
		neurondb_llm_endpoint :
		"https://api-inference.huggingface.co";
	cfg.model = model_str != NULL ? model_str :
		(neurondb_llm_model != NULL ?
		 neurondb_llm_model :
		 "sentence-transformers/clip-ViT-B-32");
	cfg.api_key = neurondb_llm_api_key;
	cfg.timeout_ms = neurondb_llm_timeout_ms;
	cfg.prefer_gpu = neurondb_gpu_enabled;
	cfg.require_gpu = false;
	if (cfg.provider != NULL &&
		(pg_strcasecmp(cfg.provider, "huggingface-local") == 0 ||
		 pg_strcasecmp(cfg.provider, "hf-local") == 0) &&
		!neurondb_llm_fail_open)
		cfg.require_gpu = true;

	{
		NdbLLMCallOptions call_opts;
		bytea	   *detoasted_image = NULL;
		size_t		image_size;
		const unsigned char *image_bytes;
		bool		need_free_detoasted = false;

		/* Detoast the bytea to ensure we have uncompressed data */
		detoasted_image = (bytea *) PG_DETOAST_DATUM(PointerGetDatum(image_data));
		/* Check if detoasting created a new copy */
		if (detoasted_image != image_data)
			need_free_detoasted = true;
		image_size = VARSIZE_ANY_EXHDR(detoasted_image);
		image_bytes = (const unsigned char *) VARDATA_ANY(detoasted_image);

		call_opts.task = "embed";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = cfg.require_gpu;
		call_opts.fail_open = neurondb_llm_fail_open;

		if (ndb_llm_route_image_embed(&cfg, &call_opts,
									  image_bytes, image_size, &vec_data, &dim) != 0 ||
			vec_data == NULL || dim <= 0)
		{
			dim = 512;
			vec_data = (float *) palloc0(dim * sizeof(float));
		}

		/* Free detoasted image if we created a copy */
		if (need_free_detoasted && detoasted_image != NULL)
			pfree(detoasted_image);
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

	/* Validate inputs */
	if (image_data == NULL || VARSIZE_ANY_EXHDR(image_data) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("embed_multimodal: image_data must not be NULL or empty")));

	if (model_text != NULL)
		model_str = text_to_cstring(model_text);
	else
		model_str = pstrdup("sentence-transformers/clip-ViT-B-32");

	{
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;
		bytea	   *detoasted_image = NULL;
		size_t		image_size;
		const unsigned char *image_bytes;
		bool		need_free_detoasted = false;

		/* Detoast the bytea to ensure we have uncompressed data */

		/*
		 * PG_GETARG_BYTEA_PP already handles detoasting, but we need to
		 * ensure it's fully detoasted
		 */
		detoasted_image = (bytea *) PG_DETOAST_DATUM(PointerGetDatum(image_data));
		/* Check if detoasting created a new copy */
		if (detoasted_image != image_data)
			need_free_detoasted = true;
		image_size = VARSIZE_ANY_EXHDR(detoasted_image);
		image_bytes = (const unsigned char *) VARDATA_ANY(detoasted_image);

		memset(&cfg, 0, sizeof(cfg));
		cfg.provider = (neurondb_llm_provider != NULL) ? neurondb_llm_provider : "huggingface";
		cfg.endpoint = (neurondb_llm_endpoint != NULL) ?
			neurondb_llm_endpoint :
			"https://api-inference.huggingface.co";
		cfg.model = model_str != NULL ? model_str :
			(neurondb_llm_model != NULL ?
			 neurondb_llm_model :
			 "sentence-transformers/clip-ViT-B-32");
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

		/* Try true multimodal embedding first */
		if (ndb_llm_route_multimodal_embed(&cfg, &call_opts,
										   input_str, image_bytes, image_size, &vec_data, &dim) != 0 ||
			vec_data == NULL || dim <= 0)
		{
			/* Fallback: concatenate text and image embeddings */
			float	   *text_vec = NULL;
			float	   *img_vec = NULL;
			int			text_dim = 0;
			int			img_dim = 0;

			if (ndb_llm_route_embed(&cfg, &call_opts, input_str, &text_vec, &text_dim) == 0 &&
				ndb_llm_route_image_embed(&cfg, &call_opts, image_bytes, image_size, &img_vec, &img_dim) == 0 &&
				text_vec != NULL && img_vec != NULL && text_dim > 0 && img_dim > 0)
			{
				dim = text_dim + img_dim;
				vec_data = (float *) palloc(dim * sizeof(float));
				memcpy(vec_data, text_vec, text_dim * sizeof(float));
				memcpy(vec_data + text_dim, img_vec, img_dim * sizeof(float));
				pfree(text_vec);
				pfree(img_vec);
			}
			else
			{
				dim = 512;
				vec_data = (float *) palloc0(dim * sizeof(float));
				if (text_vec)
					pfree(text_vec);
				if (img_vec)
					pfree(img_vec);
			}
		}

		/* Free detoasted image if we created a copy */
		if (need_free_detoasted && detoasted_image != NULL)
			pfree(detoasted_image);
	}

	result = (Vector *) palloc0(VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	result->dim = dim;

	for (i = 0; i < dim; i++)
		result->data[i] = vec_data[i];

	SAFE_PFREE(vec_data);
	SAFE_PFREE(input_str);
	SAFE_PFREE(model_str);

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

	/* Copy cache_key to current context */
	cache_key = pstrdup(key_buf.data);
	pfree(key_buf.data);
	max_age = neurondb_llm_cache_ttl;

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("embed_cached: SPI_connect failed")));

	/* Cache lookup (uses SPI) */
	if (ndb_llm_cache_lookup(cache_key, max_age, &cached_text) && cached_text != NULL)
	{
		result = parse_vector_from_text(cached_text);
		SAFE_PFREE(cached_text);
	}

	/* If cache miss or parse failed, generate embedding */
	if (result == NULL)
	{
		StringInfoData query;
		int			spi_ret;
		bool		isnull;
		Datum		vec_datum;

		initStringInfo(&query);
		if (model_text != NULL)
		{
			appendStringInfo(&query, "SELECT embed_text(%s, %s)",
							 quote_literal_cstr(input_str), quote_literal_cstr(model_str));
		}
		else
		{
			appendStringInfo(&query, "SELECT embed_text(%s)",
							 quote_literal_cstr(input_str));
		}

		spi_ret = SPI_execute(query.data, true, 1);
		pfree(query.data);

		if (spi_ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			vec_datum = SPI_getbinval(SPI_tuptable->vals[0],
									  SPI_tuptable->tupdesc,
									  1,
									  &isnull);
			if (!isnull)
			{
				result = (Vector *) PG_DETOAST_DATUM_COPY(vec_datum);
			}
		}
	}

	/* Bulletproof fallback: if result is still NULL, return zero vector */
	if (result == NULL)
	{
		int			dim = 384;

		result = (Vector *) palloc0(VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
		SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
		result->dim = dim;
	}

	/* Store in cache if result is valid and not from cache */
	if (result != NULL && cached_text == NULL)
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

	SPI_finish();
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
	Jsonb	   *config_jsonb = NULL;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	int			r;
	bool		valid = true;
	int			spi_ret;
	Oid			argtypes[2];
	Datum		values[2];
	char		nulls[2];

	model_name = PG_GETARG_TEXT_PP(0);
	config_json = PG_GETARG_TEXT_PP(1);

	if (model_name == NULL || config_json == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("configure_embedding_model: model_name and config_json must not be NULL")));

	model_str = text_to_cstring(model_name);
	cfg_str = text_to_cstring(config_json);

	/* Parse and validate JSON */
	{
		Oid			jsonb_oid = JSONBOID;
		Oid			typinput;
		Oid			typioparam;

		getTypeInputInfo(jsonb_oid, &typinput, &typioparam);
		config_jsonb = (Jsonb *) DatumGetPointer(
												 OidFunctionCall3(typinput,
																  CStringGetDatum(cfg_str),
																  ObjectIdGetDatum(InvalidOid),
																  Int32GetDatum(-1)));

		if (config_jsonb == NULL)
		{
			pfree(model_str);
			pfree(cfg_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("configure_embedding_model: invalid JSON in config_json")));
		}
	}

	/* Validate JSON structure and values */
	it = JsonbIteratorInit(&config_jsonb->root);
	while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE && valid)
	{
		if (r == WJB_KEY)
		{
			char	   *key = NULL;
			int			key_len = v.val.string.len;

			key = pnstrdup(v.val.string.val, key_len);
			r = JsonbIteratorNext(&it, &v, false);
			if (r == WJB_VALUE)
			{
				if (strcmp(key, "batch_size") == 0)
				{
					if (v.type == jbvNumeric)
					{
						int32		batch_size = DatumGetInt32(
															   DirectFunctionCall1(numeric_int4,
																				   NumericGetDatum(v.val.numeric)));

						if (batch_size < 1 || batch_size > 10000)
						{
							valid = false;
							elog(WARNING,
								 "configure_embedding_model: batch_size must be between 1 and 10000, got %d",
								 batch_size);
						}
					}
					else
					{
						valid = false;
						elog(WARNING,
							 "configure_embedding_model: batch_size must be a number");
					}
				}
				else if (strcmp(key, "normalize") == 0)
				{
					if (v.type != jbvBool)
					{
						valid = false;
						elog(WARNING,
							 "configure_embedding_model: normalize must be a boolean");
					}
				}
				else if (strcmp(key, "device") == 0)
				{
					if (v.type != jbvString)
					{
						valid = false;
						elog(WARNING,
							 "configure_embedding_model: device must be a string");
					}
				}
				else if (strcmp(key, "timeout_ms") == 0)
				{
					if (v.type == jbvNumeric)
					{
						int32		timeout = DatumGetInt32(
															DirectFunctionCall1(numeric_int4,
																				NumericGetDatum(v.val.numeric)));

						if (timeout < 100 || timeout > 300000)
						{
							valid = false;
							elog(WARNING,
								 "configure_embedding_model: timeout_ms must be between 100 and 300000, got %d",
								 timeout);
						}
					}
					else
					{
						valid = false;
						elog(WARNING,
							 "configure_embedding_model: timeout_ms must be a number");
					}
				}
			}
			pfree(key);
		}
	}

	if (!valid)
	{
		pfree(model_str);
		pfree(cfg_str);
		PG_RETURN_BOOL(false);
	}

	/* Store configuration in catalog table using SPI */
	if (SPI_connect() != SPI_OK_CONNECT)
	{
		pfree(model_str);
		pfree(cfg_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("configure_embedding_model: SPI_connect failed")));
	}

	memset(nulls, ' ', sizeof(nulls));
	argtypes[0] = TEXTOID;
	argtypes[1] = JSONBOID;

	values[0] = CStringGetTextDatum(model_str);
	values[1] = JsonbPGetDatum(config_jsonb);

	spi_ret = SPI_execute_with_args(
									"INSERT INTO neurondb.neurondb_embedding_model_config "
									"(model_name, config_json, created_at, updated_at) "
									"VALUES ($1, $2, now(), now()) "
									"ON CONFLICT (model_name) DO UPDATE "
									"SET config_json = EXCLUDED.config_json, "
									"    updated_at = now()",
									2,
									argtypes,
									values,
									nulls,
									false,
									0);

	SPI_finish();

	if (spi_ret != SPI_OK_INSERT && spi_ret != SPI_OK_UPDATE)
	{
		pfree(model_str);
		pfree(cfg_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("configure_embedding_model: failed to store configuration")));
	}

	elog(DEBUG1,
		 "neurondb: configure_embedding_model: model='%s', config stored successfully",
		 model_str);

	pfree(model_str);
	pfree(cfg_str);

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(get_embedding_model_config);
/*
 * get_embedding_model_config
 *    Retrieve stored configuration for a model.
 *
 * model_name: TEXT
 * Returns: JSONB
 */
Datum
get_embedding_model_config(PG_FUNCTION_ARGS)
{
	text	   *model_name = NULL;
	char	   *model_str = NULL;
	Jsonb	   *result = NULL;

	model_name = PG_GETARG_TEXT_PP(0);
	if (model_name == NULL)
		PG_RETURN_NULL();

	model_str = text_to_cstring(model_name);
	result = get_embedding_model_config_internal(model_str);
	pfree(model_str);

	if (result == NULL)
		PG_RETURN_NULL();

	PG_RETURN_JSONB_P(result);
}

PG_FUNCTION_INFO_V1(list_embedding_model_configs);
/*
 * list_embedding_model_configs
 *    List all stored embedding model configurations.
 *
 * Returns: TABLE(model_name TEXT, config_json JSONB, created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ)
 */
Datum
list_embedding_model_configs(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	MemoryContext per_query_ctx;
	MemoryContext oldcontext;
	StringInfoData sql;
	int			spi_ret;
	int			i;

	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("set-valued function called in context that cannot accept a set")));

	if (rsinfo->expectedDesc == NULL)
	{
		TupleDesc	desc;

		desc = CreateTemplateTupleDesc(4);
		TupleDescInitEntry(desc, (AttrNumber) 1, "model_name", TEXTOID, -1, 0);
		TupleDescInitEntry(desc, (AttrNumber) 2, "config_json", JSONBOID, -1, 0);
		TupleDescInitEntry(desc, (AttrNumber) 3, "created_at", TIMESTAMPTZOID, -1, 0);
		TupleDescInitEntry(desc, (AttrNumber) 4, "updated_at", TIMESTAMPTZOID, -1, 0);
		tupdesc = BlessTupleDesc(desc);
		rsinfo->returnMode = SFRM_Materialize;
		rsinfo->setDesc = tupdesc;
		rsinfo->expectedDesc = tupdesc;
	}
	else
	{
		tupdesc = rsinfo->expectedDesc;
		rsinfo->returnMode = SFRM_Materialize;
		rsinfo->setDesc = tupdesc;
	}
	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

	/* Get work_mem GUC setting */
	{
		const char *work_mem_str = GetConfigOption("work_mem", true, false);
		int			work_mem_kb = 262144;	/* Default 256MB */

		if (work_mem_str)
		{
			work_mem_kb = atoi(work_mem_str);
			if (work_mem_kb <= 0)
				work_mem_kb = 262144;
		}
		tupstore = tuplestore_begin_heap(true, false, work_mem_kb);
	}
	rsinfo->setResult = tupstore;

	MemoryContextSwitchTo(oldcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("get_embedding_model_config: SPI_connect failed")));

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT model_name, config_json, created_at, updated_at "
					 "FROM neurondb.neurondb_embedding_model_config "
					 "ORDER BY model_name");

	spi_ret = SPI_execute(sql.data, true, 0);

	if (spi_ret == SPI_OK_SELECT)
	{
		for (i = 0; i < (int) SPI_processed; i++)
		{
			HeapTuple	spi_tuple = SPI_tuptable->vals[i];
			Datum		values[4];
			bool		nulls[4];
			int			j;
			MemoryContext tuple_context;

			memset(nulls, 0, sizeof(nulls));

			for (j = 0; j < 4; j++)
			{
				Datum		datum;
				bool		isnull;

				datum = SPI_getbinval(spi_tuple, SPI_tuptable->tupdesc, j + 1, &isnull);
				if (!isnull)
				{
					if (j == 1)
					{
						/* config_json: copy JSONB to tuple context */
						Jsonb	   *temp_jsonb;
						MemoryContext oldctx;

						temp_jsonb = DatumGetJsonbP(datum);
						tuple_context = MemoryContextSwitchTo(per_query_ctx);
						oldctx = MemoryContextSwitchTo(tuple_context);
						values[j] = JsonbPGetDatum((Jsonb *) PG_DETOAST_DATUM_COPY((Datum) temp_jsonb));
						MemoryContextSwitchTo(oldctx);
					}
					else
					{
						/* Other columns: copy to tuple context */
						tuple_context = MemoryContextSwitchTo(per_query_ctx);
						values[j] = datum;
					}
				}
				else
				{
					nulls[j] = true;
				}
			}

			tuplestore_putvalues(tupstore, tupdesc, values, nulls);
		}
	}

	pfree(sql.data);
	SPI_finish();

	return (Datum) 0;
}

PG_FUNCTION_INFO_V1(delete_embedding_model_config);
/*
 * delete_embedding_model_config
 *    Delete stored configuration for a model.
 *
 * model_name: TEXT
 * Returns: BOOLEAN
 */
Datum
delete_embedding_model_config(PG_FUNCTION_ARGS)
{
	text	   *model_name = NULL;
	char	   *model_str = NULL;
	int			spi_ret;
	Oid			argtypes[1];
	Datum		values[1];
	char		nulls[1];

	model_name = PG_GETARG_TEXT_PP(0);
	if (model_name == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("delete_embedding_model_config: model_name must not be NULL")));

	model_str = text_to_cstring(model_name);

	if (SPI_connect() != SPI_OK_CONNECT)
	{
		pfree(model_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("delete_embedding_model_config: SPI_connect failed")));
	}

	memset(nulls, ' ', sizeof(nulls));
	argtypes[0] = TEXTOID;
	values[0] = CStringGetTextDatum(model_str);

	spi_ret = SPI_execute_with_args(
									"DELETE FROM neurondb.neurondb_embedding_model_config WHERE model_name = $1",
									1,
									argtypes,
									values,
									nulls,
									false,
									0);

	SPI_finish();
	pfree(model_str);

	if (spi_ret == SPI_OK_DELETE)
		PG_RETURN_BOOL(true);
	else
		PG_RETURN_BOOL(false);
}
