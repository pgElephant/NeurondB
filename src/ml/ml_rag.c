/*-------------------------------------------------------------------------
 *
 * ml_rag.c
 *	  RAG (Retrieval-Augmented Generation) Pipeline for NeuronDB
 *
 * Implements text chunking, embedding, ranking, and data transformation for RAG support,
 * fully following PostgreSQL C coding standards and conventions.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *	  src/ml/ml_rag.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <ctype.h>

#include "neurondb.h"

PG_FUNCTION_INFO_V1(neurondb_chunk_text);
PG_FUNCTION_INFO_V1(neurondb_embed_text);
PG_FUNCTION_INFO_V1(neurondb_rank_documents);
PG_FUNCTION_INFO_V1(neurondb_transform_data);

/*
 * neurondb_chunk_text
 *	  Chunk text for RAG with configurable size, overlap, and separator.
 */
Datum
neurondb_chunk_text(PG_FUNCTION_ARGS)
{
	text	   *input_text;
	int32		chunk_size;
	int32		overlap;
	text	   *separator_text;
	char	   *input_str;
	char	   *separator;
	int			input_len;
	int			chunk_count = 0;
	Datum	   *chunk_datums;
	ArrayType  *result_array;
	int			i;
	int			start;
	int			end;
	int			chunk_len;
	int			max_chunks;
	int			sep_pos;

	CHECK_NARGS_RANGE(1, 4);
	input_text = PG_GETARG_TEXT_PP(0);
	chunk_size = PG_ARGISNULL(1) ? 512 : PG_GETARG_INT32(1);
	overlap = PG_ARGISNULL(2) ? 128 : PG_GETARG_INT32(2);
	separator_text = PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);

	/* Assert: Internal invariants */
	Assert(input_text != NULL);

	/* Defensive: Check NULL input */
	if (input_text == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("neurondb: chunk_text input text cannot be NULL")));

	/* Defensive: Validate chunk_size */
	if (chunk_size <= 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: chunk_text chunk_size must be "
				       "positive, got %d", chunk_size)));

	/* Defensive: Validate overlap */
	if (overlap < 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: chunk_text overlap must be "
				       "non-negative, got %d", overlap)));

	/* Defensive: Validate chunk_size > overlap */
	if (chunk_size <= overlap)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: chunk_text chunk_size (%d) must be "
				       "greater than overlap (%d)",
				       chunk_size, overlap)));

	/* Convert input and separator to C string */
	input_str = text_to_cstring(input_text);
	input_len = strlen(input_str);
	separator = separator_text ? text_to_cstring(separator_text)
		: pstrdup("\n\n");

	/* Defensive: Validate separator length */
	if (separator && strlen(separator) > 1024)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: chunk_text separator too long "
				       "(max 1024 characters)")));

	/* Conservative estimate of number of chunks */
	if (input_len == 0)
	{
		chunk_datums = (Datum *) palloc(sizeof(Datum));
		chunk_datums[0] = CStringGetTextDatum("");
		result_array = construct_array(chunk_datums, 1, TEXTOID, -1, false,
									   TYPALIGN_INT);
		pfree(chunk_datums);
		if (separator_text)
			pfree(separator);
		pfree(input_str);
		PG_RETURN_ARRAYTYPE_P(result_array);
	}

	/* Defensive: Check for overflow in max_chunks calculation */
	if (chunk_size - overlap <= 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: chunk_text chunk_size - overlap "
				       "must be positive")));

	max_chunks = (input_len + chunk_size - overlap - 1) / (chunk_size - overlap);
	if (max_chunks < 1)
		max_chunks = 1;

	/* Defensive: Limit maximum chunks to prevent excessive memory allocation */
	if (max_chunks > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: chunk_text too many chunks (%d), "
				       "maximum 100000", max_chunks)));

	chunk_datums = (Datum *) palloc0(sizeof(Datum) * max_chunks);

	start = 0;
	while (start < input_len)
	{
		end = start + chunk_size;
		if (end > input_len)
			end = input_len;

		/* Try to split on the separator, if provided and found */
		if (separator && strlen(separator) > 0 && end < input_len)
		{
			size_t		sep_len = strlen(separator);

			sep_pos = -1;
			/* Defensive: Check bounds before strncmp */
			for (i = end; i > start && i >= (int) sep_len; i--)
			{
				if (strncmp(input_str + i - sep_len, separator, sep_len) == 0)
				{
					sep_pos = i;
					break;
				}
			}
			if (sep_pos > start)
				end = sep_pos;
		}

		chunk_len = end - start;
		if (chunk_len <= 0)
			break;

		/* Defensive: Check chunk_count bounds */
		if (chunk_count >= max_chunks)
			ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					errmsg("neurondb: chunk_text chunk count "
					       "exceeded maximum")));

		{
			char	   *chunk_buf = (char *) palloc(chunk_len + 1);

			memcpy(chunk_buf, input_str + start, chunk_len);
			chunk_buf[chunk_len] = '\0';
			chunk_datums[chunk_count++] = CStringGetTextDatum(chunk_buf);
			pfree(chunk_buf);
		}

		if (end == input_len)
			break;

		start = end - overlap;
		if (start < 0)
			start = 0;
	}
	/* Defensive: Ensure at least one chunk */
	if (chunk_count == 0)
	{
		chunk_datums[0] = CStringGetTextDatum("");
		chunk_count = 1;
	}

	result_array = construct_array(chunk_datums, chunk_count, TEXTOID, -1, false,
								   TYPALIGN_INT);
	pfree(chunk_datums);
	if (separator_text)
		pfree(separator);
	pfree(input_str);
	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * neurondb_embed_text
 *	  Generate vector embeddings for text with mock GPU support.
 */
Datum
neurondb_embed_text(PG_FUNCTION_ARGS)
{
	text	   *model_text;
	text	   *input_text;
	bool		use_gpu;
	char	   *model_name;
	char	   *input_str;
	int			input_len;
	const int	embedding_dim = 384;
	float	   *embedding_data;
	Vector	   *result;
	uint32		hash = 5381;
	size_t		model_name_len;
	int			i;

	CHECK_NARGS_RANGE(2, 3);
	model_text = PG_GETARG_TEXT_PP(0);
	input_text = PG_GETARG_TEXT_PP(1);
	use_gpu = PG_ARGISNULL(2) ? true : PG_GETARG_BOOL(2);

	/* Assert: Internal invariants */
	Assert(model_text != NULL);
	Assert(input_text != NULL);

	/* Defensive: Check NULL inputs */
	if (model_text == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("neurondb: embed_text model text cannot be NULL")));

	if (input_text == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("neurondb: embed_text input text cannot be NULL")));

	model_name = text_to_cstring(model_text);
	input_str = text_to_cstring(input_text);
	input_len = strlen(input_str);
	model_name_len = strlen(model_name);

	/* Defensive: Validate input length */
	if (input_len > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: embed_text input text too long "
				       "(max 1000000 characters)")));

	/* Defensive: Validate model name length */
	if (model_name_len == 0 || model_name_len > 256)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: embed_text invalid model name length")));

	elog(DEBUG2,
	     "neurondb: embed_text model='%s', use_gpu=%d",
	     model_name, use_gpu);

	embedding_data = (float *) palloc(sizeof(float) * embedding_dim);
	if (embedding_data == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: embed_text failed to allocate "
				       "embedding data")));

	/*
	 * Generate a deterministic pseudo-random embedding vector.
	 * Uses a djb2-style hash seeded by input and model for plausible output.
	 */
	for (i = 0; i < input_len; i++)
		hash = ((hash << 5) + hash) ^ (unsigned char) input_str[i];

	for (i = 0; i < embedding_dim; i++)
	{
		hash = ((hash << 5) + hash)
			^ (unsigned char) (model_name[i % model_name_len]);
		embedding_data[i] = ((float) ((hash % 2000) - 1000)) / 1000.0f;

		/* Defensive: Validate embedding value */
		if (isnan(embedding_data[i]) || isinf(embedding_data[i]))
			ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					errmsg("neurondb: embed_text embedding value "
					       "is NaN or Infinity at index %d", i)));
	}

	result = (Vector *) palloc(VARHDRSZ + sizeof(int16) * 2 +
								sizeof(float) * embedding_dim);
	if (result == NULL)
	{
		pfree(embedding_data);
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: embed_text failed to allocate vector")));
	}

	SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 +
				sizeof(float) * embedding_dim);
	result->dim = embedding_dim;
	result->unused = 0;
	memcpy(result->data, embedding_data, sizeof(float) * embedding_dim);

	pfree(embedding_data);
	pfree(model_name);
	pfree(input_str);

	PG_RETURN_POINTER(result);
}

/*
 * neurondb_rank_documents
 *	  Rank (rerank) array of documents based on query using simple heuristics
 *	  (supports: bm25, cosine, edit_distance ranking method names for demonstration).
 */
static float
simple_bm25(const char *query, const char *doc)
{
	/*
	 * Naive: score = matches of every query word in doc, normalized.
	 */
	float		score = 0.0f;
	int			total = 0;
	char	   *doc_lc;
	char	   *query_lc;
	char	   *tok;
	char	   *saveptr_q;
	bool		found = false;
	int			i;

	/* Defensive: Check NULL inputs */
	if (query == NULL || doc == NULL)
		return 0.0f;

	doc_lc = pstrdup(doc);
	query_lc = pstrdup(query);

	/* Defensive: Check allocation */
	if (doc_lc == NULL || query_lc == NULL)
	{
		if (doc_lc)
			pfree(doc_lc);
		if (query_lc)
			pfree(query_lc);
		return 0.0f;
	}

	/* Lowercase for case-insensitive matching */
	for (i = 0; doc_lc[i]; i++)
		doc_lc[i] = tolower((unsigned char) doc_lc[i]);
	for (i = 0; query_lc[i]; i++)
		query_lc[i] = tolower((unsigned char) query_lc[i]);

	for (tok = strtok_r(query_lc, " \t\r\n", &saveptr_q); tok != NULL;
		 tok = strtok_r(NULL, " \t\r\n", &saveptr_q))
	{
		total++;
		found = false;
		if (strstr(doc_lc, tok) != NULL)
			found = true;
		if (found)
			score += 1.0f;

		/* Defensive: Prevent overflow */
		if (score > 1e6f)
			break;
	}

	if (total > 0)
		score /= (float) total;

	pfree(doc_lc);
	pfree(query_lc);

	/* Defensive: Validate score */
	if (isnan(score) || isinf(score))
		return 0.5f;

	return score + 0.5f;		/* Base for sort stability */
}

static float
simple_cosine(const char *query, const char *doc)
{
	/*
	 * Naive "cosine": overlap count for tokens.
	 */
	char	   *doc_lc;
	char	   *query_lc;
	int			score = 0;
	int			qcount = 0;
	char	   *tok_q;
	char	   *saveptr_q;
	char	   *tok_d;
	char	   *saveptr_d;
	int			i;

	/* Defensive: Check NULL inputs */
	if (query == NULL || doc == NULL)
		return 0.0f;

	doc_lc = pstrdup(doc);
	query_lc = pstrdup(query);

	/* Defensive: Check allocation */
	if (doc_lc == NULL || query_lc == NULL)
	{
		if (doc_lc)
			pfree(doc_lc);
		if (query_lc)
			pfree(query_lc);
		return 0.0f;
	}

	for (i = 0; doc_lc[i]; i++)
		doc_lc[i] = tolower((unsigned char) doc_lc[i]);
	for (i = 0; query_lc[i]; i++)
		query_lc[i] = tolower((unsigned char) query_lc[i]);
	for (tok_q = strtok_r(query_lc, " \t\r\n", &saveptr_q); tok_q != NULL;
		 tok_q = strtok_r(NULL, " \t\r\n", &saveptr_q))
	{
		qcount++;
		/* Here, check if token appears in doc */
		for (tok_d = strtok_r(doc_lc, " \t\r\n", &saveptr_d);
			 tok_d != NULL;
			 tok_d = strtok_r(NULL, " \t\r\n", &saveptr_d))
		{
			if (strcmp(tok_d, tok_q) == 0)
			{
				score++;
				break;
			}
		}
		/* important: re-initialize for next q token */
		strcpy(doc_lc, doc);
		for (i = 0; doc_lc[i]; i++)
			doc_lc[i] = tolower((unsigned char) doc_lc[i]);
	}

	if (qcount == 0)
		qcount = 1;

	pfree(doc_lc);
	pfree(query_lc);

	{
		float		result = ((float) score / (float) qcount) + 0.5f;

		/* Defensive: Validate result */
		if (isnan(result) || isinf(result))
			return 0.5f;

		return result;		/* stabilization */
	}
}

static int
levenshtein(const char *s1, const char *s2)
{
	int			len1;
	int			len2;
	int		   *v0;
	int		   *v1;
	int			i;
	int			j;
	int			cost;
	int			min1;
	int			min2;
	int			min3;
	int			result;

	/* Defensive: Check NULL inputs */
	if (s1 == NULL || s2 == NULL)
		return -1;

	len1 = strlen(s1);
	len2 = strlen(s2);

	/* Defensive: Limit string length to prevent excessive memory */
	if (len1 > 10000 || len2 > 10000)
		return -1;

	v0 = (int *) palloc(sizeof(int) * (len2 + 1));
	v1 = (int *) palloc(sizeof(int) * (len2 + 1));

	if (v0 == NULL || v1 == NULL)
	{
		if (v0)
			pfree(v0);
		if (v1)
			pfree(v1);
		return -1;
	}

	for (j = 0; j <= len2; j++)
		v0[j] = j;

	for (i = 0; i < len1; i++)
	{
		v1[0] = i + 1;
		for (j = 0; j < len2; j++)
		{
			cost = (tolower((unsigned char) s1[i]) ==
					tolower((unsigned char) s2[j])) ? 0 : 1;
			min1 = v1[j] + 1;
			min2 = v0[j + 1] + 1;
			min3 = v0[j] + cost;
			if (min1 > min2)
				min1 = min2;
			if (min1 > min3)
				min1 = min3;
			v1[j + 1] = min1;
		}
		memcpy(v0, v1, sizeof(int) * (len2 + 1));
	}

	result = v1[len2];
	pfree(v0);
	pfree(v1);
	return result;
}

static float
simple_edit_distance(const char *query, const char *doc)
{
	/*
	 * Return normalized similarity based on edit distance:
	 * similarity = 1.0 - (edit_distance / max(len1, len2))
	 */
	int			lenq;
	int			lend;
	int			maxl;
	int			edit;
	float		result;

	/* Defensive: Check NULL inputs */
	if (query == NULL || doc == NULL)
		return 0.0f;

	lenq = strlen(query);
	lend = strlen(doc);
	maxl = (lenq > lend) ? lenq : lend;

	if (maxl == 0)
		return 1.0f;

	edit = levenshtein(query, doc);

	/* Defensive: Check for error from levenshtein */
	if (edit < 0)
		return 0.0f;

	result = 1.0f - ((float) edit / (float) maxl);

	/* Defensive: Validate result */
	if (isnan(result) || isinf(result))
		return 0.0f;

	return result;
}

typedef struct
{
	char *doc;
	float score;
	int rawidx;
} doc_with_score;

static int
doc_with_score_cmp(const void *a, const void *b)
{
	const doc_with_score *da = (const doc_with_score *)a;
	const doc_with_score *db = (const doc_with_score *)b;
	if (da->score > db->score)
		return -1;
	if (da->score < db->score)
		return 1;
	return da->rawidx - db->rawidx;
}

Datum
neurondb_rank_documents(PG_FUNCTION_ARGS)
{
	text	   *query_text;
	ArrayType  *documents_array;
	text	   *algorithm_text;
	char	   *query;
	char	   *algorithm;
	int			ndims;
	int			nelems;
	int		   *dims;
	Datum	   *elem_values;
	bool	   *elem_nulls;
	doc_with_score *ranklist;
	int			i;
	int			limit = 10;
	int			count = 0;
	StringInfoData json_result;

	CHECK_NARGS_RANGE(2, 3);
	query_text = PG_GETARG_TEXT_PP(0);
	documents_array = PG_GETARG_ARRAYTYPE_P(1);
	algorithm_text = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);

	/* Assert: Internal invariants */
	Assert(query_text != NULL);
	Assert(documents_array != NULL);

	/* Defensive: Check NULL inputs */
	if (query_text == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("neurondb: rank_documents query text cannot be NULL")));

	if (documents_array == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("neurondb: rank_documents documents array "
				       "cannot be NULL")));

	query = text_to_cstring(query_text);
	algorithm = algorithm_text ? text_to_cstring(algorithm_text)
		: pstrdup("bm25");

	/* Defensive: Check allocation */
	if (query == NULL || algorithm == NULL)
	{
		if (query)
			pfree(query);
		if (algorithm)
			pfree(algorithm);
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: rank_documents failed to allocate memory")));
	}

	ndims = ARR_NDIM(documents_array);
	dims = ARR_DIMS(documents_array);
	nelems = ArrayGetNItems(ndims, dims);

	/* Defensive: Validate array dimensions */
	if (ndims < 0 || ndims > 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: rank_documents invalid array "
				       "dimensions %d", ndims)));

	/* Defensive: Validate array size */
	if (nelems < 0 || nelems > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: rank_documents array too large "
				       "(%d elements, max 100000)", nelems)));

	deconstruct_array(documents_array, TEXTOID, -1, false, TYPALIGN_INT,
					   &elem_values, &elem_nulls, &nelems);

	/* Defensive: Check deconstruct_array result */
	if (elem_values == NULL || elem_nulls == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: rank_documents failed to "
				       "deconstruct array")));

	ranklist = (doc_with_score *) palloc0(nelems * sizeof(doc_with_score));
	if (ranklist == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: rank_documents failed to allocate ranklist")));

	for (i = 0; i < nelems; i++)
	{
		if (elem_nulls[i])
		{
			ranklist[i].doc = NULL;
			ranklist[i].score = -FLT_MAX;
			ranklist[i].rawidx = i;
			continue;
		}
		ranklist[i].doc = TextDatumGetCString(elem_values[i]);
		if (ranklist[i].doc == NULL)
		{
			ranklist[i].doc = NULL;
			ranklist[i].score = -FLT_MAX;
			ranklist[i].rawidx = i;
			continue;
		}

		if (strcmp(algorithm, "bm25") == 0)
			ranklist[i].score = simple_bm25(query, ranklist[i].doc);
		else if (strcmp(algorithm, "cosine") == 0)
			ranklist[i].score = simple_cosine(query, ranklist[i].doc);
		else if (strcmp(algorithm, "edit_distance") == 0)
			ranklist[i].score = simple_edit_distance(query, ranklist[i].doc);
		else
			ranklist[i].score = simple_bm25(query, ranklist[i].doc);

		/* Defensive: Validate score */
		if (isnan(ranklist[i].score) || isinf(ranklist[i].score))
			ranklist[i].score = -FLT_MAX;

		ranklist[i].rawidx = i;
	}

	qsort(ranklist, nelems, sizeof(doc_with_score), doc_with_score_cmp);

	initStringInfo(&json_result);
	appendStringInfoString(&json_result, "{\"ranked\": [");

	count = 0;
	for (i = 0; i < nelems && count < limit; i++)
	{
		if (ranklist[i].doc)
		{
			if (count != 0)
				appendStringInfoString(&json_result, ", ");
			appendStringInfo(&json_result,
				"{\"document\": %s, \"score\": %.4f, \"rank\": "
				"%d}",
				quote_literal_cstr(ranklist[i].doc),
				ranklist[i].score,
				count + 1);
			count++;
		}
	}
	appendStringInfoString(&json_result, "]}");

	/* cleanup */
	for (i = 0; i < nelems; i++)
	{
		if (ranklist[i].doc)
			pfree(ranklist[i].doc);
	}
	pfree(ranklist);
	if (algorithm_text)
		pfree(algorithm);
	pfree(query);

	PG_RETURN_JSONB_P(DatumGetJsonbP(DirectFunctionCall1(
		jsonb_in, CStringGetDatum(json_result.data))));
}

/*
 * neurondb_transform_data
 *	  Apply transformation (normalize, standardize, min_max) to float8 array.
 */

Datum
neurondb_transform_data(PG_FUNCTION_ARGS)
{
	text	   *pipeline_text;
	ArrayType  *input_array;
	char	   *pipeline_name;
	int			ndims;
	int			nelems;
	int		   *dims;
	Oid			element_type;
	Datum	   *elem_values = NULL;
	bool	   *elem_nulls = NULL;
	float8	   *transformed_data;
	Datum	   *result_datums;
	ArrayType  *result_array;
	float8		sum = 0.0;
	float8		sum_sq = 0.0;
	float8		min_v = DBL_MAX;
	float8		max_v = -DBL_MAX;
	float8		mean;
	float8		stddev;
	float8		range;
	float8		x;
	float8		norm;
	int			i;

	CHECK_NARGS(2);
	pipeline_text = PG_GETARG_TEXT_PP(0);
	input_array = PG_GETARG_ARRAYTYPE_P(1);

	/* Assert: Internal invariants */
	Assert(pipeline_text != NULL);
	Assert(input_array != NULL);

	/* Defensive: Check NULL inputs */
	if (pipeline_text == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("neurondb: transform_data pipeline text cannot be NULL")));

	if (input_array == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("neurondb: transform_data input array cannot be NULL")));

	pipeline_name = text_to_cstring(pipeline_text);
	if (pipeline_name == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: transform_data failed to allocate "
				       "pipeline name")));

	ndims = ARR_NDIM(input_array);
	dims = ARR_DIMS(input_array);
	nelems = ArrayGetNItems(ndims, dims);
	element_type = ARR_ELEMTYPE(input_array);

	/* Defensive: Validate array dimensions */
	if (ndims < 0 || ndims > 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: transform_data invalid array "
				       "dimensions %d", ndims)));

	/* Defensive: Validate array size */
	if (nelems < 0 || nelems > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: transform_data array too large "
				       "(%d elements, max 1000000)", nelems)));

	deconstruct_array(input_array, element_type, sizeof(float8),
					   FLOAT8PASSBYVAL, 'd', &elem_values, &elem_nulls, &nelems);

	/* Defensive: Check deconstruct_array result */
	if (elem_values == NULL || elem_nulls == NULL)
	{
		pfree(pipeline_name);
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: transform_data failed to "
				       "deconstruct array")));
	}

	if (nelems == 0)
	{
		pfree(pipeline_name);
		result_array = construct_empty_array(FLOAT8OID);
		PG_RETURN_ARRAYTYPE_P(result_array);
	}

	/* Stats for transformation */
	for (i = 0; i < nelems; i++)
	{
		if (!elem_nulls[i])
		{
			x = DatumGetFloat8(elem_values[i]);

			/* Defensive: Check for NaN/Inf */
			if (isnan(x) || isinf(x))
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: transform_data NaN or "
						       "Infinity at index %d", i)));

			sum += x;
			sum_sq += x * x;

			/* Defensive: Check for overflow */
			if (isinf(sum) || isnan(sum) || isinf(sum_sq) || isnan(sum_sq))
				ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
						errmsg("neurondb: transform_data overflow "
						       "in statistics calculation")));

			if (x < min_v)
				min_v = x;
			if (x > max_v)
				max_v = x;
		}
	}

	mean = sum / (float8) nelems;
	stddev = sqrt((sum_sq / (float8) nelems) - (mean * mean));

	/* Defensive: Validate mean and stddev */
	if (isnan(mean) || isinf(mean) || isnan(stddev) || isinf(stddev))
		ereport(ERROR,
			(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				errmsg("neurondb: transform_data invalid statistics "
				       "(mean or stddev is NaN/Inf)")));

	transformed_data = (float8 *) palloc0(sizeof(float8) * nelems);
	result_datums = (Datum *) palloc0(sizeof(Datum) * nelems);

	if (transformed_data == NULL || result_datums == NULL)
	{
		if (transformed_data)
			pfree(transformed_data);
		if (result_datums)
			pfree(result_datums);
		pfree(pipeline_name);
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: transform_data failed to allocate "
				       "result arrays")));
	}

	if (strcmp(pipeline_name, "normalize") == 0)
	{
		/* L2 normalization */
		norm = sqrt(sum_sq);

		/* Defensive: Validate norm */
		if (isnan(norm) || isinf(norm))
			norm = 0.0;

		if (norm <= 0.0)
		{
			for (i = 0; i < nelems; i++)
				transformed_data[i] = 0.0;
		}
		else
		{
			for (i = 0; i < nelems; i++)
			{
				if (elem_nulls[i])
					transformed_data[i] = 0.0;
				else
				{
					x = DatumGetFloat8(elem_values[i]);
					transformed_data[i] = x / norm;

					/* Defensive: Validate result */
					if (isnan(transformed_data[i]) || isinf(transformed_data[i]))
						transformed_data[i] = 0.0;
				}
			}
		}
	}
	else if (strcmp(pipeline_name, "standardize") == 0)
	{
		/* Z-score */
		if (stddev > 0.0)
		{
			for (i = 0; i < nelems; i++)
			{
				if (elem_nulls[i])
					transformed_data[i] = 0.0;
				else
				{
					x = DatumGetFloat8(elem_values[i]);
					transformed_data[i] = (x - mean) / stddev;

					/* Defensive: Validate result */
					if (isnan(transformed_data[i]) || isinf(transformed_data[i]))
						transformed_data[i] = 0.0;
				}
			}
		}
		else
		{
			for (i = 0; i < nelems; i++)
				transformed_data[i] = 0.0;
		}
	}
	else if (strcmp(pipeline_name, "min_max") == 0)
	{
		/* Min-max scaling [0, 1] */
		range = max_v - min_v;

		/* Defensive: Validate range */
		if (isnan(range) || isinf(range))
			range = 0.0;

		if (range > 0.0)
		{
			for (i = 0; i < nelems; i++)
			{
				if (elem_nulls[i])
					transformed_data[i] = 0.0;
				else
				{
					x = DatumGetFloat8(elem_values[i]);
					transformed_data[i] = (x - min_v) / range;

					/* Defensive: Validate result */
					if (isnan(transformed_data[i]) || isinf(transformed_data[i]))
						transformed_data[i] = 0.5;
				}
			}
		}
		else
		{
			for (i = 0; i < nelems; i++)
				transformed_data[i] = 0.5;
		}
	}
	else
	{
		pfree(transformed_data);
		pfree(result_datums);
		pfree(pipeline_name);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: transform_data unsupported "
				       "transformation pipeline \"%s\"",
				       pipeline_name),
				errhint("Supported pipelines: normalize, standardize, min_max")));
	}

	for (i = 0; i < nelems; i++)
		result_datums[i] = Float8GetDatum(transformed_data[i]);

	result_array = construct_array(result_datums, nelems, FLOAT8OID,
								   sizeof(float8), FLOAT8PASSBYVAL, 'd');

	pfree(result_datums);
	pfree(transformed_data);
	pfree(pipeline_name);

	PG_RETURN_ARRAYTYPE_P(result_array);
}
