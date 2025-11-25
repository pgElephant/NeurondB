/*-------------------------------------------------------------------------
 *
 * sparse_vector_types.c
 *    Learned sparse vector type for SPLADE/ColBERTv2 retrieval
 *
 * Implements sparse_vector type optimized for learned sparse retrieval
 * models like SPLADE and ColBERTv2. Stores sparse representations with
 * token IDs and learned weights.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/types/sparse_vector_types.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/varlena.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "neurondb_sparse.h"
#include <string.h>
#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"

/*
 * sparse_vector_in: Parse sparse vector from text
 * Format: "{vocab_size:30522, model:SPLADE, tokens:[100,200,300], weights:[0.5,0.8,0.3]}"
 */
PG_FUNCTION_INFO_V1(sparse_vector_in);
Datum
sparse_vector_in(PG_FUNCTION_ARGS)
{
	char *str = PG_GETARG_CSTRING(0);
	SparseVector *result;
	int32 vocab_size = 0;
	int32 nnz = 0;
	uint16 model_type = 1; /* Default to SPLADE */
	int32 *token_ids = NULL;
	float4 *weights = NULL;
	char *ptr;
	int capacity = 16;

	/* Simple parser - in production, use proper JSON parsing */
	ptr = str;
	while (*ptr && *ptr != '{')
		ptr++;

	/* Parse vocab_size */
	if (strstr(ptr, "vocab_size:") != NULL)
	{
		sscanf(strstr(ptr, "vocab_size:"), "vocab_size:%d", &vocab_size);
	}

	/* Parse model type */
	if (strstr(ptr, "model:BM25") != NULL)
		model_type = 0;
	else if (strstr(ptr, "model:SPLADE") != NULL)
		model_type = 1;
	else if (strstr(ptr, "model:ColBERTv2") != NULL)
		model_type = 2;

	/* Allocate temporary arrays */
	token_ids = (int32 *)palloc(sizeof(int32) * capacity);
	weights = (float4 *)palloc(sizeof(float4) * capacity);

	/* Parse tokens array */
	if (strstr(ptr, "tokens:[") != NULL)
	{
		char *tokens_start = strstr(ptr, "tokens:[") + 8;
		char *tokens_end = strchr(tokens_start, ']');
		char *tok_ptr = tokens_start;

		while (tok_ptr < tokens_end && *tok_ptr)
		{
			if (nnz >= capacity)
			{
				capacity *= 2;
				token_ids = (int32 *)repalloc(token_ids,
					sizeof(int32) * capacity);
				weights = (float4 *)repalloc(weights,
					sizeof(float4) * capacity);
			}

			while (*tok_ptr == ' ' || *tok_ptr == ',')
				tok_ptr++;

			if (*tok_ptr == ']')
				break;

			token_ids[nnz] = atoi(tok_ptr);
			while (*tok_ptr && *tok_ptr != ',' && *tok_ptr != ']')
				tok_ptr++;
			nnz++;
		}
	}

	/* Parse weights array */
	if (strstr(ptr, "weights:[") != NULL)
	{
		char *weights_start = strstr(ptr, "weights:[") + 9;
		char *weights_end = strchr(weights_start, ']');
		char *wgt_ptr = weights_start;
		int idx = 0;

		while (wgt_ptr < weights_end && *wgt_ptr && idx < nnz)
		{
			while (*wgt_ptr == ' ' || *wgt_ptr == ',')
				wgt_ptr++;

			if (*wgt_ptr == ']')
				break;

			weights[idx] = strtof(wgt_ptr, NULL);
			while (*wgt_ptr && *wgt_ptr != ',' && *wgt_ptr != ']')
				wgt_ptr++;
			idx++;
		}
	}

	if (nnz == 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				errmsg("sparse_vector must have at least one token")));

	if (vocab_size == 0)
		vocab_size = 30522; /* Default BERT vocab size */

	/* Build result */
	result = (SparseVector *)palloc0(SPARSE_VEC_SIZE(nnz));
	SET_VARSIZE(result, SPARSE_VEC_SIZE(nnz));
	result->vocab_size = vocab_size;
	result->nnz = nnz;
	result->model_type = model_type;

	memcpy(SPARSE_VEC_TOKEN_IDS(result), token_ids, sizeof(int32) * nnz);
	memcpy(SPARSE_VEC_WEIGHTS(result), weights, sizeof(float4) * nnz);

	NDB_SAFE_PFREE_AND_NULL(token_ids);
	NDB_SAFE_PFREE_AND_NULL(weights);

	PG_RETURN_POINTER(result);
}

/*
 * sparse_vector_out: Convert sparse vector to text
 */
PG_FUNCTION_INFO_V1(sparse_vector_out);
Datum
sparse_vector_out(PG_FUNCTION_ARGS)
{
	SparseVector *sv = (SparseVector *)PG_GETARG_POINTER(0);
	StringInfoData buf;
	int32 *token_ids;
	float4 *weights;
	int i;
	const char *model_name;

	if (sv == NULL)
		PG_RETURN_CSTRING(pstrdup("NULL"));

	token_ids = SPARSE_VEC_TOKEN_IDS(sv);
	weights = SPARSE_VEC_WEIGHTS(sv);

	switch (sv->model_type)
	{
	case 0:
		model_name = "BM25";
		break;
	case 1:
		model_name = "SPLADE";
		break;
	case 2:
		model_name = "ColBERTv2";
		break;
	default:
		model_name = "UNKNOWN";
		break;
	}

	initStringInfo(&buf);
	appendStringInfo(&buf,
		"{vocab_size:%d, model:%s, tokens:[",
		sv->vocab_size,
		model_name);

	for (i = 0; i < sv->nnz; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%d", token_ids[i]);
	}

	appendStringInfoString(&buf, "], weights:[");

	for (i = 0; i < sv->nnz; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%g", weights[i]);
	}

	appendStringInfoChar(&buf, '}');

	PG_RETURN_CSTRING(buf.data);
}

/*
 * sparse_vector_recv: Binary receive
 */
PG_FUNCTION_INFO_V1(sparse_vector_recv);
Datum
sparse_vector_recv(PG_FUNCTION_ARGS)
{
	StringInfo buf = (StringInfo)PG_GETARG_POINTER(0);
	SparseVector *result;
	int32 vocab_size;
	int32 nnz;
	uint16 model_type;
	int i;

	vocab_size = pq_getmsgint(buf, sizeof(int32));
	nnz = pq_getmsgint(buf, sizeof(int32));
	model_type = pq_getmsgint(buf, sizeof(uint16));

	if (vocab_size <= 0 || vocab_size > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				errmsg("invalid sparse_vector vocab_size: %d",
					vocab_size)));

	if (nnz < 0 || nnz > 10000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				errmsg("invalid sparse_vector nnz: %d", nnz)));

	result = (SparseVector *)palloc0(SPARSE_VEC_SIZE(nnz));
	SET_VARSIZE(result, SPARSE_VEC_SIZE(nnz));
	result->vocab_size = vocab_size;
	result->nnz = nnz;
	result->model_type = model_type;

	for (i = 0; i < nnz; i++)
		SPARSE_VEC_TOKEN_IDS(result)[i] = pq_getmsgint(buf, sizeof(int32));

	for (i = 0; i < nnz; i++)
		SPARSE_VEC_WEIGHTS(result)[i] = pq_getmsgfloat4(buf);

	PG_RETURN_POINTER(result);
}

/*
 * sparse_vector_send: Binary send
 */
PG_FUNCTION_INFO_V1(sparse_vector_send);
Datum
sparse_vector_send(PG_FUNCTION_ARGS)
{
	SparseVector *sv = (SparseVector *)PG_GETARG_POINTER(0);
	StringInfoData buf;
	int32 *token_ids;
	float4 *weights;
	int i;

	token_ids = SPARSE_VEC_TOKEN_IDS(sv);
	weights = SPARSE_VEC_WEIGHTS(sv);

	pq_begintypsend(&buf);
	pq_sendint(&buf, sv->vocab_size, sizeof(int32));
	pq_sendint(&buf, sv->nnz, sizeof(int32));
	pq_sendint(&buf, sv->model_type, sizeof(uint16));

	for (i = 0; i < sv->nnz; i++)
		pq_sendint(&buf, token_ids[i], sizeof(int32));

	for (i = 0; i < sv->nnz; i++)
		pq_sendfloat4(&buf, weights[i]);

	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/*
 * sparse_vector_dot_product: Compute dot product between two sparse vectors
 */
PG_FUNCTION_INFO_V1(sparse_vector_dot_product);
Datum
sparse_vector_dot_product(PG_FUNCTION_ARGS)
{
	SparseVector *a = (SparseVector *)PG_GETARG_POINTER(0);
	SparseVector *b = (SparseVector *)PG_GETARG_POINTER(1);
	int32 *a_tokens, *b_tokens;
	float4 *a_weights, *b_weights;
	float4 result = 0.0;
	int i, j;

	if (a == NULL || b == NULL)
		PG_RETURN_FLOAT4(0.0);

	a_tokens = SPARSE_VEC_TOKEN_IDS(a);
	b_tokens = SPARSE_VEC_TOKEN_IDS(b);
	a_weights = SPARSE_VEC_WEIGHTS(a);
	b_weights = SPARSE_VEC_WEIGHTS(b);

	/* Compute dot product by matching token IDs */
	for (i = 0; i < a->nnz; i++)
	{
		for (j = 0; j < b->nnz; j++)
		{
			if (a_tokens[i] == b_tokens[j])
			{
				result += a_weights[i] * b_weights[j];
				break;
			}
		}
	}

	PG_RETURN_FLOAT4(result);
}

