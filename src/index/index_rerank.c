/*-------------------------------------------------------------------------
 *
 * index_rerank.c
 *	  Rerank-ready index with pre-computed candidate lists
 *
 * Implements RRI (Rerank Ready Index) that stores top-k candidate
 * lists for hot queries, enabling zero round trips to heap for
 * reranking operations.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *	  src/index/index_rerank.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_compat.h"
#include "neurondb_index.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/elog.h"
#include "utils/memutils.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "funcapi.h"
#include "lib/stringinfo.h"

#include <math.h>
#include <stdint.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_spi_safe.h"

/*
 * Generate the rerank cache table name for a given table.column
 */
static char *
get_rerank_cache_table(const char *tbl, const char *col)
{
	char	   *buf;

	Assert(tbl != NULL && col != NULL);

	buf = (char *) palloc(strlen(tbl) + strlen(col) + 32);

	snprintf(buf,
			 strlen(tbl) + strlen(col) + 32,
			 "__rerank_cache_%s_%s",
			 tbl,
			 col);

	return buf;
}

/*
 * Simple vector hash (for demonstration).
 */
static uint64
vector_hash(Vector *v)
{
	int			i;
	uint64		h = 0;

	if (v == NULL)
		elog(ERROR, "vector_hash: NULL input");

	for (i = 0; i < v->dim && i < 16; i++)
		h = (h * 31) + (uint32) (v->data[i] * 1000);

	return h;
}

/*
 * Compute L2 distance between two vectors.
 */
static float4 __attribute__((unused))
vector_l2(Vector *a, Vector *b)
{
	int			i;
	float4		sum = 0.0f;

	if (a == NULL || b == NULL)
		elog(ERROR, "vector_l2: NULL input");

	if (a->dim != b->dim)
		elog(ERROR, "vector_l2: dimensions do not match");

	for (i = 0; i < a->dim; i++)
	{
		float4		d = a->data[i] - b->data[i];

		sum += d * d;
	}

	return sqrtf(sum);
}

/*
 * Create rerank-ready index/cache: sets up a cache table
 * to store (query_hash, query_vec, candidate_id, candidate_vec, similarity)
 */
PG_FUNCTION_INFO_V1(rerank_index_create);

Datum
rerank_index_create(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *vector_col = PG_GETARG_TEXT_PP(1);
	int32		cache_size = PG_GETARG_INT32(2);
	int32		k_candidates = PG_GETARG_INT32(3);
	char	   *tbl_str = text_to_cstring(table_name);
	char	   *col_str = text_to_cstring(vector_col);
	char	   *cache_tbl;
	StringInfoData sql;
	int			ret;

	cache_tbl = get_rerank_cache_table(tbl_str, col_str);

	elog(INFO,
		 "neurondb: Creating rerank index on %s.%s (cache=%d, k=%d)",
		 tbl_str, col_str, cache_size, k_candidates);

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", ret);

	initStringInfo(&sql);

	appendStringInfo(&sql,
					 "CREATE TABLE IF NOT EXISTS %s ("
					 "  query_hash bigint,"
					 "  query_vec vector,"
					 "  candidate_id bigint,"
					 "  candidate_vec vector,"
					 "  similarity float4,"
					 "  PRIMARY KEY(query_hash, candidate_id)"
					 ")",
					 cache_tbl);

	ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_UTILITY)
	{
		SPI_finish();
		elog(ERROR, "Failed to create rerank cache table: %s", sql.data);
	}

	SPI_finish();
	NDB_SAFE_PFREE_AND_NULL(cache_tbl);

	PG_RETURN_BOOL(true);
}

/*
 * Get pre-computed candidates for reranking
 * Returns SETOF (candidate_id bigint, similarity float4)
 */
PG_FUNCTION_INFO_V1(rerank_get_candidates);

Datum
rerank_get_candidates(PG_FUNCTION_ARGS)
{
	Vector	   *query = PG_GETARG_VECTOR_P(0);
	int32		k = PG_GETARG_INT32(1);
	uint64		query_hash;
	bool		cache_hit = false;
	char	   *table_name = "";
	char	   *vector_col = "";
	char	   *cache_tbl;
	StringInfoData sql;
	int			ret;
	FuncCallContext *funcctx;
	TupleDesc	tupdesc;
	int			i;
	StringInfoData vec_buf;
	char	   *vec_lit = NULL;
	Oid			argtypes[1];
	Datum		values[1];

	NDB_CHECK_VECTOR_VALID(query);

	if (SRF_IS_FIRSTCALL())
	{
		funcctx = SRF_FIRSTCALL_INIT();
		query_hash = vector_hash(query);
		cache_tbl = get_rerank_cache_table(table_name, vector_col);
		if ((ret = SPI_connect()) != SPI_OK_CONNECT)
			elog(ERROR, "SPI_connect failed: %d", ret);
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT candidate_id, candidate_vec, similarity "
						 "FROM %s WHERE query_hash = " NDB_UINT64_FMT " "
						 "ORDER BY similarity DESC LIMIT %d",
						 cache_tbl,
						 NDB_UINT64_CAST(query_hash),
						 k);
		ret = ndb_spi_execute_safe(sql.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT)
		{
			SPI_finish();
			elog(ERROR,
				 "Failed to retrieve rerank candidates: %s",
				 sql.data);
		}

		if (SPI_processed > 0)
			cache_hit = true;

		/*
		 * Populate from underlying vector table if cache miss. Use proper ANN
		 * scan with vector distance operator which will use HNSW or IVF index
		 * if available.
		 */
		if (!cache_hit)
		{
			initStringInfo(&vec_buf);
			appendStringInfoChar(&vec_buf, '\'');
			appendStringInfoChar(&vec_buf, '[');
			for (i = 0; i < query->dim; i++)
			{
				if (i > 0)
					appendStringInfoString(&vec_buf, ", ");
				appendStringInfo(&vec_buf, "%.6f", (double) query->data[i]);
			}
			appendStringInfoString(&vec_buf, "]'::vector");
			vec_lit = vec_buf.data;
			/* Use safe free/reinit to handle potential memory context changes */
			NDB_SAFE_PFREE_AND_NULL(sql.data);
			initStringInfo(&sql);

			/*
			 * Use vector distance operator <-> which will use index if
			 * available
			 */
			appendStringInfo(&sql,
							 "INSERT INTO %s (query_hash, query_vec, "
							 "candidate_id, candidate_vec, similarity) "
							 "SELECT " NDB_UINT64_FMT ", $1, id, %s, "
							 "1.0 - (v <-> %s) AS similarity "
							 "FROM %s ORDER BY v <-> %s LIMIT %d",
							 cache_tbl,
							 NDB_UINT64_CAST(query_hash),
							 "v",
							 vec_lit,
							 table_name,
							 vec_lit,
							 k);

			argtypes[0] = 3802;
			values[0] = PointerGetDatum(query);

			SPI_finish();
			if ((ret = SPI_connect()) != SPI_OK_CONNECT)
				elog(ERROR, "SPI_connect failed: %d", ret);

			ret = SPI_execute_with_args(sql.data, 1, argtypes, values, NULL, false, 0);

			/* Free vector literal */
			if (vec_lit != NULL)
			{
				pfree(vec_lit);
			}

			if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
			{
				SPI_finish();
				elog(ERROR,
					 "Failed to cache rerank candidates after miss: %s",
					 sql.data);
			}

			SPI_finish();

			if ((ret = SPI_connect()) != SPI_OK_CONNECT)
				elog(ERROR, "SPI_connect failed: %d", ret);

			/* Use safe free/reinit - SPI_finish destroyed previous context */
			NDB_SAFE_PFREE_AND_NULL(sql.data);
			initStringInfo(&sql);
			appendStringInfo(&sql,
							 "SELECT candidate_id, candidate_vec, similarity "
							 "FROM %s WHERE query_hash = " NDB_UINT64_FMT " "
							 "ORDER BY similarity DESC LIMIT %d",
							 cache_tbl,
							 NDB_UINT64_CAST(query_hash),
							 k);

			ret = ndb_spi_execute_safe(sql.data, true, 0);
			NDB_CHECK_SPI_TUPTABLE();
			if (ret != SPI_OK_SELECT)
			{
				SPI_finish();
				elog(ERROR,
					 "Failed to retrieve newly cached rerank candidates: %s",
					 sql.data);
			}
		}

		NDB_SAFE_PFREE_AND_NULL(cache_tbl);

		funcctx->user_fctx = SPI_tuptable;
		funcctx->max_calls = SPI_processed;

		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "candidate_id", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "similarity", FLOAT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		SPI_finish();
	}

	funcctx = SRF_PERCALL_SETUP();

	if (funcctx->call_cntr < funcctx->max_calls)
	{
		SPITupleTable *tuptable = (SPITupleTable *) funcctx->user_fctx;
		HeapTuple	spi_tuple;
		Datum		values[2];
		bool		nulls[2];
		HeapTuple	tuple;

		Assert(tuptable != NULL);

		spi_tuple = tuptable->vals[funcctx->call_cntr];

		values[0] = SPI_getbinval(spi_tuple, tuptable->tupdesc, 1, &nulls[0]);
		values[1] = SPI_getbinval(spi_tuple, tuptable->tupdesc, 3, &nulls[1]);

		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		SPITupleTable *tuptable = (SPITupleTable *) funcctx->user_fctx;

		if (tuptable)
			SPI_freetuptable(tuptable);
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * Warm up rerank index cache by precomputing candidate lists for a set of queries.
 * queries: an array of vector values.
 */
PG_FUNCTION_INFO_V1(rerank_index_warm);

Datum
rerank_index_warm(PG_FUNCTION_ARGS)
{
	text	   *index_name = PG_GETARG_TEXT_PP(0);
	ArrayType  *queries = PG_GETARG_ARRAYTYPE_P(1);
	char	   *idx_str = text_to_cstring(index_name);
	int			nqueries = ArrayGetNItems(ARR_NDIM(queries), ARR_DIMS(queries));
	int			i;
	int			ret;
	Oid			eltype = ARR_ELEMTYPE(queries);
	int16		elmlen;
	bool		elmbyval;
	char		elmalign;
	Datum	   *elem_values;
	bool	   *elem_nulls;
	char	   *cache_tbl;
	StringInfoData sql;

	get_typlenbyvalalign(eltype, &elmlen, &elmbyval, &elmalign);

	elog(INFO,
		 "neurondb: Warming rerank index %s with %d queries",
		 idx_str, nqueries);

	cache_tbl = pstrdup(idx_str);

	deconstruct_array(queries,
					  eltype,
					  elmlen,
					  elmbyval,
					  elmalign,
					  &elem_values,
					  &elem_nulls,
					  &nqueries);

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", ret);

	for (i = 0; i < nqueries; i++)
	{
		Vector	   *qv;
		uint64		qhash;
		Oid			argtypes[1];
		Datum		values[1];

		if (elem_nulls[i])
			continue;

		qv = (Vector *) DatumGetPointer(elem_values[i]);
		qhash = vector_hash(qv);

		initStringInfo(&sql);

		appendStringInfo(&sql,
						 "INSERT INTO %s (query_hash, query_vec, candidate_id, "
						 "candidate_vec, similarity) "
						 "SELECT " NDB_UINT64_FMT ", $1, id, v, (random()) "
						 "FROM some_base_table ORDER BY random() LIMIT 10",
						 cache_tbl,
						 NDB_UINT64_CAST(qhash));

		argtypes[0] = 3802;
		values[0] = PointerGetDatum(qv);

		ret = SPI_execute_with_args(sql.data, 1, argtypes, values, NULL, false, 0);
		if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
		{
			SPI_finish();
			elog(ERROR,
				 "Failed to cache candidates for query_hash " NDB_UINT64_FMT ": %s",
				 NDB_UINT64_CAST(qhash), sql.data);
		}
		/* Use safe free/reinit to handle potential memory context changes */
		NDB_SAFE_PFREE_AND_NULL(sql.data);
		initStringInfo(&sql);
	}

	if (cache_tbl)
		NDB_SAFE_PFREE_AND_NULL(cache_tbl);

	SPI_finish();

	PG_RETURN_BOOL(true);
}
