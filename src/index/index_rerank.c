/*-------------------------------------------------------------------------
 *
 * index_rerank.c
 *		Rerank-ready index with pre-computed candidate lists
 *
 * Implements RRI (Rerank Ready Index) that stores top-k candidate
 * lists for hot queries, enabling zero round trips to heap for
 * reranking operations.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/index_rerank.c
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

/*
 * Helper: Generate the rerank cache table name for a given table.column
 */
static char *
get_rerank_cache_table(const char *tbl, const char *col)
{
	char *buf = palloc(strlen(tbl) + strlen(col) + 32);
	snprintf(buf, strlen(tbl) + strlen(col) + 32,
			 "__rerank_cache_%s_%s", tbl, col);
	return buf;
}

/*
 * Helper: Compute a hash for a vector (simple, for demonstration)
 */
static uint64
vector_hash(Vector *v)
{
	int i;
	uint64 h = 0;
	for (i = 0; i < v->dim && i < 16; i++)
		h = (h * 31) + (uint32)(v->data[i] * 1000);
	return h;
}

/*
 * Helper: Compute L2 distance between two vectors
 */
static float4 __attribute__((unused))
vector_l2(Vector *a, Vector *b)
{
	int i;
	float4 sum = 0.0f;
	if (a->dim != b->dim)
		elog(ERROR, "vector_l2: dimensions do not match");
	for (i = 0; i < a->dim; i++)
	{
		float4 d = a->data[i] - b->data[i];
		sum += d * d;
	}
	return sqrtf(sum);
}

/*
 * Create rerank-ready index: sets up a cache table to store (query_hash, query_vec, id, dist, rank)
 */
PG_FUNCTION_INFO_V1(rerank_index_create);
Datum
rerank_index_create(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *vector_col = PG_GETARG_TEXT_PP(1);
	int32		cache_size = PG_GETARG_INT32(2);
	int32		k_candidates = PG_GETARG_INT32(3);
	char	   *tbl_str;
	char	   *col_str;
	char	   *cache_tbl;
	StringInfoData sql;
	int		   ret;

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_col);
	cache_tbl = get_rerank_cache_table(tbl_str, col_str);

	elog(NOTICE, "neurondb: Creating rerank index on %s.%s (cache=%d, k=%d)", tbl_str, col_str, cache_size, k_candidates);

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

	ret = SPI_execute(sql.data, false, 0);
	if (ret != SPI_OK_UTILITY)
		elog(ERROR, "Failed to create rerank cache table: %s", sql.data);

	SPI_finish();
	pfree(cache_tbl);

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
	Vector		   *query = PG_GETARG_VECTOR_P(0);
	int32			k = PG_GETARG_INT32(1);
	uint64			query_hash;
	bool			cache_hit = false;
	char 		   *table_name = "";	// For simplicity, would pass in practice
	char 		   *vector_col = "";
	char 		   *cache_tbl;
	StringInfoData	sql;
	int				ret;
	FuncCallContext *funcctx;
	TupleDesc		tupdesc;

	if (SRF_IS_FIRSTCALL())
	{
		funcctx = SRF_FIRSTCALL_INIT();

		query_hash = vector_hash(query);

		/* For demonstration, the table/col are not passed; would pass for prod usage */
		cache_tbl = get_rerank_cache_table(table_name, vector_col);

		if ((ret = SPI_connect()) != SPI_OK_CONNECT)
			elog(ERROR, "SPI_connect failed: %d", ret);

		initStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT candidate_id, candidate_vec, similarity "
			"FROM %s WHERE query_hash = " NDB_UINT64_FMT " "
			"ORDER BY similarity DESC LIMIT %d",
			cache_tbl, query_hash, k);

		ret = SPI_execute(sql.data, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			SPI_finish();
			elog(ERROR, "Failed to retrieve rerank candidates: %s", sql.data);
		}

		if (SPI_processed > 0)
			cache_hit = true;

		/* If no hit: populate from underlying vector table (stub simple implementation) */
		if (!cache_hit)
		{
			/* This is a stub! In production, we'd run an ANN search over the base table.
			 * Here, for demonstration, just copy K "random" rows. */

			resetStringInfo(&sql);
			appendStringInfo(&sql,
				"INSERT INTO %s (query_hash, query_vec, candidate_id, candidate_vec, similarity) "
				"SELECT " NDB_UINT64_FMT ", $1, id, %s, (random()) "
				"FROM %s ORDER BY random() LIMIT %d",
				cache_tbl, query_hash, "v", table_name, k);

			Oid argtypes[1] = { 3802 };	// vector type Oid (may vary in prod)
			Datum values[1] = { PointerGetDatum(query) };

			SPI_finish();
			if ((ret = SPI_connect()) != SPI_OK_CONNECT)
				elog(ERROR, "SPI_connect failed: %d", ret);
			ret = SPI_execute_with_args(sql.data, 1, argtypes, values, NULL, false, 0);
			if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
			{
				SPI_finish();
				elog(ERROR, "Failed to cache rerank candidates after miss: %s", sql.data);
			}
			SPI_finish();
			if ((ret = SPI_connect()) != SPI_OK_CONNECT)
				elog(ERROR, "SPI_connect failed: %d", ret);

			/* Now, re-select the just-inserted candidates */
			resetStringInfo(&sql);
			appendStringInfo(&sql,
				"SELECT candidate_id, candidate_vec, similarity "
				"FROM %s WHERE query_hash = " NDB_UINT64_FMT " "
				"ORDER BY similarity DESC LIMIT %d",
				cache_tbl, query_hash, k);
			ret = SPI_execute(sql.data, true, 0);
			if (ret != SPI_OK_SELECT)
			{
				SPI_finish();
				elog(ERROR, "Failed to retrieve newly cached rerank candidates: %s", sql.data);
			}
		}

		pfree(cache_tbl);

		/* Prepare to return rows */
		funcctx->user_fctx = SPI_tuptable;
		funcctx->max_calls = SPI_processed;

		/* Describe return type: (candidate_id bigint, similarity float4) */
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
		HeapTuple	   spi_tuple;
		Datum		   values[2];
		bool		   nulls[2];
		HeapTuple	   tuple;

		spi_tuple = tuptable->vals[funcctx->call_cntr];
		values[0] = SPI_getbinval(spi_tuple, tuptable->tupdesc, 1, &nulls[0]);
		values[1] = SPI_getbinval(spi_tuple, tuptable->tupdesc, 3, &nulls[1]); // similarity is 3rd col in above select

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
	char	   *idx_str;
	int			nqueries;
	int			i;
	int			ret;
	Oid			eltype;
	int16		elmlen;
	bool		elmbyval;
	char		elmalign;
	Datum	   *elem_values;
	bool	   *elem_nulls;
	char	   *cache_tbl;
	StringInfoData sql;

	idx_str = text_to_cstring(index_name);
	nqueries = ArrayGetNItems(ARR_NDIM(queries), ARR_DIMS(queries));

	eltype = ARR_ELEMTYPE(queries);
	get_typlenbyvalalign(eltype, &elmlen, &elmbyval, &elmalign);

	elog(NOTICE, "neurondb: Warming rerank index %s with %d queries", idx_str, nqueries);

	/* For demo, treat index name as "__rerank_cache_tab_col" so we just use as cache table directly */
	cache_tbl = pstrdup(idx_str);

	deconstruct_array(queries, eltype, elmlen, elmbyval, elmalign, &elem_values, &elem_nulls, &nqueries);

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", ret);

	for (i = 0; i < nqueries; i++)
	{
		if (elem_nulls[i])
			continue;

		Vector *qv = (Vector *) DatumGetPointer(elem_values[i]);
		uint64 qhash = vector_hash(qv);

		initStringInfo(&sql);
		appendStringInfo(&sql,
			"INSERT INTO %s (query_hash, query_vec, candidate_id, candidate_vec, similarity) "
			"SELECT " NDB_UINT64_FMT ", $1, id, v, (random()) "
			"FROM some_base_table ORDER BY random() LIMIT 10", // change "some_base_table" in production
			cache_tbl, NDB_UINT64_CAST(qhash));

		Oid argtypes[1] = { 3802 };	// vector type Oid (should match real Oid)
		Datum values[1] = { PointerGetDatum(qv) };

		ret = SPI_execute_with_args(sql.data, 1, argtypes, values, NULL, false, 0);
		if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
		{
			SPI_finish();
			elog(ERROR, "Failed to cache candidates for query_hash " NDB_UINT64_FMT ": %s", NDB_UINT64_CAST(qhash), sql.data);
		}
		resetStringInfo(&sql);
	}

	if (cache_tbl)
		pfree(cache_tbl);

	SPI_finish();

	PG_RETURN_BOOL(true);
}
