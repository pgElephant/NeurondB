/*-------------------------------------------------------------------------
 *
 * index_temporal.c
 *		Temporal vector index with decay on insert time
 *
 * Implements TVX index that applies time-based decay to similarity
 * scores, enabling time-gated kNN queries where recent vectors
 * are automatically boosted.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/index_temporal.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_index.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
#include "executor/spi.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "utils/memutils.h"
#include "utils/elog.h"
#include "lib/stringinfo.h"
#include <math.h>

/* Forward declarations */
static char *VectorToLiteral(Vector *v);

/* Helper: returns qualified TVX index table name for a table/col */
static char *
get_temporal_index_table(const char *table, const char *col)
{
	char *buf = palloc(strlen(table) + strlen(col) + 32);
	snprintf(buf,
		strlen(table) + strlen(col) + 32,
		"__tvx_index_%s_%s",
		table,
		col);
	return buf;
}

/*
 * Create temporal vector index.
 * This sets up a separate index table holding (id, vector, insert_ts) for the given table/col.
 * metadata could be extended to store decay rate per index.
 */
PG_FUNCTION_INFO_V1(temporal_index_create);
Datum
temporal_index_create(PG_FUNCTION_ARGS)
{
	text *table_name = PG_GETARG_TEXT_PP(0);
	text *vector_col = PG_GETARG_TEXT_PP(1);
	text *timestamp_col = PG_GETARG_TEXT_PP(2);
	float8 decay_rate = PG_GETARG_FLOAT8(3);
	char *tbl_str;
	char *vec_str;
	char *ts_str;
	char *idx_tbl;
	StringInfoData sql;
	int ret;

	tbl_str = text_to_cstring(table_name);
	vec_str = text_to_cstring(vector_col);
	ts_str = text_to_cstring(timestamp_col);
	idx_tbl = get_temporal_index_table(tbl_str, vec_str);

	elog(NOTICE,
		"neurondb: Creating temporal index on %s.%s with timestamp %s "
		"(decay=%.4f/day)",
		tbl_str,
		vec_str,
		ts_str,
		decay_rate);

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", ret);

	/* 1. Create index table if it does not exist */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"CREATE TABLE IF NOT EXISTS %s ("
		"  id bigint PRIMARY KEY,"
		"  v vector,"
		"  insert_ts timestamptz"
		")",
		idx_tbl);
	ret = SPI_execute(sql.data, false, 0);
	if (ret != SPI_OK_UTILITY)
		elog(ERROR, "Failed to create TVX index table: %s", sql.data);

	resetStringInfo(&sql);

	/* 2. Clear index in case we recreate */
	appendStringInfo(&sql, "TRUNCATE %s", idx_tbl);
	if (SPI_execute(sql.data, false, 0) != SPI_OK_UTILITY)
		elog(ERROR, "Failed to truncate TVX index table: %s", idx_tbl);
	resetStringInfo(&sql);

	/* 3. Bulk insert all current vectors from source table */
	appendStringInfo(&sql,
		"INSERT INTO %s (id, v, insert_ts) "
		"SELECT id, %s, %s FROM %s",
		idx_tbl,
		vec_str,
		ts_str,
		tbl_str);

	ret = SPI_execute(sql.data, false, 0);
	if (ret != SPI_OK_INSERT)
		elog(ERROR,
			"Failed to bulk insert vectors into TVX index: %s",
			sql.data);

	SPI_finish();

	/* Could, in practice, store metadata about decay_rate, etc, for this index */

	pfree(idx_tbl);
	PG_RETURN_BOOL(true);
}

/*
 * Time-gated temporal kNN search.
 * 
 * Args:
 *   query: vector (similarity, higher is better)
 *   k: number of results
 *   cutoff_time: cutoff on insert_ts (timestamptz); only entries inserted at or before this are considered
 *   table_name, vector_col (optional): we fetch these as CSTRING input for full implementation.
 *   decay_rate (optional): fetch from metadata or as argument for testing
 *
 * RETURNS: SETOF (id bigint, score float4, insert_ts timestamptz)
 */
PG_FUNCTION_INFO_V1(temporal_knn_search);
Datum
temporal_knn_search(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	TupleDesc tupdesc;
	MemoryContext oldcontext;

	if (SRF_IS_FIRSTCALL())
	{
		Vector *query = PG_GETARG_VECTOR_P(0);
		int32 k = PG_GETARG_INT32(1);
		TimestampTz cutoff_time = PG_GETARG_TIMESTAMPTZ(2);
		text *table_name = PG_GETARG_TEXT_PP(3);
		text *vector_col = PG_GETARG_TEXT_PP(4);
		text *ts_col = PG_GETARG_TEXT_PP(5);
		float8 decay_rate = PG_GETARG_FLOAT8(6);

		/* Suppress unused parameter warning - may be used in future */
		(void)ts_col;

		char *tbl_str = text_to_cstring(table_name);
		char *vec_str = text_to_cstring(vector_col);

		char *idx_tbl = get_temporal_index_table(tbl_str, vec_str);
		StringInfoData sql;
		int ret;

		funcctx = SRF_FIRSTCALL_INIT();

		/* Compose tuple desc: (id bigint, score real, insert_ts timestamptz) */
		tupdesc = CreateTemplateTupleDesc(3);
		TupleDescInitEntry(
			tupdesc, (AttrNumber)1, "id", INT8OID, -1, 0);
		TupleDescInitEntry(
			tupdesc, (AttrNumber)2, "score", FLOAT4OID, -1, 0);
		TupleDescInitEntry(tupdesc,
			(AttrNumber)3,
			"insert_ts",
			TIMESTAMPTZOID,
			-1,
			0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if ((ret = SPI_connect()) != SPI_OK_CONNECT)
			elog(ERROR, "SPI_connect failed: %d", ret);

		initStringInfo(&sql);

		/* 
		 * The similarity computation is delegated to SQL using a UDF temporal_score.
		 * We only consider rows with insert_ts <= cutoff_time.
		 * We assume vector_l2 exists (or replace with your similarity UDF).
		 */
		appendStringInfo(&sql,
			"SELECT id, "
			"  temporal_score(-vector_l2(v, %s), insert_ts, now(), "
			"%f) AS score,"
			"  insert_ts "
			"FROM %s "
			"WHERE insert_ts <= to_timestamp(%lld) "
			"ORDER BY score DESC "
			"LIMIT %d",
			/* Arguments for vector, decay_rate, index table, cutoff_time, k */
			VectorToLiteral(query),
			decay_rate,
			idx_tbl,
			(long long)cutoff_time,
			k);

		ret = SPI_execute(sql.data, true, 0);
		if (ret != SPI_OK_SELECT)
			elog(ERROR, "Failed temporal index scan: %s", sql.data);

		funcctx->user_fctx = SPI_tuptable;

		MemoryContextSwitchTo(oldcontext);

		pfree(idx_tbl);
	}

	funcctx = SRF_PERCALL_SETUP();
	uint64 call_cntr = funcctx->call_cntr;
	uint64 max_calls = SPI_processed;
	SPITupleTable *tuptable = (SPITupleTable *)funcctx->user_fctx;

	if (call_cntr < max_calls)
	{
		HeapTuple spi_tuple = tuptable->vals[call_cntr];
		Datum values[3];
		bool nulls[3] = { false, false, false };
		bool isnull;
		HeapTuple result_tuple;

		values[0] =
			SPI_getbinval(spi_tuple, tuptable->tupdesc, 1, &isnull);
		nulls[0] = isnull;
		values[1] =
			SPI_getbinval(spi_tuple, tuptable->tupdesc, 2, &isnull);
		nulls[1] = isnull;
		values[2] =
			SPI_getbinval(spi_tuple, tuptable->tupdesc, 3, &isnull);
		nulls[2] = isnull;

		result_tuple =
			heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(result_tuple));
	} else
	{
		if (tuptable)
			SPI_freetuptable(tuptable);
		SPI_finish();
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * Compute time-decayed similarity score.
 * We assume base_score is typically similarity (e.g., negative L2 distance).
 * insert_time, current_time in microseconds.
 * decay_rate = in 1/days, e.g. 0.1 means about e^-0.1 decay per day.
 */
PG_FUNCTION_INFO_V1(temporal_score);
Datum
temporal_score(PG_FUNCTION_ARGS)
{
	float4 base_score = PG_GETARG_FLOAT4(0);
	TimestampTz insert_time = PG_GETARG_TIMESTAMPTZ(1);
	TimestampTz current_time = PG_GETARG_TIMESTAMPTZ(2);
	float8 decay_rate = PG_GETARG_FLOAT8(3);
	float8 age_days;
	float8 decay_factor;
	float4 final_score;

	/* Compute age in days */
	age_days = (float8)(current_time - insert_time) / USECS_PER_DAY;

	/* Apply exponential decay */
	decay_factor = exp(-decay_rate * age_days);
	final_score = base_score * decay_factor;

	PG_RETURN_FLOAT4(final_score);
}

/*
 * Helper: Converts a Vector * to a single-quoted SQL literal, for use in SPI SQL.
 * Caller responsible for pfree.
 */
static char *
VectorToLiteral(Vector *v)
{
	char *out = vector_out_internal(v);
	char *buf = palloc(strlen(out) + 4);
	snprintf(buf, strlen(out) + 4, "'%s'", out);
	return buf;
}

/*
 * Example: L2 distance function
 * You should provide this in your extension, or use similarity/vectors if available.
 */
PG_FUNCTION_INFO_V1(vector_l2);
Datum
vector_l2(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	int i;
	float4 sum = 0.0;

	if (a->dim != b->dim)
		elog(ERROR, "vector_l2: dimensions do not match");

	for (i = 0; i < a->dim; i++)
	{
		float4 d = a->data[i] - b->data[i];
		sum += d * d;
	}
	PG_RETURN_FLOAT4(sqrt(sum));
}
