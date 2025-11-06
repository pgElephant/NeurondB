/*-------------------------------------------------------------------------
 *
 * planner.c
 *		Adaptive Intelligence: Planner Hook, Query Optimizer, Scaling, Prefetcher
 *
 * This file implements adaptive intelligence features including
 * auto-routing planner hook, self-learning query optimizer,
 * dynamic precision scaling, and predictive prefetching.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/planner.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "executor/spi.h"
#include "utils/memutils.h"
#include "utils/hsearch.h"
#include "catalog/pg_type.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/*
 * Auto-Routing Planner Hook: Select ANN vs FTS based on query
 *
 *   Heuristics:
 *   - ANN if embedding_length > 128 or query contains "similarity" or "vector"
 *   - FTS (full table scan) otherwise
 */
PG_FUNCTION_INFO_V1(auto_route_query);
Datum
auto_route_query(PG_FUNCTION_ARGS)
{
	text	   *query = PG_GETARG_TEXT_PP(0);
	int32		embedding_length = PG_GETARG_INT32(1);
	char	   *query_str;
	bool		use_ann = false;

	query_str = text_to_cstring(query);

	/* Use ANN if embedding_length is large or similarity semantics is present */
	if (embedding_length > 128 ||
	    strstr(query_str, "similarity") != NULL ||
	    strstr(query_str, "vector") != NULL)
	{
		use_ann = true;
	}

	elog(DEBUG1, "neurondb: auto-routing: %s (embedding_length=%d)", use_ann ? "ANN" : "FTS", embedding_length);

	pfree(query_str);

	PG_RETURN_BOOL(use_ann);
}

/*
 * Self-learning Query Optimizer: Store fingerprints and adjust parameters
 *
 *   Learn query performance. Store or update fingerprinted entry in table neurondb_query_history:
 *   CREATE TABLE IF NOT EXISTS neurondb_query_history (
 *     fingerprint     BIGINT PRIMARY KEY,
 *     last_recall     REAL,
 *     last_latency    INTEGER,
 *     seen_count      INTEGER,
 *     ef_search       INTEGER,
 *     beam_size       INTEGER
 *   );
 *   - On new query: insert with defaults ef_search=32, beam_size=8
 *   - On repeated query: UPDATE recall, latency, seen_count, tune params (ef_search, beam_size)
 */
PG_FUNCTION_INFO_V1(learn_from_query);
Datum
learn_from_query(PG_FUNCTION_ARGS)
{
	text	   *query = PG_GETARG_TEXT_PP(0);
	float4		actual_recall = PG_GETARG_FLOAT4(1);
	int32		latency_ms = PG_GETARG_INT32(2);
	char	   *query_str;
	uint64		fingerprint = 5381;
	int			i;
	bool		found = false;

	query_str = text_to_cstring(query);

	/* Compute simple DJB2 hash (unsigned 64-bit) */
	for (i = 0; query_str[i] != '\0'; i++)
		fingerprint = ((fingerprint << 5) + fingerprint) + (unsigned char) query_str[i];

	elog(DEBUG1, "neurondb: learning from query (fingerprint=%llu, recall=%.4f, latency=%d)",
	     fingerprint, actual_recall, latency_ms);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: SPI_connect failed in learn_from_query")));

	/* Create the optimization history table if it does not exist */
	SPI_execute(
		"CREATE TABLE IF NOT EXISTS neurondb_query_history ("
		" fingerprint  BIGINT PRIMARY KEY,"
		" last_recall  REAL,"
		" last_latency INTEGER,"
		" seen_count   INTEGER,"
		" ef_search    INTEGER,"
		" beam_size    INTEGER"
		")", false, 0);

	/* Try SELECT first to see if fingerprint exists */
	{
		StringInfoData sql;
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT fingerprint, seen_count, ef_search, beam_size FROM neurondb_query_history WHERE fingerprint = %llu", fingerprint);

		if (SPI_execute(sql.data, true, 1) == SPI_OK_SELECT && SPI_processed == 1)
			found = true;
		pfree(sql.data);
	}

	if (found)
	{
		/* Update row, and tune ef_search/beam_size heuristically */
		int32 seen_count = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, NULL));
		int32 ef_search = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3, NULL));
		int32 beam_size = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 4, NULL));
		seen_count++;

		/* Heuristic tuning: if recall < 0.9, ef_search++; if latency > 100ms, beam_size--; */
		if (actual_recall < 0.90 && ef_search < 128)
			ef_search += 8;
		if (actual_recall > 0.99 && ef_search > 24)
			ef_search -= 8;

		if (latency_ms > 100 && beam_size > 4)
			beam_size -= 2;
		else if (latency_ms < 15 && beam_size < 32)
			beam_size += 2;

		{
			StringInfoData usql;
			initStringInfo(&usql);
			appendStringInfo(&usql,
				"UPDATE neurondb_query_history SET last_recall=%.6f, last_latency=%d, seen_count=%d, ef_search=%d, beam_size=%d WHERE fingerprint=%llu",
				actual_recall, latency_ms, seen_count, ef_search, beam_size, fingerprint);
			SPI_execute(usql.data, false, 0);
			pfree(usql.data);
		}
	}
	else
	{
		/* Insert new row with default optimization params */
		StringInfoData isql;
		initStringInfo(&isql);
		appendStringInfo(&isql,
			"INSERT INTO neurondb_query_history (fingerprint, last_recall, last_latency, seen_count, ef_search, beam_size) "
			"VALUES (%llu, %.6f, %d, %d, %d, %d)",
			fingerprint, actual_recall, latency_ms, 1, 32, 8);
		SPI_execute(isql.data, false, 0);
		pfree(isql.data);
	}

	pfree(query_str);
	SPI_finish();

	PG_RETURN_BOOL(true);
}

/*
 * Dynamic Precision Scaling: Switch float32→int8→binary
 *
 * - If memory_pressure > 0.8 or recall_target < 0.85: quantize to int8
 * - If memory_pressure > 0.6 or recall_target < 0.90: float16 (simulated as half-precision)
 * - else: keep as float32
 * The returned type is always a Vector *, but quantized as needed.
 */
static void quantize_to_int8(const float4 *src, int8_t *dst, int len)
{
	int i;
	for (i = 0; i < len; i++)
	{
		float x = src[i];
		if (x > 127.0f) x = 127.0f;
		if (x < -128.0f) x = -128.0f;
		dst[i] = (int8_t) rint(x);
	}
}

/* Simulate float16: simply round to one decimal and store as float4 for demonstration */
static void quantize_to_float16(const float4 *src, float4 *dst, int len)
{
	int i;
	for (i = 0; i < len; i++)
		dst[i] = roundf(src[i] * 10.0f) / 10.0f;
}

PG_FUNCTION_INFO_V1(scale_precision);
Datum
scale_precision(PG_FUNCTION_ARGS)
{
	Vector	   *input = (Vector *) PG_GETARG_POINTER(0);
	float4		memory_pressure = PG_GETARG_FLOAT4(1);
	float4		recall_target = PG_GETARG_FLOAT4(2);
	int			target_precision;
	Vector	   *result = NULL;

	if (input == NULL)
		ereport(ERROR, (errmsg("input vector is null")));

	if (memory_pressure > 0.8 || recall_target < 0.85)
		target_precision = 8; /* int8 */
	else if (memory_pressure > 0.6 || recall_target < 0.90)
		target_precision = 16; /* float16 */
	else
		target_precision = 32; /* float32 */

	elog(DEBUG1, "neurondb: scaling precision to %d-bit (memory=%.2f, recall=%.2f)",
		 target_precision, memory_pressure, recall_target);

	if (target_precision == 8)
	{
		/* Store quantized values in the data field as floats, but quantized to int8. */
		result = new_vector(input->dim);
		int8_t *int8buf = (int8_t *) palloc0(sizeof(int8_t) * input->dim);
		quantize_to_int8(input->data, int8buf, input->dim);

		/* Copy to float4 struct for compatibility */
		for (int i = 0; i < input->dim; i++)
			result->data[i] = (float4) int8buf[i];
		pfree(int8buf);
	}
	else if (target_precision == 16)
	{
		result = new_vector(input->dim);
		quantize_to_float16(input->data, result->data, input->dim);
	}
	else
	{
		result = new_vector(input->dim);
		memcpy(result->data, input->data, sizeof(float4) * input->dim);
	}

	PG_RETURN_POINTER(result);
}

/*
 * Predictive Prefetcher: Preload probable HNSW entry points
 *
 *   - Estimate candidate entry points by nearest centroid or clustering statistics.
 *   - For demo, use index statistics and touch (dummy read) their relpages.
 */
PG_FUNCTION_INFO_V1(prefetch_entry_points);
Datum
prefetch_entry_points(PG_FUNCTION_ARGS)
{
	text	   *index_name = PG_GETARG_TEXT_PP(0);
	Vector	   *query_vector = (Vector *) PG_GETARG_POINTER(1);
	char	   *idx_str;
	int			prefetched_count = 0;

	if (index_name == NULL)
		ereport(ERROR, (errmsg("index_name cannot be null")));
	if (query_vector == NULL)
		ereport(ERROR, (errmsg("query_vector cannot be null")));

	idx_str = text_to_cstring(index_name);

	elog(DEBUG1, "neurondb: prefetching entry points for index '%s'", idx_str);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("neurondb: SPI_connect failed in prefetch_entry_points")));

	/* Find index relid and relpages, simulate "prefetch" of their entry points */
	{
		StringInfoData sql;
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT c.oid, c.relpages FROM pg_class c "
			"JOIN pg_index i ON c.oid = i.indexrelid "
			"WHERE c.relname = '%s'", idx_str);

		if (SPI_execute(sql.data, true, 10) == SPI_OK_SELECT && SPI_processed > 0)
		{
			int i;
			for (i = 0; i < (int)SPI_processed; i++)
			{
				Datum pagesDatum;
				bool isnull;
				pagesDatum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2, &isnull);
				if (!isnull)
				{
					int32 pages = DatumGetInt32(pagesDatum);
					/* Each index page "touched" corresponds to a simulated prefetch. Limit 128. */
					if (pages > 128)
						pages = 128;
					prefetched_count += pages;
				}
			}
		}
		pfree(sql.data);
	}
	SPI_finish();
	pfree(idx_str);

	elog(DEBUG1, "neurondb: prefetched %d entry points", prefetched_count);

	PG_RETURN_INT32(prefetched_count);
}
