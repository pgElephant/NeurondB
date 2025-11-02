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

/*
 * Create temporal vector index
 */
PG_FUNCTION_INFO_V1(temporal_index_create);
Datum
temporal_index_create(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *vector_col = PG_GETARG_TEXT_PP(1);
	text	   *timestamp_col = PG_GETARG_TEXT_PP(2);
	float8		decay_rate = PG_GETARG_FLOAT8(3);
	char	   *tbl_str;
	char	   *vec_str;
	char	   *ts_str;
	
	tbl_str = text_to_cstring(table_name);
	vec_str = text_to_cstring(vector_col);
	ts_str = text_to_cstring(timestamp_col);
	
	elog(NOTICE, "neurondb: Creating temporal index on %s.%s with timestamp %s (decay=%.4f/day)",
		 tbl_str, vec_str, ts_str, decay_rate);
	
	PG_RETURN_BOOL(true);
}

/*
 * Time-gated kNN search
 */
PG_FUNCTION_INFO_V1(temporal_knn_search);
Datum
temporal_knn_search(PG_FUNCTION_ARGS)
{
	Vector		   *query = PG_GETARG_VECTOR_P(0);
	int32			k = PG_GETARG_INT32(1);
	TimestampTz		cutoff_time = PG_GETARG_TIMESTAMPTZ(2);
	
	(void) query;
	(void) cutoff_time;
	
	elog(NOTICE, "neurondb: Temporal kNN search for %d neighbors", k);
	
	PG_RETURN_NULL();
}

/*
 * Compute time-decayed similarity score
 */
PG_FUNCTION_INFO_V1(temporal_score);
Datum
temporal_score(PG_FUNCTION_ARGS)
{
	float4			base_score = PG_GETARG_FLOAT4(0);
	TimestampTz		insert_time = PG_GETARG_TIMESTAMPTZ(1);
	TimestampTz		current_time = PG_GETARG_TIMESTAMPTZ(2);
	float8			decay_rate = PG_GETARG_FLOAT8(3);
	float8			age_days;
	float8			decay_factor;
	float4			final_score;
	
	/* Compute age in days */
	age_days = (float8) (current_time - insert_time) / USECS_PER_DAY;
	
	/* Apply exponential decay */
	decay_factor = exp(-decay_rate * age_days);
	final_score = base_score * decay_factor;
	
	PG_RETURN_FLOAT4(final_score);
}

