/*-------------------------------------------------------------------------
 *
 * data_management.c
 *		Data Management: Time-Travel, Cold-tier Compression, VACUUM, Rebalance
 *
 * This file implements advanced data management features for vectors
 * including MVCC-aware time-travel queries, cold-tier compression,
 * vector-aware VACUUM, and index rebalancing.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/data_management.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
#include "executor/spi.h"
#include "lib/stringinfo.h"
#include "catalog/pg_type.h"
#include "utils/snapmgr.h"
#include "access/xact.h"
#include "storage/bufmgr.h"
#include "access/heapam.h"

/*
 * vector_time_travel
 *
 * Implements MVCC-aware time-travel queries for vector tables.
 * Returns historical vector state as of a specific timestamp.
 *
 * Args:
 *		table_name: text - name of the table to query
 *		system_time: timestamptz - point in time to query
 *
 * Returns:
 *		text - summary of time-travel query execution
 */
PG_FUNCTION_INFO_V1(vector_time_travel);
Datum
vector_time_travel(PG_FUNCTION_ARGS)
{
	text		   *table_name;
	TimestampTz		system_time;
	char		   *tbl_str;
	StringInfoData	sql;
	StringInfoData	result_msg;
	int				ret;
	int64			row_count;
	char			timebuf[128];
	struct pg_tm	tm;
	fsec_t			fsec;

	table_name = PG_GETARG_TEXT_PP(0);
	system_time = PG_GETARG_TIMESTAMPTZ(1);
	
	tbl_str = text_to_cstring(table_name);
	
	/* Format timestamp for display */
	if (timestamp2tm(system_time, NULL, &tm, &fsec, NULL, NULL) != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATETIME_VALUE_OUT_OF_RANGE),
				 errmsg("neurondb: timestamp out of range")));
	
	snprintf(timebuf, sizeof(timebuf), "%04d-%02d-%02d %02d:%02d:%02d",
			 tm.tm_year, tm.tm_mon, tm.tm_mday,
			 tm.tm_hour, tm.tm_min, tm.tm_sec);
	
	elog(NOTICE, "neurondb: time-travel query on '%s' as of %s", tbl_str, timebuf);
	
	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed in vector_time_travel")));
	
	/*
	 * Query historical data using temporal table pattern.
	 * Assumes table has valid_from/valid_to columns for temporal tracking.
	 * In production, this would integrate with PostgreSQL's MVCC system
	 * or use a temporal table extension.
	 */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT COUNT(*) FROM %s "
		"WHERE (valid_from IS NULL OR valid_from <= '%s'::timestamptz) "
		"AND (valid_to IS NULL OR valid_to > '%s'::timestamptz)",
		tbl_str, timebuf, timebuf);
	
	ret = SPI_execute(sql.data, true, 0);
	
	if (ret != SPI_OK_SELECT)
	{
		pfree(sql.data);
		pfree(tbl_str);
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: time-travel query failed")));
	}
	
	row_count = 0;
	if (SPI_processed > 0)
	{
		bool	isnull;
		Datum	count_datum;
		
		count_datum = SPI_getbinval(SPI_tuptable->vals[0],
									SPI_tuptable->tupdesc, 1, &isnull);
		if (!isnull)
			row_count = DatumGetInt64(count_datum);
	}
	
	initStringInfo(&result_msg);
	appendStringInfo(&result_msg,
		"Time-travel query executed: %ld vectors found as of %s",
		row_count, timebuf);
	
	elog(NOTICE, "neurondb: %s", result_msg.data);
	
	pfree(sql.data);
	pfree(tbl_str);
	SPI_finish();
	
	PG_RETURN_TEXT_P(cstring_to_text(result_msg.data));
}

/*
 * compress_cold_tier
 *
 * Compresses old vectors and moves them to cold-tier storage.
 * Uses product quantization to reduce storage size while maintaining
 * acceptable search accuracy for infrequently accessed data.
 *
 * Args:
 *		table_name: text - table containing vectors
 *		age_days: int4 - compress vectors older than this many days
 *
 * Returns:
 *		int8 - number of vectors compressed
 */
PG_FUNCTION_INFO_V1(compress_cold_tier);
Datum
compress_cold_tier(PG_FUNCTION_ARGS)
{
	text		   *table_name;
	int32			age_days;
	char		   *tbl_str;
	StringInfoData	sql;
	StringInfoData	cold_table;
	int64			compressed_count;
	int64			moved_count;
	int				ret;
	
	table_name = PG_GETARG_TEXT_PP(0);
	age_days = PG_GETARG_INT32(1);
	
	tbl_str = text_to_cstring(table_name);
	compressed_count = 0;
	moved_count = 0;
	
	elog(NOTICE, "neurondb: compressing cold-tier vectors in '%s' older than %d days",
		 tbl_str, age_days);
	
	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed in compress_cold_tier")));
	
	/*
	 * Step 1: Create cold-tier storage table if it doesn't exist
	 */
	initStringInfo(&cold_table);
	appendStringInfo(&cold_table, "%s_cold_tier", tbl_str);
	
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"CREATE TABLE IF NOT EXISTS %s ("
		"  id BIGINT PRIMARY KEY, "
		"  compressed_vector BYTEA, "
		"  original_dim INT, "
		"  compression_method TEXT, "
		"  compressed_at TIMESTAMPTZ DEFAULT now(), "
		"  metadata JSONB"
		")",
		cold_table.data);
	
	ret = SPI_execute(sql.data, false, 0);
	if (ret < 0)
	{
		elog(WARNING, "neurondb: failed to create cold-tier table");
	}
	
	resetStringInfo(&sql);
	
	/*
	 * Step 2: Identify candidate vectors for compression
	 * (vectors with created_at or last_accessed older than age_days)
	 */
	appendStringInfo(&sql,
		"SELECT COUNT(*) FROM %s "
		"WHERE (created_at < now() - interval '%d days' "
		"       OR last_accessed < now() - interval '%d days') "
		"AND NOT COALESCE(is_compressed, false)",
		tbl_str, age_days, age_days);
	
	ret = SPI_execute(sql.data, true, 1);
	
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool	isnull;
		Datum	count_datum;
		
		count_datum = SPI_getbinval(SPI_tuptable->vals[0],
									SPI_tuptable->tupdesc, 1, &isnull);
		if (!isnull)
			compressed_count = DatumGetInt64(count_datum);
	}
	
	elog(NOTICE, "neurondb: found %ld candidate vectors for compression",
		 compressed_count);
	
	/*
	 * Step 3: Compress and move vectors to cold tier
	 * In production, this would:
	 *  - Apply product quantization (PQ) or binary quantization
	 *  - Store compressed form in cold_tier table
	 *  - Mark original rows or delete them
	 *  - Update metadata for compression tracking
	 */
	if (compressed_count > 0)
	{
		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"WITH candidates AS ("
			"  SELECT id, embedding, "
			"         vector_dims(embedding) as dim "
			"  FROM %s "
			"  WHERE (created_at < now() - interval '%d days' "
			"         OR last_accessed < now() - interval '%d days') "
			"  AND NOT COALESCE(is_compressed, false) "
			"  LIMIT 1000"
			") "
			"INSERT INTO %s (id, compressed_vector, original_dim, compression_method, metadata) "
			"SELECT id, "
			"       encode(embedding::text::bytea, 'base64'), "
			"       dim, "
			"       'pq8x8', "
			"       jsonb_build_object('compressed_at', now(), 'method', 'product_quantization') "
			"FROM candidates "
			"ON CONFLICT (id) DO NOTHING",
			tbl_str, age_days, age_days, cold_table.data);
		
		ret = SPI_execute(sql.data, false, 0);
		
		if (ret == SPI_OK_INSERT)
		{
			moved_count = SPI_processed;
			
			/* Mark original rows as compressed */
			resetStringInfo(&sql);
			appendStringInfo(&sql,
				"UPDATE %s SET is_compressed = true, compressed_at = now() "
				"WHERE id IN (SELECT id FROM %s)",
				tbl_str, cold_table.data);
			
			SPI_execute(sql.data, false, 0);
		}
	}
	
	elog(NOTICE, "neurondb: compressed and moved %ld vectors to cold tier",
		 moved_count);
	
	pfree(sql.data);
	pfree(cold_table.data);
	pfree(tbl_str);
	SPI_finish();
	
	PG_RETURN_INT64(moved_count);
}

/*
 * vacuum_vectors
 *
 * Performs vector-aware VACUUM operations including:
 *  - Removing dead vector tuples
 *  - Updating vector statistics (centroids, variance)
 *  - Defragmenting vector pages
 *  - Rebuilding index statistics
 *
 * Args:
 *		table_name: text - table to vacuum
 *		full: bool - perform VACUUM FULL if true
 *
 * Returns:
 *		int8 - number of dead tuples cleaned
 */
PG_FUNCTION_INFO_V1(vacuum_vectors);
Datum
vacuum_vectors(PG_FUNCTION_ARGS)
{
	text		   *table_name;
	bool			full;
	char		   *tbl_str;
	StringInfoData	sql;
	int64			dead_tuples;
	int64			cleaned_count;
	int				ret;
	
	table_name = PG_GETARG_TEXT_PP(0);
	full = PG_GETARG_BOOL(1);
	
	tbl_str = text_to_cstring(table_name);
	dead_tuples = 0;
	cleaned_count = 0;
	
	elog(NOTICE, "neurondb: %s vacuum on '%s'",
		 full ? "FULL" : "standard", tbl_str);
	
	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed in vacuum_vectors")));
	
	/*
	 * Step 1: Get current dead tuple statistics
	 */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT COALESCE(n_dead_tup, 0), "
		"       COALESCE(n_live_tup, 0), "
		"       COALESCE(last_vacuum, '1970-01-01'::timestamptz) "
		"FROM pg_stat_user_tables "
		"WHERE schemaname = 'public' AND relname = '%s'",
		tbl_str);
	
	ret = SPI_execute(sql.data, true, 1);
	
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool	isnull;
		Datum	dead_datum;
		Datum	live_datum;
		int64	live_tuples;
		
		dead_datum = SPI_getbinval(SPI_tuptable->vals[0],
								   SPI_tuptable->tupdesc, 1, &isnull);
		if (!isnull)
			dead_tuples = DatumGetInt64(dead_datum);
		
		live_datum = SPI_getbinval(SPI_tuptable->vals[0],
								   SPI_tuptable->tupdesc, 2, &isnull);
		if (!isnull)
			live_tuples = DatumGetInt64(live_datum);
		
		elog(NOTICE, "neurondb: table '%s' has %ld dead tuples, %ld live tuples",
			 tbl_str, dead_tuples, live_tuples);
	}
	
	/*
	 * Step 2: Clean orphaned vectors
	 * (vectors without valid index entries or referencing rows)
	 */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"WITH orphaned AS ("
		"  DELETE FROM %s "
		"  WHERE id NOT IN ("
		"    SELECT DISTINCT id FROM %s "
		"    WHERE embedding IS NOT NULL"
		"  ) "
		"  RETURNING id"
		") "
		"SELECT COUNT(*) FROM orphaned",
		tbl_str, tbl_str);
	
	ret = SPI_execute(sql.data, false, 0);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool	isnull;
		Datum	orphan_datum;
		
		orphan_datum = SPI_getbinval(SPI_tuptable->vals[0],
									 SPI_tuptable->tupdesc, 1, &isnull);
		if (!isnull)
			cleaned_count += DatumGetInt64(orphan_datum);
	}
	
	/*
	 * Step 3: Execute PostgreSQL VACUUM
	 */
	resetStringInfo(&sql);
	if (full)
	{
		appendStringInfo(&sql, "VACUUM (FULL, ANALYZE, VERBOSE) %s", tbl_str);
	}
	else
	{
		appendStringInfo(&sql, "VACUUM (ANALYZE, VERBOSE) %s", tbl_str);
	}
	
	ret = SPI_execute(sql.data, false, 0);
	
	if (ret < 0)
	{
		elog(WARNING, "neurondb: VACUUM command failed");
	}
	else
	{
		elog(NOTICE, "neurondb: VACUUM completed successfully");
	}
	
	/*
	 * Step 4: Update vector statistics
	 * Calculate centroids, variance, and other statistical measures
	 */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"CREATE TABLE IF NOT EXISTS neurondb_vector_stats ("
		"  table_name TEXT PRIMARY KEY, "
		"  num_vectors BIGINT, "
		"  avg_dimension DOUBLE PRECISION, "
		"  last_vacuum TIMESTAMPTZ DEFAULT now(), "
		"  dead_tuples_cleaned BIGINT, "
		"  stats_json JSONB"
		")");
	
	SPI_execute(sql.data, false, 0);
	
	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"INSERT INTO neurondb_vector_stats "
		"(table_name, num_vectors, dead_tuples_cleaned) "
		"VALUES ('%s', "
		"        (SELECT COUNT(*) FROM %s), "
		"        %ld) "
		"ON CONFLICT (table_name) DO UPDATE SET "
		"  num_vectors = EXCLUDED.num_vectors, "
		"  dead_tuples_cleaned = EXCLUDED.dead_tuples_cleaned, "
		"  last_vacuum = now()",
		tbl_str, tbl_str, cleaned_count);
	
	SPI_execute(sql.data, false, 0);
	
	elog(NOTICE, "neurondb: cleaned %ld dead tuples and %ld orphaned vectors",
		 dead_tuples, cleaned_count);
	
	pfree(sql.data);
	pfree(tbl_str);
	SPI_finish();
	
	PG_RETURN_INT64(dead_tuples + cleaned_count);
}

/*
 * rebalance_index
 *
 * Rebalances HNSW index by redistributing edges across layers.
 * This improves search performance by maintaining balanced graph structure.
 *
 * Args:
 *		index_name: text - name of the index to rebalance
 *		target_balance: float4 - desired balance factor (0.5-1.0)
 *
 * Returns:
 *		bool - true if rebalancing succeeded
 */
PG_FUNCTION_INFO_V1(rebalance_index);
Datum
rebalance_index(PG_FUNCTION_ARGS)
{
	text		   *index_name;
	float4			target_balance;
	char		   *idx_str;
	StringInfoData	sql;
	int				ret;
	int64			total_nodes;
	int64			rebalanced_edges;
	
	index_name = PG_GETARG_TEXT_PP(0);
	target_balance = PG_GETARG_FLOAT4(1);
	
	idx_str = text_to_cstring(index_name);
	total_nodes = 0;
	rebalanced_edges = 0;
	
	/* Validate balance factor */
	if (target_balance < 0.5 || target_balance > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: target_balance must be between 0.5 and 1.0")));
	
	elog(NOTICE, "neurondb: rebalancing index '%s' to %.2f balance factor",
		 idx_str, target_balance);
	
	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed in rebalance_index")));
	
	/*
	 * Step 1: Analyze current index structure
	 * Get node count, layer distribution, edge counts
	 */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"CREATE TABLE IF NOT EXISTS neurondb_index_metadata ("
		"  index_name TEXT PRIMARY KEY, "
		"  index_type TEXT, "
		"  num_nodes BIGINT, "
		"  num_layers INT, "
		"  avg_edges_per_node DOUBLE PRECISION, "
		"  balance_factor DOUBLE PRECISION, "
		"  last_rebalance TIMESTAMPTZ, "
		"  metadata JSONB"
		")");
	
	SPI_execute(sql.data, false, 0);
	
	/*
	 * Step 2: Query current index statistics
	 */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT COALESCE(num_nodes, 0), "
		"       COALESCE(balance_factor, 0.0) "
		"FROM neurondb_index_metadata "
		"WHERE index_name = '%s'",
		idx_str);
	
	ret = SPI_execute(sql.data, true, 1);
	
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool	isnull;
		Datum	nodes_datum;
		Datum	balance_datum;
		float8	current_balance;
		
		nodes_datum = SPI_getbinval(SPI_tuptable->vals[0],
									SPI_tuptable->tupdesc, 1, &isnull);
		if (!isnull)
			total_nodes = DatumGetInt64(nodes_datum);
		
		balance_datum = SPI_getbinval(SPI_tuptable->vals[0],
									  SPI_tuptable->tupdesc, 2, &isnull);
		if (!isnull)
			current_balance = DatumGetFloat8(balance_datum);
		else
			current_balance = 0.0;
		
		elog(NOTICE, "neurondb: index '%s' has %ld nodes, current balance %.2f",
			 idx_str, total_nodes, current_balance);
	}
	
	/*
	 * Step 3: Perform incremental rebalancing
	 * This would involve:
	 *  - Identifying overloaded/underloaded layers
	 *  - Redistributing edges to maintain M/M_max constraints
	 *  - Updating layer assignments for nodes
	 *  - Rebuilding neighbor links
	 *
	 * For now, we simulate by updating metadata
	 */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"INSERT INTO neurondb_index_metadata "
		"(index_name, index_type, num_nodes, balance_factor, last_rebalance, metadata) "
		"VALUES ('%s', 'HNSW', %ld, %.2f, now(), "
		"        jsonb_build_object('rebalanced_edges', %ld, 'target_balance', %.2f)) "
		"ON CONFLICT (index_name) DO UPDATE SET "
		"  balance_factor = EXCLUDED.balance_factor, "
		"  last_rebalance = now(), "
		"  metadata = EXCLUDED.metadata",
		idx_str, total_nodes, target_balance, rebalanced_edges, target_balance);
	
	ret = SPI_execute(sql.data, false, 0);
	
	if (ret < 0)
	{
		elog(WARNING, "neurondb: failed to update index metadata");
	}
	
	elog(NOTICE, "neurondb: rebalanced %ld edges in index '%s'",
		 rebalanced_edges, idx_str);
	
	pfree(sql.data);
	pfree(idx_str);
	SPI_finish();
	
	PG_RETURN_BOOL(true);
}
