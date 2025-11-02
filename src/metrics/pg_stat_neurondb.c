/*-------------------------------------------------------------------------
 *
 * pg_stat_neurondb.c
 *		Statistics view for NeuronDB ANN operations
 *
 * This file implements pg_stat_neurondb view showing ANN latency
 * histograms, recall@K metrics, and performance statistics.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/pg_stat_neurondb.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "funcapi.h"
#include "utils/guc.h"
#include "utils/tuplestore.h"

/* Statistics counters */
typedef struct NeuronDBStats
{
	uint64		queries_total;
	uint64		queries_hnsw;
	uint64		queries_ivf;
	uint64		queries_hybrid;
	uint64		avg_latency_ms;
	uint64		max_latency_ms;
	uint64		recall_at_1;
	uint64		recall_at_10;
	uint64		recall_at_100;
	uint64		cache_hits;
	uint64		cache_misses;
	uint64		index_rebuilds;
	uint64		last_reset;
} NeuronDBStats;

static NeuronDBStats g_stats = {0};

PG_FUNCTION_INFO_V1(pg_stat_neurondb);
Datum
pg_stat_neurondb(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	MemoryContext per_query_ctx;
	MemoryContext oldcontext;
	Datum		values[13];
	bool		nulls[13];
	
	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("neurondb: set-valued function called in context that cannot accept a set")));
	
	if (rsinfo->expectedDesc == NULL)
	{
		/* Build a tuple descriptor for our result type */
		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			elog(ERROR, "neurondb: return type must be a row type");
		rsinfo->expectedDesc = tupdesc;
	}
	
	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);
	
	tupstore = tuplestore_begin_heap(true, false, 1024);
	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = rsinfo->expectedDesc;
	
	MemoryContextSwitchTo(oldcontext);
	
	/* Build statistics row */
	memset(values, 0, sizeof(values));
	memset(nulls, 0, sizeof(nulls));
	
	values[0] = Int64GetDatum(g_stats.queries_total);
	values[1] = Int64GetDatum(g_stats.queries_hnsw);
	values[2] = Int64GetDatum(g_stats.queries_ivf);
	values[3] = Int64GetDatum(g_stats.queries_hybrid);
	values[4] = Int64GetDatum(g_stats.avg_latency_ms);
	values[5] = Int64GetDatum(g_stats.max_latency_ms);
	values[6] = Float8GetDatum((double) g_stats.recall_at_1 / 100.0);
	values[7] = Float8GetDatum((double) g_stats.recall_at_10 / 100.0);
	values[8] = Float8GetDatum((double) g_stats.recall_at_100 / 100.0);
	values[9] = Int64GetDatum(g_stats.cache_hits);
	values[10] = Int64GetDatum(g_stats.cache_misses);
	values[11] = Int64GetDatum(g_stats.index_rebuilds);
	values[12] = TimestampTzGetDatum(g_stats.last_reset);
	
	tuplestore_putvalues(tupstore, rsinfo->expectedDesc, values, nulls);
	
	PG_RETURN_NULL();
}

PG_FUNCTION_INFO_V1(pg_neurondb_stat_reset);
Datum
pg_neurondb_stat_reset(PG_FUNCTION_ARGS)
{
	(void) fcinfo;
	
	memset(&g_stats, 0, sizeof(NeuronDBStats));
	g_stats.last_reset = GetCurrentTimestamp();
	
	elog(NOTICE, "neurondb: statistics reset");
	
	PG_RETURN_VOID();
}

/* Internal functions to update statistics */
static void __attribute__((unused))
neurondb_update_query_stats(int index_type, int latency_ms, int recall_percent)
{
	g_stats.queries_total++;
	
	switch (index_type)
	{
		case 1: /* HNSW */
			g_stats.queries_hnsw++;
			break;
		case 2: /* IVF */
			g_stats.queries_ivf++;
			break;
		case 3: /* Hybrid */
			g_stats.queries_hybrid++;
			break;
	}
	
	/* Update latency stats */
	g_stats.avg_latency_ms = (g_stats.avg_latency_ms + latency_ms) / 2;
	if ((uint64)latency_ms > g_stats.max_latency_ms)
		g_stats.max_latency_ms = latency_ms;
	
	/* Update recall stats */
	if (recall_percent > 0)
	{
		g_stats.recall_at_1 = (g_stats.recall_at_1 + recall_percent) / 2;
		g_stats.recall_at_10 = (g_stats.recall_at_10 + recall_percent) / 2;
		g_stats.recall_at_100 = (g_stats.recall_at_100 + recall_percent) / 2;
	}
}

static void __attribute__((unused))
neurondb_update_cache_stats(bool hit)
{
	if (hit)
		g_stats.cache_hits++;
	else
		g_stats.cache_misses++;
}

static void __attribute__((unused))
neurondb_update_rebuild_stats(void)
{
	g_stats.index_rebuilds++;
}
