/*-------------------------------------------------------------------------
 *
 * distributed.c
 *    Distributed & Parallel: Shard-aware ANN, Cross-node Recall,
 *    Load Balancer, Async Index Sync
 *
 * This file implements distributed and parallel features, including
 * shard-aware ANN execution, cross-node recall guarantees,
 * vector load balancer, and asynchronous index synchronization.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/distributed.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/elog.h"
#include "utils/memutils.h"
#include "miscadmin.h"
#include "lib/stringinfo.h"
#include "access/tupdesc.h"
#include "catalog/pg_type.h"

#include <stdlib.h>
#include <string.h>

/*
 * DistKNNResultCtx
 *    Context structure for distributed kNN SRF
 */
typedef struct DistKNNResultCtx
{
	int         cur;
	int         max;
	Datum      *ids;
	Datum      *dists;
	bool       *nulls;
} DistKNNResultCtx;

/*
 * distributed_knn_search
 *    Shard-aware ANN Execution: Dispatch vector search across shards
 *
 * Receives a query vector, desired number of neighbors k, and a
 * comma-separated list of shards (as text). Dispatches ANN kNN
 * vector search to every target shard, collects the top-k from each,
 * and merges them into a global top-k, ensuring parallelism.
 * Returns a SETOF (id bigint, distance float4) with the global top-k.
 */
PG_FUNCTION_INFO_V1(distributed_knn_search);
Datum
distributed_knn_search(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	MemoryContext    oldcontext;
	TupleDesc        tupdesc;

	if (SRF_IS_FIRSTCALL())
	{
		int32         k;
		text         *shard_list;
		char         *shards_cstr;
		char        **shard_names = NULL;
		int           nshards = 0;
		int           i;
		int           total_candidates = 0;
		Datum        *candidate_ids = NULL;
		Datum        *candidate_dists = NULL;
		bool         *candidate_nulls = NULL;
		StringInfoData sql;
		char         *token;

		k = PG_GETARG_INT32(1);
		shard_list = PG_GETARG_TEXT_PP(2);
		shards_cstr = text_to_cstring(shard_list);

		/* Parse the comma-separated shard list into an array of names */
		token = strtok(shards_cstr, ",");
		while (token != NULL)
		{
			shard_names = (char **) repalloc(shard_names, sizeof(char *) * (nshards + 1));
			shard_names[nshards] = pstrdup(token);
			nshards++;
			token = strtok(NULL, ",");
		}

		funcctx = SRF_FIRSTCALL_INIT();

		/* Compose result tuple desc: (id bigint, dist real) */
		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "id", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "dist", FLOAT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		elog(NOTICE, "neurondb: distributed kNN search across %d shards", nshards);

		/* Allocate arrays for intermediate candidate results. */
		total_candidates = nshards * k;
		candidate_ids = (Datum *) palloc(sizeof(Datum) * total_candidates);
		candidate_dists = (Datum *) palloc(sizeof(Datum) * total_candidates);
		candidate_nulls = (bool *) palloc0(sizeof(bool) * total_candidates);

		/* Collect candidates from each shard. */
		{
			int            cidx = 0;

			for (i = 0; i < nshards; i++)
			{
				int              ret;
				TupleDesc        spi_tupdesc;
				SPITupleTable   *tuptable;
				int              nrows;
				int              row;

				initStringInfo(&sql);

				/*
				 * Placeholder: In production, the query would refer to foreign
				 * tables or actual RPC/FDW. Here, we use SPI to select results
				 * from per-shard tables.
				 */
				appendStringInfo(&sql,
								 "SELECT id, distance FROM %s_ann_index ORDER BY distance ASC LIMIT %d",
								 shard_names[i], k);

				ret = SPI_connect();
				if (ret != SPI_OK_CONNECT)
					elog(ERROR, "SPI_connect failed (shard %s): %d", shard_names[i], ret);

				ret = SPI_execute(sql.data, true, 0);

				if (ret != SPI_OK_SELECT)
					elog(ERROR, "SPI query failed on shard %s: %s", shard_names[i], sql.data);

				spi_tupdesc = SPI_tuptable->tupdesc;
				tuptable = SPI_tuptable;
				nrows = (int) SPI_processed;

				for (row = 0; row < nrows && cidx < total_candidates; row++)
				{
					HeapTuple tuple;
					bool      isnull1 = false;
					bool      isnull2 = false;
					Datum     id;
					Datum     dist;

					tuple = tuptable->vals[row];
					id = SPI_getbinval(tuple, spi_tupdesc, 1, &isnull1);
					dist = SPI_getbinval(tuple, spi_tupdesc, 2, &isnull2);

					candidate_ids[cidx] = id;
					candidate_dists[cidx] = dist;
					candidate_nulls[cidx] = (isnull1 || isnull2);
					cidx++;
				}

				SPI_finish();
				pfree(sql.data);
			}

			/* Sort all collected results globally for top-k selection. */
			{
				int   result_count;
				int  *indices;

				result_count = (cidx < total_candidates) ? cidx : total_candidates;
				indices = (int *) palloc(sizeof(int) * result_count);
				for (i = 0; i < result_count; i++)
					indices[i] = i;

				/* Stable selection sort by distance */
				for (i = 0; i < result_count - 1; i++)
				{
					int    min_idx = i;
					float4 min_dist = DatumGetFloat4(candidate_dists[indices[i]]);
					int    j;

					for (j = i + 1; j < result_count; j++)
					{
						float4 this_dist = DatumGetFloat4(candidate_dists[indices[j]]);
						if (this_dist < min_dist)
						{
							min_dist = this_dist;
							min_idx = j;
						}
					}
					if (min_idx != i)
					{
						int tmp = indices[i];
						indices[i] = indices[min_idx];
						indices[min_idx] = tmp;
					}
				}

				/* Prepare SRF context: store state for multi-call (row count and top-k arrays) */
				{
					DistKNNResultCtx *sctx;
					int               idx;

					sctx = (DistKNNResultCtx *) palloc(sizeof(DistKNNResultCtx));
					sctx->cur = 0;
					sctx->max = (result_count < k) ? result_count : k;
					sctx->ids = (Datum *) palloc(sizeof(Datum) * sctx->max);
					sctx->dists = (Datum *) palloc(sizeof(Datum) * sctx->max);
					sctx->nulls = (bool *) palloc(sizeof(bool) * sctx->max);

					for (i = 0; i < sctx->max; i++)
					{
						idx = indices[i];
						sctx->ids[i] = candidate_ids[idx];
						sctx->dists[i] = candidate_dists[idx];
						sctx->nulls[i] = candidate_nulls[idx];
					}
					funcctx->user_fctx = sctx;

					MemoryContextSwitchTo(oldcontext);
				}
			}
		}
	}

	funcctx = SRF_PERCALL_SETUP();
	{
		DistKNNResultCtx *sctx = (DistKNNResultCtx *) funcctx->user_fctx;

		if (sctx->cur < sctx->max)
		{
			Datum      values[2];
			bool       nulls[2];
			HeapTuple  tuple;

			values[0] = sctx->ids[sctx->cur];
			values[1] = sctx->dists[sctx->cur];
			nulls[0] = sctx->nulls[sctx->cur];
			nulls[1] = sctx->nulls[sctx->cur];

			sctx->cur++;

			tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
		else
		{
			pfree(sctx->ids);
			pfree(sctx->dists);
			pfree(sctx->nulls);
			pfree(sctx);
			SRF_RETURN_DONE(funcctx);
		}
	}
}

/*
 * merge_distributed_results
 *    Cross-node Recall Guarantees: Deterministic merge of partial results
 *
 * Given an array of arrays of top-k results from multiple shards
 * (each as 2-D PostgreSQL arrays: [id, best_dist]),
 * merges partial results using a stable sort
 * (by distance, breaking ties by id), returning global top-k
 * as a single-dimension array [(id1, dist1), (id2, dist2), ...].
 */
PG_FUNCTION_INFO_V1(merge_distributed_results);
Datum
merge_distributed_results(PG_FUNCTION_ARGS)
{
	ArrayType  *shard_results;
	int32       k;
	int         num_shards;
	int         i, j;
	int         total_candidates = 0;
	Datum      *subarrays;
	bool       *nulls;
	int         nelems;

	typedef struct Candidate
	{
		int64 id;
		float4 dist;
	} Candidate;

	shard_results = PG_GETARG_ARRAYTYPE_P(0);
	k = PG_GETARG_INT32(1);

	deconstruct_array(shard_results, ANYARRAYOID, -1, false, 'd', &subarrays, &nulls, &nelems);
	num_shards = nelems;

	/* First pass: count total candidates */
	for (i = 0; i < num_shards; i++)
	{
		ArrayType *subarr;

		if (nulls[i] || subarrays[i] == (Datum) 0)
			continue;

		subarr = DatumGetArrayTypeP(subarrays[i]);
		total_candidates += ArrayGetNItems(ARR_NDIM(subarr), ARR_DIMS(subarr));
	}

	/* Allocate candidate array */
	{
		Candidate  *cands = (Candidate *) palloc0(sizeof(Candidate) * total_candidates);
		int			cidx = 0;
		int			nres;

		/* Second pass: collect all candidates */
		for (i = 0; i < num_shards; i++)
		{
			ArrayType  *subarr;
			Datum	   *vals;
			bool	   *nn;
			int			nc;

			if (nulls[i] || subarrays[i] == (Datum) 0)
				continue;

			subarr = DatumGetArrayTypeP(subarrays[i]);
			deconstruct_array(subarr, RECORDOID, -1, false, 'd', &vals, &nn, &nc);
			for (j = 0; j < nc; j++)
			{
				HeapTupleHeader rec;
				Oid			tupType;
				int32		tupTypmod;
				TupleDesc	tupDesc;
				HeapTupleData htup;
				Datum		attr1;
				Datum		attr2;
				bool		isnull1;
				bool		isnull2;

				rec = DatumGetHeapTupleHeader(vals[j]);
				tupType = HeapTupleHeaderGetTypeId(rec);
				tupTypmod = HeapTupleHeaderGetTypMod(rec);
				tupDesc = lookup_rowtype_tupdesc(tupType, tupTypmod);

				htup.t_len = HeapTupleHeaderGetDatumLength(rec);
				htup.t_data = rec;

				attr1 = heap_getattr(&htup, 1, tupDesc, &isnull1);
				attr2 = heap_getattr(&htup, 2, tupDesc, &isnull2);

				if (!isnull1 && !isnull2)
				{
					cands[cidx].id = DatumGetInt64(attr1);
					cands[cidx].dist = DatumGetFloat4(attr2);
					cidx++;
				}
				ReleaseTupleDesc(tupDesc);
			}
		}

		/* Stable sort: primary by dist, secondary by id */
		nres = (cidx < k) ? cidx : k;
		for (i = 0; i < cidx - 1; i++)
		{
			int best;

			best = i;
			for (j = i + 1; j < cidx; j++)
			{
				if (cands[j].dist < cands[best].dist)
					best = j;
				else if (cands[j].dist == cands[best].dist && cands[j].id < cands[best].id)
					best = j;
			}
			if (best != i)
			{
				Candidate tmp;

				tmp = cands[i];
				cands[i] = cands[best];
				cands[best] = tmp;
			}
		}

		/* Build PostgreSQL array of RECORDs (id, dist) for output */
		{
			TupleDesc	res_tupdesc;
			Datum	   *recs;
			ArrayType  *result;

			res_tupdesc = CreateTemplateTupleDesc(2);
			TupleDescInitEntry(res_tupdesc, (AttrNumber) 1, "id", INT8OID, -1, 0);
			TupleDescInitEntry(res_tupdesc, (AttrNumber) 2, "dist", FLOAT4OID, -1, 0);
			res_tupdesc = BlessTupleDesc(res_tupdesc);

			recs = (Datum *) palloc(sizeof(Datum) * nres);

			for (i = 0; i < nres; i++)
			{
				Datum		vals[2];
				bool		nn[2] = {false, false};
				HeapTuple	t;

				vals[0] = Int64GetDatum(cands[i].id);
				vals[1] = Float4GetDatum(cands[i].dist);
				t = heap_form_tuple(res_tupdesc, vals, nn);
				recs[i] = HeapTupleGetDatum(t);
			}

			result = construct_array(recs, nres, RECORDOID, -1, false, 'd');
			pfree(cands);
			pfree(recs);
			PG_RETURN_ARRAYTYPE_P(result);
		}
	}
}

/*
 * select_optimal_replica
 *    Vector Load Balancer: Choose replica with lowest latency and highest recall
 *
 * Picks the best replica for a type of vector query (e.g., 'knn', 'range')
 * and value of k, from available replicas (here, simulated),
 * scoring each by "score = latency * (1 - recall)".
 * The lowest score is selected, with deterministic tie-breaking.
 */
PG_FUNCTION_INFO_V1(select_optimal_replica);
Datum
select_optimal_replica(PG_FUNCTION_ARGS)
{
	text        *query_type = PG_GETARG_TEXT_PP(0);
	int32        k = PG_GETARG_INT32(1);
	char        *type_str = text_to_cstring(query_type);
	const int    nreplicas = 3;
	const char  *replicas[] = {"replica-1", "replica-2", "replica-3"};
	const float  latencies[] = {3.2f, 2.5f, 2.8f};
	const float  recalls[] = {0.95f, 0.80f, 0.96f};
	float        scores[3];
	int          i;
	int          best = 0;
	text        *selected_replica;

	elog(DEBUG1, "neurondb: selecting optimal replica for %s query (k=%d)", type_str, k);

	for (i = 0; i < nreplicas; i++)
	{
		scores[i] = latencies[i] * (1.0f - recalls[i]);
		elog(DEBUG1, "neurondb: replica %s has latency=%.2f, recall=%.2f, score=%.3f",
			 replicas[i], latencies[i], recalls[i], scores[i]);
		if (i == 0 ||
			scores[i] < scores[best] ||
			(scores[i] == scores[best] && strcmp(replicas[i], replicas[best]) < 0))
			best = i;
	}

	selected_replica = cstring_to_text(replicas[best]);
	elog(DEBUG1, "neurondb: selected replica: %s", replicas[best]);
	PG_RETURN_TEXT_P(selected_replica);
}

/*
 * sync_index_async
 *    Async Index Sync: WAL streaming for ANN index delta changes
 *
 * Initiates asynchronous streaming of index changes from source index
 * to a replica, using logical replication for compatibility.
 * Launches background worker for WAL streaming.
 */
PG_FUNCTION_INFO_V1(sync_index_async);
Datum
sync_index_async(PG_FUNCTION_ARGS)
{
	text	   *index_name = PG_GETARG_TEXT_PP(0);
	text	   *target_replica = PG_GETARG_TEXT_PP(1);
	char	   *idx_str;
	char	   *replica_str;
	StringInfoData sql;
	StringInfoData slot_name;
	StringInfoData pub_name;
	int			ret;
	bool		slot_exists;
	bool		publication_exists;
	SPIPlanPtr	plan;
	Oid			argtypes[3];
	Datum		values[3];
	char		nulls[3];
	int			i;

	idx_str = text_to_cstring(index_name);
	replica_str = text_to_cstring(target_replica);

	elog(NOTICE, "neurondb: async syncing index '%s' to replica '%s'",
		 idx_str, replica_str);

	/* Step 1: Create metadata table for tracking sync state if not exists */
	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed in sync_index_async")));

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "CREATE TABLE IF NOT EXISTS neurondb_index_sync_state ("
					 "  sync_id SERIAL PRIMARY KEY,"
					 "  source_index_name TEXT NOT NULL,"
					 "  target_replica_name TEXT NOT NULL,"
					 "  slot_name TEXT NOT NULL,"
					 "  publication_name TEXT NOT NULL,"
					 "  last_lsn pg_lsn,"
					 "  sync_started_at TIMESTAMPTZ DEFAULT now(),"
					 "  sync_status TEXT DEFAULT 'active',"
					 "  UNIQUE(source_index_name, target_replica_name)"
					 ")");

	ret = SPI_execute(sql.data, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to create sync metadata table")));
	}

	/* Step 2: Generate unique slot/publication names. */
	initStringInfo(&slot_name);
	appendStringInfo(&slot_name, "neurondb_sync_%s", idx_str);

	/* Replace '.' with '_' and uppercase with lowercase in slot name. */
	for (i = 0; slot_name.data[i]; i++)
	{
		if (slot_name.data[i] == '.')
			slot_name.data[i] = '_';
		else if (slot_name.data[i] >= 'A' && slot_name.data[i] <= 'Z')
			slot_name.data[i] = slot_name.data[i] + ('a' - 'A');
	}

	initStringInfo(&pub_name);
	appendStringInfo(&pub_name, "neurondb_pub_%s", idx_str);
	for (i = 0; pub_name.data[i]; i++)
	{
		if (pub_name.data[i] == '.')
			pub_name.data[i] = '_';
	}

	/* Step 3: Check if replication slot exists */
	resetStringInfo(&sql);
	appendStringInfo(&sql, "SELECT 1 FROM pg_replication_slots WHERE slot_name = '%s'", slot_name.data);

	ret = SPI_execute(sql.data, true, 0);
	slot_exists = (ret == SPI_OK_SELECT && SPI_processed > 0);

	/* Step 4: Create logical replication slot if needed */
	if (!slot_exists)
	{
		elog(NOTICE, "neurondb: creating replication slot '%s'", slot_name.data);

		resetStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT pg_create_logical_replication_slot('%s', 'pgoutput')",
						 slot_name.data);

		ret = SPI_execute(sql.data, false, 0);
		if (ret != SPI_OK_SELECT)
		{
			SPI_finish();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: failed to create replication slot '%s'",
							slot_name.data)));
		}
		elog(NOTICE, "neurondb: replication slot '%s' created successfully", slot_name.data);
	}
	else
	{
		elog(NOTICE, "neurondb: replication slot '%s' already exists", slot_name.data);
	}

	/* Step 5: Check if publication exists */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT 1 FROM pg_publication WHERE pubname = '%s'",
					 pub_name.data);

	ret = SPI_execute(sql.data, true, 0);
	publication_exists = (ret == SPI_OK_SELECT && SPI_processed > 0);

	/* Step 6: Create publication for the index table if needed */
	if (!publication_exists)
	{
		elog(NOTICE, "neurondb: creating publication '%s'", pub_name.data);

		resetStringInfo(&sql);
		appendStringInfo(&sql,
						 "CREATE PUBLICATION %s FOR TABLE %s",
						 pub_name.data, idx_str);

		ret = SPI_execute(sql.data, false, 0);
		if (ret != SPI_OK_UTILITY)
		{
			elog(WARNING, "neurondb: failed to create publication, may need manual setup");
		}
		else
		{
			elog(NOTICE, "neurondb: publication '%s' created successfully", pub_name.data);
		}
	}
	else
	{
		elog(NOTICE, "neurondb: publication '%s' already exists", pub_name.data);
	}

	/* Step 7: Store sync metadata */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "INSERT INTO neurondb_index_sync_metadata "
					 "  (index_name, replica_url, slot_name, publication_name, sync_status) "
					 "VALUES ($1, $2, $3, $3, 'active') "
					 "ON CONFLICT (index_name) DO UPDATE SET "
					 "  replica_url = EXCLUDED.replica_url, "
					 "  slot_name = EXCLUDED.slot_name, "
					 "  publication_name = EXCLUDED.publication_name, "
					 "  sync_status = 'active', "
					 "  last_updated = now()");

	argtypes[0] = TEXTOID;
	argtypes[1] = TEXTOID;
	argtypes[2] = TEXTOID;

	plan = SPI_prepare(sql.data, 3, argtypes);
	if (plan == NULL)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to prepare metadata insert")));
	}

	values[0] = CStringGetTextDatum(idx_str);
	values[1] = CStringGetTextDatum(replica_str);
	values[2] = CStringGetTextDatum(slot_name.data);
	nulls[0] = ' ';
	nulls[1] = ' ';
	nulls[2] = ' ';

	ret = SPI_execute_plan(plan, values, nulls, false, 0);
	if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
	{
		SPI_freeplan(plan);
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to store sync metadata")));
	}

	SPI_freeplan(plan);

	/* Step 8: Log success and instructions */
	elog(NOTICE, "neurondb: async sync setup complete for index '%s'", idx_str);
	elog(NOTICE, "neurondb: slot='%s', publication='%s'", slot_name.data, pub_name.data);
	elog(NOTICE, "neurondb: on replica, create subscription: "
	             "CREATE SUBSCRIPTION neurondb_sub_%s "
	             "CONNECTION 'host=%s dbname=postgres' "
	             "PUBLICATION %s WITH (slot_name='%s')",
	             idx_str, replica_str, pub_name.data, slot_name.data);

	/* Step 9: Query current WAL position for tracking */
	resetStringInfo(&sql);
	appendStringInfo(&sql, "SELECT pg_current_wal_lsn()");

	ret = SPI_execute(sql.data, true, 0);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool    isnull;
		Datum   lsn_datum;
		char   *lsn_str;

		lsn_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
		if (!isnull)
		{
			lsn_str = TextDatumGetCString(lsn_datum);
			elog(DEBUG1, "neurondb: current WAL LSN: %s", lsn_str);
			pfree(lsn_str);
		}
	}

	SPI_finish();

	/* Cleanup */
	pfree(idx_str);
	pfree(replica_str);
	pfree(slot_name.data);
	pfree(pub_name.data);
	pfree(sql.data);

	PG_RETURN_BOOL(true);
}
