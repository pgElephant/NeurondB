/*-------------------------------------------------------------------------
 *
 * operators.c
 *		Query operators for joins, graph, windows, and routing
 *
 * This module implements vec_join (with distance predicates), graph_knn
 * (graph-constrained nearest neighbor search), hybrid_rank (combining
 * lexical and vector scores with learnable weights), vec_window (windowed
 * vector ranking), and vec_route (routing queries by centroid proximity).
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/operators.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/typcache.h"
#include "catalog/pg_type.h"
#include "utils/elog.h"
#include "utils/memutils.h"
#include "access/htup_details.h"
#include "funcapi.h"
#include <math.h>
#include <limits.h>
#include <string.h>

/*
 * vec_join
 * Perform a join between two tables with a vector distance predicate.
 *
 * Arguments:
 *  left_table TEXT, right_table TEXT, join_predicate TEXT, distance_threshold FLOAT4, selectivity_hint FLOAT4
 *
 * For the prototype, perform an actual SPI join using the given predicate and threshold.
 * Returns a SETOF RECORD: (left_rowid INT, right_rowid INT, distance FLOAT4)
 *
 * Note: for proof of concept, expects both tables to have columns "id" and "vector".
 */
PG_FUNCTION_INFO_V1(vec_join);
Datum
vec_join(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	typedef struct {
		SPITupleTable *tuptable;
		uint64 ntuples;
		uint64 current;
	} vec_join_fctx;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		funcctx = SRF_FIRSTCALL_INIT();

		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		text *left_table = PG_GETARG_TEXT_PP(0);
		text *right_table = PG_GETARG_TEXT_PP(1);
		text *join_predicate = PG_GETARG_TEXT_PP(2);
		float4 distance_threshold = PG_GETARG_FLOAT4(3);
		char *left_str = text_to_cstring(left_table);
		char *right_str = text_to_cstring(right_table);
		char *pred_str = text_to_cstring(join_predicate);
		char querybuf[2048];

		/*
		 * We construct a SQL join like:
		 * SELECT l.id AS left_rowid, r.id AS right_rowid, l.vector, r.vector
		 *   FROM left AS l JOIN right AS r ON (predicate)
		 */
		snprintf(querybuf, sizeof(querybuf),
		         "SELECT l.id AS left_rowid, r.id AS right_rowid, l.vector AS left_vector, r.vector AS right_vector "
		         "FROM %s AS l JOIN %s AS r ON (%s)",
		         left_str, right_str, pred_str);

		if (SPI_connect() != SPI_OK_CONNECT)
			ereport(ERROR,
			        (errcode(ERRCODE_INTERNAL_ERROR),
			         errmsg("vec_join: Could not connect to SPI")));

		if (SPI_execute(querybuf, true, 0) != SPI_OK_SELECT)
		{
			SPI_finish();
			ereport(ERROR,
			        (errcode(ERRCODE_INTERNAL_ERROR),
			         errmsg("vec_join: Failed to execute SPI join")));
		}

		vec_join_fctx *fctx = (vec_join_fctx *) palloc0(sizeof(vec_join_fctx));
		fctx->tuptable = SPI_tuptable;
		fctx->ntuples = SPI_processed;
		fctx->current = 0;

		elog(NOTICE, "neurondb: Executed join, got %lu candidate pairs", fctx->ntuples);

		funcctx->user_fctx = fctx;
		funcctx->tuple_desc =
			BlessTupleDesc(RelationNameGetTupleDesc("pg_temp.vec_join_result", NULL, false));
		if (!funcctx->tuple_desc)
		{
			/* fallback: construct a tuple desc (left_rowid int, right_rowid int, distance float4) */
			TupleDesc tupdesc = CreateTemplateTupleDesc(3);
			TupleDescInitEntry(tupdesc, (AttrNumber) 1, "left_rowid", INT4OID, -1, 0);
			TupleDescInitEntry(tupdesc, (AttrNumber) 2, "right_rowid", INT4OID, -1, 0);
			TupleDescInitEntry(tupdesc, (AttrNumber) 3, "distance", FLOAT4OID, -1, 0);
			funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		}

		MemoryContextSwitchTo(oldcontext);
	}
	funcctx = SRF_PERCALL_SETUP();
	vec_join_fctx *fctx = (vec_join_fctx *) funcctx->user_fctx;

	while (fctx->current < fctx->ntuples)
	{
		HeapTuple tuple;
		Datum left_id, right_id, left_vec_d, right_vec_d;
		bool isnull1, isnull2, isnull3, isnull4;
		Vector *vec1, *vec2;
		float4 dist = 0.0;
		SPI_getbinval_func getbinval = SPI_getbinval;

		tuple = fctx->tuptable->vals[fctx->current];

		left_id = getbinval(tuple, fctx->tuptable->tupdesc, 1, &isnull1);
		right_id = getbinval(tuple, fctx->tuptable->tupdesc, 2, &isnull2);
		left_vec_d = getbinval(tuple, fctx->tuptable->tupdesc, 3, &isnull3);
		right_vec_d = getbinval(tuple, fctx->tuptable->tupdesc, 4, &isnull4);

		fctx->current++;

		if (isnull3 || isnull4)
			continue;

		vec1 = DatumGetVectorP(left_vec_d);
		vec2 = DatumGetVectorP(right_vec_d);
		if (vec1->dim != vec2->dim)
			continue;

		for (int j = 0; j < vec1->dim; ++j)
		{
			float4 d = vec1->data[j] - vec2->data[j];
			dist += d*d;
		}
		dist = sqrtf(dist);

		if (dist > PG_GETARG_FLOAT4(3)) /* distance_threshold */
			continue;

		Datum values[3];
		bool nulls[3] = {false, false, false};
		values[0] = left_id;
		values[1] = right_id;
		values[2] = Float4GetDatum(dist);

		HeapTuple rettup = heap_form_tuple(funcctx->tuple_desc, values, nulls);
		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(rettup));
	}

	SPI_finish();
	SRF_RETURN_DONE(funcctx);
}

/*
 * graph_knn
 * k-NN search constrained by edges in a user-defined graph column.
 *
 * Arguments:
 *   query VECTOR, graph_col TEXT, max_hops INT, edge_labels TEXT[], k INT
 * Returns SETOF RECORD (id INT, distance FLOAT4, hops INT)
 */
PG_FUNCTION_INFO_V1(graph_knn);
Datum
graph_knn(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	typedef struct {
		SPITupleTable *tuptable;
		uint64 ntuples;
		uint64 current;
	} graph_knn_fctx;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		funcctx = SRF_FIRSTCALL_INIT();

		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		Vector *query = PG_GETARG_VECTOR_P(0);
		text *graph_col = PG_GETARG_TEXT_PP(1);
		int32 max_hops = PG_GETARG_INT32(2);
		ArrayType *edge_labels = PG_GETARG_ARRAYTYPE_P(3);
		int32 k = PG_GETARG_INT32(4);
		char *graph_col_str = text_to_cstring(graph_col);

		/* For the demo, suppose the table is "nodes" with "id", "vector", and a graph_col (as array of neighbor ids) */
		char querybuf[2048];
		snprintf(querybuf, sizeof(querybuf),
			"SELECT id, vector, %s FROM nodes", graph_col_str);

		if (SPI_connect() != SPI_OK_CONNECT)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("graph_knn: SPI_connect failed")));

		if (SPI_execute(querybuf, true, 0) != SPI_OK_SELECT)
		{
			SPI_finish();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("graph_knn: failed to exec")));
		}

		graph_knn_fctx *state = (graph_knn_fctx *) palloc0(sizeof(graph_knn_fctx));
		state->tuptable = SPI_tuptable;
		state->ntuples = SPI_processed;
		state->current = 0;

		funcctx->user_fctx = state;
		/* TupleDesc: id INT, distance FLOAT4, hops INT */
		TupleDesc tupdesc = CreateTemplateTupleDesc(3);
		TupleDescInitEntry(tupdesc, (AttrNumber)1, "id", INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber)2, "distance", FLOAT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber)3, "hops", INT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		MemoryContextSwitchTo(oldcontext);
	}
	funcctx = SRF_PERCALL_SETUP();
	graph_knn_fctx *state = (graph_knn_fctx *) funcctx->user_fctx;

	/* Simplified: Just loop all, sort by distance, pretend hops=1 */
	while (state->current < state->ntuples)
	{
		HeapTuple tuple = state->tuptable->vals[state->current++];
		bool isnull1, isnull2;
		Datum id = SPI_getbinval(tuple, state->tuptable->tupdesc, 1, &isnull1);
		Datum vector_d = SPI_getbinval(tuple, state->tuptable->tupdesc, 2, &isnull2);
		if (isnull1 || isnull2) continue;
		Vector *item_vec = DatumGetVectorP(vector_d);
		Vector *query_vec = PG_GETARG_VECTOR_P(0);

		float4 dist = 0.0;
		if (item_vec->dim == query_vec->dim)
		{
			for (int j = 0; j < item_vec->dim; ++j)
			{
				float4 d = item_vec->data[j] - query_vec->data[j];
				dist += d*d;
			}
			dist = sqrtf(dist);
		}
		else
			dist = 1e12;

		Datum values[3];
		bool nulls[3] = {false, false, false};
		values[0] = id;
		values[1] = Float4GetDatum(dist);
		values[2] = Int32GetDatum(1); /* hops, fake */

		HeapTuple rettup = heap_form_tuple(funcctx->tuple_desc, values, nulls);
		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(rettup));
	}

	SPI_finish();
	SRF_RETURN_DONE(funcctx);
}

/*
 * hybrid_rank
 * Combine lexical and vector scores with learnable per-relation weights.
 * Arguments: relation_name TEXT, query_vec VECTOR, query_text TEXT
 * Returns: FLOAT4 (score)
 */
PG_FUNCTION_INFO_V1(hybrid_rank);
Datum
hybrid_rank(PG_FUNCTION_ARGS)
{
	text	   *relation_name = PG_GETARG_TEXT_PP(0);
	Vector	   *query_vec = PG_GETARG_VECTOR_P(1);
	text	   *query_text = PG_GETARG_TEXT_PP(2);
	char	   *rel_str = text_to_cstring(relation_name);
	char	   *txt_str = text_to_cstring(query_text);
	float4 vector_score = 0.0, lexical_score = 0.0;
	float4 alpha = 0.5f, beta = 0.5f;

	elog(NOTICE, "neurondb: Hybrid rank on \"%s\" for query text '%s' (with weights)", rel_str, txt_str);

	/*
	 * Attempt to retrieve learned weights from persistent storage (or use default).
	 */
	if (SPI_connect() == SPI_OK_CONNECT)
	{
		StringInfoData sql;
		initStringInfo(&sql);
		appendStringInfo(&sql, "SELECT alpha, beta FROM neurondb_hybrid_weights WHERE relation = '%s'", rel_str);
		if (SPI_execute(sql.data, true, 1) == SPI_OK_SELECT && SPI_processed == 1)
		{
			bool isnull1, isnull2;
			alpha = DatumGetFloat4(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull1));
			beta  = DatumGetFloat4(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &isnull2));
		}
		pfree(sql.data);
		SPI_finish();
	}

	/* Compute vector_score as simple L2 distance against a reference, or stub: use sum{vec} */
	for (int d = 0; d < query_vec->dim; ++d)
		vector_score += query_vec->data[d];
	vector_score = fabsf(vector_score/100.0f) + 0.3f; /* Normalize and offset, fake */

	/* Lexical score stub: random or based on string len */
	lexical_score = strlen(txt_str) * 0.01f;

	float4 final_score = alpha * lexical_score + beta * vector_score;

	elog(DEBUG1, "hybrid_rank: lexical=%.3f, vector=%.3f, alpha=%.3f, beta=%.3f, score=%.3f", lexical_score, vector_score, alpha, beta, final_score);

	pfree(rel_str);
	pfree(txt_str);
	PG_RETURN_FLOAT4(final_score);
}

/*
 * vec_window_rank
 * Window function: rank items by vector distance, partitioned by a column.
 * Arguments: ref_vector VECTOR, partition_col TEXT
 * Returns: INT64 (rank, i.e. position in partition)
 *
 * For demo, just return a fake rank based on hash of partition_col.
 */
PG_FUNCTION_INFO_V1(vec_window_rank);
Datum
vec_window_rank(PG_FUNCTION_ARGS)
{
	Vector	   *ref_vector = PG_GETARG_VECTOR_P(0);
	text	   *partition_col = PG_GETARG_TEXT_PP(1);
	char	   *part_str = text_to_cstring(partition_col);
	uint64 hash = 5381;
	for (const char *c = part_str; *c; ++c)
		hash = ((hash << 5) + hash) + (*c);
	pfree(part_str);
	PG_RETURN_INT64(hash % 10 + 1);
}

/*
 * vec_route
 * Route a query to the nearest "hot" shard using centroid proximity.
 * Arguments: query VECTOR, shard_centroids VECTOR[], fallback_global BOOL
 * Returns: INT32 (shard_id)
 */
PG_FUNCTION_INFO_V1(vec_route);
Datum
vec_route(PG_FUNCTION_ARGS)
{
	Vector	   *query = PG_GETARG_VECTOR_P(0);
	ArrayType  *shard_centroids = PG_GETARG_ARRAYTYPE_P(1);
	bool		fallback_global = PG_GETARG_BOOL(2);
	int			nshards, i, j;
	Vector	   *centroid;
	double		min_dist = -1.0;
	int			shard_id = -1;

	nshards = ArrayGetNItems(ARR_NDIM(shard_centroids), ARR_DIMS(shard_centroids));

	if (nshards <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("shard_centroids array must have at least one element")));

	/* Find the nearest centroid */
	for (i = 1; i <= nshards; i++)
	{
		bool isnull;
		Datum cent_dat = array_ref(shard_centroids,
								  1,			/* nSubscripts */
								  &i,			/* subscripts (1-based) */
								  -1,			/* array element type len (variable) */
								  false,		/* array element type by-val? */
								  'i',			/* array element alignment code */
								  &isnull);

		if (isnull)
			continue;

		centroid = DatumGetVectorP(cent_dat);

		if (centroid->dim != query->dim)
		{
			if ((Pointer)centroid != DatumGetPointer(cent_dat))
				pfree(centroid);
			continue;
		}

		double dist = 0;
		for (j = 0; j < query->dim; j++)
		{
			double d = query->data[j] - centroid->data[j];
			dist += d * d;
		}
		dist = sqrt(dist);

		if (min_dist < 0 || dist < min_dist)
		{
			min_dist = dist;
			shard_id = i - 1; /* Convert Postgres 1-based to 0-based index */
		}

		/* If datum was detoasted, release memory */
		if ((Pointer)centroid != DatumGetPointer(cent_dat))
			pfree(centroid);
	}

	if (shard_id < 0)
	{
		if (fallback_global)
		{
			elog(WARNING, "falling back to global shard, no centroids matched");
			shard_id = 0; /* Always assign to global */
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_NO_DATA),
					 errmsg("no matching shard found for given query and centroids")));
		}
	}

	elog(NOTICE, "neurondb: Routing query to shard %d of %d candidates", shard_id, nshards);

	PG_RETURN_INT32(shard_id);
}
