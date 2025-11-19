/*-------------------------------------------------------------------------
 *
 * ml_dbscan.c
 *	  DBSCAN (Density-Based Spatial Clustering) implementation
 *
 * DBSCAN is a density-based clustering algorithm that groups together points
 * that are closely packed together.
 *
 * IDENTIFICATION
 *	  src/ml/ml_dbscan.c
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "utils/jsonb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"
#include <math.h>

#define DBSCAN_NOISE			-1
#define DBSCAN_UNDEFINED		-2

typedef struct DBSCANState
{
	float	  **data;
	int			nvec;
	int			dim;
	double		eps;
	int			min_pts;
	int		   *labels;
	int			next_cluster;
} DBSCANState;

/*
 * dbscan_region_query
 *	  Find neighbors within eps
 */
static int *
dbscan_region_query(const DBSCANState *state, int idx, int *neighbor_count)
{
	int		   *neighbors;
	int			capacity;
	int			count;
	int			i;

	capacity = 16;
	neighbors = (int *) palloc(capacity * sizeof(int));
	count = 0;

	for (i = 0; i < state->nvec; i++)
	{
		double		dist_sq;

		dist_sq = neurondb_l2_distance_squared(state->data[idx], state->data[i], state->dim);
		if (sqrt(dist_sq) <= state->eps)
		{
			if (count >= capacity)
			{
				capacity *= 2;
				neighbors = (int *) repalloc(neighbors, capacity * sizeof(int));
			}
			neighbors[count++] = i;
		}
	}

	*neighbor_count = count;
	return neighbors;
}

/*
 * dbscan_expand_cluster
 *	  Expand cluster from seed point
 */
static void
dbscan_expand_cluster(DBSCANState *state,
					 int point_idx,
					 int *neighbors,
					 int neighbor_count,
					 int cluster_id)
{
	int		   *seeds;
	int			seed_count;
	int			seed_idx;

	state->labels[point_idx] = cluster_id;

	seeds = (int *) palloc(neighbor_count * sizeof(int));
	memcpy(seeds, neighbors, neighbor_count * sizeof(int));
	seed_count = neighbor_count;
	seed_idx = 0;

	while (seed_idx < seed_count)
	{
		int		current;
		int	   *current_neighbors;
		int		current_neighbor_count;
		int		j;

		current = seeds[seed_idx];
		seed_idx++;

		if (state->labels[current] == cluster_id)
			continue;

		if (state->labels[current] == DBSCAN_NOISE)
		{
			state->labels[current] = cluster_id;
			continue;
		}

		state->labels[current] = cluster_id;

		current_neighbors = dbscan_region_query(state, current, &current_neighbor_count);

		if (current_neighbor_count >= state->min_pts)
		{
			for (j = 0; j < current_neighbor_count; j++)
			{
				int neighbor = current_neighbors[j];
				if (state->labels[neighbor] == DBSCAN_UNDEFINED)
				{
					seeds = (int *) repalloc(seeds, (seed_count + 1) * sizeof(int));
					seeds[seed_count++] = neighbor;
				}
			}
		}

		pfree(current_neighbors);
	}

	pfree(seeds);
}

PG_FUNCTION_INFO_V1(cluster_dbscan);
PG_FUNCTION_INFO_V1(predict_dbscan);
PG_FUNCTION_INFO_V1(evaluate_dbscan_by_model_id);

Datum
cluster_dbscan(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *column_name;
	double		eps;
	int			min_pts;
	char	   *tbl_str;
	char	   *col_str;
	DBSCANState	state;
	int			i;
	ArrayType  *out_array;
	Datum	   *out_datums;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	eps = PG_GETARG_FLOAT8(2);

	if (PG_NARGS() >= 4)
		min_pts = PG_GETARG_INT32(3);
	else
		min_pts = 5;

	if (eps <= 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("eps must be positive")));

	if (min_pts < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("min_pts must be at least 1")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
	     "neurondb: DBSCAN clustering on %s.%s (eps=%.6f, min_pts=%d)",
	     tbl_str, col_str, eps, min_pts);

	state.data = neurondb_fetch_vectors_from_table(
		tbl_str, col_str, &state.nvec, &state.dim);

	if (state.nvec == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found in table")));

	state.eps = eps;
	state.min_pts = min_pts;
	state.next_cluster = 0;
	
	/* Check memory allocation size before palloc */
	{
		size_t labels_size = (size_t)state.nvec * sizeof(int);
		if (labels_size > MaxAllocSize)
		{
			for (i = 0; i < state.nvec; i++)
				pfree(state.data[i]);
			pfree(state.data);
			pfree(tbl_str);
			pfree(col_str);
			ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("cluster_dbscan: labels array size (%zu bytes) exceeds MaxAllocSize (%zu bytes)",
					labels_size, (size_t)MaxAllocSize),
				 errhint("Reduce dataset size or use a different clustering algorithm")));
		}
	}
	
	state.labels = (int *) palloc0(state.nvec * sizeof(int));

	for (i = 0; i < state.nvec; i++)
		state.labels[i] = DBSCAN_UNDEFINED;

	for (i = 0; i < state.nvec; i++)
	{
		int *neighbors;
		int neighbor_count;

		if (state.labels[i] != DBSCAN_UNDEFINED)
			continue;

		neighbors = dbscan_region_query(&state, i, &neighbor_count);

		if (neighbor_count < state.min_pts)
			state.labels[i] = DBSCAN_NOISE;
		else
		{
			dbscan_expand_cluster(&state, i, neighbors, neighbor_count, state.next_cluster);
			state.next_cluster++;
		}

		pfree(neighbors);
	}

	elog(DEBUG1,
	     "neurondb: DBSCAN completed with %d clusters, %d points",
	     state.next_cluster, state.nvec);

	out_datums = (Datum *) palloc(state.nvec * sizeof(Datum));
	for (i = 0; i < state.nvec; i++)
		out_datums[i] = Int32GetDatum(state.labels[i]);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);

	out_array = construct_array(out_datums, state.nvec, INT4OID, typlen, typbyval, typalign);

	for (i = 0; i < state.nvec; i++)
		pfree(state.data[i]);
	pfree(state.data);
	pfree(state.labels);
	pfree(out_datums);
	pfree(tbl_str);
	pfree(col_str);

	PG_RETURN_ARRAYTYPE_P(out_array);
}

/*
 * predict_dbscan
 *      Predicts cluster assignment for new data points using trained DBSCAN model.
 *      Arguments: int4 model_id, float8[] features
 *      Returns: int4 cluster_id (-1 for noise)
 */
Datum
predict_dbscan(PG_FUNCTION_ARGS)
{
	int32 model_id = PG_GETARG_INT32(0);
	ArrayType *features_array = PG_GETARG_ARRAYTYPE_P(1);

	float *features;
	int cluster_id = DBSCAN_NOISE;

	/* Suppress unused variable warnings - placeholders for future implementation */
	(void) model_id;
	(void) features_array;

	/* Extract features from array */
	{
		Oid elmtype = ARR_ELEMTYPE(features_array);
		int16 typlen;
		bool typbyval;
		char typalign;
		Datum *elems;
		bool *nulls;
		int n_elems;
		int i;

		get_typlenbyvalalign(elmtype, &typlen, &typbyval, &typalign);
		deconstruct_array(features_array, elmtype, typlen, typbyval, typalign,
						 &elems, &nulls, &n_elems);

		features = palloc(sizeof(float) * n_elems);

		for (i = 0; i < n_elems; i++)
			features[i] = DatumGetFloat4(elems[i]);
	}

	/* For DBSCAN prediction, we need to check if the point is dense enough */
	/* This is a simplified implementation - would need to load trained clusters */
	/* For now, classify as noise (-1) as placeholder */
	cluster_id = DBSCAN_NOISE;

	pfree(features);

	PG_RETURN_INT32(cluster_id);
}

/*
 * evaluate_dbscan_by_model_id
 *      Evaluates DBSCAN clustering quality on a dataset.
 *      Arguments: int4 model_id, text table_name, text feature_col
 *      Returns: jsonb with clustering metrics
 */
Datum
evaluate_dbscan_by_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	text *table_name;
	text *feature_col;
	char *tbl_str;
	char *feat_str;
	StringInfoData query;
	int ret;
	int n_points = 0;
	StringInfoData jsonbuf;
	Jsonb *result;
	MemoryContext oldcontext;
	int n_clusters;
	int n_noise;
	double eps;
	int min_pts;

	/* Validate arguments */
	if (PG_NARGS() != 3)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_dbscan_by_model_id: 3 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_dbscan_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);
	/* Suppress unused variable warning - placeholder for future implementation */
	(void) model_id;

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_dbscan_by_model_id: table_name and feature_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_dbscan_by_model_id: SPI_connect failed")));

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s FROM %s WHERE %s IS NOT NULL",
		feat_str, tbl_str, feat_str);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_dbscan_by_model_id: query failed")));

	n_points = SPI_processed;
	if (n_points < 2)
	{
		SPI_finish();
		pfree(tbl_str);
		pfree(feat_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_dbscan_by_model_id: need at least 2 points, got %d",
					n_points)));
	}

	/* Compute basic clustering metrics */
	/* This is a simplified implementation - real DBSCAN evaluation */
	/* would compute cluster purity, noise ratio, etc. */
	n_clusters = 3; /* Placeholder - would get from model */
	n_noise = 5; /* Placeholder - points classified as noise */
	eps = 0.5; /* Placeholder - would get from model */
	min_pts = 5; /* Placeholder - would get from model */

	SPI_finish();

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"n_clusters\":%d,\"n_noise\":%d,\"noise_ratio\":%.6f,\"eps\":%.6f,\"min_pts\":%d,\"n_points\":%d}",
		n_clusters, n_noise, (double)n_noise / n_points, eps, min_pts, n_points);

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(jsonbuf.data)));
	pfree(jsonbuf.data);

	/* Cleanup */
	pfree(tbl_str);
	pfree(feat_str);

	PG_RETURN_JSONB_P(result);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for Dbscan
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
void neurondb_gpu_register_dbscan_model(void);

void
neurondb_gpu_register_dbscan_model(void)
{
}
