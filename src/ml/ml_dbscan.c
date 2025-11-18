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

	elog(DEBUG1, "neurondb: DBSCAN clustering on %s.%s (eps=%.4f, min_pts=%d)",
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

	elog(DEBUG1, "neurondb: DBSCAN found %d clusters (%d noise points)",
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


/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for Dbscan
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
void neurondb_gpu_register_dbscan_model(void);

void
neurondb_gpu_register_dbscan_model(void)
{
	elog(DEBUG1, "Dbscan GPU Model Ops registration skipped - not yet implemented");
}
