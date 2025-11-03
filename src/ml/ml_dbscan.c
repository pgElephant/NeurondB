/*
 * ml_dbscan.c
 *     DBSCAN (Density-Based Spatial Clustering) implementation
 *
 * DBSCAN is a density-based clustering algorithm that groups together points
 * that are closely packed together.
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"

#include "neurondb_ml.h"
#include "neurondb_simd.h"

/* DBSCAN constants */
#define DBSCAN_NOISE -1
#define DBSCAN_UNDEFINED -2

/* DBSCAN state */
typedef struct
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
 * Helper: region query (find neighbors within eps)
 */
static int *
dbscan_region_query(const DBSCANState *state, int idx, int *neighbor_count)
{
	int		   *neighbors;
	int			capacity;
	int			count;
	int			i;

	capacity = 16;
	neighbors = (int *) palloc(sizeof(int) * capacity);
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
				neighbors = (int *) repalloc(neighbors, sizeof(int) * capacity);
			}
			neighbors[count++] = i;
		}
	}

	*neighbor_count = count;
	return neighbors;
}

/*
 * Helper: expand cluster
 */
static void
dbscan_expand_cluster(DBSCANState *state, int point_idx, int *neighbors, int neighbor_count, int cluster_id)
{
	int			i;
	int		   *seeds;
	int			seed_count;
	int			seed_idx;

	/* Assign cluster */
	state->labels[point_idx] = cluster_id;

	/* Copy neighbors to seed list */
	seeds = (int *) palloc(sizeof(int) * neighbor_count);
	memcpy(seeds, neighbors, sizeof(int) * neighbor_count);
	seed_count = neighbor_count;
	seed_idx = 0;

	/* Process seeds */
	while (seed_idx < seed_count)
	{
		int			current;
		int		   *current_neighbors;
		int			current_neighbor_count;
		int			j;

		current = seeds[seed_idx];
		seed_idx++;

		/* Skip already processed */
		if (state->labels[current] == cluster_id)
			continue;

		/* If noise, mark as border point */
		if (state->labels[current] == DBSCAN_NOISE)
		{
			state->labels[current] = cluster_id;
			continue;
		}

		/* Assign cluster */
		state->labels[current] = cluster_id;

		/* Find neighbors of current */
		current_neighbors = dbscan_region_query(state, current, &current_neighbor_count);

		/* If core point, add its neighbors to seeds */
		if (current_neighbor_count >= state->min_pts)
		{
			for (j = 0; j < current_neighbor_count; j++)
			{
				int			neighbor;

				neighbor = current_neighbors[j];
				if (state->labels[neighbor] == DBSCAN_UNDEFINED)
				{
					/* Add to seeds */
					seeds = (int *) repalloc(seeds, sizeof(int) * (seed_count + 1));
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
	text		*table_name;
	text		*column_name;
	double		eps;
	int			min_pts;
	char		*tbl_str;
	char		*col_str;
	DBSCANState	state;
	int			i;
	ArrayType	*out_array;
	Datum		*out_datums;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	/* Parse arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	eps = PG_GETARG_FLOAT8(2);
	if (PG_NARGS() >= 4)
		min_pts = PG_GETARG_INT32(3);
	else
		min_pts = 5;  /* Default: 5 */

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

	/* Fetch vectors from table */
	state.data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &state.nvec, &state.dim);
	if (state.nvec == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found in table")));

	state.eps = eps;
	state.min_pts = min_pts;
	state.next_cluster = 0;

	/* Initialize all labels as undefined */
	state.labels = (int *) palloc0(sizeof(int) * state.nvec);
	for (i = 0; i < state.nvec; i++)
		state.labels[i] = DBSCAN_UNDEFINED;

	/* Main DBSCAN algorithm */
	for (i = 0; i < state.nvec; i++)
	{
		int *neighbors;
		int neighbor_count;

		/* Skip already processed points */
		if (state.labels[i] != DBSCAN_UNDEFINED)
			continue;

		/* Find neighbors */
		neighbors = dbscan_region_query(&state, i, &neighbor_count);

		/* Check if core point */
		if (neighbor_count < state.min_pts)
		{
			/* Mark as noise */
			state.labels[i] = DBSCAN_NOISE;
		}
		else
		{
			/* Start new cluster */
			dbscan_expand_cluster(&state, i, neighbors, neighbor_count, state.next_cluster);
			state.next_cluster++;
		}

		pfree(neighbors);
	}

	elog(DEBUG1, "neurondb: DBSCAN found %d clusters (%d noise points)",
		 state.next_cluster, state.nvec);

	/* Build result array */
	out_datums = (Datum *) palloc(sizeof(Datum) * state.nvec);
	for (i = 0; i < state.nvec; i++)
		out_datums[i] = Int32GetDatum(state.labels[i]);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);
	out_array = construct_array(out_datums, state.nvec, INT4OID,
								typlen, typbyval, typalign);

	/* Cleanup */
	for (i = 0; i < state.nvec; i++)
		pfree(state.data[i]);
	pfree(state.data);
	pfree(state.labels);
	pfree(out_datums);
	pfree(tbl_str);
	pfree(col_str);

	PG_RETURN_ARRAYTYPE_P(out_array);
}

