/*-------------------------------------------------------------------------
 *
 * ml_kmeans.c
 *    K-Means clustering algorithm implementation
 *
 * Implements Lloyd's algorithm with K-Means++ initialization for
 * intelligent centroid seeding using D² weighting.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/ml/ml_kmeans.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

/*
 * kmeanspp_init
 *    K-means++ seeding (D^2 weighting).
 */
static void
kmeanspp_init(float **data, int nvec, int dim, int k, int *centroids)
{
	bool *selected;
	double *dist;
	int c, i, d;

	/* Assert: Internal invariants */
	Assert(data != NULL);
	Assert(centroids != NULL);
	Assert(nvec > 0);
	Assert(dim > 0);
	Assert(k > 0);
	Assert(k <= nvec);

	/* Defensive: Check for NULL pointers */
	if (data == NULL || centroids == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("kmeanspp_init: NULL pointer argument")));

	if (nvec <= 0 || dim <= 0 || k <= 0 || k > nvec)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("kmeanspp_init: invalid parameters (nvec=%d, dim=%d, k=%d)",
					nvec, dim, k)));

	selected = (bool *)palloc0(sizeof(bool) * nvec);
	dist = (double *)palloc(sizeof(double) * nvec);

	/* Defensive: Validate allocations */
	if (selected == NULL || dist == NULL)
	{
		if (selected)
			pfree(selected);
		if (dist)
			pfree(dist);
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate arrays in kmeanspp_init")));
	}

	/* Pick first centroid randomly */
	centroids[0] = rand() % nvec;
	selected[centroids[0]] = true;

	/* Compute initial distances from first centroid */
	for (i = 0; i < nvec; i++)
	{
		double acc = 0.0;

		for (d = 0; d < dim; d++)
		{
			float diff = data[i][d] - data[centroids[0]][d];
			acc += diff * diff;
		}
		dist[i] = acc;
	}

	/* Pick remaining centroids */
	for (c = 1; c < k; c++)
	{
		double sum = 0.0;
		double r;
		int picked = -1;

		for (i = 0; i < nvec; i++)
			if (!selected[i])
				sum += dist[i];

		r = ((double)rand() / (double)RAND_MAX) * sum;

		for (i = 0; i < nvec; i++)
		{
			if (selected[i])
				continue;
			r -= dist[i];
			if (r <= 0)
			{
				picked = i;
				break;
			}
		}
		if (picked < 0)
		{
			for (i = 0; i < nvec; i++)
			{
				if (!selected[i])
				{
					picked = i;
					break;
				}
			}
		}

		/* Safety check */
		if (picked < 0 || picked >= nvec)
		{
			pfree(dist);
			pfree(selected);
			elog(ERROR,
				"neurondb: k-means++ failed to select centroid "
				"%d",
				c);
		}

		centroids[c] = picked;
		selected[picked] = true;

		/* Update distances if new centroid is closer */
		for (i = 0; i < nvec; i++)
		{
			double acc = 0.0;
			for (d = 0; d < dim; d++)
				acc += (data[i][d] - data[picked][d])
					* (data[i][d] - data[picked][d]);
			if (acc < dist[i])
				dist[i] = acc;
		}
	}

	pfree(dist);
	pfree(selected);
}

/*
 * cluster_kmeans
 *    K-means actual implementation: Lloyd's algorithm with k-means++ seeding
 */
PG_FUNCTION_INFO_V1(cluster_kmeans);

Datum
cluster_kmeans(PG_FUNCTION_ARGS)
{
	text *table_name = PG_GETARG_TEXT_PP(0);
	text *vector_col = PG_GETARG_TEXT_PP(1);
	int32 num_clusters = PG_GETARG_INT32(2);
	int32 max_iters = PG_GETARG_INT32(3);

	char *tbl_str;
	char *col_str;
	int nvec = 0;
	int dim = 0;
	int *assignments = NULL;
	int *centroids_idx = NULL;
	float **data = NULL;
	float **centers = NULL;
	bool changed = false;
	int iter, i, c, d;

	CHECK_NARGS(4);

	/* Defensive: Check NULL inputs */
	if (table_name == NULL || vector_col == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("cluster_kmeans: table_name and vector_col cannot be NULL")));

	/* Defensive: Validate parameters */
	if (num_clusters < 1 || num_clusters > 1000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("num_clusters must be between 1 and 1000, got %d", num_clusters)));

	if (max_iters < 1 || max_iters > 10000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("max_iters must be between 1 and 10000, got %d", max_iters)));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_col);

	/* Defensive: Validate allocations */
	if (tbl_str == NULL || col_str == NULL)
	{
		if (tbl_str)
			pfree(tbl_str);
		if (col_str)
			pfree(col_str);
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate strings")));
	}

	/* Defensive: Validate string lengths */
	if (strlen(tbl_str) == 0 || strlen(tbl_str) > NAMEDATALEN ||
		strlen(col_str) == 0 || strlen(col_str) > NAMEDATALEN)
	{
		pfree(tbl_str);
		pfree(col_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_NAME),
				errmsg("cluster_kmeans: invalid table or column name length")));
	}

	elog(DEBUG1, "cluster_kmeans: Starting k-means clustering (k=%d, max_iters=%d)",
		num_clusters, max_iters);

	if (num_clusters <= 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("number of clusters must be at least "
				       "2")));
	if (max_iters < 1)
		max_iters = 100;

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (!data || nvec < num_clusters)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("not enough vectors for cluster count "
				       "(need >= %d, have %d)",
					num_clusters,
					nvec)));

	assignments = (int *)palloc0(sizeof(int) * nvec);
	centroids_idx = (int *)palloc(sizeof(int) * num_clusters);

	/* Initialize centroids using k-means++ */
	kmeanspp_init(data, nvec, dim, num_clusters, centroids_idx);

	centers = (float **)palloc(sizeof(float *) * num_clusters);
	for (c = 0; c < num_clusters; c++)
	{
		centers[c] = (float *)palloc(sizeof(float) * dim);
		memcpy(centers[c], data[centroids_idx[c]], sizeof(float) * dim);
	}

	changed = true;
	for (i = 0; i < nvec; i++)
		assignments[i] = -1;

	for (iter = 0; iter < max_iters && changed; iter++)
	{
		int *counts;

		changed = false;
		/* Assignment phase - SIMD optimized */
		for (i = 0; i < nvec; i++)
		{
			double min_dist = DBL_MAX;
			int best = -1;

			for (c = 0; c < num_clusters; c++)
			{
				/* Use SIMD-optimized L2 distance */
				double dist = neurondb_l2_distance_squared(
					data[i], centers[c], dim);

				if (dist < min_dist)
				{
					min_dist = dist;
					best = c;
				}
			}
			if (assignments[i] != best)
			{
				assignments[i] = best;
				changed = true;
			}
		}

		/* Update phase */
		for (c = 0; c < num_clusters; c++)
			memset(centers[c], 0, sizeof(float) * dim);

		counts = (int *)palloc0(sizeof(int) * num_clusters);

		for (i = 0; i < nvec; i++)
		{
			c = assignments[i];
			for (d = 0; d < dim; d++)
				centers[c][d] += data[i][d];
			counts[c]++;
		}
		for (c = 0; c < num_clusters; c++)
		{
			if (counts[c] > 0)
			{
				for (d = 0; d < dim; d++)
					centers[c][d] /= counts[c];
			}
		}
		pfree(counts);
	}

	/* Build output int array (nvec x 1) for 1-based cluster labels */
	out_datums = (Datum *)palloc(sizeof(Datum) * nvec);
	for (i = 0; i < nvec; i++)
		out_datums[i] = Int32GetDatum(assignments[i] + 1);

	out_array = construct_array(out_datums, nvec, INT4OID, 4, true, 'i');

	for (i = 0; i < nvec; i++)
		pfree(data[i]);
	for (c = 0; c < num_clusters; c++)
		pfree(centers[c]);
	pfree(data);
	pfree(centers);
	pfree(assignments);
	pfree(centroids_idx);
	pfree(tbl_str);
	pfree(col_str);
	pfree(out_datums);

	PG_RETURN_ARRAYTYPE_P(out_array);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for K-Means
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"

void
neurondb_gpu_register_kmeans_model(void)
{
	elog(DEBUG1, "K-Means GPU Model Ops registration skipped - not yet implemented");
}
