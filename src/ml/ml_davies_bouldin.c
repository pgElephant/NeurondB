/*-------------------------------------------------------------------------
 *
 * ml_davies_bouldin.c
 *	  Davies-Bouldin Index for cluster quality evaluation
 *
 * The Davies-Bouldin Index (DBI) measures the average similarity between each
 * cluster and its most similar cluster. Lower values indicate better clustering
 * (more separated clusters). Unlike silhouette score, DBI uses cluster centroids
 * and is computationally efficient.
 *
 * Formula:
 *   DBI = (1/k) * Σ_i max_j≠i [(s_i + s_j) / d(c_i, c_j)]
 *
 * Where:
 *   - k: number of clusters
 *   - s_i: average distance of points in cluster i to centroid c_i
 *   - d(c_i, c_j): distance between centroids i and j
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *	  src/ml/ml_davies_bouldin.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"

#include "neurondb.h"
#include "neurondb_ml.h"

#include <math.h>
#include <float.h>

/*
 * euclidean_distance
 *	 Compute Euclidean distance between two vectors.
 */
static inline double
euclidean_distance(const float *a, const float *b, int dim)
{
	double		sum = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		double		diff = (double) a[i] - (double) b[i];
		sum += diff * diff;
	}
	return sqrt(sum);
}

PG_FUNCTION_INFO_V1(davies_bouldin_index);

Datum
davies_bouldin_index(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	text	   *cluster_column;
	char	   *tbl_str;
	char	   *vec_col_str;
	char	   *cluster_col_str;
	float	  **vectors;
	int		   *cluster_labels;
	int			nvec;
	int			dim;
	int			num_clusters;
	int		   *cluster_sizes;
	float	  **cluster_centroids;
	double	   *cluster_scatter;
	double		db_index;
	int			i;
	int			c;
	int			d;
	StringInfoData sql;
#include "ml_gpu_registry.h"

	/* Parse input arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	cluster_column = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	vec_col_str = text_to_cstring(vector_column);
	cluster_col_str = text_to_cstring(cluster_column);

	elog(DEBUG1, "neurondb: Computing Davies-Bouldin Index for %s.%s (clusters=%s)",
		 tbl_str, vec_col_str, cluster_col_str);

	vectors = neurondb_fetch_vectors_from_table(tbl_str, vec_col_str, &nvec, &dim);

	if (nvec < 2)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 2 vectors for DBI calculation")));

	cluster_labels = (int *) palloc(sizeof(int) * nvec);

	int ret;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	{
		initStringInfo(&sql);
		appendStringInfo(&sql, "SELECT %s FROM %s ORDER BY ctid", cluster_col_str, tbl_str);
		ret = SPI_execute(sql.data, true, 0);
	}

	if (ret != SPI_OK_SELECT || (int) SPI_processed != nvec)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Failed to fetch cluster labels (expected %d, got %d)",
						nvec, (int) SPI_processed)));
	}

	num_clusters = 0;
	for (i = 0; i < nvec; i++)
	{
		bool		isnull;
		Datum		val;

		val = SPI_getbinval(SPI_tuptable->vals[i],
							SPI_tuptable->tupdesc,
							1,
							&isnull);

		cluster_labels[i] = isnull ? -1 : DatumGetInt32(val);
		if (cluster_labels[i] > num_clusters)
			num_clusters = cluster_labels[i];
	}

	SPI_finish();

	if (num_clusters < 2)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 2 clusters for DBI (found %d)", num_clusters)));

	elog(DEBUG1, "neurondb: Found %d clusters in data", num_clusters);

	cluster_sizes = (int *) palloc0(sizeof(int) * num_clusters);
	cluster_centroids = (float **) palloc(sizeof(float *) * num_clusters);

	for (c = 0; c < num_clusters; c++)
		cluster_centroids[c] = (float *) palloc0(sizeof(float) * dim);

	for (i = 0; i < nvec; i++)
	{
		int		cluster = cluster_labels[i];

		if (cluster < 1 || cluster > num_clusters)
			continue;

		cluster = cluster - 1;

		cluster_sizes[cluster]++;

		for (d = 0; d < dim; d++)
			cluster_centroids[cluster][d] += vectors[i][d];
	}

	for (c = 0; c < num_clusters; c++)
	{
		if (cluster_sizes[c] > 0)
		{
			for (d = 0; d < dim; d++)
				cluster_centroids[c][d] /= cluster_sizes[c];
		}
	}

	cluster_scatter = (double *) palloc0(sizeof(double) * num_clusters);

	for (i = 0; i < nvec; i++)
	{
		int		cluster = cluster_labels[i];

		if (cluster < 1 || cluster > num_clusters)
			continue;

		cluster = cluster - 1;
		cluster_scatter[cluster] += euclidean_distance(
				vectors[i], cluster_centroids[cluster], dim);
	}

	for (c = 0; c < num_clusters; c++)
	{
		if (cluster_sizes[c] > 0)
			cluster_scatter[c] /= cluster_sizes[c];
	}

	db_index = 0.0;
	{
		int		valid_clusters = 0;

		for (i = 0; i < num_clusters; i++)
		{
			double	max_ratio = 0.0;
			int		j;

			if (cluster_sizes[i] < 2)
				continue;

			for (j = 0; j < num_clusters; j++)
			{
				double	centroid_dist;
				double	ratio;

				if (i == j || cluster_sizes[j] < 2)
					continue;

				centroid_dist = euclidean_distance(cluster_centroids[i],
												   cluster_centroids[j],
												   dim);

				if (centroid_dist < 1e-10)
					continue;

				ratio = (cluster_scatter[i] + cluster_scatter[j]) / centroid_dist;

				if (ratio > max_ratio)
					max_ratio = ratio;
			}

			db_index += max_ratio;
			valid_clusters++;
		}

		if (valid_clusters > 0)
			db_index /= valid_clusters;
		else
			db_index = -1.0;
	}

	elog(DEBUG1, "neurondb: Davies-Bouldin Index = %.4f", db_index);

	for (i = 0; i < nvec; i++)
		pfree(vectors[i]);
	pfree(vectors);
	pfree(cluster_labels);
	pfree(cluster_sizes);

	for (c = 0; c < num_clusters; c++)
		pfree(cluster_centroids[c]);
	pfree(cluster_centroids);

	pfree(cluster_scatter);
	pfree(tbl_str);
	pfree(vec_col_str);
	pfree(cluster_col_str);

	PG_RETURN_FLOAT8(db_index);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for DaviesBouldin
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"

void
neurondb_gpu_register_davies_bouldin_model(void)
{
	elog(DEBUG1, "DaviesBouldin GPU Model Ops registration skipped - not yet implemented");
}
