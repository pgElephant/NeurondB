/*-------------------------------------------------------------------------
 *
 * ml_davies_bouldin.c
 *    Davies-Bouldin Index for cluster quality evaluation
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
 * Interpretation:
 *   - Lower values = better clustering
 *   - 0 is the best possible score
 *   - Typical range: 0.5 to 2.5
 *   - Values > 3.0 suggest poor clustering
 *
 * Advantages over Silhouette:
 *   - O(n) vs O(n²) complexity
 *   - Faster for large datasets
 *   - Based on cluster centroids (interpretable)
 *
 * Reference:
 *   Davies, D. L., & Bouldin, D. W. (1979). "A cluster separation measure."
 *   IEEE Transactions on Pattern Analysis and Machine Intelligence.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/ml/ml_davies_bouldin.c
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
 * Compute Euclidean distance between two vectors
 */
static inline double
euclidean_distance(const float *a, const float *b, int dim)
{
	double sum = 0.0;
	int i;

	for (i = 0; i < dim; i++)
	{
		double diff = (double)a[i] - (double)b[i];
		sum += diff * diff;
	}
	return sqrt(sum);
}

/*
 * davies_bouldin_index
 * --------------------
 * Compute Davies-Bouldin Index for cluster quality evaluation.
 *
 * SQL Arguments:
 *   table_name    - Source table containing vectors
 *   vector_column - Column with vector data
 *   cluster_column - Column with cluster assignments (integer labels)
 *
 * Returns:
 *   Float8: Davies-Bouldin Index (lower = better clustering)
 *
 * Typical values:
 *   - < 1.0: Excellent clustering
 *   - 1.0-2.0: Good clustering
 *   - 2.0-3.0: Acceptable clustering
 *   - > 3.0: Poor clustering (consider different k or algorithm)
 *
 * Example Usage:
 *   -- After running K-means:
 *   SELECT davies_bouldin_index('documents', 'embedding', 'cluster_id');
 *   
 *   -- Compare different k values:
 *   SELECT k, davies_bouldin_index('docs', 'vec', 'k' || k || '_cluster')
 *   FROM generate_series(2, 10) k;
 *
 * Notes:
 *   - Clusters with < 2 points are excluded from calculation
 *   - Noise points (cluster_id = -1) are ignored
 *   - Complexity: O(n*d + k²*d) where n=points, d=dimensions, k=clusters
 */
PG_FUNCTION_INFO_V1(davies_bouldin_index);

Datum
davies_bouldin_index(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *vector_column;
	text *cluster_column;
	char *tbl_str;
	char *vec_col_str;
	char *cluster_col_str;
	float **vectors;
	int *cluster_labels;
	int nvec, dim;
	int num_clusters;
	int *cluster_sizes;
	float **cluster_centroids;
	double *cluster_scatter; /* Average distance to centroid */
	double db_index;
	int i, c, d;
	StringInfoData sql;
	int ret;

	/* Parse arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	cluster_column = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	vec_col_str = text_to_cstring(vector_column);
	cluster_col_str = text_to_cstring(cluster_column);

	elog(DEBUG1,
		"neurondb: Computing Davies-Bouldin Index for %s.%s "
		"(clusters=%s)",
		tbl_str,
		vec_col_str,
		cluster_col_str);

	/* Fetch vectors */
	vectors = neurondb_fetch_vectors_from_table(
		tbl_str, vec_col_str, &nvec, &dim);
	if (nvec < 2)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("Need at least 2 vectors for DBI "
				       "calculation")));

	/* Fetch cluster assignments */
	cluster_labels = (int *)palloc(sizeof(int) * nvec);

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT %s FROM %s ORDER BY ctid",
		cluster_col_str,
		tbl_str);
	ret = SPI_execute(sql.data, true, 0);

	if (ret != SPI_OK_SELECT || (int)SPI_processed != nvec)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("Failed to fetch cluster labels "
				       "(expected %d, got %d)",
					nvec,
					(int)SPI_processed)));
	}

	num_clusters = 0;
	for (i = 0; i < nvec; i++)
	{
		bool isnull;
		Datum val = SPI_getbinval(SPI_tuptable->vals[i],
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
				errmsg("Need at least 2 clusters for DBI "
				       "(found %d)",
					num_clusters)));

	elog(DEBUG1, "neurondb: Found %d clusters in data", num_clusters);

	/* 
	 * Compute cluster centroids and sizes
	 * Cluster IDs are 1-based in output, internally we use 0-based
	 */
	cluster_sizes = (int *)palloc0(sizeof(int) * num_clusters);
	cluster_centroids = (float **)palloc(sizeof(float *) * num_clusters);
	for (c = 0; c < num_clusters; c++)
	{
		cluster_centroids[c] = (float *)palloc0(sizeof(float) * dim);
	}

	/* Sum vectors for each cluster */
	for (i = 0; i < nvec; i++)
	{
		int cluster = cluster_labels[i];

		if (cluster < 1 || cluster > num_clusters)
			continue; /* Skip noise points or invalid labels */

		cluster = cluster - 1; /* Convert to 0-based */
		cluster_sizes[cluster]++;

		for (d = 0; d < dim; d++)
			cluster_centroids[cluster][d] += vectors[i][d];
	}

	/* Compute centroids (mean of points) */
	for (c = 0; c < num_clusters; c++)
	{
		if (cluster_sizes[c] > 0)
		{
			for (d = 0; d < dim; d++)
				cluster_centroids[c][d] /= cluster_sizes[c];
		}
	}

	/*
	 * Compute cluster scatter: s_i = average distance of points to centroid
	 */
	cluster_scatter = (double *)palloc0(sizeof(double) * num_clusters);

	for (i = 0; i < nvec; i++)
	{
		int cluster = cluster_labels[i];

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

	/*
	 * Compute Davies-Bouldin Index
	 * DBI = (1/k) * Σ_i max_j≠i [(s_i + s_j) / d(c_i, c_j)]
	 */
	db_index = 0.0;
	{
		int valid_clusters = 0;

		for (i = 0; i < num_clusters; i++)
		{
			double max_ratio = 0.0;
			int j;

			/* Skip clusters with < 2 points */
			if (cluster_sizes[i] < 2)
				continue;

			/* Find max ratio with other clusters */
			for (j = 0; j < num_clusters; j++)
			{
				double centroid_dist;
				double ratio;

				if (i == j || cluster_sizes[j] < 2)
					continue;

				centroid_dist =
					euclidean_distance(cluster_centroids[i],
						cluster_centroids[j],
						dim);

				if (centroid_dist < 1e-10)
					continue; /* Avoid division by zero */

				ratio = (cluster_scatter[i]
						+ cluster_scatter[j])
					/ centroid_dist;

				if (ratio > max_ratio)
					max_ratio = ratio;
			}

			db_index += max_ratio;
			valid_clusters++;
		}

		if (valid_clusters > 0)
			db_index /= valid_clusters;
		else
			db_index = -1.0; /* Invalid clustering */
	}

	elog(DEBUG1, "neurondb: Davies-Bouldin Index = %.4f", db_index);

	/* Cleanup */
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
