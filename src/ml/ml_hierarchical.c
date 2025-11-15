/*-------------------------------------------------------------------------
 *
 * ml_hierarchical.c
 *	  Hierarchical Agglomerative Clustering
 *
 * Hierarchical clustering builds a dendrogram of nested clusters by
 * iteratively merging the closest pair of clusters. Unlike K-means,
 * it does not require pre-specifying the number of clusters.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *	  src/ml/ml_hierarchical.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"

#include "neurondb.h"
#include "neurondb_ml.h"

#include <math.h>
#include <float.h>

/*--------------------
 * ClusterNode - Represents a cluster in the hierarchy.
 *--------------------
 */
typedef struct ClusterNode
{
	int			id;			/* Cluster ID */
	int		   *members;	/* Indices of data points in this cluster */
	int			size;		/* Number of points */
	double	   *centroid;	/* Centroid coordinates */
} ClusterNode;

static inline double
euclidean_dist(const float *a, const float *b, int dim)
{
	double		sum = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		double	diff = (double) a[i] - (double) b[i];
		sum += diff * diff;
	}
	return sqrt(sum);
}

static void
compute_centroid(float **data, int *members, int size, int dim, double *centroid)
{
	int			i;
	int			d;

	for (d = 0; d < dim; d++)
		centroid[d] = 0.0;

	for (i = 0; i < size; i++)
	{
		for (d = 0; d < dim; d++)
			centroid[d] += (double) data[members[i]][d];
	}

	for (d = 0; d < dim; d++)
		centroid[d] /= (double) size;
}

static double
cluster_distance_average(float **data, ClusterNode *c1, ClusterNode *c2, int dim)
{
	double		sum = 0.0;
	int			i;
	int			j;
#include "ml_gpu_registry.h"
	for (i = 0; i < c1->size; i++)
	{
		for (j = 0; j < c2->size; j++)
			sum += euclidean_dist(data[c1->members[i]],
								  data[c2->members[j]],
								  dim);
	}
	return sum / ((double) c1->size * (double) c2->size);
}

static double
cluster_distance_complete(float **data, ClusterNode *c1, ClusterNode *c2, int dim)
{
	double		max_dist = 0.0;
	int			i;
	int			j;

	for (i = 0; i < c1->size; i++)
	{
		for (j = 0; j < c2->size; j++)
		{
			double	dist = euclidean_dist(data[c1->members[i]],
										 data[c2->members[j]],
										 dim);
			if (dist > max_dist)
				max_dist = dist;
		}
	}
	return max_dist;
}

static double
cluster_distance_single(float **data, ClusterNode *c1, ClusterNode *c2, int dim)
{
	double		min_dist = DBL_MAX;
	int			i;
	int			j;

	for (i = 0; i < c1->size; i++)
	{
		for (j = 0; j < c2->size; j++)
		{
			double	dist = euclidean_dist(data[c1->members[i]],
										 data[c2->members[j]],
										 dim);
			if (dist < min_dist)
				min_dist = dist;
		}
	}
	return min_dist;
}

PG_FUNCTION_INFO_V1(cluster_hierarchical);

Datum
cluster_hierarchical(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	int			num_clusters;
	text	   *linkage_text;
	char	   *tbl_str;
	char	   *col_str;
	char	   *linkage;
	float	  **data;
	int			nvec;
	int			dim;
	ClusterNode *clusters;
	int		   *cluster_assignments;
	int			n_active_clusters;
	int			iter, i, j, k, d;
	ArrayType  *result;
	Datum	   *result_datums;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	/* Parse input arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_clusters = PG_GETARG_INT32(2);
	linkage_text = PG_ARGISNULL(3) ? cstring_to_text("average")
								   : PG_GETARG_TEXT_PP(3);

	if (num_clusters < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("num_clusters must be at least 1")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);
	linkage = text_to_cstring(linkage_text);

	/* Validate linkage */
	if (strcmp(linkage, "average") != 0 &&
		strcmp(linkage, "complete") != 0 &&
		strcmp(linkage, "single") != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("linkage must be 'average', 'complete', or 'single'")));

	elog(DEBUG1,
		 "neurondb: hierarchical clustering (k=%d, linkage=%s)",
		 num_clusters, linkage);

	/* Fetch data from table */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (nvec < num_clusters)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("not enough vectors (%d) for %d clusters",
						nvec, num_clusters)));

	if (nvec > 5000)
		elog(WARNING,
			 "hierarchical clustering on %d points may be slow, "
			 "consider K-means", nvec);

	/* Initialize clusters: each point as singleton cluster */
	clusters = (ClusterNode *) palloc(sizeof(ClusterNode) * nvec);
	for (i = 0; i < nvec; i++)
	{
		clusters[i].id = i;
		clusters[i].size = 1;
		clusters[i].members = (int *) palloc(sizeof(int));
		clusters[i].members[0] = i;
		clusters[i].centroid = (double *) palloc(sizeof(double) * dim);
		for (d = 0; d < dim; d++)
			clusters[i].centroid[d] = (double) data[i][d];
	}

	n_active_clusters = nvec;

	/* Agglomerative merge */
	for (iter = 0; iter < nvec - num_clusters; iter++)
	{
		double		min_dist = DBL_MAX;
		int			merge_i = -1;
		int			merge_j = -1;

		for (i = 0; i < nvec; i++)
		{
			double	dist;

			if (clusters[i].size == 0)
				continue;

			for (j = i + 1; j < nvec; j++)
			{
				if (clusters[j].size == 0)
					continue;

				if (strcmp(linkage, "average") == 0)
					dist = cluster_distance_average(data, &clusters[i], &clusters[j], dim);
				else if (strcmp(linkage, "complete") == 0)
					dist = cluster_distance_complete(data, &clusters[i], &clusters[j], dim);
				else
					dist = cluster_distance_single(data, &clusters[i], &clusters[j], dim);

				if (dist < min_dist)
				{
					min_dist = dist;
					merge_i = i;
					merge_j = j;
				}
			}
		}

		if (merge_i < 0 || merge_j < 0)
			break;

		{
			int		new_size;
			int	   *new_members;

			new_size = clusters[merge_i].size + clusters[merge_j].size;
			new_members = (int *) palloc(sizeof(int) * new_size);

			for (k = 0; k < clusters[merge_i].size; k++)
				new_members[k] = clusters[merge_i].members[k];
			for (k = 0; k < clusters[merge_j].size; k++)
				new_members[clusters[merge_i].size + k] = clusters[merge_j].members[k];

			pfree(clusters[merge_i].members);
			clusters[merge_i].members = new_members;
			clusters[merge_i].size = new_size;

			compute_centroid(data,
							 clusters[merge_i].members,
							 clusters[merge_i].size,
							 dim,
							 clusters[merge_i].centroid);

			pfree(clusters[merge_j].members);
			pfree(clusters[merge_j].centroid);
			clusters[merge_j].size = 0;
			clusters[merge_j].members = NULL;
			clusters[merge_j].centroid = NULL;

			n_active_clusters--;
		}

		if ((iter + 1) % 100 == 0)
			elog(DEBUG2,
				 "neurondb: hierarchical merge iteration %d (%d clusters remain)",
				 iter + 1, n_active_clusters);
	}

	/* Assign cluster labels: 1-based */
	cluster_assignments = (int *) palloc(sizeof(int) * nvec);
	{
		int		cluster_id = 1;

		for (i = 0; i < nvec; i++)
		{
			if (clusters[i].size > 0)
			{
				for (k = 0; k < clusters[i].size; k++)
					cluster_assignments[clusters[i].members[k]] = cluster_id;
				cluster_id++;
			}
		}
	}

	/* Build result array */
	result_datums = (Datum *) palloc(sizeof(Datum) * nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = Int32GetDatum(cluster_assignments[i]);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);
	result = construct_array(result_datums, nvec, INT4OID, typlen, typbyval, typalign);

	/* Free memory */
	for (i = 0; i < nvec; i++)
	{
		pfree(data[i]);
		if (clusters[i].size > 0)
		{
			pfree(clusters[i].members);
			pfree(clusters[i].centroid);
		}
	}
	pfree(data);
	pfree(clusters);
	pfree(cluster_assignments);
	pfree(result_datums);
	pfree(tbl_str);
	pfree(col_str);
	pfree(linkage);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for Hierarchical Clustering
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"

void
neurondb_gpu_register_hierarchical_model(void)
{
	elog(DEBUG1, "Hierarchical GPU Model Ops registration skipped - not yet implemented");
}
