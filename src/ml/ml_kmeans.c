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
#include "executor/spi.h"
#include "utils/jsonb.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"
#include "ml_gpu_registry.h"
#include "ml_catalog.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

	selected = (bool *)palloc0(sizeof(bool) * nvec);
	dist = (double *)palloc(sizeof(double) * nvec);

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
	Datum *out_datums;
	ArrayType *out_array;

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_col);

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

/*
 * Serialize K-Means model (centroids) to bytea
 */
static bytea *
kmeans_model_serialize_to_bytea(float **centers, int num_clusters, int dim)
{
	StringInfoData buf;
	int i, j;
	int total_size;
	bytea *result;

	initStringInfo(&buf);

	/* Write header: num_clusters, dim */
	appendBinaryStringInfo(&buf, (char *)&num_clusters, sizeof(int));
	appendBinaryStringInfo(&buf, (char *)&dim, sizeof(int));

	/* Write centroids */
	for (i = 0; i < num_clusters; i++)
		for (j = 0; j < dim; j++)
			appendBinaryStringInfo(&buf, (char *)&centers[i][j], sizeof(float));

	/* Convert to bytea */
	total_size = VARHDRSZ + buf.len;
	result = (bytea *)palloc(total_size);
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	pfree(buf.data);

	return result;
}

/*
 * Deserialize K-Means model from bytea
 */
static int
kmeans_model_deserialize_from_bytea(const bytea *data, float ***centers_out, int *num_clusters_out, int *dim_out)
{
	const char *buf;
	int offset = 0;
	int i, j;
	float **centers;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 2)
		return -1;

	buf = VARDATA(data);

	/* Read header */
	memcpy(num_clusters_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(dim_out, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (*num_clusters_out <= 0 || *num_clusters_out > 10000 || *dim_out <= 0 || *dim_out > 100000)
		return -1;

	/* Allocate centroids */
	centers = (float **)palloc(sizeof(float *) * *num_clusters_out);
	for (i = 0; i < *num_clusters_out; i++)
	{
		centers[i] = (float *)palloc(sizeof(float) * *dim_out);
		for (j = 0; j < *dim_out; j++)
		{
			memcpy(&centers[i][j], buf + offset, sizeof(float));
			offset += sizeof(float);
		}
	}

	*centers_out = centers;
	return 0;
}

/*
 * train_kmeans_model_id
 *
 * Trains K-Means and stores centroids in catalog, returns model_id
 */
PG_FUNCTION_INFO_V1(train_kmeans_model_id);

Datum
train_kmeans_model_id(PG_FUNCTION_ARGS)
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
	bytea *model_data;
	MLCatalogModelSpec spec;
	Jsonb *metrics;
	StringInfoData metrics_json;
	int32 model_id;

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_col);

	if (num_clusters <= 1)
	{
		pfree(tbl_str);
		pfree(col_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("number of clusters must be at least 2")));
	}
	if (max_iters < 1)
		max_iters = 100;

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (!data || nvec < num_clusters)
	{
		pfree(tbl_str);
		pfree(col_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("not enough vectors for cluster count "
				       "(need >= %d, have %d)",
					num_clusters,
					nvec)));
	}

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

	/* K-Means iteration */
	for (iter = 0; iter < max_iters && changed; iter++)
	{
		int *counts;

		changed = false;
		/* Assignment phase */
		for (i = 0; i < nvec; i++)
		{
			double min_dist = DBL_MAX;
			int best = -1;

			for (c = 0; c < num_clusters; c++)
			{
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

	/* Serialize model to bytea */
	model_data = kmeans_model_serialize_to_bytea(centers, num_clusters, dim);

	/* Build metrics JSONB */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json, "{\"storage\": \"cpu\", \"k\": %d, \"dim\": %d, \"max_iters\": %d, \"n_samples\": %d}",
		num_clusters, dim, max_iters, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metrics_json.data)));
	pfree(metrics_json.data);

	/* Store model in catalog */
	memset(&spec, 0, sizeof(MLCatalogModelSpec));
	spec.project_name = NULL;
	spec.algorithm = "kmeans";
	spec.training_table = tbl_str;
	spec.training_column = NULL;
	spec.model_data = model_data;
	spec.metrics = metrics;
	spec.num_samples = nvec;
	spec.num_features = dim;

	model_id = ml_catalog_register_model(&spec);

	/* Cleanup */
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

	PG_RETURN_INT32(model_id);
}

/*
 * Compute Euclidean distance squared
 */
static inline double
euclidean_distance_squared(const float *a, const float *b, int dim)
{
	double sum = 0.0;
	int i;

	for (i = 0; i < dim; i++)
	{
		double diff = (double)a[i] - (double)b[i];
		sum += diff * diff;
	}
	return sum;
}

/*
 * Compute Euclidean distance
 */
static inline double
euclidean_distance(const float *a, const float *b, int dim)
{
	return sqrt(euclidean_distance_squared(a, b, dim));
}

/*
 * evaluate_kmeans_by_model_id
 *
 * Evaluates K-Means clustering model by computing:
 * - Inertia (within-cluster sum of squares)
 * - Silhouette score
 * - Davies-Bouldin index
 */
PG_FUNCTION_INFO_V1(evaluate_kmeans_by_model_id);

Datum
evaluate_kmeans_by_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	text *table_name;
	text *vector_col;
	char *tbl_str;
	char *col_str;
	int ret;
	int nvec = 0;
	int i, j, c;
	float **data = NULL;
	int dim = 0;
	float **centers = NULL;
	int num_clusters = 0;
	int *assignments = NULL;
	int *cluster_sizes = NULL;
	double inertia = 0.0;
	double silhouette = 0.0;
	double davies_bouldin = 0.0;
	double *cluster_scatter = NULL;
	double *a_scores = NULL;
	double *b_scores = NULL;
	MemoryContext oldcontext;
	StringInfoData jsonbuf;
	Jsonb *result_jsonb = NULL;
	bytea *model_payload = NULL;
	Jsonb *model_metrics = NULL;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_kmeans_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_kmeans_by_model_id: table_name and vector_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	vector_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_col);

	oldcontext = CurrentMemoryContext;

	/* Load model from catalog */
	if (!ml_catalog_fetch_model_payload(model_id, &model_payload, NULL, &model_metrics))
	{
		pfree(tbl_str);
		pfree(col_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_kmeans_by_model_id: model %d not found",
					model_id)));
	}

	if (model_payload == NULL)
	{
		pfree(tbl_str);
		pfree(col_str);
		if (model_metrics)
			pfree(model_metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_kmeans_by_model_id: model %d has no model_data",
					model_id)));
	}

	/* Deserialize model */
	if (kmeans_model_deserialize_from_bytea(model_payload, &centers, &num_clusters, &dim) != 0)
	{
		pfree(tbl_str);
		pfree(col_str);
		if (model_payload)
			pfree(model_payload);
		if (model_metrics)
			pfree(model_metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_kmeans_by_model_id: failed to deserialize model")));
	}

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
	{
		for (c = 0; c < num_clusters; c++)
			pfree(centers[c]);
		pfree(centers);
		pfree(tbl_str);
		pfree(col_str);
		if (model_payload)
			pfree(model_payload);
		if (model_metrics)
			pfree(model_metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_kmeans_by_model_id: SPI_connect failed")));
	}

	/* Fetch test data */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (!data || nvec < 1)
	{
		SPI_finish();
		for (c = 0; c < num_clusters; c++)
			pfree(centers[c]);
		pfree(centers);
		pfree(tbl_str);
		pfree(col_str);
		if (model_payload)
			pfree(model_payload);
		if (model_metrics)
			pfree(model_metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_kmeans_by_model_id: no valid data found")));
	}

	/* Assign points to clusters */
	assignments = (int *)palloc(sizeof(int) * nvec);
	cluster_sizes = (int *)palloc0(sizeof(int) * num_clusters);

	for (i = 0; i < nvec; i++)
	{
		double min_dist = DBL_MAX;
		int best = 0;

		for (c = 0; c < num_clusters; c++)
		{
			double dist = euclidean_distance_squared(data[i], centers[c], dim);
			if (dist < min_dist)
			{
				min_dist = dist;
				best = c;
			}
		}
		assignments[i] = best;
		cluster_sizes[best]++;
		inertia += min_dist;
	}

	/* Compute silhouette score */
	a_scores = (double *)palloc0(sizeof(double) * nvec);
	b_scores = (double *)palloc0(sizeof(double) * nvec);

	for (i = 0; i < nvec; i++)
	{
		int my_cluster = assignments[i];
		int same_count = 0;
		double same_dist = 0.0;
		double min_other_dist = DBL_MAX;

		if (cluster_sizes[my_cluster] <= 1)
		{
			a_scores[i] = 0.0;
			b_scores[i] = 0.0;
			continue;
		}

		/* Average distance to same cluster */
		for (j = 0; j < nvec; j++)
		{
			if (i == j)
				continue;
			if (assignments[j] == my_cluster)
			{
				same_dist += euclidean_distance(data[i], data[j], dim);
				same_count++;
			}
		}
		if (same_count > 0)
			a_scores[i] = same_dist / (double)same_count;
		else
			a_scores[i] = 0.0;

		/* Minimum average distance to other clusters */
		{
			double other_dist = 0.0;
			int other_count = 0;
			int other_cluster_loop;

			for (other_cluster_loop = 0; other_cluster_loop < num_clusters; other_cluster_loop++)
			{
				if (other_cluster_loop == my_cluster)
					continue;
				if (cluster_sizes[other_cluster_loop] == 0)
					continue;

				for (j = 0; j < nvec; j++)
				{
					if (assignments[j] == other_cluster_loop)
					{
						other_dist += euclidean_distance(data[i], data[j], dim);
						other_count++;
					}
				}
				if (other_count > 0)
				{
					other_dist /= (double)other_count;
					if (other_dist < min_other_dist)
						min_other_dist = other_dist;
				}
			}
		}
		b_scores[i] = min_other_dist;
	}

	/* Compute average silhouette */
	{
		int valid_count = 0;
		double sum_silhouette = 0.0;

		for (i = 0; i < nvec; i++)
		{
			double max_ab = (a_scores[i] > b_scores[i]) ? a_scores[i] : b_scores[i];
			if (max_ab > 0.0)
			{
				double s = (b_scores[i] - a_scores[i]) / max_ab;
				sum_silhouette += s;
				valid_count++;
			}
		}
		if (valid_count > 0)
			silhouette = sum_silhouette / (double)valid_count;
	}

	/* Compute Davies-Bouldin index */
	cluster_scatter = (double *)palloc0(sizeof(double) * num_clusters);

	for (i = 0; i < nvec; i++)
	{
		c = assignments[i];
		cluster_scatter[c] += euclidean_distance(data[i], centers[c], dim);
	}

	for (c = 0; c < num_clusters; c++)
	{
		if (cluster_sizes[c] > 0)
			cluster_scatter[c] /= (double)cluster_sizes[c];
	}

	{
		int valid_clusters = 0;
		double sum_dbi = 0.0;

		for (i = 0; i < num_clusters; i++)
		{
			double max_ratio = 0.0;

			if (cluster_sizes[i] < 2)
				continue;

			for (j = 0; j < num_clusters; j++)
			{
				double centroid_dist;
				double ratio;

				if (i == j || cluster_sizes[j] < 2)
					continue;

				centroid_dist = euclidean_distance(centers[i], centers[j], dim);
				if (centroid_dist < 1e-10)
					continue;

				ratio = (cluster_scatter[i] + cluster_scatter[j]) / centroid_dist;
				if (ratio > max_ratio)
					max_ratio = ratio;
			}
			sum_dbi += max_ratio;
			valid_clusters++;
		}
		if (valid_clusters > 0)
			davies_bouldin = sum_dbi / (double)valid_clusters;
	}

	SPI_finish();

	/* Build jsonb result */
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"inertia\":%.6f,\"silhouette_score\":%.6f,\"davies_bouldin_index\":%.6f,\"n_samples\":%d,\"n_clusters\":%d}",
		inertia,
		silhouette,
		davies_bouldin,
		nvec,
		num_clusters);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
		CStringGetDatum(jsonbuf.data)));

	pfree(jsonbuf.data);

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		pfree(data[i]);
	pfree(data);
	for (c = 0; c < num_clusters; c++)
		pfree(centers[c]);
	pfree(centers);
	pfree(assignments);
	pfree(cluster_sizes);
	pfree(cluster_scatter);
	pfree(a_scores);
	pfree(b_scores);
	pfree(tbl_str);
	pfree(col_str);
	if (model_payload)
		pfree(model_payload);
	if (model_metrics)
		pfree(model_metrics);

	MemoryContextSwitchTo(oldcontext);
	PG_RETURN_JSONB_P(result_jsonb);
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
