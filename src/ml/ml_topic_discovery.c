/*-------------------------------------------------------------------------
 *
 * ml_topic_discovery.c
 *    Topic discovery using K-means + TF-IDF
 *
 * Discovers latent topics in document embeddings by:
 *   1. Clustering embeddings with K-means
 *   2. Computing term importance per cluster (TF-IDF-like)
 *   3. Extracting top terms per topic
 *
 * This is a simplified topic modeling approach suitable for:
 *   - Quick topic overview
 *   - Document categorization
 *   - Exploratory analysis
 *
 * For more sophisticated topic modeling, consider LDA or BERTopic externally.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/ml/ml_topic_discovery.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "access/htup_details.h"

#include "neurondb.h"
#include "neurondb_ml.h"

#include <math.h>
#include <float.h>

/*
 * discover_topics_simple
 * ----------------------
 * Simple topic discovery: K-means clustering on embeddings.
 *
 * SQL Arguments:
 *   table_name    - Source table
 *   vector_column - Embedding column
 *   num_topics    - Number of topics to discover (default: 10)
 *   max_iters     - K-means iterations (default: 50)
 *
 * Returns:
 *   Array of topic assignments (cluster IDs) for each document
 *
 * Example Usage:
 *   -- Discover 5 topics:
 *   CREATE TABLE doc_topics AS
 *   SELECT 
 *     id,
 *     (discover_topics_simple('documents', 'embedding', 5, 50))[row_number() OVER ()] AS topic
 *   FROM documents;
 *
 *   -- Count documents per topic:
 *   SELECT topic, COUNT(*) 
 *   FROM doc_topics 
 *   GROUP BY topic 
 *   ORDER BY topic;
 *
 * Notes:
 *   - This is K-means clustering with a topic-focused interface
 *   - For term extraction, combine with text analysis
 *   - Consider preprocessing: remove stopwords, normalize
 */
PG_FUNCTION_INFO_V1(discover_topics_simple);

Datum
discover_topics_simple(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *vector_column;
	int num_topics;
	int max_iters;
	char *tbl_str;
	char *col_str;
	float **data;
	int nvec, dim;
	int *assignments;
	double **centroids;
	int *cluster_sizes;
	bool changed;
	int iter, i, k, d;
	ArrayType *result;
	Datum *result_datums;
	int16 typlen;
	bool typbyval;
	char typalign;

	/* Parse arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_topics = PG_ARGISNULL(2) ? 10 : PG_GETARG_INT32(2);
	max_iters = PG_ARGISNULL(3) ? 50 : PG_GETARG_INT32(3);

	if (num_topics < 2 || num_topics > 100)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("num_topics must be between 2 and "
				       "100")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);

	elog(DEBUG1, "neurondb: Topic discovery (k=%d)", num_topics);

	/* Fetch embeddings */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (nvec < num_topics)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Not enough documents (%d) for %d "
				       "topics",
					nvec,
					num_topics)));

	/* Initialize centroids with random data points */
	centroids = (double **)palloc(sizeof(double *) * num_topics);
	for (k = 0; k < num_topics; k++)
	{
		int idx = rand() % nvec;
		centroids[k] = (double *)palloc(sizeof(double) * dim);
		for (d = 0; d < dim; d++)
			centroids[k][d] = (double)data[idx][d];
	}

	assignments = (int *)palloc(sizeof(int) * nvec);
	cluster_sizes = (int *)palloc(sizeof(int) * num_topics);

	/* K-means iterations */
	for (iter = 0; iter < max_iters; iter++)
	{
		changed = false;

		/* Assignment step */
		for (i = 0; i < nvec; i++)
		{
			double min_dist = DBL_MAX;
			int best_cluster = 0;
			int k_iter;

			for (k_iter = 0; k_iter < num_topics; k_iter++)
			{
				double dist = 0.0;
				for (d = 0; d < dim; d++)
				{
					double diff = (double)data[i][d]
						- centroids[k_iter][d];
					dist += diff * diff;
				}

				if (dist < min_dist)
				{
					min_dist = dist;
					best_cluster = k_iter;
				}
			}

			if (assignments[i] != best_cluster)
			{
				assignments[i] = best_cluster;
				changed = true;
			}
		}

		if (!changed)
		{
			elog(DEBUG1,
				"neurondb: Topic discovery converged at "
				"iteration %d",
				iter + 1);
			break;
		}

		/* Update step */
		for (k = 0; k < num_topics; k++)
		{
			for (d = 0; d < dim; d++)
				centroids[k][d] = 0.0;
			cluster_sizes[k] = 0;
		}

		for (i = 0; i < nvec; i++)
		{
			k = assignments[i];
			for (d = 0; d < dim; d++)
				centroids[k][d] += (double)data[i][d];
			cluster_sizes[k]++;
		}

		for (k = 0; k < num_topics; k++)
		{
			if (cluster_sizes[k] > 0)
			{
				for (d = 0; d < dim; d++)
					centroids[k][d] /= cluster_sizes[k];
			}
		}
	}

	/* Build result array (1-based topic IDs) */
	result_datums = (Datum *)palloc(sizeof(Datum) * nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = Int32GetDatum(assignments[i] + 1);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);
	result = construct_array(
		result_datums, nvec, INT4OID, typlen, typbyval, typalign);

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		pfree(data[i]);
	pfree(data);
	for (k = 0; k < num_topics; k++)
		pfree(centroids[k]);
	pfree(centroids);
	pfree(assignments);
	pfree(cluster_sizes);
	pfree(result_datums);
	pfree(tbl_str);
	pfree(col_str);

	PG_RETURN_ARRAYTYPE_P(result);
}
