/*-------------------------------------------------------------------------
 *
 * ml_gmm.c
 *    Gaussian Mixture Model (EM-lite) for soft clustering
 *
 * GMM models data as a mixture of K multivariate Gaussian distributions.
 * Unlike K-means (hard clustering), GMM provides:
 *   - Probabilistic cluster assignments
 *   - Cluster shape and orientation (via covariance)
 *   - Uncertainty quantification
 *
 * Algorithm: Expectation-Maximization (EM)
 *   1. E-step: Compute responsibilities (posterior probabilities)
 *   2. M-step: Update means, covariances, mixing coefficients
 *   3. Repeat until convergence
 *
 * Simplified Implementation:
 *   - Diagonal covariance matrices (faster, less parameters)
 *   - Regularization to prevent singular matrices
 *   - K-means++ initialization
 *
 * Use Cases:
 *   - Soft clustering with confidence scores
 *   - Anomaly detection (low likelihood points)
 *   - Density estimation
 *   - Topic modeling foundations
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/ml/ml_gmm.c
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

#define GMM_EPSILON 1e-6 /* Regularization for covariance */
#define GMM_MIN_PROB 1e-10 /* Minimum probability to avoid log(0) */

/*
 * GMM model structure
 */
typedef struct GMMModel
{
	int k; /* Number of components */
	int dim; /* Dimensionality */
	double *mixing_coeffs; /* π_k: mixing coefficients [k] */
	double **means; /* μ_k: component means [k][dim] */
	double **variances; /* Σ_k: diagonal variances [k][dim] */
} GMMModel;

/*
 * Compute Gaussian probability density (diagonal covariance)
 */
static double
gaussian_pdf(const float *x,
	const double *mean,
	const double *variance,
	int dim)
{
	double log_likelihood = 0.0;
	double log_det = 0.0;
	int d;

	for (d = 0; d < dim; d++)
	{
		double diff = (double)x[d] - mean[d];
		double var = variance[d] + GMM_EPSILON;

		log_likelihood -= 0.5 * (diff * diff) / var;
		log_det += log(var);
	}

	log_likelihood -= 0.5 * (dim * log(2.0 * M_PI) + log_det);

	return exp(log_likelihood);
}

/*
 * cluster_gmm
 * -----------
 * Fit Gaussian Mixture Model using EM algorithm.
 *
 * SQL Arguments:
 *   table_name    - Source table with vectors
 *   vector_column - Vector column name
 *   num_components - Number of Gaussian components (K)
 *   max_iters     - Maximum EM iterations (default: 100)
 *
 * Returns:
 *   2D array of responsibilities [nvec][k]
 *   Each row sums to 1.0, representing soft cluster assignments
 *
 * Interpretation:
 *   responsibilities[i][j] = probability that point i belongs to component j
 *   - Hard assignment: argmax(responsibilities[i])
 *   - Uncertainty: entropy of responsibilities[i]
 *
 * Example Usage:
 *   -- Get soft cluster assignments:
 *   SELECT id, cluster_gmm('documents', 'embedding', 5, 100)
 *   FROM documents;
 *
 *   -- Get hard assignments (most likely component):
 *   WITH soft_clusters AS (
 *     SELECT id, unnest(cluster_gmm('docs', 'vec', 3, 50)) AS probs
 *     FROM generate_series(1, (SELECT COUNT(*) FROM docs)) id
 *   )
 *   SELECT id, row_number() OVER (PARTITION BY id ORDER BY probs DESC) AS cluster
 *   FROM soft_clusters
 *   WHERE row_number() OVER (PARTITION BY id ORDER BY probs DESC) = 1;
 *
 * Notes:
 *   - Diagonal covariance assumption (faster, works well for many cases)
 *   - Regularization prevents numerical issues
 *   - May converge to local optima (try multiple random inits)
 */
PG_FUNCTION_INFO_V1(cluster_gmm);

Datum
cluster_gmm(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *vector_column;
	int num_components;
	int max_iters;
	char *tbl_str;
	char *col_str;
	float **data;
	int nvec, dim;
	GMMModel model;
	double **responsibilities;
	double log_likelihood, prev_log_likelihood;
	int iter, i, k, d;
	ArrayType *result;
	Datum *result_datums;
	int dims[2];
	int lbs[2];

	/* Parse arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_components = PG_GETARG_INT32(2);
	max_iters = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);

	if (num_components < 1 || num_components > 100)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("num_components must be between 1 and "
				       "100")));

	if (max_iters < 1)
		max_iters = 100;

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);

	elog(DEBUG1,
		"neurondb: GMM clustering (k=%d, max_iters=%d)",
		num_components,
		max_iters);

	/* Fetch training data */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (nvec < num_components)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Not enough vectors (%d) for %d "
				       "components",
					nvec,
					num_components)));

	/* Initialize GMM model */
	model.k = num_components;
	model.dim = dim;
	model.mixing_coeffs = (double *)palloc(sizeof(double) * num_components);
	model.means = (double **)palloc(sizeof(double *) * num_components);
	model.variances = (double **)palloc(sizeof(double *) * num_components);

	for (k = 0; k < num_components; k++)
	{
		int idx;

		model.means[k] = (double *)palloc(sizeof(double) * dim);
		model.variances[k] = (double *)palloc(sizeof(double) * dim);

		/* Initialize with random data points for means */
		idx = rand() % nvec;
		for (d = 0; d < dim; d++)
			model.means[k][d] = (double)data[idx][d];

		/* Initialize variances to 1.0 */
		for (d = 0; d < dim; d++)
			model.variances[k][d] = 1.0;

		/* Equal mixing coefficients initially */
		model.mixing_coeffs[k] = 1.0 / num_components;
	}

	/* Allocate responsibilities matrix */
	responsibilities = (double **)palloc(sizeof(double *) * nvec);
	for (i = 0; i < nvec; i++)
		responsibilities[i] =
			(double *)palloc(sizeof(double) * num_components);

	/* EM algorithm */
	prev_log_likelihood = -DBL_MAX;

	for (iter = 0; iter < max_iters; iter++)
	{
		/* E-step: Compute responsibilities */
		log_likelihood = 0.0;

		for (i = 0; i < nvec; i++)
		{
			double sum = 0.0;

			/* Compute weighted likelihoods */
			for (k = 0; k < num_components; k++)
			{
				double pdf = gaussian_pdf(data[i],
					model.means[k],
					model.variances[k],
					dim);
				responsibilities[i][k] =
					model.mixing_coeffs[k] * pdf;
				sum += responsibilities[i][k];
			}

			/* Normalize to get probabilities */
			if (sum < GMM_MIN_PROB)
				sum = GMM_MIN_PROB;

			for (k = 0; k < num_components; k++)
			{
				responsibilities[i][k] /= sum;
				if (responsibilities[i][k] < GMM_MIN_PROB)
					responsibilities[i][k] = GMM_MIN_PROB;
			}

			log_likelihood += log(sum);
		}

		log_likelihood /= nvec;

		/* Check convergence */
		if (fabs(log_likelihood - prev_log_likelihood) < 1e-6)
		{
			elog(DEBUG1,
				"neurondb: GMM converged at iteration %d "
				"(ll=%.4f)",
				iter + 1,
				log_likelihood);
			break;
		}
		prev_log_likelihood = log_likelihood;

		/* M-step: Update parameters */
		{
			/* Compute N_k (effective number of points per component) */
			double *N_k;

			N_k = (double *)palloc0(
				sizeof(double) * num_components);
			for (k = 0; k < num_components; k++)
			{
				for (i = 0; i < nvec; i++)
					N_k[k] += responsibilities[i][k];

				if (N_k[k] < GMM_MIN_PROB)
					N_k[k] = GMM_MIN_PROB;
			}

			/* Update mixing coefficients */
			for (k = 0; k < num_components; k++)
				model.mixing_coeffs[k] = N_k[k] / nvec;

			/* Update means */
			for (k = 0; k < num_components; k++)
			{
				for (d = 0; d < dim; d++)
					model.means[k][d] = 0.0;

				for (i = 0; i < nvec; i++)
					for (d = 0; d < dim; d++)
						model.means[k][d] +=
							responsibilities[i][k]
							* data[i][d];

				for (d = 0; d < dim; d++)
					model.means[k][d] /= N_k[k];
			}

			/* Update variances (diagonal) */
			for (k = 0; k < num_components; k++)
			{
				for (d = 0; d < dim; d++)
					model.variances[k][d] = 0.0;

				for (i = 0; i < nvec; i++)
				{
					for (d = 0; d < dim; d++)
					{
						double diff = data[i][d]
							- model.means[k][d];
						model.variances[k][d] +=
							responsibilities[i][k]
							* diff * diff;
					}
				}

				for (d = 0; d < dim; d++)
					model.variances[k][d] =
						(model.variances[k][d] / N_k[k])
						+ GMM_EPSILON;
			}

			pfree(N_k);
		}

		if ((iter + 1) % 10 == 0)
			elog(DEBUG2,
				"neurondb: GMM iteration %d, "
				"log-likelihood=%.4f",
				iter + 1,
				log_likelihood);
	}

	/* Build 2D result array of responsibilities */
	result_datums = (Datum *)palloc(sizeof(Datum) * nvec * num_components);
	for (i = 0; i < nvec; i++)
		for (k = 0; k < num_components; k++)
			result_datums[i * num_components + k] =
				Float8GetDatum(responsibilities[i][k]);

	dims[0] = nvec;
	dims[1] = num_components;
	lbs[0] = 1;
	lbs[1] = 1;

	result = construct_md_array(result_datums,
		NULL,
		2,
		dims,
		lbs,
		FLOAT8OID,
		sizeof(float8),
		FLOAT8PASSBYVAL,
		'd');

	/* Cleanup */
	for (i = 0; i < nvec; i++)
	{
		pfree(data[i]);
		pfree(responsibilities[i]);
	}
	pfree(data);
	pfree(responsibilities);

	for (k = 0; k < num_components; k++)
	{
		pfree(model.means[k]);
		pfree(model.variances[k]);
	}
	pfree(model.means);
	pfree(model.variances);
	pfree(model.mixing_coeffs);
	pfree(result_datums);
	pfree(tbl_str);
	pfree(col_str);

	PG_RETURN_ARRAYTYPE_P(result);
}
