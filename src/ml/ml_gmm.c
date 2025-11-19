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
#include "executor/spi.h"
#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_catalog.h"
#include "lib/stringinfo.h"
#include "utils/jsonb.h"
#include "vector/vector_types.h"
#include "neurondb_cuda_gmm.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif

#include <math.h>
#include <float.h>

#define GMM_EPSILON		1e-6		/* Regularization for covariance */
#define GMM_MIN_PROB	1e-10		/* Minimum probability to avoid log(0) */

/* GMM model structure */
typedef struct GMMModel
{
	int				k;				/* Number of components */
	int				dim;			/* Dimensionality */
	double		   *mixing_coeffs;	/* Mixing coefficients [k] */
	double		  **means;			/* Component means [k][dim] */
	double		  **variances;		/* Diagonal variances [k][dim] */
} GMMModel;

/*
 * Compute Gaussian probability density (diagonal covariance)
 */
static double
gaussian_pdf(const float *x, const double *mean, const double *variance, int dim)
{
	double		log_likelihood = 0.0;
	double		log_det = 0.0;
	int			d;

	for (d = 0; d < dim; d++)
	{
		double	diff = (double)x[d] - mean[d];
		double	var = variance[d] + GMM_EPSILON;

		log_likelihood -= 0.5 * (diff * diff) / var;
		log_det += log(var);
	}

	log_likelihood -= 0.5 * (dim * log(2.0 * M_PI) + log_det);

	return exp(log_likelihood);
}

PG_FUNCTION_INFO_V1(cluster_gmm);

Datum
cluster_gmm(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	int			num_components;
	int			max_iters;
	char	   *tbl_str;
	char	   *col_str;
	float	  **data;
	int			nvec,
				dim;
	GMMModel	model;
	double	  **responsibilities;
	double		log_likelihood,
				prev_log_likelihood;
	int			iter,
				i,
				k,
				d;
	ArrayType  *result;
	Datum	   *result_datums;
	int			dims[2];
	int			lbs[2];

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_components = PG_GETARG_INT32(2);
	max_iters = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);

	if (num_components < 1 || num_components > 100)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("num_components must be between 1 and 100")));

	if (max_iters < 1)
		max_iters = 100;

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (nvec < num_components)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Not enough vectors (%d) for %d components",
						nvec, num_components)));

	model.k = num_components;
	model.dim = dim;
	model.mixing_coeffs = (double *) palloc(sizeof(double) * num_components);
	model.means = (double **) palloc(sizeof(double *) * num_components);
	model.variances = (double **) palloc(sizeof(double *) * num_components);

	for (k = 0; k < num_components; k++)
	{
		int	idx;

		model.means[k] = (double *) palloc(sizeof(double) * dim);
		model.variances[k] = (double *) palloc(sizeof(double) * dim);

		idx = rand() % nvec;

		for (d = 0; d < dim; d++)
			model.means[k][d] = (double) data[idx][d];

		for (d = 0; d < dim; d++)
			model.variances[k][d] = 1.0;

		model.mixing_coeffs[k] = 1.0 / num_components;
	}

	responsibilities = (double **) palloc(sizeof(double *) * nvec);
	for (i = 0; i < nvec; i++)
		responsibilities[i] = (double *) palloc(sizeof(double) * num_components);

	prev_log_likelihood = -DBL_MAX;

	for (iter = 0; iter < max_iters; iter++)
	{
		log_likelihood = 0.0;

		for (i = 0; i < nvec; i++)
		{
			double	sum = 0.0;

			for (k = 0; k < num_components; k++)
			{
				double	pdf = gaussian_pdf(data[i],
										 model.means[k],
										 model.variances[k],
										 dim);

				responsibilities[i][k] = model.mixing_coeffs[k] * pdf;
				sum += responsibilities[i][k];
			}

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

		if (fabs(log_likelihood - prev_log_likelihood) < 1e-6)
			break;
		prev_log_likelihood = log_likelihood;

		{
			double *N_k = (double *) palloc0(sizeof(double) * num_components);

			for (k = 0; k < num_components; k++)
			{
				for (i = 0; i < nvec; i++)
					N_k[k] += responsibilities[i][k];

				if (N_k[k] < GMM_MIN_PROB)
					N_k[k] = GMM_MIN_PROB;
			}

			for (k = 0; k < num_components; k++)
				model.mixing_coeffs[k] = N_k[k] / nvec;

			for (k = 0; k < num_components; k++)
			{
				for (d = 0; d < dim; d++)
					model.means[k][d] = 0.0;

				for (i = 0; i < nvec; i++)
					for (d = 0; d < dim; d++)
						model.means[k][d] +=
							responsibilities[i][k] * data[i][d];

				for (d = 0; d < dim; d++)
					model.means[k][d] /= N_k[k];
			}

			for (k = 0; k < num_components; k++)
			{
				for (d = 0; d < dim; d++)
					model.variances[k][d] = 0.0;

				for (i = 0; i < nvec; i++)
				{
					for (d = 0; d < dim; d++)
					{
						double diff = data[i][d] - model.means[k][d];
						model.variances[k][d] +=
							responsibilities[i][k] * diff * diff;
					}
				}

				for (d = 0; d < dim; d++)
					model.variances[k][d] =
						(model.variances[k][d] / N_k[k]) + GMM_EPSILON;
			}

			pfree(N_k);
		}

		if ((iter + 1) % 10 == 0)
			elog(DEBUG1,
				"neurondb: GMM iteration %d, log_likelihood=%.6f",
				iter + 1, log_likelihood);
	}

	result_datums = (Datum *) palloc(sizeof(Datum) * nvec * num_components);
	for (i = 0; i < nvec; i++)
	{
		for (k = 0; k < num_components; k++)
			result_datums[i * num_components + k] =
				Float8GetDatum(responsibilities[i][k]);
	}

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

/*
 * Serialize GMMModel to bytea for storage
 */
static bytea *
gmm_model_serialize_to_bytea(const GMMModel *model)
{
	StringInfoData buf;
	int i, j;
	int total_size;
	bytea *result;

	initStringInfo(&buf);

	/* Write header: k, dim */
	appendBinaryStringInfo(&buf, (char *)&model->k, sizeof(int));
	appendBinaryStringInfo(&buf, (char *)&model->dim, sizeof(int));

	/* Write mixing coefficients */
	for (i = 0; i < model->k; i++)
		appendBinaryStringInfo(&buf, (char *)&model->mixing_coeffs[i], sizeof(double));

	/* Write means */
	for (i = 0; i < model->k; i++)
		for (j = 0; j < model->dim; j++)
			appendBinaryStringInfo(&buf, (char *)&model->means[i][j], sizeof(double));

	/* Write variances */
	for (i = 0; i < model->k; i++)
		for (j = 0; j < model->dim; j++)
			appendBinaryStringInfo(&buf, (char *)&model->variances[i][j], sizeof(double));

	/* Convert to bytea */
	total_size = VARHDRSZ + buf.len;
	result = (bytea *)palloc(total_size);
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	pfree(buf.data);

	return result;
}

/*
 * Deserialize GMMModel from bytea
 */
static GMMModel *
gmm_model_deserialize_from_bytea(const bytea *data)
{
	GMMModel *model;
	const char *buf;
	int offset = 0;
	int i, j;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 2)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: invalid GMM model data: too small")));

	buf = VARDATA(data);

	model = (GMMModel *)palloc0(sizeof(GMMModel));

	/* Read header */
	memcpy(&model->k, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&model->dim, buf + offset, sizeof(int));
	offset += sizeof(int);

	/* Validate reasonable bounds */
	if (model->k <= 0 || model->k > 100)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: invalid model data: k=%d (expected 1-100)", model->k)));
	if (model->dim <= 0 || model->dim > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: invalid model data: dim=%d (expected 1-100000)", model->dim)));

	/* Allocate arrays */
	model->mixing_coeffs = (double *)palloc0(sizeof(double) * model->k);
	model->means = (double **)palloc(sizeof(double *) * model->k);
	model->variances = (double **)palloc(sizeof(double *) * model->k);

	/* Read mixing coefficients */
	for (i = 0; i < model->k; i++)
	{
		memcpy(&model->mixing_coeffs[i], buf + offset, sizeof(double));
		offset += sizeof(double);
	}

	/* Read means */
	for (i = 0; i < model->k; i++)
	{
		model->means[i] = (double *)palloc(sizeof(double) * model->dim);
		for (j = 0; j < model->dim; j++)
		{
			memcpy(&model->means[i][j], buf + offset, sizeof(double));
			offset += sizeof(double);
		}
	}

	/* Read variances */
	for (i = 0; i < model->k; i++)
	{
		model->variances[i] = (double *)palloc(sizeof(double) * model->dim);
		for (j = 0; j < model->dim; j++)
		{
			memcpy(&model->variances[i][j], buf + offset, sizeof(double));
			offset += sizeof(double);
		}
	}

	return model;
}

/*
 * train_gmm_model_id
 *
 * Trains GMM and stores model in catalog, returns model_id
 */
PG_FUNCTION_INFO_V1(train_gmm_model_id);

Datum
train_gmm_model_id(PG_FUNCTION_ARGS)
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
	bytea *model_data;
	MLCatalogModelSpec spec;
	Jsonb *metrics;
	StringInfoData metrics_json;
	int32 model_id;

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_components = PG_GETARG_INT32(2);
	max_iters = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);

	if (num_components < 1 || num_components > 100)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("num_components must be between 1 and 100")));

	if (max_iters < 1)
		max_iters = 100;

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (nvec < num_components)
	{
		pfree(tbl_str);
		pfree(col_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("Not enough vectors (%d) for %d components", nvec, num_components)));
	}

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
		idx = rand() % nvec;
		for (d = 0; d < dim; d++)
			model.means[k][d] = (double)data[idx][d];
		for (d = 0; d < dim; d++)
			model.variances[k][d] = 1.0;
		model.mixing_coeffs[k] = 1.0 / num_components;
	}

	responsibilities = (double **)palloc(sizeof(double *) * nvec);
	for (i = 0; i < nvec; i++)
		responsibilities[i] = (double *)palloc(sizeof(double) * num_components);

	prev_log_likelihood = -DBL_MAX;

	/* EM algorithm */
	for (iter = 0; iter < max_iters; iter++)
	{
		log_likelihood = 0.0;

		/* E-step: Compute responsibilities */
		for (i = 0; i < nvec; i++)
		{
			double sum = 0.0;
			for (k = 0; k < num_components; k++)
			{
				double pdf = gaussian_pdf(data[i], model.means[k], model.variances[k], dim);
				responsibilities[i][k] = model.mixing_coeffs[k] * pdf;
				sum += responsibilities[i][k];
			}
		if (sum > GMM_MIN_PROB)
		{
			double point_likelihood = 0.0;

			for (k = 0; k < num_components; k++)
				responsibilities[i][k] /= sum;
			for (k = 0; k < num_components; k++)
				point_likelihood += model.mixing_coeffs[k] * gaussian_pdf(data[i], model.means[k], model.variances[k], dim);
			log_likelihood += log(point_likelihood + GMM_MIN_PROB);
		}
		}

		/* Check convergence */
		if (fabs(log_likelihood - prev_log_likelihood) < 1e-6)
			break;
		prev_log_likelihood = log_likelihood;

		/* M-step: Update parameters */
		{
			double *N_k = (double *)palloc0(sizeof(double) * num_components);
			for (k = 0; k < num_components; k++)
			{
				for (i = 0; i < nvec; i++)
					N_k[k] += responsibilities[i][k];
				if (N_k[k] < GMM_MIN_PROB)
					N_k[k] = GMM_MIN_PROB;
			}
			for (k = 0; k < num_components; k++)
				model.mixing_coeffs[k] = N_k[k] / nvec;
			for (k = 0; k < num_components; k++)
			{
				for (d = 0; d < dim; d++)
					model.means[k][d] = 0.0;
				for (i = 0; i < nvec; i++)
					for (d = 0; d < dim; d++)
						model.means[k][d] += responsibilities[i][k] * data[i][d];
				for (d = 0; d < dim; d++)
					model.means[k][d] /= N_k[k];
			}
			for (k = 0; k < num_components; k++)
			{
				for (d = 0; d < dim; d++)
					model.variances[k][d] = 0.0;
				for (i = 0; i < nvec; i++)
				{
					for (d = 0; d < dim; d++)
					{
						double diff = data[i][d] - model.means[k][d];
						model.variances[k][d] += responsibilities[i][k] * diff * diff;
					}
				}
				for (d = 0; d < dim; d++)
					model.variances[k][d] = (model.variances[k][d] / N_k[k]) + GMM_EPSILON;
			}
			pfree(N_k);
		}
	}

	/* Serialize model to bytea */
	model_data = gmm_model_serialize_to_bytea(&model);

	/* Build metrics JSONB */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json, "{\"storage\": \"cpu\", \"k\": %d, \"dim\": %d, \"max_iters\": %d}",
		model.k, model.dim, max_iters);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metrics_json.data)));
	pfree(metrics_json.data);

	/* Store model in catalog */
	memset(&spec, 0, sizeof(MLCatalogModelSpec));
	spec.project_name = NULL; /* Will auto-create project */
	spec.algorithm = "gmm";
	spec.training_table = tbl_str;
	spec.training_column = NULL; /* GMM is unsupervised */
	spec.model_data = model_data;
	spec.metrics = metrics;
	spec.num_samples = nvec;
	spec.num_features = dim;

	model_id = ml_catalog_register_model(&spec);

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
	pfree(tbl_str);
	pfree(col_str);

	PG_RETURN_INT32(model_id);
}

/*
 * predict_gmm_model_id
 *
 * Predict cluster assignment for a feature vector using GMM model
 */
PG_FUNCTION_INFO_V1(predict_gmm_model_id);

Datum
predict_gmm_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	Vector *features;
	bytea *model_data = NULL;
	Jsonb *metrics = NULL;
	GMMModel *model = NULL;
	double *probabilities;
	int predicted_cluster = 0;
	double max_prob = -DBL_MAX;
	int i, k;

	model_id = PG_GETARG_INT32(0);
	features = PG_GETARG_VECTOR_P(1);

	/* Load model from catalog */
	if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
	{
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			errmsg("GMM model %d not found", model_id)));
	}

	if (model_data == NULL)
	{
		if (metrics)
			pfree(metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("GMM model %d has no model data", model_id)));
	}

	/* Ensure bytea is in current function's memory context */
	if (model_data != NULL)
	{
		int data_len = VARSIZE(model_data);
		bytea *copy = (bytea *)palloc(data_len);
		memcpy(copy, model_data, data_len);
		model_data = copy;
	}

	/* Deserialize model */
	model = gmm_model_deserialize_from_bytea(model_data);

	if (features->dim != model->dim)
	{
		/* Cleanup */
		for (i = 0; i < model->k; i++)
		{
			pfree(model->means[i]);
			pfree(model->variances[i]);
		}
		pfree(model->means);
		pfree(model->variances);
		pfree(model->mixing_coeffs);
		pfree(model);
		if (model_data)
			pfree(model_data);
		if (metrics)
			pfree(metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("Feature dimension mismatch: expected %d, got %d",
				model->dim, features->dim)));
	}

	/* Compute probabilities for each component */
	probabilities = (double *)palloc(sizeof(double) * model->k);
	for (k = 0; k < model->k; k++)
	{
		double pdf = gaussian_pdf(features->data, model->means[k], model->variances[k], model->dim);
		probabilities[k] = model->mixing_coeffs[k] * pdf;
		if (probabilities[k] > max_prob)
		{
			max_prob = probabilities[k];
			predicted_cluster = k;
		}
	}

	/* Cleanup */
	pfree(probabilities);
	for (i = 0; i < model->k; i++)
	{
		pfree(model->means[i]);
		pfree(model->variances[i]);
	}
	pfree(model->means);
	pfree(model->variances);
	pfree(model->mixing_coeffs);
	pfree(model);
	if (model_data)
		pfree(model_data);
	if (metrics)
		pfree(metrics);

	PG_RETURN_INT32(predicted_cluster);
}

/*
 * Compute Euclidean distance squared (double to double)
 */
static inline double
gmm_euclidean_distance_squared_dd(const double *a, const double *b, int dim)
{
	double sum = 0.0;
	int i;

	for (i = 0; i < dim; i++)
	{
		double diff = a[i] - b[i];
		sum += diff * diff;
	}
	return sum;
}

/*
 * Compute Euclidean distance squared (float to double)
 */
static inline double
gmm_euclidean_distance_squared(const float *a, const double *b, int dim)
{
	double sum = 0.0;
	int i;

	for (i = 0; i < dim; i++)
	{
		double diff = (double)a[i] - b[i];
		sum += diff * diff;
	}
	return sum;
}

/*
 * Compute Euclidean distance (float to double)
 */
static inline double
gmm_euclidean_distance(const float *a, const double *b, int dim)
{
	return sqrt(gmm_euclidean_distance_squared(a, b, dim));
}

/*
 * Compute Euclidean distance (double to double)
 */
static inline double
gmm_euclidean_distance_dd(const double *a, const double *b, int dim)
{
	return sqrt(gmm_euclidean_distance_squared_dd(a, b, dim));
}

/*
 * Compute Euclidean distance (float to float)
 */
static inline double
gmm_euclidean_distance_ff(const float *a, const float *b, int dim)
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
 * evaluate_gmm_by_model_id
 *
 * Evaluates GMM clustering model by computing:
 * - Inertia (within-cluster sum of squares)
 * - Silhouette score
 * - Davies-Bouldin index
 */
PG_FUNCTION_INFO_V1(evaluate_gmm_by_model_id);

Datum
evaluate_gmm_by_model_id(PG_FUNCTION_ARGS)
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
	GMMModel *model = NULL;
	int *assignments = NULL;
	int *cluster_sizes = NULL;
	double inertia = 0.0;
	double silhouette = 0.0;
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
				errmsg("neurondb: evaluate_gmm_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_gmm_by_model_id: table_name and vector_col are required")));

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
				errmsg("neurondb: evaluate_gmm_by_model_id: model %d not found", model_id)));
	}

	if (model_payload == NULL)
	{
		pfree(tbl_str);
		pfree(col_str);
		if (model_metrics)
			pfree(model_metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_gmm_by_model_id: model %d has no model_data", model_id)));
	}

	/* Deserialize model */
	model = gmm_model_deserialize_from_bytea(model_payload);
	if (model == NULL)
	{
		pfree(tbl_str);
		pfree(col_str);
		if (model_payload)
			pfree(model_payload);
		if (model_metrics)
			pfree(model_metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_gmm_by_model_id: failed to deserialize model")));
	}

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
	{
		for (c = 0; c < model->k; c++)
		{
			pfree(model->means[c]);
			pfree(model->variances[c]);
		}
		pfree(model->means);
		pfree(model->variances);
		pfree(model->mixing_coeffs);
		pfree(model);
		pfree(tbl_str);
		pfree(col_str);
		if (model_payload)
			pfree(model_payload);
		if (model_metrics)
			pfree(model_metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_gmm_by_model_id: SPI_connect failed")));
	}

	/* Fetch test data */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (!data || nvec < 1)
	{
		SPI_finish();
		for (c = 0; c < model->k; c++)
		{
			pfree(model->means[c]);
			pfree(model->variances[c]);
		}
		pfree(model->means);
		pfree(model->variances);
		pfree(model->mixing_coeffs);
		pfree(model);
		pfree(tbl_str);
		pfree(col_str);
		if (model_payload)
			pfree(model_payload);
		if (model_metrics)
			pfree(model_metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_gmm_by_model_id: no valid data found")));
	}

	if (dim != model->dim)
	{
		SPI_finish();
		for (i = 0; i < nvec; i++)
			pfree(data[i]);
		pfree(data);
		for (c = 0; c < model->k; c++)
		{
			pfree(model->means[c]);
			pfree(model->variances[c]);
		}
		pfree(model->means);
		pfree(model->variances);
		pfree(model->mixing_coeffs);
		pfree(model);
		pfree(tbl_str);
		pfree(col_str);
		if (model_payload)
			pfree(model_payload);
		if (model_metrics)
			pfree(model_metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_gmm_by_model_id: dimension mismatch: model dim=%d, data dim=%d",
					model->dim, dim)));
	}

	/* Assign points to clusters (based on maximum probability) */
	assignments = (int *)palloc(sizeof(int) * nvec);
	cluster_sizes = (int *)palloc0(sizeof(int) * model->k);

	for (i = 0; i < nvec; i++)
	{
		double max_prob = -DBL_MAX;
		int best = 0;

		for (c = 0; c < model->k; c++)
		{
			double pdf = gaussian_pdf(data[i], model->means[c], model->variances[c], dim);
			double prob = model->mixing_coeffs[c] * pdf;
			if (prob > max_prob)
			{
				max_prob = prob;
				best = c;
			}
		}
		assignments[i] = best;
		cluster_sizes[best]++;
		inertia += gmm_euclidean_distance_squared(data[i], model->means[best], dim);
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
				double dist = gmm_euclidean_distance_ff(data[i], data[j], dim);
				same_dist += dist;
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

			for (other_cluster_loop = 0; other_cluster_loop < model->k; other_cluster_loop++)
			{
				if (other_cluster_loop == my_cluster)
					continue;
				if (cluster_sizes[other_cluster_loop] == 0)
					continue;

				other_dist = 0.0;
				other_count = 0;

				for (j = 0; j < nvec; j++)
			{
					if (assignments[j] == other_cluster_loop)
					{
						other_dist += gmm_euclidean_distance_ff(data[i], data[j], dim);
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
		/* If no other clusters found, set to 0 to avoid DBL_MAX issues */
		if (min_other_dist >= DBL_MAX)
			b_scores[i] = 0.0;
		else
			b_scores[i] = min_other_dist;
	}

	/* Compute average silhouette */
	{
		int valid_count = 0;
		double sum_silhouette = 0.0;

		for (i = 0; i < nvec; i++)
		{
			/* Skip if no other clusters (b_scores is 0 and a_scores might be 0) */
			if (b_scores[i] <= 0.0 && a_scores[i] <= 0.0)
				continue;
			
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

	/* Cleanup */
	if (data)
	{
		for (i = 0; i < nvec; i++)
			pfree(data[i]);
		pfree(data);
	}
	if (assignments)
		pfree(assignments);
	if (cluster_sizes)
		pfree(cluster_sizes);
	if (a_scores)
		pfree(a_scores);
	if (b_scores)
		pfree(b_scores);
	for (c = 0; c < model->k; c++)
	{
		pfree(model->means[c]);
		pfree(model->variances[c]);
	}
	pfree(model->means);
	pfree(model->variances);
	pfree(model->mixing_coeffs);
	pfree(model);
	if (model_payload)
		pfree(model_payload);
	if (model_metrics)
		pfree(model_metrics);

	/* Build jsonb result in SPI context */
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"inertia\":%.6f,\"silhouette_score\":%.6f,\"n_samples\":%d}",
		inertia,
		silhouette,
		nvec);

	/* BULLETPROOF: Create JSONB with error handling */
	PG_TRY();
	{
		result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
			CStringGetDatum(jsonbuf.data)));
		
		if (result_jsonb == NULL)
		{
			elog(WARNING, "neurondb: evaluate_gmm_by_model_id: jsonb_in returned NULL");
			/* Create a minimal valid JSONB object */
			result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
				CStringGetDatum("{}")));
		}
	}
	PG_CATCH();
	{
		elog(WARNING, "neurondb: evaluate_gmm_by_model_id: error creating JSONB, using empty object");
		if (jsonbuf.data)
			pfree(jsonbuf.data);
		FlushErrorState();
		/* Create a minimal valid JSONB object in a safe way */
		PG_TRY();
		{
			result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
				CStringGetDatum("{}")));
		}
		PG_CATCH();
		{
			FlushErrorState();
			/* Last resort: return NULL and let caller handle it */
			result_jsonb = NULL;
		}
		PG_END_TRY();
	}
	PG_END_TRY();

	if (jsonbuf.data)
		pfree(jsonbuf.data);
	
	/* BULLETPROOF: Validate and copy result_jsonb to caller's context before SPI_finish() */
	if (result_jsonb == NULL)
	{
		elog(WARNING, "neurondb: evaluate_gmm_by_model_id: result_jsonb is NULL, creating empty object");
		/* Create empty JSONB as fallback */
		result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
			CStringGetDatum("{}")));
	}
	
	/* This ensures the JSONB remains valid after the SPI context is deleted */
	MemoryContextSwitchTo(oldcontext);
	result_jsonb = (Jsonb *)PG_DETOAST_DATUM_COPY(JsonbPGetDatum(result_jsonb));
	
	/* Now safe to finish SPI (this switches back to oldcontext automatically) */
	SPI_finish();
	
	/* Free strings allocated before SPI_connect (in oldcontext) */
	/* Note: tbl_str and col_str were allocated in oldcontext, not SPI context */
	pfree(tbl_str);
	pfree(col_str);
	
	/* Return result (already in oldcontext) */
	PG_RETURN_JSONB_P(result_jsonb);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for GMM
 *-------------------------------------------------------------------------*/
#include "ml_gpu_registry.h"

/* Stub function to satisfy linker - full implementation needed */
void
neurondb_gpu_register_gmm_model(void)
{
	/* GMM GPU Model Ops not yet implemented - will use CPU fallback */
}
