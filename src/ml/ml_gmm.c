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
#include "ml_catalog.h"
#include "lib/stringinfo.h"
#include "utils/jsonb.h"
#include "vector/vector_types.h"
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
	double		result;

	/* Assert: Internal invariants */
	Assert(x != NULL);
	Assert(mean != NULL);
	Assert(variance != NULL);
	Assert(dim > 0);

	/* Defensive: Check for NULL pointers */
	if (x == NULL || mean == NULL || variance == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("gaussian_pdf: NULL pointer argument")));

	if (dim <= 0 || dim > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("gaussian_pdf: invalid dimension: %d", dim)));

	for (d = 0; d < dim; d++)
	{
		/* Defensive: Check for NaN/Inf */
		if (isnan(x[d]) || isnan(mean[d]) || isnan(variance[d]) ||
			isinf(x[d]) || isinf(mean[d]) || isinf(variance[d]))
		{
			elog(WARNING, "gaussian_pdf: NaN or Infinity detected at dimension %d", d);
		}

		double	diff = (double)x[d] - mean[d];
		double	var = variance[d] + GMM_EPSILON;

		/* Defensive: Ensure variance is positive */
		if (var <= 0.0)
		{
			elog(WARNING, "gaussian_pdf: non-positive variance at dimension %d: %f", d, var);
			var = GMM_EPSILON;
		}

		log_likelihood -= 0.5 * (diff * diff) / var;
		log_det += log(var);
	}

	log_likelihood -= 0.5 * (dim * log(2.0 * M_PI) + log_det);

	/* Defensive: Check for overflow */
	if (isnan(log_likelihood) || isinf(log_likelihood))
	{
		elog(WARNING, "gaussian_pdf: log_likelihood overflow");
		return 0.0;
	}

	result = exp(log_likelihood);

	/* Defensive: Validate result */
	if (isnan(result) || isinf(result))
	{
		elog(WARNING, "gaussian_pdf: result is NaN or Infinity");
		return 0.0;
	}

	return result;
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

	CHECK_NARGS_RANGE(3, 4);

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_components = PG_GETARG_INT32(2);
	max_iters = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);

	/* Defensive: Check NULL inputs */
	if (table_name == NULL || vector_column == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("cluster_gmm: table_name and vector_column cannot be NULL")));

	/* Defensive: Validate parameters */
	if (num_components < 1 || num_components > 100)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("num_components must be between 1 and 100, got %d", num_components)));

	if (max_iters < 1)
	{
		elog(WARNING, "cluster_gmm: max_iters < 1, using default 100");
		max_iters = 100;
	}

	if (max_iters > 10000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("max_iters must be at most 10000, got %d", max_iters)));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);

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
				errmsg("cluster_gmm: invalid table or column name length")));
	}

	elog(DEBUG1, "neurondb: GMM clustering (k=%d, max_iters=%d)",
		 num_components, max_iters);

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
		{
			elog(DEBUG1, "neurondb: GMM converged at iteration %d (ll=%.4f)",
				 iter + 1, log_likelihood);
			break;
		}
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
			elog(DEBUG2, "neurondb: GMM iteration %d, log-likelihood=%.4f",
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
		ereport(ERROR, (errmsg("Invalid GMM model data: too small")));

	buf = VARDATA(data);

	model = (GMMModel *)palloc0(sizeof(GMMModel));

	/* Read header */
	memcpy(&model->k, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&model->dim, buf + offset, sizeof(int));
	offset += sizeof(int);

	/* Validate reasonable bounds */
	if (model->k <= 0 || model->k > 100)
		ereport(ERROR, (errmsg("Invalid model data: k=%d (expected 1-100)", model->k)));
	if (model->dim <= 0 || model->dim > 100000)
		ereport(ERROR, (errmsg("Invalid model data: dim=%d (expected 1-100000)", model->dim)));

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
				for (k = 0; k < num_components; k++)
					responsibilities[i][k] /= sum;
				double point_likelihood = 0.0;
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

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for GMM
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"

/* Stub function to satisfy linker - full implementation needed */
void
neurondb_gpu_register_gmm_model(void)
{
	/* GMM GPU Model Ops not yet implemented - will use CPU fallback */
	elog(DEBUG1, "GMM GPU Model Ops registration skipped - not yet implemented");
}
