/*-------------------------------------------------------------------------
 *
 * ml_naive_bayes.c
 *    Naive Bayes classifier implementation
 *
 * Implements Gaussian Naive Bayes for continuous features.
 * Assumes features follow Gaussian distribution within each class.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_naive_bayes.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/memutils.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "neurondb_gpu_backend.h"
#include "ml_catalog.h"
#include "neurondb_cuda_nb.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
extern int ndb_cuda_nb_evaluate(const bytea *model_data,
	const float *features,
	const int *labels,
	int n_samples,
	int feature_dim,
	double *accuracy_out,
	double *precision_out,
	double *recall_out,
	double *f1_out,
	char **errstr);
#endif

#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Gaussian Naive Bayes model parameters
 */
typedef struct
{
	double *class_priors; /* P(class) */
	double **means; /* Mean for each feature per class */
	double **variances; /* Variance for each feature per class */
	int n_classes;
	int n_features;
} GaussianNBModel;

/*
 * Gaussian probability density function
 */
static double
gaussian_pdf(double x, double mean, double variance)
{
	double exponent;

	if (variance < 1e-9)
		variance = 1e-9; /* Avoid division by zero */

	exponent = -0.5 * pow((x - mean), 2) / variance;
	return (1.0 / sqrt(2.0 * M_PI * variance)) * exp(exponent);
}

/*
 * train_naive_bayes_classifier
 *
 * Trains a Gaussian Naive Bayes classifier
 * Returns model parameters as JSONB
 */
PG_FUNCTION_INFO_V1(train_naive_bayes_classifier);

Datum
train_naive_bayes_classifier(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *feature_col;
	text *label_col;
	char *tbl_str;
	char *feat_str;
	char *label_str;
	StringInfoData query;
	int ret;
	int nvec = 0;
	int dim = 0;
	float **X = NULL;
	double *y = NULL;
	GaussianNBModel model;
	int i, j, class;
	int n_params;
	int idx;
	int *class_counts;
	double ***class_samples; /* [class][sample_idx][feature] */
	int *class_sizes;
	Datum *result_datums;
	ArrayType *result_array;
	MemoryContext oldcontext;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: SPI_connect failed")));

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		feat_str,
		label_str,
		tbl_str,
		feat_str,
		label_str);
	elog(DEBUG1, "train_naive_bayes_classifier: executing query: %s", query.data);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: query failed")));

	nvec = SPI_processed;

	if (nvec < 10)
		ereport(ERROR,
			(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				errmsg("neurondb: need at least 10 samples")));

	/* Allocate arrays */
	MemoryContextSwitchTo(oldcontext);
	X = (float **)palloc(sizeof(float *) * nvec);
	y = (double *)palloc(sizeof(double) * nvec);

	/* Extract data */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum feat_datum;
		Datum label_datum;
		bool feat_null;
		bool label_null;
		Vector *vec;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);

		if (feat_null || label_null)
			continue;

		vec = DatumGetVector(feat_datum);
		if (vec == NULL || vec->dim <= 0)
			continue;

		if (i == 0)
			dim = vec->dim;

		X[i] = (float *)palloc(sizeof(float) * dim);
		memcpy(X[i], vec->data, sizeof(float) * dim);

		y[i] = DatumGetFloat8(label_datum);
	}

	SPI_finish();

	/* Initialize model for binary classification */
	model.n_classes = 2;
	model.n_features = dim;
	model.class_priors = (double *)palloc0(sizeof(double) * 2);
	model.means = (double **)palloc(sizeof(double *) * 2);
	model.variances = (double **)palloc(sizeof(double *) * 2);

	class_counts = (int *)palloc0(sizeof(int) * 2);
	class_sizes = (int *)palloc0(sizeof(int) * 2);
	class_samples = (double ***)palloc(sizeof(double **) * 2);

	/* Count samples per class */
	for (i = 0; i < nvec; i++)
	{
		class = (int)y[i];
		if (class >= 0 && class < 2)
			class_counts[class]++;
	}

	/* Allocate class sample arrays */
	for (class = 0; class < 2; class ++)
	{
		class_samples[class] = (double **)palloc(
			sizeof(double *) * class_counts[class]);
		model.means[class] = (double *)palloc0(sizeof(double) * dim);
		model.variances[class] =
			(double *)palloc0(sizeof(double) * dim);
	}

	/* Group samples by class */
	for (i = 0; i < nvec; i++)
	{
		class = (int)y[i];
		if (class >= 0 && class < 2)
		{
			class_samples[class][class_sizes[class]] =
				(double *)palloc(sizeof(double) * dim);
			for (j = 0; j < dim; j++)
				class_samples[class][class_sizes[class]][j] =
					X[i][j];
			class_sizes[class]++;
		}
	}

	/* Compute class priors, means, and variances */
	for (class = 0; class < 2; class ++)
	{
		model.class_priors[class] = (double)class_sizes[class] / nvec;

		/* Check for empty class to avoid division by zero */
		if (class_sizes[class] == 0)
		{
			/* Set default values for empty class */
			for (j = 0; j < dim; j++)
			{
				model.means[class][j] = 0.0;
				model.variances[class][j] = 1.0;
			}
			continue;
		}

		/* Compute means */
		for (j = 0; j < dim; j++)
		{
			double sum = 0.0;
			for (i = 0; i < class_sizes[class]; i++)
				sum += class_samples[class][i][j];
			model.means[class][j] = sum / class_sizes[class];
		}

		/* Compute variances */
		for (j = 0; j < dim; j++)
		{
			double sum_sq = 0.0;
			for (i = 0; i < class_sizes[class]; i++)
			{
				double diff = class_samples[class][i][j]
					- model.means[class][j];
				sum_sq += diff * diff;
			}
			model.variances[class][j] = sum_sq / class_sizes[class];
		}
	}

	/* Serialize model parameters to array */
	/* Format: [n_classes, n_features, prior0, prior1, mean0_0, mean0_1, ..., var0_0, var0_1, ...] */
	n_params = 2 + 2 + (2 * dim)
		+ (2 * dim); /* metadata + priors + means + variances */
	result_datums = (Datum *)palloc(sizeof(Datum) * n_params);

	result_datums[0] = Float8GetDatum(model.n_classes);
	result_datums[1] = Float8GetDatum(model.n_features);
	result_datums[2] = Float8GetDatum(model.class_priors[0]);
	result_datums[3] = Float8GetDatum(model.class_priors[1]);

	idx = 4;
	for (class = 0; class < 2; class ++)
		for (j = 0; j < dim; j++)
			result_datums[idx++] =
				Float8GetDatum(model.means[class][j]);

	for (class = 0; class < 2; class ++)
		for (j = 0; j < dim; j++)
			result_datums[idx++] =
				Float8GetDatum(model.variances[class][j]);

	result_array = construct_array(result_datums,
		n_params,
		FLOAT8OID,
		sizeof(float8),
		FLOAT8PASSBYVAL,
		'd');

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		pfree(X[i]);
	pfree(X);
	pfree(y);
	for (class = 0; class < 2; class ++)
	{
		for (i = 0; i < class_sizes[class]; i++)
			pfree(class_samples[class][i]);
		pfree(class_samples[class]);
		pfree(model.means[class]);
		pfree(model.variances[class]);
	}
	pfree(class_samples);
	pfree(class_counts);
	pfree(class_sizes);
	pfree(model.class_priors);
	pfree(model.means);
	pfree(model.variances);
	pfree(result_datums);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * Serialize GaussianNBModel to bytea for storage
 */
static bytea *
nb_model_serialize_to_bytea(const GaussianNBModel *model)
{
	StringInfoData buf;
	int i, j;
	int total_size;
	bytea *result;

	initStringInfo(&buf);

	/* Write header: n_classes, n_features */
	appendBinaryStringInfo(&buf, (char *)&model->n_classes, sizeof(int));
	appendBinaryStringInfo(&buf, (char *)&model->n_features, sizeof(int));

	/* Write class priors */
	for (i = 0; i < model->n_classes; i++)
		appendBinaryStringInfo(&buf, (char *)&model->class_priors[i], sizeof(double));

	/* Write means */
	for (i = 0; i < model->n_classes; i++)
		for (j = 0; j < model->n_features; j++)
			appendBinaryStringInfo(&buf, (char *)&model->means[i][j], sizeof(double));

	/* Write variances */
	for (i = 0; i < model->n_classes; i++)
		for (j = 0; j < model->n_features; j++)
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
 * Deserialize bytea to GaussianNBModel
 */
static GaussianNBModel *
nb_model_deserialize_from_bytea(const bytea *data)
{
	GaussianNBModel *model;
	const char *buf;
	int offset = 0;
	int i, j;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 2)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: invalid model data: too small")));

	buf = VARDATA(data);

	model = (GaussianNBModel *)palloc0(sizeof(GaussianNBModel));

	/* Read header */
	memcpy(&model->n_classes, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&model->n_features, buf + offset, sizeof(int));
	offset += sizeof(int);

	/* Validate reasonable bounds */
	if (model->n_classes <= 0 || model->n_classes > 1000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: invalid model data: n_classes=%d (expected 1-1000)",
					model->n_classes)));
	if (model->n_features <= 0 || model->n_features > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: invalid model data: n_features=%d (expected 1-100000)",
					model->n_features)));
	
	/* Verify we have enough data */
	{
		int expected_size = sizeof(int) * 2 + /* header */
			sizeof(double) * model->n_classes + /* priors */
			sizeof(double) * model->n_classes * model->n_features + /* means */
			sizeof(double) * model->n_classes * model->n_features; /* variances */
		int actual_size = VARSIZE(data) - VARHDRSZ;
		if (actual_size < expected_size)
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: invalid model data: expected %d bytes, got %d bytes",
						expected_size, actual_size)));
	}

	/* Allocate arrays */
	model->class_priors = (double *)palloc0(sizeof(double) * model->n_classes);
	model->means = (double **)palloc(sizeof(double *) * model->n_classes);
	model->variances = (double **)palloc(sizeof(double *) * model->n_classes);

	/* Read class priors */
	for (i = 0; i < model->n_classes; i++)
	{
		memcpy(&model->class_priors[i], buf + offset, sizeof(double));
		offset += sizeof(double);
	}

	/* Read means */
	for (i = 0; i < model->n_classes; i++)
	{
		model->means[i] = (double *)palloc(sizeof(double) * model->n_features);
		for (j = 0; j < model->n_features; j++)
		{
			memcpy(&model->means[i][j], buf + offset, sizeof(double));
			offset += sizeof(double);
		}
	}

	/* Read variances */
	for (i = 0; i < model->n_classes; i++)
	{
		model->variances[i] = (double *)palloc(sizeof(double) * model->n_features);
		for (j = 0; j < model->n_features; j++)
		{
			memcpy(&model->variances[i][j], buf + offset, sizeof(double));
			offset += sizeof(double);
		}
	}

	return model;
}

/*
 * train_naive_bayes_classifier_model_id
 *
 * Trains Naive Bayes and stores model in catalog, returns model_id
 */
PG_FUNCTION_INFO_V1(train_naive_bayes_classifier_model_id);

Datum
train_naive_bayes_classifier_model_id(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *feature_col;
	text *label_col;
	char *tbl_str;
	char *feat_str;
	char *label_str;
	StringInfoData query;
	int ret;
	int nvec = 0;
	int dim = 0;
	float **X = NULL;
	double *y = NULL;
	GaussianNBModel model;
	int i, j, class;
	int *class_counts;
	double ***class_samples;
	int *class_sizes;
	MemoryContext oldcontext;
	bytea *model_data;
	MLCatalogModelSpec spec;
	Jsonb *metrics;
	StringInfoData metrics_json;
	int32 model_id;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: SPI_connect failed")));

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		feat_str,
		label_str,
		tbl_str,
		feat_str,
		label_str);
	elog(DEBUG1, "train_naive_bayes_classifier_model_id: executing query: %s", query.data);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: query failed")));

	nvec = SPI_processed;

	if (nvec < 10)
		ereport(ERROR,
			(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				errmsg("neurondb: need at least 10 samples")));

	/* Allocate arrays */
	MemoryContextSwitchTo(oldcontext);
	X = (float **)palloc(sizeof(float *) * nvec);
	y = (double *)palloc(sizeof(double) * nvec);

	/* Extract data */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum feat_datum;
		Datum label_datum;
		bool feat_null;
		bool label_null;
		Vector *vec;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);

		if (feat_null || label_null)
			continue;

		vec = DatumGetVector(feat_datum);
		if (vec == NULL || vec->dim <= 0)
			continue;

		if (i == 0)
			dim = vec->dim;

		X[i] = (float *)palloc(sizeof(float) * dim);
		memcpy(X[i], vec->data, sizeof(float) * dim);

		y[i] = DatumGetFloat8(label_datum);
	}

	SPI_finish();

	/* Initialize model for binary classification */
	model.n_classes = 2;
	model.n_features = dim;
	model.class_priors = (double *)palloc0(sizeof(double) * 2);
	model.means = (double **)palloc(sizeof(double *) * 2);
	model.variances = (double **)palloc(sizeof(double *) * 2);

	class_counts = (int *)palloc0(sizeof(int) * 2);
	class_sizes = (int *)palloc0(sizeof(int) * 2);
	class_samples = (double ***)palloc(sizeof(double **) * 2);

	/* Count samples per class */
	for (i = 0; i < nvec; i++)
	{
		class = (int)y[i];
		if (class >= 0 && class < 2)
			class_counts[class]++;
	}

	/* Allocate class sample arrays */
	for (class = 0; class < 2; class++)
	{
		class_samples[class] = (double **)palloc(
			sizeof(double *) * class_counts[class]);
		model.means[class] = (double *)palloc0(sizeof(double) * dim);
		model.variances[class] =
			(double *)palloc0(sizeof(double) * dim);
	}

	/* Group samples by class */
	for (i = 0; i < nvec; i++)
	{
		class = (int)y[i];
		if (class >= 0 && class < 2)
		{
			class_samples[class][class_sizes[class]] =
				(double *)palloc(sizeof(double) * dim);
			for (j = 0; j < dim; j++)
				class_samples[class][class_sizes[class]][j] =
					X[i][j];
			class_sizes[class]++;
		}
	}

	/* Compute class priors, means, and variances */
	for (class = 0; class < 2; class++)
	{
		model.class_priors[class] = (double)class_sizes[class] / nvec;

		/* Check for empty class to avoid division by zero */
		if (class_sizes[class] == 0)
		{
			/* Set default values for empty class */
			for (j = 0; j < dim; j++)
			{
				model.means[class][j] = 0.0;
				model.variances[class][j] = 1.0;
			}
			continue;
		}

		/* Compute means */
		for (j = 0; j < dim; j++)
		{
			double sum = 0.0;
			for (i = 0; i < class_sizes[class]; i++)
				sum += class_samples[class][i][j];
			model.means[class][j] = sum / class_sizes[class];
		}

		/* Compute variances */
		for (j = 0; j < dim; j++)
		{
			double sum_sq = 0.0;
			for (i = 0; i < class_sizes[class]; i++)
			{
				double diff = class_samples[class][i][j]
					- model.means[class][j];
				sum_sq += diff * diff;
			}
			model.variances[class][j] = sum_sq / class_sizes[class];
		}
	}

	/* Serialize model to bytea */
	model_data = nb_model_serialize_to_bytea(&model);

	/* Build metrics JSONB */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json, "{\"storage\": \"cpu\", \"n_classes\": %d, \"n_features\": %d}",
		model.n_classes, model.n_features);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metrics_json.data)));
	pfree(metrics_json.data);

	/* Store model in catalog */
	memset(&spec, 0, sizeof(MLCatalogModelSpec));
	spec.project_name = NULL; /* Will auto-create project */
	spec.algorithm = "naive_bayes";
	spec.training_table = tbl_str;
	spec.training_column = label_str;
	spec.model_data = model_data;
	spec.metrics = metrics;
	spec.num_samples = nvec;
	spec.num_features = dim;

	model_id = ml_catalog_register_model(&spec);

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		pfree(X[i]);
	pfree(X);
	pfree(y);
	for (class = 0; class < 2; class++)
	{
		for (i = 0; i < class_sizes[class]; i++)
			pfree(class_samples[class][i]);
		pfree(class_samples[class]);
		pfree(model.means[class]);
		pfree(model.variances[class]);
	}
	pfree(class_samples);
	pfree(class_counts);
	pfree(class_sizes);
	pfree(model.class_priors);
	pfree(model.means);
	pfree(model.variances);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(label_str);

	PG_RETURN_INT32(model_id);
}

/*
 * predict_naive_bayes
 *
 * Predicts class using trained Naive Bayes model
 */
PG_FUNCTION_INFO_V1(predict_naive_bayes);

Datum
predict_naive_bayes(PG_FUNCTION_ARGS)
{
	ArrayType *model_params;
	Vector *features;
	float8 *params;
	int n_params;
	int n_classes;
	int n_features;
	double *class_priors;
	double **means;
	double **variances;
	double log_probs[2] = { 0.0, 0.0 };
	int predicted_class;
	int i, j, idx;

	model_params = PG_GETARG_ARRAYTYPE_P(0);
	features = PG_GETARG_VECTOR_P(1);

	if (ARR_NDIM(model_params) != 1)
		ereport(ERROR,
			(errmsg("Model parameters must be 1-dimensional "
				"array")));

	n_params = ARR_DIMS(model_params)[0];
	(void)n_params; /* Suppress unused variable warning */
	params = (float8 *)ARR_DATA_PTR(model_params);

	n_classes = (int)params[0];
	n_features = (int)params[1];

	if (features->dim != n_features)
		ereport(ERROR,
			(errmsg("neurondb: predict_naive_bayes: feature dimension mismatch: expected %d, got %d",
				n_features, features->dim)));

	/* Extract model parameters */
	class_priors = (double *)palloc(sizeof(double) * n_classes);
	means = (double **)palloc(sizeof(double *) * n_classes);
	variances = (double **)palloc(sizeof(double *) * n_classes);

	class_priors[0] = params[2];
	class_priors[1] = params[3];

	idx = 4;
	for (i = 0; i < n_classes; i++)
	{
		means[i] = (double *)palloc(sizeof(double) * n_features);
		for (j = 0; j < n_features; j++)
			means[i][j] = params[idx++];
	}

	for (i = 0; i < n_classes; i++)
	{
		variances[i] = (double *)palloc(sizeof(double) * n_features);
		for (j = 0; j < n_features; j++)
			variances[i][j] = params[idx++];
	}

	/* Compute log probabilities for each class */
	for (i = 0; i < n_classes; i++)
	{
		log_probs[i] = log(class_priors[i]);

		for (j = 0; j < n_features; j++)
		{
			double pdf = gaussian_pdf(features->data[j],
				means[i][j],
				variances[i][j]);
			log_probs[i] += log(pdf
				+ 1e-10); /* Add small constant to avoid log(0) */
		}
	}

	/* Return class with highest log probability */
	predicted_class = (log_probs[1] > log_probs[0]) ? 1 : 0;

	/* Cleanup */
	pfree(class_priors);
	for (i = 0; i < n_classes; i++)
	{
		pfree(means[i]);
		pfree(variances[i]);
	}
	pfree(means);
	pfree(variances);

	PG_RETURN_INT32(predicted_class);
}

/*
 * predict_naive_bayes_model_id
 *
 * Predicts class using Naive Bayes model from catalog (model_id)
 */
PG_FUNCTION_INFO_V1(predict_naive_bayes_model_id);

Datum
predict_naive_bayes_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	Vector *features;
	bytea *model_data = NULL;
	Jsonb *metrics = NULL;
	GaussianNBModel *model = NULL;
	double log_probs[2] = { 0.0, 0.0 };
	int predicted_class;
	int i, j;

	model_id = PG_GETARG_INT32(0);
	features = PG_GETARG_VECTOR_P(1);

	/* Load model from catalog (ml_catalog_fetch_model_payload handles SPI internally) */
	if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
	{
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			errmsg("Naive Bayes model %d not found", model_id)));
	}

	if (model_data == NULL)
	{
		if (metrics)
			pfree(metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("Naive Bayes model %d has no model data", model_id)));
	}

	/* Ensure bytea is in current function's memory context */
	/* ml_catalog_fetch_model_payload may return data in a different context */
	if (model_data != NULL)
	{
		int data_len = VARSIZE(model_data);
		bytea *copy = (bytea *)palloc(data_len);
		memcpy(copy, model_data, data_len);
		/* Free original if it was allocated (ml_catalog_fetch_model_payload uses caller context) */
		/* But we can't free it here as we don't know the context */
		model_data = copy;
	}

	/* model_data is now guaranteed to be in current memory context and detoasted */
	/* Deserialize model */
	model = nb_model_deserialize_from_bytea(model_data);

	if (features->dim != model->n_features)
	{
		/* Cleanup */
		pfree(model->class_priors);
		for (i = 0; i < model->n_classes; i++)
		{
			pfree(model->means[i]);
			pfree(model->variances[i]);
		}
		pfree(model->means);
		pfree(model->variances);
		pfree(model);
		if (model_data)
			pfree(model_data);
		if (metrics)
			pfree(metrics);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("Feature dimension mismatch: expected %d, got %d",
				model->n_features, features->dim)));
	}

	/* Compute log probabilities for each class */
	for (i = 0; i < model->n_classes; i++)
	{
		log_probs[i] = log(model->class_priors[i]);

		for (j = 0; j < model->n_features; j++)
		{
			double pdf = gaussian_pdf(features->data[j],
				model->means[i][j],
				model->variances[i][j]);
			log_probs[i] += log(pdf + 1e-10); /* Add small constant to avoid log(0) */
		}
	}

	/* Return class with highest log probability */
	predicted_class = (log_probs[1] > log_probs[0]) ? 1 : 0;

	/* Cleanup */
	pfree(model->class_priors);
	for (i = 0; i < model->n_classes; i++)
	{
		pfree(model->means[i]);
		pfree(model->variances[i]);
	}
	pfree(model->means);
	pfree(model->variances);
	pfree(model);
	if (model_data)
		pfree(model_data);
	if (metrics)
		pfree(metrics);

	PG_RETURN_INT32(predicted_class);
}

/*
 * nb_predict_batch
 *
 * Helper function to predict a batch of samples using Naive Bayes model.
 * Updates confusion matrix.
 */
static void
nb_predict_batch(const GaussianNBModel *model,
	const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	int *tp_out,
	int *tn_out,
	int *fp_out,
	int *fn_out)
{
	int i;
	int tp = 0;
	int tn = 0;
	int fp = 0;
	int fn = 0;

	if (model == NULL || features == NULL || labels == NULL || n_samples <= 0)
	{
		if (tp_out)
			*tp_out = 0;
		if (tn_out)
			*tn_out = 0;
		if (fp_out)
			*fp_out = 0;
		if (fn_out)
			*fn_out = 0;
		return;
	}

	for (i = 0; i < n_samples; i++)
	{
		const float *row = features + (i * feature_dim);
		double y_true = labels[i];
		int true_class;
		double log_probs[2] = { 0.0, 0.0 };
		int pred_class;
		int j;
		int c;

		if (!isfinite(y_true))
			continue;

		true_class = (int)rint(y_true);
		if (true_class < 0 || true_class > 1)
			continue;

		/* Compute log probabilities for each class */
		for (c = 0; c < model->n_classes; c++)
		{
			log_probs[c] = log(model->class_priors[c]);

			for (j = 0; j < feature_dim; j++)
			{
				double pdf = gaussian_pdf((double)row[j],
					model->means[c][j],
					model->variances[c][j]);
				log_probs[c] += log(pdf + 1e-10); /* Add small constant to avoid log(0) */
			}
		}

		/* Return class with highest log probability */
		pred_class = (log_probs[1] > log_probs[0]) ? 1 : 0;

		/* Update confusion matrix */
		if (true_class == 1 && pred_class == 1)
			tp++;
		else if (true_class == 0 && pred_class == 0)
			tn++;
		else if (true_class == 0 && pred_class == 1)
			fp++;
		else if (true_class == 1 && pred_class == 0)
			fn++;
	}

	if (tp_out)
		*tp_out = tp;
	if (tn_out)
		*tn_out = tn;
	if (fp_out)
		*fp_out = fp;
	if (fn_out)
		*fn_out = fn;
}

/*
 * evaluate_naive_bayes_by_model_id
 *
 * Evaluates Naive Bayes model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: accuracy, precision, recall, f1_score, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_naive_bayes_by_model_id);

Datum
evaluate_naive_bayes_by_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	text *table_name;
	text *feature_col;
	text *label_col;
	char *tbl_str;
	char *feat_str;
	char *targ_str;
	int ret;
	int nvec = 0;
	int i;
	int j;
	int feat_dim = 0;
	Oid feat_type_oid = InvalidOid;
	bool feat_is_array = false;
	double accuracy = 0.0;
	double precision = 0.0;
	double recall = 0.0;
	double f1_score = 0.0;
	int tp = 0;
	int tn = 0;
	int fp = 0;
	int fn = 0;
	MemoryContext oldcontext;
	StringInfoData query;
	GaussianNBModel *model = NULL;
	StringInfoData jsonbuf;
	Jsonb *result_jsonb = NULL;
	bytea *gpu_payload = NULL;
	Jsonb *gpu_metrics = NULL;
	bool is_gpu_model = false;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_naive_bayes_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_naive_bayes_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Load model from catalog - try CPU first, then GPU */
	if (!ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
	{
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_naive_bayes_by_model_id: model %d not found",
					model_id)));
	}

	/* Check if GPU model */
	if (gpu_payload != NULL && gpu_metrics != NULL)
	{
		char *meta_txt = NULL;
		bool is_gpu = false;

		PG_TRY();
		{
			meta_txt = DatumGetCString(DirectFunctionCall1(
				jsonb_out, JsonbPGetDatum(gpu_metrics)));
			/* Check for both "storage":"gpu" and "storage": "gpu" formats */
			if (strstr(meta_txt, "\"storage\":\"gpu\"") != NULL ||
				strstr(meta_txt, "\"storage\": \"gpu\"") != NULL)
				is_gpu = true;
			if (meta_txt != NULL)
				pfree(meta_txt);
		}
		PG_CATCH();
		{
			/* Invalid JSONB, assume CPU */
			is_gpu = false;
		}
		PG_END_TRY();

		is_gpu_model = is_gpu;
	}

	/* For CPU models, deserialize from bytea */
	if (!is_gpu_model && gpu_payload != NULL)
	{
		bytea *copy;
		int data_len = VARSIZE(gpu_payload);

		copy = (bytea *)palloc(data_len);
		memcpy(copy, gpu_payload, data_len);
		model = nb_model_deserialize_from_bytea(copy);
		pfree(copy);
	}

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
	{
		if (model != NULL)
		{
			pfree(model->class_priors);
			for (i = 0; i < model->n_classes; i++)
			{
				pfree(model->means[i]);
				pfree(model->variances[i]);
			}
			pfree(model->means);
			pfree(model->variances);
			pfree(model);
		}
		if (gpu_payload)
			pfree(gpu_payload);
		if (gpu_metrics)
			pfree(gpu_metrics);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_naive_bayes_by_model_id: SPI_connect failed")));
	}

	/* Build query - single query to fetch all data */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quote_identifier(feat_str),
		quote_identifier(targ_str),
		quote_identifier(tbl_str),
		quote_identifier(feat_str),
		quote_identifier(targ_str));
	elog(DEBUG1, "evaluate_naive_bayes_by_model_id: executing query: %s", query.data);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		pfree(query.data);
		if (model != NULL)
		{
			pfree(model->class_priors);
			for (i = 0; i < model->n_classes; i++)
			{
				pfree(model->means[i]);
				pfree(model->variances[i]);
			}
			pfree(model->means);
			pfree(model->variances);
			pfree(model);
		}
		if (gpu_payload)
			pfree(gpu_payload);
		if (gpu_metrics)
			pfree(gpu_metrics);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_naive_bayes_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		/* Check if table/view exists and has any rows at all */
		StringInfoData check_query;
		int check_ret;
		int total_rows = 0;
		
		initStringInfo(&check_query);
		appendStringInfo(&check_query,
			"SELECT COUNT(*) FROM %s",
			quote_identifier(tbl_str));
		check_ret = SPI_execute(check_query.data, true, 0);
		if (check_ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool isnull;
			Datum count_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
			if (!isnull)
				total_rows = DatumGetInt64(count_datum);
		}
		pfree(check_query.data);
		
		pfree(query.data);
		if (model != NULL)
		{
			pfree(model->class_priors);
			for (i = 0; i < model->n_classes; i++)
			{
				pfree(model->means[i]);
				pfree(model->variances[i]);
			}
			pfree(model->means);
			pfree(model->variances);
			pfree(model);
		}
		if (gpu_payload)
			pfree(gpu_payload);
		if (gpu_metrics)
			pfree(gpu_metrics);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		SPI_finish();
		
		if (total_rows == 0)
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_naive_bayes_by_model_id: table/view '%s' has no rows",
						tbl_str),
					errhint("Ensure the table/view exists and contains data")));
		}
		else
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_naive_bayes_by_model_id: no valid rows found in '%s' (table has %d total rows, but all have NULL values in '%s' or '%s')",
						tbl_str, total_rows, feat_str, targ_str),
					errhint("Ensure columns '%s' and '%s' are not NULL for evaluation rows",
						feat_str, targ_str)));
		}
	}

	/* Determine feature column type */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/* GPU batch evaluation path for GPU models - uses optimized evaluation kernel */
	if (is_gpu_model && neurondb_gpu_is_available())
	{
#ifdef NDB_GPU_CUDA
		const NdbCudaNbModelHeader *gpu_hdr;
		int *h_labels = NULL;
		float *h_features = NULL;
		int valid_rows = 0;
		size_t payload_size;

		/* Defensive check: validate payload size */
		payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
		if (payload_size < sizeof(NdbCudaNbModelHeader))
		{
				 elog(DEBUG1,
				 	"neurondb: evaluate_naive_bayes_by_model_id: GPU payload too small (%zu bytes), falling back to CPU",
				 payload_size);
			goto cpu_evaluation_path;
		}

		/* Load GPU model header with defensive checks */
		gpu_hdr = (const NdbCudaNbModelHeader *)VARDATA(gpu_payload);
		if (gpu_hdr == NULL)
		{
			elog(DEBUG1, "neurondb: evaluate_naive_bayes_by_model_id: NULL GPU header, falling back to CPU");
			goto cpu_evaluation_path;
		}

		feat_dim = gpu_hdr->n_features;
		if (feat_dim <= 0 || feat_dim > 100000)
		{
			elog(DEBUG1, "neurondb: evaluate_naive_bayes_by_model_id: invalid feature_dim (%d), falling back to CPU", feat_dim);
			goto cpu_evaluation_path;
		}

		/* Allocate host buffers for features and labels with size checks */
		{
			size_t features_size = sizeof(float) * (size_t)nvec * (size_t)feat_dim;
			size_t labels_size = sizeof(int) * (size_t)nvec;

			if (features_size > MaxAllocSize || labels_size > MaxAllocSize)
			{
					 elog(DEBUG1,
					 	"neurondb: evaluate_naive_bayes_by_model_id: allocation size too large (features=%zu, labels=%zu), falling back to CPU",
					 features_size, labels_size);
				goto cpu_evaluation_path;
			}

			h_features = (float *)palloc(features_size);
			h_labels = (int *)palloc(labels_size);

			if (h_features == NULL || h_labels == NULL)
			{
				elog(DEBUG1, "neurondb: evaluate_naive_bayes_by_model_id: memory allocation failed, falling back to CPU");
				if (h_features)
					pfree(h_features);
				if (h_labels)
					pfree(h_labels);
				goto cpu_evaluation_path;
			}
		}

		/* Extract features and labels from SPI results - optimized batch extraction */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc tupdesc = SPI_tuptable->tupdesc;

			if (tupdesc == NULL)
			{
				elog(DEBUG1,
					"neurondb: evaluate_naive_bayes_by_model_id: NULL TupleDesc, falling back to CPU");
				pfree(h_features);
				pfree(h_labels);
				goto cpu_evaluation_path;
			}

			for (i = 0; i < nvec; i++)
			{
				HeapTuple tuple;
				Datum feat_datum;
				Datum targ_datum;
				bool feat_null;
				bool targ_null;
				Vector *vec;
				ArrayType *arr;
				float *feat_row;

				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || i >= SPI_processed)
					break;

				tuple = SPI_tuptable->vals[i];
				if (tuple == NULL)
					continue;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				/* Bounds check */
				if (valid_rows >= nvec)
				{
					elog(DEBUG1,
						"neurondb: evaluate_naive_bayes_by_model_id: valid_rows overflow, breaking");
					break;
				}

				feat_row = h_features + (valid_rows * feat_dim);
				if (feat_row == NULL || feat_row < h_features || feat_row >= h_features + (nvec * feat_dim))
				{
					elog(DEBUG1,
						"neurondb: evaluate_naive_bayes_by_model_id: feat_row out of bounds, skipping row");
					continue;
				}

				h_labels[valid_rows] = (int)rint(DatumGetFloat8(targ_datum));

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						/* Optimized: bulk conversion with loop unrolling hint */
						float8 *data = (float8 *)ARR_DATA_PTR(arr);
						int j_remain = feat_dim % 4;
						int j_end = feat_dim - j_remain;

						/* Process 4 elements at a time for better cache locality */
						for (j = 0; j < j_end; j += 4)
						{
							feat_row[j] = (float)data[j];
							feat_row[j + 1] = (float)data[j + 1];
							feat_row[j + 2] = (float)data[j + 2];
							feat_row[j + 3] = (float)data[j + 3];
						}
						/* Handle remaining elements */
						for (j = j_end; j < feat_dim; j++)
							feat_row[j] = (float)data[j];
					}
					else
					{
						/* FLOAT4ARRAYOID: direct memcpy (already optimal) */
						float4 *data = (float4 *)ARR_DATA_PTR(arr);
						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					/* Vector type: direct memcpy (already optimal) */
					vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_rows++;
			}
		}

		if (valid_rows == 0)
		{
			pfree(h_features);
			pfree(h_labels);
			if (gpu_payload)
				pfree(gpu_payload);
			if (gpu_metrics)
				pfree(gpu_metrics);
			pfree(query.data);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_naive_bayes_by_model_id: no valid rows found in '%s' (query returned %d rows, but all were filtered out due to NULL values or dimension mismatches)",
						tbl_str, nvec),
					errhint("Ensure columns '%s' and '%s' are not NULL and feature dimensions match the model (expected %d)",
						feat_str, targ_str, feat_dim)));
		}

		/* Use optimized GPU batch evaluation */
		{
			int rc;
			char *gpu_errstr = NULL;

			/* Defensive checks before GPU call */
			if (h_features == NULL || h_labels == NULL || valid_rows <= 0 || feat_dim <= 0)
			{
					 elog(DEBUG1,
					 	"neurondb: evaluate_naive_bayes_by_model_id: invalid inputs for GPU evaluation (features=%p, labels=%p, rows=%d, dim=%d), falling back to CPU",
					 (void *)h_features, (void *)h_labels, valid_rows, feat_dim);
				pfree(h_features);
				pfree(h_labels);
				goto cpu_evaluation_path;
			}

			PG_TRY();
			{
				rc = ndb_cuda_nb_evaluate_batch(gpu_payload,
					h_features,
					h_labels,
					valid_rows,
					feat_dim,
					&accuracy,
					&precision,
					&recall,
					&f1_score,
					&gpu_errstr);

				if (rc == 0)
				{
					/* Success - build result and return */
					initStringInfo(&jsonbuf);
					appendStringInfo(&jsonbuf,
						"{\"accuracy\":%.6f,\"precision\":%.6f,\"recall\":%.6f,\"f1_score\":%.6f,\"n_samples\":%d}",
						accuracy,
						precision,
						recall,
						f1_score,
						valid_rows);

					result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
						CStringGetDatum(jsonbuf.data)));

					pfree(jsonbuf.data);
					pfree(h_features);
					pfree(h_labels);
					if (gpu_payload)
						pfree(gpu_payload);
					if (gpu_metrics)
						pfree(gpu_metrics);
					if (gpu_errstr)
						pfree(gpu_errstr);
					pfree(query.data);
					pfree(tbl_str);
					pfree(feat_str);
					pfree(targ_str);
					SPI_finish();
					MemoryContextSwitchTo(oldcontext);
					PG_RETURN_JSONB_P(result_jsonb);
				}
				else
				{
					/* GPU evaluation failed - fall back to CPU */
						 elog(DEBUG1,
						 	"evaluate_naive_bayes_by_model_id: GPU batch evaluation failed: %s, falling back to CPU",
						 gpu_errstr ? gpu_errstr : "unknown error");
					if (gpu_errstr)
						pfree(gpu_errstr);
					pfree(h_features);
					pfree(h_labels);
					goto cpu_evaluation_path;
				}
			}
			PG_CATCH();
			{
				elog(DEBUG1,
					"evaluate_naive_bayes_by_model_id: exception during GPU evaluation, falling back to CPU");
				if (h_features)
					pfree(h_features);
				if (h_labels)
					pfree(h_labels);
				goto cpu_evaluation_path;
			}
			PG_END_TRY();
		}
#endif	/* NDB_GPU_CUDA */
	}

cpu_evaluation_path:

	/* CPU evaluation path */
	/* Use optimized batch prediction */
	{
		float *h_features = NULL;
		double *h_labels = NULL;
		int valid_rows = 0;

		/* Determine feature dimension from model */
		if (model != NULL)
			feat_dim = model->n_features;
		else if (is_gpu_model && gpu_payload != NULL)
		{
			const NdbCudaNbModelHeader *gpu_hdr;

			gpu_hdr = (const NdbCudaNbModelHeader *)VARDATA(gpu_payload);
			feat_dim = gpu_hdr->n_features;
		}
		else
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_naive_bayes_by_model_id: could not determine feature dimension")));
		}

		if (feat_dim <= 0)
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_naive_bayes_by_model_id: invalid feature dimension %d",
						feat_dim)));
		}

		/* Allocate host buffers for features and labels */
		h_features = (float *)palloc(sizeof(float) * (size_t)nvec * (size_t)feat_dim);
		h_labels = (double *)palloc(sizeof(double) * (size_t)nvec);

		/* Extract features and labels from SPI results - optimized batch extraction */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc tupdesc = SPI_tuptable->tupdesc;

			for (i = 0; i < nvec; i++)
			{
				HeapTuple tuple = SPI_tuptable->vals[i];
				Datum feat_datum;
				Datum targ_datum;
				bool feat_null;
				bool targ_null;
				Vector *vec;
				ArrayType *arr;
				float *feat_row;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				feat_row = h_features + (valid_rows * feat_dim);
				h_labels[valid_rows] = DatumGetFloat8(targ_datum);

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						/* Optimized: bulk conversion with loop unrolling hint */
						float8 *data = (float8 *)ARR_DATA_PTR(arr);
						int j_remain = feat_dim % 4;
						int j_end = feat_dim - j_remain;

						/* Process 4 elements at a time for better cache locality */
						for (j = 0; j < j_end; j += 4)
						{
							feat_row[j] = (float)data[j];
							feat_row[j + 1] = (float)data[j + 1];
							feat_row[j + 2] = (float)data[j + 2];
							feat_row[j + 3] = (float)data[j + 3];
						}
						/* Handle remaining elements */
						for (j = j_end; j < feat_dim; j++)
							feat_row[j] = (float)data[j];
					}
					else
					{
						/* FLOAT4ARRAYOID: direct memcpy (already optimal) */
						float4 *data = (float4 *)ARR_DATA_PTR(arr);
						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					/* Vector type: direct memcpy (already optimal) */
					vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_rows++;
			}
		}

		if (valid_rows == 0)
		{
			pfree(h_features);
			pfree(h_labels);
			if (model != NULL)
			{
				pfree(model->class_priors);
				for (i = 0; i < model->n_classes; i++)
				{
					pfree(model->means[i]);
					pfree(model->variances[i]);
				}
				pfree(model->means);
				pfree(model->variances);
				pfree(model);
			}
			if (gpu_payload)
				pfree(gpu_payload);
			if (gpu_metrics)
				pfree(gpu_metrics);
			pfree(query.data);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_naive_bayes_by_model_id: no valid rows found in '%s' (query returned %d rows, but all were filtered out due to NULL values or dimension mismatches)",
						tbl_str, nvec),
					errhint("Ensure columns '%s' and '%s' are not NULL and feature dimensions match the model (expected %d)",
						feat_str, targ_str, feat_dim)));
		}

		/* For GPU models, we cannot evaluate on CPU without model conversion */
		if (is_gpu_model && model == NULL)
		{
			pfree(h_features);
			pfree(h_labels);
			if (gpu_payload)
				pfree(gpu_payload);
			if (gpu_metrics)
				pfree(gpu_metrics);
			pfree(query.data);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					errmsg("neurondb: evaluate_naive_bayes_by_model_id: GPU model evaluation requires GPU evaluation kernel (not yet implemented)")));
		}

		/* Use batch prediction helper */
		nb_predict_batch(model,
			h_features,
			h_labels,
			valid_rows,
			feat_dim,
			&tp,
			&tn,
			&fp,
			&fn);

		/* Compute metrics */
		if (valid_rows > 0)
		{
			accuracy = (double)(tp + tn) / (double)valid_rows;

			if ((tp + fp) > 0)
				precision = (double)tp / (double)(tp + fp);
			else
				precision = 0.0;

			if ((tp + fn) > 0)
				recall = (double)tp / (double)(tp + fn);
			else
				recall = 0.0;

			if ((precision + recall) > 0.0)
				f1_score = 2.0 * (precision * recall) / (precision + recall);
			else
				f1_score = 0.0;
		}

		/* Cleanup */
		pfree(h_features);
		pfree(h_labels);
		if (model != NULL)
		{
			pfree(model->class_priors);
			for (i = 0; i < model->n_classes; i++)
			{
				pfree(model->means[i]);
				pfree(model->variances[i]);
			}
			pfree(model->means);
			pfree(model->variances);
			pfree(model);
		}
		if (gpu_payload)
			pfree(gpu_payload);
		if (gpu_metrics)
			pfree(gpu_metrics);
	}

	SPI_finish();
	pfree(query.data);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(targ_str);

	/* Build jsonb result */
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"accuracy\":%.6f,\"precision\":%.6f,\"recall\":%.6f,\"f1_score\":%.6f,\"n_samples\":%d}",
		accuracy,
		precision,
		recall,
		f1_score,
		nvec);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
		CStringGetDatum(jsonbuf.data)));

	pfree(jsonbuf.data);
	MemoryContextSwitchTo(oldcontext);
	PG_RETURN_JSONB_P(result_jsonb);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Implementation for Naive Bayes
 *-------------------------------------------------------------------------
 */

typedef struct NbGpuModelState
{
	bytea *model_blob;
	Jsonb *metrics;
	int feature_dim;
	int n_samples;
	int n_classes;
} NbGpuModelState;

static void
nb_gpu_release_state(NbGpuModelState *state)
{
	if (state == NULL)
		return;
	if (state->model_blob)
		pfree(state->model_blob);
	if (state->metrics)
		pfree(state->metrics);
	pfree(state);
}

static bool
nb_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	NbGpuModelState *state;
	bytea *payload;
	Jsonb *metrics;
	int rc;
	const ndb_gpu_backend *backend;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
		return false;
	if (!neurondb_gpu_is_available())
		return false;
	if (spec->feature_matrix == NULL || spec->label_vector == NULL)
		return false;
	if (spec->sample_count <= 0 || spec->feature_dim <= 0)
		return false;

	backend = ndb_gpu_get_active_backend();
	if (backend == NULL || backend->nb_train == NULL)
		return false;

	/* Ensure CUDA is initialized in this backend process */
	if (backend->init && backend->init() != 0)
	{
		if (errstr)
			*errstr = pstrdup("Failed to initialize GPU backend");
		return false;
	}

	payload = NULL;
	metrics = NULL;
	rc = backend->nb_train(spec->feature_matrix, spec->label_vector,
						   spec->sample_count, spec->feature_dim, spec->class_count,
						   NULL,  /* hyperparams */
						   &payload, &metrics, errstr);
	
	if (rc != 0 || payload == NULL)
		return false;

	state = (NbGpuModelState *)palloc0(sizeof(NbGpuModelState));
	state->model_blob = payload;
	state->metrics = metrics;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;
	state->n_classes = spec->class_count;

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
nb_gpu_predict(const MLGpuModel *model,
			   const float *input,
			   int input_dim,
			   float *output,
			   int output_dim,
			   char **errstr)
{
	const NbGpuModelState *state;
	int class_out;
	double probability_out;
	int rc;
	const ndb_gpu_backend *backend;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || input == NULL || output == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const NbGpuModelState *)model->backend_state;
	if (state->model_blob == NULL)
		return false;

	backend = ndb_gpu_get_active_backend();
	if (backend == NULL || backend->nb_predict == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Naive Bayes GPU backend not available");
		return false;
	}

	/* Ensure CUDA is initialized in this backend process */
	if (backend->init && backend->init() != 0)
	{
		if (errstr)
			*errstr = pstrdup("Failed to initialize GPU backend");
		return false;
	}

	rc = backend->nb_predict(state->model_blob,
							 input,
							 input_dim,
							 &class_out,
							 &probability_out,
							 errstr);
	if (rc != 0)
		return false;

	if (output_dim >= 1)
		output[0] = (float)class_out;
	if (output_dim >= 2)
		output[1] = (float)probability_out;

	return true;
}

static __attribute__((unused)) bool
nb_gpu_evaluate(const MLGpuModel *model,
				const MLGpuEvalSpec *spec,
				MLGpuMetrics *out,
				char **errstr)
{
	/* Defensive: validate inputs */
	if (model == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("nb_gpu_evaluate: model is NULL");
		return false;
	}

	if (spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("nb_gpu_evaluate: spec is NULL");
		return false;
	}

	if (out == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("nb_gpu_evaluate: out is NULL");
		return false;
	}

	if (errstr != NULL)
		*errstr = NULL;
	memset(out, 0, sizeof(MLGpuMetrics));

	/* Evaluation not implemented for GPU models yet */
	if (errstr != NULL)
		*errstr = pstrdup("GPU evaluation not yet implemented for Naive Bayes");
	return false;
}

static bool
nb_gpu_serialize(const MLGpuModel *model,
				 bytea **payload_out,
				 Jsonb **metadata_out,
				 char **errstr)
{
	/* GPU serialize disabled for Naive Bayes (CUDA fork-safety issue) */
	(void)model;
	(void)payload_out;
	(void)metadata_out;
	
	if (errstr)
		*errstr = pstrdup("GPU serialize disabled for Naive Bayes (use CPU fallback)");
	return false;
}

static bool
nb_gpu_deserialize(MLGpuModel *model,
				   const bytea *payload,
				   const Jsonb *metadata,
				   char **errstr)
{
	/* GPU deserialize disabled for Naive Bayes (CUDA fork-safety issue) */
	(void)model;
	(void)payload;
	(void)metadata;
	
	if (errstr)
		*errstr = pstrdup("GPU deserialize disabled for Naive Bayes (use CPU fallback)");
	return false;
}

static void
nb_gpu_destroy(MLGpuModel *model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		nb_gpu_release_state((NbGpuModelState *)model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps nb_gpu_model_ops = {
	.algorithm = "naive_bayes",
	.train = nb_gpu_train,
	.predict = nb_gpu_predict,
	.evaluate = NULL, /* Not yet implemented with correct signature */
	.serialize = nb_gpu_serialize,
	.deserialize = nb_gpu_deserialize,
	.destroy = nb_gpu_destroy,
};

#include "ml_gpu_registry.h"

void
neurondb_gpu_register_nb_model(void)
{
	static bool registered = false;

	if (registered)
		return;

	ndb_gpu_register_model_ops(&nb_gpu_model_ops);
	registered = true;
}
