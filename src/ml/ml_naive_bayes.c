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

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "neurondb_gpu_backend.h"
#include "ml_catalog.h"

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
	double result;

	/* Defensive: Check for NaN/Inf inputs */
	if (isnan(x) || isnan(mean) || isnan(variance) ||
		isinf(x) || isinf(mean) || isinf(variance))
	{
		elog(WARNING, "gaussian_pdf: NaN or Infinity in inputs (x=%f, mean=%f, variance=%f)",
			x, mean, variance);
		return 0.0;
	}

	/* Defensive: Validate variance */
	if (variance < 0.0)
	{
		elog(WARNING, "gaussian_pdf: negative variance %f, using absolute value", variance);
		variance = fabs(variance);
	}

	if (variance < 1e-9)
		variance = 1e-9; /* Avoid division by zero */

	exponent = -0.5 * pow((x - mean), 2) / variance;

	/* Defensive: Check for overflow in exponent */
	if (isnan(exponent) || isinf(exponent))
	{
		elog(WARNING, "gaussian_pdf: exponent overflow (x=%f, mean=%f, variance=%f)",
			x, mean, variance);
		return 0.0;
	}

	result = (1.0 / sqrt(2.0 * M_PI * variance)) * exp(exponent);

	/* Defensive: Validate result */
	if (isnan(result) || isinf(result))
	{
		elog(WARNING, "gaussian_pdf: result is NaN or Infinity");
		return 0.0;
	}

	return result;
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

	CHECK_NARGS(3);

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);

	/* Defensive: Check NULL inputs */
	if (table_name == NULL || feature_col == NULL || label_col == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("train_naive_bayes_classifier: table_name, feature_col, and label_col cannot be NULL")));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	/* Defensive: Validate allocations */
	if (tbl_str == NULL || feat_str == NULL || label_str == NULL)
	{
		if (tbl_str)
			pfree(tbl_str);
		if (feat_str)
			pfree(feat_str);
		if (label_str)
			pfree(label_str);
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate strings")));
	}

	/* Defensive: Validate string lengths */
	if (strlen(tbl_str) == 0 || strlen(tbl_str) > NAMEDATALEN ||
		strlen(feat_str) == 0 || strlen(feat_str) > NAMEDATALEN ||
		strlen(label_str) == 0 || strlen(label_str) > NAMEDATALEN)
	{
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_NAME),
				errmsg("train_naive_bayes_classifier: invalid table or column name length")));
	}

	elog(DEBUG1, "train_naive_bayes_classifier: Starting training on table '%s'", tbl_str);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_connect failed")));

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		feat_str,
		label_str,
		tbl_str,
		feat_str,
		label_str);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR, (errmsg("Query failed")));

	nvec = SPI_processed;

	if (nvec < 10)
		ereport(ERROR, (errmsg("Need at least 10 samples")));

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
		ereport(ERROR, (errmsg("Invalid model data: too small")));

	buf = VARDATA(data);

	model = (GaussianNBModel *)palloc0(sizeof(GaussianNBModel));

	/* Read header */
	memcpy(&model->n_classes, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&model->n_features, buf + offset, sizeof(int));
	offset += sizeof(int);

	/* Validate reasonable bounds */
	if (model->n_classes <= 0 || model->n_classes > 1000)
		ereport(ERROR, (errmsg("Invalid model data: n_classes=%d (expected 1-1000)",
			model->n_classes)));
	if (model->n_features <= 0 || model->n_features > 100000)
		ereport(ERROR, (errmsg("Invalid model data: n_features=%d (expected 1-100000)",
			model->n_features)));
	
	/* Verify we have enough data */
	{
		int expected_size = sizeof(int) * 2 + /* header */
			sizeof(double) * model->n_classes + /* priors */
			sizeof(double) * model->n_classes * model->n_features + /* means */
			sizeof(double) * model->n_classes * model->n_features; /* variances */
		int actual_size = VARSIZE(data) - VARHDRSZ;
		if (actual_size < expected_size)
			ereport(ERROR, (errmsg("Invalid model data: expected %d bytes, got %d bytes",
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
	int project_id = 1; /* Default project */
	int version = 1;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("SPI_connect failed")));

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		feat_str,
		label_str,
		tbl_str,
		feat_str,
		label_str);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR, (errmsg("Query failed")));

	nvec = SPI_processed;

	if (nvec < 10)
		ereport(ERROR, (errmsg("Need at least 10 samples")));

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
			(errmsg("Feature dimension mismatch: expected %d, got "
				"%d",
				n_features,
				features->dim)));

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
	int ret;

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

static bool
nb_gpu_evaluate(const MLGpuModel *model,
				const float *features,
				const double *labels,
				int n_samples,
				int feature_dim,
				Jsonb **metrics_out,
				char **errstr)
{
	if (errstr != NULL)
		*errstr = NULL;
	if (metrics_out != NULL)
		*metrics_out = NULL;

	/* Evaluation not implemented for GPU models yet */
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
	.evaluate = nb_gpu_evaluate,
	.serialize = nb_gpu_serialize,
	.deserialize = nb_gpu_deserialize,
	.destroy = nb_gpu_destroy,
};

void
neurondb_gpu_register_nb_model(void)
{
	static bool registered = false;

	if (registered)
		return;

	elog(DEBUG1, "Registering Naive Bayes GPU Model Ops");
	ndb_gpu_register_model_ops(&nb_gpu_model_ops);
	registered = true;
	elog(DEBUG1, "Naive Bayes GPU Model Ops registered successfully");
}
