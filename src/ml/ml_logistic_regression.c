/*-------------------------------------------------------------------------
 *
 * ml_logistic_regression.c
 *    Logistic Regression implementation for binary classification
 *
 * Implements logistic regression using gradient descent with L2 regularization.
 * Supports binary classification with sigmoid activation.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_logistic_regression.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/memutils.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_logistic_regression_internal.h"
#include "ml_catalog.h"
#include "neurondb_gpu_bridge.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "neurondb_gpu_backend.h"
#include "ml_gpu_registry.h"
#include "ml_gpu_logistic_regression.h"
#include "utils/builtins.h"

#include <math.h>
#include <float.h>

typedef struct LRDataset
{
	float *features;
	double *labels;
	int n_samples;
	int feature_dim;
} LRDataset;

static void lr_dataset_init(LRDataset *dataset);
static void lr_dataset_free(LRDataset *dataset);
static void lr_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_label,
	LRDataset *dataset,
	StringInfo query);
static bytea *lr_model_serialize(const LRModel *model);
static LRModel *lr_model_deserialize(const bytea *data);
static bool lr_metadata_is_gpu(Jsonb *metadata);
static bool lr_try_gpu_predict_catalog(int32 model_id,
	const Vector *feature_vec,
	double *result_out);
static bool lr_load_model_from_catalog(int32 model_id, LRModel **out);

/*
 * Sigmoid function: 1 / (1 + exp(-z))
 */
static inline double
sigmoid(double z)
{
	if (z > 500)
		return 1.0;
	if (z < -500)
		return 0.0;
	return 1.0 / (1.0 + exp(-z));
}

/*
 * Compute log loss (binary cross-entropy)
 */
static double
compute_log_loss(double *y_true, double *y_pred, int n)
{
	double loss = 0.0;
	int i;

	for (i = 0; i < n; i++)
	{
		double pred = y_pred[i];
		/* Clip predictions to avoid log(0) */
		pred = fmax(1e-15, fmin(1.0 - 1e-15, pred));

		if (y_true[i] > 0.5)
			loss -= log(pred);
		else
			loss -= log(1.0 - pred);
	}

	return loss / n;
}

/*
 * train_logistic_regression
 *
 * Trains a logistic regression model using gradient descent
 * Returns coefficients (intercept + feature weights)
 */
PG_FUNCTION_INFO_V1(train_logistic_regression);

Datum
train_logistic_regression(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *feature_col;
	text *target_col;
	int max_iters = PG_GETARG_INT32(3);
	double learning_rate = PG_GETARG_FLOAT8(4);
	double lambda = PG_GETARG_FLOAT8(5); /* L2 regularization */
	char *tbl_str;
	char *feat_str;
	char *targ_str;
	StringInfoData query;
	int nvec = 0;
	int dim = 0;
	double *y = NULL;
	double *weights = NULL;
	double bias = 0.0;
	int iter;
	int i;
	int j;
	MemoryContext oldcontext;
	LRDataset dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_label;
	MLGpuTrainResult gpu_result;
	char *gpu_err = NULL;
	Jsonb *gpu_hyperparams = NULL;
	StringInfoData hyperbuf;
	int32 model_id = 0;

	if (PG_NARGS() != 6)
		ereport(ERROR,
			(errmsg("Usage: train_logistic_regression(table_name, "
				"feature_col, target_col, max_iters, "
				"learning_rate, lambda)")));

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	oldcontext = CurrentMemoryContext;

	lr_dataset_init(&dataset);
	initStringInfo(&query);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_label = quote_identifier(targ_str);

	lr_dataset_load(
		quoted_tbl, quoted_feat, quoted_label, &dataset, &query);

	nvec = dataset.n_samples;
	dim = dataset.feature_dim;

	if (nvec < 10)
	{
		lr_dataset_free(&dataset);
		pfree(query.data);
		ereport(ERROR,
			(errmsg("Need at least 10 samples for logistic "
				"regression, have %d",
				nvec)));
	}

	/* Try GPU training first */
	if (neurondb_gpu_is_available() && nvec > 0 && dim > 0)
	{
		initStringInfo(&hyperbuf);
		appendStringInfo(&hyperbuf,
			"{\"max_iters\":%d,\"learning_rate\":%.6f,\"lambda\":%."
			"6f}",
			max_iters,
			learning_rate,
			lambda);
		gpu_hyperparams = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(hyperbuf.data)));

		if (ndb_gpu_try_train_model("logistic_regression",
			    NULL,
			    NULL,
			    tbl_str,
			    targ_str,
			    NULL,
			    0,
			    gpu_hyperparams,
			    dataset.features,
			    dataset.labels,
			    nvec,
			    dim,
			    2,
			    &gpu_result,
			    &gpu_err)
			&& gpu_result.spec.model_data != NULL)
		{
			elog(NOTICE,
				"logistic_regression: GPU training succeeded");
			MLCatalogModelSpec spec = gpu_result.spec;

			if (spec.training_table == NULL)
				spec.training_table = tbl_str;
			if (spec.training_column == NULL)
				spec.training_column = targ_str;
			if (spec.parameters == NULL)
			{
				spec.parameters = gpu_hyperparams;
				gpu_hyperparams = NULL;
			}

			spec.algorithm = "logistic_regression";
			spec.model_type = "classification";

			/* Debug: verify metrics are set */
			if (spec.metrics != NULL)
			{
				char *metrics_txt = DatumGetCString(
					DirectFunctionCall1(jsonb_out,
						JsonbPGetDatum(spec.metrics)));
				elog(DEBUG1,
					"logistic_regression: GPU metrics: %s",
					metrics_txt);
				pfree(metrics_txt);
			} else
			{
				elog(WARNING,
					"logistic_regression: GPU metrics are "
					"NULL!");
			}

			model_id = ml_catalog_register_model(&spec);

			lr_dataset_free(&dataset);
			pfree(query.data);
			if (gpu_hyperparams)
				pfree(gpu_hyperparams);

			PG_RETURN_INT32(model_id);
		} else
		{
			if (gpu_err != NULL)
				elog(NOTICE,
					"logistic_regression: GPU training "
					"failed: %s",
					gpu_err);
			else
				elog(NOTICE,
					"logistic_regression: GPU training "
					"failed, falling back to CPU");
			if (gpu_hyperparams != NULL)
			{
				pfree(gpu_hyperparams);
				gpu_hyperparams = NULL;
			}
		}
	}

	/* Fall back to CPU training - use dataset directly */
	elog(NOTICE, "logistic_regression: Using CPU training path");
	/* Use dataset.features directly as 1D array, no need for X double pointer */
	y = dataset.labels;

	/* Initialize weights and bias */
	weights = (double *)palloc0(sizeof(double) * dim);

	/* Gradient descent */
	for (iter = 0; iter < max_iters; iter++)
	{
		double *predictions = (double *)palloc(sizeof(double) * nvec);
		double *grad_w = (double *)palloc0(sizeof(double) * dim);
		double grad_b = 0.0;

		/* Forward pass: compute predictions */
		for (i = 0; i < nvec; i++)
		{
			double z = bias;
			for (j = 0; j < dim; j++)
				z += weights[j] * dataset.features[i * dim + j];
			predictions[i] = sigmoid(z);
		}

		/* Backward pass: compute gradients */
		for (i = 0; i < nvec; i++)
		{
			double error = predictions[i] - y[i];
			grad_b += error;
			for (j = 0; j < dim; j++)
				grad_w[j] +=
					error * dataset.features[i * dim + j];
		}

		/* Average gradients and add L2 regularization */
		grad_b /= nvec;
		for (j = 0; j < dim; j++)
		{
			grad_w[j] = grad_w[j] / nvec + lambda * weights[j];
		}

		/* Update weights and bias */
		bias -= learning_rate * grad_b;
		for (j = 0; j < dim; j++)
			weights[j] -= learning_rate * grad_w[j];

		pfree(predictions);
		pfree(grad_w);

		/* Log progress every 100 iterations */
		if (iter % 100 == 0)
		{
			double *preds;
			double loss;

			preds = (double *)palloc(sizeof(double) * nvec);
			for (i = 0; i < nvec; i++)
			{
				double z = bias;
				for (j = 0; j < dim; j++)
					z += weights[j]
						* dataset.features[i * dim + j];
				preds[i] = sigmoid(z);
			}
			loss = compute_log_loss(y, preds, nvec);
			elog(NOTICE, "Iteration %d: loss = %f", iter, loss);
			pfree(preds);
		}
	}

	/* Build LRModel and register in catalog */
	{
		LRModel model;
		bytea *serialized;
		MLCatalogModelSpec spec;
		Jsonb *params_jsonb;
		Jsonb *metrics_jsonb;
		StringInfoData hyperbuf;
		StringInfoData metricsbuf;
		double final_loss;
		double *preds;
		int correct = 0;

		/* Compute final loss and accuracy */
		preds = (double *)palloc(sizeof(double) * nvec);
		for (i = 0; i < nvec; i++)
		{
			double z = bias;
			for (j = 0; j < dim; j++)
				z += weights[j] * dataset.features[i * dim + j];
			preds[i] = sigmoid(z);
			if ((preds[i] >= 0.5 && y[i] > 0.5)
				|| (preds[i] < 0.5 && y[i] <= 0.5))
				correct++;
		}
		final_loss = compute_log_loss(y, preds, nvec);
		pfree(preds);

		/* Build model struct */
		memset(&model, 0, sizeof(model));
		model.n_features = dim;
		model.n_samples = nvec;
		model.bias = bias;
		model.weights = (double *)palloc(sizeof(double) * dim);
		memcpy(model.weights, weights, sizeof(double) * dim);
		model.learning_rate = learning_rate;
		model.lambda = lambda;
		model.max_iters = max_iters;
		model.final_loss = final_loss;
		model.accuracy = (double)correct / (double)nvec;

		/* Serialize model */
		serialized = lr_model_serialize(&model);

		/* Build hyperparameters JSON */
		initStringInfo(&hyperbuf);
		appendStringInfo(&hyperbuf,
			"{\"max_iters\":%d,\"learning_rate\":%.6f,\"lambda\":%."
			"6f}",
			max_iters,
			learning_rate,
			lambda);
		params_jsonb = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(hyperbuf.data)));

		/* Build metrics JSON */
		initStringInfo(&metricsbuf);
		appendStringInfo(&metricsbuf,
			"{\"final_loss\":%.6f,\"accuracy\":%.6f,\"n_samples\":%"
			"d,\"n_features\":%d}",
			final_loss,
			model.accuracy,
			nvec,
			dim);
		metrics_jsonb = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(metricsbuf.data)));

		/* Register in catalog */
		memset(&spec, 0, sizeof(spec));
		spec.algorithm = "logistic_regression";
		spec.model_type = "classification";
		spec.training_table = tbl_str;
		spec.training_column = targ_str;
		spec.parameters = params_jsonb;
		spec.metrics = metrics_jsonb;
		spec.model_data = serialized;
		spec.training_time_ms = -1;
		spec.num_samples = nvec;
		spec.num_features = dim;

		model_id = ml_catalog_register_model(&spec);
		elog(NOTICE,
			"logistic_regression: CPU training completed, "
			"model_id=%d",
			model_id);

		/* Cleanup */
		pfree(model.weights);
		/* Note: serialized, params_jsonb, metrics_jsonb are owned by catalog now */
		/* StringInfo buffers will be cleaned up when function returns */
	}

	if (weights)
		pfree(weights);
	lr_dataset_free(&dataset);
	if (query.data)
		pfree(query.data);
	if (tbl_str)
		pfree(tbl_str);
	if (feat_str)
		pfree(feat_str);
	if (targ_str)
		pfree(targ_str);
	PG_RETURN_INT32(model_id);
}

/*
 * predict_logistic_regression
 *
 * Predicts probability for logistic regression
 */
PG_FUNCTION_INFO_V1(predict_logistic_regression);

Datum
predict_logistic_regression(PG_FUNCTION_ARGS)
{
	ArrayType *coef_array;
	Vector *features;
	int ncoef;
	float8 *coef;
	float *x;
	int dim;
	double z;
	double probability;
	int i;

	coef_array = PG_GETARG_ARRAYTYPE_P(0);
	features = PG_GETARG_VECTOR_P(1);

	/* Extract coefficients */
	if (ARR_NDIM(coef_array) != 1)
		ereport(ERROR,
			(errmsg("Coefficients must be 1-dimensional array")));

	ncoef = ARR_DIMS(coef_array)[0];
	coef = (float8 *)ARR_DATA_PTR(coef_array);

	x = features->data;
	dim = features->dim;

	if (ncoef != dim + 1)
		ereport(ERROR,
			(errmsg("Coefficient dimension mismatch: expected %d, "
				"got %d",
				dim + 1,
				ncoef)));

	/* Compute z = bias + w1*x1 + w2*x2 + ... */
	z = coef[0]; /* bias */
	for (i = 0; i < dim; i++)
		z += coef[i + 1] * x[i];

	/* Apply sigmoid */
	probability = sigmoid(z);

	PG_RETURN_FLOAT8(probability);
}

/*
 * predict_logistic_regression(model_id, features)
 *
 * Predicts using a model loaded from the catalog by model_id.
 * Supports both CPU and GPU models.
 */
PG_FUNCTION_INFO_V1(predict_logistic_regression_model_id);

Datum
predict_logistic_regression_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	Vector *features;
	LRModel *model = NULL;
	double probability;
	double z;
	int i;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("logistic_regression: model_id is "
				       "required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("logistic_regression: features vector "
				       "is required")));

	features = PG_GETARG_VECTOR_P(1);

	/* Try GPU prediction first */
	if (lr_try_gpu_predict_catalog(model_id, features, &probability))
	{
		elog(DEBUG1,
			"logistic_regression: GPU prediction succeeded, "
			"probability=%.6f",
			probability);
		PG_RETURN_FLOAT8(probability);
	} else
	{
		elog(DEBUG1,
			"logistic_regression: GPU prediction failed or not "
			"available, trying CPU");
	}

	/* Load model from catalog */
	if (!lr_load_model_from_catalog(model_id, &model))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("logistic_regression: model %d not "
				       "found",
					model_id)));

	/* Validate feature dimension */
	if (model->n_features > 0 && features->dim != model->n_features)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("logistic_regression: feature dimension "
				       "mismatch "
				       "(expected %d, got %d)",
					model->n_features,
					features->dim)));

	/* Compute z = bias + w1*x1 + w2*x2 + ... */
	z = model->bias;
	for (i = 0; i < model->n_features && i < features->dim; i++)
		z += model->weights[i] * features->data[i];

	/* Apply sigmoid */
	probability = sigmoid(z);

	/* Cleanup */
	if (model != NULL)
	{
		if (model->weights != NULL)
			pfree(model->weights);
		pfree(model);
	}

	PG_RETURN_FLOAT8(probability);
}

/*
 * evaluate_logistic_regression
 *
 * Evaluates model performance (accuracy, precision, recall, F1, AUC)
 * Returns: [accuracy, precision, recall, f1_score, log_loss]
 */
PG_FUNCTION_INFO_V1(evaluate_logistic_regression);

Datum
evaluate_logistic_regression(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *feature_col;
	text *target_col;
	ArrayType *coef_array;
	double threshold = PG_GETARG_FLOAT8(4);
	char *tbl_str;
	char *feat_str;
	char *targ_str;
	StringInfoData query;
	int ret;
	int nvec = 0;
	int ncoef;
	float8 *coef;
	int tp = 0, tn = 0, fp = 0, fn = 0;
	double log_loss = 0.0;
	double accuracy, precision, recall, f1_score;
	int i, j;
	Datum *result_datums;
	ArrayType *result_array;
	MemoryContext oldcontext;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);
	coef_array = PG_GETARG_ARRAYTYPE_P(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	/* Extract coefficients */
	if (ARR_NDIM(coef_array) != 1)
		ereport(ERROR,
			(errmsg("Coefficients must be 1-dimensional array")));

	ncoef = ARR_DIMS(coef_array)[0];
	(void)ncoef; /* Suppress unused variable warning */
	coef = (float8 *)ARR_DATA_PTR(coef_array);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("SPI_connect failed")));

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		feat_str,
		targ_str,
		tbl_str,
		feat_str,
		targ_str);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR, (errmsg("Query failed")));

	nvec = SPI_processed;

	/* Compute predictions and metrics */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum feat_datum;
		Datum targ_datum;
		bool feat_null;
		bool targ_null;
		Vector *vec;
		double y_true;
		double z;
		double probability;
		int y_pred;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		vec = DatumGetVector(feat_datum);
		y_true = DatumGetFloat8(targ_datum);

		/* Compute probability */
		z = coef[0]; /* bias */
		for (j = 0; j < vec->dim; j++)
			z += coef[j + 1] * vec->data[j];
		probability = sigmoid(z);

		/* Apply threshold */
		y_pred = (probability >= threshold) ? 1 : 0;

		/* Update confusion matrix */
		if (y_true > 0.5)
		{
			if (y_pred == 1)
				tp++;
			else
				fn++;
		} else
		{
			if (y_pred == 1)
				fp++;
			else
				tn++;
		}

		/* Update log loss */
		probability = fmax(1e-15, fmin(1.0 - 1e-15, probability));
		if (y_true > 0.5)
			log_loss -= log(probability);
		else
			log_loss -= log(1.0 - probability);
	}

	log_loss /= nvec;

	SPI_finish();

	/* Compute metrics */
	accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
	precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
	recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
	f1_score = (precision + recall > 0)
		? 2.0 * precision * recall / (precision + recall)
		: 0.0;

	/* Build result array: [accuracy, precision, recall, f1_score, log_loss] */
	MemoryContextSwitchTo(oldcontext);

	result_datums = (Datum *)palloc(sizeof(Datum) * 5);
	result_datums[0] = Float8GetDatum(accuracy);
	result_datums[1] = Float8GetDatum(precision);
	result_datums[2] = Float8GetDatum(recall);
	result_datums[3] = Float8GetDatum(f1_score);
	result_datums[4] = Float8GetDatum(log_loss);

	result_array = construct_array(result_datums,
		5,
		FLOAT8OID,
		sizeof(float8),
		FLOAT8PASSBYVAL,
		'd');

	pfree(result_datums);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(targ_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

static void
lr_dataset_init(LRDataset *dataset)
{
	if (dataset == NULL)
		return;
	memset(dataset, 0, sizeof(LRDataset));
}

static void
lr_dataset_free(LRDataset *dataset)
{
	if (dataset == NULL)
		return;
	if (dataset->features != NULL)
	{
		pfree(dataset->features);
		dataset->features = NULL;
	}
	if (dataset->labels != NULL)
	{
		pfree(dataset->labels);
		dataset->labels = NULL;
	}
	lr_dataset_init(dataset);
}

static void
lr_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_label,
	LRDataset *dataset,
	StringInfo query)
{
	int feature_dim = 0;
	int n_samples = 0;
	int i;
	int ret;
	MemoryContext oldcontext;

	if (dataset == NULL || query == NULL)
		return;

	lr_dataset_init(dataset);

	oldcontext = CurrentMemoryContext;

	/* Initialize query in caller's context before SPI_connect */
	initStringInfo(query);
	MemoryContextSwitchTo(oldcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("lr_dataset_load: SPI_connect failed")));
	appendStringInfo(query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quoted_feat,
		quoted_label,
		quoted_tbl,
		quoted_feat,
		quoted_label);

	ret = SPI_execute(query->data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		ereport(ERROR, (errmsg("lr_dataset_load: query failed")));
	}

	n_samples = SPI_processed;
	if (n_samples <= 0)
	{
		SPI_finish();
		return;
	}

	/* Get feature dimension from first row before allocating */
	if (SPI_processed > 0)
	{
		HeapTuple first_tuple = SPI_tuptable->vals[0];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum feat_datum;
		bool feat_null;
		Vector *vec;

		feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &feat_null);
		if (!feat_null)
		{
			vec = DatumGetVector(feat_datum);
			feature_dim = vec->dim;
		}
	}

	if (feature_dim <= 0)
	{
		SPI_finish();
		ereport(ERROR,
			(errmsg("lr_dataset_load: could not determine feature "
				"dimension")));
	}

	MemoryContextSwitchTo(oldcontext);
	dataset->features = (float *)palloc(
		sizeof(float) * (size_t)n_samples * (size_t)feature_dim);
	dataset->labels = (double *)palloc(sizeof(double) * (size_t)n_samples);

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum feat_datum;
		Datum targ_datum;
		bool feat_null;
		bool targ_null;
		Vector *vec;
		float *row;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		if (feat_null)
			continue;

		vec = DatumGetVector(feat_datum);
		if (vec->dim != feature_dim)
		{
			SPI_finish();
			ereport(ERROR,
				(errmsg("lr_dataset_load: inconsistent vector "
					"dimensions")));
		}

		row = dataset->features + (i * feature_dim);
		memcpy(row, vec->data, sizeof(float) * feature_dim);

		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (targ_null)
			continue;

		{
			Oid targ_type = SPI_gettypeid(tupdesc, 2);

			if (targ_type == INT2OID || targ_type == INT4OID
				|| targ_type == INT8OID)
				dataset->labels[i] =
					(double)DatumGetInt32(targ_datum);
			else
				dataset->labels[i] = DatumGetFloat8(targ_datum);
		}

		if (dataset->labels[i] != 0.0 && dataset->labels[i] != 1.0)
		{
			SPI_finish();
			ereport(ERROR,
				(errmsg("lr_dataset_load: binary target "
					"required (0 or 1)")));
		}
	}

	dataset->n_samples = n_samples;
	dataset->feature_dim = feature_dim;

	SPI_finish();
}

/* GPU model state for Logistic Regression */
typedef struct LRGpuModelState
{
	bytea *model_blob;
	Jsonb *metrics;
	int feature_dim;
	int n_samples;
} LRGpuModelState;

static void
lr_gpu_release_state(LRGpuModelState *state)
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
lr_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	LRGpuModelState *state;
	bytea *payload;
	Jsonb *metrics;
	int rc;

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

	payload = NULL;
	metrics = NULL;

	rc = ndb_gpu_lr_train(spec->feature_matrix,
		spec->label_vector,
		spec->sample_count,
		spec->feature_dim,
		spec->hyperparameters,
		&payload,
		&metrics,
		errstr);
	if (rc != 0 || payload == NULL)
	{
		if (payload != NULL)
			pfree(payload);
		if (metrics != NULL)
			pfree(metrics);
		return false;
	}

	if (model->backend_state != NULL)
	{
		lr_gpu_release_state((LRGpuModelState *)model->backend_state);
		model->backend_state = NULL;
	}

	state = (LRGpuModelState *)palloc0(sizeof(LRGpuModelState));
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;

	/* Store metrics in model state for later retrieval */
	if (metrics != NULL)
	{
		/* Copy metrics to ensure they're in the correct memory context */
		state->metrics = (Jsonb *)PG_DETOAST_DATUM_COPY(
			PointerGetDatum(metrics));
		elog(DEBUG1,
			"lr_gpu_train: stored metrics in state: %p",
			(void *)state->metrics);
	} else
	{
		state->metrics = NULL;
		elog(WARNING,
			"lr_gpu_train: metrics is NULL, cannot store in "
			"state!");
	}

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
lr_gpu_predict(const MLGpuModel *model,
	const float *input,
	int input_dim,
	float *output,
	int output_dim,
	char **errstr)
{
	const LRGpuModelState *state;
	double probability;
	int rc;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = -1.0f;
	if (model == NULL || input == NULL || output == NULL)
		return false;
	if (output_dim <= 0)
		return false;
	if (!model->gpu_ready || model->backend_state == NULL)
		return false;

	state = (const LRGpuModelState *)model->backend_state;
	if (state->model_blob == NULL)
		return false;

	rc = ndb_gpu_lr_predict(state->model_blob,
		input,
		state->feature_dim > 0 ? state->feature_dim : input_dim,
		&probability,
		errstr);
	if (rc != 0)
		return false;

	output[0] = (float)probability;

	return true;
}

static bool
lr_gpu_evaluate(const MLGpuModel *model,
	const MLGpuEvalSpec *spec,
	MLGpuMetrics *out,
	char **errstr)
{
	const LRGpuModelState *state;
	Jsonb *metrics_json;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const LRGpuModelState *)model->backend_state;

	/* Build metrics JSON from stored model metadata */
	{
		StringInfoData buf;

		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"logistic_regression\","
			"\"storage\":\"gpu\","
			"\"n_features\":%d,"
			"\"n_samples\":%d}",
			state->feature_dim > 0 ? state->feature_dim : 0,
			state->n_samples > 0 ? state->n_samples : 0);

		metrics_json = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(buf.data)));
		pfree(buf.data);
	}

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
lr_gpu_serialize(const MLGpuModel *model,
	bytea **payload_out,
	Jsonb **metadata_out,
	char **errstr)
{
	const LRGpuModelState *state;
	bytea *payload_copy;
	int payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
		return false;

	state = (const LRGpuModelState *)model->backend_state;
	if (state->model_blob == NULL)
		return false;

	payload_size = VARSIZE(state->model_blob);
	payload_copy = (bytea *)palloc(payload_size);
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		pfree(payload_copy);

	/* Return stored metrics */
	if (metadata_out != NULL && state->metrics != NULL)
	{
		/* Copy metrics to ensure they're in the correct memory context */
		*metadata_out = (Jsonb *)PG_DETOAST_DATUM_COPY(
			PointerGetDatum(state->metrics));
		elog(DEBUG1,
			"lr_gpu_serialize: returning metrics: %p",
			(void *)*metadata_out);
	} else if (metadata_out != NULL)
	{
		*metadata_out = NULL;
		elog(WARNING,
			"lr_gpu_serialize: state->metrics is NULL, cannot "
			"return metrics!");
	}

	return true;
}

static bool
lr_gpu_deserialize(MLGpuModel *model,
	const bytea *payload,
	const Jsonb *metadata,
	char **errstr)
{
	LRGpuModelState *state;
	bytea *payload_copy;
	int payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

	payload_size = VARSIZE(payload);
	payload_copy = (bytea *)palloc(payload_size);
	memcpy(payload_copy, payload, payload_size);

	state = (LRGpuModelState *)palloc0(sizeof(LRGpuModelState));
	state->model_blob = payload_copy;
	state->feature_dim = -1;
	state->n_samples = -1;

	if (model->backend_state != NULL)
		lr_gpu_release_state((LRGpuModelState *)model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static void
lr_gpu_destroy(MLGpuModel *model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		lr_gpu_release_state((LRGpuModelState *)model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps lr_gpu_model_ops = {
	.algorithm = "logistic_regression",
	.train = lr_gpu_train,
	.predict = lr_gpu_predict,
	.evaluate = lr_gpu_evaluate,
	.serialize = lr_gpu_serialize,
	.deserialize = lr_gpu_deserialize,
	.destroy = lr_gpu_destroy,
};

static bytea *
lr_model_serialize(const LRModel *model)
{
	StringInfoData buf;
	int i;

	if (model == NULL)
		return NULL;

	pq_begintypsend(&buf);

	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendfloat8(&buf, model->bias);
	pq_sendfloat8(&buf, model->learning_rate);
	pq_sendfloat8(&buf, model->lambda);
	pq_sendint32(&buf, model->max_iters);
	pq_sendfloat8(&buf, model->final_loss);
	pq_sendfloat8(&buf, model->accuracy);

	if (model->weights != NULL && model->n_features > 0)
	{
		for (i = 0; i < model->n_features; i++)
			pq_sendfloat8(&buf, model->weights[i]);
	}

	return pq_endtypsend(&buf);
}

static LRModel *
lr_model_deserialize(const bytea *data)
{
	LRModel *model;
	StringInfoData buf;
	int i;

	if (data == NULL)
		return NULL;

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.cursor = 0;

	model = (LRModel *)palloc0(sizeof(LRModel));

	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->bias = pq_getmsgfloat8(&buf);
	model->learning_rate = pq_getmsgfloat8(&buf);
	model->lambda = pq_getmsgfloat8(&buf);
	model->max_iters = pq_getmsgint(&buf, 4);
	model->final_loss = pq_getmsgfloat8(&buf);
	model->accuracy = pq_getmsgfloat8(&buf);

	if (model->n_features > 0)
	{
		model->weights =
			(double *)palloc(sizeof(double) * model->n_features);
		for (i = 0; i < model->n_features; i++)
			model->weights[i] = pq_getmsgfloat8(&buf);
	}

	return model;
}

/*
 * lr_metadata_is_gpu
 *
 * Checks if a model's metadata indicates it's a GPU-backed model.
 */
static bool
lr_metadata_is_gpu(Jsonb *metadata)
{
	char *meta_txt;
	bool is_gpu = false;

	if (metadata == NULL)
		return false;

	PG_TRY();
	{
		meta_txt = DatumGetCString(DirectFunctionCall1(
			jsonb_out, JsonbPGetDatum(metadata)));
		if (meta_txt != NULL)
		{
			if (strstr(meta_txt, "\"storage\":\"gpu\"") != NULL
				|| strstr(meta_txt, "\"storage\": \"gpu\"")
					!= NULL)
				is_gpu = true;
			pfree(meta_txt);
		}
	}
	PG_CATCH();
	{
		/* Invalid JSONB, assume CPU */
		is_gpu = false;
	}
	PG_END_TRY();

	return is_gpu;
}

/*
 * lr_try_gpu_predict_catalog
 *
 * Attempts GPU prediction for a model loaded from the catalog.
 * Returns true if GPU prediction succeeded, false otherwise.
 */
static bool
lr_try_gpu_predict_catalog(int32 model_id,
	const Vector *feature_vec,
	double *result_out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	char *gpu_err = NULL;
	double probability = 0.0;
	bool success = false;

	elog(DEBUG1,
		"lr_try_gpu_predict_catalog: entry, model_id=%d, "
		"feature_dim=%d",
		model_id,
		feature_vec ? feature_vec->dim : -1);

	if (!neurondb_gpu_is_available())
	{
		elog(DEBUG1, "lr_try_gpu_predict_catalog: GPU not available");
		return false;
	}
	if (feature_vec == NULL)
		return false;
	if (feature_vec->dim <= 0)
		return false;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
	{
		elog(DEBUG1,
			"lr_try_gpu_predict_catalog: failed to fetch model "
			"payload for model %d",
			model_id);
		return false;
	}

	if (payload == NULL)
	{
		elog(DEBUG1,
			"lr_try_gpu_predict_catalog: payload is NULL for model "
			"%d",
			model_id);
		goto cleanup;
	}

	if (!lr_metadata_is_gpu(metrics))
	{
		elog(DEBUG1,
			"lr_try_gpu_predict_catalog: model %d is not a GPU "
			"model",
			model_id);
		goto cleanup;
	}

	elog(DEBUG1,
		"lr_try_gpu_predict_catalog: model %d is GPU model, payload "
		"size=%d",
		model_id,
		VARSIZE(payload) - VARHDRSZ);

	if (ndb_gpu_lr_predict(payload,
		    feature_vec->data,
		    feature_vec->dim,
		    &probability,
		    &gpu_err)
		== 0)
	{
		if (result_out != NULL)
			*result_out = probability;
		elog(DEBUG1,
			"logistic_regression: GPU prediction used for model %d "
			"probability=%.6f",
			model_id,
			probability);
		success = true;
	} else if (gpu_err != NULL)
	{
		elog(WARNING,
			"logistic_regression: GPU prediction failed for model "
			"%d "
			"(%s)",
			model_id,
			gpu_err);
	}

cleanup:
	if (payload != NULL)
		pfree(payload);
	if (metrics != NULL)
		pfree(metrics);
	if (gpu_err != NULL)
		pfree(gpu_err);

	return success;
}

/*
 * lr_load_model_from_catalog
 *
 * Loads a Logistic Regression model from the catalog by model_id.
 * Returns true on success, false on failure.
 */
static bool
lr_load_model_from_catalog(int32 model_id, LRModel **out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	LRModel *decoded;

	if (out == NULL)
		return false;

	*out = NULL;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
	{
		if (metrics != NULL)
			pfree(metrics);
		return false;
	}

	/* Skip GPU models - they should be handled by GPU prediction */
	if (lr_metadata_is_gpu(metrics))
	{
		pfree(payload);
		if (metrics != NULL)
			pfree(metrics);
		return false;
	}

	decoded = lr_model_deserialize(payload);

	pfree(payload);
	if (metrics != NULL)
		pfree(metrics);

	if (decoded == NULL)
		return false;

	*out = decoded;
	return true;
}

void
neurondb_gpu_register_lr_model(void)
{
	static bool registered = false;

	if (registered)
		return;

	ndb_gpu_register_model_ops(&lr_gpu_model_ops);
	registered = true;
}
