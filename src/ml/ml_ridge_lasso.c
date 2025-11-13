/*-------------------------------------------------------------------------
 *
 * ml_ridge_lasso.c
 *    Ridge and Lasso Regression implementations
 *
 * Implements regularized linear regression with L2 (Ridge) and L1 (Lasso)
 * penalties to prevent overfitting.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_ridge_lasso.c
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

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_ridge_regression_internal.h"
#include "ml_lasso_regression_internal.h"
#include "ml_catalog.h"
#include "neurondb_gpu_bridge.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "neurondb_gpu_backend.h"
#include "ml_gpu_registry.h"
#include "ml_gpu_ridge_regression.h"
#include "ml_gpu_lasso_regression.h"

#include <math.h>
#include <float.h>

/* Ridge Regression dataset structure */
typedef struct RidgeDataset
{
	float *features;
	double *targets;
	int n_samples;
	int feature_dim;
} RidgeDataset;

/* Lasso Regression dataset structure (same as Ridge) */
typedef RidgeDataset LassoDataset;

/* Forward declarations for Ridge Regression */
static void ridge_dataset_init(RidgeDataset *dataset);
static void ridge_dataset_free(RidgeDataset *dataset);
static void ridge_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	RidgeDataset *dataset);
static bytea *ridge_model_serialize(const RidgeModel *model);
static RidgeModel *ridge_model_deserialize(const bytea *data);
static bool ridge_metadata_is_gpu(Jsonb *metadata);
static bool ridge_try_gpu_predict_catalog(int32 model_id,
	const Vector *feature_vec,
	double *result_out);
static bool ridge_load_model_from_catalog(int32 model_id, RidgeModel **out);

/* Forward declarations for Lasso Regression */
static void lasso_dataset_init(LassoDataset *dataset);
static void lasso_dataset_free(LassoDataset *dataset);
static void lasso_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	LassoDataset *dataset);
static bytea *lasso_model_serialize(const LassoModel *model);
static LassoModel *lasso_model_deserialize(const bytea *data);
static bool lasso_metadata_is_gpu(Jsonb *metadata);
static bool lasso_try_gpu_predict_catalog(int32 model_id,
	const Vector *feature_vec,
	double *result_out);
static bool lasso_load_model_from_catalog(int32 model_id, LassoModel **out);

/*
 * Matrix inversion using Gauss-Jordan elimination
 * Returns false if matrix is singular
 */
static bool
matrix_invert(double **matrix, int n, double **result)
{
	double **augmented;
	int i, j, k;
	double pivot, factor;

	/* Create augmented matrix [A | I] */
	augmented = (double **)palloc(sizeof(double *) * n);
	for (i = 0; i < n; i++)
	{
		augmented[i] = (double *)palloc(sizeof(double) * 2 * n);
		for (j = 0; j < n; j++)
		{
			augmented[i][j] = matrix[i][j];
			augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
		}
	}

	/* Gauss-Jordan elimination */
	for (i = 0; i < n; i++)
	{
		/* Find pivot */
		pivot = augmented[i][i];
		if (fabs(pivot) < 1e-10)
		{
			/* Try to swap with a row below */
			bool found = false;
			for (k = i + 1; k < n; k++)
			{
				if (fabs(augmented[k][i]) > 1e-10)
				{
					/* Swap rows */
					double *temp = augmented[i];
					augmented[i] = augmented[k];
					augmented[k] = temp;
					pivot = augmented[i][i];
					found = true;
					break;
				}
			}
			if (!found)
			{
				/* Matrix is singular */
				for (j = 0; j < n; j++)
					pfree(augmented[j]);
				pfree(augmented);
				return false;
			}
		}

		/* Normalize pivot row */
		for (j = 0; j < 2 * n; j++)
			augmented[i][j] /= pivot;

		/* Eliminate column */
		for (k = 0; k < n; k++)
		{
			if (k != i)
			{
				factor = augmented[k][i];
				for (j = 0; j < 2 * n; j++)
					augmented[k][j] -=
						factor * augmented[i][j];
			}
		}
	}

	/* Extract result matrix from augmented matrix */
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			result[i][j] = augmented[i][j + n];

	/* Cleanup */
	for (i = 0; i < n; i++)
		pfree(augmented[i]);
	pfree(augmented);

	return true;
}

/*
 * ridge_dataset_init
 */
static void
ridge_dataset_init(RidgeDataset *dataset)
{
	if (dataset == NULL)
		return;
	memset(dataset, 0, sizeof(RidgeDataset));
}

/*
 * ridge_dataset_free
 */
static void
ridge_dataset_free(RidgeDataset *dataset)
{
	if (dataset == NULL)
		return;
	if (dataset->features != NULL)
		pfree(dataset->features);
	if (dataset->targets != NULL)
		pfree(dataset->targets);
	ridge_dataset_init(dataset);
}

/*
 * ridge_dataset_load
 */
static void
ridge_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	RidgeDataset *dataset)
{
	StringInfoData query;
	MemoryContext oldcontext;
	int ret;
	int n_samples = 0;
	int feature_dim = 0;
	int i;

	if (dataset == NULL)
		ereport(ERROR, (errmsg("ridge_dataset_load: dataset is NULL")));

	oldcontext = CurrentMemoryContext;

	/* Initialize query in caller's context before SPI_connect */
	initStringInfo(&query);
	MemoryContextSwitchTo(oldcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errmsg("ridge_dataset_load: SPI_connect failed")));
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quoted_feat,
		quoted_target,
		quoted_tbl,
		quoted_feat,
		quoted_target);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		ereport(ERROR, (errmsg("ridge_dataset_load: query failed")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		SPI_finish();
		ereport(ERROR,
			(errmsg("ridge_dataset_load: need at least 10 samples, "
				"got %d",
				n_samples)));
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
			(errmsg("ridge_dataset_load: could not determine "
				"feature dimension")));
	}

	MemoryContextSwitchTo(oldcontext);
	dataset->features = (float *)palloc(
		sizeof(float) * (size_t)n_samples * (size_t)feature_dim);
	dataset->targets = (double *)palloc(sizeof(double) * (size_t)n_samples);

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
				(errmsg("ridge_dataset_load: inconsistent "
					"vector dimensions")));
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
				dataset->targets[i] =
					(double)DatumGetInt32(targ_datum);
			else
				dataset->targets[i] =
					DatumGetFloat8(targ_datum);
		}
	}

	dataset->n_samples = n_samples;
	dataset->feature_dim = feature_dim;

	SPI_finish();
}

/*
 * ridge_model_serialize
 */
static bytea *
ridge_model_serialize(const RidgeModel *model)
{
	StringInfoData buf;
	int i;

	if (model == NULL)
		return NULL;

	/* Validate model before serialization */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("ridge_model_serialize: invalid "
				       "n_features %d (corrupted model?)",
					model->n_features)));
	}

	pq_begintypsend(&buf);

	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendfloat8(&buf, model->intercept);
	pq_sendfloat8(&buf, model->lambda);
	pq_sendfloat8(&buf, model->r_squared);
	pq_sendfloat8(&buf, model->mse);
	pq_sendfloat8(&buf, model->mae);

	if (model->coefficients != NULL && model->n_features > 0)
	{
		for (i = 0; i < model->n_features; i++)
			pq_sendfloat8(&buf, model->coefficients[i]);
	}

	return pq_endtypsend(&buf);
}

/*
 * ridge_model_deserialize
 */
static RidgeModel *
ridge_model_deserialize(const bytea *data)
{
	RidgeModel *model;
	StringInfoData buf;
	int i;

	if (data == NULL)
		return NULL;

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	model = (RidgeModel *)palloc0(sizeof(RidgeModel));
	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->intercept = pq_getmsgfloat8(&buf);
	model->lambda = pq_getmsgfloat8(&buf);
	model->r_squared = pq_getmsgfloat8(&buf);
	model->mse = pq_getmsgfloat8(&buf);
	model->mae = pq_getmsgfloat8(&buf);

	/* Validate deserialized values */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		pfree(model);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("ridge: invalid n_features %d in "
				       "deserialized model (corrupted data?)",
					model->n_features)));
	}
	if (model->n_samples < 0 || model->n_samples > 100000000)
	{
		pfree(model);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("ridge: invalid n_samples %d in "
				       "deserialized model (corrupted data?)",
					model->n_samples)));
	}

	if (model->n_features > 0)
	{
		model->coefficients =
			(double *)palloc(sizeof(double) * model->n_features);
		for (i = 0; i < model->n_features; i++)
			model->coefficients[i] = pq_getmsgfloat8(&buf);
	}

	return model;
}

/*
 * ridge_metadata_is_gpu
 */
static bool
ridge_metadata_is_gpu(Jsonb *metadata)
{
	char *meta_text = NULL;
	bool is_gpu = false;

	if (metadata == NULL)
		return false;

	PG_TRY();
	{
		meta_text = DatumGetCString(DirectFunctionCall1(
			jsonb_out, JsonbPGetDatum(metadata)));
		if (strstr(meta_text, "\"storage\":\"gpu\"") != NULL)
			is_gpu = true;
		pfree(meta_text);
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
 * ridge_try_gpu_predict_catalog
 */
static bool
ridge_try_gpu_predict_catalog(int32 model_id,
	const Vector *feature_vec,
	double *result_out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	char *gpu_err = NULL;
	double prediction = 0.0;
	bool success = false;

	if (!neurondb_gpu_is_available())
		return false;
	if (feature_vec == NULL)
		return false;
	if (feature_vec->dim <= 0)
		return false;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
		goto cleanup;

	if (!ridge_metadata_is_gpu(metrics))
		goto cleanup;

	if (ndb_gpu_ridge_predict(payload,
		    feature_vec->data,
		    feature_vec->dim,
		    &prediction,
		    &gpu_err)
		== 0)
	{
		if (result_out != NULL)
			*result_out = prediction;
		success = true;
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
 * ridge_load_model_from_catalog
 */
static bool
ridge_load_model_from_catalog(int32 model_id, RidgeModel **out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;

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
	if (ridge_metadata_is_gpu(metrics))
	{
		pfree(payload);
		if (metrics != NULL)
			pfree(metrics);
		return false;
	}

	*out = ridge_model_deserialize(payload);

	pfree(payload);
	if (metrics != NULL)
		pfree(metrics);

	return (*out != NULL);
}

/*
 * lasso_dataset_init
 */
static void
lasso_dataset_init(LassoDataset *dataset)
{
	if (dataset == NULL)
		return;
	memset(dataset, 0, sizeof(LassoDataset));
}

/*
 * lasso_dataset_free
 */
static void
lasso_dataset_free(LassoDataset *dataset)
{
	if (dataset == NULL)
		return;
	if (dataset->features != NULL)
		pfree(dataset->features);
	if (dataset->targets != NULL)
		pfree(dataset->targets);
	lasso_dataset_init(dataset);
}

/*
 * lasso_dataset_load
 * Reuses ridge_dataset_load since they have the same structure
 */
static void
lasso_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	LassoDataset *dataset)
{
	ridge_dataset_load(quoted_tbl,
		quoted_feat,
		quoted_target,
		(RidgeDataset *)dataset);
}

/*
 * lasso_model_serialize
 */
static bytea *
lasso_model_serialize(const LassoModel *model)
{
	StringInfoData buf;
	int i;

	if (model == NULL)
		return NULL;

	/* Validate model before serialization */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("lasso_model_serialize: invalid "
				       "n_features %d (corrupted model?)",
					model->n_features)));
	}

	pq_begintypsend(&buf);

	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendfloat8(&buf, model->intercept);
	pq_sendfloat8(&buf, model->lambda);
	pq_sendint32(&buf, model->max_iters);
	pq_sendfloat8(&buf, model->r_squared);
	pq_sendfloat8(&buf, model->mse);
	pq_sendfloat8(&buf, model->mae);

	if (model->coefficients != NULL && model->n_features > 0)
	{
		for (i = 0; i < model->n_features; i++)
			pq_sendfloat8(&buf, model->coefficients[i]);
	}

	return pq_endtypsend(&buf);
}

/*
 * lasso_model_deserialize
 */
static LassoModel *
lasso_model_deserialize(const bytea *data)
{
	LassoModel *model;
	StringInfoData buf;
	int i;

	if (data == NULL)
		return NULL;

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	model = (LassoModel *)palloc0(sizeof(LassoModel));
	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->intercept = pq_getmsgfloat8(&buf);
	model->lambda = pq_getmsgfloat8(&buf);
	model->max_iters = pq_getmsgint(&buf, 4);
	model->r_squared = pq_getmsgfloat8(&buf);
	model->mse = pq_getmsgfloat8(&buf);
	model->mae = pq_getmsgfloat8(&buf);

	/* Validate deserialized values */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		pfree(model);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("lasso: invalid n_features %d in "
				       "deserialized model (corrupted data?)",
					model->n_features)));
	}
	if (model->n_samples < 0 || model->n_samples > 100000000)
	{
		pfree(model);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("lasso: invalid n_samples %d in "
				       "deserialized model (corrupted data?)",
					model->n_samples)));
	}

	if (model->n_features > 0)
	{
		model->coefficients =
			(double *)palloc(sizeof(double) * model->n_features);
		for (i = 0; i < model->n_features; i++)
			model->coefficients[i] = pq_getmsgfloat8(&buf);
	}

	return model;
}

/*
 * lasso_metadata_is_gpu
 */
static bool
lasso_metadata_is_gpu(Jsonb *metadata)
{
	char *meta_text = NULL;
	bool is_gpu = false;

	if (metadata == NULL)
		return false;

	PG_TRY();
	{
		meta_text = DatumGetCString(DirectFunctionCall1(
			jsonb_out, JsonbPGetDatum(metadata)));
		if (strstr(meta_text, "\"storage\":\"gpu\"") != NULL)
			is_gpu = true;
		pfree(meta_text);
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
 * lasso_try_gpu_predict_catalog
 */
static bool
lasso_try_gpu_predict_catalog(int32 model_id,
	const Vector *feature_vec,
	double *result_out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	char *gpu_err = NULL;
	double prediction = 0.0;
	bool success = false;

	if (!neurondb_gpu_is_available())
		return false;
	if (feature_vec == NULL)
		return false;
	if (feature_vec->dim <= 0)
		return false;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
		goto cleanup;

	if (!lasso_metadata_is_gpu(metrics))
		goto cleanup;

	if (ndb_gpu_lasso_predict(payload,
		    feature_vec->data,
		    feature_vec->dim,
		    &prediction,
		    &gpu_err)
		== 0)
	{
		if (result_out != NULL)
			*result_out = prediction;
		success = true;
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
 * lasso_load_model_from_catalog
 */
static bool
lasso_load_model_from_catalog(int32 model_id, LassoModel **out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;

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
	if (lasso_metadata_is_gpu(metrics))
	{
		pfree(payload);
		if (metrics != NULL)
			pfree(metrics);
		return false;
	}

	*out = lasso_model_deserialize(payload);

	pfree(payload);
	if (metrics != NULL)
		pfree(metrics);

	return (*out != NULL);
}

/*
 * Soft thresholding operator for Lasso (coordinate descent)
 */
static double
soft_threshold(double x, double lambda)
{
	if (x > lambda)
		return x - lambda;
	else if (x < -lambda)
		return x + lambda;
	else
		return 0.0;
}

/*
 * train_ridge_regression
 *
 * Trains Ridge Regression (L2 regularization)
 * Uses closed-form solution: β = (X'X + λI)^(-1)X'y
 * Returns model_id
 */
PG_FUNCTION_INFO_V1(train_ridge_regression);

Datum
train_ridge_regression(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *feature_col;
	text *target_col;
	double lambda = PG_GETARG_FLOAT8(3); /* Regularization parameter */
	char *tbl_str;
	char *feat_str;
	char *targ_str;
	int nvec = 0;
	int dim = 0;
	RidgeDataset dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_target;
	MLGpuTrainResult gpu_result;
	char *gpu_err = NULL;
	Jsonb *gpu_hyperparams = NULL;
	StringInfoData hyperbuf;
	int32 model_id = 0;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);

	if (lambda < 0.0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("ridge: lambda must be non-negative, "
				       "got %.6f",
					lambda)));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	ridge_dataset_init(&dataset);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_target = quote_identifier(targ_str);

	ridge_dataset_load(quoted_tbl, quoted_feat, quoted_target, &dataset);

	nvec = dataset.n_samples;
	dim = dataset.feature_dim;

	if (nvec < 10)
	{
		ridge_dataset_free(&dataset);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
			(errmsg("Need at least 10 samples for Ridge "
				"regression, have %d",
				nvec)));
	}

	/* Try GPU training first */
	if (neurondb_gpu_is_available() && nvec > 0 && dim > 0)
	{
		initStringInfo(&hyperbuf);
		appendStringInfo(&hyperbuf, "{\"lambda\":%.6f}", lambda);
		gpu_hyperparams = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(hyperbuf.data)));

		if (ndb_gpu_try_train_model("ridge",
			    NULL,
			    NULL,
			    tbl_str,
			    targ_str,
			    NULL,
			    0,
			    gpu_hyperparams,
			    dataset.features,
			    dataset.targets,
			    nvec,
			    dim,
			    0,
			    &gpu_result,
			    &gpu_err)
			&& gpu_result.spec.model_data != NULL)
		{
			MLCatalogModelSpec spec;

			elog(DEBUG1, "ridge: GPU training succeeded");
			spec = gpu_result.spec;

			if (spec.training_table == NULL)
				spec.training_table = tbl_str;
			if (spec.training_column == NULL)
				spec.training_column = targ_str;
			if (spec.parameters == NULL)
			{
				spec.parameters = gpu_hyperparams;
				gpu_hyperparams = NULL;
			}

			spec.algorithm = "ridge";
			spec.model_type = "regression";

			model_id = ml_catalog_register_model(&spec);

			if (gpu_err != NULL)
				pfree(gpu_err);
			if (gpu_hyperparams != NULL)
				pfree(gpu_hyperparams);
			ndb_gpu_free_train_result(&gpu_result);
			ridge_dataset_free(&dataset);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);

			PG_RETURN_INT32(model_id);
		} else
		{
			elog(DEBUG1,
				"ridge: GPU training unavailable, using CPU");
			if (gpu_err != NULL)
				pfree(gpu_err);
			if (gpu_hyperparams != NULL)
				pfree(gpu_hyperparams);
			ndb_gpu_free_train_result(&gpu_result);
		}
	}

	/* CPU training path */
	{
		double **XtX = NULL;
		double **XtX_inv = NULL;
		double *Xty = NULL;
		double *beta = NULL;
		int i, j, k;
		int dim_with_intercept;
		RidgeModel *model;
		bytea *model_blob;
		Jsonb *metrics_json;
		StringInfoData metricsbuf;

		/* Allocate matrices for normal equations: β = (X'X + λI)^(-1)X'y */
		dim_with_intercept = dim + 1; /* Add 1 for intercept */
		XtX = (double **)palloc(sizeof(double *) * dim_with_intercept);
		XtX_inv = (double **)palloc(
			sizeof(double *) * dim_with_intercept);
		for (i = 0; i < dim_with_intercept; i++)
		{
			XtX[i] = (double *)palloc0(
				sizeof(double) * dim_with_intercept);
			XtX_inv[i] = (double *)palloc(
				sizeof(double) * dim_with_intercept);
		}
		Xty = (double *)palloc0(sizeof(double) * dim_with_intercept);
		beta = (double *)palloc(sizeof(double) * dim_with_intercept);

		/* Compute X'X and X'y using dataset */
		for (i = 0; i < nvec; i++)
		{
			/* Add intercept term (1.0) */
			double *xi = (double *)palloc(
				sizeof(double) * dim_with_intercept);
			float *row = dataset.features + (i * dim);

			xi[0] = 1.0;
			for (k = 1; k < dim_with_intercept; k++)
				xi[k] = row[k - 1];

			/* X'X accumulation */
			for (j = 0; j < dim_with_intercept; j++)
			{
				for (k = 0; k < dim_with_intercept; k++)
					XtX[j][k] += xi[j] * xi[k];

				/* X'y accumulation */
				Xty[j] += xi[j] * dataset.targets[i];
			}

			pfree(xi);
		}

		/* Add Ridge penalty (λI) to diagonal (excluding intercept) */
		for (i = 1; i < dim_with_intercept; i++)
		{
			XtX[i][i] += lambda;
		}

		/* Invert X'X + λI */
		if (!matrix_invert(XtX, dim_with_intercept, XtX_inv))
		{
			for (i = 0; i < dim_with_intercept; i++)
			{
				pfree(XtX[i]);
				pfree(XtX_inv[i]);
			}
			pfree(XtX);
			pfree(XtX_inv);
			pfree(Xty);
			pfree(beta);
			ridge_dataset_free(&dataset);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("ridge: matrix is singular, "
					       "cannot compute Ridge "
					       "regression"),
					errhint("Try increasing lambda or "
						"removing correlated "
						"features")));
		}

		/* Compute β = (X'X + λI)^(-1)X'y */
		for (i = 0; i < dim_with_intercept; i++)
		{
			beta[i] = 0.0;
			for (j = 0; j < dim_with_intercept; j++)
				beta[i] += XtX_inv[i][j] * Xty[j];
		}

		/* Build RidgeModel */
		model = (RidgeModel *)palloc0(sizeof(RidgeModel));
		model->n_features = dim;
		model->n_samples = nvec;
		model->intercept = beta[0];
		model->lambda = lambda;
		model->coefficients = (double *)palloc(sizeof(double) * dim);
		for (i = 0; i < dim; i++)
			model->coefficients[i] = beta[i + 1];

		/* Compute metrics (R², MSE, MAE) */
		{
			double y_mean = 0.0;
			double ss_tot = 0.0;
			double ss_res = 0.0;
			double mse = 0.0;
			double mae = 0.0;

			for (i = 0; i < nvec; i++)
				y_mean += dataset.targets[i];
			y_mean /= nvec;

			for (i = 0; i < nvec; i++)
			{
				float *row = dataset.features + (i * dim);
				double y_pred = model->intercept;
				double error;
				int feat_idx;

				for (feat_idx = 0; feat_idx < dim; feat_idx++)
					y_pred += model->coefficients[feat_idx]
						* row[feat_idx];

				error = dataset.targets[i] - y_pred;
				mse += error * error;
				mae += fabs(error);
				ss_res += error * error;
				ss_tot += (dataset.targets[i] - y_mean)
					* (dataset.targets[i] - y_mean);
			}

			mse /= nvec;
			mae /= nvec;
			model->r_squared = (ss_tot > 0.0)
				? (1.0 - (ss_res / ss_tot))
				: 0.0;
			model->mse = mse;
			model->mae = mae;
		}

		/* Validate model before serialization */
		if (model->n_features <= 0 || model->n_features > 10000)
		{
			if (model->coefficients != NULL)
				pfree(model->coefficients);
			pfree(model);
			for (i = 0; i < dim_with_intercept; i++)
			{
				pfree(XtX[i]);
				pfree(XtX_inv[i]);
			}
			pfree(XtX);
			pfree(XtX_inv);
			pfree(Xty);
			pfree(beta);
			ridge_dataset_free(&dataset);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("ridge: model.n_features is "
					       "invalid (%d) before "
					       "serialization",
						model->n_features)));
		}

		elog(DEBUG1,
			"ridge: serializing model with n_features=%d, "
			"n_samples=%d, lambda=%.6f",
			model->n_features,
			model->n_samples,
			model->lambda);

		/* Serialize model */
		model_blob = ridge_model_serialize(model);

		/* Build metrics JSON */
		initStringInfo(&metricsbuf);
		appendStringInfo(&metricsbuf,
			"{\"algorithm\":\"ridge\","
			"\"storage\":\"cpu\","
			"\"n_features\":%d,"
			"\"n_samples\":%d,"
			"\"lambda\":%.6f,"
			"\"r_squared\":%.6f,"
			"\"mse\":%.6f,"
			"\"mae\":%.6f}",
			model->n_features,
			model->n_samples,
			model->lambda,
			model->r_squared,
			model->mse,
			model->mae);

		metrics_json = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(metricsbuf.data)));

		/* Register in catalog */
		{
			MLCatalogModelSpec spec;

			memset(&spec, 0, sizeof(MLCatalogModelSpec));
			spec.algorithm = "ridge";
			spec.model_type = "regression";
			spec.training_table = tbl_str;
			spec.training_column = targ_str;
			spec.model_data = model_blob;
			spec.metrics = metrics_json;

			/* Build hyperparameters JSON */
			{
				StringInfoData hyperbuf_cpu;
				initStringInfo(&hyperbuf_cpu);
				appendStringInfo(&hyperbuf_cpu,
					"{\"lambda\":%.6f}",
					lambda);
				spec.parameters = DatumGetJsonbP(
					DirectFunctionCall1(jsonb_in,
						CStringGetDatum(
							hyperbuf_cpu.data)));
			}

			model_id = ml_catalog_register_model(&spec);
		}

		/* Cleanup */
		for (i = 0; i < dim_with_intercept; i++)
		{
			pfree(XtX[i]);
			pfree(XtX_inv[i]);
		}
		pfree(XtX);
		pfree(XtX_inv);
		pfree(Xty);
		pfree(beta);
		if (model->coefficients != NULL)
			pfree(model->coefficients);
		pfree(model);
		ridge_dataset_free(&dataset);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);

		PG_RETURN_INT32(model_id);
	}
}

/*
 * predict_ridge_regression_model_id
 *
 * Makes predictions using trained Ridge Regression model from catalog
 */
PG_FUNCTION_INFO_V1(predict_ridge_regression_model_id);

Datum
predict_ridge_regression_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	Vector *features;
	RidgeModel *model = NULL;
	double prediction;
	int i;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("ridge: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("ridge: features vector is required")));

	features = PG_GETARG_VECTOR_P(1);

	/* Try GPU prediction first */
	if (ridge_try_gpu_predict_catalog(model_id, features, &prediction))
	{
		elog(DEBUG1,
			"ridge: GPU prediction succeeded, prediction=%.6f",
			prediction);
		PG_RETURN_FLOAT8(prediction);
	} else
	{
		elog(DEBUG1,
			"ridge: GPU prediction failed or not available, trying "
			"CPU");
	}

	/* Load model from catalog */
	if (!ridge_load_model_from_catalog(model_id, &model))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("ridge: model %d not found", model_id)));

	/* Validate feature dimension */
	if (model->n_features > 0 && features->dim != model->n_features)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("ridge: feature dimension mismatch "
				       "(expected %d, got %d)",
					model->n_features,
					features->dim)));

	/* Compute prediction: y = intercept + coef1*x1 + coef2*x2 + ... */
	prediction = model->intercept;
	for (i = 0; i < model->n_features && i < features->dim; i++)
		prediction += model->coefficients[i] * features->data[i];

	/* Cleanup */
	if (model != NULL)
	{
		if (model->coefficients != NULL)
			pfree(model->coefficients);
		pfree(model);
	}

	PG_RETURN_FLOAT8(prediction);
}

/*
 * train_lasso_regression
 *
 * Trains Lasso Regression (L1 regularization)
 * Uses coordinate descent algorithm
 * Returns model_id
 */
PG_FUNCTION_INFO_V1(train_lasso_regression);

Datum
train_lasso_regression(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *feature_col;
	text *target_col;
	double lambda = PG_GETARG_FLOAT8(3); /* Regularization parameter */
	int max_iters = PG_NARGS() > 4 ? PG_GETARG_INT32(4) : 1000;
	char *tbl_str;
	char *feat_str;
	char *targ_str;
	int nvec = 0;
	int dim = 0;
	LassoDataset dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_target;
	MLGpuTrainResult gpu_result;
	char *gpu_err = NULL;
	Jsonb *gpu_hyperparams = NULL;
	StringInfoData hyperbuf;
	int32 model_id = 0;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);

	if (lambda < 0.0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("lasso: lambda must be non-negative, "
				       "got %.6f",
					lambda)));
	if (max_iters <= 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("lasso: max_iters must be positive, got "
				       "%d",
					max_iters)));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	lasso_dataset_init(&dataset);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_target = quote_identifier(targ_str);

	lasso_dataset_load(quoted_tbl, quoted_feat, quoted_target, &dataset);

	nvec = dataset.n_samples;
	dim = dataset.feature_dim;

	if (nvec < 10)
	{
		lasso_dataset_free(&dataset);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
			(errmsg("Need at least 10 samples for Lasso "
				"regression, have %d",
				nvec)));
	}

	/* Try GPU training first */
	if (neurondb_gpu_is_available() && nvec > 0 && dim > 0)
	{
		initStringInfo(&hyperbuf);
		appendStringInfo(&hyperbuf,
			"{\"lambda\":%.6f,\"max_iters\":%d}",
			lambda,
			max_iters);
		gpu_hyperparams = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(hyperbuf.data)));

		if (ndb_gpu_try_train_model("lasso",
			    NULL,
			    NULL,
			    tbl_str,
			    targ_str,
			    NULL,
			    0,
			    gpu_hyperparams,
			    dataset.features,
			    dataset.targets,
			    nvec,
			    dim,
			    0,
			    &gpu_result,
			    &gpu_err)
			&& gpu_result.spec.model_data != NULL)
		{
			MLCatalogModelSpec spec;

			elog(DEBUG1, "lasso: GPU training succeeded");
			spec = gpu_result.spec;

			if (spec.training_table == NULL)
				spec.training_table = tbl_str;
			if (spec.training_column == NULL)
				spec.training_column = targ_str;
			if (spec.parameters == NULL)
			{
				spec.parameters = gpu_hyperparams;
				gpu_hyperparams = NULL;
			}

			spec.algorithm = "lasso";
			spec.model_type = "regression";

			model_id = ml_catalog_register_model(&spec);

			if (gpu_err != NULL)
				pfree(gpu_err);
			if (gpu_hyperparams != NULL)
				pfree(gpu_hyperparams);
			ndb_gpu_free_train_result(&gpu_result);
			lasso_dataset_free(&dataset);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);

			PG_RETURN_INT32(model_id);
		} else
		{
			elog(DEBUG1,
				"lasso: GPU training unavailable, using CPU");
			if (gpu_err != NULL)
				pfree(gpu_err);
			if (gpu_hyperparams != NULL)
				pfree(gpu_hyperparams);
			ndb_gpu_free_train_result(&gpu_result);
		}
	}

	/* CPU training path - Coordinate Descent */
	{
		double *weights = NULL;
		double *weights_old = NULL;
		double *residuals = NULL;
		double y_mean = 0.0;
		int iter, i, j;
		bool converged = false;
		LassoModel *model;
		bytea *model_blob;
		Jsonb *metrics_json;
		StringInfoData metricsbuf;

		/* Compute mean of targets */
		for (i = 0; i < nvec; i++)
			y_mean += dataset.targets[i];
		y_mean /= nvec;

		/* Initialize weights and residuals */
		weights = (double *)palloc0(sizeof(double) * dim);
		weights_old = (double *)palloc(sizeof(double) * dim);
		residuals = (double *)palloc(sizeof(double) * nvec);

		/* Initialize residuals */
		for (i = 0; i < nvec; i++)
			residuals[i] = dataset.targets[i] - y_mean;

		/* Coordinate descent */
		for (iter = 0; iter < max_iters && !converged; iter++)
		{
			double diff;

			memcpy(weights_old, weights, sizeof(double) * dim);

			/* Update each coordinate */
			for (j = 0; j < dim; j++)
			{
				double rho = 0.0;
				double z = 0.0;
				double old_weight;
				float *feature_col_j;

				/* Compute rho = X_j^T * residuals */
				/* Access feature column j using 1D indexing: features[i * dim + j] */
				for (i = 0; i < nvec; i++)
				{
					feature_col_j = dataset.features
						+ (i * dim + j);
					rho += (*feature_col_j) * residuals[i];
				}

				/* Compute z = X_j^T * X_j */
				for (i = 0; i < nvec; i++)
				{
					feature_col_j = dataset.features
						+ (i * dim + j);
					z += (*feature_col_j)
						* (*feature_col_j);
				}

				if (z < 1e-10)
					continue;

				/* Soft thresholding */
				old_weight = weights[j];
				weights[j] =
					soft_threshold(rho / z, lambda / z);

				/* Update residuals */
				if (weights[j] != old_weight)
				{
					double weight_diff;

					weight_diff = weights[j] - old_weight;
					for (i = 0; i < nvec; i++)
					{
						feature_col_j = dataset.features
							+ (i * dim + j);
						residuals[i] -= (*feature_col_j)
							* weight_diff;
					}
				}
			}

			/* Check convergence */
			diff = 0.0;
			for (j = 0; j < dim; j++)
			{
				double d = weights[j] - weights_old[j];
				diff += d * d;
			}

			if (sqrt(diff) < 1e-6)
			{
				converged = true;
				elog(DEBUG1,
					"lasso: converged after %d iterations",
					iter + 1);
			}
		}

		if (!converged)
		{
			elog(DEBUG1,
				"lasso: did not converge after %d iterations",
				max_iters);
		}

		/* Build LassoModel */
		model = (LassoModel *)palloc0(sizeof(LassoModel));
		model->n_features = dim;
		model->n_samples = nvec;
		model->intercept = y_mean;
		model->lambda = lambda;
		model->max_iters = max_iters;
		model->coefficients = (double *)palloc(sizeof(double) * dim);
		for (i = 0; i < dim; i++)
			model->coefficients[i] = weights[i];

		/* Compute metrics (R², MSE, MAE) */
		{
			double ss_tot = 0.0;
			double ss_res = 0.0;
			double mse = 0.0;
			double mae = 0.0;

			for (i = 0; i < nvec; i++)
			{
				float *row = dataset.features + (i * dim);
				double y_pred = model->intercept;
				double error;
				int feat_idx;

				for (feat_idx = 0; feat_idx < dim; feat_idx++)
					y_pred += model->coefficients[feat_idx]
						* row[feat_idx];

				error = dataset.targets[i] - y_pred;
				mse += error * error;
				mae += fabs(error);
				ss_res += error * error;
				ss_tot += (dataset.targets[i] - y_mean)
					* (dataset.targets[i] - y_mean);
			}

			mse /= nvec;
			mae /= nvec;
			model->r_squared = (ss_tot > 0.0)
				? (1.0 - (ss_res / ss_tot))
				: 0.0;
			model->mse = mse;
			model->mae = mae;
		}

		/* Validate model before serialization */
		if (model->n_features <= 0 || model->n_features > 10000)
		{
			if (model->coefficients != NULL)
				pfree(model->coefficients);
			pfree(model);
			pfree(weights);
			pfree(weights_old);
			pfree(residuals);
			lasso_dataset_free(&dataset);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("lasso: model.n_features is "
					       "invalid (%d) before "
					       "serialization",
						model->n_features)));
		}

		elog(DEBUG1,
			"lasso: serializing model with n_features=%d, "
			"n_samples=%d, lambda=%.6f",
			model->n_features,
			model->n_samples,
			model->lambda);

		/* Serialize model */
		model_blob = lasso_model_serialize(model);

		/* Build metrics JSON */
		initStringInfo(&metricsbuf);
		appendStringInfo(&metricsbuf,
			"{\"algorithm\":\"lasso\","
			"\"storage\":\"cpu\","
			"\"n_features\":%d,"
			"\"n_samples\":%d,"
			"\"lambda\":%.6f,"
			"\"max_iters\":%d,"
			"\"r_squared\":%.6f,"
			"\"mse\":%.6f,"
			"\"mae\":%.6f}",
			model->n_features,
			model->n_samples,
			model->lambda,
			model->max_iters,
			model->r_squared,
			model->mse,
			model->mae);

		metrics_json = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(metricsbuf.data)));

		/* Register in catalog */
		{
			MLCatalogModelSpec spec;

			memset(&spec, 0, sizeof(MLCatalogModelSpec));
			spec.algorithm = "lasso";
			spec.model_type = "regression";
			spec.training_table = tbl_str;
			spec.training_column = targ_str;
			spec.model_data = model_blob;
			spec.metrics = metrics_json;

			/* Build hyperparameters JSON */
			{
				StringInfoData hyperbuf_cpu;
				initStringInfo(&hyperbuf_cpu);
				appendStringInfo(&hyperbuf_cpu,
					"{\"lambda\":%.6f,\"max_iters\":%d}",
					lambda,
					max_iters);
				spec.parameters = DatumGetJsonbP(
					DirectFunctionCall1(jsonb_in,
						CStringGetDatum(
							hyperbuf_cpu.data)));
			}

			model_id = ml_catalog_register_model(&spec);
		}

		/* Cleanup */
		pfree(weights);
		pfree(weights_old);
		pfree(residuals);
		if (model->coefficients != NULL)
			pfree(model->coefficients);
		pfree(model);
		lasso_dataset_free(&dataset);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);

		PG_RETURN_INT32(model_id);
	}
}

/*
 * predict_lasso_regression_model_id
 *
 * Makes predictions using trained Lasso Regression model from catalog
 */
PG_FUNCTION_INFO_V1(predict_lasso_regression_model_id);

Datum
predict_lasso_regression_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	Vector *features;
	LassoModel *model = NULL;
	double prediction;
	int i;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("lasso: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("lasso: features vector is required")));

	features = PG_GETARG_VECTOR_P(1);

	/* Try GPU prediction first */
	if (lasso_try_gpu_predict_catalog(model_id, features, &prediction))
	{
		elog(DEBUG1,
			"lasso: GPU prediction succeeded, prediction=%.6f",
			prediction);
		PG_RETURN_FLOAT8(prediction);
	} else
	{
		elog(DEBUG1,
			"lasso: GPU prediction failed or not available, trying "
			"CPU");
	}

	/* Load model from catalog */
	if (!lasso_load_model_from_catalog(model_id, &model))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("lasso: model %d not found", model_id)));

	/* Validate feature dimension */
	if (model->n_features > 0 && features->dim != model->n_features)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("lasso: feature dimension mismatch "
				       "(expected %d, got %d)",
					model->n_features,
					features->dim)));

	/* Compute prediction: y = intercept + coef1*x1 + coef2*x2 + ... */
	prediction = model->intercept;
	for (i = 0; i < model->n_features && i < features->dim; i++)
		prediction += model->coefficients[i] * features->data[i];

	/* Cleanup */
	if (model != NULL)
	{
		if (model->coefficients != NULL)
			pfree(model->coefficients);
		pfree(model);
	}

	PG_RETURN_FLOAT8(prediction);
}

/*
 * train_elastic_net
 *
 * Trains Elastic Net (L1 + L2 regularization)
 * Combines Ridge and Lasso penalties
 */
PG_FUNCTION_INFO_V1(train_elastic_net);

Datum
train_elastic_net(PG_FUNCTION_ARGS)
{
	text *table_name = PG_GETARG_TEXT_PP(0);
	text *feature_col = PG_GETARG_TEXT_PP(1);
	text *target_col = PG_GETARG_TEXT_PP(2);
	double alpha =
		PG_GETARG_FLOAT8(3); /* Overall regularization strength */
	double l1_ratio =
		PG_GETARG_FLOAT8(4); /* L1 vs L2 ratio (0=Ridge, 1=Lasso) */

	Datum *result_datums;
	ArrayType *result_array;

	/* Placeholder implementation - would combine Ridge and Lasso */
	elog(NOTICE, "Elastic Net: alpha=%.4f, l1_ratio=%.4f", alpha, l1_ratio);
	elog(NOTICE,
		"Training on table: %s, features: %s, target: %s",
		text_to_cstring(table_name),
		text_to_cstring(feature_col),
		text_to_cstring(target_col));

	/* Return dummy coefficients */
	result_datums = (Datum *)palloc(sizeof(Datum) * 6);
	result_datums[0] = Float8GetDatum(0.0); /* bias */
	for (int i = 1; i < 6; i++)
		result_datums[i] = Float8GetDatum(0.1);

	result_array = construct_array(
		result_datums, 6, FLOAT8OID, 8, FLOAT8PASSBYVAL, 'd');

	PG_RETURN_ARRAYTYPE_P(result_array);
}
