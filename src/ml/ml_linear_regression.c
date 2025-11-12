/*-------------------------------------------------------------------------
 *
 * ml_linear_regression.c
 *    Linear Regression implementation for supervised learning
 *
 * Implements ordinary least squares (OLS) linear regression using
 * normal equations: β = (X'X)^(-1)X'y
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_linear_regression.c
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
#include "ml_linear_regression_internal.h"
#include "ml_catalog.h"
#include "neurondb_gpu_bridge.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "ml_gpu_linear_regression.h"

#include <math.h>
#include <float.h>

typedef struct LinRegDataset
{
	float *features;
	double *targets;
	int n_samples;
	int feature_dim;
} LinRegDataset;

static void linreg_dataset_init(LinRegDataset *dataset);
static void linreg_dataset_free(LinRegDataset *dataset);
static void linreg_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	LinRegDataset *dataset);
static bytea *linreg_model_serialize(const LinRegModel *model);
static LinRegModel *linreg_model_deserialize(const bytea *data);
static bool linreg_metadata_is_gpu(Jsonb *metadata);
static bool linreg_try_gpu_predict_catalog(int32 model_id,
	const Vector *feature_vec,
	double *result_out);
static bool linreg_load_model_from_catalog(int32 model_id, LinRegModel **out);

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
	augmented = (double **) palloc(sizeof(double *) * n);
	for (i = 0; i < n; i++)
	{
		augmented[i] = (double *) palloc(sizeof(double) * 2 * n);
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
					augmented[k][j] -= factor * augmented[i][j];
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
 * linreg_dataset_init
 */
static void
linreg_dataset_init(LinRegDataset *dataset)
{
	if (dataset == NULL)
		return;
	memset(dataset, 0, sizeof(LinRegDataset));
}

/*
 * linreg_dataset_free
 */
static void
linreg_dataset_free(LinRegDataset *dataset)
{
	if (dataset == NULL)
		return;
	if (dataset->features != NULL)
		pfree(dataset->features);
	if (dataset->targets != NULL)
		pfree(dataset->targets);
	linreg_dataset_init(dataset);
}

/*
 * linreg_dataset_load
 */
static void
linreg_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	LinRegDataset *dataset)
{
	StringInfoData query;
	MemoryContext oldcontext;
	int ret;
	int n_samples = 0;
	int feature_dim = 0;
	int i;

	if (dataset == NULL)
		ereport(ERROR,
			(errmsg("linreg_dataset_load: dataset is NULL")));

	oldcontext = CurrentMemoryContext;

	/* Initialize query in caller's context before SPI_connect */
	initStringInfo(&query);
	MemoryContextSwitchTo(oldcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errmsg("linreg_dataset_load: SPI_connect failed")));
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quoted_feat, quoted_target, quoted_tbl, quoted_feat, quoted_target);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		ereport(ERROR,
			(errmsg("linreg_dataset_load: query failed")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		SPI_finish();
		ereport(ERROR,
			(errmsg("linreg_dataset_load: need at least 10 samples, got %d",
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
			(errmsg("linreg_dataset_load: could not determine feature dimension")));
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
				(errmsg("linreg_dataset_load: inconsistent vector dimensions")));
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
				dataset->targets[i] = (double)DatumGetInt32(targ_datum);
			else
				dataset->targets[i] = DatumGetFloat8(targ_datum);
		}
	}

	dataset->n_samples = n_samples;
	dataset->feature_dim = feature_dim;

	SPI_finish();
}

/*
 * linreg_model_serialize
 */
static bytea *
linreg_model_serialize(const LinRegModel *model)
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
				errmsg("linreg_model_serialize: invalid n_features %d (corrupted model?)",
					model->n_features)));
	}
	
	pq_begintypsend(&buf);
	
	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendfloat8(&buf, model->intercept);
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
 * linreg_model_deserialize
 */
static LinRegModel *
linreg_model_deserialize(const bytea *data)
{
	LinRegModel *model;
	StringInfoData buf;
	int i;
	
	if (data == NULL)
		return NULL;
	
	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;
	
	model = (LinRegModel *)palloc0(sizeof(LinRegModel));
	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->intercept = pq_getmsgfloat8(&buf);
	model->r_squared = pq_getmsgfloat8(&buf);
	model->mse = pq_getmsgfloat8(&buf);
	model->mae = pq_getmsgfloat8(&buf);
	
	/* Validate deserialized values */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		pfree(model);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("linreg: invalid n_features %d in deserialized model (corrupted data?)",
					model->n_features)));
	}
	if (model->n_samples < 0 || model->n_samples > 100000000)
	{
		pfree(model);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("linreg: invalid n_samples %d in deserialized model (corrupted data?)",
					model->n_samples)));
	}
	
	if (model->n_features > 0)
	{
		model->coefficients = (double *)palloc(sizeof(double) * model->n_features);
		for (i = 0; i < model->n_features; i++)
			model->coefficients[i] = pq_getmsgfloat8(&buf);
	}
	
	return model;
}

/*
 * linreg_metadata_is_gpu
 */
static bool
linreg_metadata_is_gpu(Jsonb *metadata)
{
	char *meta_text = NULL;
	bool is_gpu = false;
	
	if (metadata == NULL)
		return false;
	
	PG_TRY();
	{
		meta_text = DatumGetCString(
			DirectFunctionCall1(jsonb_out, JsonbPGetDatum(metadata)));
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
 * linreg_try_gpu_predict_catalog
 */
static bool
linreg_try_gpu_predict_catalog(int32 model_id,
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
	
	if (!ml_catalog_fetch_model_payload(
			model_id, &payload, NULL, &metrics))
		return false;
	
	if (payload == NULL)
		goto cleanup;
	
	if (!linreg_metadata_is_gpu(metrics))
		goto cleanup;
	
	if (ndb_gpu_linreg_predict(payload,
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
 * linreg_load_model_from_catalog
 */
static bool
linreg_load_model_from_catalog(int32 model_id, LinRegModel **out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	
	if (out == NULL)
		return false;
	
	*out = NULL;
	
	if (!ml_catalog_fetch_model_payload(
			model_id, &payload, NULL, &metrics))
		return false;
	
	if (payload == NULL)
	{
		if (metrics != NULL)
			pfree(metrics);
		return false;
	}
	
	/* Skip GPU models - they should be handled by GPU prediction */
	if (linreg_metadata_is_gpu(metrics))
	{
		pfree(payload);
		if (metrics != NULL)
			pfree(metrics);
		return false;
	}
	
	*out = linreg_model_deserialize(payload);
	
	pfree(payload);
	if (metrics != NULL)
		pfree(metrics);
	
	return (*out != NULL);
}

/*
 * train_linear_regression
 *
 * Trains a linear regression model using OLS
 * Returns model_id (for GPU path) or coefficients array (for CPU path)
 */
PG_FUNCTION_INFO_V1(train_linear_regression);

Datum
train_linear_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	int			nvec = 0;
	int			dim = 0;
	LinRegDataset dataset;
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
	
	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);
	
	linreg_dataset_init(&dataset);
	
	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_target = quote_identifier(targ_str);
	
	linreg_dataset_load(quoted_tbl, quoted_feat, quoted_target, &dataset);
	
	nvec = dataset.n_samples;
	dim = dataset.feature_dim;
	
	if (nvec < 10)
	{
		linreg_dataset_free(&dataset);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
			(errmsg("Need at least 10 samples for linear regression, have %d", nvec)));
	}
	
	/* Try GPU training first */
	if (neurondb_gpu_is_available() && nvec > 0 && dim > 0)
	{
		initStringInfo(&hyperbuf);
		appendStringInfo(&hyperbuf, "{}");
		gpu_hyperparams = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(hyperbuf.data)));
		
		if (ndb_gpu_try_train_model("linear_regression",
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
			elog(NOTICE, "linear_regression: GPU training succeeded");
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
			
			spec.algorithm = "linear_regression";
			spec.model_type = "regression";
			
			model_id = ml_catalog_register_model(&spec);
			
			if (gpu_err != NULL)
				pfree(gpu_err);
			if (gpu_hyperparams != NULL)
				pfree(gpu_hyperparams);
			ndb_gpu_free_train_result(&gpu_result);
			linreg_dataset_free(&dataset);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			
			PG_RETURN_INT32(model_id);
		}
		else
		{
			elog(DEBUG1, "linear_regression: GPU training unavailable, using CPU");
			if (gpu_err != NULL)
				pfree(gpu_err);
			if (gpu_hyperparams != NULL)
				pfree(gpu_hyperparams);
			ndb_gpu_free_train_result(&gpu_result);
		}
	}
	
	/* CPU training path */
	{
		double	  **XtX = NULL;
		double	  **XtX_inv = NULL;
		double	   *Xty = NULL;
		double	   *beta = NULL;
		int			i, j, k;
		int			dim_with_intercept;
		LinRegModel *model;
		bytea *model_blob;
		Jsonb *metrics_json;
		StringInfoData metricsbuf;
		
		/* Allocate matrices for normal equations: β = (X'X)^(-1)X'y */
		dim_with_intercept = dim + 1; /* Add 1 for intercept */
		XtX = (double **) palloc(sizeof(double *) * dim_with_intercept);
		XtX_inv = (double **) palloc(sizeof(double *) * dim_with_intercept);
		for (i = 0; i < dim_with_intercept; i++)
		{
			XtX[i] = (double *) palloc0(sizeof(double) * dim_with_intercept);
			XtX_inv[i] = (double *) palloc(sizeof(double) * dim_with_intercept);
		}
		Xty = (double *) palloc0(sizeof(double) * dim_with_intercept);
		beta = (double *) palloc(sizeof(double) * dim_with_intercept);
		
		/* Compute X'X and X'y using dataset */
		for (i = 0; i < nvec; i++)
		{
			/* Add intercept term (1.0) */
			double *xi = (double *) palloc(sizeof(double) * dim_with_intercept);
			float *row = dataset.features + (i * dim);
			
			xi[0] = 1.0;
			for (k = 1; k < dim_with_intercept; k++)
				xi[k] = row[k-1];
			
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
	
		/* Add small regularization (ridge) to diagonal to handle near-singular matrices */
		{
			double lambda = 1e-6; /* Small regularization parameter */
			for (i = 0; i < dim_with_intercept; i++)
			{
				XtX[i][i] += lambda;
			}
		}
		
		/* Invert X'X */
		if (!matrix_invert(XtX, dim_with_intercept, XtX_inv))
		{
			/* If still singular after regularization, try larger lambda */
			double lambda = 1e-3;
			for (i = 0; i < dim_with_intercept; i++)
			{
				XtX[i][i] += lambda;
			}
			
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
				linreg_dataset_free(&dataset);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(targ_str);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("linear_regression: matrix is singular even after regularization"),
						errhint("Try removing correlated features, reducing feature count, or using ridge regression")));
			}
			else
			{
				elog(DEBUG1, "linear_regression: used regularization (lambda=1e-3) to handle near-singular matrix");
			}
		}
		
		/* Compute β = (X'X)^(-1)X'y */
		for (i = 0; i < dim_with_intercept; i++)
		{
			beta[i] = 0.0;
			for (j = 0; j < dim_with_intercept; j++)
				beta[i] += XtX_inv[i][j] * Xty[j];
		}
		
		/* Build LinRegModel */
		model = (LinRegModel *)palloc0(sizeof(LinRegModel));
		model->n_features = dim;
		model->n_samples = nvec;
		model->intercept = beta[0];
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
				int j;
				
				for (j = 0; j < dim; j++)
					y_pred += model->coefficients[j] * row[j];
				
				error = dataset.targets[i] - y_pred;
				mse += error * error;
				mae += fabs(error);
				ss_res += error * error;
				ss_tot += (dataset.targets[i] - y_mean) * (dataset.targets[i] - y_mean);
			}
			
			mse /= nvec;
			mae /= nvec;
			model->r_squared = 1.0 - (ss_res / ss_tot);
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
			linreg_dataset_free(&dataset);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("linreg: model.n_features is invalid (%d) before serialization", model->n_features)));
		}
		
		elog(DEBUG1, "linreg: serializing model with n_features=%d, n_samples=%d",
			model->n_features, model->n_samples);
		
		/* Serialize model */
		model_blob = linreg_model_serialize(model);
		
		/* Note: GPU packing is disabled for CPU-trained models to avoid format conflicts.
		 * GPU packing should only be used when the model was actually trained on GPU.
		 * CPU models must use CPU serialization format for proper deserialization.
		 */
		
		/* Build metrics JSON */
		initStringInfo(&metricsbuf);
		appendStringInfo(&metricsbuf,
			"{\"algorithm\":\"linear_regression\","
			"\"storage\":\"cpu\","
			"\"n_features\":%d,"
			"\"n_samples\":%d,"
			"\"r_squared\":%.6f,"
			"\"mse\":%.6f,"
			"\"mae\":%.6f}",
			model->n_features,
			model->n_samples,
			model->r_squared,
			model->mse,
			model->mae);
		
		metrics_json = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(metricsbuf.data)));
		
		/* Register in catalog */
		{
			MLCatalogModelSpec spec;
			
			memset(&spec, 0, sizeof(MLCatalogModelSpec));
			spec.algorithm = "linear_regression";
			spec.model_type = "regression";
			spec.training_table = tbl_str;
			spec.training_column = targ_str;
			spec.model_data = model_blob;
			spec.metrics = metrics_json;
			
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
		linreg_dataset_free(&dataset);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		
		PG_RETURN_INT32(model_id);
	}
}

/*
 * predict_linear_regression_model_id
 *
 * Makes predictions using trained linear regression model from catalog
 */
PG_FUNCTION_INFO_V1(predict_linear_regression_model_id);

Datum
predict_linear_regression_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	Vector *features;
	LinRegModel *model = NULL;
	double prediction;
	int i;
	
	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("linear_regression: model_id is required")));
	
	model_id = PG_GETARG_INT32(0);
	
	if (PG_ARGISNULL(1))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("linear_regression: features vector is required")));
	
	features = PG_GETARG_VECTOR_P(1);
	
	/* Try GPU prediction first */
	if (linreg_try_gpu_predict_catalog(model_id, features, &prediction))
	{
		elog(DEBUG1, "linear_regression: GPU prediction succeeded, prediction=%.6f", prediction);
		PG_RETURN_FLOAT8(prediction);
	}
	else
	{
		elog(DEBUG1, "linear_regression: GPU prediction failed or not available, trying CPU");
	}
	
	/* Load model from catalog */
	if (!linreg_load_model_from_catalog(model_id, &model))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("linear_regression: model %d not found",
					model_id)));
	
	/* Validate feature dimension */
	if (model->n_features > 0 && features->dim != model->n_features)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("linear_regression: feature dimension mismatch "
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
 * predict_linear_regression
 *
 * Makes predictions using trained linear regression coefficients (legacy)
 */
PG_FUNCTION_INFO_V1(predict_linear_regression);

Datum
predict_linear_regression(PG_FUNCTION_ARGS)
{
	ArrayType  *coef_array;
	Vector	   *features;
	int			ncoef;
	float8	   *coef;
	float	   *x;
	int			dim;
	double		prediction;
	int			i;
	
	coef_array = PG_GETARG_ARRAYTYPE_P(0);
	features = PG_GETARG_VECTOR_P(1);
	
	/* Extract coefficients */
	if (ARR_NDIM(coef_array) != 1)
		ereport(ERROR,
				(errmsg("Coefficients must be 1-dimensional array")));
	
	ncoef = ARR_DIMS(coef_array)[0];
	coef = (float8 *) ARR_DATA_PTR(coef_array);
	
	x = features->data;
	dim = features->dim;
	
	if (ncoef != dim + 1)
		ereport(ERROR,
				(errmsg("Coefficient dimension mismatch: expected %d, got %d", dim + 1, ncoef)));
	
	/* Compute prediction: y = β0 + β1*x1 + β2*x2 + ... */
	prediction = coef[0]; /* intercept */
	for (i = 0; i < dim; i++)
		prediction += coef[i + 1] * x[i];
	
	PG_RETURN_FLOAT8(prediction);
}

/*
 * evaluate_linear_regression
 *
 * Evaluates model performance (R², MSE, MAE)
 * Returns: [r_squared, mse, mae, rmse]
 */
PG_FUNCTION_INFO_V1(evaluate_linear_regression);

Datum
evaluate_linear_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	ArrayType  *coef_array;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			ncoef;
	float8	   *coef;
	double		mse = 0.0;
	double		mae = 0.0;
	double		ss_tot = 0.0;
	double		ss_res = 0.0;
	double		y_mean = 0.0;
	double		r_squared;
	double		rmse;
	int			i;
	Datum	   *result_datums;
	ArrayType  *result_array;
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
	(void) ncoef;  /* Suppress unused variable warning */
	coef = (float8 *) ARR_DATA_PTR(coef_array);

	oldcontext = CurrentMemoryContext;
	
	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
				(errmsg("SPI_connect failed")));
	
	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query, "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str, targ_str, tbl_str, feat_str, targ_str);
	
	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
				(errmsg("Query failed")));
	
	nvec = SPI_processed;
	
	/* First pass: compute mean of y */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		targ_datum;
		bool		targ_null;
		
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (!targ_null)
			y_mean += DatumGetFloat8(targ_datum);
	}
	y_mean /= nvec;
	
	/* Second pass: compute predictions and metrics */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec;
		double		y_true;
		double		y_pred;
		double		error;
		int			j;
		
		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		
		if (feat_null || targ_null)
			continue;
		
		vec = DatumGetVector(feat_datum);
		y_true = DatumGetFloat8(targ_datum);
		
		/* Compute prediction */
		y_pred = coef[0]; /* intercept */
		for (j = 0; j < vec->dim; j++)
			y_pred += coef[j + 1] * vec->data[j];
		
		/* Compute errors */
		error = y_true - y_pred;
		mse += error * error;
		mae += fabs(error);
		ss_res += error * error;
		ss_tot += (y_true - y_mean) * (y_true - y_mean);
	}
	
	mse /= nvec;
	mae /= nvec;
	rmse = sqrt(mse);
	r_squared = 1.0 - (ss_res / ss_tot);
	
	SPI_finish();
	
	/* Build result array: [r_squared, mse, mae, rmse] */
	MemoryContextSwitchTo(oldcontext);
	
	result_datums = (Datum *) palloc(sizeof(Datum) * 4);
	result_datums[0] = Float8GetDatum(r_squared);
	result_datums[1] = Float8GetDatum(mse);
	result_datums[2] = Float8GetDatum(mae);
	result_datums[3] = Float8GetDatum(rmse);
	
	result_array = construct_array(result_datums, 4,
								   FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
	
	pfree(result_datums);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(targ_str);
	
	PG_RETURN_ARRAYTYPE_P(result_array);
}

/* GPU model state for Linear Regression */
typedef struct LinRegGpuModelState
{
	bytea *model_blob;
	Jsonb *metrics;
	int feature_dim;
	int n_samples;
} LinRegGpuModelState;

static void
linreg_gpu_release_state(LinRegGpuModelState *state)
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
linreg_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	LinRegGpuModelState *state;
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
	
	rc = ndb_gpu_linreg_train(spec->feature_matrix,
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
		linreg_gpu_release_state((LinRegGpuModelState *)model->backend_state);
		model->backend_state = NULL;
	}
	
	state = (LinRegGpuModelState *)palloc0(sizeof(LinRegGpuModelState));
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;
	
	/* Store metrics in model state for later retrieval */
	if (metrics != NULL)
	{
		state->metrics = (Jsonb *)PG_DETOAST_DATUM_COPY(PointerGetDatum(metrics));
		elog(DEBUG1, "linreg_gpu_train: stored metrics in state: %p", (void *)state->metrics);
	}
	else
	{
		state->metrics = NULL;
		elog(WARNING, "linreg_gpu_train: metrics is NULL, cannot store in state!");
	}
	
	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;
	
	return true;
}

static bool
linreg_gpu_predict(const MLGpuModel *model,
	const float *input,
	int input_dim,
	float *output,
	int output_dim,
	char **errstr)
{
	const LinRegGpuModelState *state;
	double prediction;
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
	
	state = (const LinRegGpuModelState *)model->backend_state;
	if (state->model_blob == NULL)
		return false;
	
	rc = ndb_gpu_linreg_predict(state->model_blob,
		input,
		state->feature_dim > 0 ? state->feature_dim : input_dim,
		&prediction,
		errstr);
	if (rc != 0)
		return false;
	
	output[0] = (float)prediction;
	
	return true;
}

static bool
linreg_gpu_evaluate(const MLGpuModel *model,
	const MLGpuEvalSpec *spec,
	MLGpuMetrics *out,
	char **errstr)
{
	const LinRegGpuModelState *state;
	Jsonb *metrics_json;
	
	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;
	
	state = (const LinRegGpuModelState *)model->backend_state;
	
	/* Build metrics JSON from stored model metadata */
	{
		StringInfoData buf;
		
		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"linear_regression\","
			"\"storage\":\"gpu\","
			"\"n_features\":%d,"
			"\"n_samples\":%d}",
			state->feature_dim > 0 ? state->feature_dim : 0,
			state->n_samples > 0 ? state->n_samples : 0);
		
		metrics_json = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(buf.data)));
		pfree(buf.data);
	}
	
	if (out != NULL)
		out->payload = metrics_json;
	
	return true;
}

static bool
linreg_gpu_serialize(const MLGpuModel *model,
	bytea **payload_out,
	Jsonb **metadata_out,
	char **errstr)
{
	const LinRegGpuModelState *state;
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
	
	state = (const LinRegGpuModelState *)model->backend_state;
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
		*metadata_out = (Jsonb *)PG_DETOAST_DATUM_COPY(PointerGetDatum(state->metrics));
		elog(DEBUG1, "linreg_gpu_serialize: returning metrics: %p", (void *)*metadata_out);
	}
	else if (metadata_out != NULL)
	{
		*metadata_out = NULL;
		elog(WARNING, "linreg_gpu_serialize: state->metrics is NULL, cannot return metrics!");
	}
	
	return true;
}

static bool
linreg_gpu_deserialize(MLGpuModel *model,
	const bytea *payload,
	const Jsonb *metadata,
	char **errstr)
{
	LinRegGpuModelState *state;
	bytea *payload_copy;
	int payload_size;
	
	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;
	
	payload_size = VARSIZE(payload);
	payload_copy = (bytea *)palloc(payload_size);
	memcpy(payload_copy, payload, payload_size);
	
	state = (LinRegGpuModelState *)palloc0(sizeof(LinRegGpuModelState));
	state->model_blob = payload_copy;
	state->feature_dim = -1;
	state->n_samples = -1;
	
	if (model->backend_state != NULL)
		linreg_gpu_release_state((LinRegGpuModelState *)model->backend_state);
	
	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;
	
	return true;
}

static void
linreg_gpu_destroy(MLGpuModel *model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		linreg_gpu_release_state((LinRegGpuModelState *)model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps linreg_gpu_model_ops = {
	.algorithm = "linear_regression",
	.train = linreg_gpu_train,
	.predict = linreg_gpu_predict,
	.evaluate = linreg_gpu_evaluate,
	.serialize = linreg_gpu_serialize,
	.deserialize = linreg_gpu_deserialize,
	.destroy = linreg_gpu_destroy,
};

void
neurondb_gpu_register_linreg_model(void)
{
	ndb_gpu_register_model_ops(&linreg_gpu_model_ops);
}

