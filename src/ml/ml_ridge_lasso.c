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
#include "utils/memutils.h"
#include "utils/jsonb.h"
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
#include "neurondb_cuda_ridge.h"
#include "neurondb_cuda_lasso.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_safe_memory.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
extern int ndb_cuda_ridge_evaluate(const bytea *model_data,
	const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	double *mse_out,
	double *mae_out,
	double *rmse_out,
	double *r_squared_out,
	char **errstr);
extern int ndb_cuda_lasso_evaluate(const bytea *model_data,
	const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	double *mse_out,
	double *mae_out,
	double *rmse_out,
	double *r_squared_out,
	char **errstr);
#endif

#include <math.h>
#include <float.h>

typedef struct RidgeDataset
{
	float *features;
	double *targets;
	int n_samples;
	int feature_dim;
} RidgeDataset;

typedef RidgeDataset LassoDataset;

/*
 * Streaming accumulator for incremental X'X and X'y computation for Ridge
 * This avoids loading all data into memory at once
 */
typedef struct RidgeStreamAccum
{
	double **XtX;
	double *Xty;
	int feature_dim;
	int n_samples;
	double y_sum;
	double y_sq_sum;
	bool initialized;
} RidgeStreamAccum;

static void ridge_dataset_init(RidgeDataset *dataset);
static void ridge_dataset_free(RidgeDataset *dataset);
static void ridge_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	RidgeDataset *dataset);
static void ridge_dataset_load_limited(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	RidgeDataset *dataset,
	int max_rows);
static void ridge_stream_accum_init(RidgeStreamAccum *accum, int dim);
static void ridge_stream_accum_free(RidgeStreamAccum *accum);
static void ridge_stream_accum_add_row(RidgeStreamAccum *accum,
	const float *features,
	double target);
static void ridge_stream_process_chunk(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	RidgeStreamAccum *accum,
	int chunk_size,
	int offset,
	int *rows_processed);
static bytea *ridge_model_serialize(const RidgeModel *model);
static RidgeModel *ridge_model_deserialize(const bytea *data);
static bool ridge_metadata_is_gpu(Jsonb *metadata);
static bool ridge_try_gpu_predict_catalog(int32 model_id,
	const Vector *feature_vec,
	double *result_out);
static bool ridge_load_model_from_catalog(int32 model_id, RidgeModel **out);

/*
 * Streaming accumulator for Lasso coordinate descent
 * Stores residuals and feature data for incremental updates
 */
typedef struct LassoStreamAccum
{
	double *residuals;
	float *features;
	double *targets;
	int feature_dim;
	int n_samples;
	double y_mean;
	bool initialized;
} LassoStreamAccum;

static void lasso_dataset_init(LassoDataset *dataset);
static void lasso_dataset_free(LassoDataset *dataset);
static void lasso_dataset_load_limited(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	LassoDataset *dataset,
	int max_rows);
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

	for (i = 0; i < n; i++)
	{
		pivot = augmented[i][i];
		if (fabs(pivot) < 1e-10)
		{
			bool found = false;
			for (k = i + 1; k < n; k++)
			{
				if (fabs(augmented[k][i]) > 1e-10)
				{
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
				for (j = 0; j < n; j++)
				{
					NDB_SAFE_PFREE_AND_NULL(augmented[j]);
				}
				NDB_SAFE_PFREE_AND_NULL(augmented);
				return false;
			}
		}

		for (j = 0; j < 2 * n; j++)
			augmented[i][j] /= pivot;

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

	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			result[i][j] = augmented[i][j + n];

	/* Cleanup */
	for (i = 0; i < n; i++)
	{
		NDB_SAFE_PFREE_AND_NULL(augmented[i]);
	}
	NDB_SAFE_PFREE_AND_NULL(augmented);

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
	{
		NDB_SAFE_PFREE_AND_NULL(dataset->features);
		dataset->features = NULL;
	}
	if (dataset->targets != NULL)
	{
		NDB_SAFE_PFREE_AND_NULL(dataset->targets);
		dataset->targets = NULL;
	}
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
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: ridge_dataset_load: dataset is NULL")));

	oldcontext = CurrentMemoryContext;

	/* Initialize query in caller's context before SPI_connect */
	initStringInfo(&query);
	MemoryContextSwitchTo(oldcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: ridge_dataset_load: SPI_connect failed")));
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quoted_feat,
		quoted_target,
		quoted_tbl,
		quoted_feat,
		quoted_target);
	elog(DEBUG1, "ridge_dataset_load: executing query: %s", query.data);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: ridge_dataset_load: query failed")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: ridge_dataset_load: need at least 10 samples, got %d",
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
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: ridge_dataset_load: could not determine "
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
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: ridge_dataset_load: inconsistent "
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
 * ridge_dataset_load_limited
 *
 * Load dataset with LIMIT clause to avoid loading too much data
 */
static void
ridge_dataset_load_limited(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	RidgeDataset *dataset,
	int max_rows)
{
	StringInfoData query;
	MemoryContext oldcontext;
	int ret;
	int n_samples = 0;
	int feature_dim = 0;
	int i;

	Oid feat_type_oid = InvalidOid;
	bool feat_is_array = false;

	if (dataset == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: ridge_dataset_load_limited: dataset is NULL")));

	if (max_rows <= 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: ridge_dataset_load_limited: max_rows must be positive")));

	oldcontext = CurrentMemoryContext;

	/* Initialize query in caller's context before SPI_connect */
	initStringInfo(&query);
	MemoryContextSwitchTo(oldcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: ridge_dataset_load_limited: SPI_connect failed")));
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %d",
		quoted_feat,
		quoted_target,
		quoted_tbl,
		quoted_feat,
		quoted_target,
		max_rows);
	elog(DEBUG1, "ridge_dataset_load_limited: executing query: %s", query.data);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: ridge_dataset_load_limited: query failed")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: ridge_dataset_load_limited: need at least 10 samples, got %d",
					n_samples)));
	}

	/* Determine feature column type and dimension before allocating */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

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
			if (feat_is_array)
			{
				ArrayType *arr = DatumGetArrayTypeP(feat_datum);

				if (ARR_NDIM(arr) != 1)
				{
					SPI_finish();
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("neurondb: ridge_dataset_load_limited: features array must be 1-D")));
				}
				feature_dim = ARR_DIMS(arr)[0];
			}
			else
			{
				vec = DatumGetVector(feat_datum);
				feature_dim = vec->dim;
			}
		}
	}

	if (feature_dim <= 0)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: ridge_dataset_load_limited: could not determine "
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

		row = dataset->features + (i * feature_dim);
		if (feat_is_array)
		{
			ArrayType *arr = DatumGetArrayTypeP(feat_datum);
			int ndims = ARR_NDIM(arr);
			int dimlen = (ndims == 1) ? ARR_DIMS(arr)[0] : 0;

			if (ndims != 1 || dimlen != feature_dim)
			{
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: ridge_dataset_load_limited: inconsistent array feature dimensions")));
			}
			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				float8 *data = (float8 *)ARR_DATA_PTR(arr);
				int j;

				for (j = 0; j < feature_dim; j++)
					row[j] = (float)data[j];
			}
			else
			{
				float4 *data = (float4 *)ARR_DATA_PTR(arr);

				memcpy(row, data, sizeof(float) * feature_dim);
			}
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			if (vec->dim != feature_dim)
			{
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: ridge_dataset_load_limited: inconsistent "
							"vector dimensions")));
			}
			memcpy(row, vec->data, sizeof(float) * feature_dim);
		}

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
 * ridge_stream_accum_init
 *
 * Initialize streaming accumulator for incremental X'X and X'y computation
 */
static void
ridge_stream_accum_init(RidgeStreamAccum *accum, int dim)
{
	int i;

	int dim_with_intercept;

	if (accum == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: ridge_stream_accum_init: accum is NULL")));

	if (dim <= 0 || dim > 10000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: ridge_stream_accum_init: invalid feature dimension %d",
					dim)));

	dim_with_intercept = dim + 1;

	memset(accum, 0, sizeof(RidgeStreamAccum));

	accum->feature_dim = dim;
	accum->n_samples = 0;
	accum->y_sum = 0.0;
	accum->y_sq_sum = 0.0;
	accum->initialized = false;

	/* Allocate X'X matrix */
	accum->XtX = (double **)palloc(sizeof(double *) * dim_with_intercept);
	if (accum->XtX == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: ridge_stream_accum_init: failed to allocate XtX matrix")));

	for (i = 0; i < dim_with_intercept; i++)
	{
		accum->XtX[i] = (double *)palloc0(sizeof(double) * dim_with_intercept);
		if (accum->XtX[i] == NULL)
		{
		/* Cleanup on failure */
		for (i--; i >= 0; i--)
		{
			NDB_SAFE_PFREE_AND_NULL(accum->XtX[i]);
			accum->XtX[i] = NULL;
		}
		NDB_SAFE_PFREE_AND_NULL(accum->XtX);
		accum->XtX = NULL;
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: ridge_stream_accum_init: failed to allocate XtX row")));
		}
	}

	/* Allocate X'y vector */
	accum->Xty = (double *)palloc0(sizeof(double) * dim_with_intercept);
	if (accum->Xty == NULL)
	{
		/* Cleanup on failure */
		for (i = 0; i < dim_with_intercept; i++)
		{
			NDB_SAFE_PFREE_AND_NULL(accum->XtX[i]);
			accum->XtX[i] = NULL;
		}
		NDB_SAFE_PFREE_AND_NULL(accum->XtX);
		accum->XtX = NULL;
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: ridge_stream_accum_init: failed to allocate Xty vector")));
	}

	accum->initialized = true;
}

/*
 * ridge_stream_accum_free
 *
 * Free memory allocated for streaming accumulator
 */
static void
ridge_stream_accum_free(RidgeStreamAccum *accum)
{
	int i;

	if (accum == NULL)
		return;

	if (accum->XtX != NULL)
	{
		int dim_with_intercept = accum->feature_dim + 1;

		for (i = 0; i < dim_with_intercept; i++)
		{
			if (accum->XtX[i] != NULL)
			{
				NDB_SAFE_PFREE_AND_NULL(accum->XtX[i]);
				accum->XtX[i] = NULL;
			}
		}
		NDB_SAFE_PFREE_AND_NULL(accum->XtX);
		accum->XtX = NULL;
	}

	if (accum->Xty != NULL)
	{
		NDB_SAFE_PFREE_AND_NULL(accum->Xty);
		accum->Xty = NULL;
	}

	memset(accum, 0, sizeof(RidgeStreamAccum));
}

/*
 * ridge_stream_accum_add_row
 *
 * Add a single row to the streaming accumulator, updating X'X and X'y
 */
static void
ridge_stream_accum_add_row(RidgeStreamAccum *accum,
	const float *features,
	double target)
{
	int i;

	int j;
	int dim_with_intercept;
	double *xi;

	if (accum == NULL || !accum->initialized)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: ridge_stream_accum_add_row: accumulator not initialized")));

	if (features == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: ridge_stream_accum_add_row: features is NULL")));

	dim_with_intercept = accum->feature_dim + 1;

	/* Allocate temporary vector for this row (with intercept) */
	xi = (double *)palloc(sizeof(double) * dim_with_intercept);
	if (xi == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: ridge_stream_accum_add_row: failed to allocate row vector")));

	/* Build row vector: [1, x1, x2, ..., xd] */
	xi[0] = 1.0; /* intercept term */
	for (i = 0; i < accum->feature_dim; i++)
		xi[i + 1] = (double)features[i];

	/* Update X'X: XtX[j][k] += xi[j] * xi[k] */
	for (j = 0; j < dim_with_intercept; j++)
	{
		for (i = 0; i < dim_with_intercept; i++)
			accum->XtX[j][i] += xi[j] * xi[i];

		/* Update X'y: Xty[j] += xi[j] * y */
		accum->Xty[j] += xi[j] * target;
	}

	/* Update statistics for metrics computation */
	accum->n_samples++;
	accum->y_sum += target;
	accum->y_sq_sum += target * target;

	NDB_SAFE_PFREE_AND_NULL(xi);
}

/*
 * ridge_stream_process_chunk
 *
 * Process a chunk of data from the table, accumulating statistics
 * Returns number of rows processed in this chunk
 */
static void
ridge_stream_process_chunk(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	RidgeStreamAccum *accum,
	int chunk_size,
	int offset,
	int *rows_processed)
{
	StringInfoData query;
	int ret;
	int i __attribute__((unused));
	int n_rows __attribute__((unused));
	Oid feat_type_oid __attribute__((unused)) = InvalidOid;
	bool feat_is_array __attribute__((unused)) = false;
	TupleDesc tupdesc __attribute__((unused));
	float *row_buffer __attribute__((unused)) = NULL;

	if (quoted_tbl == NULL || quoted_feat == NULL || quoted_target == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: ridge_stream_process_chunk: NULL parameter")));

	if (accum == NULL || !accum->initialized)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: ridge_stream_process_chunk: accumulator not initialized")));

	if (chunk_size <= 0 || chunk_size > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: ridge_stream_process_chunk: invalid chunk_size %d",
					chunk_size)));

	if (rows_processed == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: ridge_stream_process_chunk: rows_processed is NULL")));

	*rows_processed = 0;

	/* Build query with LIMIT and OFFSET for chunking */
	/* Note: For views, we can't use ctid, so we use LIMIT/OFFSET without ORDER BY */
	/* This is non-deterministic but efficient for large datasets */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL "
		"LIMIT %d OFFSET %d",
		quoted_feat,
		quoted_target,
		quoted_tbl,
		quoted_feat,
		quoted_target,
		chunk_size,
		offset);

	elog(DEBUG1, "ridge_stream_process_chunk: query=%s", query.data);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		char *query_str = pstrdup(query.data);
		const char *error_msg;
		
		
		/* Provide more specific error messages for common SPI errors */
		switch (ret)
		{
			case SPI_ERROR_UNCONNECTED:
				error_msg = "SPI not connected";
				break;
			case SPI_ERROR_COPY:
				error_msg = "COPY command in progress (possible nested SPI issue or SPI disconnected)";
				break;
			case SPI_ERROR_TRANSACTION:
				error_msg = "transaction state error";
				break;
			case SPI_ERROR_ARGUMENT:
				error_msg = "invalid argument";
				break;
			default:
				error_msg = "unknown SPI error";
				break;
		}
		
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: ridge_stream_process_chunk: query failed (ret=%d, %s)",
					ret, error_msg),
				errhint("Query: %s. Ensure SPI is connected and no COPY command is in progress.", query_str)));
		NDB_SAFE_PFREE_AND_NULL(query_str);
	}

	n_rows = SPI_processed;
	if (n_rows == 0)
	{
		*rows_processed = 0;
		return;
	}

	/* Determine feature type from first row */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
	{
		tupdesc = SPI_tuptable->tupdesc;
		feat_type_oid = SPI_gettypeid(tupdesc, 1);
		if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
			feat_is_array = true;
	}

	/* Allocate temporary buffer for one row */
	row_buffer = (float *)palloc(sizeof(float) * accum->feature_dim);
	if (row_buffer == NULL)
	{
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: ridge_stream_process_chunk: failed to allocate row buffer")));
	}

	/* Process each row in the chunk */
	for (i = 0; i < n_rows; i++)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		Datum feat_datum;
		Datum targ_datum;
		bool feat_null;
		bool targ_null;
		double target;
		Vector *vec;
		ArrayType *arr;
		int j;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		/* Extract features */
		if (feat_is_array)
		{
			arr = DatumGetArrayTypeP(feat_datum);
			if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != accum->feature_dim)
			{
				NDB_SAFE_PFREE_AND_NULL(row_buffer);
				row_buffer = NULL;
				continue; /* Skip inconsistent rows */
			}
			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				float8 *data = (float8 *)ARR_DATA_PTR(arr);
				for (j = 0; j < accum->feature_dim; j++)
					row_buffer[j] = (float)data[j];
			}
			else
			{
				float4 *data = (float4 *)ARR_DATA_PTR(arr);
				memcpy(row_buffer, data, sizeof(float) * accum->feature_dim);
			}
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			if (vec->dim != accum->feature_dim)
			{
				NDB_SAFE_PFREE_AND_NULL(row_buffer);
				row_buffer = NULL;
				continue; /* Skip inconsistent rows */
			}
			memcpy(row_buffer, vec->data, sizeof(float) * accum->feature_dim);
		}

		/* Extract target */
		{
			Oid targ_type = SPI_gettypeid(tupdesc, 2);

			if (targ_type == INT2OID || targ_type == INT4OID || targ_type == INT8OID)
				target = (double)DatumGetInt32(targ_datum);
			else
				target = DatumGetFloat8(targ_datum);
		}

		/* Add row to accumulator */
		ridge_stream_accum_add_row(accum, row_buffer, target);
		(*rows_processed)++;
	}

	NDB_SAFE_PFREE_AND_NULL(row_buffer);
	row_buffer = NULL;
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
		elog(DEBUG1, "neurondb: ridge_model_serialize: invalid n_features %d (corrupted model?)",
		     model->n_features);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: ridge_model_serialize: invalid n_features %d (corrupted model?)",
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
		NDB_SAFE_PFREE_AND_NULL(model);
		model = NULL;
		elog(DEBUG1, "neurondb: ridge_model_deserialize: invalid n_features %d in deserialized model (corrupted data?)",
		     model ? model->n_features : 0);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: ridge_model_deserialize: invalid n_features in deserialized model (corrupted data?)")));
	}
	if (model != NULL && (model->n_samples < 0 || model->n_samples > 100000000))
	{
		NDB_SAFE_PFREE_AND_NULL(model);
		model = NULL;
		elog(DEBUG1, "ridge_model_deserialize: invalid n_samples in deserialized model (corrupted data?)");
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: ridge_model_deserialize: invalid n_samples in deserialized model")));
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
		NDB_SAFE_PFREE_AND_NULL(meta_text);
		meta_text = NULL;
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
		NDB_SAFE_PFREE_AND_NULL(payload);
	if (metrics != NULL)
		NDB_SAFE_PFREE_AND_NULL(metrics);
	if (gpu_err != NULL)
		NDB_SAFE_PFREE_AND_NULL(gpu_err);

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
		{
			NDB_SAFE_PFREE_AND_NULL(metrics);
			metrics = NULL;
		}
		return false;
	}

	/* Skip GPU models - they should be handled by GPU prediction */
	if (ridge_metadata_is_gpu(metrics))
	{
		NDB_SAFE_PFREE_AND_NULL(payload);
		payload = NULL;
		if (metrics != NULL)
		{
			NDB_SAFE_PFREE_AND_NULL(metrics);
			metrics = NULL;
		}
		return false;
	}

	*out = ridge_model_deserialize(payload);

	NDB_SAFE_PFREE_AND_NULL(payload);
	payload = NULL;
	if (metrics != NULL)
	{
		NDB_SAFE_PFREE_AND_NULL(metrics);
		metrics = NULL;
	}

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
		NDB_SAFE_PFREE_AND_NULL(dataset->features);
	if (dataset->targets != NULL)
		NDB_SAFE_PFREE_AND_NULL(dataset->targets);
	lasso_dataset_init(dataset);
}

/*
 * lasso_dataset_load_limited
 *
 * Load dataset with LIMIT clause to avoid loading too much data
 * Reuses ridge_dataset_load_limited since they have the same structure
 */
static void
lasso_dataset_load_limited(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	LassoDataset *dataset,
	int max_rows)
{
	ridge_dataset_load_limited(quoted_tbl,
		quoted_feat,
		quoted_target,
		(RidgeDataset *)dataset,
		max_rows);
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
		elog(DEBUG1,
		     "neurondb: lasso_model_serialize: invalid n_features %d (corrupted model?)",
		     model->n_features);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: lasso_model_serialize: invalid n_features %d (corrupted model?)",
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
		NDB_SAFE_PFREE_AND_NULL(model);
		model = NULL;
		elog(DEBUG1,
		     "neurondb: lasso_model_deserialize: invalid n_features in deserialized model (corrupted data?)");
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: lasso_model_deserialize: invalid n_features in deserialized model (corrupted data?)")));
	}
	if (model != NULL && (model->n_samples < 0 || model->n_samples > 100000000))
	{
		NDB_SAFE_PFREE_AND_NULL(model);
		model = NULL;
		elog(DEBUG1, "lasso_model_deserialize: invalid n_samples in deserialized model (corrupted data?)");
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: lasso_model_deserialize: invalid n_samples in deserialized model")));
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
		NDB_SAFE_PFREE_AND_NULL(meta_text);
		meta_text = NULL;
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
		NDB_SAFE_PFREE_AND_NULL(payload);
	if (metrics != NULL)
		NDB_SAFE_PFREE_AND_NULL(metrics);
	if (gpu_err != NULL)
		NDB_SAFE_PFREE_AND_NULL(gpu_err);

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
			NDB_SAFE_PFREE_AND_NULL(metrics);
		return false;
	}

	/* Skip GPU models - they should be handled by GPU prediction */
	if (lasso_metadata_is_gpu(metrics))
	{
		NDB_SAFE_PFREE_AND_NULL(payload);
		if (metrics != NULL)
			NDB_SAFE_PFREE_AND_NULL(metrics);
		return false;
	}

	*out = lasso_model_deserialize(payload);

	NDB_SAFE_PFREE_AND_NULL(payload);
	if (metrics != NULL)
		NDB_SAFE_PFREE_AND_NULL(metrics);

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
	{
		elog(DEBUG1,
		     "neurondb: train_ridge_regression: lambda must be non-negative, got %.6f",
		     lambda);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: train_ridge_regression: lambda must be non-negative, got %.6f",
					lambda)));
	}

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_target = quote_identifier(targ_str);

	/* First, determine feature dimension and row count without loading all data */
	{
		StringInfoData count_query;
		int ret;
		Oid feat_type_oid = InvalidOid;
		bool feat_is_array = false;

		if (SPI_connect() != SPI_OK_CONNECT)
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("neurondb: train_ridge_regression: SPI_connect failed")));

		/* Get feature dimension from first row */
		initStringInfo(&count_query);
		appendStringInfo(&count_query,
			"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT 1",
			quoted_feat,
			quoted_target,
			quoted_tbl,
			quoted_feat,
			quoted_target);
		elog(DEBUG1,
			"neurondb: train_ridge_regression: getting feature dimension: %s",
			count_query.data);

		ret = ndb_spi_execute_safe(count_query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			NDB_SAFE_PFREE_AND_NULL(count_query.data);
			SPI_finish();
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: train_ridge_regression: no valid rows found")));
		}

		/* Determine feature dimension */
		if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		{
			HeapTuple first_tuple = SPI_tuptable->vals[0];
			TupleDesc tupdesc = SPI_tuptable->tupdesc;
			Datum feat_datum;
			bool feat_null;

			feat_type_oid = SPI_gettypeid(tupdesc, 1);
			if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
				feat_is_array = true;

			feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &feat_null);
			if (!feat_null)
			{
				if (feat_is_array)
				{
					ArrayType *arr = DatumGetArrayTypeP(feat_datum);

					if (ARR_NDIM(arr) != 1)
					{
						NDB_SAFE_PFREE_AND_NULL(count_query.data);
						SPI_finish();
						NDB_SAFE_PFREE_AND_NULL(tbl_str);
						NDB_SAFE_PFREE_AND_NULL(feat_str);
						NDB_SAFE_PFREE_AND_NULL(targ_str);
						ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								errmsg("neurondb: train_ridge_regression: features array must be 1-D")));
					}
					dim = ARR_DIMS(arr)[0];
				}
				else
				{
					Vector *vec = DatumGetVector(feat_datum);

					dim = vec->dim;
				}
			}
		}

		if (dim <= 0)
		{
			NDB_SAFE_PFREE_AND_NULL(count_query.data);
			SPI_finish();
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: train_ridge_regression: could not determine feature dimension")));
		}

		/* Get row count */
		NDB_SAFE_PFREE_AND_NULL(count_query.data);
		initStringInfo(&count_query);
		appendStringInfo(&count_query,
			"SELECT COUNT(*) FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
			quoted_tbl,
			quoted_feat,
			quoted_target);
		elog(DEBUG1,
			"neurondb: train_ridge_regression: counting rows: %s",
			count_query.data);

		ret = ndb_spi_execute_safe(count_query.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool count_null;
			HeapTuple tuple = SPI_tuptable->vals[0];
			Datum count_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, &count_null);

			if (!count_null)
				nvec = DatumGetInt32(count_datum);
		}

		NDB_SAFE_PFREE_AND_NULL(count_query.data);
		SPI_finish();

		if (nvec < 10)
		{
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: train_ridge_regression: need at least 10 samples, got %d",
						nvec)));
		}
	}

	/* Define max_samples limit for large datasets */
	{
		int max_samples = 500000; /* Limit to 500k samples for very large datasets */

		/* Limit sample size for very large datasets to avoid excessive training time */
		if (nvec > max_samples)
		{
			elog(DEBUG1,
			     "neurondb: ridge_regression: dataset has %d rows, limiting to %d samples for training",
			     nvec,
			     max_samples);
			elog(INFO,
			     "neurondb: ridge_regression: dataset has %d rows, limiting to %d samples for training",
			     nvec,
			     max_samples);
			nvec = max_samples;
		}

		/* Try GPU training first - always use GPU when enabled and kernel available */
		/* Initialize GPU if needed (lazy initialization) */
		if (neurondb_gpu_enabled)
		{
			ndb_gpu_init_if_needed();

			elog(DEBUG1,
				"neurondb: ridge_regression: checking GPU - enabled=%d, available=%d, kernel_enabled=%d",
			neurondb_gpu_enabled ? 1 : 0,
			neurondb_gpu_is_available() ? 1 : 0,
			ndb_gpu_kernel_enabled("ridge_train") ? 1 : 0);
		}

		if (neurondb_gpu_is_available() && nvec > 0 && dim > 0
			&& ndb_gpu_kernel_enabled("ridge_train"))
		{
			int gpu_sample_limit = nvec;

			elog(DEBUG1,
			     "neurondb: ridge_regression: attempting GPU training with %d samples (kernel enabled)",
			     gpu_sample_limit);
			elog(INFO,
			     "neurondb: ridge_regression: attempting GPU training with %d samples (kernel enabled)",
			     gpu_sample_limit);

			/* Load limited dataset for GPU training */
			ridge_dataset_init(&dataset);
			ridge_dataset_load_limited(quoted_tbl,
				quoted_feat,
				quoted_target,
				&dataset,
				gpu_sample_limit);

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
					dataset.n_samples,
					dataset.feature_dim,
					0,
					&gpu_result,
					&gpu_err)
				&& gpu_result.spec.model_data != NULL)
			{
				MLCatalogModelSpec spec;

				elog(INFO,
				     "neurondb: ridge_regression: GPU training succeeded");
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
					NDB_SAFE_PFREE_AND_NULL(gpu_err);
				if (gpu_hyperparams != NULL)
					NDB_SAFE_PFREE_AND_NULL(gpu_hyperparams);
				ndb_gpu_free_train_result(&gpu_result);
				ridge_dataset_free(&dataset);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);

				PG_RETURN_INT32(model_id);
			} else
			{
				/* GPU training failed - log reason and fall back to CPU */
				if (gpu_err != NULL)
				{
						elog(DEBUG1,
							"neurondb: ridge_regression: GPU training failed: %s, falling back to CPU streaming",
						gpu_err);
					NDB_SAFE_PFREE_AND_NULL(gpu_err);
				}
				else
				{
					elog(INFO,
					     "neurondb: ridge_regression: GPU training not available (backend may not support ridge_train), falling back to CPU streaming");
				}
				if (gpu_hyperparams != NULL)
					NDB_SAFE_PFREE_AND_NULL(gpu_hyperparams);
				ndb_gpu_free_train_result(&gpu_result);
				ridge_dataset_free(&dataset);
			}
		} else if (neurondb_gpu_is_available() && !ndb_gpu_kernel_enabled("ridge_train"))
		{
			elog(INFO,
			     "neurondb: ridge_regression: GPU available but ridge_train kernel not enabled, using CPU streaming");
		}

		/* CPU training path using streaming accumulator */
		{
			RidgeStreamAccum stream_accum;
			double **XtX_inv = NULL;
			double *beta = NULL;
			int i, j;
			int dim_with_intercept;
			RidgeModel *model;
			bytea *model_blob;
			Jsonb *metrics_json;
			StringInfoData metricsbuf;
			int chunk_size;
			int offset = 0;
			int rows_in_chunk = 0;
			int spi_ret;

			/* Ensure SPI is connected for CPU streaming path */
			/* (ridge_dataset_load_limited may have disconnected SPI) */
			if ((spi_ret = SPI_connect()) != SPI_OK_CONNECT)
			{
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("neurondb: train_ridge_regression: SPI_connect failed for CPU streaming (ret=%d)",
							spi_ret)));
			}

			/* Use larger chunks for better performance */
			if (nvec > 1000000)
				chunk_size = 100000; /* 100k chunks for very large datasets */
			else if (nvec > 100000)
				chunk_size = 50000; /* 50k chunks for large datasets */
			else
				chunk_size = 10000; /* 10k chunks for smaller datasets */

			/* Initialize streaming accumulator */
			ridge_stream_accum_init(&stream_accum, dim);
			dim_with_intercept = dim + 1;

			/* Process data in chunks */
				elog(DEBUG1,
					"neurondb: ridge_regression: processing %d samples in chunks of %d",
				nvec,
				chunk_size);

			while (offset < nvec)
			{
				ridge_stream_process_chunk(quoted_tbl,
					quoted_feat,
					quoted_target,
					&stream_accum,
					chunk_size,
					offset,
					&rows_in_chunk);

				if (rows_in_chunk == 0)
					break; /* No more rows */

				offset += rows_in_chunk;

				if (offset % (chunk_size * 10) == 0)
						elog(DEBUG1,
							"neurondb: ridge_regression: processed %d/%d samples",
						offset,
						nvec);
			}

			if (stream_accum.n_samples < 10)
			{
				ridge_stream_accum_free(&stream_accum);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: train_ridge_regression: need at least 10 samples, got %d",
							stream_accum.n_samples)));
			}

			/* Allocate matrices for normal equations: β = (X'X + λI)^(-1)X'y */
			XtX_inv = (double **)palloc(sizeof(double *) * dim_with_intercept);
			for (i = 0; i < dim_with_intercept; i++)
				XtX_inv[i] = (double *)palloc(sizeof(double) * dim_with_intercept);
			beta = (double *)palloc(sizeof(double) * dim_with_intercept);

			/* Add Ridge penalty (λI) to diagonal (excluding intercept) */
			for (i = 1; i < dim_with_intercept; i++)
				stream_accum.XtX[i][i] += lambda;

			/* Invert X'X + λI */
			if (!matrix_invert(stream_accum.XtX, dim_with_intercept, XtX_inv))
			{
				for (i = 0; i < dim_with_intercept; i++)
					NDB_SAFE_PFREE_AND_NULL(XtX_inv[i]);
				NDB_SAFE_PFREE_AND_NULL(XtX_inv);
				NDB_SAFE_PFREE_AND_NULL(beta);
				ridge_stream_accum_free(&stream_accum);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: train_ridge_regression: matrix is singular, "
						       "cannot compute Ridge regression"),
						errhint("Try increasing lambda or "
							"removing correlated "
							"features")));
			}

			/* Compute β = (X'X + λI)^(-1)X'y */
			for (i = 0; i < dim_with_intercept; i++)
			{
				beta[i] = 0.0;
				for (j = 0; j < dim_with_intercept; j++)
					beta[i] += XtX_inv[i][j] * stream_accum.Xty[j];
			}

			/* Build RidgeModel */
			model = (RidgeModel *)palloc0(sizeof(RidgeModel));
			model->n_features = dim;
			model->n_samples = stream_accum.n_samples;
			model->intercept = beta[0];
			model->lambda = lambda;
			model->coefficients = (double *)palloc(sizeof(double) * dim);
			for (i = 0; i < dim; i++)
				model->coefficients[i] = beta[i + 1];

			/* Compute metrics (R², MSE, MAE) using streaming accumulator statistics */
			{
				double ss_tot;
				double ss_res = 0.0;
				double mse = 0.0;
				double mae = 0.0;
				int metrics_chunk_size;
				int metrics_offset = 0;

				/* Compute ss_tot from accumulator */
				ss_tot = stream_accum.y_sq_sum - (stream_accum.y_sum * stream_accum.y_sum / stream_accum.n_samples);

				/* Compute MSE and MAE by processing chunks for metrics */
				/* Limit metrics computation to avoid excessive time */
				metrics_chunk_size = (stream_accum.n_samples > 100000) ? 100000 : stream_accum.n_samples;

					elog(DEBUG1,
						"neurondb: ridge_regression: computing metrics on %d samples",
					metrics_chunk_size);

				/* Connect to SPI for metrics computation */
				if (SPI_connect() != SPI_OK_CONNECT)
				{
					elog(WARNING,
					     "neurondb: train_ridge_regression: SPI_connect failed for metrics, skipping detailed metrics");
					mse = 0.0;
					mae = 0.0;
					model->r_squared = 0.0;
				}
				else
				{
					while (metrics_offset < metrics_chunk_size)
					{
						StringInfoData metrics_query;
						int metrics_ret;
						int metrics_n_rows;
						TupleDesc metrics_tupdesc;
						float *metrics_row_buffer = NULL;
						int metrics_i;

						initStringInfo(&metrics_query);
						elog(DEBUG1,
						     "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %d OFFSET %d",
						     quoted_feat,
						     quoted_target,
						     quoted_tbl,
						     quoted_feat,
						     quoted_target,
						     10000,
						     metrics_offset);
						appendStringInfo(&metrics_query,
							"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %d OFFSET %d",
							quoted_feat,
							quoted_target,
							quoted_tbl,
							quoted_feat,
							quoted_target,
							10000,
							metrics_offset);

						metrics_ret = ndb_spi_execute_safe(metrics_query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
						if (metrics_ret != SPI_OK_SELECT)
						{
							NDB_SAFE_PFREE_AND_NULL(metrics_query.data);
							break;
						}
						NDB_CHECK_SPI_TUPTABLE();
						metrics_n_rows = SPI_processed;
						if (metrics_n_rows == 0)
						{
							NDB_SAFE_PFREE_AND_NULL(metrics_query.data);
							break;
						}

						metrics_tupdesc = SPI_tuptable->tupdesc;
						metrics_row_buffer = (float *)palloc(sizeof(float) * dim);

						for (metrics_i = 0; metrics_i < metrics_n_rows && (metrics_offset + metrics_i) < metrics_chunk_size; metrics_i++)
					{
						HeapTuple metrics_tuple = SPI_tuptable->vals[metrics_i];
						Datum metrics_feat_datum;
						Datum metrics_targ_datum;
						bool metrics_feat_null;
						bool metrics_targ_null;
						double metrics_y_true;
						double metrics_y_pred;
						double metrics_error;
						Vector *metrics_vec;
						ArrayType *metrics_arr;
						int metrics_j;
						Oid metrics_feat_type_oid = SPI_gettypeid(metrics_tupdesc, 1);
						bool metrics_feat_is_array = (metrics_feat_type_oid == FLOAT8ARRAYOID || metrics_feat_type_oid == FLOAT4ARRAYOID);

						metrics_feat_datum = SPI_getbinval(metrics_tuple, metrics_tupdesc, 1, &metrics_feat_null);
						metrics_targ_datum = SPI_getbinval(metrics_tuple, metrics_tupdesc, 2, &metrics_targ_null);

						if (metrics_feat_null || metrics_targ_null)
							continue;

						/* Extract features */
						if (metrics_feat_is_array)
						{
							metrics_arr = DatumGetArrayTypeP(metrics_feat_datum);
							if (ARR_NDIM(metrics_arr) != 1 || ARR_DIMS(metrics_arr)[0] != dim)
								continue;
							if (metrics_feat_type_oid == FLOAT8ARRAYOID)
							{
								float8 *data = (float8 *)ARR_DATA_PTR(metrics_arr);
								for (metrics_j = 0; metrics_j < dim; metrics_j++)
									metrics_row_buffer[metrics_j] = (float)data[metrics_j];
							}
							else
							{
								float4 *data = (float4 *)ARR_DATA_PTR(metrics_arr);
								memcpy(metrics_row_buffer, data, sizeof(float) * dim);
							}
						}
						else
						{
							metrics_vec = DatumGetVector(metrics_feat_datum);
							if (metrics_vec->dim != dim)
								continue;
							memcpy(metrics_row_buffer, metrics_vec->data, sizeof(float) * dim);
						}

						/* Extract target */
						{
							Oid metrics_targ_type = SPI_gettypeid(metrics_tupdesc, 2);

							if (metrics_targ_type == INT2OID || metrics_targ_type == INT4OID || metrics_targ_type == INT8OID)
								metrics_y_true = (double)DatumGetInt32(metrics_targ_datum);
							else
								metrics_y_true = DatumGetFloat8(metrics_targ_datum);
						}

						/* Compute prediction */
						metrics_y_pred = model->intercept;
						for (metrics_j = 0; metrics_j < dim; metrics_j++)
							metrics_y_pred += model->coefficients[metrics_j] * metrics_row_buffer[metrics_j];

						/* Accumulate errors */
						metrics_error = metrics_y_true - metrics_y_pred;
						mse += metrics_error * metrics_error;
						mae += fabs(metrics_error);
						ss_res += metrics_error * metrics_error;
					}

						NDB_SAFE_PFREE_AND_NULL(metrics_row_buffer);
						NDB_SAFE_PFREE_AND_NULL(metrics_query.data);

						metrics_offset += metrics_n_rows;
						if (metrics_offset >= metrics_chunk_size)
							break;
					}

					SPI_finish();

					/* Normalize metrics */
					if (metrics_chunk_size > 0)
					{
						mse /= metrics_chunk_size;
						mae /= metrics_chunk_size;
					}
				}

				/* Compute R² */
				if (ss_tot > 1e-10)
					model->r_squared = 1.0 - (ss_res / ss_tot);
				else
					model->r_squared = 0.0;
				model->mse = mse;
				model->mae = mae;
			}

			/* Validate model before serialization */
			if (model->n_features <= 0 || model->n_features > 10000)
			{
				if (model->coefficients != NULL)
				{
					NDB_SAFE_PFREE_AND_NULL(model->coefficients);
					model->coefficients = NULL;
				}
				NDB_SAFE_PFREE_AND_NULL(model);
				model = NULL;
				for (i = 0; i < dim_with_intercept; i++)
				{
					NDB_SAFE_PFREE_AND_NULL(XtX_inv[i]);
					XtX_inv[i] = NULL;
				}
				NDB_SAFE_PFREE_AND_NULL(XtX_inv);
				XtX_inv = NULL;
				NDB_SAFE_PFREE_AND_NULL(beta);
				beta = NULL;
				ridge_stream_accum_free(&stream_accum);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				elog(DEBUG1,
				     "neurondb: train_ridge_regression: model.n_features is invalid (%d) before serialization",
				     model->n_features);
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("neurondb: train_ridge_regression: model.n_features is invalid (%d) before serialization",
							model->n_features)));
			}

			elog(DEBUG1,
			     "neurondb: ridge_regression: serializing model with n_features=%d, n_samples=%d, lambda=%.6f",
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
				NDB_SAFE_PFREE_AND_NULL(XtX_inv[i]);
				XtX_inv[i] = NULL;
			}
			NDB_SAFE_PFREE_AND_NULL(XtX_inv);
			XtX_inv = NULL;
			NDB_SAFE_PFREE_AND_NULL(beta);
			beta = NULL;
			if (model->coefficients != NULL)
			{
				NDB_SAFE_PFREE_AND_NULL(model->coefficients);
				model->coefficients = NULL;
			}
			NDB_SAFE_PFREE_AND_NULL(model);
			model = NULL;
			ridge_stream_accum_free(&stream_accum);
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			
			/* Finish SPI connection before returning */
			SPI_finish();

			PG_RETURN_INT32(model_id);
		}
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
 NDB_CHECK_VECTOR_VALID(features);

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
		     "ridge: GPU prediction failed or not available, trying CPU");
	}

	/* Load model from catalog */
	if (!ridge_load_model_from_catalog(model_id, &model))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("ridge: model %d not found", model_id)));

	/* Validate feature dimension */
	if (model->n_features > 0 && features->dim != model->n_features)
	{
		elog(DEBUG1,
		     "ridge: feature dimension mismatch (expected %d, got %d)",
		     model->n_features,
		     features->dim);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("ridge: feature dimension mismatch (expected %d, got %d)",
					model->n_features,
					features->dim)));
	}

	/* Compute prediction: y = intercept + coef1*x1 + coef2*x2 + ... */
	prediction = model->intercept;
	for (i = 0; i < model->n_features && i < features->dim; i++)
		prediction += model->coefficients[i] * features->data[i];

	/* Cleanup */
	if (model != NULL)
	{
		if (model->coefficients != NULL)
			NDB_SAFE_PFREE_AND_NULL(model->coefficients);
		NDB_SAFE_PFREE_AND_NULL(model);
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
	{
		elog(DEBUG1,
		     "neurondb: train_lasso_regression: lambda must be non-negative, got %.6f",
		     lambda);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: train_lasso_regression: lambda must be non-negative, got %.6f",
					lambda)));
	}
	if (max_iters <= 0)
	{
		elog(DEBUG1,
		     "neurondb: train_lasso_regression: max_iters must be positive, got %d",
		     max_iters);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: train_lasso_regression: max_iters must be positive, got %d",
					max_iters)));
	}

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_target = quote_identifier(targ_str);

	/* First, determine feature dimension and row count without loading all data */
	{
		StringInfoData count_query;
		int ret;
		Oid feat_type_oid = InvalidOid;
		bool feat_is_array = false;

		if (SPI_connect() != SPI_OK_CONNECT)
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("neurondb: train_lasso_regression: SPI_connect failed")));

		/* Get feature dimension from first row */
		initStringInfo(&count_query);
		appendStringInfo(&count_query,
			"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT 1",
			quoted_feat,
			quoted_target,
			quoted_tbl,
			quoted_feat,
			quoted_target);
		elog(DEBUG1,
			"neurondb: train_lasso_regression: getting feature dimension: %s",
			count_query.data);

		ret = ndb_spi_execute_safe(count_query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			NDB_SAFE_PFREE_AND_NULL(count_query.data);
			SPI_finish();
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: train_lasso_regression: no valid rows found")));
		}

		/* Determine feature dimension */
		if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		{
			HeapTuple first_tuple = SPI_tuptable->vals[0];
			TupleDesc tupdesc = SPI_tuptable->tupdesc;
			Datum feat_datum;
			bool feat_null;

			feat_type_oid = SPI_gettypeid(tupdesc, 1);
			if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
				feat_is_array = true;

			feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &feat_null);
			if (!feat_null)
			{
				if (feat_is_array)
				{
					ArrayType *arr = DatumGetArrayTypeP(feat_datum);

					if (ARR_NDIM(arr) != 1)
					{
						NDB_SAFE_PFREE_AND_NULL(count_query.data);
						SPI_finish();
						NDB_SAFE_PFREE_AND_NULL(tbl_str);
						NDB_SAFE_PFREE_AND_NULL(feat_str);
						NDB_SAFE_PFREE_AND_NULL(targ_str);
						ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								errmsg("neurondb: train_lasso_regression: features array must be 1-D")));
					}
					dim = ARR_DIMS(arr)[0];
				}
				else
				{
					Vector *vec = DatumGetVector(feat_datum);

					dim = vec->dim;
				}
			}
		}

		if (dim <= 0)
		{
			NDB_SAFE_PFREE_AND_NULL(count_query.data);
			SPI_finish();
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: train_lasso_regression: could not determine feature dimension")));
		}

		/* Get row count */
		NDB_SAFE_PFREE_AND_NULL(count_query.data);
		initStringInfo(&count_query);
		appendStringInfo(&count_query,
			"SELECT COUNT(*) FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
			quoted_tbl,
			quoted_feat,
			quoted_target);
		elog(DEBUG1,
			"neurondb: train_lasso_regression: counting rows: %s",
			count_query.data);

		ret = ndb_spi_execute_safe(count_query.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool count_null;
			HeapTuple tuple = SPI_tuptable->vals[0];
			Datum count_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, &count_null);

			if (!count_null)
				nvec = DatumGetInt32(count_datum);
		}

		NDB_SAFE_PFREE_AND_NULL(count_query.data);
		SPI_finish();

		if (nvec < 10)
		{
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: train_lasso_regression: need at least 10 samples, got %d",
						nvec)));
		}
	}

	/* Define max_samples limit for large datasets */
	{
		int max_samples = 500000; /* Limit to 500k samples for very large datasets */

		/* Limit sample size for very large datasets to avoid excessive training time */
		if (nvec > max_samples)
		{
			elog(DEBUG1,
			     "neurondb: lasso_regression: dataset has %d rows, limiting to %d samples for training",
			     nvec,
			     max_samples);
			elog(INFO,
			     "neurondb: lasso_regression: dataset has %d rows, limiting to %d samples for training",
			     nvec,
			     max_samples);
			nvec = max_samples;
		}

		/* Try GPU training first - always use GPU when enabled and kernel available */
		/* Initialize GPU if needed (lazy initialization) */
		if (neurondb_gpu_enabled)
		{
			ndb_gpu_init_if_needed();

			elog(INFO,
				"neurondb: lasso_regression: checking GPU - enabled=%d, available=%d, kernel_enabled=%d",
			neurondb_gpu_enabled ? 1 : 0,
			neurondb_gpu_is_available() ? 1 : 0,
			ndb_gpu_kernel_enabled("lasso_train") ? 1 : 0);
		}

		if (neurondb_gpu_is_available() && nvec > 0 && dim > 0
			&& ndb_gpu_kernel_enabled("lasso_train"))
		{
			int gpu_sample_limit = nvec;

			elog(DEBUG1,
			     "neurondb: lasso_regression: attempting GPU training with %d samples (kernel enabled)",
			     gpu_sample_limit);
			elog(INFO,
			     "neurondb: lasso_regression: attempting GPU training with %d samples (kernel enabled)",
			     gpu_sample_limit);

			/* Load limited dataset for GPU training */
			lasso_dataset_init(&dataset);
			lasso_dataset_load_limited(quoted_tbl,
				quoted_feat,
				quoted_target,
				&dataset,
				gpu_sample_limit);

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
					dataset.n_samples,
					dataset.feature_dim,
					0,
					&gpu_result,
					&gpu_err)
				&& gpu_result.spec.model_data != NULL)
			{
				MLCatalogModelSpec spec;

				elog(INFO,
				     "neurondb: lasso_regression: GPU training succeeded");
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
					NDB_SAFE_PFREE_AND_NULL(gpu_err);
				if (gpu_hyperparams != NULL)
					NDB_SAFE_PFREE_AND_NULL(gpu_hyperparams);
				ndb_gpu_free_train_result(&gpu_result);
				lasso_dataset_free(&dataset);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);

				PG_RETURN_INT32(model_id);
			} else
			{
				elog(INFO,
				     "neurondb: lasso_regression: GPU training failed, falling back to CPU");
				if (gpu_err != NULL)
				{
					elog(INFO,
					     "neurondb: lasso_regression: GPU training error: %s",
					     gpu_err);
					NDB_SAFE_PFREE_AND_NULL(gpu_err);
				}
				else
				{
					elog(INFO,
					     "neurondb: lasso_regression: GPU training failed with no error message (check GPU backend and kernel availability)");
				}
				if (gpu_hyperparams != NULL)
					NDB_SAFE_PFREE_AND_NULL(gpu_hyperparams);
				ndb_gpu_free_train_result(&gpu_result);
				lasso_dataset_free(&dataset);
			}
		} else if (neurondb_gpu_is_available() && !ndb_gpu_kernel_enabled("lasso_train"))
		{
			elog(INFO,
			     "neurondb: lasso_regression: GPU available but lasso_train kernel not enabled, using CPU");
		}

		/* CPU training path - use limited dataset loading */
		/* For CPU training, use a smaller limit to avoid excessive training time */
		{
			int cpu_sample_limit = nvec;
			if (cpu_sample_limit > 100000)
			{
				elog(DEBUG1,
				     "neurondb: lasso_regression: CPU training on large dataset (%d rows), limiting to 100000 samples for reasonable training time",
				     cpu_sample_limit);
				elog(INFO,
				     "neurondb: lasso_regression: CPU training on large dataset (%d rows), limiting to 100000 samples for reasonable training time",
				     cpu_sample_limit);
				cpu_sample_limit = 100000;
			}
			
			lasso_dataset_init(&dataset);
			lasso_dataset_load_limited(quoted_tbl,
				quoted_feat,
				quoted_target,
				&dataset,
				cpu_sample_limit);

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

				nvec = dataset.n_samples;
				dim = dataset.feature_dim;

				if (nvec < 10)
				{
					lasso_dataset_free(&dataset);
					NDB_SAFE_PFREE_AND_NULL(tbl_str);
					NDB_SAFE_PFREE_AND_NULL(feat_str);
					NDB_SAFE_PFREE_AND_NULL(targ_str);
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("neurondb: train_lasso_regression: need at least 10 samples, got %d",
								nvec)));
				}

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
								"neurondb: lasso_regression: converged after %d iterations",
							iter + 1);
					}
				}

			if (!converged)
			{
					elog(DEBUG1,
						"neurondb: lasso_regression: did not converge after %d iterations",
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
				{
					NDB_SAFE_PFREE_AND_NULL(model->coefficients);
					model->coefficients = NULL;
				}
				NDB_SAFE_PFREE_AND_NULL(model);
				model = NULL;
				NDB_SAFE_PFREE_AND_NULL(weights);
				weights = NULL;
				NDB_SAFE_PFREE_AND_NULL(weights_old);
				weights_old = NULL;
				NDB_SAFE_PFREE_AND_NULL(residuals);
				residuals = NULL;
				lasso_dataset_free(&dataset);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				elog(DEBUG1,
				     "neurondb: train_lasso_regression: model.n_features is invalid (%d) before serialization",
				     model->n_features);
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("neurondb: train_lasso_regression: model.n_features is invalid (%d) before serialization",
							model->n_features)));
			}

			elog(DEBUG1,
			     "neurondb: lasso_regression: serializing model with n_features=%d, n_samples=%d, lambda=%.6f",
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
			NDB_SAFE_PFREE_AND_NULL(weights);
			weights = NULL;
			NDB_SAFE_PFREE_AND_NULL(weights_old);
			weights_old = NULL;
			NDB_SAFE_PFREE_AND_NULL(residuals);
			residuals = NULL;
			if (model->coefficients != NULL)
			{
				NDB_SAFE_PFREE_AND_NULL(model->coefficients);
				model->coefficients = NULL;
			}
			NDB_SAFE_PFREE_AND_NULL(model);
			model = NULL;
			lasso_dataset_free(&dataset);
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);

			PG_RETURN_INT32(model_id);
		}
	}
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
				errmsg("neurondb: predict_lasso_regression_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: predict_lasso_regression_model_id: features vector is required")));

	features = PG_GETARG_VECTOR_P(1);
 NDB_CHECK_VECTOR_VALID(features);

	/* Try GPU prediction first */
	if (lasso_try_gpu_predict_catalog(model_id, features, &prediction))
	{
			elog(DEBUG1,
				"neurondb: lasso_regression: GPU prediction succeeded, prediction=%.6f",
			prediction);
		PG_RETURN_FLOAT8(prediction);
	} else
	{
		elog(DEBUG1,
		     "neurondb: lasso_regression: GPU prediction failed or not available, trying CPU");
	}

	/* Load model from catalog */
	if (!lasso_load_model_from_catalog(model_id, &model))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: predict_lasso_regression_model_id: model %d not found", model_id)));

	/* Validate feature dimension */
	if (model->n_features > 0 && features->dim != model->n_features)
	{
		elog(DEBUG1,
		     "neurondb: predict_lasso_regression_model_id: feature dimension mismatch (expected %d, got %d)",
		     model->n_features,
		     features->dim);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: predict_lasso_regression_model_id: feature dimension mismatch (expected %d, got %d)",
					model->n_features,
					features->dim)));
	}

	/* Compute prediction: y = intercept + coef1*x1 + coef2*x2 + ... */
	prediction = model->intercept;
	for (i = 0; i < model->n_features && i < features->dim; i++)
		prediction += model->coefficients[i] * features->data[i];

	/* Cleanup */
	if (model != NULL)
	{
		if (model->coefficients != NULL)
			NDB_SAFE_PFREE_AND_NULL(model->coefficients);
		NDB_SAFE_PFREE_AND_NULL(model);
	}

	PG_RETURN_FLOAT8(prediction);
}

/*
 * evaluate_ridge_regression_by_model_id
 *
 * Evaluates Ridge Regression model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: mse, mae, rmse, r_squared, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_ridge_regression_by_model_id);

Datum
evaluate_ridge_regression_by_model_id(PG_FUNCTION_ARGS)
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
	Oid feat_type_oid = InvalidOid;
	bool feat_is_array = false;
	double mse = 0.0;
	double mae = 0.0;
	double rmse = 0.0;
	double r_squared = 0.0;
	double y_mean = 0.0;
	MemoryContext oldcontext;
	StringInfoData query;
	RidgeModel *model = NULL;
	StringInfoData jsonbuf;
	Jsonb *result_jsonb = NULL;
	bytea *gpu_payload = NULL;
	Jsonb *gpu_metrics = NULL;
	bool is_gpu_model = false;
	float *h_features = NULL;
	double *h_targets = NULL;
	int valid_rows = 0;
	double ss_res = 0.0;
	double ss_tot = 0.0;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_ridge_regression_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_ridge_regression_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Load model from catalog - try CPU first, then GPU */
	if (!ridge_load_model_from_catalog(model_id, &model))
	{
		/* Try GPU model */
		if (ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
		{
			is_gpu_model = ridge_metadata_is_gpu(gpu_metrics);
			if (!is_gpu_model)
			{
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_ridge_regression_by_model_id: model %d not found",
							model_id)));
			}
		}
		else
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_ridge_regression_by_model_id: model %d not found",
						model_id)));
		}
	}

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
	{
		if (model != NULL)
		{
			if (model->coefficients != NULL)
			{
				NDB_SAFE_PFREE_AND_NULL(model->coefficients);
				model->coefficients = NULL;
			}
			NDB_SAFE_PFREE_AND_NULL(model);
			model = NULL;
		}
		if (tbl_str)
		{
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			targ_str = NULL;
		}
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_ridge_regression_by_model_id: SPI_connect failed")));
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
	elog(DEBUG1, "evaluate_ridge_regression_by_model_id: executing query: %s", query.data);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		if (model != NULL)
		{
			if (model->coefficients != NULL)
			{
				NDB_SAFE_PFREE_AND_NULL(model->coefficients);
				model->coefficients = NULL;
			}
			NDB_SAFE_PFREE_AND_NULL(model);
			model = NULL;
		}
		if (tbl_str)
		{
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			targ_str = NULL;
		}
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_ridge_regression_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		if (model != NULL)
		{
			if (model->coefficients != NULL)
			{
				NDB_SAFE_PFREE_AND_NULL(model->coefficients);
				model->coefficients = NULL;
			}
			NDB_SAFE_PFREE_AND_NULL(model);
			model = NULL;
		}
		if (tbl_str)
		{
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			targ_str = NULL;
		}
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_ridge_regression_by_model_id: no valid rows found")));
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
		const NdbCudaRidgeModelHeader *gpu_hdr;
		int feat_dim = 0;

		/* Load GPU model header */
		gpu_hdr = (const NdbCudaRidgeModelHeader *)VARDATA(gpu_payload);
		feat_dim = gpu_hdr->feature_dim;

		/* Allocate host buffers for features and targets */
		h_features = (float *)palloc(sizeof(float) * (size_t)nvec * (size_t)feat_dim);
		h_targets = (double *)palloc(sizeof(double) * (size_t)nvec);

		/* Extract features and targets from SPI results - optimized batch extraction */
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
				h_targets[valid_rows] = DatumGetFloat8(targ_datum);
				y_mean += h_targets[valid_rows];

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
						int j;
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
			NDB_SAFE_PFREE_AND_NULL(h_features);
			NDB_SAFE_PFREE_AND_NULL(h_targets);
			if (gpu_payload)
				NDB_SAFE_PFREE_AND_NULL(gpu_payload);
			if (gpu_metrics)
				NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_ridge_regression_by_model_id: no valid rows found")));
		}

		/* Use optimized GPU evaluation kernel instead of cuBLAS + CPU loop */
		/* This kernel computes predictions and accumulates SSE/SAE in one pass */
		{
			double gpu_mse = 0.0;
			double gpu_mae = 0.0;
			double gpu_rmse = 0.0;
			double gpu_r_squared = 0.0;
			char *gpu_err = NULL;
			int eval_rc;

				elog(DEBUG1,
					"neurondb: evaluate_ridge_regression_by_model_id: using GPU evaluation kernel for %d samples",
				valid_rows);

			eval_rc = ndb_cuda_ridge_evaluate(gpu_payload,
				h_features,
				h_targets,
				valid_rows,
				feat_dim,
				&gpu_mse,
				&gpu_mae,
				&gpu_rmse,
				&gpu_r_squared,
				&gpu_err);

			if (eval_rc == 0)
			{
				/* GPU evaluation succeeded */
				mse = gpu_mse;
				mae = gpu_mae;
				rmse = gpu_rmse;
				r_squared = gpu_r_squared;
				nvec = valid_rows;

				/* Cleanup */
				NDB_SAFE_PFREE_AND_NULL(h_features);
				h_features = NULL;
				NDB_SAFE_PFREE_AND_NULL(h_targets);
				h_targets = NULL;
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
			}
			else
			{
				/* GPU evaluation failed, fall back to CPU */
					elog(DEBUG1,
						"neurondb: evaluate_ridge_regression_by_model_id: GPU evaluation kernel failed: %s, falling back to CPU",
					gpu_err ? gpu_err : "unknown error");
				if (gpu_err)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_err);
					gpu_err = NULL;
				}
				NDB_SAFE_PFREE_AND_NULL(h_features);
				h_features = NULL;
				NDB_SAFE_PFREE_AND_NULL(h_targets);
				h_targets = NULL;
				goto gpu_eval_fallback;
			}
		}

		SPI_finish();
		NDB_SAFE_PFREE_AND_NULL(tbl_str);
		NDB_SAFE_PFREE_AND_NULL(feat_str);
		NDB_SAFE_PFREE_AND_NULL(targ_str);

		/* Build jsonb result */
		initStringInfo(&jsonbuf);
		appendStringInfo(&jsonbuf,
			"{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
			mse,
			mae,
			rmse,
			r_squared,
			nvec);

		result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
			CStringGetDatum(jsonbuf.data)));

		NDB_SAFE_PFREE_AND_NULL(jsonbuf.data);
		jsonbuf.data = NULL;
		MemoryContextSwitchTo(oldcontext);
		PG_RETURN_JSONB_P(result_jsonb);
#endif	/* NDB_GPU_CUDA */
	}

gpu_eval_fallback:
	/* CPU batch evaluation path (also used as fallback for GPU models) */
	/* Use the already-loaded h_features and h_targets arrays for batch evaluation */
	{
		/* Load GPU model into CPU structure for CPU evaluation */
		const NdbCudaRidgeModelHeader *gpu_hdr;
		const float *gpu_coefficients;
		double *cpu_coefficients = NULL;
		double cpu_intercept = 0.0;
		int feat_dim;
		int valid_samples = 0;
		double *predictions = NULL;

		if (is_gpu_model && gpu_payload)
		{
			gpu_hdr = (const NdbCudaRidgeModelHeader *)VARDATA(gpu_payload);
			gpu_coefficients = (const float *)((const char *)gpu_hdr + sizeof(NdbCudaRidgeModelHeader));
			feat_dim = gpu_hdr->feature_dim;
			cpu_intercept = gpu_hdr->intercept;

			/* Convert coefficients from float to double */
			cpu_coefficients = (double *)palloc(sizeof(double) * feat_dim);
			for (i = 0; i < feat_dim; i++)
				cpu_coefficients[i] = (double)gpu_coefficients[i];
		}
		else if (model != NULL)
		{
			cpu_coefficients = model->coefficients;
			cpu_intercept = model->intercept;
			feat_dim = model->n_features;
		}
		else
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_ridge_regression_by_model_id: CPU model evaluation failed")));
		}

		/* Re-execute query to get data for CPU evaluation (since GPU arrays may not be loaded) */
		if (!is_gpu_model || h_features == NULL || h_targets == NULL)
		{
			float *cpu_features = NULL;
			double *cpu_targets = NULL;
			size_t features_size, targets_size;

			/* Re-execute query for CPU batch evaluation */
			ret = ndb_spi_execute_safe(query.data, true, 0);
			NDB_CHECK_SPI_TUPTABLE();
			if (ret != SPI_OK_SELECT)
			{
				if (cpu_coefficients != model->coefficients)
				{
					NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
				}
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
				if (tbl_str)
				{
					NDB_SAFE_PFREE_AND_NULL(tbl_str);
					tbl_str = NULL;
				}
				if (feat_str)
				{
					NDB_SAFE_PFREE_AND_NULL(feat_str);
					feat_str = NULL;
				}
				if (targ_str)
				{
					NDB_SAFE_PFREE_AND_NULL(targ_str);
					targ_str = NULL;
				}
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("neurondb: evaluate_ridge_regression_by_model_id: query failed for CPU batch evaluation")));
			}

			nvec = SPI_processed;
			if (nvec < 2)
			{
				if (cpu_coefficients != model->coefficients)
				{
					NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
					cpu_coefficients = NULL;
				}
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
				if (tbl_str)
				{
					NDB_SAFE_PFREE_AND_NULL(tbl_str);
					tbl_str = NULL;
				}
				if (feat_str)
				{
					NDB_SAFE_PFREE_AND_NULL(feat_str);
					feat_str = NULL;
				}
				if (targ_str)
				{
					NDB_SAFE_PFREE_AND_NULL(targ_str);
					targ_str = NULL;
				}
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_ridge_regression_by_model_id: need at least 2 samples for CPU evaluation, got %d",
							nvec)));
			}

			/* Allocate arrays for CPU batch evaluation */
			features_size = sizeof(float) * (size_t)nvec * (size_t)feat_dim;
			targets_size = sizeof(double) * (size_t)nvec;

			if (features_size > MaxAllocSize || targets_size > MaxAllocSize)
			{
				if (cpu_coefficients != model->coefficients)
				{
					NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
					cpu_coefficients = NULL;
				}
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
				if (tbl_str)
				{
					NDB_SAFE_PFREE_AND_NULL(tbl_str);
					tbl_str = NULL;
				}
				if (feat_str)
				{
					NDB_SAFE_PFREE_AND_NULL(feat_str);
					feat_str = NULL;
				}
				if (targ_str)
				{
					NDB_SAFE_PFREE_AND_NULL(targ_str);
					targ_str = NULL;
				}
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
						errmsg("neurondb: evaluate_ridge_regression_by_model_id: data too large for CPU batch evaluation")));
			}

			cpu_features = (float *)palloc(features_size);
			cpu_targets = (double *)palloc(targets_size);

			/* Extract features and targets in batch */
			for (i = 0; i < nvec; i++)
			{
				HeapTuple tuple = SPI_tuptable->vals[i];
				TupleDesc tupdesc = SPI_tuptable->tupdesc;
				Datum feat_datum, targ_datum;
				bool feat_null, targ_null;
				float *feat_row;
				int k;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				feat_row = cpu_features + (valid_samples * feat_dim);
				cpu_targets[valid_samples] = DatumGetFloat8(targ_datum);

				/* Extract feature vector */
				if (feat_is_array)
				{
					ArrayType *arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;

					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						float8 *data = (float8 *)ARR_DATA_PTR(arr);
						for (k = 0; k < feat_dim; k++)
							feat_row[k] = (float)data[k];
					}
					else
					{
						float4 *data = (float4 *)ARR_DATA_PTR(arr);
						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					Vector *vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_samples++;
				y_mean += cpu_targets[valid_samples - 1];
			}

			if (valid_samples < 2)
			{
				NDB_SAFE_PFREE_AND_NULL(cpu_features);
				cpu_features = NULL;
				NDB_SAFE_PFREE_AND_NULL(cpu_targets);
				cpu_targets = NULL;
				if (cpu_coefficients != model->coefficients)
				{
					NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
					cpu_coefficients = NULL;
				}
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
				if (tbl_str)
				{
					NDB_SAFE_PFREE_AND_NULL(tbl_str);
					tbl_str = NULL;
				}
				if (feat_str)
				{
					NDB_SAFE_PFREE_AND_NULL(feat_str);
					feat_str = NULL;
				}
				if (targ_str)
				{
					NDB_SAFE_PFREE_AND_NULL(targ_str);
					targ_str = NULL;
				}
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_ridge_regression_by_model_id: need at least 2 valid samples for CPU evaluation, got %d",
							valid_samples)));
			}

			h_features = cpu_features;
			h_targets = cpu_targets;
			valid_rows = valid_samples;
		}

		y_mean /= valid_rows;

		/* Allocate predictions array */
		predictions = (double *)palloc(sizeof(double) * valid_rows);

		/* Batch prediction: compute all predictions at once */
		for (i = 0; i < valid_rows; i++)
		{
			const float *feat_row = h_features + (i * feat_dim);
			double prediction = cpu_intercept;
			int k;

			for (k = 0; k < feat_dim; k++)
				prediction += cpu_coefficients[k] * (double)feat_row[k];

			predictions[i] = prediction;
		}

		/* Compute metrics in batch */
		for (i = 0; i < valid_rows; i++)
		{
			double y_true = h_targets[i];
			double y_pred = predictions[i];
			double error = y_true - y_pred;

			mse += error * error;
			mae += fabs(error);
			ss_res += error * error;
			ss_tot += (y_true - y_mean) * (y_true - y_mean);
		}

		mse /= valid_rows;
		mae /= valid_rows;
		rmse = sqrt(mse);

		/* Handle R² calculation */
		if (ss_tot == 0.0)
			r_squared = 0.0;
		else
			r_squared = 1.0 - (ss_res / ss_tot);

		nvec = valid_rows;

		/* Cleanup */
		NDB_SAFE_PFREE_AND_NULL(predictions);
		predictions = NULL;
		NDB_SAFE_PFREE_AND_NULL(h_features);
		h_features = NULL;
		NDB_SAFE_PFREE_AND_NULL(h_targets);
		h_targets = NULL;
		if (cpu_coefficients != model->coefficients)
		{
			NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
			cpu_coefficients = NULL;
		}
	}

	/* Build jsonb result in old context to survive SPI_finish */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
		mse,
		mae,
		rmse,
		r_squared,
		nvec);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
		CStringGetDatum(jsonbuf.data)));

	NDB_SAFE_PFREE_AND_NULL(jsonbuf.data);
	jsonbuf.data = NULL;
	SPI_finish();
	PG_RETURN_JSONB_P(result_jsonb);
}

/*
 * evaluate_lasso_regression_by_model_id
 *
 * Evaluates Lasso Regression model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: mse, mae, rmse, r_squared, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_lasso_regression_by_model_id);

Datum
evaluate_lasso_regression_by_model_id(PG_FUNCTION_ARGS)
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
	Oid feat_type_oid = InvalidOid;
	bool feat_is_array = false;
	double mse = 0.0;
	double mae = 0.0;
	double rmse = 0.0;
	double r_squared = 0.0;
	double y_mean = 0.0;
	MemoryContext oldcontext;
	StringInfoData query;
	LassoModel *model = NULL;
	StringInfoData jsonbuf;
	Jsonb *result_jsonb = NULL;
	bytea *gpu_payload = NULL;
	Jsonb *gpu_metrics = NULL;
	bool is_gpu_model = false;
	float *h_features = NULL;
	double *h_targets = NULL;
	int valid_rows = 0;
	double ss_res = 0.0;
	double ss_tot = 0.0;

	/* Defensive: validate model_id parameter */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_lasso_regression_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	/* Defensive: validate model_id range */
	if (model_id <= 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_lasso_regression_by_model_id: model_id must be positive, got %d",
					model_id)));

	/* Defensive: validate required parameters */
	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_lasso_regression_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	/* Defensive: validate text pointers */
	if (table_name == NULL || feature_col == NULL || label_col == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_lasso_regression_by_model_id: table_name, feature_col, and label_col cannot be NULL")));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	/* Defensive: validate converted strings */
	if (tbl_str == NULL || feat_str == NULL || targ_str == NULL)
	{
		if (tbl_str)
		{
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			targ_str = NULL;
		}
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_lasso_regression_by_model_id: failed to convert text parameters")));
	}

	oldcontext = CurrentMemoryContext;

	/* Load model from catalog - try CPU first, then GPU */
	if (!lasso_load_model_from_catalog(model_id, &model))
	{
		/* Try GPU model */
		if (ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
		{
			is_gpu_model = lasso_metadata_is_gpu(gpu_metrics);
			if (!is_gpu_model)
			{
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
				if (tbl_str)
				{
					NDB_SAFE_PFREE_AND_NULL(tbl_str);
					tbl_str = NULL;
				}
				if (feat_str)
				{
					NDB_SAFE_PFREE_AND_NULL(feat_str);
					feat_str = NULL;
				}
				if (targ_str)
				{
					NDB_SAFE_PFREE_AND_NULL(targ_str);
					targ_str = NULL;
				}
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: model %d not found",
							model_id)));
			}
		}
		else
		{
			if (tbl_str)
			{
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				tbl_str = NULL;
			}
			if (feat_str)
			{
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				feat_str = NULL;
			}
			if (targ_str)
			{
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				targ_str = NULL;
			}
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_lasso_regression_by_model_id: model %d not found",
						model_id)));
		}
	}

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
	{
		if (model != NULL)
		{
			if (model->coefficients != NULL)
			{
				NDB_SAFE_PFREE_AND_NULL(model->coefficients);
				model->coefficients = NULL;
			}
			NDB_SAFE_PFREE_AND_NULL(model);
			model = NULL;
		}
		if (gpu_payload)
		{
			NDB_SAFE_PFREE_AND_NULL(gpu_payload);
			gpu_payload = NULL;
		}
		if (gpu_metrics)
		{
			NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
			gpu_metrics = NULL;
		}
		if (tbl_str)
		{
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			targ_str = NULL;
		}
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_lasso_regression_by_model_id: SPI_connect failed")));
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
	elog(DEBUG1, "evaluate_lasso_regression_by_model_id: executing query: %s", query.data);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		if (model != NULL)
		{
			if (model->coefficients != NULL)
			{
				NDB_SAFE_PFREE_AND_NULL(model->coefficients);
				model->coefficients = NULL;
			}
			NDB_SAFE_PFREE_AND_NULL(model);
			model = NULL;
		}
		if (gpu_payload)
		{
			NDB_SAFE_PFREE_AND_NULL(gpu_payload);
			gpu_payload = NULL;
		}
		if (gpu_metrics)
		{
			NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
			gpu_metrics = NULL;
		}
		if (tbl_str)
		{
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			targ_str = NULL;
		}
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_lasso_regression_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		if (model != NULL)
		{
			if (model->coefficients != NULL)
			{
				NDB_SAFE_PFREE_AND_NULL(model->coefficients);
				model->coefficients = NULL;
			}
			NDB_SAFE_PFREE_AND_NULL(model);
			model = NULL;
		}
		if (gpu_payload)
		{
			NDB_SAFE_PFREE_AND_NULL(gpu_payload);
			gpu_payload = NULL;
		}
		if (gpu_metrics)
		{
			NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
			gpu_metrics = NULL;
		}
		if (tbl_str)
		{
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			targ_str = NULL;
		}
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_lasso_regression_by_model_id: no valid rows found")));
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
		const NdbCudaLassoModelHeader *gpu_hdr;
		int feat_dim = 0;

		/* Defensive: validate GPU payload */
		if (gpu_payload == NULL || VARSIZE(gpu_payload) < sizeof(NdbCudaLassoModelHeader))
		{
			if (gpu_payload)
				NDB_SAFE_PFREE_AND_NULL(gpu_payload);
			if (gpu_metrics)
				NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_lasso_regression_by_model_id: invalid GPU model payload")));
		}

		/* Load GPU model header */
		gpu_hdr = (const NdbCudaLassoModelHeader *)VARDATA(gpu_payload);
		feat_dim = gpu_hdr->feature_dim;

		/* Defensive: validate feature dimension */
		if (feat_dim <= 0 || feat_dim > 100000)
		{
			if (gpu_payload)
				NDB_SAFE_PFREE_AND_NULL(gpu_payload);
			if (gpu_metrics)
				NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_lasso_regression_by_model_id: invalid feature dimension %d",
						feat_dim)));
		}

		/* Allocate host buffers for features and targets */
		{
			size_t features_size = sizeof(float) * (size_t)nvec * (size_t)feat_dim;
			size_t targets_size = sizeof(double) * (size_t)nvec;

			/* Defensive: check for overflow */
			if (features_size > MaxAllocSize || targets_size > MaxAllocSize)
			{
				if (gpu_payload)
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
				if (gpu_metrics)
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: data too large for GPU evaluation")));
			}

			h_features = (float *)palloc(features_size);
			h_targets = (double *)palloc(targets_size);

			/* Defensive: validate allocation */
			if (h_features == NULL || h_targets == NULL)
			{
				if (h_features)
				{
					NDB_SAFE_PFREE_AND_NULL(h_features);
					h_features = NULL;
				}
				if (h_targets)
				{
					NDB_SAFE_PFREE_AND_NULL(h_targets);
					h_targets = NULL;
				}
				if (gpu_payload)
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
				if (gpu_metrics)
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: failed to allocate memory")));
			}
		}

		/* Extract features and targets from SPI results - optimized batch extraction */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc tupdesc = SPI_tuptable->tupdesc;

			if (tupdesc == NULL)
			{
				NDB_SAFE_PFREE_AND_NULL(h_features);
				h_features = NULL;
				NDB_SAFE_PFREE_AND_NULL(h_targets);
				h_targets = NULL;
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
				if (tbl_str)
				{
					NDB_SAFE_PFREE_AND_NULL(tbl_str);
					tbl_str = NULL;
				}
				if (feat_str)
				{
					NDB_SAFE_PFREE_AND_NULL(feat_str);
					feat_str = NULL;
				}
				if (targ_str)
				{
					NDB_SAFE_PFREE_AND_NULL(targ_str);
					targ_str = NULL;
				}
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: invalid tuple descriptor")));
			}

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

				if (tuple == NULL)
					continue;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				feat_row = h_features + (valid_rows * feat_dim);
				h_targets[valid_rows] = DatumGetFloat8(targ_datum);
				y_mean += h_targets[valid_rows];

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (arr == NULL)
						continue;
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						/* Optimized: bulk conversion with loop unrolling hint */
						float8 *data = (float8 *)ARR_DATA_PTR(arr);
						int j;
						int j_remain = feat_dim % 4;
						int j_end = feat_dim - j_remain;

						if (data == NULL)
							continue;

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
						if (data == NULL)
							continue;
						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					/* Vector type: direct memcpy (already optimal) */
					vec = DatumGetVector(feat_datum);
					if (vec == NULL)
						continue;
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_rows++;
			}
		}

		if (valid_rows == 0)
		{
			NDB_SAFE_PFREE_AND_NULL(h_features);
			NDB_SAFE_PFREE_AND_NULL(h_targets);
			if (gpu_payload)
				NDB_SAFE_PFREE_AND_NULL(gpu_payload);
			if (gpu_metrics)
				NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_lasso_regression_by_model_id: no valid rows found")));
		}

		/* Use optimized GPU evaluation kernel instead of cuBLAS + CPU loop */
		/* This kernel computes predictions and accumulates SSE/SAE in one pass */
		{
			double gpu_mse = 0.0;
			double gpu_mae = 0.0;
			double gpu_rmse = 0.0;
			double gpu_r_squared = 0.0;
			char *gpu_err = NULL;
			int eval_rc;

			elog(DEBUG1,
				"neurondb: evaluate_lasso_regression_by_model_id: using GPU evaluation kernel for %d samples",
				valid_rows);

			eval_rc = ndb_cuda_lasso_evaluate(gpu_payload,
				h_features,
				h_targets,
				valid_rows,
				feat_dim,
				&gpu_mse,
				&gpu_mae,
				&gpu_rmse,
				&gpu_r_squared,
				&gpu_err);

			if (eval_rc == 0)
			{
				/* GPU evaluation succeeded */
				mse = gpu_mse;
				mae = gpu_mae;
				rmse = gpu_rmse;
				r_squared = gpu_r_squared;
				nvec = valid_rows;

				/* Cleanup */
				NDB_SAFE_PFREE_AND_NULL(h_features);
				h_features = NULL;
				NDB_SAFE_PFREE_AND_NULL(h_targets);
				h_targets = NULL;
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
			}
			else
			{
				/* GPU evaluation failed, fall back to CPU */
				elog(DEBUG1,
					"neurondb: evaluate_lasso_regression_by_model_id: GPU evaluation kernel failed: %s, falling back to CPU",
					gpu_err ? gpu_err : "unknown error");
				if (gpu_err)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_err);
					gpu_err = NULL;
				}
				NDB_SAFE_PFREE_AND_NULL(h_features);
				h_features = NULL;
				NDB_SAFE_PFREE_AND_NULL(h_targets);
				h_targets = NULL;
				goto gpu_eval_fallback;
			}
		}

		NDB_SAFE_PFREE_AND_NULL(tbl_str);
		NDB_SAFE_PFREE_AND_NULL(feat_str);
		NDB_SAFE_PFREE_AND_NULL(targ_str);

		/* Build jsonb result in old context to survive SPI_finish */
		MemoryContextSwitchTo(oldcontext);
		initStringInfo(&jsonbuf);
		appendStringInfo(&jsonbuf,
			"{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
			mse,
			mae,
			rmse,
			r_squared,
			nvec);

		result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
			CStringGetDatum(jsonbuf.data)));

		NDB_SAFE_PFREE_AND_NULL(jsonbuf.data);
		SPI_finish();
		PG_RETURN_JSONB_P(result_jsonb);
#endif	/* NDB_GPU_CUDA */
	}

gpu_eval_fallback:
	/* CPU batch evaluation path (also used as fallback for GPU models) */
	/* Use the already-loaded h_features and h_targets arrays for batch evaluation */
	{
		/* Load GPU model into CPU structure for CPU evaluation */
		const NdbCudaLassoModelHeader *gpu_hdr;
		const float *gpu_coefficients;
		double *cpu_coefficients = NULL;
		double cpu_intercept = 0.0;
		int feat_dim;
		int valid_samples = 0;
		double *predictions = NULL;

		if (is_gpu_model && gpu_payload)
		{
			/* Defensive: validate GPU payload */
			if (VARSIZE(gpu_payload) < sizeof(NdbCudaLassoModelHeader))
			{
				if (gpu_payload)
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
				if (gpu_metrics)
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: invalid GPU model payload")));
			}

			gpu_hdr = (const NdbCudaLassoModelHeader *)VARDATA(gpu_payload);
			gpu_coefficients = (const float *)((const char *)gpu_hdr + sizeof(NdbCudaLassoModelHeader));
			feat_dim = gpu_hdr->feature_dim;
			cpu_intercept = gpu_hdr->intercept;

			/* Defensive: validate feature dimension */
			if (feat_dim <= 0 || feat_dim > 100000)
			{
				if (gpu_payload)
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
				if (gpu_metrics)
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: invalid feature dimension %d",
							feat_dim)));
			}

			/* Convert coefficients from float to double */
			cpu_coefficients = (double *)palloc(sizeof(double) * feat_dim);
			if (cpu_coefficients == NULL)
			{
				if (gpu_payload)
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
				if (gpu_metrics)
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: failed to allocate coefficients")));
			}
			for (i = 0; i < feat_dim; i++)
				cpu_coefficients[i] = (double)gpu_coefficients[i];
		}
		else if (model != NULL)
		{
			/* Defensive: validate CPU model */
			if (model->coefficients == NULL || model->n_features <= 0)
			{
				if (model->coefficients != NULL)
					NDB_SAFE_PFREE_AND_NULL(model->coefficients);
				NDB_SAFE_PFREE_AND_NULL(model);
				if (gpu_payload)
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
				if (gpu_metrics)
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: invalid CPU model")));
			}
			cpu_coefficients = model->coefficients;
			cpu_intercept = model->intercept;
			feat_dim = model->n_features;
		}
		else
		{
			if (gpu_payload)
				NDB_SAFE_PFREE_AND_NULL(gpu_payload);
			if (gpu_metrics)
				NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_lasso_regression_by_model_id: CPU model evaluation failed")));
		}

		/* Re-execute query to get data for CPU evaluation (since GPU arrays may not be loaded) */
		if (!is_gpu_model || h_features == NULL || h_targets == NULL)
		{
			float *cpu_features = NULL;
			double *cpu_targets = NULL;
			size_t features_size, targets_size;

			/* Re-execute query for CPU batch evaluation */
			ret = ndb_spi_execute_safe(query.data, true, 0);
			NDB_CHECK_SPI_TUPTABLE();
			if (ret != SPI_OK_SELECT)
			{
				if (cpu_coefficients != NULL && (model == NULL || cpu_coefficients != model->coefficients))
					NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
				NDB_SAFE_PFREE_AND_NULL(gpu_payload);
				NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: query failed for CPU batch evaluation")));
			}

			nvec = SPI_processed;
			if (nvec < 2)
			{
				if (cpu_coefficients != NULL && (model == NULL || cpu_coefficients != model->coefficients))
					NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
				if (gpu_payload)
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
				if (gpu_metrics)
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: need at least 2 samples for CPU evaluation, got %d",
							nvec)));
			}

			/* Allocate arrays for CPU batch evaluation */
			features_size = sizeof(float) * (size_t)nvec * (size_t)feat_dim;
			targets_size = sizeof(double) * (size_t)nvec;

			/* Defensive: check for overflow */
			if (features_size > MaxAllocSize || targets_size > MaxAllocSize)
			{
				if (cpu_coefficients != NULL && (model == NULL || cpu_coefficients != model->coefficients))
				{
					NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
					cpu_coefficients = NULL;
				}
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
				if (tbl_str)
				{
					NDB_SAFE_PFREE_AND_NULL(tbl_str);
					tbl_str = NULL;
				}
				if (feat_str)
				{
					NDB_SAFE_PFREE_AND_NULL(feat_str);
					feat_str = NULL;
				}
				if (targ_str)
				{
					NDB_SAFE_PFREE_AND_NULL(targ_str);
					targ_str = NULL;
				}
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: data too large for CPU batch evaluation")));
			}

			cpu_features = (float *)palloc(features_size);
			cpu_targets = (double *)palloc(targets_size);

			/* Defensive: validate allocation */
			if (cpu_features == NULL || cpu_targets == NULL)
			{
				if (cpu_features)
				{
					NDB_SAFE_PFREE_AND_NULL(cpu_features);
					cpu_features = NULL;
				}
				if (cpu_targets)
				{
					NDB_SAFE_PFREE_AND_NULL(cpu_targets);
					cpu_targets = NULL;
				}
				if (cpu_coefficients != NULL && (model == NULL || cpu_coefficients != model->coefficients))
				{
					NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
					cpu_coefficients = NULL;
				}
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
				if (tbl_str)
				{
					NDB_SAFE_PFREE_AND_NULL(tbl_str);
					tbl_str = NULL;
				}
				if (feat_str)
				{
					NDB_SAFE_PFREE_AND_NULL(feat_str);
					feat_str = NULL;
				}
				if (targ_str)
				{
					NDB_SAFE_PFREE_AND_NULL(targ_str);
					targ_str = NULL;
				}
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: failed to allocate memory for CPU evaluation")));
			}

			/* Extract features and targets in batch */
			{
				TupleDesc tupdesc = SPI_tuptable->tupdesc;

				if (tupdesc == NULL)
				{
					NDB_SAFE_PFREE_AND_NULL(cpu_features);
					NDB_SAFE_PFREE_AND_NULL(cpu_targets);
					if (cpu_coefficients != NULL && (model == NULL || cpu_coefficients != model->coefficients))
						NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
					if (gpu_payload)
						NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					if (gpu_metrics)
						NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					NDB_SAFE_PFREE_AND_NULL(tbl_str);
					NDB_SAFE_PFREE_AND_NULL(feat_str);
					NDB_SAFE_PFREE_AND_NULL(targ_str);
					SPI_finish();
					ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
							errmsg("neurondb: evaluate_lasso_regression_by_model_id: invalid tuple descriptor")));
				}

				for (i = 0; i < nvec; i++)
				{
					HeapTuple tuple = SPI_tuptable->vals[i];
					Datum feat_datum, targ_datum;
					bool feat_null, targ_null;
					float *feat_row;
					int k;

					if (tuple == NULL)
						continue;

					feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
					targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

					if (feat_null || targ_null)
						continue;

					feat_row = cpu_features + (valid_samples * feat_dim);
					cpu_targets[valid_samples] = DatumGetFloat8(targ_datum);

					/* Extract feature vector */
					if (feat_is_array)
					{
						ArrayType *arr = DatumGetArrayTypeP(feat_datum);
						if (arr == NULL)
							continue;
						if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
							continue;

						if (feat_type_oid == FLOAT8ARRAYOID)
						{
							float8 *data = (float8 *)ARR_DATA_PTR(arr);
							if (data == NULL)
								continue;
							for (k = 0; k < feat_dim; k++)
								feat_row[k] = (float)data[k];
						}
						else
						{
							float4 *data = (float4 *)ARR_DATA_PTR(arr);
							if (data == NULL)
								continue;
							memcpy(feat_row, data, sizeof(float) * feat_dim);
						}
					}
					else
					{
						Vector *vec = DatumGetVector(feat_datum);
						if (vec == NULL)
							continue;
						if (vec->dim != feat_dim)
							continue;
						memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
					}

					valid_samples++;
					y_mean += cpu_targets[valid_samples - 1];
				}
			}

			if (valid_samples < 2)
			{
				NDB_SAFE_PFREE_AND_NULL(cpu_features);
				cpu_features = NULL;
				NDB_SAFE_PFREE_AND_NULL(cpu_targets);
				cpu_targets = NULL;
				if (cpu_coefficients != NULL && (model == NULL || cpu_coefficients != model->coefficients))
				{
					NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
					cpu_coefficients = NULL;
				}
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
				if (tbl_str)
				{
					NDB_SAFE_PFREE_AND_NULL(tbl_str);
					tbl_str = NULL;
				}
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: need at least 2 valid samples for CPU evaluation, got %d",
							valid_samples)));
			}

			h_features = cpu_features;
			h_targets = cpu_targets;
			valid_rows = valid_samples;
		}

		/* Defensive: validate valid_rows */
		if (valid_rows < 2)
		{
			if (cpu_coefficients != NULL && (model == NULL || cpu_coefficients != model->coefficients))
				NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
			if (h_features != NULL)
				NDB_SAFE_PFREE_AND_NULL(h_features);
			if (h_targets != NULL)
				NDB_SAFE_PFREE_AND_NULL(h_targets);
			if (gpu_payload)
				NDB_SAFE_PFREE_AND_NULL(gpu_payload);
			if (gpu_metrics)
				NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_lasso_regression_by_model_id: need at least 2 valid rows, got %d",
						valid_rows)));
		}

		y_mean /= valid_rows;

		/* Allocate predictions array */
		predictions = (double *)palloc(sizeof(double) * valid_rows);
		if (predictions == NULL)
		{
			if (cpu_coefficients != NULL && (model == NULL || cpu_coefficients != model->coefficients))
				NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
			if (h_features)
				NDB_SAFE_PFREE_AND_NULL(h_features);
			if (h_targets)
				NDB_SAFE_PFREE_AND_NULL(h_targets);
			if (gpu_payload)
				NDB_SAFE_PFREE_AND_NULL(gpu_payload);
			if (gpu_metrics)
				NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
			NDB_SAFE_PFREE_AND_NULL(tbl_str);
			NDB_SAFE_PFREE_AND_NULL(feat_str);
			NDB_SAFE_PFREE_AND_NULL(targ_str);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
					errmsg("neurondb: evaluate_lasso_regression_by_model_id: failed to allocate predictions array")));
		}

		/* Batch prediction: compute all predictions at once */
		for (i = 0; i < valid_rows; i++)
		{
			const float *feat_row = h_features + (i * feat_dim);
			double prediction = cpu_intercept;
			int k;

			/* Defensive: validate feat_row pointer */
			if (feat_row == NULL)
			{
				NDB_SAFE_PFREE_AND_NULL(predictions);
				predictions = NULL;
				if (cpu_coefficients != NULL && (model == NULL || cpu_coefficients != model->coefficients))
				{
					NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
					cpu_coefficients = NULL;
				}
				if (h_features)
				{
					NDB_SAFE_PFREE_AND_NULL(h_features);
					h_features = NULL;
				}
				if (h_targets)
				{
					NDB_SAFE_PFREE_AND_NULL(h_targets);
					h_targets = NULL;
				}
				if (gpu_payload)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics)
				{
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
					gpu_metrics = NULL;
				}
				if (tbl_str)
				{
					NDB_SAFE_PFREE_AND_NULL(tbl_str);
					tbl_str = NULL;
				}
				if (feat_str)
				{
					NDB_SAFE_PFREE_AND_NULL(feat_str);
					feat_str = NULL;
				}
				if (targ_str)
				{
					NDB_SAFE_PFREE_AND_NULL(targ_str);
					targ_str = NULL;
				}
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("neurondb: evaluate_lasso_regression_by_model_id: invalid feature row pointer")));
			}

			for (k = 0; k < feat_dim; k++)
				prediction += cpu_coefficients[k] * (double)feat_row[k];

			predictions[i] = prediction;
		}

		/* Compute metrics in batch */
		for (i = 0; i < valid_rows; i++)
		{
			double y_true = h_targets[i];
			double y_pred = predictions[i];
			double error = y_true - y_pred;

			mse += error * error;
			mae += fabs(error);
			ss_res += error * error;
			ss_tot += (y_true - y_mean) * (y_true - y_mean);
		}

		mse /= valid_rows;
		mae /= valid_rows;
		rmse = sqrt(mse);

		/* Handle R² calculation */
		if (ss_tot == 0.0)
			r_squared = 0.0;
		else
			r_squared = 1.0 - (ss_res / ss_tot);

		nvec = valid_rows;

		/* Cleanup */
		NDB_SAFE_PFREE_AND_NULL(predictions);
		predictions = NULL;
		if (h_features != NULL)
		{
			NDB_SAFE_PFREE_AND_NULL(h_features);
			h_features = NULL;
		}
		if (h_targets != NULL)
		{
			NDB_SAFE_PFREE_AND_NULL(h_targets);
			h_targets = NULL;
		}
		if (cpu_coefficients != NULL && (model == NULL || cpu_coefficients != model->coefficients))
		{
			NDB_SAFE_PFREE_AND_NULL(cpu_coefficients);
			cpu_coefficients = NULL;
		}
	}

	/* Build jsonb result in old context to survive SPI_finish */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
		mse,
		mae,
		rmse,
		r_squared,
		nvec);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
		CStringGetDatum(jsonbuf.data)));

	NDB_SAFE_PFREE_AND_NULL(jsonbuf.data);
	jsonbuf.data = NULL;
	if (model != NULL)
	{
		if (model->coefficients != NULL)
		{
			NDB_SAFE_PFREE_AND_NULL(model->coefficients);
			model->coefficients = NULL;
		}
		NDB_SAFE_PFREE_AND_NULL(model);
		model = NULL;
	}
	if (gpu_payload)
	{
		NDB_SAFE_PFREE_AND_NULL(gpu_payload);
		gpu_payload = NULL;
	}
	if (gpu_metrics)
	{
		NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
		gpu_metrics = NULL;
	}
	NDB_SAFE_PFREE_AND_NULL(tbl_str);
	NDB_SAFE_PFREE_AND_NULL(feat_str);
	NDB_SAFE_PFREE_AND_NULL(targ_str);
	SPI_finish();
	PG_RETURN_JSONB_P(result_jsonb);
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

	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	RidgeDataset dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_target;
	int			nvec;
	int			dim;
	int			i;
	int			j;
	double	   *coefficients = NULL;
	double		intercept = 0.0;
	double		l1_penalty;
	double		l2_penalty;
	double	   *Xty = NULL;
	double	   *beta = NULL;
	Datum	   *result_datums;
	ArrayType  *result_array;
	MemoryContext oldcontext;
	MemoryContext elastic_context;

	/* Defensive: validate inputs */
	if (alpha < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("elastic_net: alpha must be non-negative")));

	if (l1_ratio < 0.0 || l1_ratio > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("elastic_net: l1_ratio must be between 0 and 1")));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	/* Create memory context */
	elog(DEBUG1, "elastic_net: creating training context");
	elastic_context = AllocSetContextCreate(CurrentMemoryContext,
										   "elastic net training context",
										   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(elastic_context);

	ridge_dataset_init(&dataset);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_target = quote_identifier(targ_str);

	ridge_dataset_load(quoted_tbl, quoted_feat, quoted_target, &dataset);

	nvec = dataset.n_samples;
	dim = dataset.feature_dim;

	if (nvec < 10)
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(elastic_context);
		ridge_dataset_free(&dataset);
		NDB_SAFE_PFREE_AND_NULL(tbl_str);
		NDB_SAFE_PFREE_AND_NULL(feat_str);
		NDB_SAFE_PFREE_AND_NULL(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("elastic_net: need at least 10 samples, got %d",
						nvec)));
	}

	/* Calculate penalty terms */
	l1_penalty = alpha * l1_ratio;
	l2_penalty = alpha * (1.0 - l1_ratio);

	/* Allocate coefficient arrays */
	coefficients = (double *)palloc0(dim * sizeof(double));
	beta = (double *)palloc0((dim + 1) * sizeof(double));

	/* Build X'X + regularization matrix */
	/* Allocate as double ** for matrix_invert compatibility */
	{
		double	   **XtX_2d = NULL;
		double	   **XtX_inv_2d = NULL;
		int			dim_with_intercept = dim + 1;
		int			k;

		XtX_2d = (double **)palloc(sizeof(double *) * dim_with_intercept);
		XtX_inv_2d = (double **)palloc(
			sizeof(double *) * dim_with_intercept);

		for (i = 0; i < dim_with_intercept; i++)
		{
			XtX_2d[i] = (double *)palloc0(
				sizeof(double) * dim_with_intercept);
			XtX_inv_2d[i] = (double *)palloc0(
				sizeof(double) * dim_with_intercept);
		}

		Xty = (double *)palloc0((dim + 1) * sizeof(double));

		/* Compute X'X with L2 regularization */
		for (i = 0; i < dim; i++)
		{
			for (j = 0; j < dim; j++)
			{
				double		sum = 0.0;

				for (k = 0; k < nvec; k++)
					sum += dataset.features[k * dim + i] *
						dataset.features[k * dim + j];

				XtX_2d[i + 1][j + 1] = sum;
				if (i == j)
					XtX_2d[i + 1][j + 1] += l2_penalty;
			}
		}

		/* Add intercept column (all ones) */
		for (i = 0; i < dim; i++)
		{
			double		sum = 0.0;

			for (k = 0; k < nvec; k++)
				sum += dataset.features[k * dim + i];
			XtX_2d[i + 1][0] = sum;
			XtX_2d[0][i + 1] = sum;
		}
		XtX_2d[0][0] = nvec;

		/* Compute X'y */
		for (i = 0; i < dim; i++)
		{
			double		sum = 0.0;

			for (k = 0; k < nvec; k++)
				sum += dataset.features[k * dim + i] * dataset.targets[k];
			Xty[i + 1] = sum;
		}
		{
			double		sum = 0.0;

			for (k = 0; k < nvec; k++)
				sum += dataset.targets[k];
			Xty[0] = sum;
		}

		/* Solve (X'X + λI)β = X'y using coordinate descent for Elastic Net */
		/* Simplified: use Ridge solution as starting point, then apply L1 shrinkage */
		{
			bool		invert_ok;

			invert_ok = matrix_invert(XtX_2d, dim_with_intercept,
									  XtX_inv_2d);

			if (invert_ok)
			{
				/* Compute beta = (X'X + λI)^(-1)X'y */
				for (i = 0; i < dim_with_intercept; i++)
				{
					beta[i] = 0.0;
					for (j = 0; j < dim_with_intercept; j++)
						beta[i] += XtX_inv_2d[i][j] * Xty[j];
				}

				/* Apply L1 shrinkage (soft thresholding) */
				for (i = 1; i < dim_with_intercept; i++)
				{
					if (beta[i] > l1_penalty)
						beta[i] -= l1_penalty;
					else if (beta[i] < -l1_penalty)
						beta[i] += l1_penalty;
					else
						beta[i] = 0.0;
				}

				intercept = beta[0];
				for (i = 0; i < dim; i++)
					coefficients[i] = beta[i + 1];
			}
			else
			{
				/* Cleanup on inversion failure */
				for (i = 0; i < dim_with_intercept; i++)
				{
					NDB_SAFE_PFREE_AND_NULL(XtX_2d[i]);
					NDB_SAFE_PFREE_AND_NULL(XtX_inv_2d[i]);
				}
				NDB_SAFE_PFREE_AND_NULL(XtX_2d);
				NDB_SAFE_PFREE_AND_NULL(XtX_inv_2d);
				NDB_SAFE_PFREE_AND_NULL(Xty);
				NDB_SAFE_PFREE_AND_NULL(beta);
				NDB_SAFE_PFREE_AND_NULL(coefficients);
				MemoryContextSwitchTo(oldcontext);
				MemoryContextDelete(elastic_context);
				ridge_dataset_free(&dataset);
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("elastic_net: failed to solve linear system")));
			}

			/* Cleanup matrix allocations */
			for (i = 0; i < dim_with_intercept; i++)
			{
				NDB_SAFE_PFREE_AND_NULL(XtX_2d[i]);
				NDB_SAFE_PFREE_AND_NULL(XtX_inv_2d[i]);
			}
			NDB_SAFE_PFREE_AND_NULL(XtX_2d);
			NDB_SAFE_PFREE_AND_NULL(XtX_inv_2d);
		}
	}


	/* Build result array */
	result_datums = (Datum *)palloc((dim + 1) * sizeof(Datum));
	result_datums[0] = Float8GetDatum(intercept);
	for (i = 0; i < dim; i++)
		result_datums[i + 1] = Float8GetDatum(coefficients[i]);

	result_array = construct_array(result_datums,
								   dim + 1,
								   FLOAT8OID,
								   8,
								   FLOAT8PASSBYVAL,
								   'd');

	/* Cleanup */
	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(elastic_context);
	ridge_dataset_free(&dataset);
	NDB_SAFE_PFREE_AND_NULL(tbl_str);
	NDB_SAFE_PFREE_AND_NULL(feat_str);
	NDB_SAFE_PFREE_AND_NULL(targ_str);
	NDB_SAFE_PFREE_AND_NULL(Xty);
	NDB_SAFE_PFREE_AND_NULL(beta);
	NDB_SAFE_PFREE_AND_NULL(coefficients);
	NDB_SAFE_PFREE_AND_NULL(result_datums);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * predict_elastic_net
 *      Predicts using a trained Elastic Net model.
 *      Arguments: int4 model_id, float8[] features
 *      Returns: float8 prediction
 */
PG_FUNCTION_INFO_V1(predict_elastic_net);

Datum
predict_elastic_net(PG_FUNCTION_ARGS)
{
	int32 model_id = PG_GETARG_INT32(0);
	ArrayType *features_array = PG_GETARG_ARRAYTYPE_P(1);

	/* Elastic Net uses the same prediction as Ridge regression */
	/* since it's just a linear model with combined regularization */
	return DirectFunctionCall2(predict_ridge_regression_model_id,
							   Int32GetDatum(model_id),
							   PointerGetDatum(features_array));
}

/*
 * evaluate_elastic_net_by_model_id
 *      Evaluates Elastic Net model performance on a dataset.
 *      Arguments: int4 model_id, text table_name, text feature_col, text label_col
 *      Returns: jsonb with regression metrics
 */
PG_FUNCTION_INFO_V1(evaluate_elastic_net_by_model_id);

Datum
evaluate_elastic_net_by_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	text *table_name;
	text *feature_col;
	text *label_col;
	char *tbl_str;
	char *feat_str;
	char *targ_str;
	StringInfoData query;
	int ret;
	int nvec = 0;
	int i;
	double mse = 0.0;

	Oid feat_type_oid = InvalidOid;
	bool feat_is_array = false;
	double mae = 0.0;
	double ss_tot = 0.0;
	double ss_res = 0.0;
	double y_mean = 0.0;
	double r_squared;
	double rmse;
	
	StringInfoData jsonbuf;
	Jsonb *result;
	MemoryContext oldcontext;

	/* Validate arguments */
	if (PG_NARGS() != 4)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_elastic_net_by_model_id: 4 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_elastic_net_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_elastic_net_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_elastic_net_by_model_id: SPI_connect failed")));

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		feat_str, targ_str, tbl_str, feat_str, targ_str);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_elastic_net_by_model_id: query failed")));

	nvec = SPI_processed;
	if (nvec < 2)
	{
		SPI_finish();
		NDB_SAFE_PFREE_AND_NULL(tbl_str);
		NDB_SAFE_PFREE_AND_NULL(feat_str);
		NDB_SAFE_PFREE_AND_NULL(targ_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_elastic_net_by_model_id: need at least 2 samples, got %d",
					nvec)));
	}

	/* First pass: compute mean of y */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum targ_datum;
		bool targ_null;

		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (!targ_null)
			y_mean += DatumGetFloat8(targ_datum);
	}
	y_mean /= nvec;
	feat_type_oid = InvalidOid;
	feat_is_array = false;
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
	{
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
		if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
			feat_is_array = true;
	}

	/* Second pass: compute predictions and metrics */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum feat_datum;
		Datum targ_datum;
		bool feat_null;
		bool targ_null;
		ArrayType *arr;
		Vector *vec;
		double y_true;
		double y_pred;
		double error;
		int actual_dim;
		int j;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		y_true = DatumGetFloat8(targ_datum);

		/* Extract features and determine dimension */
		if (feat_is_array)
		{
			arr = DatumGetArrayTypeP(feat_datum);
			if (ARR_NDIM(arr) != 1)
			{
				SPI_finish();
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(targ_str);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("elastic_net: features array must be 1-D")));
			}
			actual_dim = ARR_DIMS(arr)[0];
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			actual_dim = vec->dim;
		}

		/* Make prediction using Elastic Net model */
		if (feat_is_array)
		{
			/* Create a temporary array for prediction */
			Datum features_datum = feat_datum;
			y_pred = DatumGetFloat8(DirectFunctionCall2(predict_elastic_net,
													   Int32GetDatum(model_id),
													   features_datum));
		}
		else
		{
			/* Convert vector to array for prediction */
			int ndims = 1;
			int dims[1];
			int lbs[1];
			Datum *elems;
			ArrayType *feature_array;
			Datum features_datum;

			dims[0] = actual_dim;
			lbs[0] = 1;
			elems = palloc(sizeof(Datum) * actual_dim);

			for (j = 0; j < actual_dim; j++)
				elems[j] = Float8GetDatum(vec->data[j]);

			feature_array = construct_md_array(elems, NULL, ndims, dims, lbs,
														FLOAT8OID, sizeof(float8), true, 'd');
			features_datum = PointerGetDatum(feature_array);

			y_pred = DatumGetFloat8(DirectFunctionCall2(predict_elastic_net,
													   Int32GetDatum(model_id),
													   features_datum));

			NDB_SAFE_PFREE_AND_NULL(elems);
			NDB_SAFE_PFREE_AND_NULL(feature_array);
		}

		/* Compute errors */
		error = y_true - y_pred;
		mse += error * error;
		mae += fabs(error);
		ss_res += error * error;
		ss_tot += (y_true - y_mean) * (y_true - y_mean);
	}

	SPI_finish();

	mse /= nvec;
	mae /= nvec;
	rmse = sqrt(mse);

	/* Handle R² calculation - if ss_tot is zero (no variance in y), R² is undefined */
	if (ss_tot == 0.0)
		r_squared = 0.0; /* Convention: set to 0 when there's no variance to explain */
	else
		r_squared = 1.0 - (ss_res / ss_tot);

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
		mse, mae, rmse, r_squared, nvec);

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(jsonbuf.data)));
	NDB_SAFE_PFREE_AND_NULL(jsonbuf.data);
	jsonbuf.data = NULL;

	/* Cleanup */
	if (tbl_str)
	{
		NDB_SAFE_PFREE_AND_NULL(tbl_str);
		tbl_str = NULL;
	}
	if (feat_str)
	{
		NDB_SAFE_PFREE_AND_NULL(feat_str);
		feat_str = NULL;
	}
	if (targ_str)
	{
		NDB_SAFE_PFREE_AND_NULL(targ_str);
		targ_str = NULL;
	}

	PG_RETURN_JSONB_P(result);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops for Ridge Regression
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"

typedef struct RidgeGpuModelState
{
	bytea *model_blob;
	Jsonb *metrics;
	int feature_dim;
	int n_samples;
} RidgeGpuModelState;

static void
ridge_gpu_release_state(RidgeGpuModelState *state)
{
	if (state == NULL)
		return;
	if (state->model_blob != NULL)
		NDB_SAFE_PFREE_AND_NULL(state->model_blob);
	if (state->metrics != NULL)
		NDB_SAFE_PFREE_AND_NULL(state->metrics);
	NDB_SAFE_PFREE_AND_NULL(state);
}

static bool
ridge_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	RidgeGpuModelState *state;
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

	rc = ndb_gpu_ridge_train(spec->feature_matrix,
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
			NDB_SAFE_PFREE_AND_NULL(payload);
		if (metrics != NULL)
			NDB_SAFE_PFREE_AND_NULL(metrics);
		return false;
	}

	if (model->backend_state != NULL)
	{
		ridge_gpu_release_state((RidgeGpuModelState *)model->backend_state);
		model->backend_state = NULL;
	}

	state = (RidgeGpuModelState *)palloc0(sizeof(RidgeGpuModelState));
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;

	if (metrics != NULL)
	{
		state->metrics = (Jsonb *)PG_DETOAST_DATUM_COPY(PointerGetDatum(metrics));
	}
	else
	{
		state->metrics = NULL;
	}

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
ridge_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
	float *output, int output_dim, char **errstr)
{
	const RidgeGpuModelState *state;
	double prediction;
	int rc;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = 0.0f;
	if (model == NULL || input == NULL || output == NULL)
		return false;
	if (output_dim <= 0)
		return false;
	if (!model->gpu_ready || model->backend_state == NULL)
		return false;

	state = (const RidgeGpuModelState *)model->backend_state;
	if (state->model_blob == NULL)
		return false;

	rc = ndb_gpu_ridge_predict(state->model_blob, input,
		state->feature_dim > 0 ? state->feature_dim : input_dim,
		&prediction, errstr);
	if (rc != 0)
		return false;

	output[0] = (float)prediction;
	return true;
}

static bool
ridge_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
	MLGpuMetrics *out, char **errstr)
{
	const RidgeGpuModelState *state;
	Jsonb *metrics_json;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const RidgeGpuModelState *)model->backend_state;
	{
		StringInfoData buf;
		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"ridge\",\"storage\":\"gpu\",\"n_features\":%d,\"n_samples\":%d}",
			state->feature_dim > 0 ? state->feature_dim : 0,
			state->n_samples > 0 ? state->n_samples : 0);
		metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(buf.data)));
		NDB_SAFE_PFREE_AND_NULL(buf.data);
	}
	if (out != NULL)
		out->payload = metrics_json;
	return true;
}

static bool
ridge_gpu_serialize(const MLGpuModel *model, bytea **payload_out,
	Jsonb **metadata_out, char **errstr)
{
	const RidgeGpuModelState *state;
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

	state = (const RidgeGpuModelState *)model->backend_state;
	if (state->model_blob == NULL)
		return false;

	payload_size = VARSIZE(state->model_blob);
	payload_copy = (bytea *)palloc(payload_size);
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		NDB_SAFE_PFREE_AND_NULL(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
	{
		*metadata_out = (Jsonb *)PG_DETOAST_DATUM_COPY(PointerGetDatum(state->metrics));
	}
	else if (metadata_out != NULL)
	{
		*metadata_out = NULL;
	}
	return true;
}

static bool
ridge_gpu_deserialize(MLGpuModel *model, const bytea *payload,
	const Jsonb *metadata, char **errstr)
{
	RidgeGpuModelState *state;
	bytea *payload_copy;
	int payload_size;
	int feature_dim = -1;
	int n_samples = -1;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

	payload_size = VARSIZE(payload);
	payload_copy = (bytea *)palloc(payload_size);
	memcpy(payload_copy, payload, payload_size);

	/* Extract feature_dim and n_samples from metadata if available */
	if (metadata != NULL)
	{
		JsonbIterator *it = NULL;
		JsonbValue	v;
		int			r;
		
		PG_TRY();
		{
			it = JsonbIteratorInit((JsonbContainer *)&metadata->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY && v.type == jbvString)
				{
					char *key = pnstrdup(v.val.string.val, v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "n_features") == 0 && v.type == jbvNumeric)
					{
						feature_dim = DatumGetInt32(DirectFunctionCall1(numeric_int4,
							NumericGetDatum(v.val.numeric)));
					}
					else if (strcmp(key, "n_samples") == 0 && v.type == jbvNumeric)
					{
						n_samples = DatumGetInt32(DirectFunctionCall1(numeric_int4,
							NumericGetDatum(v.val.numeric)));
					}
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}
		PG_CATCH();
		{
			/* If metadata parsing fails, use defaults */
			feature_dim = -1;
			n_samples = -1;
		}
		PG_END_TRY();
	}

	state = (RidgeGpuModelState *)palloc0(sizeof(RidgeGpuModelState));
	state->model_blob = payload_copy;
	state->feature_dim = feature_dim;
	state->n_samples = n_samples;

	if (metadata != NULL)
	{
		state->metrics = (Jsonb *)PG_DETOAST_DATUM_COPY(PointerGetDatum(metadata));
	}
	else
	{
		state->metrics = NULL;
	}

	if (model->backend_state != NULL)
		ridge_gpu_release_state((RidgeGpuModelState *)model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;
	return true;
}

static void
ridge_gpu_destroy(MLGpuModel *model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		ridge_gpu_release_state((RidgeGpuModelState *)model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps ridge_gpu_model_ops = {
	.algorithm = "ridge",
	.train = ridge_gpu_train,
	.predict = ridge_gpu_predict,
	.evaluate = ridge_gpu_evaluate,
	.serialize = ridge_gpu_serialize,
	.deserialize = ridge_gpu_deserialize,
	.destroy = ridge_gpu_destroy,
};

/*-------------------------------------------------------------------------
 * GPU Model Ops for Lasso Regression
 *-------------------------------------------------------------------------
 */

typedef struct LassoGpuModelState
{
	bytea *model_blob;
	Jsonb *metrics;
	int feature_dim;
	int n_samples;
} LassoGpuModelState;

static void
lasso_gpu_release_state(LassoGpuModelState *state)
{
	if (state == NULL)
		return;
	if (state->model_blob != NULL)
		NDB_SAFE_PFREE_AND_NULL(state->model_blob);
	if (state->metrics != NULL)
		NDB_SAFE_PFREE_AND_NULL(state->metrics);
	NDB_SAFE_PFREE_AND_NULL(state);
}

static bool
lasso_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	LassoGpuModelState *state;
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

	rc = ndb_gpu_lasso_train(spec->feature_matrix,
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
			NDB_SAFE_PFREE_AND_NULL(payload);
		if (metrics != NULL)
			NDB_SAFE_PFREE_AND_NULL(metrics);
		return false;
	}

	if (model->backend_state != NULL)
	{
		lasso_gpu_release_state((LassoGpuModelState *)model->backend_state);
		model->backend_state = NULL;
	}

	state = (LassoGpuModelState *)palloc0(sizeof(LassoGpuModelState));
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;

	if (metrics != NULL)
	{
		state->metrics = (Jsonb *)PG_DETOAST_DATUM_COPY(PointerGetDatum(metrics));
	}
	else
	{
		state->metrics = NULL;
	}

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
lasso_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
	float *output, int output_dim, char **errstr)
{
	const LassoGpuModelState *state;
	double prediction;
	int rc;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = 0.0f;
	if (model == NULL || input == NULL || output == NULL)
		return false;
	if (output_dim <= 0)
		return false;
	if (!model->gpu_ready || model->backend_state == NULL)
		return false;

	state = (const LassoGpuModelState *)model->backend_state;
	if (state->model_blob == NULL)
		return false;

	rc = ndb_gpu_lasso_predict(state->model_blob, input,
		state->feature_dim > 0 ? state->feature_dim : input_dim,
		&prediction, errstr);
	if (rc != 0)
		return false;

	output[0] = (float)prediction;
	return true;
}

static bool
lasso_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
	MLGpuMetrics *out, char **errstr)
{
	const LassoGpuModelState *state;
	Jsonb *metrics_json;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const LassoGpuModelState *)model->backend_state;
	{
		StringInfoData buf;
		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"lasso\",\"storage\":\"gpu\",\"n_features\":%d,\"n_samples\":%d}",
			state->feature_dim > 0 ? state->feature_dim : 0,
			state->n_samples > 0 ? state->n_samples : 0);
		metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(buf.data)));
		NDB_SAFE_PFREE_AND_NULL(buf.data);
	}
	if (out != NULL)
		out->payload = metrics_json;
	return true;
}

static bool
lasso_gpu_serialize(const MLGpuModel *model, bytea **payload_out,
	Jsonb **metadata_out, char **errstr)
{
	const LassoGpuModelState *state;
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

	state = (const LassoGpuModelState *)model->backend_state;
	if (state->model_blob == NULL)
		return false;

	payload_size = VARSIZE(state->model_blob);
	payload_copy = (bytea *)palloc(payload_size);
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		NDB_SAFE_PFREE_AND_NULL(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
	{
		*metadata_out = (Jsonb *)PG_DETOAST_DATUM_COPY(PointerGetDatum(state->metrics));
	}
	else if (metadata_out != NULL)
	{
		*metadata_out = NULL;
	}
	return true;
}

static bool
lasso_gpu_deserialize(MLGpuModel *model, const bytea *payload,
	const Jsonb *metadata, char **errstr)
{
	LassoGpuModelState *state;
	bytea *payload_copy;
	int payload_size;
	int feature_dim = -1;
	int n_samples = -1;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

	payload_size = VARSIZE(payload);
	payload_copy = (bytea *)palloc(payload_size);
	memcpy(payload_copy, payload, payload_size);

	/* Extract feature_dim and n_samples from metadata if available */
	if (metadata != NULL)
	{
		JsonbIterator *it = NULL;
		JsonbValue	v;
		int			r;
		
		PG_TRY();
		{
			it = JsonbIteratorInit((JsonbContainer *)&metadata->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY && v.type == jbvString)
				{
					char *key = pnstrdup(v.val.string.val, v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "n_features") == 0 && v.type == jbvNumeric)
					{
						feature_dim = DatumGetInt32(DirectFunctionCall1(numeric_int4,
							NumericGetDatum(v.val.numeric)));
					}
					else if (strcmp(key, "n_samples") == 0 && v.type == jbvNumeric)
					{
						n_samples = DatumGetInt32(DirectFunctionCall1(numeric_int4,
							NumericGetDatum(v.val.numeric)));
					}
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}
		PG_CATCH();
		{
			/* If metadata parsing fails, use defaults */
			feature_dim = -1;
			n_samples = -1;
		}
		PG_END_TRY();
	}

	state = (LassoGpuModelState *)palloc0(sizeof(LassoGpuModelState));
	state->model_blob = payload_copy;
	state->feature_dim = feature_dim;
	state->n_samples = n_samples;

	if (metadata != NULL)
	{
		state->metrics = (Jsonb *)PG_DETOAST_DATUM_COPY(PointerGetDatum(metadata));
	}
	else
	{
		state->metrics = NULL;
	}

	if (model->backend_state != NULL)
		lasso_gpu_release_state((LassoGpuModelState *)model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;
	return true;
}

static void
lasso_gpu_destroy(MLGpuModel *model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		lasso_gpu_release_state((LassoGpuModelState *)model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps lasso_gpu_model_ops = {
	.algorithm = "lasso",
	.train = lasso_gpu_train,
	.predict = lasso_gpu_predict,
	.evaluate = lasso_gpu_evaluate,
	.serialize = lasso_gpu_serialize,
	.deserialize = lasso_gpu_deserialize,
	.destroy = lasso_gpu_destroy,
};

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration
 *-------------------------------------------------------------------------
 */

void
neurondb_gpu_register_ridge_model(void)
{
	static bool registered = false;
	if (registered)
		return;
	ndb_gpu_register_model_ops(&ridge_gpu_model_ops);
	registered = true;
}

void
neurondb_gpu_register_lasso_model(void)
{
	static bool registered = false;
	if (registered)
		return;
	ndb_gpu_register_model_ops(&lasso_gpu_model_ops);
	registered = true;
}
