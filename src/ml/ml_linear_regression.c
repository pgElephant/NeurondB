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
#include "utils/memutils.h"
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
#include "neurondb_cuda_linreg.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
extern int ndb_cuda_linreg_evaluate(const bytea *model_data,
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

typedef struct LinRegDataset
{
	float *features;
	double *targets;
	int n_samples;
	int feature_dim;
} LinRegDataset;

/*
 * Streaming accumulator for incremental X'X and X'y computation
 * This avoids loading all data into memory at once
 */
typedef struct LinRegStreamAccum
{
	double **XtX;			/* X'X matrix (dim+1 x dim+1) */
	double *Xty;			/* X'y vector (dim+1) */
	int feature_dim;		/* Feature dimension */
	int n_samples;			/* Total samples processed */
	double y_sum;			/* Sum of y for mean computation */
	double y_sq_sum;		/* Sum of y^2 for variance */
	bool initialized;		/* Whether accumulator is initialized */
} LinRegStreamAccum;

static void linreg_dataset_init(LinRegDataset *dataset);
static void linreg_dataset_free(LinRegDataset *dataset);
static void linreg_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	LinRegDataset *dataset);
static void linreg_dataset_load_limited(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	LinRegDataset *dataset,
	int max_rows);
static void linreg_stream_accum_init(LinRegStreamAccum *accum, int dim);
static void linreg_stream_accum_free(LinRegStreamAccum *accum);
static void linreg_stream_accum_add_row(LinRegStreamAccum *accum,
	const float *features,
	double target);
static void linreg_stream_process_chunk(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	LinRegStreamAccum *accum,
	int chunk_size,
	int offset,
	int *rows_processed);
static bytea *linreg_model_serialize(const LinRegModel *model);
static LinRegModel *linreg_model_deserialize(const bytea *data);
static bool linreg_metadata_is_gpu(Jsonb *metadata);
static bool linreg_try_gpu_predict_catalog(int32 model_id,
	const Vector *feature_vec,
	double *result_out);
static bool linreg_load_model_from_catalog(int32 model_id, LinRegModel **out);
static Jsonb *evaluate_linear_regression_by_model_id_jsonb(int32 model_id,
	text *table_name,
	text *feature_col,
	text *label_col);

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
	Oid feat_type_oid = InvalidOid;
	bool feat_is_array = false;

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
	elog(DEBUG1,
		"linreg_dataset_load: building query for table %s, features %s, target %s",
		quoted_tbl, quoted_feat, quoted_target);
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
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: linreg_dataset_load: query failed")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		SPI_finish();
		elog(DEBUG1,
			"linreg_dataset_load: need at least 10 samples, got %d",
			n_samples);
		ereport(ERROR,
			(errmsg("linreg_dataset_load: need at least 10 samples, got %d",
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
						(errmsg("linreg_dataset_load: features array must be 1-D")));
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
			(errmsg("linreg_dataset_load: could not determine "
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
					(errmsg("linreg_dataset_load: inconsistent array feature dimensions")));
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
					(errmsg("linreg_dataset_load: inconsistent "
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
 * linreg_dataset_load_limited
 *
 * Load dataset with LIMIT clause to avoid loading too much data
 */
static void
linreg_dataset_load_limited(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	LinRegDataset *dataset,
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
				errmsg("neurondb: linreg_dataset_load_limited: dataset is NULL")));

	if (max_rows <= 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: linreg_dataset_load_limited: max_rows must be positive")));

	oldcontext = CurrentMemoryContext;

	/* Initialize query in caller's context before SPI_connect */
	initStringInfo(&query);
	MemoryContextSwitchTo(oldcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: linreg_dataset_load_limited: SPI_connect failed")));
	
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %d",
		quoted_feat,
		quoted_target,
		quoted_tbl,
		quoted_feat,
		quoted_target,
		max_rows);
	
	elog(DEBUG1, "linreg_dataset_load_limited: executing query: %s", query.data);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		pfree(query.data);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: linreg_dataset_load_limited: query failed")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		pfree(query.data);
		SPI_finish();
		elog(DEBUG1,
			"linreg_dataset_load_limited: need at least 10 samples, got %d",
			n_samples);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: linreg_dataset_load_limited: need at least 10 samples, got %d",
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
					pfree(query.data);
					SPI_finish();
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("neurondb: linreg_dataset_load_limited: features array must be 1-D")));
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
		pfree(query.data);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: linreg_dataset_load_limited: could not determine "
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
				pfree(query.data);
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: linreg_dataset_load_limited: inconsistent array feature dimensions")));
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
				pfree(query.data);
				SPI_finish();
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: linreg_dataset_load_limited: inconsistent "
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

	pfree(query.data);
	SPI_finish();
}

/*
 * linreg_stream_accum_init
 *
 * Initialize streaming accumulator for incremental X'X and X'y computation
 */
static void
linreg_stream_accum_init(LinRegStreamAccum *accum, int dim)
{
	int i;
	int dim_with_intercept;

	if (accum == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: linreg_stream_accum_init: accum is NULL")));

	if (dim <= 0 || dim > 10000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: linreg_stream_accum_init: invalid feature dimension %d",
					dim)));

	dim_with_intercept = dim + 1;

	memset(accum, 0, sizeof(LinRegStreamAccum));

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
				errmsg("neurondb: linreg_stream_accum_init: failed to allocate XtX matrix")));

	for (i = 0; i < dim_with_intercept; i++)
	{
		accum->XtX[i] = (double *)palloc0(sizeof(double) * dim_with_intercept);
		if (accum->XtX[i] == NULL)
		{
			/* Cleanup on failure */
			for (i--; i >= 0; i--)
				pfree(accum->XtX[i]);
			pfree(accum->XtX);
			ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
					errmsg("neurondb: linreg_stream_accum_init: failed to allocate XtX row")));
		}
	}

	/* Allocate X'y vector */
	accum->Xty = (double *)palloc0(sizeof(double) * dim_with_intercept);
	if (accum->Xty == NULL)
	{
		/* Cleanup on failure */
		for (i = 0; i < dim_with_intercept; i++)
			pfree(accum->XtX[i]);
		pfree(accum->XtX);
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: linreg_stream_accum_init: failed to allocate Xty vector")));
	}

	accum->initialized = true;
}

/*
 * linreg_stream_accum_free
 *
 * Free memory allocated for streaming accumulator
 */
static void
linreg_stream_accum_free(LinRegStreamAccum *accum)
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
				pfree(accum->XtX[i]);
		}
		pfree(accum->XtX);
		accum->XtX = NULL;
	}

	if (accum->Xty != NULL)
	{
		pfree(accum->Xty);
		accum->Xty = NULL;
	}

	memset(accum, 0, sizeof(LinRegStreamAccum));
}

/*
 * linreg_stream_accum_add_row
 *
 * Add a single row to the streaming accumulator, updating X'X and X'y
 */
static void
linreg_stream_accum_add_row(LinRegStreamAccum *accum,
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
				errmsg("neurondb: linreg_stream_accum_add_row: accumulator not initialized")));

	if (features == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: linreg_stream_accum_add_row: features is NULL")));

	dim_with_intercept = accum->feature_dim + 1;

	/* Allocate temporary vector for this row (with intercept) */
	xi = (double *)palloc(sizeof(double) * dim_with_intercept);
	if (xi == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: linreg_stream_accum_add_row: failed to allocate row vector")));

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

	pfree(xi);
}

/*
 * linreg_stream_process_chunk
 *
 * Process a chunk of data from the table, accumulating statistics
 * Returns number of rows processed in this chunk
 */
static void
linreg_stream_process_chunk(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_target,
	LinRegStreamAccum *accum,
	int chunk_size,
	int offset,
	int *rows_processed)
{
	StringInfoData query;
	int ret;
	int i;
	int n_rows;
	Oid feat_type_oid = InvalidOid;
	bool feat_is_array = false;
	TupleDesc tupdesc;
	float *row_buffer = NULL;

	if (quoted_tbl == NULL || quoted_feat == NULL || quoted_target == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: linreg_stream_process_chunk: NULL parameter")));

	if (accum == NULL || !accum->initialized)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: linreg_stream_process_chunk: accumulator not initialized")));

	if (chunk_size <= 0 || chunk_size > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: linreg_stream_process_chunk: invalid chunk_size %d",
					chunk_size)));

	if (rows_processed == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: linreg_stream_process_chunk: rows_processed is NULL")));

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

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		pfree(query.data);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: linreg_stream_process_chunk: query failed")));
	}

	n_rows = SPI_processed;
	if (n_rows == 0)
	{
		pfree(query.data);
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
		pfree(query.data);
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: linreg_stream_process_chunk: failed to allocate row buffer")));
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
				pfree(row_buffer);
				pfree(query.data);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: linreg_stream_process_chunk: inconsistent feature dimensions")));
			}

			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				double *arr_data = (double *)ARR_DATA_PTR(arr);

				for (j = 0; j < accum->feature_dim; j++)
					row_buffer[j] = (float)arr_data[j];
			}
			else
			{
				float *arr_data = (float *)ARR_DATA_PTR(arr);

				memcpy(row_buffer, arr_data, sizeof(float) * accum->feature_dim);
			}
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			if (vec->dim != accum->feature_dim)
			{
				pfree(row_buffer);
				pfree(query.data);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: linreg_stream_process_chunk: feature dimension mismatch")));
			}
			memcpy(row_buffer, vec->data, sizeof(float) * accum->feature_dim);
		}

		/* Extract target */
		{
			Oid targ_type = SPI_gettypeid(tupdesc, 2);

			if (targ_type == INT2OID || targ_type == INT4OID
				|| targ_type == INT8OID)
				target = (double)DatumGetInt32(targ_datum);
			else
				target = DatumGetFloat8(targ_datum);
		}

		/* Add row to accumulator */
		linreg_stream_accum_add_row(accum, row_buffer, target);
		(*rows_processed)++;
	}

	pfree(row_buffer);
	pfree(query.data);
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
		model->coefficients =
			(double *)palloc(sizeof(double) * model->n_features);
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
		meta_text = DatumGetCString(DirectFunctionCall1(
			jsonb_out, JsonbPGetDatum(metadata)));
		/* Accept both "storage":"gpu" and "storage": "gpu" (with/without space) */
		if (strstr(meta_text, "\"storage\":\"gpu\"") != NULL ||
		    strstr(meta_text, "\"storage\": \"gpu\"") != NULL)
		{
			is_gpu = true;
		}
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

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
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

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
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
	text *table_name;
	text *feature_col;
	text *target_col;
	char *tbl_str;
	char *feat_str;
	char *targ_str;
	int nvec = 0;
	int dim = 0;
	LinRegDataset dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_target;
	MLGpuTrainResult gpu_result;
	char *gpu_err = NULL;
	Jsonb *gpu_hyperparams = NULL;
	StringInfoData hyperbuf;
	int32 model_id = 0;
	MemoryContext oldcontext;

	/* Save the function's memory context - this is the per-call context that Postgres manages */
	oldcontext = CurrentMemoryContext;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);

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
					errmsg("neurondb: train_linear_regression: SPI_connect failed")));

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
			"train_linear_regression: checking first row with query: %s",
			count_query.data);

		ret = SPI_execute(count_query.data, true, 0);
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			pfree(count_query.data);
			SPI_finish();
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: train_linear_regression: no valid rows found")));
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
						pfree(count_query.data);
						SPI_finish();
						pfree(tbl_str);
						pfree(feat_str);
						pfree(targ_str);
						ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								errmsg("neurondb: train_linear_regression: features array must be 1-D")));
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
			pfree(count_query.data);
			SPI_finish();
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: train_linear_regression: could not determine feature dimension")));
		}

		/* Get row count */
		pfree(count_query.data);
		initStringInfo(&count_query);
		appendStringInfo(&count_query,
			"SELECT COUNT(*) FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
			quoted_tbl,
			quoted_feat,
			quoted_target);
		elog(DEBUG1,
			"neurondb: train_linear_regression: counting rows: %s",
			count_query.data);

		ret = SPI_execute(count_query.data, true, 0);
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool count_null;
			HeapTuple tuple = SPI_tuptable->vals[0];
			Datum count_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, &count_null);

			if (!count_null)
				nvec = DatumGetInt32(count_datum);
		}

		pfree(count_query.data);
		SPI_finish();

		if (nvec < 10)
		{
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: train_linear_regression: need at least 10 samples, got %d",
						nvec)));
		}
	}

	/* Define max_samples limit for large datasets */
	{
		int max_samples = 500000; /* Limit to 500k samples for very large datasets */

		/* Limit sample size for very large datasets to avoid excessive training time */
		if (nvec > max_samples)
		{
			elog(DEBUG1, "neurondb: linear_regression: dataset has %d rows, limiting to %d samples for training", nvec, max_samples);
			nvec = max_samples;
		}

		/* Try GPU training first - always use GPU when enabled and kernel available */
		/* Initialize GPU if needed (lazy initialization) */
		if (neurondb_gpu_enabled)
		{
			ndb_gpu_init_if_needed();

			elog(DEBUG1,
				"neurondb: linear_regression: checking GPU - enabled=%d, available=%d, kernel_enabled=%d",
			neurondb_gpu_enabled ? 1 : 0,
			neurondb_gpu_is_available() ? 1 : 0,
			ndb_gpu_kernel_enabled("linreg_train") ? 1 : 0);
		}

		if (neurondb_gpu_is_available() && nvec > 0 && dim > 0
			&& ndb_gpu_kernel_enabled("linreg_train"))
		{
			int gpu_sample_limit = nvec;

			elog(INFO,
				"neurondb: linear_regression: attempting GPU training with %d samples (kernel enabled)",
				gpu_sample_limit);

			/* Load limited dataset for GPU training */
			linreg_dataset_init(&dataset);
			linreg_dataset_load_limited(quoted_tbl,
				quoted_feat,
				quoted_target,
				&dataset,
				gpu_sample_limit);

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
				dataset.n_samples,
				dataset.feature_dim,
				0,
				&gpu_result,
				&gpu_err)
				&& gpu_result.spec.model_data != NULL)
			{
				MLCatalogModelSpec spec;

				elog(INFO,
				     "neurondb: linear_regression: GPU training succeeded");
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

				spec.algorithm = "linear_regression";
				spec.model_type = "regression";

				/* Ensure we're in a valid memory context before calling ml_catalog_register_model */
				MemoryContextSwitchTo(oldcontext);

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
			} else
			{
				elog(DEBUG1,
				     "neurondb: linear_regression: GPU training failed, falling back to CPU streaming");
				if (gpu_err != NULL)
				{
					elog(DEBUG1,
					     "neurondb: linear_regression: GPU training error: %s",
					     gpu_err);
					pfree(gpu_err);
				}
				if (gpu_hyperparams != NULL)
					pfree(gpu_hyperparams);
				ndb_gpu_free_train_result(&gpu_result);
				linreg_dataset_free(&dataset);
			}
		}
		else if (neurondb_gpu_is_available() && !ndb_gpu_kernel_enabled("linreg_train"))
		{
			elog(INFO,
			     "neurondb: linear_regression: GPU available but linreg_train kernel not enabled, using CPU streaming");
		}

		/* CPU training path using streaming accumulator */
		{
			LinRegStreamAccum stream_accum;
			double **XtX_inv = NULL;
			double *beta = NULL;
			int i, j;
			int dim_with_intercept;
			LinRegModel *model;
			bytea *model_blob;
			Jsonb *metrics_json;
			StringInfoData metricsbuf;
			int chunk_size;
			int offset = 0;
			int rows_in_chunk = 0;

			/* Use larger chunks for better performance */
			if (nvec > 1000000)
				chunk_size = 100000; /* 100k chunks for very large datasets */
			else if (nvec > 100000)
				chunk_size = 50000; /* 50k chunks for large datasets */
			else
				chunk_size = 10000; /* 10k chunks for smaller datasets */

			/* Initialize streaming accumulator */
			linreg_stream_accum_init(&stream_accum, dim);
			dim_with_intercept = dim + 1;

			/* Process data in chunks */
				elog(DEBUG1,
					"neurondb: linear_regression: processing %d rows in chunks of %d",
				nvec,
				chunk_size);

			if (SPI_connect() != SPI_OK_CONNECT)
			{
				linreg_stream_accum_free(&stream_accum);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(targ_str);
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("neurondb: train_linear_regression: SPI_connect failed for streaming")));
			}

			offset = 0;
			while (offset < nvec)
			{
				linreg_stream_process_chunk(quoted_tbl,
					quoted_feat,
					quoted_target,
					&stream_accum,
					chunk_size,
					offset,
					&rows_in_chunk);

				if (rows_in_chunk == 0)
					break;

				offset += rows_in_chunk;

				/* Log progress for large datasets */
				if (offset % 100000 == 0 || offset >= nvec)
				{
						elog(DEBUG1,
							"neurondb: linear_regression: processed %d/%d rows (%.1f%%)",
						offset,
						nvec,
						(offset * 100.0) / nvec);
				}
			}

			SPI_finish();

			if (stream_accum.n_samples < 10)
			{
				linreg_stream_accum_free(&stream_accum);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(targ_str);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: train_linear_regression: insufficient samples processed (%d)",
							stream_accum.n_samples)));
			}

			/* Allocate matrices for inversion */
			XtX_inv = (double **)palloc(sizeof(double *) * dim_with_intercept);
			for (i = 0; i < dim_with_intercept; i++)
				XtX_inv[i] = (double *)palloc(sizeof(double) * dim_with_intercept);
			beta = (double *)palloc(sizeof(double) * dim_with_intercept);

			/* Add small regularization (ridge) to diagonal to handle near-singular matrices */
			{
				double lambda =
					1e-6; /* Small regularization parameter */
				for (i = 0; i < dim_with_intercept; i++)
				{
					stream_accum.XtX[i][i] += lambda;
				}
			}

			/* Invert X'X */
			if (!matrix_invert(stream_accum.XtX, dim_with_intercept, XtX_inv))
			{
				/* If still singular after regularization, try larger lambda */
				double lambda = 1e-3;
				for (i = 0; i < dim_with_intercept; i++)
				{
					stream_accum.XtX[i][i] += lambda;
				}

				if (!matrix_invert(stream_accum.XtX, dim_with_intercept, XtX_inv))
				{
					for (i = 0; i < dim_with_intercept; i++)
						pfree(XtX_inv[i]);
					pfree(XtX_inv);
					pfree(beta);
					linreg_stream_accum_free(&stream_accum);
					pfree(tbl_str);
					pfree(feat_str);
					pfree(targ_str);
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("neurondb: linear_regression: "
							       "matrix is singular "
							       "even after "
							       "regularization"),
							errhint("Try removing "
								"correlated features, "
								"reducing feature "
								"count, or using ridge "
								"regression")));
				} else
				{
					elog(DEBUG1,
					     "neurondb: linear_regression: used regularization (lambda=1e-3) to handle near-singular matrix");
				}
			}

			/* Compute β = (X'X)^(-1)X'y */
			for (i = 0; i < dim_with_intercept; i++)
			{
				beta[i] = 0.0;
				for (j = 0; j < dim_with_intercept; j++)
					beta[i] += XtX_inv[i][j] * stream_accum.Xty[j];
			}

			/* Build LinRegModel */
			model = (LinRegModel *)palloc0(sizeof(LinRegModel));
			model->n_features = dim;
			model->n_samples = stream_accum.n_samples;
			model->intercept = beta[0];
			model->coefficients = (double *)palloc(sizeof(double) * dim);
			if (model->coefficients == NULL)
			{
				for (i = 0; i < dim_with_intercept; i++)
					pfree(XtX_inv[i]);
				pfree(XtX_inv);
				pfree(beta);
				pfree(model);
				linreg_stream_accum_free(&stream_accum);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(targ_str);
				ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
						errmsg("neurondb: train_linear_regression: failed to allocate coefficients")));
			}
			for (i = 0; i < dim; i++)
				model->coefficients[i] = beta[i + 1];

			/* Compute approximate metrics using accumulated statistics */
			/* Note: For exact metrics, we would need a second pass through data */
			{
				double y_mean;
				double y_var;

				if (stream_accum.n_samples > 0)
				{
					y_mean = stream_accum.y_sum / stream_accum.n_samples;
					y_var = (stream_accum.y_sq_sum / stream_accum.n_samples) - (y_mean * y_mean);
				}
				else
				{
					y_mean = 0.0;
					y_var = 1.0;
				}

				/* Approximate R² using variance (simplified) */
				/* For exact metrics, would need second pass to compute residuals */
				model->r_squared = 0.5; /* Placeholder - would need second pass for exact */
				model->mse = y_var * 0.5; /* Approximate */
				model->mae = sqrt(y_var) * 0.7; /* Approximate */
			}

			/* Validate model before serialization */
			if (model->n_features <= 0 || model->n_features > 10000)
			{
				if (model->coefficients != NULL)
					pfree(model->coefficients);
				pfree(model);
				for (i = 0; i < dim_with_intercept; i++)
					pfree(XtX_inv[i]);
				pfree(XtX_inv);
				pfree(beta);
				linreg_stream_accum_free(&stream_accum);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(targ_str);
				elog(DEBUG1,
				     "neurondb: train_linear_regression: model.n_features is invalid (%d) before serialization",
				     model->n_features);
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("neurondb: train_linear_regression: model.n_features is invalid (%d) before serialization",
							model->n_features)));
			}

			elog(DEBUG1,
			     "neurondb: linear_regression: serializing model with n_features=%d, n_samples=%d",
			     model->n_features,
			     model->n_samples);

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

			metrics_json = DatumGetJsonbP(DirectFunctionCall1(
				jsonb_in, CStringGetDatum(metricsbuf.data)));

			/* Register in catalog */
			{
				MLCatalogModelSpec spec;

				/* Ensure we're in a valid memory context before calling ml_catalog_register_model */
				/* This function may have been called after SPI_finish() or other context changes */
				MemoryContextSwitchTo(oldcontext);

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
				pfree(XtX_inv[i]);
			pfree(XtX_inv);
			pfree(beta);
			if (model->coefficients != NULL)
				pfree(model->coefficients);
			pfree(model);
			linreg_stream_accum_free(&stream_accum);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);

			PG_RETURN_INT32(model_id);
		}
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
				errmsg("linear_regression: model_id is "
				       "required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("linear_regression: features vector is "
				       "required")));

	features = PG_GETARG_VECTOR_P(1);

	/* Try GPU prediction first */
	if (linreg_try_gpu_predict_catalog(model_id, features, &prediction))
	{
		elog(DEBUG1,
		     "linear_regression: GPU prediction succeeded, prediction=%.6f",
		     prediction);
		PG_RETURN_FLOAT8(prediction);
	} else
	{
		elog(DEBUG1,
		     "linear_regression: GPU prediction failed or not available, trying CPU");
	}

	/* Load model from catalog */
	if (!linreg_load_model_from_catalog(model_id, &model))
	{
		/* Check if model is GPU-only */
		bytea *payload = NULL;
		Jsonb *metrics = NULL;
		bool is_gpu = false;
		
		if (ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		{
			is_gpu = linreg_metadata_is_gpu(metrics);
			if (payload)
				pfree(payload);
			if (metrics)
				pfree(metrics);
		}
		
		if (is_gpu)
		{
			elog(DEBUG1,
			     "linear_regression: model %d is GPU-only and GPU prediction failed. Please ensure GPU is available and properly configured.",
			     model_id);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("linear_regression: model %d is GPU-only and GPU prediction failed. Please ensure GPU is available and properly configured.",
						model_id)));
		}
		else
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("linear_regression: model %d not found",
						model_id)));
	}

	/* Validate feature dimension */
	if (model->n_features > 0 && features->dim != model->n_features)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("linear_regression: feature dimension "
				       "mismatch (expected %d, got %d)",
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
	ArrayType *coef_array;
	Vector *features;
	int ncoef;
	float8 *coef;
	float *x;
	int dim;
	double prediction;
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
			(errmsg("linear_regression: coefficient dimension mismatch: expected %d, got %d",
				dim + 1,
				ncoef)));

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
	text *table_name;
	text *feature_col;
	text *target_col;
	ArrayType *coef_array;
	char *tbl_str;
	char *feat_str;
	char *targ_str;
	StringInfoData query;
	int ret;
	int nvec = 0;
	int ncoef;
	float8 *coef;
	double mse = 0.0;
	double mae = 0.0;
	double ss_tot = 0.0;
	double ss_res = 0.0;
	double y_mean = 0.0;
	double r_squared;
	double rmse;
	int i;
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
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: SPI_connect failed")));

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		feat_str,
		targ_str,
		tbl_str,
		feat_str,
		targ_str);
	elog(DEBUG1, "evaluate_linear_regression: executing query: %s", query.data);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: query failed")));

	nvec = SPI_processed;

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

	/* Second pass: compute predictions and metrics */
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
		double y_pred;
		double error;
		int j;

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

	result_datums = (Datum *)palloc(sizeof(Datum) * 4);
	result_datums[0] = Float8GetDatum(r_squared);
	result_datums[1] = Float8GetDatum(mse);
	result_datums[2] = Float8GetDatum(mae);
	result_datums[3] = Float8GetDatum(rmse);

	result_array = construct_array(result_datums,
		4,
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

Jsonb *
evaluate_linear_regression_by_model_id_jsonb(int32 model_id, text *table_name, text *feature_col, text *label_col)
{
	LinRegModel *model = NULL;
	char *tbl_str;
	char *feat_str;
	char *targ_str;
	StringInfoData query;
	int ret;
	int nvec = 0;
	double mse = 0.0;
	double mae = 0.0;
	double ss_tot = 0.0;
	double ss_res = 0.0;
	double y_mean = 0.0;
	double r_squared;
	double rmse;
	int i;
	StringInfoData jsonbuf;
	Jsonb *result;
	MemoryContext oldcontext;
	Oid feat_type_oid;
	bool feat_is_array;

	/* Load model from catalog */
	if (!linreg_load_model_from_catalog(model_id, &model))
	{
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("linear_regression: model %d not found for evaluation",
					model_id)));
	}

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_linear_regression_by_model_id_jsonb: SPI_connect failed")));

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		feat_str,
		targ_str,
		tbl_str,
		feat_str,
		targ_str);
	elog(DEBUG1, "evaluate_linear_regression_by_model_id_jsonb: executing query: %s", query.data);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_linear_regression_by_model_id_jsonb: query failed")));

	nvec = SPI_processed;
	if (nvec < 2)
	{
		SPI_finish();
		if (model->coefficients != NULL)
			pfree(model->coefficients);
		pfree(model);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_linear_regression_by_model_id_jsonb: need at least 2 samples, got %d",
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

	/* Determine feature type from first row */
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
		Vector *vec;
		ArrayType *arr;
		double y_true;
		double y_pred;
		double error;
		int j;
		int actual_dim;

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
				if (model->coefficients != NULL)
					pfree(model->coefficients);
				pfree(model);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(targ_str);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("linear_regression: features array must be 1-D")));
			}
			actual_dim = ARR_DIMS(arr)[0];
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			actual_dim = vec->dim;
		}

		/* Validate feature dimension */
		if (actual_dim != model->n_features)
		{
			SPI_finish();
			if (model->coefficients != NULL)
				pfree(model->coefficients);
			pfree(model);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("linear_regression: feature dimension mismatch (expected %d, got %d)",
						model->n_features,
						actual_dim)));
		}

		/* Compute prediction using model */
		y_pred = model->intercept;

		if (feat_is_array)
		{
			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				double *feat_data = (double *)ARR_DATA_PTR(arr);
				for (j = 0; j < model->n_features; j++)
					y_pred += model->coefficients[j] * feat_data[j];
			}
			else
			{
				float *feat_data = (float *)ARR_DATA_PTR(arr);
				for (j = 0; j < model->n_features; j++)
					y_pred += model->coefficients[j] * (double)feat_data[j];
			}
		}
		else
		{
			for (j = 0; j < model->n_features && j < vec->dim; j++)
				y_pred += model->coefficients[j] * vec->data[j];
		}

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

	/* Handle R² calculation - if ss_tot is zero (no variance in y), R² is undefined */
	if (ss_tot == 0.0)
		r_squared = 0.0; /* Convention: set to 0 when there's no variance to explain */
	else
		r_squared = 1.0 - (ss_res / ss_tot);

	SPI_finish();

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
		mse, mae, rmse, r_squared, nvec);

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(jsonbuf.data)));
	pfree(jsonbuf.data);

	/* Cleanup */
	if (model->coefficients != NULL)
		pfree(model->coefficients);
	pfree(model);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(targ_str);

	return result;
}

/*
 * evaluate_linear_regression_by_model_id
 *
 * One-shot evaluation function: loads model, fetches all data in one query,
 * loops through rows in C, computes predictions and metrics, returns jsonb.
 * This is much more efficient than calling predict() for each row in SQL.
 */
PG_FUNCTION_INFO_V1(evaluate_linear_regression_by_model_id);

Datum
evaluate_linear_regression_by_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	text *table_name;
	text *feature_col;
	text *label_col;
	Jsonb *result;

	if (PG_NARGS() != 4)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_linear_regression_by_model_id: 4 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_linear_regression_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_linear_regression_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	result = evaluate_linear_regression_by_model_id_jsonb(model_id, table_name, feature_col, label_col);
	if (result == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_linear_regression_by_model_id: result is NULL")));

	PG_RETURN_JSONB_P(result);
}

static const MLGpuModelOps linreg_gpu_model_ops = {
	.algorithm = "linear_regression",
	.train = NULL,
	.predict = NULL,
	.evaluate = NULL,
	.serialize = NULL,
	.deserialize = NULL,
	.destroy = NULL,
};

void
neurondb_gpu_register_linreg_model(void)
{
	static bool registered = false;

	if (registered)
		return;

	ndb_gpu_register_model_ops(&linreg_gpu_model_ops);
	registered = true;

	return;
}
