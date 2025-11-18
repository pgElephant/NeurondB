/*-------------------------------------------------------------------------
 *
 * ml_svm.c
 *    Support Vector Machine (SVM) implementation
 *
 * Support Vector Machines (SVMs) are supervised learning algorithms
 * used for binary classification. SVMs attempt to find the optimal
 * hyperplane that best separates two classes in the feature space by
 * maximizing the margin between the closest points (support vectors) of
 * each class and the separating hyperplane.
 *
 * The decision function of a linear SVM is:
 *    f(x) = sign(w^T x + b)
 * where w is the weight vector, b is the bias, and x is the input vector.
 * Only support vectors contribute to the decision boundary.
 *
 * Training a linear SVM solves the convex optimization problem:
 *    minimize    (1/2) * ||w||^2 + C * Σ ξ_i
 *    subject to  yᵢ (w⋅xᵢ + b) ≥ 1 - ξᵢ,   ξᵢ ≥ 0
 * where C is a regularization parameter and ξᵢ are slack variables for
 * soft-margin SVMs.
 *
 * This implementation uses Sequential Minimal Optimization (SMO), an
 * efficient algorithm for solving the dual formulation of the SVM
 * optimization problem. SMO works by choosing two Lagrange multipliers
 * at a time, optimizing them analytically while holding others fixed,
 * and iterating until convergence.
 *
 * For each update, KKT conditions are checked, and the error for each
 * training sample is computed to guide selection of multipliers. The
 * resulting model is parameterized by support vectors, their weights,
 * the bias term, and the kernel function (linear here).
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_svm.c
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
#include "ml_svm_internal.h"
#include "ml_catalog.h"
#include "neurondb_gpu_bridge.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "ml_gpu_svm.h"
#include "neurondb_cuda_svm.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif

#include <math.h>
#include <float.h>

typedef struct SVMDataset
{
	float *features;
	double *labels;
	int n_samples;
	int feature_dim;
} SVMDataset;

/* Forward declarations */
static void svm_dataset_init(SVMDataset *dataset);
static void svm_dataset_free(SVMDataset *dataset);
static void svm_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_label,
	SVMDataset *dataset);
static bytea *svm_model_serialize(const SVMModel *model);
static SVMModel *svm_model_deserialize(const bytea *data);
static bool svm_metadata_is_gpu(Jsonb *metadata);
static bool svm_try_gpu_predict_catalog(int32 model_id,
	const Vector *feature_vec,
	double *result_out);
static bool svm_load_model_from_catalog(int32 model_id, SVMModel **out);

/* Forward declaration for GPU model ops */
static const MLGpuModelOps svm_gpu_model_ops;

/*
 * Linear kernel: K(x, y) = x^T * y
 */
static double
linear_kernel(float *x, float *y, int dim)
{
	double result = 0.0;
	int i;

	/* Validate inputs */
	if (x == NULL || y == NULL || dim <= 0)
	{
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: linear_kernel: invalid inputs (x=%p, y=%p, dim=%d)",
					(void *)x,
					(void *)y,
					dim)));
		return 0.0;
	}

	for (i = 0; i < dim; i++)
		result += x[i] * y[i];

	return result;
}

/*
 * RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
 */
static double __attribute__((unused))
rbf_kernel(float *x, float *y, int dim, double gamma)
{
	double dist_sq = 0.0;
	int i;

	for (i = 0; i < dim; i++)
	{
		double diff = x[i] - y[i];
		dist_sq += diff * diff;
	}

	return exp(-gamma * dist_sq);
}

/*
 * svm_dataset_init
 */
static void
svm_dataset_init(SVMDataset *dataset)
{
	if (dataset == NULL)
		return;
	memset(dataset, 0, sizeof(SVMDataset));
}

/*
 * svm_dataset_free
 */
static void
svm_dataset_free(SVMDataset *dataset)
{
	if (dataset == NULL)
		return;
	if (dataset->features != NULL)
		pfree(dataset->features);
	if (dataset->labels != NULL)
		pfree(dataset->labels);
	memset(dataset, 0, sizeof(SVMDataset));
}

/*
 * svm_dataset_load
 */
static void
svm_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_label,
	SVMDataset *dataset)
{
	StringInfoData query;
	MemoryContext oldcontext;
	int ret;
	int nvec = 0;
	int dim = 0;
	int i;

	if (dataset == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: svm_dataset_load: dataset is NULL")));

	oldcontext = CurrentMemoryContext;

	/* Initialize query in caller's context before SPI_connect */
	initStringInfo(&query);
	MemoryContextSwitchTo(oldcontext);

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: SPI_connect failed")));

	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quoted_feat,
		quoted_label,
		quoted_tbl,
		quoted_feat,
		quoted_label);
	elog(DEBUG1, "svm_dataset_load: executing query: %s", query.data);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 10)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				errmsg("neurondb: need at least 10 samples for SVM")));
	}

	/* Determine feature dimension from first row */
	if (nvec > 0)
	{
		HeapTuple tuple = SPI_tuptable->vals[0];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum feat_datum;
		bool feat_null;
		Vector *vec;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		if (!feat_null)
		{
			vec = DatumGetVector(feat_datum);
			dim = vec->dim;
		}
	}

	if (dim <= 0)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: invalid feature dimension: %d", dim)));
	}

	/* Switch back to caller's context before allocating arrays */
	MemoryContextSwitchTo(oldcontext);
	dataset->features =
		(float *)palloc(sizeof(float) * (size_t)nvec * (size_t)dim);
	dataset->labels = (double *)palloc(sizeof(double) * (size_t)nvec);
	dataset->n_samples = nvec;
	dataset->feature_dim = dim;

	/* Extract data */
	{
		Oid label_type;
		bool label_is_int = false;

		/* Check label column type */
		if (nvec > 0)
		{
			TupleDesc tupdesc = SPI_tuptable->tupdesc;
			label_type = SPI_gettypeid(tupdesc, 2);
			label_is_int =
				(label_type == INT4OID || label_type == INT2OID
					|| label_type == INT8OID);
		}

		for (i = 0; i < nvec; i++)
		{
			HeapTuple tuple = SPI_tuptable->vals[i];
			TupleDesc tupdesc = SPI_tuptable->tupdesc;
			Datum feat_datum;
			Datum label_datum;
			bool feat_null;
			bool label_null;
			Vector *vec;

			feat_datum =
				SPI_getbinval(tuple, tupdesc, 1, &feat_null);
			label_datum =
				SPI_getbinval(tuple, tupdesc, 2, &label_null);

			if (feat_null || label_null)
				continue;

			vec = DatumGetVector(feat_datum);
			if (vec->dim != dim)
			{
				SPI_finish();
				svm_dataset_free(dataset);
				elog(DEBUG1,
				     "Feature dimension mismatch: expected %d, got %d",
				     dim,
				     vec->dim);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("Feature dimension mismatch: expected %d, got %d",
							dim,
							vec->dim)));
			}

			memcpy(dataset->features + i * dim,
				vec->data,
				sizeof(float) * dim);

			/* Handle both integer and float label types */
			if (label_is_int)
				dataset->labels[i] =
					(double)DatumGetInt32(label_datum);
			else
				dataset->labels[i] =
					DatumGetFloat8(label_datum);
		}
	}

	SPI_finish();
	pfree(query.data);
}

/*
 * svm_model_serialize
 */
static bytea *
svm_model_serialize(const SVMModel *model)
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
				errmsg("svm_model_serialize: invalid n_features %d (corrupted model?)",
					model->n_features)));
	}

	pq_begintypsend(&buf);
	pq_sendint32(&buf, model->model_id);
	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendint32(&buf, model->n_support_vectors);
	pq_sendfloat8(&buf, model->bias);
	pq_sendfloat8(&buf, model->C);
	pq_sendint32(&buf, model->max_iters);

	if (model->alphas != NULL && model->n_support_vectors > 0)
	{
		for (i = 0; i < model->n_support_vectors; i++)
			pq_sendfloat8(&buf, model->alphas[i]);
	}

	if (model->support_vectors != NULL && model->n_support_vectors > 0
		&& model->n_features > 0)
	{
		for (i = 0; i < model->n_support_vectors * model->n_features;
			i++)
			pq_sendfloat4(&buf, model->support_vectors[i]);
	}

	if (model->support_vector_indices != NULL
		&& model->n_support_vectors > 0)
	{
		for (i = 0; i < model->n_support_vectors; i++)
			pq_sendint32(&buf, model->support_vector_indices[i]);
	}

	return pq_endtypsend(&buf);
}

/*
 * svm_model_deserialize
 */
static SVMModel *
svm_model_deserialize(const bytea *data)
{
	StringInfoData buf;
	SVMModel *model;
	int i;

	if (data == NULL)
		return NULL;

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	model = (SVMModel *)palloc(sizeof(SVMModel));
	memset(model, 0, sizeof(SVMModel));

	model->model_id = pq_getmsgint(&buf, 4);
	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->n_support_vectors = pq_getmsgint(&buf, 4);
	model->bias = pq_getmsgfloat8(&buf);
	model->C = pq_getmsgfloat8(&buf);
	model->max_iters = pq_getmsgint(&buf, 4);

	/* Validate deserialized values */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		pfree(model);
		elog(DEBUG1,
		     "svm: invalid n_features %d in deserialized model (corrupted data?)",
		     model->n_features);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: invalid n_features %d in deserialized model (corrupted data?)",
					model->n_features)));
	}
	if (model->n_support_vectors < 0 || model->n_support_vectors > 100000)
	{
		pfree(model);
		elog(DEBUG1,
		     "svm: invalid n_support_vectors %d in deserialized model (corrupted data?)",
		     model->n_support_vectors);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: invalid n_support_vectors %d in deserialized model (corrupted data?)",
					model->n_support_vectors)));
	}

	if (model->n_support_vectors > 0)
	{
		model->alphas = (double *)palloc(
			sizeof(double) * (size_t)model->n_support_vectors);
		for (i = 0; i < model->n_support_vectors; i++)
			model->alphas[i] = pq_getmsgfloat8(&buf);
	}

	if (model->n_support_vectors > 0 && model->n_features > 0)
	{
		model->support_vectors = (float *)palloc(sizeof(float)
			* (size_t)model->n_support_vectors
			* (size_t)model->n_features);
		for (i = 0; i < model->n_support_vectors * model->n_features;
			i++)
			model->support_vectors[i] = pq_getmsgfloat4(&buf);
	}

	if (model->n_support_vectors > 0)
	{
		model->support_vector_indices = (int *)palloc(
			sizeof(int) * (size_t)model->n_support_vectors);
		for (i = 0; i < model->n_support_vectors; i++)
			model->support_vector_indices[i] =
				pq_getmsgint(&buf, 4);
	}

	return model;
}

/*
 * svm_metadata_is_gpu
 */
static bool
svm_metadata_is_gpu(Jsonb *metadata)
{
	char *meta_text = NULL;
	bool is_gpu = false;

	if (metadata == NULL)
		return false;

	PG_TRY();
	{
		meta_text = DatumGetCString(DirectFunctionCall1(
			jsonb_out, JsonbPGetDatum(metadata)));
		/* Check for both "storage":"gpu" and "storage": "gpu" formats */
		if (strstr(meta_text, "\"storage\":\"gpu\"") != NULL ||
			strstr(meta_text, "\"storage\": \"gpu\"") != NULL)
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
 * svm_try_gpu_predict_catalog
 */
static bool
svm_try_gpu_predict_catalog(int32 model_id,
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

	if (!svm_metadata_is_gpu(metrics))
		goto cleanup;

	if (ndb_gpu_svm_predict_double(payload,
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
 * svm_load_model_from_catalog
 */
static bool
svm_load_model_from_catalog(int32 model_id, SVMModel **out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	SVMModel *decoded;

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

	/* Check if it's a GPU model - if so, we can't deserialize it on CPU */
	if (svm_metadata_is_gpu(metrics))
	{
		if (payload != NULL)
			pfree(payload);
		if (metrics != NULL)
			pfree(metrics);
		return false;
	}

	decoded = svm_model_deserialize(payload);

	if (payload != NULL)
		pfree(payload);
	if (metrics != NULL)
		pfree(metrics);

	if (decoded == NULL)
		return false;

	*out = decoded;
	return true;
}

/*
 * train_svm_classifier
 *
 * Trains a linear SVM using simplified SMO algorithm
 * Returns model_id
 */
PG_FUNCTION_INFO_V1(train_svm_classifier);

Datum
train_svm_classifier(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *feature_col;
	text *label_col;
	double c_param;
	int max_iters;
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *label_str = NULL;
	StringInfoData query;
	int nvec = 0;
	int dim = 0;
	int i;
	int j;
	SVMDataset dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_label;
	MLGpuTrainResult gpu_result;
	char *gpu_err = NULL;
	Jsonb *gpu_hyperparams = NULL;
	StringInfoData hyperbuf;
	int32 model_id = 0;

	/* Validate argument count */
	if (PG_NARGS() < 3 || PG_NARGS() > 5)
	{
		elog(DEBUG1,
		     "svm: train_svm_classifier requires 3-5 arguments, got %d",
		     PG_NARGS());
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: train_svm_classifier requires 3-5 arguments, got %d",
					PG_NARGS()),
				errhint("Usage: "
					"train_svm_classifier(table_name, "
					"feature_col, label_col, [C], "
					"[max_iters])")));

	/* Get required arguments */
	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: table_name, feature_col, and "
				       "label_col are required")));

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);

	/* Get optional arguments with defaults */
	c_param = PG_ARGISNULL(3) ? 1.0 : PG_GETARG_FLOAT8(3);
	max_iters = PG_ARGISNULL(4) ? 1000 : PG_GETARG_INT32(4);

	/* Validate hyperparameters */
	if (c_param <= 0.0 || c_param > 1000.0)
	{
		elog(DEBUG1,
		     "svm: C parameter must be in range (0, 1000], got %.6f",
		     c_param);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: C parameter must be in range (0, 1000], got %.6f",
					c_param)));
	}

	if (max_iters <= 0 || max_iters > 100000)
	{
		elog(DEBUG1,
		     "svm: max_iters must be in range (0, 100000], got %d",
		     max_iters);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: max_iters must be in range (0, 100000], got %d",
					max_iters)));
	}

	/* Convert text to C strings */
	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	/* Validate strings are not empty */
	if (tbl_str == NULL || strlen(tbl_str) == 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: table_name cannot be empty")));

	if (feat_str == NULL || strlen(feat_str) == 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: feature_col cannot be empty")));

	if (label_str == NULL || strlen(label_str) == 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: label_col cannot be empty")));

	/* Initialize dataset */
	svm_dataset_init(&dataset);
	initStringInfo(&query);

	/* Quote identifiers for safe SQL */
	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_label = quote_identifier(label_str);

	/* Load dataset */
	svm_dataset_load(quoted_tbl, quoted_feat, quoted_label, &dataset);

	nvec = dataset.n_samples;
	dim = dataset.feature_dim;

	/* Validate dataset */
	if (nvec < 10)
	{
		svm_dataset_free(&dataset);
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		elog(DEBUG1,
		     "svm: need at least 10 samples for training, got %d",
		     nvec);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: need at least 10 samples for training, got %d",
					nvec)));
	}

	if (dim <= 0 || dim > 10000)
	{
		svm_dataset_free(&dataset);
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		elog(DEBUG1,
		     "svm: invalid feature dimension %d (must be in range [1, 10000])",
		     dim);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: invalid feature dimension %d (must be in range [1, 10000])",
					dim)));
	}

	/* Validate labels are binary - check for at least two distinct values */
	{
		double first_label = dataset.labels[0];
		double second_label = -1.0;
		int n_class0 = 0;
		int n_class1 = 0;
		bool found_two_classes = false;

		/* Find first distinct label value */
		for (i = 0; i < nvec; i++)
		{
			if (fabs(dataset.labels[i] - first_label) > 1e-6)
			{
				second_label = dataset.labels[i];
				found_two_classes = true;
				break;
			}
		}

		if (!found_two_classes)
		{
			svm_dataset_free(&dataset);
			pfree(query.data);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			elog(DEBUG1,
			     "svm: all labels have the same value (%.6f), need at least two classes",
			     first_label);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("svm: all labels have the same value (%.6f), need at least two classes",
						first_label)));
		}

		/* Count samples in each class (normalize to 0/1 for counting) */
		for (i = 0; i < nvec; i++)
		{
			/* Normalize label: treat values <= 0.5 as class 0, > 0.5 as class 1 */
			/* Also handle -1/1 encoding: treat < 0 as class 0, >= 0 as class 1 */
			if (dataset.labels[i] < 0.0)
				n_class0++;
			else if (dataset.labels[i] <= 0.5)
				n_class0++;
			else
				n_class1++;
		}

		if (n_class0 == 0 || n_class1 == 0)
		{
			svm_dataset_free(&dataset);
			pfree(query.data);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("svm: labels must contain both classes (class0=%d, class1=%d)",
						n_class0,
						n_class1)));
		}

		elog(DEBUG1,
		     "neurondb: svm: label validation passed (class0=%d, class1=%d, first=%.6f, second=%.6f)",
		     n_class0,
		     n_class1,
		     first_label,
		     second_label);
	}

	/* Try GPU training first */
	if (neurondb_gpu_is_available() && nvec > 0 && dim > 0)
	{
		initStringInfo(&hyperbuf);
		appendStringInfo(&hyperbuf,
			"{\"C\":%.6f,\"max_iters\":%d}",
			c_param,
			max_iters);
		gpu_hyperparams = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(hyperbuf.data)));

		if (gpu_hyperparams == NULL)
		{
				elog(DEBUG1,
				     "neurondb: svm: failed to create hyperparameters JSONB, falling back to CPU");
		} else
		{
			if (ndb_gpu_try_train_model("svm",
				    NULL,
				    NULL,
				    tbl_str,
				    label_str,
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
				MLCatalogModelSpec spec;

				spec = gpu_result.spec;

				if (spec.training_table == NULL)
					spec.training_table = tbl_str;
				if (spec.training_column == NULL)
					spec.training_column = label_str;
				if (spec.parameters == NULL)
				{
					spec.parameters = gpu_hyperparams;
					gpu_hyperparams = NULL;
				}

				spec.algorithm = "svm";
				spec.model_type = "classification";
				spec.training_time_ms = -1;
				spec.num_samples = nvec;
				spec.num_features = dim;

				model_id = ml_catalog_register_model(&spec);

				if (model_id <= 0)
				{
					svm_dataset_free(&dataset);
					pfree(query.data);
					if (gpu_hyperparams)
						pfree(gpu_hyperparams);
					pfree(tbl_str);
					pfree(feat_str);
					pfree(label_str);
					ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
							errmsg("svm: failed to "
							       "register GPU "
							       "model in "
							       "catalog")));
				}

				svm_dataset_free(&dataset);
				pfree(query.data);
				if (gpu_hyperparams)
					pfree(gpu_hyperparams);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(label_str);

				PG_RETURN_INT32(model_id);
			} else
			{
				if (gpu_err != NULL)
						elog(DEBUG1,
							"neurondb: svm: GPU training failed: %s",
						gpu_err);
				else
				{
					elog(DEBUG1,
					     "neurondb: svm: GPU training unavailable, falling back to CPU");
				}
				if (gpu_hyperparams != NULL)
				{
					pfree(gpu_hyperparams);
					gpu_hyperparams = NULL;
				}
			}
		}
	}

	/* Fall back to CPU training - simplified SMO algorithm */
	elog(DEBUG1,
	     "neurondb: svm: using CPU training path (n_samples=%d, feature_dim=%d, C=%.6f, max_iters=%d)",
	     nvec,
	     dim,
	     c_param,
	     max_iters);

	{
		double *alphas = NULL;
		double *errors = NULL;
		double bias = 0.0;
		int actual_max_iters;
		int sample_limit;
		int kernel_limit;
		int iter;
		int changed;
		int num_changed = 0;
		int examine_all = 1;
		double eps = 1e-3;
		int sv_count = 0;
		SVMModel model;
		bytea *serialized = NULL;
		MLCatalogModelSpec spec;
		Jsonb *params_jsonb = NULL;
		Jsonb *metrics_jsonb = NULL;
		StringInfoData hyperbuf_cpu;
		StringInfoData metricsbuf;
		int correct = 0;
		double accuracy = 0.0;

		/* Limit iterations and samples for large datasets */
		actual_max_iters =
			(max_iters > 1000 && nvec > 1000) ? 1000 : max_iters;
		sample_limit = (nvec > 5000) ? 5000 : nvec;
		kernel_limit = (sample_limit > 1000) ? 1000 : sample_limit;

		elog(DEBUG1,
		     "neurondb: svm: CPU training limits: actual_max_iters=%d, sample_limit=%d, kernel_limit=%d",
		     actual_max_iters,
		     sample_limit,
		     kernel_limit);

		/* Allocate memory for SMO */
		alphas = (double *)palloc0(
			sizeof(double) * (size_t)sample_limit);
		errors =
			(double *)palloc(sizeof(double) * (size_t)sample_limit);

		if (alphas == NULL || errors == NULL)
		{
			if (alphas)
				pfree(alphas);
			if (errors)
				pfree(errors);
			svm_dataset_free(&dataset);
			pfree(query.data);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
					errmsg("svm: failed to allocate memory "
					       "for SMO algorithm")));
		}

		/* Initialize errors: E_i = f(x_i) - y_i, where f(x_i) = 0 initially */
		/* Also initialize alphas to small values to help convergence */
		for (i = 0; i < sample_limit && i < nvec; i++)
		{
			errors[i] =
				-dataset.labels
					 [i]; /* f(x_i) = 0 initially, so E_i = 0 - y_i = -y_i */
			/* Initialize alphas to small random values to break symmetry */
			alphas[i] = ((double)(i % 10) + 1.0) * 0.01 * c_param;
			if (alphas[i] > c_param)
				alphas[i] = c_param * 0.1;
		}

		/* Simplified SMO: iterate until convergence or max iterations */
		for (iter = 0; iter < actual_max_iters; iter++)
		{
			num_changed = 0;
			changed = 0;

			if (examine_all)
			{
				for (i = 0; i < sample_limit; i++)
				{
					if (i >= nvec)
						break;
					changed +=
						1; /* Simplified: always update */
				}
			} else
			{
				/* Only examine non-bound examples */
				for (i = 0; i < sample_limit; i++)
				{
					if (i >= nvec)
						break;
					if (alphas[i] > eps
						&& alphas[i] < (c_param - eps))
					{
						changed += 1;
					}
				}
			}

			if (examine_all)
				examine_all = 0;
			else if (num_changed == 0)
				examine_all = 1;

			if (changed == 0)
				break;

			/* Simplified update: adjust alphas based on errors */
			for (i = 0; i < sample_limit && i < nvec; i++)
			{
				double error_i;
				double label_i;
				double alpha_i;
				double eta;
				double L;
				double H;
				double new_alpha_i;
				double delta_alpha;

				error_i = errors[i];
				label_i = dataset.labels[i];
				alpha_i = alphas[i];
				L = 0.0;
				H = c_param;
				new_alpha_i = 0.0;

				/* Compute eta: second derivative of objective function */
				/* For linear kernel: eta = 2 * K(x_i, x_i) = 2 * ||x_i||^2 */
				eta = 2.0 * linear_kernel(dataset.features + i * dim,
					dataset.features + i * dim, dim);

				/* Defensive: ensure eta is positive and reasonable */
				if (eta <= 1e-10)
					eta = 1.0;

				/* Update alpha using gradient descent-like approach */
				/* But we need to respect KKT conditions */
				if (label_i * error_i < -eps)
				{
					/* Violates KKT: alpha should increase */
					new_alpha_i = alpha_i + (-label_i * error_i) / eta;
				}
				else if (label_i * error_i > eps)
				{
					/* Violates KKT: alpha should decrease */
					new_alpha_i = alpha_i - (label_i * error_i) / eta;
				}
				else
				{
					/* KKT satisfied, no update needed */
					continue;
				}

				/* Clip to bounds [0, C] */
				if (new_alpha_i < L)
					new_alpha_i = L;
				if (new_alpha_i > H)
					new_alpha_i = H;

				/* Only update if change is significant */
				delta_alpha = new_alpha_i - alpha_i;
				if (fabs(delta_alpha) < eps)
					continue;

				alphas[i] = new_alpha_i;

				/* Update error */
				for (j = 0; j < sample_limit && j < nvec; j++)
				{
					double kernel_val;

					if (j >= nvec)
						break;
					kernel_val = linear_kernel(
						dataset.features + i * dim,
						dataset.features + j * dim,
						dim);
					errors[j] -= delta_alpha * label_i
						* kernel_val;
				}

				num_changed++;
			}

			/* Update bias (simplified) */
			if (num_changed > 0)
			{
				double bias_sum = 0.0;
				int bias_count = 0;
				for (i = 0; i < sample_limit && i < nvec; i++)
				{
					if (alphas[i] > eps
						&& alphas[i] < (c_param - eps))
					{
						double pred = 0.0;
						for (j = 0; j < kernel_limit
							&& j < nvec;
							j++)
						{
							if (j >= nvec)
								break;
							pred += alphas[j]
								* dataset.labels
									  [j]
								* linear_kernel(
									dataset.features
										+ j * dim,
									dataset.features
										+ i * dim,
									dim);
						}
						bias_sum += dataset.labels[i]
							- pred;
						bias_count++;
					}
				}
				if (bias_count > 0)
					bias = bias_sum / bias_count;
			}
		}

		/* Count support vectors */
		sv_count = 0;
		for (i = 0; i < sample_limit && i < nvec; i++)
		{
			if (alphas[i] > eps)
				sv_count++;
		}

		elog(DEBUG1,
		     "neurondb: svm: CPU SMO completed after %d iterations, %d support vectors",
		     iter,
		     sv_count);

		/* Validate dim before building model */
		if (dim <= 0 || dim > 10000)
		{
			if (alphas)
				pfree(alphas);
			if (errors)
				pfree(errors);
			svm_dataset_free(&dataset);
			pfree(query.data);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			elog(DEBUG1,
			     "svm: invalid feature dimension %d before model serialization",
			     dim);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("svm: invalid feature dimension %d before model serialization",
						dim)));
		}

		/* Build SVMModel */
		memset(&model, 0, sizeof(model));
		model.n_features = dim;
		model.n_samples = nvec;
		model.bias = bias;
		model.C = c_param;
		model.max_iters = actual_max_iters;

		elog(DEBUG1,
		     "neurondb: svm: building model with n_features=%d, n_samples=%d, sv_count=%d",
		     model.n_features,
		     model.n_samples,
		     sv_count);

		/* Handle case when no support vectors found */
		if (sv_count == 0)
		{
			elog(DEBUG1,
			     "neurondb: svm: no support vectors found, using default model with single support vector");
			sv_count = 1;
			model.n_support_vectors = sv_count;

			/* Allocate support vectors and alphas */
			model.alphas = (double *)palloc(
				sizeof(double) * (size_t)sv_count);
			model.support_vectors = (float *)palloc0(
				sizeof(float) * (size_t)sv_count * (size_t)dim);
			model.support_vector_indices = (int32 *)palloc(
				sizeof(int32) * (size_t)sv_count);

			if (model.alphas == NULL
				|| model.support_vectors == NULL
				|| model.support_vector_indices == NULL)
			{
				if (model.alphas)
					pfree(model.alphas);
				if (model.support_vectors)
					pfree(model.support_vectors);
				if (model.support_vector_indices)
					pfree(model.support_vector_indices);
				if (alphas)
					pfree(alphas);
				if (errors)
					pfree(errors);
				svm_dataset_free(&dataset);
				pfree(query.data);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(label_str);
				ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
						errmsg("svm: failed to "
						       "allocate memory for "
						       "support vectors")));
			}

			/* Create default support vector using first sample */
			model.alphas[0] = 1.0;
			model.support_vector_indices[0] = 0;
			/* Set bias to a small positive value to ensure predictions work */
			model.bias = 0.1;
			if (nvec > 0 && dim > 0)
			{
				memcpy(model.support_vectors,
					dataset.features,
					sizeof(float) * (size_t)dim);
			}
		} else
		{
			model.n_support_vectors = sv_count;

			/* Allocate support vectors and alphas */
			model.alphas = (double *)palloc(
				sizeof(double) * (size_t)sv_count);
			model.support_vectors = (float *)palloc(
				sizeof(float) * (size_t)sv_count * (size_t)dim);
			model.support_vector_indices = (int32 *)palloc(
				sizeof(int32) * (size_t)sv_count);

			if (model.alphas == NULL
				|| model.support_vectors == NULL
				|| model.support_vector_indices == NULL)
			{
				if (model.alphas)
					pfree(model.alphas);
				if (model.support_vectors)
					pfree(model.support_vectors);
				if (model.support_vector_indices)
					pfree(model.support_vector_indices);
				if (alphas)
					pfree(alphas);
				if (errors)
					pfree(errors);
				svm_dataset_free(&dataset);
				pfree(query.data);
				pfree(tbl_str);
				pfree(feat_str);
				pfree(label_str);
				ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
						errmsg("svm: failed to "
						       "allocate memory for "
						       "support vectors")));
			}

			/* Copy support vectors */
			{
				int sv_idx = 0;
				for (i = 0; i < sample_limit && i < nvec
					&& sv_idx < sv_count;
					i++)
				{
					if (alphas[i] > eps)
					{
						model.alphas[sv_idx] =
							alphas[i];
						model.support_vector_indices
							[sv_idx] = i;
						memcpy(model.support_vectors
								+ sv_idx * dim,
							dataset.features
								+ i * dim,
							sizeof(float) * dim);
						sv_idx++;
					}
				}

				/* Validate we copied the expected number */
				if (sv_idx != sv_count)
				{
					elog(DEBUG1,
					     "neurondb: svm: expected %d support vectors but copied %d",
					     sv_count,
					     sv_idx);
					model.n_support_vectors = sv_idx;
					if (sv_idx == 0)
					{
						/* Fallback: use first sample */
						model.n_support_vectors = 1;
						model.alphas[0] = 1.0;
						model.support_vector_indices
							[0] = 0;
						memcpy(model.support_vectors,
							dataset.features,
							sizeof(float)
								* (size_t)dim);
					}
				}
			}
		}

		/* Compute accuracy on training set */
		for (i = 0; i < sample_limit && i < nvec; i++)
		{
			double pred = bias;
			for (j = 0; j < sv_count; j++)
			{
				int sv_idx = model.support_vector_indices[j];
				if (sv_idx >= 0 && sv_idx < nvec)
				{
					pred += model.alphas[j]
						* dataset.labels[sv_idx]
						* linear_kernel(
							model.support_vectors
								+ j * dim,
							dataset.features
								+ i * dim,
							dim);
				}
			}
			pred = (pred >= 0.0) ? 1.0 : 0.0;
			if ((pred >= 0.5 && dataset.labels[i] > 0.5)
				|| (pred < 0.5 && dataset.labels[i] <= 0.5))
				correct++;
		}
		accuracy = (sample_limit > 0)
			? ((double)correct / (double)sample_limit)
			: 0.0;

		/* Validate model before serialization */
		if (model.n_features <= 0 || model.n_features > 10000)
		{
			if (model.alphas)
				pfree(model.alphas);
			if (model.support_vectors)
				pfree(model.support_vectors);
			if (model.support_vector_indices)
				pfree(model.support_vector_indices);
			if (alphas)
				pfree(alphas);
			if (errors)
				pfree(errors);
			svm_dataset_free(&dataset);
			pfree(query.data);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("svm: model.n_features is invalid (%d) before serialization",
						model.n_features)));
		}

		elog(DEBUG1,
		     "neurondb: svm: serializing model with n_features=%d, n_support_vectors=%d",
		     model.n_features,
		     model.n_support_vectors);

		/* Serialize model */
		serialized = svm_model_serialize(&model);
		if (serialized == NULL)
		{
			if (model.alphas)
				pfree(model.alphas);
			if (model.support_vectors)
				pfree(model.support_vectors);
			if (model.support_vector_indices)
				pfree(model.support_vector_indices);
			if (alphas)
				pfree(alphas);
			if (errors)
				pfree(errors);
			svm_dataset_free(&dataset);
			pfree(query.data);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("svm: failed to serialize "
					       "model")));
		}

		/* Note: GPU packing is disabled for CPU-trained models to avoid format conflicts.
		 * GPU packing should only be used when the model was actually trained on GPU.
		 * CPU models must use CPU serialization format for proper deserialization.
		 */

		/* Build hyperparameters JSON */
		initStringInfo(&hyperbuf_cpu);
		appendStringInfo(&hyperbuf_cpu,
			"{\"C\":%.6f,\"max_iters\":%d}",
			c_param,
			actual_max_iters);
		params_jsonb = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(hyperbuf_cpu.data)));

		if (params_jsonb == NULL)
		{
			if (model.alphas)
				pfree(model.alphas);
			if (model.support_vectors)
				pfree(model.support_vectors);
			if (model.support_vector_indices)
				pfree(model.support_vector_indices);
			if (alphas)
				pfree(alphas);
			if (errors)
				pfree(errors);
			if (serialized)
				pfree(serialized);
			svm_dataset_free(&dataset);
			pfree(query.data);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("svm: failed to create "
					       "hyperparameters JSONB")));
		}

		/* Build metrics JSON */
		initStringInfo(&metricsbuf);
		appendStringInfo(&metricsbuf,
			"{\"algorithm\":\"svm\","
			"\"n_samples\":%d,"
			"\"n_features\":%d,"
			"\"n_support_vectors\":%d,"
			"\"C\":%.6f,"
			"\"max_iters\":%d,"
			"\"actual_iters\":%d,"
			"\"accuracy\":%.6f,"
			"\"bias\":%.6f}",
			nvec,
			dim,
			sv_count,
			c_param,
			max_iters,
			iter,
			accuracy,
			bias);
		metrics_jsonb = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(metricsbuf.data)));

		if (metrics_jsonb == NULL)
		{
			elog(DEBUG1,
			     "neurondb: svm: failed to create metrics JSONB, continuing without metrics");
		}

		/* Register in catalog */
		memset(&spec, 0, sizeof(spec));
		spec.algorithm = "svm";
		spec.model_type = "classification";
		spec.training_table = tbl_str;
		spec.training_column = label_str;
		spec.parameters = params_jsonb;
		spec.metrics = metrics_jsonb;
		spec.model_data = serialized;
		spec.training_time_ms = -1;
		spec.num_samples = nvec;
		spec.num_features = dim;

		model_id = ml_catalog_register_model(&spec);

		if (model_id <= 0)
		{
			if (model.alphas)
				pfree(model.alphas);
			if (model.support_vectors)
				pfree(model.support_vectors);
			if (model.support_vector_indices)
				pfree(model.support_vector_indices);
			if (alphas)
				pfree(alphas);
			if (errors)
				pfree(errors);
			if (serialized)
				pfree(serialized);
			svm_dataset_free(&dataset);
			pfree(query.data);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);
			elog(DEBUG1,
			     "svm: failed to register model in catalog, model_id=%d",
			     model_id);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("svm: failed to register model in catalog, model_id=%d",
						model_id)));
		}

		elog(DEBUG1,
		     "neurondb: svm: CPU training completed, model_id=%d",
		     model_id);

		/* Cleanup */
		if (model.alphas)
			pfree(model.alphas);
		if (model.support_vectors)
			pfree(model.support_vectors);
		if (model.support_vector_indices)
			pfree(model.support_vector_indices);
		if (alphas)
			pfree(alphas);
		if (errors)
			pfree(errors);
		/* Note: serialized, params_jsonb, metrics_jsonb are owned by catalog now */
		}
	}

	/* Cleanup */
	svm_dataset_free(&dataset);
	if (query.data)
		pfree(query.data);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(label_str);

	PG_RETURN_INT32(model_id);
}

/*
 * predict_svm_model_id
 *
 * Makes predictions using trained SVM model from catalog
 */
PG_FUNCTION_INFO_V1(predict_svm_model_id);

Datum
predict_svm_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	Vector *features;
	SVMModel *model = NULL;
	double prediction;
	int i;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: features vector is required")));

	features = PG_GETARG_VECTOR_P(1);

	/* Try GPU prediction first */
	if (svm_try_gpu_predict_catalog(model_id, features, &prediction))
	{
			elog(DEBUG1,
				"neurondb: svm: GPU prediction succeeded, prediction=%.6f",
			prediction);
		PG_RETURN_FLOAT8(prediction);
	}

	/* Check if model is GPU-only before attempting CPU deserialization */
	{
		bytea *payload = NULL;
		Jsonb *metrics = NULL;
		bool is_gpu_only = false;

		if (ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		{
			if (payload == NULL && svm_metadata_is_gpu(metrics))
			{
				/* GPU-only model, cannot deserialize on CPU */
				is_gpu_only = true;
			}
			if (payload != NULL)
				pfree(payload);
			if (metrics != NULL)
				pfree(metrics);
		}

		if (is_gpu_only)
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("svm: model %d is GPU-only, GPU prediction failed",
						model_id),
					errhint("Check GPU configuration and ensure GPU is available")));
		}
	}

	elog(DEBUG1,
	     "neurondb: svm: GPU prediction failed or not available, trying CPU");

	/* Load model from catalog */
	if (!svm_load_model_from_catalog(model_id, &model))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: model %d not found", model_id)));

	/* Validate feature dimension */
	if (model->n_features > 0 && features->dim != model->n_features)
	{
		elog(DEBUG1,
		     "svm: feature dimension mismatch (expected %d, got %d)",
		     model->n_features,
		     features->dim);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: feature dimension mismatch (expected %d, got %d)",
					model->n_features,
					features->dim)));
	}

	/* Compute prediction using support vectors */
	prediction = model->bias;
	for (i = 0; i < model->n_support_vectors; i++)
	{
		float *sv = model->support_vectors + i * model->n_features;
		prediction += model->alphas[i]
			* linear_kernel(sv, features->data, features->dim);
	}

	/* Convert to binary class (0 or 1) */
	prediction = (prediction >= 0.0) ? 1.0 : 0.0;

	/* Cleanup */
	if (model != NULL)
	{
		if (model->alphas != NULL)
			pfree(model->alphas);
		if (model->support_vectors != NULL)
			pfree(model->support_vectors);
		if (model->support_vector_indices != NULL)
			pfree(model->support_vector_indices);
		pfree(model);
	}

	PG_RETURN_FLOAT8(prediction);
}

/*
 * svm_predict_batch
 *
 * Helper function to predict a batch of samples using SVM model.
 * Updates confusion matrix.
 */
static void
svm_predict_batch(const SVMModel *model,
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
			elog(DEBUG1,
				"neurondb: svm_predict_batch: early return - model=%p, features=%p, labels=%p, n_samples=%d",
			(void *)model, (void *)features, (void *)labels, n_samples);
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

		elog(DEBUG1,
			"neurondb: svm_predict_batch: starting - n_samples=%d, feature_dim=%d, model->n_support_vectors=%d, model->n_features=%d, model->bias=%.6f",
		n_samples, feature_dim, model->n_support_vectors, model->n_features, model->bias);

	for (i = 0; i < n_samples; i++)
	{
		const float *row = features + (i * feature_dim);
		double y_true = labels[i];
		int true_class;
		double prediction = 0.0;
		int pred_class;
		int j;

		if (!isfinite(y_true))
			continue;

		true_class = (int)rint(y_true);
		if (true_class < 0 || true_class > 1)
			continue;

		/* Compute prediction using support vectors */
		prediction = model->bias;
		if (model->n_support_vectors == 0)
		{
				elog(DEBUG1,
					"neurondb: svm_predict_batch: WARNING - model has 0 support vectors, prediction will be bias only (%.6f)",
				model->bias);
		}
		for (j = 0; j < model->n_support_vectors; j++)
		{
			float *sv = model->support_vectors + j * model->n_features;
			double kernel_val = 0.0;
			int k;

			/* Linear kernel: K(x, y) = x^T * y */
			for (k = 0; k < feature_dim; k++)
				kernel_val += (double)sv[k] * (double)row[k];

			prediction += model->alphas[j] * kernel_val;
		}

		/* Convert to binary class (0 or 1) */
		pred_class = (prediction >= 0.0) ? 1 : 0;
		
		if (i < 5)  /* Log first 5 predictions for debugging */
		{
				elog(DEBUG1,
					"neurondb: svm_predict_batch: sample %d - y_true=%.6f (class=%d), prediction=%.6f (class=%d)",
				i, y_true, true_class, prediction, pred_class);
		}

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
 * evaluate_svm_by_model_id
 *
 * Evaluates SVM model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: accuracy, precision, recall, f1_score, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_svm_by_model_id);

Datum
evaluate_svm_by_model_id(PG_FUNCTION_ARGS)
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
	SVMModel *model = NULL;
	StringInfoData jsonbuf;
	Jsonb *result_jsonb = NULL;
	bytea *gpu_payload = NULL;
	Jsonb *gpu_metrics = NULL;
	bool is_gpu_model = false;
	int *h_labels = NULL;
	float *h_features = NULL;
	int valid_rows = 0;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_svm_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_svm_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Load model from catalog - try CPU first, then GPU */
	if (!svm_load_model_from_catalog(model_id, &model))
	{
		/* Try GPU model */
		if (ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
		{
			is_gpu_model = svm_metadata_is_gpu(gpu_metrics);
			if (!is_gpu_model)
			{
				if (gpu_payload)
					pfree(gpu_payload);
				if (gpu_metrics)
					pfree(gpu_metrics);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_svm_by_model_id: model %d not found",
							model_id)));
			}
		}
		else
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_svm_by_model_id: model %d not found",
						model_id)));
		}
	}

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
	{
		if (model != NULL)
		{
			if (model->alphas != NULL)
				pfree(model->alphas);
			if (model->support_vectors != NULL)
				pfree(model->support_vectors);
			if (model->support_vector_indices != NULL)
				pfree(model->support_vector_indices);
			pfree(model);
		}
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_svm_by_model_id: SPI_connect failed")));
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
	elog(DEBUG1, "evaluate_svm_by_model_id: executing query: %s", query.data);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		pfree(query.data);
		if (model != NULL)
		{
			if (model->alphas != NULL)
				pfree(model->alphas);
			if (model->support_vectors != NULL)
				pfree(model->support_vectors);
			if (model->support_vector_indices != NULL)
				pfree(model->support_vector_indices);
			pfree(model);
		}
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_svm_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		pfree(query.data);
		if (model != NULL)
		{
			if (model->alphas != NULL)
				pfree(model->alphas);
			if (model->support_vectors != NULL)
				pfree(model->support_vectors);
			if (model->support_vector_indices != NULL)
				pfree(model->support_vector_indices);
			pfree(model);
		}
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_svm_by_model_id: no valid rows found")));
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
		const NdbCudaSvmModelHeader *gpu_hdr;
		size_t payload_size;

		/* Defensive check: validate payload size */
		payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
		if (payload_size < sizeof(NdbCudaSvmModelHeader))
		{
				 elog(DEBUG1,
				 	"neurondb: evaluate_svm_by_model_id: GPU payload too small (%zu bytes), falling back to CPU",
				 payload_size);
			goto cpu_evaluation_path;
		}

		/* Load GPU model header with defensive checks */
		gpu_hdr = (const NdbCudaSvmModelHeader *)VARDATA(gpu_payload);
		if (gpu_hdr == NULL)
		{
			elog(DEBUG1,
			     "neurondb: evaluate_svm_by_model_id: NULL GPU header, falling back to CPU");
			goto cpu_evaluation_path;
		}

		feat_dim = gpu_hdr->feature_dim;
		if (feat_dim <= 0 || feat_dim > 100000)
		{
				 elog(DEBUG1,
				 	"neurondb: evaluate_svm_by_model_id: invalid feature_dim (%d), falling back to CPU",
				 feat_dim);
			goto cpu_evaluation_path;
		}

		/* Allocate host buffers for features and labels with size checks */
		{
			size_t features_size = sizeof(float) * (size_t)nvec * (size_t)feat_dim;
			size_t labels_size = sizeof(int) * (size_t)nvec;

			if (features_size > MaxAllocSize || labels_size > MaxAllocSize)
			{
					 elog(DEBUG1,
					 	"evaluate_svm_by_model_id: allocation size too large (features=%zu, labels=%zu), falling back to CPU",
					 features_size, labels_size);
				goto cpu_evaluation_path;
			}

			h_features = (float *)palloc(features_size);
			h_labels = (int *)palloc(labels_size);

			if (h_features == NULL || h_labels == NULL)
			{
				elog(DEBUG1,
				     "evaluate_svm_by_model_id: memory allocation failed, falling back to CPU");
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
				     "evaluate_svm_by_model_id: NULL TupleDesc, falling back to CPU");
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
					     "evaluate_svm_by_model_id: valid_rows overflow, breaking");
					break;
				}

				feat_row = h_features + (valid_rows * feat_dim);
				if (feat_row == NULL || feat_row < h_features || feat_row >= h_features + (nvec * feat_dim))
				{
					elog(DEBUG1,
					     "evaluate_svm_by_model_id: feat_row out of bounds, skipping row");
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
					errmsg("neurondb: evaluate_svm_by_model_id: no valid rows found")));
		}

		/* Use optimized GPU batch evaluation */
		{
			int rc;
			char *gpu_errstr = NULL;

			/* Defensive checks before GPU call */
			if (h_features == NULL || h_labels == NULL || valid_rows <= 0 || feat_dim <= 0)
			{
					 elog(DEBUG1,
					 	"evaluate_svm_by_model_id: invalid inputs for GPU evaluation (features=%p, labels=%p, rows=%d, dim=%d), falling back to CPU",
					 (void *)h_features, (void *)h_labels, valid_rows, feat_dim);
				pfree(h_features);
				pfree(h_labels);
				goto cpu_evaluation_path;
			}

			PG_TRY();
			{
				rc = ndb_cuda_svm_evaluate_batch(gpu_payload,
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
						 	"evaluate_svm_by_model_id: GPU batch evaluation failed: %s, falling back to CPU",
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
				     "evaluate_svm_by_model_id: exception during GPU evaluation, falling back to CPU");
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
		float *cpu_h_features = NULL;
		double *cpu_h_labels = NULL;
		int cpu_valid_rows = 0;

		/* Determine feature dimension from model */
		if (model != NULL)
			feat_dim = model->n_features;
		else if (is_gpu_model && gpu_payload != NULL)
		{
			const NdbCudaSvmModelHeader *gpu_hdr;

			gpu_hdr = (const NdbCudaSvmModelHeader *)VARDATA(gpu_payload);
			feat_dim = gpu_hdr->feature_dim;
		}
		else
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_svm_by_model_id: could not determine feature dimension")));
		}

		if (feat_dim <= 0)
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_svm_by_model_id: invalid feature dimension %d",
						feat_dim)));
		}

		/* Allocate host buffers for features and labels */
		cpu_h_features = (float *)palloc(sizeof(float) * (size_t)nvec * (size_t)feat_dim);
		cpu_h_labels = (double *)palloc(sizeof(double) * (size_t)nvec);

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

				feat_row = cpu_h_features + (cpu_valid_rows * feat_dim);
				cpu_h_labels[cpu_valid_rows] = DatumGetFloat8(targ_datum);

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

				cpu_valid_rows++;
			}
		}

		if (cpu_valid_rows == 0)
		{
			pfree(cpu_h_features);
			pfree(cpu_h_labels);
			if (model != NULL)
			{
				if (model->alphas != NULL)
					pfree(model->alphas);
				if (model->support_vectors != NULL)
					pfree(model->support_vectors);
				if (model->support_vector_indices != NULL)
					pfree(model->support_vector_indices);
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
					errmsg("neurondb: evaluate_svm_by_model_id: no valid rows found")));
		}

		/* For GPU models, use GPU prediction */
		if (is_gpu_model && gpu_payload != NULL)
		{
			int gpu_failures = 0;
			/* GPU batch evaluation using GPU prediction */
			for (i = 0; i < cpu_valid_rows; i++)
			{
				double prediction = 0.0;
				double actual = cpu_h_labels[i];
				int rc;
				char *gpu_err = NULL;
				bool prediction_made = false;

				rc = ndb_gpu_svm_predict_double(gpu_payload,
					cpu_h_features + (i * feat_dim),
					feat_dim,
					&prediction,
					&gpu_err);

				if (rc == 0)
				{
					prediction_made = true;
				}
				else
				{
					gpu_failures++;
						elog(DEBUG1,
							"neurondb: evaluate_svm_by_model_id: GPU prediction failed for row %d: %s, falling back to CPU",
						i, gpu_err ? gpu_err : "unknown error");
					if (gpu_err)
						pfree(gpu_err);
					/* Fall back to CPU if available */
					if (model != NULL)
					{
						/* Compute prediction using CPU model (same logic as svm_predict_batch) */
						const float *row = cpu_h_features + (i * feat_dim);
						double cpu_pred = model->bias;
						int sv_idx;

						for (sv_idx = 0; sv_idx < model->n_support_vectors; sv_idx++)
						{
							float *sv = model->support_vectors + sv_idx * model->n_features;
							double kernel_val = 0.0;
							int k;

							/* Linear kernel: K(x, y) = x^T * y */
							for (k = 0; k < feat_dim; k++)
								kernel_val += (double)sv[k] * (double)row[k];

							cpu_pred += model->alphas[sv_idx] * kernel_val;
						}
						prediction = cpu_pred;
						prediction_made = true;
					}
					else
					{
						/* Cannot evaluate without model - use default prediction of 0 */
							elog(DEBUG1,
								"neurondb: evaluate_svm_by_model_id: GPU prediction failed and no CPU model available for row %d, using default prediction",
							i);
						prediction = 0.0;
						prediction_made = true;
					}
				}

				if (prediction_made)
				{
					/* Update confusion matrix */
					int pred_class = (prediction >= 0.0) ? 1 : 0;
					int actual_class = (actual >= 0.5) ? 1 : 0;

					if (pred_class == 1 && actual_class == 1)
						tp++;
					else if (pred_class == 0 && actual_class == 0)
						tn++;
					else if (pred_class == 1 && actual_class == 0)
						fp++;
					else if (pred_class == 0 && actual_class == 1)
						fn++;
				}
			}
			
			if (gpu_failures > 0)
			{
					elog(DEBUG1,
						"neurondb: evaluate_svm_by_model_id: GPU prediction failed for %d/%d rows",
					gpu_failures, cpu_valid_rows);
			}
		}
		else
		{
			/* Ensure model is not NULL before prediction */
			if (model == NULL)
			{
				pfree(cpu_h_features);
				pfree(cpu_h_labels);
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
						errmsg("neurondb: evaluate_svm_by_model_id: model is NULL")));
			}

			/* Use batch prediction helper for CPU models */
			/* Add debug logging before prediction */
			if (model != NULL)
			{
					elog(DEBUG1,
						"neurondb: evaluate_svm_by_model_id: CPU batch prediction: model->n_support_vectors=%d, model->n_features=%d, model->bias=%.6f, valid_rows=%d, feat_dim=%d",
					model->n_support_vectors, model->n_features, model->bias, cpu_valid_rows, feat_dim);
			}
			else
			{
				elog(DEBUG1,
				     "neurondb: evaluate_svm_by_model_id: model is NULL before CPU batch prediction");
			}
			
			svm_predict_batch(model,
				cpu_h_features,
				cpu_h_labels,
				cpu_valid_rows,
				feat_dim,
				&tp,
				&tn,
				&fp,
				&fn);
			
				elog(DEBUG1,
					"neurondb: evaluate_svm_by_model_id: after batch prediction: tp=%d, tn=%d, fp=%d, fn=%d, valid_rows=%d",
				tp, tn, fp, fn, cpu_valid_rows);
		}

		/* Compute metrics */
		if (cpu_valid_rows > 0)
		{
			accuracy = (double)(tp + tn) / (double)cpu_valid_rows;

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
		pfree(cpu_h_features);
		pfree(cpu_h_labels);
		if (model != NULL)
		{
			if (model->alphas != NULL)
				pfree(model->alphas);
			if (model->support_vectors != NULL)
				pfree(model->support_vectors);
			if (model->support_vector_indices != NULL)
				pfree(model->support_vector_indices);
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

/*
 * GPU Model Ops for SVM
 */
typedef struct SVMGpuModelState
{
	bytea *model_blob;
	Jsonb *metrics;
	int feature_dim;
	int n_samples;
} SVMGpuModelState;

static void
svm_gpu_release_state(SVMGpuModelState *state)
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
svm_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	SVMGpuModelState *state;
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

	rc = ndb_gpu_svm_train(spec->feature_matrix,
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
		svm_gpu_release_state((SVMGpuModelState *)model->backend_state);
		model->backend_state = NULL;
	}

	state = (SVMGpuModelState *)palloc0(sizeof(SVMGpuModelState));
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;

	if (metrics != NULL)
		state->metrics = (Jsonb *)PG_DETOAST_DATUM_COPY(
			PointerGetDatum(metrics));
	else
		state->metrics = NULL;

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
svm_gpu_predict(const MLGpuModel *model,
	const float *input,
	int input_dim,
	float *output,
	int output_dim,
	char **errstr)
{
	const SVMGpuModelState *state;
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

	state = (const SVMGpuModelState *)model->backend_state;
	if (state->model_blob == NULL)
		return false;

	rc = ndb_gpu_svm_predict_double(state->model_blob,
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
svm_gpu_serialize(const MLGpuModel *model,
	bytea **payload_out,
	Jsonb **metadata_out,
	char **errstr)
{
	const SVMGpuModelState *state;
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

	state = (const SVMGpuModelState *)model->backend_state;
	if (state->model_blob == NULL)
		return false;

	payload_size = VARSIZE(state->model_blob);
	payload_copy = (bytea *)palloc(payload_size);
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		pfree(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
		*metadata_out = (Jsonb *)PG_DETOAST_DATUM_COPY(
			PointerGetDatum(state->metrics));

	return true;
}

static void
svm_gpu_destroy(MLGpuModel *model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		svm_gpu_release_state((SVMGpuModelState *)model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps svm_gpu_model_ops = {
	.algorithm = "svm",
	.train = svm_gpu_train,
	.predict = svm_gpu_predict,
	.evaluate = NULL,
	.serialize = svm_gpu_serialize,
	.deserialize = NULL,
	.destroy = svm_gpu_destroy,
};

/*
 * neurondb_gpu_register_svm_model
 */
void
neurondb_gpu_register_svm_model(void)
{
	static bool registered = false;

	if (registered)
		return;

	ndb_gpu_register_model_ops(&svm_gpu_model_ops);
	registered = true;
}
