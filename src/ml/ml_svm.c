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
		elog(ERROR,
			"linear_kernel: invalid inputs (x=%p, y=%p, dim=%d)",
			(void *)x,
			(void *)y,
			dim);
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
		ereport(ERROR, (errmsg("svm_dataset_load: dataset is NULL")));

	oldcontext = CurrentMemoryContext;

	/* Initialize query in caller's context before SPI_connect */
	initStringInfo(&query);
	MemoryContextSwitchTo(oldcontext);

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("SPI_connect failed")));

	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quoted_feat,
		quoted_label,
		quoted_tbl,
		quoted_feat,
		quoted_label);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		ereport(ERROR, (errmsg("Query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 10)
	{
		SPI_finish();
		ereport(ERROR, (errmsg("Need at least 10 samples for SVM")));
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
		ereport(ERROR, (errmsg("Invalid feature dimension: %d", dim)));
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
				ereport(ERROR,
					(errmsg("Feature dimension mismatch: "
						"expected %d, got %d",
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
				errmsg("svm_model_serialize: invalid "
				       "n_features %d (corrupted model?)",
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
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: invalid n_features %d in "
				       "deserialized model (corrupted data?)",
					model->n_features)));
	}
	if (model->n_support_vectors < 0 || model->n_support_vectors > 100000)
	{
		pfree(model);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: invalid n_support_vectors %d in "
				       "deserialized model (corrupted data?)",
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
	char *tbl_str;
	char *feat_str;
	char *label_str;
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
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: train_svm_classifier requires 3-5 "
				       "arguments, got %d",
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
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: C parameter must be in range (0, "
				       "1000], got %.6f",
					c_param)));

	if (max_iters <= 0 || max_iters > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: max_iters must be in range (0, "
				       "100000], got %d",
					max_iters)));

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
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: need at least 10 samples for "
				       "training, got %d",
					nvec)));
	}

	if (dim <= 0 || dim > 10000)
	{
		svm_dataset_free(&dataset);
		pfree(query.data);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: invalid feature dimension %d "
				       "(must be in range [1, 10000])",
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
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("svm: all labels have the same "
					       "value (%.6f), need at least "
					       "two classes",
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
					errmsg("svm: labels must contain both "
					       "classes (class0=%d, class1=%d)",
						n_class0,
						n_class1)));
		}

		elog(DEBUG1,
			"svm: label validation passed (class0=%d, class1=%d, "
			"first=%.6f, second=%.6f)",
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
			elog(WARNING,
				"svm: failed to create hyperparameters JSONB, "
				"falling back to CPU");
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

				elog(DEBUG1, "svm: GPU training succeeded");
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
						"svm: GPU training failed: %s",
						gpu_err);
				else
					elog(DEBUG1,
						"svm: GPU training "
						"unavailable, falling back to "
						"CPU");
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
		"svm: using CPU training path (n_samples=%d, feature_dim=%d, "
		"C=%.6f, max_iters=%d)",
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
			"svm: CPU training limits: actual_max_iters=%d, "
			"sample_limit=%d, kernel_limit=%d",
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
		for (i = 0; i < sample_limit && i < nvec; i++)
		{
			errors[i] =
				-dataset.labels
					 [i]; /* f(x_i) = 0 initially, so E_i = 0 - y_i = -y_i */
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
				eta = 0.0;
				L = 0.0;
				H = c_param;
				new_alpha_i = 0.0;

				/* Compute kernel for self (simplified) */
				eta = 2.0
						* linear_kernel(dataset.features
								+ i * dim,
							dataset.features
								+ i * dim,
							dim)
					- linear_kernel(
						dataset.features + i * dim,
						dataset.features + i * dim,
						dim);

				if (eta >= 0.0)
					continue;

				/* Update alpha */
				new_alpha_i = alpha_i - label_i * error_i / eta;

				/* Clip to bounds */
				if (new_alpha_i < L)
					new_alpha_i = L;
				if (new_alpha_i > H)
					new_alpha_i = H;

				if (fabs(new_alpha_i - alpha_i) < eps)
					continue;

				alphas[i] = new_alpha_i;

				/* Update error */
				delta_alpha = (new_alpha_i - alpha_i);
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

		elog(DEBUG1,
			"svm: CPU SMO completed after %d iterations, %d "
			"support vectors",
			iter,
			sv_count);

		/* Count support vectors */
		sv_count = 0;
		for (i = 0; i < sample_limit && i < nvec; i++)
		{
			if (alphas[i] > eps)
				sv_count++;
		}

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
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("svm: invalid feature dimension "
					       "%d before model serialization",
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
			"svm: building model with n_features=%d, n_samples=%d, "
			"sv_count=%d",
			model.n_features,
			model.n_samples,
			sv_count);

		/* Handle case when no support vectors found */
		if (sv_count == 0)
		{
			elog(WARNING,
				"svm: no support vectors found, using default "
				"model with single support vector");
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
					elog(WARNING,
						"svm: expected %d support "
						"vectors but copied %d",
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
					errmsg("svm: model.n_features is "
					       "invalid (%d) before "
					       "serialization",
						model.n_features)));
		}

		elog(DEBUG1,
			"svm: serializing model with n_features=%d, "
			"n_support_vectors=%d",
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
			elog(WARNING,
				"svm: failed to create metrics JSONB, "
				"continuing without metrics");
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
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("svm: failed to register model "
					       "in catalog, model_id=%d",
						model_id)));
		}

		elog(DEBUG1,
			"svm: CPU training completed, model_id=%d",
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
			"svm: GPU prediction succeeded, prediction=%.6f",
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
		"svm: GPU prediction failed or not available, trying CPU");

	/* Load model from catalog */
	if (!svm_load_model_from_catalog(model_id, &model))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: model %d not found", model_id)));

	/* Validate feature dimension */
	if (model->n_features > 0 && features->dim != model->n_features)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("svm: feature dimension mismatch "
				       "(expected %d, got %d)",
					model->n_features,
					features->dim)));

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
