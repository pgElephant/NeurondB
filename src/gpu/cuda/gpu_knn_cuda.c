/*-------------------------------------------------------------------------
 *
 * gpu_knn_cuda.c
 *    CUDA backend bridge for K-Nearest Neighbors training and prediction.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_knn_cuda.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#ifdef NDB_GPU_CUDA

#include <float.h>
#include <math.h>
#include <string.h>

#include "neurondb_cuda_runtime.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/palloc.h"
#include "utils/memutils.h"
#include "utils/elog.h"

#include "neurondb_cuda_knn.h"

/* KNN model structure */
typedef struct KNNModel
{
	int		n_samples;
	int		n_features;
	int		k;
	int		task_type;  /* 0=classification, 1=regression */
	float	*features;  /* Training features [n_samples * n_features] */
	double	*labels;    /* Training labels [n_samples] */
} KNNModel;

int
ndb_cuda_knn_pack(const struct KNNModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	size_t payload_bytes;
	size_t features_bytes;
	size_t labels_bytes;
	bytea *blob;
	char *base;
	NdbCudaKnnModelHeader *hdr;
	float *features_dest;
	double *labels_dest;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model for CUDA pack: model or model_data is NULL");
		return -1;
	}

	/* Validate model structure */
	if (model->n_samples <= 0 || model->n_samples > 100000000)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: n_samples must be between 1 and 100000000");
		return -1;
	}
	if (model->n_features <= 0 || model->n_features > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: n_features must be between 1 and 1000000");
		return -1;
	}
	if (model->k <= 0 || model->k > model->n_samples)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: k must be between 1 and n_samples");
		return -1;
	}
	if (model->task_type != 0 && model->task_type != 1)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: task_type must be 0 (classification) or 1 (regression)");
		return -1;
	}

	/* Check for integer overflow in size calculations */
	features_bytes = sizeof(float) * (size_t)model->n_samples * (size_t)model->n_features;
	labels_bytes = sizeof(double) * (size_t)model->n_samples;

	if (features_bytes / sizeof(float) / (size_t)model->n_samples != (size_t)model->n_features)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: integer overflow in features size calculation");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaKnnModelHeader) + features_bytes + labels_bytes;

	/* Check for overflow in total payload */
	if (payload_bytes < sizeof(NdbCudaKnnModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: payload size underflow");
		return -1;
	}
	if (payload_bytes > (MaxAllocSize - VARHDRSZ))
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: payload size exceeds MaxAllocSize");
		return -1;
	}

	blob = (bytea *)palloc(VARHDRSZ + payload_bytes);
	if (blob == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN pack: memory allocation failed");
		return -1;
	}

	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaKnnModelHeader *)base;
	hdr->n_samples = model->n_samples;
	hdr->n_features = model->n_features;
	hdr->k = model->k;
	hdr->task_type = model->task_type;

	features_dest = (float *)(base + sizeof(NdbCudaKnnModelHeader));
	labels_dest = (double *)(base + sizeof(NdbCudaKnnModelHeader) + sizeof(float) * (size_t)model->n_samples * (size_t)model->n_features);

	if (model->features != NULL)
	{
		/* Validate features array for NaN/Inf */
		for (int i = 0; i < model->n_samples * model->n_features; i++)
		{
			if (!isfinite(model->features[i]))
			{
				if (errstr)
					*errstr = pstrdup("invalid KNN model: features array contains non-finite value");
				pfree(blob);
				return -1;
			}
		}
		memcpy(features_dest, model->features, features_bytes);
	}
	else
	{
		/* Initialize with zeros if NULL */
		memset(features_dest, 0, features_bytes);
	}

	if (model->labels != NULL)
	{
		/* Validate labels array for NaN/Inf and task-specific constraints */
		for (int i = 0; i < model->n_samples; i++)
		{
			if (!isfinite(model->labels[i]))
			{
				if (errstr)
					*errstr = pstrdup("invalid KNN model: labels array contains non-finite value");
				pfree(blob);
				return -1;
			}
			/* For classification, labels should be integers */
			if (model->task_type == 0)
			{
				double label = model->labels[i];
				if (label != floor(label) || label < 0.0)
				{
					if (errstr)
						*errstr = pstrdup("invalid KNN model: classification labels must be non-negative integers");
					pfree(blob);
					return -1;
				}
			}
		}
		memcpy(labels_dest, model->labels, labels_bytes);
	}
	else
	{
		/* Initialize with zeros if NULL */
		memset(labels_dest, 0, labels_bytes);
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"knn\","
			"\"storage\":\"gpu\","
			"\"n_samples\":%d,"
			"\"n_features\":%d,"
			"\"k\":%d,"
			"\"task_type\":%d}",
			model->n_samples,
			model->n_features,
			model->k,
			model->task_type);

		metrics_json = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(buf.data)));
		pfree(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

int
ndb_cuda_knn_train(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	int k,
	int task_type,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr)
{
	struct KNNModel model;
	bytea *blob = NULL;
	Jsonb *metrics_json = NULL;
	float *features_copy = NULL;
	double *labels_copy = NULL;
	int i, j;
	int extracted_task;
	int rc = -1;

	if (errstr)
		*errstr = NULL;

	/* Comprehensive input validation */
	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: features array is NULL");
		return -1;
	}
	if (labels == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: labels array is NULL");
		return -1;
	}
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: model_data output pointer is NULL");
		return -1;
	}
	if (n_samples <= 0 || n_samples > 100000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: n_samples must be between 1 and 100000000");
		return -1;
	}
	if (feature_dim <= 0 || feature_dim > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: feature_dim must be between 1 and 1000000");
		return -1;
	}
	if (k <= 0 || k > n_samples)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: k must be between 1 and n_samples");
		return -1;
	}
	if (task_type != 0 && task_type != 1)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: task_type must be 0 (classification) or 1 (regression)");
		return -1;
	}

	/* Check for integer overflow in memory allocation */
	if (feature_dim > 0 && (size_t)n_samples > MaxAllocSize / sizeof(float) / (size_t)feature_dim)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: feature array size exceeds MaxAllocSize");
		return -1;
	}

	elog(DEBUG1, "ndb_cuda_knn_train: starting training: n_samples=%d, feature_dim=%d, k=%d, task_type=%d",
		n_samples, feature_dim, k, task_type);

	/* Validate input data for NaN/Inf before processing */
	for (i = 0; i < n_samples; i++)
	{
		for (j = 0; j < feature_dim; j++)
		{
			if (!isfinite(features[i * feature_dim + j]))
			{
				if (errstr)
					*errstr = pstrdup("CUDA KNN train: non-finite value in features array");
				return -1;
			}
		}
		if (!isfinite(labels[i]))
		{
			if (errstr)
				*errstr = pstrdup("CUDA KNN train: non-finite value in labels array");
			return -1;
		}
		/* For classification, validate labels are integers */
		if (task_type == 0)
		{
			double label = labels[i];
			if (label != floor(label) || label < 0.0)
			{
				if (errstr)
					*errstr = pstrdup("CUDA KNN train: classification labels must be non-negative integers");
				return -1;
			}
		}
	}

	/* Extract hyperparameters if provided */
	if (hyperparams != NULL)
	{
		Datum k_datum;
		Datum task_type_datum;
		Datum numeric_datum;
		Numeric num;

		k_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("k"));
		if (DatumGetPointer(k_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, k_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				k = DatumGetInt32(
					DirectFunctionCall1(numeric_int4,
						NumericGetDatum(num)));
				if (k <= 0 || k > n_samples)
					k = (n_samples < 10) ? n_samples : 10;
			}
		}

		task_type_datum = DirectFunctionCall2(
			jsonb_object_field,
			JsonbPGetDatum(hyperparams),
			CStringGetTextDatum("task_type"));
		if (DatumGetPointer(task_type_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
				jsonb_numeric, task_type_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				extracted_task = DatumGetInt32(
					DirectFunctionCall1(numeric_int4,
						NumericGetDatum(num)));
				if (extracted_task == 0 || extracted_task == 1)
					task_type = extracted_task;
			}
		}
	}

	/* Copy training data (KNN is a lazy learner - just stores data) */
	if (feature_dim > 0 && (size_t)n_samples > MaxAllocSize / sizeof(float) / (size_t)feature_dim)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: features_copy array size exceeds MaxAllocSize");
		return -1;
	}
	features_copy = (float *)palloc(sizeof(float) * (size_t)n_samples * (size_t)feature_dim);
	if (features_copy == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: failed to allocate features_copy array");
		return -1;
	}

	labels_copy = (double *)palloc(sizeof(double) * (size_t)n_samples);
	if (labels_copy == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: failed to allocate labels_copy array");
		pfree(features_copy);
		return -1;
	}

	memcpy(features_copy, features, sizeof(float) * (size_t)n_samples * (size_t)feature_dim);
	memcpy(labels_copy, labels, sizeof(double) * (size_t)n_samples);

	/* Build model structure */
	model.n_samples = n_samples;
	model.n_features = feature_dim;
	model.k = k;
	model.task_type = task_type;
	model.features = features_copy;
	model.labels = labels_copy;

	/* Pack model */
	if (ndb_cuda_knn_pack(&model, &blob, &metrics_json, errstr) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("CUDA KNN model packing failed");
		goto cleanup;
	}

	*model_data = blob;
	if (metrics != NULL)
		*metrics = metrics_json;

	rc = 0;

cleanup:
	/* Note: features_copy and labels_copy are now owned by the packed model */
	/* They will be freed when the model is freed */

	return rc;
}

int
ndb_cuda_knn_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	double *prediction_out,
	char **errstr)
{
	const char *base;
	NdbCudaKnnModelHeader *hdr;
	const float *training_features;
	const double *training_labels;
	float *distances = NULL;
	int i;
	size_t expected_size;

	if (errstr)
		*errstr = NULL;

	/* Comprehensive input validation */
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: model_data is NULL");
		return -1;
	}
	if (input == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: input array is NULL");
		return -1;
	}
	if (prediction_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: prediction_out pointer is NULL");
		return -1;
	}

	/* Validate model_data bytea */
	if (VARSIZE_ANY_EXHDR(model_data) < sizeof(NdbCudaKnnModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: model_data too small (corrupted or invalid)");
		return -1;
	}

	base = VARDATA_ANY(model_data);
	hdr = (NdbCudaKnnModelHeader *)base;

	/* Validate model header */
	if (hdr->n_samples <= 0 || hdr->n_samples > 100000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: invalid n_samples in model header");
		return -1;
	}
	if (hdr->n_features <= 0 || hdr->n_features > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: invalid n_features in model header");
		return -1;
	}
	if (hdr->k <= 0 || hdr->k > hdr->n_samples)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: invalid k in model header");
		return -1;
	}
	if (hdr->task_type != 0 && hdr->task_type != 1)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: invalid task_type in model header");
		return -1;
	}
	if (feature_dim != hdr->n_features)
	{
		if (errstr)
		{
			char *msg = psprintf("CUDA KNN predict: feature dimension mismatch (expected %d, got %d)", hdr->n_features, feature_dim);
			*errstr = pstrdup(msg);
			pfree(msg);
		}
		return -1;
	}

	/* Validate bytea size matches expected payload */
	expected_size = sizeof(NdbCudaKnnModelHeader)
		+ sizeof(float) * (size_t)hdr->n_samples * (size_t)hdr->n_features
		+ sizeof(double) * (size_t)hdr->n_samples;
	if (VARSIZE_ANY_EXHDR(model_data) < expected_size)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: model_data size mismatch (corrupted model)");
		return -1;
	}

	/* Validate input array */
	for (i = 0; i < feature_dim; i++)
	{
		if (!isfinite(input[i]))
		{
			if (errstr)
				*errstr = pstrdup("CUDA KNN predict: non-finite value in input array");
			return -1;
		}
	}

	training_features = (const float *)(base + sizeof(NdbCudaKnnModelHeader));
	training_labels = (const double *)(base + sizeof(NdbCudaKnnModelHeader) + sizeof(float) * (size_t)hdr->n_samples * (size_t)hdr->n_features);

	/* Validate model data pointers */
	if (training_features == NULL || training_labels == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: model data pointers are NULL (corrupted model)");
		return -1;
	}

	/* Allocate distance array */
	if (hdr->n_samples > MaxAllocSize / sizeof(float))
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: distances array size exceeds MaxAllocSize");
		return -1;
	}
	distances = (float *)palloc(sizeof(float) * hdr->n_samples);
	if (distances == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: failed to allocate distances array");
		return -1;
	}

	/* Step 1: Compute distances using CUDA */
	if (ndb_cuda_knn_compute_distances(input, training_features, hdr->n_samples, hdr->n_features, distances) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("CUDA distance computation failed");
		pfree(distances);
		return -1;
	}

	/* Validate computed distances */
	for (i = 0; i < hdr->n_samples; i++)
	{
		if (!isfinite(distances[i]) || distances[i] < 0.0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA KNN predict: computed invalid distance");
			pfree(distances);
			return -1;
		}
	}

	/* Step 2: Find top-k and compute prediction using CUDA */
	if (ndb_cuda_knn_find_top_k(distances, training_labels, hdr->n_samples, hdr->k, hdr->task_type, prediction_out) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("CUDA top-k computation failed");
		pfree(distances);
		return -1;
	}

	/* Validate prediction output */
	if (!isfinite(*prediction_out))
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: computed non-finite prediction");
		pfree(distances);
		return -1;
	}
	/* For classification, prediction should be an integer */
	if (hdr->task_type == 0)
	{
		if (*prediction_out != floor(*prediction_out) || *prediction_out < 0.0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA KNN predict: classification prediction must be non-negative integer");
			pfree(distances);
			return -1;
		}
	}

	pfree(distances);
	return 0;
}

#else

void
ndb_cuda_knn_train(void) { }
void
ndb_cuda_knn_predict(void) { }
void
ndb_cuda_knn_pack(void) { }

#endif /* NDB_GPU_CUDA */

