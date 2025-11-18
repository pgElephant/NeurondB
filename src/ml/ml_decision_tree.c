/*-------------------------------------------------------------------------
 *
 * ml_decision_tree.c
 *    Decision Tree implementation for classification and regression
 *
 * Implements CART (Classification and Regression Trees) using
 * Gini impurity for classification and variance reduction for regression.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_decision_tree.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/jsonb.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_decision_tree_internal.h"
#include "ml_catalog.h"
#include "neurondb_gpu_bridge.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "neurondb_gpu_backend.h"
#include "ml_gpu_registry.h"
#include "ml_gpu_decision_tree.h"
#include "neurondb_cuda_dt.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif

#include <math.h>
#include <float.h>
#include <limits.h>

/* Use DTNode from internal header */
#define TreeNode DTNode

/*
 * DTDataset
 * Internal structure to hold dataset for training.
 */
typedef struct DTDataset
{
	float	*features;
	double	*labels;
	int		n_samples;
	int		feature_dim;
} DTDataset;

/* Forward declarations */
static void dt_dataset_init(DTDataset *dataset);
static void dt_dataset_free(DTDataset *dataset);
static void dt_dataset_load(const char *quoted_tbl, const char *quoted_feat, const char *quoted_label, DTDataset *dataset);
static bytea *dt_model_serialize(const DTModel *model);
static DTModel *dt_model_deserialize(const bytea *data);
static bool dt_metadata_is_gpu(Jsonb *metadata);
static bool dt_try_gpu_predict_catalog(int32 model_id, const Vector *feature_vec, double *result_out);
static bool dt_load_model_from_catalog(int32 model_id, DTModel **out);
static void dt_free_tree(DTNode *node);
static double dt_tree_predict(const DTNode *node, const float *x, int dim);

/*
 * dt_dataset_init
 *
 * Initialize a DTDataset struct to zeros.
 */
static void
dt_dataset_init(DTDataset *dataset)
{
	if (dataset == NULL)
		return;

	memset(dataset, 0, sizeof(DTDataset));
}

/*
 * dt_dataset_free
 *
 * Free memory allocated in the dataset.
 */
static void
dt_dataset_free(DTDataset *dataset)
{
	if (dataset == NULL)
		return;

	if (dataset->features)
		pfree(dataset->features);
	if (dataset->labels)
		pfree(dataset->labels);

	dt_dataset_init(dataset);
}

/*
 * dt_dataset_load
 *
 * Load feature and label data from a table into local memory.
 */
static void
dt_dataset_load(const char *quoted_tbl,
				const char *quoted_feat,
				const char *quoted_label,
				DTDataset *dataset)
{
	StringInfoData query;
	MemoryContext	oldcontext;
	int				ret;
	int				n_samples = 0;
	int				feature_dim = 0;
	int				i;

	if (!dataset)
		ereport(ERROR, (errmsg("dt_dataset_load: dataset is NULL")));

	oldcontext = CurrentMemoryContext;

	initStringInfo(&query);

	/*
	 * Save current memory context and reconnect after SPI.
	 * This follows Postgres memory management best practice.
	 */
	MemoryContextSwitchTo(oldcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("dt_dataset_load: SPI_connect failed")));

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
		ereport(ERROR, (errmsg("dt_dataset_load: query failed")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		SPI_finish();
		ereport(ERROR, (errmsg("dt_dataset_load: need at least 10 samples, got %d", n_samples)));
	}

	/* Detect feature dimension from first row. */
	if (n_samples > 0)
	{
		HeapTuple	first_tuple = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		bool		feat_null = false;
		Vector	   *vec;

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
		ereport(ERROR, (errmsg("dt_dataset_load: could not determine feature dimension")));
	}

	MemoryContextSwitchTo(oldcontext);

	dataset->features = (float *) palloc(sizeof(float) * ((Size) n_samples) * ((Size) feature_dim));
	dataset->labels = (double *) palloc(sizeof(double) * (Size) n_samples);

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		label_datum;
		bool		feat_null = false;
		bool		label_null = false;
		Vector	   *vec;
		float	   *row;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		if (feat_null)
			continue;

		vec = DatumGetVector(feat_datum);
		if (vec->dim != feature_dim)
		{
			SPI_finish();
			ereport(ERROR, (errmsg("dt_dataset_load: inconsistent vector dimensions")));
		}

		row = dataset->features + (i * feature_dim);
		memcpy(row, vec->data, sizeof(float) * feature_dim);

		label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);
		if (label_null)
			continue;

		{
			Oid label_type = SPI_gettypeid(tupdesc, 2);

			if (label_type == INT2OID)
				dataset->labels[i] = (double) DatumGetInt16(label_datum);
			else if (label_type == INT4OID)
				dataset->labels[i] = (double) DatumGetInt32(label_datum);
			else if (label_type == INT8OID)
				dataset->labels[i] = (double) DatumGetInt64(label_datum);
			else
				dataset->labels[i] = DatumGetFloat8(label_datum);
		}
	}

	dataset->n_samples = n_samples;
	dataset->feature_dim = feature_dim;

	SPI_finish();
}

/*
 * compute_gini
 *
 * Compute the Gini impurity for a vector of binary labels.
 */
static double
compute_gini(const double *labels, int n)
{
	int		i, class0 = 0, class1 = 0;
	double	p0, p1;

	if (n == 0)
		return 0.0;

	for (i = 0; i < n; i++)
	{
		if ((int) labels[i] == 0)
			class0++;
		else
			class1++;
	}

	p0 = (double) class0 / (double) n;
	p1 = (double) class1 / (double) n;
	return 1.0 - (p0 * p0 + p1 * p1);
}

/*
 * compute_variance
 *
 * Compute variance (mean squared deviation from the mean).
 */
static double
compute_variance(const double *values, int n)
{
	int			i;
	double		mean = 0.0;
	double		var = 0.0;

	if (n == 0)
		return 0.0;

	for (i = 0; i < n; i++)
		mean += values[i];
	mean /= n;

	for (i = 0; i < n; i++)
		var += (values[i] - mean) * (values[i] - mean);
	var /= n;

	return var;
}

/*
 * find_best_split_1d
 *
 * Find the best split on any feature using brute-force greedy split.
 */
static void
find_best_split_1d(const float *features,
				   const double *labels,
				   const int *indices,
				   int n_samples,
				   int dim,
				   int *best_feature,
				   float *best_threshold,
				   double *best_gain,
				   bool is_classification)
{
	int		feat;

	*best_gain = -DBL_MAX;
	*best_feature = -1;
	*best_threshold = 0.0;

	for (feat = 0; feat < dim; feat++)
	{
		float	min_val = FLT_MAX, max_val = -FLT_MAX;
		int		ii;

		for (ii = 0; ii < n_samples; ii++)
		{
			float val = features[indices[ii] * dim + feat];

			if (val < min_val)
				min_val = val;
			if (val > max_val)
				max_val = val;
		}

		if (min_val == max_val)
			continue;

		for (ii = 1; ii < 10; ii++)
		{
			float		threshold = min_val + (max_val - min_val) * ii / 10.0f;
			int			left_count = 0, right_count = 0, j;
			double	   *left_y, *right_y;
			int			l_idx = 0, r_idx = 0;
			double		left_imp, right_imp, gain;

			for (j = 0; j < n_samples; j++)
			{
				if (features[indices[j] * dim + feat] <= threshold)
					left_count++;
				else
					right_count++;
			}

			if (left_count == 0 || right_count == 0)
				continue;

			left_y = (double *) palloc(sizeof(double) * left_count);
			right_y = (double *) palloc(sizeof(double) * right_count);

			l_idx = r_idx = 0;
			for (j = 0; j < n_samples; j++)
			{
				if (features[indices[j] * dim + feat] <= threshold)
					left_y[l_idx++] = labels[indices[j]];
				else
					right_y[r_idx++] = labels[indices[j]];
			}

			Assert(l_idx == left_count);
			Assert(r_idx == right_count);

			if (is_classification)
			{
				left_imp = compute_gini(left_y, left_count);
				right_imp = compute_gini(right_y, right_count);
			}
			else
			{
				left_imp = compute_variance(left_y, left_count);
				right_imp = compute_variance(right_y, right_count);
			}

			gain = -(((double) left_count / (double) n_samples) * left_imp +
					 ((double) right_count / (double) n_samples) * right_imp);

			if (gain > *best_gain)
			{
				*best_gain = gain;
				*best_feature = feat;
				*best_threshold = threshold;
			}

			pfree(left_y);
			pfree(right_y);
		}
	}
}

/*
 * build_tree_1d
 *
 * Recursively build a decision tree using 1D flat arrays.
 */
static DTNode *
build_tree_1d(const float *features,
			  const double *labels,
			  const int *indices,
			  int n_samples,
			  int dim,
			  int max_depth,
			  int min_samples_split,
			  bool is_classification)
{
	DTNode *node;
	int			i, best_feature;
	float		best_threshold;
	double		best_gain;
	int		   *left_indices, *right_indices;
	int			left_count = 0, right_count = 0;

	node = (DTNode *) palloc0(sizeof(DTNode));

	if (max_depth == 0 || n_samples < min_samples_split)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			int class0 = 0, class1 = 0;

			for (i = 0; i < n_samples; i++)
			{
				int l = (int) labels[indices[i]];
				if (l == 0)
					class0++;
				else
					class1++;
			}
			node->leaf_value = (class1 > class0) ? 1.0 : 0.0;
		}
		else
		{
			double sum = 0.0;
			for (i = 0; i < n_samples; i++)
				sum += labels[indices[i]];
			node->leaf_value = sum / n_samples;
		}
		return node;
	}

	find_best_split_1d(features, labels, indices, n_samples, dim,
					   &best_feature, &best_threshold, &best_gain, is_classification);

	if (best_feature == -1)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			int class0 = 0, class1 = 0;

			for (i = 0; i < n_samples; i++)
			{
				int l = (int) labels[indices[i]];
				if (l == 0)
					class0++;
				else
					class1++;
			}
			node->leaf_value = (class1 > class0) ? 1.0 : 0.0;
		}
		else
		{
			double sum = 0.0;
			for (i = 0; i < n_samples; i++)
				sum += labels[indices[i]];
			node->leaf_value = sum / n_samples;
		}
		return node;
	}

	node->is_leaf = false;
	node->feature_idx = best_feature;
	node->threshold = best_threshold;

	left_indices = (int *) palloc(sizeof(int) * n_samples);
	right_indices = (int *) palloc(sizeof(int) * n_samples);

	for (i = 0; i < n_samples; i++)
	{
		if (features[indices[i] * dim + best_feature] <= best_threshold)
			left_indices[left_count++] = indices[i];
		else
			right_indices[right_count++] = indices[i];
	}
	Assert(left_count + right_count == n_samples);

	node->left = build_tree_1d(features, labels, left_indices, left_count, dim,
							   max_depth - 1, min_samples_split, is_classification);
	node->right = build_tree_1d(features, labels, right_indices, right_count, dim,
								max_depth - 1, min_samples_split, is_classification);

	pfree(left_indices);
	pfree(right_indices);

	return node;
}

/*
 * dt_tree_predict
 *
 * Predict by walking the tree for an input vector.
 */
static double
dt_tree_predict(const DTNode *node, const float *x, int dim)
{
	if (node == NULL)
		elog(ERROR, "dt_tree_predict: NULL node");
	if (node->is_leaf)
		return node->leaf_value;
	if (node->feature_idx < 0 || node->feature_idx >= dim)
		elog(ERROR, "dt_tree_predict: invalid feature_idx %d (dim=%d)",
			 node->feature_idx, dim);

	if (x[node->feature_idx] <= node->threshold)
		return dt_tree_predict(node->left, x, dim);
	else
		return dt_tree_predict(node->right, x, dim);
}

/*
 * dt_free_tree
 *
 * Free the memory allocated for the tree recursively.
 */
static void
dt_free_tree(DTNode *node)
{
	if (node == NULL)
		return;

	if (!node->is_leaf)
	{
		dt_free_tree(node->left);
		dt_free_tree(node->right);
	}
	pfree(node);
}

/*
 * dt_serialize_node
 *
 * Recursively write a tree node into a StringInfo buffer.
 */
static void
dt_serialize_node(StringInfo buf, const DTNode *node)
{
	if (!node)
	{
		pq_sendint8(buf, 0);
		return;
	}

	pq_sendint8(buf, 1);
	pq_sendint8(buf, node->is_leaf ? 1 : 0);
	pq_sendfloat8(buf, node->leaf_value);
	pq_sendint32(buf, node->feature_idx);
	pq_sendfloat4(buf, node->threshold);

	if (!node->is_leaf)
	{
		dt_serialize_node(buf, node->left);
		dt_serialize_node(buf, node->right);
	}
}

/*
 * dt_deserialize_node
 *
 * Recursively parse a tree node from a StringInfo buffer.
 */
static DTNode *
dt_deserialize_node(StringInfo buf)
{
	DTNode *node;
	int8	marker, is_leaf;

	marker = pq_getmsgint(buf, 1);
	if (marker == 0)
		return NULL;

	node = (DTNode *) palloc0(sizeof(DTNode));
	is_leaf = pq_getmsgint(buf, 1);
	node->is_leaf = (is_leaf != 0);
	node->leaf_value = pq_getmsgfloat8(buf);
	node->feature_idx = pq_getmsgint(buf, 4);
	node->threshold = pq_getmsgfloat4(buf);

	if (!node->is_leaf)
	{
		node->left = dt_deserialize_node(buf);
		node->right = dt_deserialize_node(buf);
	}

	return node;
}

/*
 * dt_model_serialize
 *
 * Serialize a DTModel structure into a binary blob.
 */
static bytea *
dt_model_serialize(const DTModel *model)
{
	StringInfoData	buf;
	bytea		   *result;

	if (model == NULL)
		return NULL;

	initStringInfo(&buf);

	pq_sendint32(&buf, model->model_id);
	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendint32(&buf, model->max_depth);
	pq_sendint32(&buf, model->min_samples_split);

	dt_serialize_node(&buf, model->root);

	result = (bytea *) palloc(VARHDRSZ + buf.len);
	SET_VARSIZE(result, VARHDRSZ + buf.len);
	memcpy(VARDATA(result), buf.data, buf.len);
	pfree(buf.data);

	return result;
}

/*
 * dt_model_deserialize
 *
 * Deserialize a binary blob into a DTModel struct.
 */
static DTModel *
dt_model_deserialize(const bytea *data)
{
	StringInfoData	buf;
	DTModel		   *model;

	if (data == NULL)
		return NULL;

	model = (DTModel *) palloc0(sizeof(DTModel));

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	model->model_id = pq_getmsgint(&buf, 4);
	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->max_depth = pq_getmsgint(&buf, 4);
	model->min_samples_split = pq_getmsgint(&buf, 4);

	if (model->n_features <= 0 || model->n_features > 10000)
	{
		pfree(model);
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("dt: invalid n_features %d in deserialized model (corrupted data?)", model->n_features)));
	}
	if (model->n_samples < 0 || model->n_samples > 100000000)
	{
		pfree(model);
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("dt: invalid n_samples %d in deserialized model (corrupted data?)", model->n_samples)));
	}

	model->root = dt_deserialize_node(&buf);

	return model;
}

/*
 * dt_metadata_is_gpu
 *
 * Return true if model metadata indicates GPU storage.
 */
static bool
dt_metadata_is_gpu(Jsonb *metadata)
{
	char   *meta_text = NULL;
	bool	is_gpu = false;

	if (metadata == NULL)
		return false;

	PG_TRY();
	{
		meta_text = DatumGetCString(DirectFunctionCall1(jsonb_out, JsonbPGetDatum(metadata)));
		if (strstr(meta_text, "\"storage\":\"gpu\"") != NULL)
			is_gpu = true;
		pfree(meta_text);
	}
	PG_CATCH();
	{
		is_gpu = false;
	}
	PG_END_TRY();

	return is_gpu;
}

/*
 * dt_try_gpu_predict_catalog
 *
 * Attempt to run prediction on GPU for given model and feature vector.
 */
static bool
dt_try_gpu_predict_catalog(int32 model_id, const Vector *feature_vec, double *result_out)
{
	bytea   *payload = NULL;
	Jsonb   *metrics = NULL;
	char    *gpu_err = NULL;
	double   prediction = 0.0;
	bool     success = false;

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

	if (!dt_metadata_is_gpu(metrics))
		goto cleanup;

	if (ndb_gpu_dt_predict(payload,
						  feature_vec->data,
						  feature_vec->dim,
						  &prediction,
						  &gpu_err) == 0)
	{
		if (result_out)
			*result_out = prediction;
		success = true;
	}

cleanup:
	if (payload)
		pfree(payload);
	if (metrics)
		pfree(metrics);
	if (gpu_err)
		pfree(gpu_err);

	return success;
}

/*
 * dt_load_model_from_catalog
 *
 * Load a model from catalog into a DTModel structure.
 */
static bool
dt_load_model_from_catalog(int32 model_id, DTModel **out)
{
	bytea   *payload = NULL;
	Jsonb   *metrics = NULL;

	if (out == NULL)
		return false;
	*out = NULL;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
	{
		if (metrics)
			pfree(metrics);
		return false;
	}

	*out = dt_model_deserialize(payload);
	if (*out == NULL)
	{
		if (metrics)
			pfree(metrics);
		return false;
	}

	if (metrics)
		pfree(metrics);

	return true;
}

PG_FUNCTION_INFO_V1(train_decision_tree_classifier);

/*
 * train_decision_tree_classifier
 *
 * SQL-callable UDF to train a new decision tree classifier, saving in catalog.
 */
Datum
train_decision_tree_classifier(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	int32		max_depth;
	int32		min_samples_split;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *label_str;
	DTDataset	dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_label;
	MLGpuTrainResult gpu_result;
	char	   *gpu_err = NULL;
	Jsonb	   *gpu_hyperparams = NULL;
	StringInfoData hyperbuf;
	int32		model_id = 0;
	int		   *indices = NULL;
	DTModel	   *model = NULL;
	bytea	   *model_blob = NULL;
	MLCatalogModelSpec spec;
	StringInfoData paramsbuf;
	StringInfoData metricsbuf;
	Jsonb	   *params_jsonb = NULL;
	Jsonb	   *metrics_jsonb = NULL;
	int			i;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);
	max_depth = PG_GETARG_INT32(3);
	min_samples_split = PG_GETARG_INT32(4);

	if (max_depth <= 0)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("dt: max_depth must be positive, got %d", max_depth)));
	if (min_samples_split < 2)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("dt: min_samples_split must be at least 2, got %d", min_samples_split)));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	dt_dataset_init(&dataset);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_label = quote_identifier(label_str);

	dt_dataset_load(quoted_tbl, quoted_feat, quoted_label, &dataset);

	if (dataset.n_samples < 10)
	{
		dt_dataset_free(&dataset);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		ereport(ERROR, (errmsg("dt: need at least 10 samples, got %d", dataset.n_samples)));
	}

	/* Try GPU path */
	if (neurondb_gpu_is_available() &&
		dataset.n_samples > 0 &&
		dataset.feature_dim > 0)
	{
		initStringInfo(&hyperbuf);
		appendStringInfo(&hyperbuf, "{\"max_depth\":%d,\"min_samples_split\":%d}",
						 max_depth, min_samples_split);
		gpu_hyperparams = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(hyperbuf.data)));

		if (ndb_gpu_try_train_model("decision_tree",
								   NULL,
								   NULL,
								   tbl_str,
								   label_str,
								   NULL,
								   0,
								   gpu_hyperparams,
								   dataset.features,
								   dataset.labels,
								   dataset.n_samples,
								   dataset.feature_dim,
								   0,
								   &gpu_result,
								   &gpu_err) &&
			gpu_result.spec.model_data != NULL)
		{
			elog(DEBUG1, "dt: GPU training succeeded");
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

			spec.algorithm = "decision_tree";
			spec.model_type = "classification";

			model_id = ml_catalog_register_model(&spec);

			if (gpu_err)
				pfree(gpu_err);
			if (gpu_hyperparams)
				pfree(gpu_hyperparams);
			ndb_gpu_free_train_result(&gpu_result);
			pfree(hyperbuf.data);
			dt_dataset_free(&dataset);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);

			PG_RETURN_INT32(model_id);
		}
		if (gpu_err)
			pfree(gpu_err);
		if (gpu_hyperparams)
			pfree(gpu_hyperparams);

		ndb_gpu_free_train_result(&gpu_result);
		pfree(hyperbuf.data);

		elog(DEBUG1, "dt: GPU training unavailable, falling back to CPU");
	}

	/* Fallback CPU training path */
	indices = (int *) palloc(sizeof(int) * dataset.n_samples);

	for (i = 0; i < dataset.n_samples; i++)
		indices[i] = i;

	model = (DTModel *) palloc0(sizeof(DTModel));
	model->n_features = dataset.feature_dim;
	model->n_samples = dataset.n_samples;
	model->max_depth = max_depth;
	model->min_samples_split = min_samples_split;
	model->root = build_tree_1d(dataset.features, dataset.labels, indices,
								dataset.n_samples, dataset.feature_dim, max_depth,
								min_samples_split, true);

	model_blob = dt_model_serialize(model);
	if (model_blob == NULL)
	{
		dt_free_tree(model->root);
		pfree(model);
		pfree(indices);
		dt_dataset_free(&dataset);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(label_str);
		ereport(ERROR, (errmsg("dt: failed to serialize model")));
	}

	initStringInfo(&paramsbuf);
	appendStringInfo(&paramsbuf, "{\"max_depth\":%d,\"min_samples_split\":%d}",
					 max_depth, min_samples_split);
	params_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(paramsbuf.data)));

	initStringInfo(&metricsbuf);
	appendStringInfo(&metricsbuf,
					 "{\"algorithm\":\"decision_tree\","
					 "\"storage\":\"cpu\","
					 "\"n_features\":%d,"
					 "\"n_samples\":%d,"
					 "\"max_depth\":%d,"
					 "\"min_samples_split\":%d}",
					 dataset.feature_dim, dataset.n_samples, max_depth, min_samples_split);
	metrics_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metricsbuf.data)));

	memset(&spec, 0, sizeof(spec));
	spec.algorithm = "decision_tree";
	spec.model_type = "classification";
	spec.training_table = tbl_str;
	spec.training_column = label_str;
	spec.model_data = model_blob;
	spec.parameters = params_jsonb;
	spec.metrics = metrics_jsonb;

	model_id = ml_catalog_register_model(&spec);
	model->model_id = model_id;

	/* Cleanup */
	dt_free_tree(model->root);
	pfree(model);
	pfree(indices);
	pfree(model_blob);

	pfree(paramsbuf.data);
	pfree(metricsbuf.data);

	dt_dataset_free(&dataset);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(label_str);

	PG_RETURN_INT32(model_id);
}

PG_FUNCTION_INFO_V1(predict_decision_tree_model_id);

/*
 * predict_decision_tree_model_id
 *
 * SQL-callable UDF for prediction using a trained decision tree model.
 */
Datum
predict_decision_tree_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	Vector	   *feature_vec = PG_GETARG_VECTOR_P(1);
	DTModel	   *model = NULL;
	double		result = 0.0;
	bool		found_gpu = false;

	if (feature_vec == NULL)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("dt: feature vector cannot be NULL")));

	if (dt_try_gpu_predict_catalog(model_id, feature_vec, &result))
	{
		found_gpu = true;
		elog(DEBUG1, "dt: GPU prediction succeeded for model_id=%d", model_id);
	}

	if (!found_gpu)
	{
		if (!dt_load_model_from_catalog(model_id, &model))
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("dt: model %d not found", model_id)));

		if (model->root == NULL)
		{
			pfree(model);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("dt: model %d has NULL root (corrupted?)", model_id)));
		}

		if (feature_vec->dim != model->n_features)
		{
			pfree(model);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("dt: feature dimension mismatch: expected %d, got %d",
								   model->n_features, feature_vec->dim)));
		}

		result = dt_tree_predict(model->root, feature_vec->data, model->n_features);

		dt_free_tree(model->root);
		pfree(model);
	}

	PG_RETURN_FLOAT8(result);
}

/*
 * dt_predict_batch
 *
 * Helper function to predict a batch of samples using Decision Tree model.
 * Updates confusion matrix.
 */
static void
dt_predict_batch(const DTModel *model,
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

	if (model == NULL || model->root == NULL || features == NULL || labels == NULL || n_samples <= 0)
	{
		elog(DEBUG1,
			"neurondb: dt_predict_batch: early return - model=%p, model->root=%p, features=%p, labels=%p, n_samples=%d",
			(void *)model, model ? (void *)model->root : NULL, (void *)features, (void *)labels, n_samples);
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
		"neurondb: dt_predict_batch: starting - n_samples=%d, feature_dim=%d, model->n_features=%d",
		n_samples, feature_dim, model->n_features);

	for (i = 0; i < n_samples; i++)
	{
		const float *row = features + (i * feature_dim);
		double y_true = labels[i];
		int true_class;
		double prediction;
		int pred_class;

		if (!isfinite(y_true))
			continue;

		true_class = (int)rint(y_true);
		if (true_class < 0 || true_class > 1)
			continue;

		/* Compute prediction using tree */
		prediction = dt_tree_predict(model->root, row, feature_dim);
		pred_class = (int)rint(prediction);
		if (pred_class < 0)
			pred_class = 0;
		if (pred_class > 1)
			pred_class = 1;
		
		if (i < 5)  /* Log first 5 predictions for debugging */
		{
			elog(DEBUG1,
				"neurondb: dt_predict_batch: sample %d - y_true=%.6f (class=%d), prediction=%.6f (class=%d)",
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
 * evaluate_decision_tree_by_model_id
 *
 * Evaluates Decision Tree model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: accuracy, precision, recall, f1_score, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_decision_tree_by_model_id);

Datum
evaluate_decision_tree_by_model_id(PG_FUNCTION_ARGS)
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
	DTModel *model = NULL;
	StringInfoData jsonbuf;
	Jsonb *result_jsonb = NULL;
	bytea *gpu_payload = NULL;
	Jsonb *gpu_metrics = NULL;
	bool is_gpu_model = false;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_decision_tree_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_decision_tree_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Load model from catalog - try CPU first, then GPU */
	if (!dt_load_model_from_catalog(model_id, &model))
	{
		/* Try GPU model */
		if (ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
		{
			is_gpu_model = dt_metadata_is_gpu(gpu_metrics);
			if (!is_gpu_model)
			{
				if (gpu_payload)
					pfree(gpu_payload);
				if (gpu_metrics)
					pfree(gpu_metrics);
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: evaluate_decision_tree_by_model_id: model %d not found",
							model_id)));
			}
		}
		else
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_decision_tree_by_model_id: model %d not found",
						model_id)));
		}
	}

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
	{
		if (model != NULL)
		{
			if (model->root != NULL)
				dt_free_tree(model->root);
			pfree(model);
		}
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_decision_tree_by_model_id: SPI_connect failed")));
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

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		pfree(query.data);
		if (model != NULL)
		{
			if (model->root != NULL)
				dt_free_tree(model->root);
			pfree(model);
		}
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_decision_tree_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		pfree(query.data);
		if (model != NULL)
		{
			if (model->root != NULL)
				dt_free_tree(model->root);
			pfree(model);
		}
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_decision_tree_by_model_id: no valid rows found")));
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
		/* For now, GPU evaluation for Decision Tree is not implemented, fall back to CPU */
		elog(DEBUG1,
			"neurondb: evaluate_decision_tree_by_model_id: GPU evaluation not yet implemented for Decision Tree, using CPU batch evaluation");
		goto gpu_eval_fallback;
#endif	/* NDB_GPU_CUDA */
	}

gpu_eval_fallback:
	/* CPU evaluation path (also used as fallback for GPU models) */
	/* Use optimized batch prediction */
	{
		float *h_features = NULL;
		double *h_labels = NULL;
		int feat_dim = 0;
		int valid_rows = 0;

		/* Determine feature dimension from model */
		if (model != NULL)
			feat_dim = model->n_features;
		else if (is_gpu_model && gpu_payload != NULL)
		{
			const NdbCudaDtModelHeader *gpu_hdr;

			gpu_hdr = (const NdbCudaDtModelHeader *)VARDATA(gpu_payload);
			feat_dim = gpu_hdr->feature_dim;
		}
		else
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_decision_tree_by_model_id: could not determine feature dimension")));
		}

		if (feat_dim <= 0)
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: evaluate_decision_tree_by_model_id: invalid feature dimension %d",
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
				if (model->root != NULL)
					dt_free_tree(model->root);
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
					errmsg("neurondb: evaluate_decision_tree_by_model_id: no valid rows found")));
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
					errmsg("neurondb: evaluate_decision_tree_by_model_id: GPU model evaluation requires GPU evaluation kernel (not yet implemented)")));
		}

		/* Ensure model is not NULL before prediction */
		if (model == NULL)
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
					errmsg("neurondb: evaluate_decision_tree_by_model_id: model is NULL")));
		}

		/* Use batch prediction helper */
		/* Add debug logging before prediction */
		if (model != NULL)
		{
			elog(DEBUG1,
				"neurondb: evaluate_decision_tree_by_model_id: CPU batch prediction: model->n_features=%d, model->root=%p, valid_rows=%d, feat_dim=%d",
				model->n_features, (void *)model->root, valid_rows, feat_dim);
		}
		else
		{
			elog(WARNING,
				"neurondb: evaluate_decision_tree_by_model_id: model is NULL before CPU batch prediction");
		}
		
		dt_predict_batch(model,
			h_features,
			h_labels,
			valid_rows,
			feat_dim,
			&tp,
			&tn,
			&fp,
			&fn);
		
		elog(DEBUG1,
			"neurondb: evaluate_decision_tree_by_model_id: after batch prediction: tp=%d, tn=%d, fp=%d, fn=%d, valid_rows=%d",
			tp, tn, fp, fn, valid_rows);

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
			if (model->root != NULL)
				dt_free_tree(model->root);
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
 * GPU Model Ops Registration Stub for Decision Tree
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"

void
neurondb_gpu_register_dt_model(void)
{
	elog(DEBUG1, "Decision Tree GPU Model Ops registration skipped - not yet implemented");
}
