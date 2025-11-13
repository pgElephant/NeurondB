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

#include <math.h>
#include <float.h>
#include <limits.h>

/* Use DTNode from internal header */
#define TreeNode DTNode

/* Decision Tree dataset structure */
typedef struct DTDataset
{
	float *features;
	double *labels;
	int n_samples;
	int feature_dim;
} DTDataset;

/* Forward declarations */
static void dt_dataset_init(DTDataset *dataset);
static void dt_dataset_free(DTDataset *dataset);
static void dt_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_label,
	DTDataset *dataset);
static bytea *dt_model_serialize(const DTModel *model);
static DTModel *dt_model_deserialize(const bytea *data);
static bool dt_metadata_is_gpu(Jsonb *metadata);
static bool dt_try_gpu_predict_catalog(int32 model_id,
	const Vector *feature_vec,
	double *result_out);
static bool dt_load_model_from_catalog(int32 model_id, DTModel **out);
static void dt_free_tree(DTNode *node);
static double dt_tree_predict(const DTNode *node, const float *x, int dim);

/*
 * dt_dataset_init
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
 */
static void
dt_dataset_free(DTDataset *dataset)
{
	if (dataset == NULL)
		return;
	if (dataset->features != NULL)
		pfree(dataset->features);
	if (dataset->labels != NULL)
		pfree(dataset->labels);
	dt_dataset_init(dataset);
}

/*
 * dt_dataset_load
 */
static void
dt_dataset_load(const char *quoted_tbl,
	const char *quoted_feat,
	const char *quoted_label,
	DTDataset *dataset)
{
	StringInfoData query;
	MemoryContext oldcontext;
	int ret;
	int n_samples = 0;
	int feature_dim = 0;
	int i;

	if (dataset == NULL)
		ereport(ERROR,
			(errmsg("dt_dataset_load: dataset is NULL")));

	oldcontext = CurrentMemoryContext;

	/* Initialize query in caller's context before SPI_connect */
	initStringInfo(&query);
	MemoryContextSwitchTo(oldcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errmsg("dt_dataset_load: SPI_connect failed")));
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quoted_feat, quoted_label, quoted_tbl, quoted_feat, quoted_label);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		ereport(ERROR,
			(errmsg("dt_dataset_load: query failed")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		SPI_finish();
		ereport(ERROR,
			(errmsg("dt_dataset_load: need at least 10 samples, got %d",
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
			(errmsg("dt_dataset_load: could not determine feature dimension")));
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
		Datum label_datum;
		bool feat_null;
		bool label_null;
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
				(errmsg("dt_dataset_load: inconsistent vector dimensions")));
		}

		row = dataset->features + (i * feature_dim);
		memcpy(row, vec->data, sizeof(float) * feature_dim);

		label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);
		if (label_null)
			continue;

		{
			Oid label_type = SPI_gettypeid(tupdesc, 2);

			if (label_type == INT2OID || label_type == INT4OID
				|| label_type == INT8OID)
				dataset->labels[i] = (double)DatumGetInt32(label_datum);
			else
				dataset->labels[i] = DatumGetFloat8(label_datum);
		}
	}

	dataset->n_samples = n_samples;
	dataset->feature_dim = feature_dim;

	SPI_finish();
}

/* ---- Data Extraction (Legacy - will be replaced) ---- */

/*
 * Helper function: Fetch all data from a table for features and label.
 * Reads into float**, double*, etc.
 *
 * The caller must pfree arrays.
 * Features will be shaped as X[sample][dim],
 * label as y[sample].
 */
static void __attribute__((unused))
extract_training_data_legacy(const char *table_name, const char *feature_col, const char *label_col,
					 float ***X_p, double **y_p, int *n_samples_p, int *dim_p)
{
	StringInfoData sql;
	int			ret;
	int			n_samples;
	int			dim;
	float	  **X;
	double	   *y;
	int			i, j;

	initStringInfo(&sql);

	/* Only allow feature_col to be an array column for simplicity */
	appendStringInfo(&sql,
		"SELECT %s, %s FROM %s",
		feature_col, label_col, table_name);

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		elog(ERROR, "SPI_execute failed: %d", ret);
	}

	n_samples = SPI_processed;
	if (n_samples <= 0)
	{
		SPI_finish();
		ereport(ERROR, (errmsg("No rows returned from training data")));
	}

	X = (float **) palloc0(sizeof(float *) * n_samples);
	y = (double *) palloc0(sizeof(double) * n_samples);

	dim = -1;

	for (i = 0; i < n_samples; i++)
	{
		bool	isnull;
		Datum	arr_datum;
		Datum	label_datum;
		ArrayType *arr;
		Oid		eltype;
		int		ndim;
		int		*dim_vec;
		int		nitems;
		float4  *vals_float4 = NULL;
		float8  *vals_float8 = NULL;
		bool    is_float8 = false;

		/* features array */
		arr_datum = SPI_getbinval(SPI_tuptable->vals[i],
								  SPI_tuptable->tupdesc,
								  1,
								  &isnull);
		if (isnull)
		{
			SPI_finish();
			ereport(ERROR, (errmsg("NULL feature array in row %d", i + 1)));
		}

		arr = DatumGetArrayTypeP(arr_datum);
		eltype = ARR_ELEMTYPE(arr);

		ndim = ARR_NDIM(arr);
		dim_vec = ARR_DIMS(arr);

		if (ndim != 1)
		{
			SPI_finish();
			ereport(ERROR,
				(errmsg("Only 1-dimensional arrays of floats supported for features")));
		}

		nitems = dim_vec[0];

		if (dim == -1)
			dim = nitems;
		else if (nitems != dim)
		{
			SPI_finish();
			ereport(ERROR,
					(errmsg("All feature arrays must have the same dimension")));
		}

		switch (eltype)
		{
			case FLOAT4OID:
				vals_float4 = (float4 *) ARR_DATA_PTR(arr);
				break;
			case FLOAT8OID:
				vals_float8 = (float8 *) ARR_DATA_PTR(arr);
				is_float8 = true;
				break;
			default:
				SPI_finish();
				ereport(ERROR,
					(errmsg("Features array must be float4[] or float8[]")));
		}

		X[i] = (float *) palloc0(sizeof(float) * dim);
		for (j = 0; j < dim; j++)
		{
			if (is_float8)
				X[i][j] = (float) vals_float8[j];
			else
				X[i][j] = (float) vals_float4[j];
		}

		/* label */
		label_datum = SPI_getbinval(SPI_tuptable->vals[i],
									SPI_tuptable->tupdesc,
									2,
									&isnull);
		if (isnull)
		{
			SPI_finish();
			ereport(ERROR, (errmsg("NULL label in row %d", i + 1)));
		}
		/* support float8 and int4 label columns */
		if (SPI_gettypeid(SPI_tuptable->tupdesc, 2) == FLOAT8OID)
			y[i] = DatumGetFloat8(label_datum);
		else if (SPI_gettypeid(SPI_tuptable->tupdesc, 2) == FLOAT4OID)
			y[i] = (double) DatumGetFloat4(label_datum);
		else if (SPI_gettypeid(SPI_tuptable->tupdesc, 2) == INT4OID)
			y[i] = (double) DatumGetInt32(label_datum);
		else if (SPI_gettypeid(SPI_tuptable->tupdesc, 2) == INT2OID)
			y[i] = (double) DatumGetInt16(label_datum);
		else
		{
			SPI_finish();
			ereport(ERROR,
				(errmsg("Label column must be float8, float4, int2, or int4")));
		}
	}

	SPI_finish();

	*X_p = X;
	*y_p = y;
	*n_samples_p = n_samples;
	*dim_p = dim;
}

/*
 * Compute Gini impurity for classification
 */
static double
compute_gini(const double *labels, int n)
{
	/* NB: Only binary class {0,1} */
	int i;
	int class0 = 0, class1 = 0;
	double p0, p1;

	if (n == 0)
		return 0.0;

	for (i = 0; i < n; i++)
	{
		if (((int)labels[i]) == 0)
			class0++;
		else
			class1++;
	}

	p0 = (double)class0 / (double)n;
	p1 = (double)class1 / (double)n;
	return 1.0 - (p0 * p0 + p1 * p1);
}

/*
 * Compute variance for regression
 */
static double
compute_variance(const double *values, int n)
{
	int i;
	double mean = 0.0, var = 0.0;

	if (n == 0)
		return 0.0;

	for (i = 0; i < n; i++)
		mean += values[i];
	mean /= n;

	for (i = 0; i < n; i++)
	{
		double d = values[i] - mean;
		var += d * d;
	}
	var /= n;

	return var;
}

/*
 * Find best split for a feature (using 1D array)
 */
static void
find_best_split_1d(const float *features, const double *labels, const int *indices,
	int n_samples, int dim,
	int *best_feature, float *best_threshold, double *best_gain,
	bool is_classification)
{
	int feat;
	*best_gain = -DBL_MAX;
	*best_feature = -1;
	*best_threshold = 0.0;

	for (feat = 0; feat < dim; feat++)
	{
		/* Determine the range for this feature */
		float min_val = FLT_MAX;
		float max_val = -FLT_MAX;
		int ii;

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

		/* Try 10 candidate thresholds, uniformly spaced */
		for (ii = 1; ii < 10; ii++)
		{
			float threshold = min_val + (max_val - min_val) * ii / 10.0f;
			int left_count, right_count, j;
			double *left_y, *right_y;
			int l_idx, r_idx;
			double left_imp, right_imp, gain;

			left_count = right_count = 0;
			for (j = 0; j < n_samples; j++)
			{
				if (features[indices[j] * dim + feat] <= threshold)
					left_count++;
				else
					right_count++;
			}
			/* Don't allow degenerated splits */
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
			/* Check all y assigned */
			Assert(l_idx == left_count && r_idx == right_count);

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

			/* Information gain (classification) or variance reduction (regression) */
			gain = -(((double)left_count / (double)n_samples) * left_imp +
					 ((double)right_count / (double)n_samples) * right_imp);

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
 * Find best split for a feature (legacy float** version)
 */
static void __attribute__((unused))
find_best_split(float **X, double *y, int *indices, int n_samples, int dim,
				int *best_feature, float *best_threshold, double *best_gain,
				bool is_classification)
{
	/* Legacy function - kept for compatibility during refactoring */
	/* This will be removed once all code is migrated to 1D arrays */
	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("find_best_split: legacy float** version is deprecated")));
}

/*
 * Build decision tree recursively (using 1D array)
 */
static DTNode *
build_tree_1d(const float *features, const double *labels, const int *indices,
	int n_samples, int dim,
	int max_depth, int min_samples_split, bool is_classification)
{
	DTNode   *node;
	int			i;
	int			best_feature;
	float		best_threshold;
	double		best_gain;
	int		   *left_indices;
	int		   *right_indices;
	int			left_count = 0;
	int			right_count = 0;

	node = (DTNode *) palloc0(sizeof(DTNode));

	/* Stopping criteria */
	if (max_depth == 0 || n_samples < min_samples_split)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			int class0 = 0, class1 = 0;
			for (i = 0; i < n_samples; i++)
			{
				int l = (int)labels[indices[i]];
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

	/* Find best split */
	find_best_split_1d(features, labels, indices, n_samples, dim,
		&best_feature, &best_threshold, &best_gain, is_classification);

	/* If no good split, also make leaf */
	if (best_feature == -1)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			int class0 = 0, class1 = 0;
			for (i = 0; i < n_samples; i++)
			{
				int l = (int)labels[indices[i]];
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

	/* Partition indices */
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
 * Build decision tree recursively (legacy float** version)
 */
static TreeNode * __attribute__((unused))
build_tree(float **X, double *y, int *indices, int n_samples, int dim,
		   int max_depth, int min_samples_split, bool is_classification)
{
	/* Legacy function - kept for compatibility during refactoring */
	/* This will be removed once all code is migrated to 1D arrays */
	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("build_tree: legacy float** version is deprecated")));
	return NULL;
}

/*
 * dt_tree_predict
 * Predict using decision tree
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
 * Serialize the tree recursively to a text representation (JSON-like).
 * Stores in buffer. Used for demonstration; a binary format would be faster.
 */
static void __attribute__((unused))
serialize_tree(const TreeNode *node, StringInfo buf, int depth)
{
	if (node == NULL)
	{
		appendStringInfoString(buf, "null");
		return;
	}
	if (node->is_leaf)
	{
		appendStringInfo(buf,
						 "{\"leaf\": %.6f}",
						 node->leaf_value);
	}
	else
	{
		appendStringInfoString(buf, "{");
		appendStringInfo(buf, "\"feature\": %d, ", node->feature_idx);
		appendStringInfo(buf, "\"threshold\": %.6f, ", node->threshold);
		appendStringInfoString(buf, "\"left\": ");
		serialize_tree(node->left, buf, depth + 1);
		appendStringInfoString(buf, ", \"right\": ");
		serialize_tree(node->right, buf, depth + 1);
		appendStringInfoString(buf, "}");
	}
}

/*
 * dt_free_tree
 * Release tree memory recursively
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
 * Serialize a tree node recursively to binary format
 */
static void
dt_serialize_node(StringInfo buf, const DTNode *node)
{
	if (node == NULL)
	{
		pq_sendint8(buf, 0); /* NULL marker */
		return;
	}

	pq_sendint8(buf, 1); /* Non-NULL marker */
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
 * Deserialize a tree node recursively from binary format
 */
static DTNode *
dt_deserialize_node(StringInfo buf)
{
	DTNode *node;
	int8 marker;
	int8 is_leaf;

	marker = pq_getmsgint(buf, 1);
	if (marker == 0)
		return NULL; /* NULL node */

	node = (DTNode *)palloc0(sizeof(DTNode));
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
 */
static bytea *
dt_model_serialize(const DTModel *model)
{
	StringInfoData buf;
	bytea *result;

	if (model == NULL)
		return NULL;

	initStringInfo(&buf);

	/* Write model header */
	pq_sendint32(&buf, model->model_id);
	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendint32(&buf, model->max_depth);
	pq_sendint32(&buf, model->min_samples_split);

	/* Serialize tree */
	dt_serialize_node(&buf, model->root);

	result = (bytea *)palloc(VARHDRSZ + buf.len);
	SET_VARSIZE(result, VARHDRSZ + buf.len);
	memcpy(VARDATA(result), buf.data, buf.len);
	pfree(buf.data);

	return result;
}

/*
 * dt_model_deserialize
 */
static DTModel *
dt_model_deserialize(const bytea *data)
{
	StringInfoData buf;
	DTModel *model;

	if (data == NULL)
		return NULL;

	model = (DTModel *)palloc0(sizeof(DTModel));

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	/* Read model header */
	model->model_id = pq_getmsgint(&buf, 4);
	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->max_depth = pq_getmsgint(&buf, 4);
	model->min_samples_split = pq_getmsgint(&buf, 4);

	/* Validate deserialized values */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		pfree(model);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("dt: invalid n_features %d in deserialized model (corrupted data?)",
					model->n_features)));
	}
	if (model->n_samples < 0 || model->n_samples > 100000000)
	{
		pfree(model);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("dt: invalid n_samples %d in deserialized model (corrupted data?)",
					model->n_samples)));
	}

	/* Deserialize tree */
	model->root = dt_deserialize_node(&buf);

	return model;
}

/*
 * dt_metadata_is_gpu
 */
static bool
dt_metadata_is_gpu(Jsonb *metadata)
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
 * dt_try_gpu_predict_catalog
 */
static bool
dt_try_gpu_predict_catalog(int32 model_id,
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

	if (!dt_metadata_is_gpu(metrics))
		goto cleanup;

	if (ndb_gpu_dt_predict(payload,
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
 * dt_load_model_from_catalog
 */
static bool
dt_load_model_from_catalog(int32 model_id, DTModel **out)
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

	*out = dt_model_deserialize(payload);
	if (*out == NULL)
	{
		if (metrics != NULL)
			pfree(metrics);
		return false;
	}

	if (metrics != NULL)
		pfree(metrics);

	return true;
}

PG_FUNCTION_INFO_V1(train_decision_tree_classifier);

/*
 * train_decision_tree_classifier
 *	Trains a decision tree for classification (binary).
 *	params:
 *		table_name text,
 *		feature_col text,
 *		label_col text,
 *		max_depth int,
 *		min_samples_split int
 *
 *	Returns model_id (integer)
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
	DTDataset dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_label;
	MLGpuTrainResult gpu_result;
	char *gpu_err = NULL;
	Jsonb *gpu_hyperparams = NULL;
	StringInfoData hyperbuf;
	int32 model_id = 0;
	int *indices = NULL;
	DTModel *model = NULL;
	bytea *model_blob = NULL;
	MLCatalogModelSpec spec;
	StringInfoData paramsbuf;
	StringInfoData metricsbuf;
	Jsonb *params_jsonb = NULL;
	Jsonb *metrics_jsonb = NULL;
	int i;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);
	max_depth = PG_GETARG_INT32(3);
	min_samples_split = PG_GETARG_INT32(4);

	if (max_depth <= 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("dt: max_depth must be positive, got %d", max_depth)));
	if (min_samples_split < 2)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("dt: min_samples_split must be at least 2, got %d",
					min_samples_split)));

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
		ereport(ERROR,
			(errmsg("dt: need at least 10 samples, got %d", dataset.n_samples)));
	}

	/* Try GPU training first */
	if (neurondb_gpu_is_available() && dataset.n_samples > 0
		&& dataset.feature_dim > 0)
	{
		initStringInfo(&hyperbuf);
		appendStringInfo(&hyperbuf,
			"{\"max_depth\":%d,\"min_samples_split\":%d}",
			max_depth, min_samples_split);
		gpu_hyperparams = DatumGetJsonbP(DirectFunctionCall1(
			jsonb_in, CStringGetDatum(hyperbuf.data)));

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
			&gpu_err)
			&& gpu_result.spec.model_data != NULL)
		{
			elog(NOTICE, "dt: GPU training succeeded");
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

			if (gpu_err != NULL)
				pfree(gpu_err);
			if (gpu_hyperparams != NULL)
				pfree(gpu_hyperparams);
			ndb_gpu_free_train_result(&gpu_result);
			pfree(hyperbuf.data);
			dt_dataset_free(&dataset);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(label_str);

			PG_RETURN_INT32(model_id);
		}

		if (gpu_err != NULL)
			pfree(gpu_err);
		if (gpu_hyperparams != NULL)
			pfree(gpu_hyperparams);
		ndb_gpu_free_train_result(&gpu_result);
		pfree(hyperbuf.data);
		elog(DEBUG1, "dt: GPU training unavailable, falling back to CPU");
	}

	/* CPU training path */
	{
		indices = (int *)palloc(sizeof(int) * dataset.n_samples);
		for (i = 0; i < dataset.n_samples; i++)
			indices[i] = i;

		model = (DTModel *)palloc0(sizeof(DTModel));
		model->n_features = dataset.feature_dim;
		model->n_samples = dataset.n_samples;
		model->max_depth = max_depth;
		model->min_samples_split = min_samples_split;
		model->root = build_tree_1d(dataset.features, dataset.labels, indices,
			dataset.n_samples, dataset.feature_dim,
			max_depth, min_samples_split, true /* classification */);

		/* Serialize model */
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
			ereport(ERROR,
				(errmsg("dt: failed to serialize model")));
		}

		/* Build hyperparameters JSON */
		initStringInfo(&paramsbuf);
		appendStringInfo(&paramsbuf,
			"{\"max_depth\":%d,\"min_samples_split\":%d}",
			max_depth, min_samples_split);
		params_jsonb = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(paramsbuf.data)));

		/* Build metrics JSON */
		initStringInfo(&metricsbuf);
		appendStringInfo(&metricsbuf,
			"{\"algorithm\":\"decision_tree\","
			"\"storage\":\"cpu\","
			"\"n_features\":%d,"
			"\"n_samples\":%d,"
			"\"max_depth\":%d,"
			"\"min_samples_split\":%d}",
			dataset.feature_dim,
			dataset.n_samples,
			max_depth,
			min_samples_split);
		metrics_jsonb = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum(metricsbuf.data)));

		/* Register model in catalog */
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
	}

	dt_dataset_free(&dataset);
	pfree(tbl_str);
	pfree(feat_str);
	pfree(label_str);

	PG_RETURN_INT32(model_id);
}

PG_FUNCTION_INFO_V1(predict_decision_tree_model_id);

/*
 * predict_decision_tree_model_id
 * Predict using Decision Tree model from catalog
 */
Datum
predict_decision_tree_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id = PG_GETARG_INT32(0);
	Vector *feature_vec = PG_GETARG_VECTOR_P(1);
	DTModel *model = NULL;
	double result = 0.0;
	bool found_gpu = false;

	if (feature_vec == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("dt: feature vector cannot be NULL")));

	/* Try GPU prediction first */
	if (dt_try_gpu_predict_catalog(model_id, feature_vec, &result))
	{
		found_gpu = true;
		elog(DEBUG1, "dt: GPU prediction succeeded for model_id=%d", model_id);
	}

	/* Fall back to CPU prediction */
	if (!found_gpu)
	{
		if (!dt_load_model_from_catalog(model_id, &model))
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("dt: model %d not found", model_id)));
		}

		if (model->root == NULL)
		{
			pfree(model);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("dt: model %d has NULL root (corrupted?)", model_id)));
		}

		if (feature_vec->dim != model->n_features)
		{
			pfree(model);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("dt: feature dimension mismatch: expected %d, got %d",
						model->n_features, feature_vec->dim)));
		}

		result = dt_tree_predict(model->root, feature_vec->data, model->n_features);

		/* Free model (tree will be freed recursively) */
		dt_free_tree(model->root);
		pfree(model);
	}

	PG_RETURN_FLOAT8(result);
}
