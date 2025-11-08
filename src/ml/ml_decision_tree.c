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

#include "neurondb.h"
#include "neurondb_ml.h"

#include <math.h>
#include <float.h>
#include <limits.h>

/* Tree node structure */
typedef struct TreeNode
{
	bool		is_leaf;
	double		leaf_value;			/* For leaves: class (classification) or value (regression) */
	int			feature_idx;		/* For internal nodes: feature to split on */
	float		threshold;			/* For internal nodes: split threshold */
	struct TreeNode *left;			/* Samples <= threshold */
	struct TreeNode *right;			/* Samples > threshold */
} TreeNode;

/* ---- Data Extraction ---- */

/*
 * Helper function: Fetch all data from a table for features and label.
 * Reads into float**, double*, etc.
 *
 * The caller must pfree arrays.
 * Features will be shaped as X[sample][dim],
 * label as y[sample].
 */
static void
extract_training_data(const char *table_name, const char *feature_col, const char *label_col,
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
		Datum label_datum;
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
 * Find best split for a feature
 */
static void
find_best_split(float **X, double *y, int *indices, int n_samples, int dim,
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
			float val = X[indices[ii]][feat];
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
				if (X[indices[j]][feat] <= threshold)
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
				if (X[indices[j]][feat] <= threshold)
					left_y[l_idx++] = y[indices[j]];
				else
					right_y[r_idx++] = y[indices[j]];
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
 * Build decision tree recursively
 */
static TreeNode *
build_tree(float **X, double *y, int *indices, int n_samples, int dim,
		   int max_depth, int min_samples_split, bool is_classification)
{
	TreeNode   *node;
	int			i;
	int			best_feature;
	float		best_threshold;
	double		best_gain;
	int		   *left_indices;
	int		   *right_indices;
	int			left_count = 0;
	int			right_count = 0;

	node = (TreeNode *) palloc0(sizeof(TreeNode));

	/* Stopping criteria */
	if (max_depth == 0 || n_samples < min_samples_split)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			int class0 = 0, class1 = 0;
			for (i = 0; i < n_samples; i++)
			{
				int l = (int)y[indices[i]];
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
				sum += y[indices[i]];
			node->leaf_value = sum / n_samples;
		}
		return node;
	}

	/* Find best split */
	find_best_split(X, y, indices, n_samples, dim, &best_feature, &best_threshold, &best_gain, is_classification);

	/* If no good split, also make leaf */
	if (best_feature == -1)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			int class0 = 0, class1 = 0;
			for (i = 0; i < n_samples; i++)
			{
				int l = (int)y[indices[i]];
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
				sum += y[indices[i]];
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
		if (X[indices[i]][best_feature] <= best_threshold)
			left_indices[left_count++] = indices[i];
		else
			right_indices[right_count++] = indices[i];
	}
	Assert(left_count + right_count == n_samples);

	node->left = build_tree(X, y, left_indices, left_count, dim,
							max_depth - 1, min_samples_split, is_classification);
	node->right = build_tree(X, y, right_indices, right_count, dim,
							max_depth - 1, min_samples_split, is_classification);

	pfree(left_indices);
	pfree(right_indices);

	return node;
}

/*
 * Predict using decision tree
 */
static double __attribute__((unused))
tree_predict(const TreeNode *node, const float *x)
{
	if (node == NULL)
		elog(ERROR, "NULL node in predict");
	if (node->is_leaf)
		return node->leaf_value;
	if (x[node->feature_idx] <= node->threshold)
		return tree_predict(node->left, x);
	else
		return tree_predict(node->right, x);
}

/*
 * Serialize the tree recursively to a text representation (JSON-like).
 * Stores in buffer. Used for demonstration; a binary format would be faster.
 */
static void
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
 * Release tree memory recursively
 */
static void
free_tree(TreeNode *node)
{
	if (node == NULL)
		return;
	if (!node->is_leaf)
	{
		free_tree(node->left);
		free_tree(node->right);
	}
	pfree(node);
}

PG_FUNCTION_INFO_V1(train_decision_tree_classifier);

/*
 * train_decision_tree_classifier
 *	Trains a decision tree for classification (binary).
 *	params:
 *		table_name text,
 *		feature_col text (column with float4[] features per row!),
 *		label_col text (int/float col, for binary)
 *		max_depth int,
 *		min_samples_split int
 *
 *	Returns JSON tree description as text.
 */
Datum
train_decision_tree_classifier(PG_FUNCTION_ARGS)
{
	text	   *table_name_text;
	text	   *feature_col_text;
	text	   *label_col_text;
	int			max_depth;
	int			min_samples_split;

	char	   *table_name;
	char	   *feature_col;
	char	   *label_col;
	float	  **X = NULL;
	double	   *y = NULL;
	int			n_samples = 0;
	int			dim = 0;
	int			i = 0;

	int		   *indices = NULL;
	TreeNode   *tree = NULL;
	StringInfoData treedesc;

	PG_TRY();
	{
		/* Validate/obtain arguments */
		table_name_text = PG_GETARG_TEXT_PP(0);
		feature_col_text = PG_GETARG_TEXT_PP(1);
		label_col_text = PG_GETARG_TEXT_PP(2);
		max_depth = PG_GETARG_INT32(3);
		min_samples_split = PG_GETARG_INT32(4);

		if (table_name_text == NULL)
			ereport(ERROR, (errmsg("table_name cannot be null")));
		if (feature_col_text == NULL)
			ereport(ERROR, (errmsg("feature_col cannot be null")));
		if (label_col_text == NULL)
			ereport(ERROR, (errmsg("label_col cannot be null")));
		if (max_depth <= 0)
			ereport(ERROR, (errmsg("max_depth must be positive")));
		if (min_samples_split < 2)
			ereport(ERROR, (errmsg("min_samples_split must be at least 2")));

		table_name = text_to_cstring(table_name_text);
		feature_col = text_to_cstring(feature_col_text);
		label_col = text_to_cstring(label_col_text);

		elog(DEBUG1, "Extracting training data from table %s, features=%s, label=%s",
			 table_name, feature_col, label_col);

		/* Data extraction (array of feature vectors, array of labels) */
		extract_training_data(table_name, feature_col, label_col,
							  &X, &y, &n_samples, &dim);

		elog(DEBUG1, "Training data: %d samples, %d features", n_samples, dim);

		indices = (int *) palloc(sizeof(int) * n_samples);
		for (i = 0; i < n_samples; i++)
			indices[i] = i;

		tree = build_tree(X, y, indices, n_samples, dim,
						  max_depth, min_samples_split, true /* classification */);

		initStringInfo(&treedesc);
		serialize_tree(tree, &treedesc, 0);

		elog(INFO, "Decision tree (JSON) = %s", treedesc.data);

		/* Free structures */
		free_tree(tree);
		if (X != NULL) {
			for (i = 0; i < n_samples; i++)
				if (X[i] != NULL)
					pfree(X[i]);
			pfree(X);
		}
		if (y != NULL)
			pfree(y);
		if (indices != NULL)
			pfree(indices);

		PG_RETURN_TEXT_P(cstring_to_text(treedesc.data));
	}
	PG_CATCH();
	{
		if (tree)
			free_tree(tree);
		if (X && n_samples > 0)
		{
			for (i = 0; i < n_samples; i++)
				if (X[i])
					pfree(X[i]);
			pfree(X);
		}
		if (y)
			pfree(y);
		if (indices)
			pfree(indices);
		PG_RE_THROW();
	}
	PG_END_TRY();
}
