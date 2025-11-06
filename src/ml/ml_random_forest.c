/*-------------------------------------------------------------------------
 *
 * ml_random_forest.c
 *    Random Forest implementation for classification and regression
 *
 * Implements Random Forest as an ensemble of decision trees using
 * bootstrap aggregating (bagging) and random feature selection.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_random_forest.c
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
#include <stdlib.h>

/*
 * Random Forest structures and hyperparameters
 */
typedef struct
{
	int		feature_index;	/* Which feature to split */
	double	threshold;		/* Threshold value for split */
	int		left;			/* Index of left child in nodes array (-1 if leaf) */
	int		right;			/* Index of right child in nodes array (-1 if leaf) */
	bool	is_leaf;		/* Is this a leaf node */
	double	value;			/* Class prediction at leaf */
} TreeNode;

typedef struct
{
	TreeNode   *nodes;			/* Array of tree nodes */
	int			num_nodes;		/* Number of nodes allocated */
	int			allocated;		/* Capacity of nodes array */
} DecisionTree;

typedef struct
{
	DecisionTree *trees;		   /* Array of decision trees */
	int			 n_trees;	       /* Number of trees */
	int			 max_depth;       /* Max depth */
	int			 min_samples_split; /* Min samples to split */
	int			 max_features;    /* Features per split */
	int			 n_classes;       /* Number of classes */
	int			 n_features;      /* Features in input */
} RandomForestModel;

/*
 * Bootstrap sample: randomly sample with replacement
 */
static void
rf_bootstrap_sample(int n_samples, int **indices_out)
{
	int i;
	int *indices = (int *) palloc(sizeof(int) * n_samples);
	for (i = 0; i < n_samples; i++)
		indices[i] = (int) (random() % n_samples);
	*indices_out = indices;
}

/*
 * Get random subset of features indices
 */
static void
rf_random_features(int n_features, int max_features, int **features_out)
{
	int *all_features;
	int *features;
	int i, j, temp;
	
	if (max_features <= 0 || max_features > n_features)
		max_features = (int) sqrt((double) n_features);
	
	features = (int *) palloc(sizeof(int) * max_features);
	all_features = (int *) palloc(sizeof(int) * n_features);

	for (i = 0; i < n_features; i++)
		all_features[i] = i;
	
	/* Fisher-Yates shuffle */
	for (i = n_features - 1; i > 0; i--)
	{
		j = (int) (random() % (i + 1));
		temp = all_features[i];
		all_features[i] = all_features[j];
		all_features[j] = temp;
	}
	
	for (i = 0; i < max_features; i++)
		features[i] = all_features[i];
	
	pfree(all_features);
	*features_out = features;
}

/*
 * Count unique classes in labels
 */
static int
rf_count_classes(double *labels, int n)
{
	int	   maxclass = 0;
	int	   i;
	for (i = 0; i < n; i++)
	{
		if ((int)labels[i] > maxclass)
			maxclass = (int)labels[i];
	}
	return maxclass + 1;
}

/*
 * Calculate Gini impurity
 */
static double
rf_gini(double *labels, int *indices, int n_indices, int n_classes)
{
	int		i, c;
	double *class_counts = (double *) palloc0(sizeof(double) * n_classes);
	double	gini = 1.0;
	double	p;

	if (n_indices == 0)
	{
		pfree(class_counts);
		return 0.0;
	}

	for (i = 0; i < n_indices; i++)
	{
		int idx = indices[i];
		int label_idx = (int) labels[idx];
		if (label_idx >= 0 && label_idx < n_classes)
			class_counts[label_idx] += 1.0;
	}
	for (c = 0; c < n_classes; c++)
	{
		p = class_counts[c] / (double) n_indices;
		gini -= p * p;
	}
	pfree(class_counts);
	return gini;
}

/*
 * Find the best split for a node
 */
static bool
rf_find_best_split(float **X, double *y, int *indices, int n_indices, int n_features, int n_classes, int max_features, int *best_feature, double *best_threshold, double *best_gini, int **left_indices_out, int *n_left_out, int **right_indices_out, int *n_right_out)
{
	int	   *features = NULL;
	int		i, f, t;
	double	min_gini = 1.0;
	bool	found_split = false;

	rf_random_features(n_features, max_features, &features);

	for (f = 0; f < max_features; f++)
	{
		int feature_idx = features[f];

		/* Get all possible split values, scan across data for that feature */
		double *feature_values = (double *) palloc(sizeof(double) * n_indices);
		for (i = 0; i < n_indices; i++)
			feature_values[i] = X[indices[i]][feature_idx];

		/* Sort vals for unique thresholds (inefficient O(n^2), but simple) */
		for (i = 0; i < n_indices - 1; i++)
		{
			for (t = i + 1; t < n_indices; t++)
			{
				if (feature_values[i] > feature_values[t])
				{
					double tmp = feature_values[i];
					feature_values[i] = feature_values[t];
					feature_values[t] = tmp;
				}
			}
		}

		for (i = 1; i < n_indices; i++)
		{
			/* candidate threshold between i-1 and i value */
			double threshold = (feature_values[i - 1] + feature_values[i]) / 2.0;
			int   *left = (int *) palloc(sizeof(int) * n_indices);
			int   *right = (int *) palloc(sizeof(int) * n_indices);
			int		n_left = 0, n_right = 0;
			int		j;

			for (j = 0; j < n_indices; j++)
			{
				if (X[indices[j]][feature_idx] <= threshold)
					left[n_left++] = indices[j];
				else
					right[n_right++] = indices[j];
			}
			if (n_left == 0 || n_right == 0)
			{
				pfree(left);
				pfree(right);
				continue;
			}
			double gini_left = rf_gini(y, left, n_left, n_classes);
			double gini_right = rf_gini(y, right, n_right, n_classes);
			double gini = ((double)n_left / n_indices) * gini_left +
						  ((double)n_right / n_indices) * gini_right;

			if (gini < min_gini)
			{
				/* Copy left/right indices for output */
				if (*left_indices_out)
					pfree(*left_indices_out);
				if (*right_indices_out)
					pfree(*right_indices_out);

				*best_feature = feature_idx;
				*best_threshold = threshold;
				*best_gini = gini;
				*left_indices_out = (int *) palloc(sizeof(int) * n_left);
				memcpy(*left_indices_out, left, sizeof(int) * n_left);
				*n_left_out = n_left;
				*right_indices_out = (int *) palloc(sizeof(int) * n_right);
				memcpy(*right_indices_out, right, sizeof(int) * n_right);
				*n_right_out = n_right;
				min_gini = gini;
				found_split = true;
			}
			pfree(left);
			pfree(right);
		}
		pfree(feature_values);
	}
	pfree(features);
	return found_split;
}

/*
 * Create a new tree node
 */
static int
rf_tree_add_node(DecisionTree *tree, int feature, double threshold, bool is_leaf, double value)
{
	if (tree->num_nodes >= tree->allocated)
	{
		int new_alloc = tree->allocated == 0 ? 32 : tree->allocated * 2;
		tree->nodes = (TreeNode *) repalloc(tree->nodes, sizeof(TreeNode) * new_alloc);
		tree->allocated = new_alloc;
	}
	tree->nodes[tree->num_nodes].feature_index = feature;
	tree->nodes[tree->num_nodes].threshold = threshold;
	tree->nodes[tree->num_nodes].is_leaf = is_leaf;
	tree->nodes[tree->num_nodes].value = value;
	tree->nodes[tree->num_nodes].left = -1;
	tree->nodes[tree->num_nodes].right = -1;
	return tree->num_nodes++;
}

/*
 * Recursively build decision tree
 */
static int
rf_build_tree(DecisionTree *tree, float **X, double *y, int *indices,
			  int n_indices, int n_features, int n_classes, int depth, int max_depth, int min_samples_split, int max_features)
{
	int		i;
	int	    left_child = -1, right_child = -1;
	int    *left_indices = NULL, *right_indices = NULL;
	int		n_left = 0, n_right = 0;
	int		best_feature = -1;
	double	best_threshold = 0.0, best_gini = 0.0;
	int		node_id;
	int		n[256];
	double  majority = -1;
	int		max_count = 0;

	/* Count occurrences for majority class */
	memset(n, 0, sizeof(n));
	for (i = 0; i < n_indices; i++)
	{
		int ylab = (int)y[indices[i]];
		n[ylab]++;
	}

	/* Majority class vote */
	for (i = 0; i < n_classes; i++)
	{
		if (n[i] > max_count)
		{
			max_count = n[i];
			majority = (double)i;
		}
	}
	/* Stop if leaf condition */
	if (depth >= max_depth || n_indices < min_samples_split || max_count == n_indices)
	{
		node_id = rf_tree_add_node(tree, -1, 0, true, majority);
		return node_id;
	}

	if (!rf_find_best_split(X, y, indices, n_indices, n_features, n_classes, max_features, &best_feature, &best_threshold, &best_gini, &left_indices, &n_left, &right_indices, &n_right))
	{
		node_id = rf_tree_add_node(tree, -1, 0, true, majority);
		return node_id;
	}
	/* Add new tree node */
	node_id = rf_tree_add_node(tree, best_feature, best_threshold, false, 0.0);
	/* Recursively build left/right subtrees */
	left_child = rf_build_tree(tree, X, y, left_indices, n_left, n_features, n_classes, depth + 1, max_depth, min_samples_split, max_features);
	right_child = rf_build_tree(tree, X, y, right_indices, n_right, n_features, n_classes, depth + 1, max_depth, min_samples_split, max_features);

	tree->nodes[node_id].left = left_child;
	tree->nodes[node_id].right = right_child;

	if (left_indices)
		pfree(left_indices);
	if (right_indices)
		pfree(right_indices);

	return node_id;
}

/*
 * Predict single tree
 */
static double
rf_tree_predict(const DecisionTree *tree, const float *x)
{
	int curr = 0;
	while (!tree->nodes[curr].is_leaf)
	{
		int f = tree->nodes[curr].feature_index;
		double t = tree->nodes[curr].threshold;
		if (x[f] <= t)
			curr = tree->nodes[curr].left;
		else
			curr = tree->nodes[curr].right;
	}
	return tree->nodes[curr].value;
}

/*
 * PG function: train_random_forest_classifier
 */
PG_FUNCTION_INFO_V1(train_random_forest_classifier);

Datum
train_random_forest_classifier(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	int			n_trees;
	int			max_depth;
	int			min_samples_split;
	int			max_features;
	
	char	   *tbl_str;
	char	   *feat_str;
	char	   *label_str;
	StringInfoData query;
	int			ret;
	int			nvec;
	int 		dim = 0;
	float	  **X = NULL;
	double	   *y = NULL;
	int			i;
	MemoryContext oldcontext;
	int			n_classes;
	RandomForestModel *model;
	bytea	   *model_bytea;
	int			tlen;
	int			model_id = 0;
	bool		isnull;
	StringInfoData insert_query;
	Datum		result;
	
	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);
	n_trees = PG_GETARG_INT32(3);
	max_depth = PG_GETARG_INT32(4);
	min_samples_split = PG_NARGS() > 5 ? PG_GETARG_INT32(5) : 2;
	max_features = PG_NARGS() > 6 ? PG_GETARG_INT32(6) : 0;
	
	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	/* Parameter validation */
	if (n_trees < 1)
		ereport(ERROR, (errmsg("n_trees must be at least 1")));
	if (max_depth < 1)
		ereport(ERROR, (errmsg("max_depth must be positive")));
	if (min_samples_split < 2)
		ereport(ERROR, (errmsg("min_samples_split must be at least 2")));
	if (max_features < 0)
		ereport(ERROR, (errmsg("max_features cannot be negative")));

	oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);
	
	/* Load training data using SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("SPI_connect failed: error code %d", ret)));
	
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str, label_str, tbl_str, feat_str, label_str);
	
	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR, (errmsg("Query failed: %s", query.data)));
	
	nvec = SPI_processed;
	
	if (nvec < 2)
		ereport(ERROR, (errmsg("Need at least 2 samples for Random Forest, have %d", nvec)));
	
	X = (float **) palloc(sizeof(float *) * nvec);
	y = (double *) palloc(sizeof(double) * nvec);
	
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple_data = SPI_tuptable->vals[i];
		TupleDesc	tupdesc_data = SPI_tuptable->tupdesc;
		Datum		feat_datum, label_datum;
		bool		feat_null, label_null;
		Vector	   *vec;
		Oid			label_type;
		
		feat_datum = SPI_getbinval(tuple_data, tupdesc_data, 1, &feat_null);
		if (feat_null)
			ereport(ERROR, (errmsg("Null encountered in feature vector")));
		
		vec = DatumGetVector(feat_datum);
		
		if (i == 0)
			dim = vec->dim;
		else if (vec->dim != dim)
			ereport(ERROR, (errmsg("Inconsistent vector dimensions (%d vs %d)", vec->dim, dim)));
		
		X[i] = (float *) palloc(sizeof(float) * dim);
		memcpy(X[i], vec->data, sizeof(float) * dim);
		
		label_datum = SPI_getbinval(tuple_data, tupdesc_data, 2, &label_null);
		if (label_null)
			ereport(ERROR, (errmsg("Null encountered in label")));

		label_type = SPI_gettypeid(tupdesc_data, 2);
		
		if (label_type == INT2OID || label_type == INT4OID || label_type == INT8OID)
			y[i] = (double) DatumGetInt32(label_datum);
		else
			y[i] = DatumGetFloat8(label_datum);
	}

	n_classes = rf_count_classes(y, nvec);
	
	SPI_finish();
	
	/* Train Random Forest */
	model = (RandomForestModel *) palloc0(sizeof(RandomForestModel));
	model->n_trees = n_trees;
	model->max_depth = max_depth;
	model->min_samples_split = min_samples_split;
	model->max_features = max_features > 0 ? max_features : (int)sqrt((double)dim);
	model->n_classes = n_classes;
	model->n_features = dim;
	model->trees = (DecisionTree *) palloc0(sizeof(DecisionTree) * n_trees);

	for (i = 0; i < n_trees; i++)
	{
		int	   *bootstrap_indices = NULL;
		DecisionTree *tree = &model->trees[i];

		rf_bootstrap_sample(nvec, &bootstrap_indices);
		tree->nodes = NULL;
		tree->num_nodes = 0;
		tree->allocated = 0;
		(void) rf_build_tree(tree, X, y, bootstrap_indices, nvec, dim, n_classes, 0, max_depth, min_samples_split, model->max_features);
		pfree(bootstrap_indices);
	}

	/*
	 * Store trained model in neurondb.ml_models table.
	 * For simplicity, we store as a bytea blob.
	 */
	tlen = sizeof(RandomForestModel) + (sizeof(DecisionTree) * n_trees);
	for (i = 0; i < n_trees; i++)
	{
		tlen += sizeof(TreeNode) * model->trees[i].num_nodes;
	}

	model_bytea = (bytea *) palloc(tlen + VARHDRSZ);
	SET_VARSIZE(model_bytea, tlen + VARHDRSZ);
	memcpy(VARDATA(model_bytea), model, sizeof(RandomForestModel));

	/* Insert model into database */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("SPI_connect failed (insert): error code %d", ret)));

	initStringInfo(&insert_query);
	appendStringInfo(&insert_query,
		"INSERT INTO neurondb.ml_models (algorithm, training_table, training_column, "
		"parameters, model_data, status) VALUES ('random_forest', '%s', '%s', "
		"'{\"n_trees\": %d, \"max_depth\": %d}'::jsonb, NULL, 'completed') RETURNING model_id",
		tbl_str, feat_str, n_trees, max_depth);

	ret = SPI_execute(insert_query.data, false, 0);
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
		ereport(ERROR, (errmsg("Failed to insert model")));

	model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	SPI_finish();

	result = Int32GetDatum(model_id);

	MemoryContextSwitchTo(oldcontext);

	if (X)
	{
		for (i = 0; i < nvec; i++)
			if (X[i])
				pfree(X[i]);
		pfree(X);
	}
	if (y)
		pfree(y);

	PG_RETURN_DATUM(result);
}

/*
 * PG function: predict_random_forest
 */
PG_FUNCTION_INFO_V1(predict_random_forest);

Datum
predict_random_forest(PG_FUNCTION_ARGS)
{
	int			model_id;
	Datum		features_datum;
	Vector	   *features;
	float	   *feature_vector;
	char		query[512];
	int			ret;
	bool		isnull;
	RandomForestModel *model = NULL;
	bytea	   *model_bytea;
	double	   *votes;
	int			i, t, n_trees, n_classes;
	double		prediction = 0.0;

	model_id = PG_GETARG_INT32(0);
	features_datum = PG_GETARG_DATUM(1);

	if (model_id < 0)
		ereport(ERROR, (errmsg("model_id must be non-negative")));

	features = DatumGetVector(features_datum);
	if (features == NULL)
		PG_RETURN_NULL();

	feature_vector = features->data;

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("SPI_connect failed: error code %d", ret)));

	snprintf(query, sizeof(query),
			 "SELECT model_data FROM neurondb.ml_models WHERE model_id = %d", model_id);

	ret = SPI_execute(query, true, 0);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Random Forest model %d not found", model_id)));
	}
	model_bytea = (bytea *) DatumGetPointer(SPI_getbinval(SPI_tuptable->vals[0],
									 SPI_tuptable->tupdesc, 1, &isnull));
	model = (RandomForestModel *)(VARDATA(model_bytea));
	n_trees = model->n_trees;
	n_classes = model->n_classes;

	votes = (double *) palloc0(sizeof(double) * n_classes);

	for (t = 0; t < n_trees; t++)
	{
		int class_pred = (int) rf_tree_predict(&model->trees[t], feature_vector);
		if (class_pred >= 0 && class_pred < n_classes)
			votes[class_pred] += 1.0;
	}

	/* Return class with max votes */
	{
		double maxv = votes[0];
		int maxi = 0;
		for (i = 1; i < n_classes; i++)
		{
			if (votes[i] > maxv)
			{
				maxv = votes[i];
				maxi = i;
			}
		}
		prediction = (double)maxi;
	}

	pfree(votes);
	SPI_finish();
	PG_RETURN_FLOAT8(prediction);
}

/*
 * PG function: evaluate_random_forest
 */
PG_FUNCTION_INFO_V1(evaluate_random_forest);

Datum
evaluate_random_forest(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	int			model_id;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *label_str;
	StringInfoData query;
	int			ret;
	int			nvec, i;
	Vector	   *vec;
	Datum	   *result_datums;
	ArrayType  *result_array;
	float	  **X;
	double	   *y;
	double		acc = 0.0, prec = 0.0, recall = 0.0, f1 = 0.0;
	int			tp = 0, tn = 0, fp = 0, fn = 0;

	model_id = PG_GETARG_INT32(3);
	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);

	if (model_id < 0)
		ereport(ERROR, (errmsg("model_id must be non-negative")));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	/* Load test data */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("SPI_connect failed: error code %d", ret)));

	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		feat_str, label_str, tbl_str, feat_str, label_str);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR, (errmsg("evaluate: SPI_execute failed: %s", query.data)));

	nvec = SPI_processed;
	X = (float **) palloc(sizeof(float *) * nvec);
	y = (double *) palloc(sizeof(double) * nvec);

	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple_data = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum, label_datum;
		bool		feat_null, label_null;
		Oid			label_type;

		feat_datum = SPI_getbinval(tuple_data, tupdesc, 1, &feat_null);
		if (feat_null)
			ereport(ERROR, (errmsg("Null encountered in feature vector")));

		vec = DatumGetVector(feat_datum);

		X[i] = (float *) palloc(sizeof(float) * vec->dim);
		memcpy(X[i], vec->data, sizeof(float) * vec->dim);

		label_datum = SPI_getbinval(tuple_data, tupdesc, 2, &label_null);
		if (label_null)
			ereport(ERROR, (errmsg("Null encountered in label")));

		label_type = SPI_gettypeid(tupdesc, 2);

		if (label_type == INT2OID || label_type == INT4OID || label_type == INT8OID)
			y[i] = (double) DatumGetInt32(label_datum);
		else
			y[i] = DatumGetFloat8(label_datum);
	}
	SPI_finish();

	/* Predict and compute metrics */
	for (i = 0; i < nvec; i++)
	{
		Datum		 args[2];
		float		*x = X[i];
		double		pred = 0.0, truth = y[i];

		args[0] = Int32GetDatum(model_id);

		{
			/* Build a Vector datum from float array */
			Vector *vf = (Vector*) palloc(sizeof(Vector) + sizeof(float) * ((vec->dim > 0 ? vec->dim : 1) - 1));
			vf->dim = vec->dim;
			memcpy(vf->data, x, sizeof(float) * vf->dim);
			args[1] = PointerGetDatum(vf);

			pred = DatumGetFloat8(DirectFunctionCall2(predict_random_forest, args[0], args[1]));

			pfree(vf);
		}
		/* Update confusion matrix for binary classification (0/1) */
		if (truth == 1.0 && pred == 1.0)
			tp++;
		else if (truth == 0.0 && pred == 0.0)
			tn++;
		else if (truth == 0.0 && pred == 1.0)
			fp++;
		else if (truth == 1.0 && pred == 0.0)
			fn++;
	}

	if ((tp + tn + fp + fn) > 0)
		acc = (double)(tp + tn) / (tp + tn + fp + fn);
	if ((tp + fp) > 0)
		prec = (double)tp / (tp + fp);
	if ((tp + fn) > 0)
		recall = (double)tp / (tp + fn);
	if ((prec + recall) > 0.0)
		f1 = 2.0 * prec * recall / (prec + recall);

	result_datums = (Datum *) palloc(sizeof(Datum) * 4);

	result_datums[0] = Float8GetDatum(acc);
	result_datums[1] = Float8GetDatum(prec);
	result_datums[2] = Float8GetDatum(recall);
	result_datums[3] = Float8GetDatum(f1);
	
	result_array = construct_array(result_datums, 4, FLOAT8OID, 8, FLOAT8PASSBYVAL, 'd');
	
	for (i = 0; i < nvec; i++)
		if (X[i])
			pfree(X[i]);
	if (X)
		pfree(X);
	if (y)
		pfree(y);
	if (result_datums)
		pfree(result_datums);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

