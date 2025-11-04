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

/*
 * Compute Gini impurity for classification
 */
static double
compute_gini(double *labels, int n)
{
	double class_counts[2] = {0.0, 0.0};
	double gini = 0.0;
	int i;
	
	/* Count classes */
	for (i = 0; i < n; i++)
	{
		int class = (int) labels[i];
		if (class >= 0 && class < 2)
			class_counts[class] += 1.0;
	}
	
	/* Compute Gini = 1 - sum(p_i^2) */
	gini = 1.0;
	for (i = 0; i < 2; i++)
	{
		double p = class_counts[i] / n;
		gini -= p * p;
	}
	
	return gini;
}

/*
 * Compute variance for regression
 */
static double
compute_variance(double *values, int n)
{
	double mean = 0.0;
	double variance = 0.0;
	int i;
	
	/* Compute mean */
	for (i = 0; i < n; i++)
		mean += values[i];
	mean /= n;
	
	/* Compute variance */
	for (i = 0; i < n; i++)
	{
		double diff = values[i] - mean;
		variance += diff * diff;
	}
	
	return variance / n;
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
	
	/* Try each feature */
	for (feat = 0; feat < dim; feat++)
	{
		float min_val = FLT_MAX;
		float max_val = -FLT_MAX;
		int i;
		
		/* Find range */
		for (i = 0; i < n_samples; i++)
		{
			float val = X[indices[i]][feat];
			if (val < min_val)
				min_val = val;
			if (val > max_val)
				max_val = val;
		}
		
		/* Try 10 threshold candidates */
		for (i = 0; i < 10; i++)
		{
			float threshold = min_val + (max_val - min_val) * i / 10.0;
			int left_count = 0;
			int right_count = 0;
			double *left_y;
			double *right_y;
			double left_impurity, right_impurity, gain;
			int j;
			
			/* Count left/right */
			for (j = 0; j < n_samples; j++)
			{
				if (X[indices[j]][feat] <= threshold)
					left_count++;
				else
					right_count++;
			}
			
			if (left_count == 0 || right_count == 0)
				continue;
			
			/* Allocate temporary arrays */
			left_y = (double *) palloc(sizeof(double) * left_count);
			right_y = (double *) palloc(sizeof(double) * right_count);
			
			/* Split samples */
			left_count = 0;
			right_count = 0;
			for (j = 0; j < n_samples; j++)
			{
				if (X[indices[j]][feat] <= threshold)
					left_y[left_count++] = y[indices[j]];
				else
					right_y[right_count++] = y[indices[j]];
			}
			
			/* Compute impurity reduction */
			if (is_classification)
			{
				left_impurity = compute_gini(left_y, left_count);
				right_impurity = compute_gini(right_y, right_count);
			}
			else
			{
				left_impurity = compute_variance(left_y, left_count);
				right_impurity = compute_variance(right_y, right_count);
			}
			
			gain = -(left_count * left_impurity + right_count * right_impurity) / n_samples;
			
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
	int			best_feature;
	float		best_threshold;
	double		best_gain;
	int		   *left_indices;
	int		   *right_indices;
	int			left_count = 0;
	int			right_count = 0;
	int			i;
	
	node = (TreeNode *) palloc0(sizeof(TreeNode));
	
	/* Check stopping criteria */
	if (max_depth == 0 || n_samples < min_samples_split)
	{
		node->is_leaf = true;
		
		if (is_classification)
		{
			/* Majority class */
			int class_counts[2] = {0, 0};
			for (i = 0; i < n_samples; i++)
			{
				int class = (int) y[indices[i]];
				if (class >= 0 && class < 2)
					class_counts[class]++;
			}
			node->leaf_value = (class_counts[1] > class_counts[0]) ? 1.0 : 0.0;
		}
		else
		{
			/* Mean value */
			double sum = 0.0;
			for (i = 0; i < n_samples; i++)
				sum += y[indices[i]];
			node->leaf_value = sum / n_samples;
		}
		
		return node;
	}
	
	/* Find best split */
	find_best_split(X, y, indices, n_samples, dim, &best_feature, &best_threshold, &best_gain, is_classification);
	
	/* If no good split found, make leaf */
	if (best_feature == -1)
	{
		node->is_leaf = true;
		
		if (is_classification)
		{
			int class_counts[2] = {0, 0};
			for (i = 0; i < n_samples; i++)
			{
				int class = (int) y[indices[i]];
				if (class >= 0 && class < 2)
					class_counts[class]++;
			}
			node->leaf_value = (class_counts[1] > class_counts[0]) ? 1.0 : 0.0;
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
	
	/* Create internal node */
	node->is_leaf = false;
	node->feature_idx = best_feature;
	node->threshold = best_threshold;
	
	/* Split samples */
	left_indices = (int *) palloc(sizeof(int) * n_samples);
	right_indices = (int *) palloc(sizeof(int) * n_samples);
	
	for (i = 0; i < n_samples; i++)
	{
		if (X[indices[i]][best_feature] <= best_threshold)
			left_indices[left_count++] = indices[i];
		else
			right_indices[right_count++] = indices[i];
	}
	
	/* Recursively build subtrees */
	node->left = build_tree(X, y, left_indices, left_count, dim, max_depth - 1, min_samples_split, is_classification);
	node->right = build_tree(X, y, right_indices, right_count, dim, max_depth - 1, min_samples_split, is_classification);
	
	pfree(left_indices);
	pfree(right_indices);
	
	return node;
}

/*
 * Predict using decision tree
 */
static double
tree_predict(TreeNode *node, float *x)
{
	if (node->is_leaf)
		return node->leaf_value;
	
	if (x[node->feature_idx] <= node->threshold)
		return tree_predict(node->left, x);
	else
		return tree_predict(node->right, x);
}

/*
 * train_decision_tree
 *
 * Trains a decision tree for classification or regression
 * Returns serialized tree structure (simplified for now - just returns max_depth used)
 */
PG_FUNCTION_INFO_V1(train_decision_tree_classifier);

Datum
train_decision_tree_classifier(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	int			max_depth = PG_GETARG_INT32(3);
	int			min_samples_split = PG_GETARG_INT32(4);
	
	/* For now, return a simple status */
	/* Full implementation would serialize the tree structure */
	ereport(NOTICE,
			(errmsg("Decision tree training: max_depth=%d, min_samples_split=%d", max_depth, min_samples_split)));
	
	PG_RETURN_INT32(max_depth);
}

