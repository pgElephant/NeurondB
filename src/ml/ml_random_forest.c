/*
 * ml_random_forest.c
 *    Refactored Random Forest implementation for classification (PostgreSQL C
 * coding standard)
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 */

#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "utils/jsonb.h"
#include "common/pg_prng.h"
#include "libpq/pqformat.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "neurondb.h"
#include "neurondb_ml.h"
#include "gtree.h"

#define RF_STUB_MAX_FEATURES 4
#define RF_BOOTSTRAP_FRACTION 0.8
#define RF_DEFAULT_TREES 3
#define RF_MAX_DEPTH 4
#define RF_MIN_SAMPLES 5

PG_FUNCTION_INFO_V1(train_random_forest_classifier);
PG_FUNCTION_INFO_V1(predict_random_forest);
PG_FUNCTION_INFO_V1(evaluate_random_forest);

typedef struct RFSplitPair {
	double value;
	int cls;
} RFSplitPair;

typedef struct RFModel RFModel;

static bool rf_select_split(const float *features, const double *labels,
	const int *indices, int count, int feature_dim, int n_classes,
	pg_prng_state *rng, int *feature_order, int *best_feature,
	double *best_threshold, double *best_impurity);
static int rf_build_branch_tree(GTree *tree, const float *features,
	const double *labels, const double *feature_vars, int feature_dim,
	int n_classes, const int *indices, int count, int depth, int max_depth,
	int min_samples, pg_prng_state *rng, int *feature_order,
	double *feature_importance, double *max_split_deviation);
static double rf_tree_predict_row(const GTree *tree, const float *row, int dim);
static void rf_serialize_tree(StringInfo buf, const GTree *tree);
static GTree *rf_deserialize_tree(StringInfo buf);
static bytea *rf_model_serialize(const RFModel *model);
static RFModel *rf_model_deserialize(const bytea *data);
static void rf_free_deserialized_model(RFModel *model);
static bool rf_load_model_from_catalog(int32 model_id, RFModel **out);
static void rf_save_model_catalog(
	int32 catalog_id, const RFModel *model, const char *hyperparams_json);
static void rf_promote_model_internal(
	int32 internal_id, int32 catalog_id, const char *hyperparams_json);
static bool rf_update_model_identifier(
	int32 internal_id, int32 new_id, RFModel **out_model);

static int rf_split_pair_cmp(const void *a, const void *b) {
	const RFSplitPair *pa = (const RFSplitPair *)a;
	const RFSplitPair *pb = (const RFSplitPair *)b;

	if (pa->value < pb->value)
		return -1;
	if (pa->value > pb->value)
		return 1;
	if (pa->cls < pb->cls)
		return -1;
	if (pa->cls > pb->cls)
		return 1;
	return 0;
}

static double rf_gini_from_counts(const int *counts, int n_classes, int total) {
	double sum_sq = 0.0;
	int i;

	if (total <= 0)
		return 0.0;

	for (i = 0; i < n_classes; i++) {
		if (counts[i] > 0) {
			double p = (double)counts[i] / (double)total;
			sum_sq += p * p;
		}
	}
	return 1.0 - sum_sq;
}

static bool rf_select_split(const float *features, const double *labels,
	const int *indices, int count, int feature_dim, int n_classes,
	pg_prng_state *rng, int *feature_order, int *best_feature,
	double *best_threshold, double *best_impurity) {
	int mtry;
	int candidates;
	int f;

	if (count <= 1 || feature_dim <= 0 || n_classes <= 0)
		return false;

	if (best_feature != NULL)
		*best_feature = -1;
	if (best_threshold != NULL)
		*best_threshold = 0.0;
	if (best_impurity != NULL)
		*best_impurity = DBL_MAX;

	if (feature_order != NULL) {
		for (f = 0; f < feature_dim; f++)
			feature_order[f] = f;
	}

	mtry = (int)sqrt((double)feature_dim);
	if (mtry < 1)
		mtry = 1;
	if (mtry > feature_dim)
		mtry = feature_dim;
	candidates = mtry;

	if (feature_order != NULL) {
		for (f = 0; f < candidates; f++) {
			int swap_idx;

			swap_idx = (int)pg_prng_uint64_range(
				rng, (uint64)f, (uint64)(feature_dim - 1));
			if (swap_idx != f) {
				int tmp = feature_order[f];

				feature_order[f] = feature_order[swap_idx];
				feature_order[swap_idx] = tmp;
			}
		}
	}

	if (feature_order == NULL)
		candidates = Min(candidates, feature_dim);

	for (f = 0; f < candidates; f++) {
		int feature_idx
			= (feature_order != NULL) ? feature_order[f] : f;
		RFSplitPair *pairs;
		int *left_counts_tmp;
		int *right_counts_tmp;
		int pair_count = 0;
		int i;
		int left_total = 0;
		int right_total = 0;

		if (feature_idx < 0 || feature_idx >= feature_dim)
			continue;

		pairs = (RFSplitPair *)palloc(sizeof(RFSplitPair) * count);

		for (i = 0; i < count; i++) {
			int idx = indices[i];
			float value;
			int cls;

			if (idx < 0)
				continue;
			cls = (int)rint(labels[idx]);
			if (cls < 0 || cls >= n_classes)
				continue;
			value = features[idx * feature_dim + feature_idx];
			if (!isfinite(value))
				continue;

			pairs[pair_count].value = (double)value;
			pairs[pair_count].cls = cls;
			pair_count++;
		}

		if (pair_count <= 1) {
			pfree(pairs);
			continue;
		}

		qsort(pairs, pair_count, sizeof(RFSplitPair),
			rf_split_pair_cmp);

		left_counts_tmp = (int *)palloc0(sizeof(int) * n_classes);
		right_counts_tmp = (int *)palloc0(sizeof(int) * n_classes);

		for (i = 0; i < pair_count; i++)
			right_counts_tmp[pairs[i].cls]++;

		right_total = pair_count;

		for (i = 0; i < pair_count - 1; i++) {
			int cls = pairs[i].cls;
			double left_imp;
			double right_imp;
			double weighted;
			double threshold_candidate;

			left_counts_tmp[cls]++;
			right_counts_tmp[cls]--;
			left_total++;
			right_total--;

			if (pairs[i].value == pairs[i + 1].value)
				continue;
			if (left_total <= 0 || right_total <= 0)
				continue;

			left_imp = rf_gini_from_counts(
				left_counts_tmp, n_classes, left_total);
			right_imp = rf_gini_from_counts(
				right_counts_tmp, n_classes, right_total);
			threshold_candidate
				= 0.5 * (pairs[i].value + pairs[i + 1].value);
			weighted = ((double)left_total / (double)pair_count)
					* left_imp
				+ ((double)right_total / (double)pair_count)
					* right_imp;

			if (weighted < *best_impurity) {
				*best_impurity = weighted;
				*best_threshold = threshold_candidate;
				*best_feature = feature_idx;
			}
		}

		pfree(left_counts_tmp);
		pfree(right_counts_tmp);
		pfree(pairs);
	}

	return (*best_feature >= 0);
}

static int rf_build_branch_tree(GTree *tree, const float *features,
	const double *labels, const double *feature_vars, int feature_dim,
	int n_classes, const int *indices, int count, int depth, int max_depth,
	int min_samples, pg_prng_state *rng, int *feature_order,
	double *feature_importance, double *max_split_deviation) {
	int *class_counts;
	int majority_idx = -1;
	int i;
	double best_impurity = DBL_MAX;
	int split_feature = -1;
	double split_threshold = 0.0;
	double gini;
	int node_idx;

	if (tree == NULL || features == NULL || labels == NULL
		|| indices == NULL || count <= 0)
		return gtree_add_leaf(tree, 0.0);

	class_counts = (int *)palloc0(sizeof(int) * n_classes);

	for (i = 0; i < count; i++) {
		int idx = indices[i];
		int cls;

		if (idx < 0)
			continue;
		if (!isfinite(labels[idx]))
			continue;
		cls = (int)rint(labels[idx]);
		if (cls < 0 || cls >= n_classes)
			continue;
		class_counts[cls]++;
		if (majority_idx < 0
			|| class_counts[cls] > class_counts[majority_idx])
			majority_idx = cls;
	}

	if (majority_idx < 0) {
		pfree(class_counts);
		return gtree_add_leaf(tree, 0.0);
	}

	gini = rf_gini_from_counts(class_counts, n_classes, count);

	if (gini <= 0.0 || depth >= max_depth || count <= min_samples) {
		double value = (double)majority_idx;

		pfree(class_counts);
		return gtree_add_leaf(tree, value);
	}

	if (!rf_select_split(features, labels, indices, count, feature_dim,
		    n_classes, rng, feature_order, &split_feature,
		    &split_threshold, &best_impurity)) {
		double value = (double)majority_idx;

		pfree(class_counts);
		return gtree_add_leaf(tree, value);
	}

	if (split_feature < 0) {
		double value = (double)majority_idx;

		pfree(class_counts);
		return gtree_add_leaf(tree, value);
	}

	{
		int *left_indices;
		int *right_indices;
		int left_count = 0;
		int right_count = 0;

		left_indices = (int *)palloc(sizeof(int) * count);
		right_indices = (int *)palloc(sizeof(int) * count);

		for (i = 0; i < count; i++) {
			int idx = indices[i];
			float value;

			if (idx < 0)
				continue;
			value = features[idx * feature_dim + split_feature];
			if (!isfinite(value))
				continue;
			if ((double)value <= split_threshold)
				left_indices[left_count++] = idx;
			else
				right_indices[right_count++] = idx;
		}

		if (left_count == 0 || right_count == 0) {
			double value = (double)majority_idx;

			pfree(left_indices);
			pfree(right_indices);
			pfree(class_counts);
			return gtree_add_leaf(tree, value);
		} else {
			int left_child;
			int right_child;

			if (feature_importance != NULL && gini > 0.0
				&& best_impurity < DBL_MAX) {
				double improvement = gini - best_impurity;

				if (improvement < 0.0)
					improvement = 0.0;
				feature_importance[split_feature]
					+= improvement;
			}

			if (feature_vars != NULL && split_feature < feature_dim
				&& feature_vars[split_feature] > 0.0
				&& max_split_deviation != NULL) {
				double split_dev = fabs(split_threshold)
					/ sqrt(feature_vars[split_feature]);

				if (split_dev > *max_split_deviation)
					*max_split_deviation = split_dev;
			}

			node_idx = gtree_add_split(
				tree, split_feature, split_threshold);

			left_child = rf_build_branch_tree(tree, features,
				labels, feature_vars, feature_dim, n_classes,
				left_indices, left_count, depth + 1, max_depth,
				min_samples, rng, feature_order,
				feature_importance, max_split_deviation);
			right_child = rf_build_branch_tree(tree, features,
				labels, feature_vars, feature_dim, n_classes,
				right_indices, right_count, depth + 1,
				max_depth, min_samples, rng, feature_order,
				feature_importance, max_split_deviation);

			gtree_set_child(tree, node_idx, left_child, true);
			gtree_set_child(tree, node_idx, right_child, false);

			pfree(left_indices);
			pfree(right_indices);
			pfree(class_counts);
			return node_idx;
		}
	}
}

static double rf_tree_predict_row(
	const GTree *tree, const float *row, int dim) {
	const GTreeNode *nodes;
	int idx;

	if (tree == NULL || row == NULL)
		return 0.0;
	if (tree->root < 0 || tree->count <= 0)
		return 0.0;

	nodes = gtree_nodes(tree);
	idx = tree->root;

	while (idx >= 0 && idx < tree->count) {
		const GTreeNode *node = &nodes[idx];

		if (node->is_leaf)
			return node->value;

		if (node->feature_idx < 0 || node->feature_idx >= dim)
			return 0.0;

		if ((double)row[node->feature_idx] <= node->threshold)
			idx = node->left;
		else
			idx = node->right;
	}

	return 0.0;
}

typedef struct RFModel {
	int32 model_id; /* model identifier */
	int n_features; /* number of features */
	int n_samples; /* number of samples */
	int n_classes; /* number of classes */
	double majority_value; /* most frequent class value */
	double majority_fraction; /* fraction of majority class */
	double gini_impurity; /* Gini impurity for root */
	int *class_counts; /* class histogram */
	double *feature_means; /* mean for each feature */
	double *feature_variances; /* variance for each feature */
	GTree *tree; /* tree structure pointer */
	int split_feature; /* feature index for split */
	double split_threshold; /* threshold used for split */
	double second_value; /* value for 2nd class by frequency */
	double second_fraction; /* fraction for 2nd class by frequency */
	double *feature_importance; /* global feature importance scores */
	double label_entropy; /* entropy of labels at root */
	double max_deviation; /* max deviation among features */
	double max_split_deviation; /* max deviation at split */
	double left_branch_value; /* predicted value for left branch */
	double left_branch_fraction; /* fraction for left branch */
	double right_branch_value; /* predicted value for right branch */
	double right_branch_fraction; /* fraction for right branch */
	int feature_limit; /* number of per-branch means stored */
	double *left_branch_means; /* per-feature left branch means */
	double *right_branch_means; /* per-feature right branch means */
	int tree_count; /* number of trained trees */
	GTree **trees; /* array of tree pointers */
	double *tree_majority; /* per-tree majority class */
	double *tree_majority_fraction; /* per-tree majority fraction */
	double *tree_second; /* per-tree second class */
	double *tree_second_fraction; /* per-tree second fraction */
	double *tree_oob_accuracy; /* per-tree out-of-bag accuracy */
	double oob_accuracy; /* forest-level OOB accuracy */
} RFModel;

static RFModel *rf_models = NULL;
static int rf_model_count = 0;
static int32 rf_next_model_id = 1;

static void rf_store_model(int32 model_id, int n_features, int n_samples,
	int n_classes, double majority, double fraction, double gini,
	double entropy, const int *class_counts, const double *feature_means,
	const double *feature_variances, const double *feature_importance,
	GTree *tree, int split_feature, double split_threshold,
	double second_value, double second_fraction, double left_value,
	double left_fraction, double right_value, double right_fraction,
	double max_deviation, double max_split_deviation, int feature_limit,
	const double *left_means, const double *right_means, int tree_count,
	GTree *const *trees, const double *tree_majority,
	const double *tree_majority_fraction, const double *tree_second,
	const double *tree_second_fraction, const double *tree_oob_accuracy,
	double oob_accuracy) {
	MemoryContext oldctx = MemoryContextSwitchTo(TopMemoryContext);
	int i;

	if (rf_model_count == 0)
		rf_models = (RFModel *)palloc(sizeof(RFModel));
	else
		rf_models = (RFModel *)repalloc(
			rf_models, sizeof(RFModel) * (rf_model_count + 1));

	rf_models[rf_model_count].model_id = model_id;
	rf_models[rf_model_count].n_features = n_features;
	rf_models[rf_model_count].n_samples = n_samples;
	rf_models[rf_model_count].n_classes = n_classes;
	rf_models[rf_model_count].majority_value = majority;
	rf_models[rf_model_count].majority_fraction = fraction;
	rf_models[rf_model_count].gini_impurity = gini;
	rf_models[rf_model_count].label_entropy = entropy;

	rf_models[rf_model_count].class_counts = NULL;
	if (n_classes > 0 && class_counts != NULL) {
		int *copy = (int *)palloc(sizeof(int) * n_classes);

		memcpy(copy, class_counts, sizeof(int) * n_classes);
		rf_models[rf_model_count].class_counts = copy;
	}

	rf_models[rf_model_count].feature_means = NULL;
	if (n_features > 0 && feature_means != NULL) {
		double *means_copy
			= (double *)palloc(sizeof(double) * n_features);

		for (i = 0; i < n_features; i++)
			means_copy[i] = feature_means[i];
		rf_models[rf_model_count].feature_means = means_copy;
	}

	rf_models[rf_model_count].feature_variances = NULL;
	if (n_features > 0 && feature_variances != NULL) {
		double *vars_copy
			= (double *)palloc(sizeof(double) * n_features);

		for (i = 0; i < n_features; i++)
			vars_copy[i] = feature_variances[i];
		rf_models[rf_model_count].feature_variances = vars_copy;
	}

	rf_models[rf_model_count].feature_importance = NULL;
	if (n_features > 0 && feature_importance != NULL) {
		double *importance_copy
			= (double *)palloc(sizeof(double) * n_features);

		for (i = 0; i < n_features; i++)
			importance_copy[i] = feature_importance[i];
		rf_models[rf_model_count].feature_importance = importance_copy;
	}

	rf_models[rf_model_count].tree = tree;
	rf_models[rf_model_count].split_feature = split_feature;
	rf_models[rf_model_count].split_threshold = split_threshold;
	rf_models[rf_model_count].second_value = second_value;
	rf_models[rf_model_count].second_fraction = second_fraction;
	rf_models[rf_model_count].oob_accuracy = oob_accuracy;
	rf_models[rf_model_count].left_branch_value = left_value;
	rf_models[rf_model_count].left_branch_fraction = left_fraction;
	rf_models[rf_model_count].right_branch_value = right_value;
	rf_models[rf_model_count].right_branch_fraction = right_fraction;
	rf_models[rf_model_count].max_deviation = max_deviation;
	rf_models[rf_model_count].max_split_deviation = max_split_deviation;
	rf_models[rf_model_count].feature_limit
		= (feature_limit > 0) ? feature_limit : 0;
	rf_models[rf_model_count].left_branch_means = NULL;
	rf_models[rf_model_count].right_branch_means = NULL;
	rf_models[rf_model_count].tree_count = 0;
	rf_models[rf_model_count].trees = NULL;
	rf_models[rf_model_count].tree_majority = NULL;
	rf_models[rf_model_count].tree_majority_fraction = NULL;
	rf_models[rf_model_count].tree_second = NULL;
	rf_models[rf_model_count].tree_second_fraction = NULL;
	rf_models[rf_model_count].tree_oob_accuracy = NULL;

	if (rf_models[rf_model_count].feature_limit > 0 && left_means != NULL) {
		double *copy = (double *)palloc(sizeof(double)
			* rf_models[rf_model_count].feature_limit);

		memcpy(copy, left_means,
			sizeof(double)
				* rf_models[rf_model_count].feature_limit);
		rf_models[rf_model_count].left_branch_means = copy;
	}

	if (rf_models[rf_model_count].feature_limit > 0
		&& right_means != NULL) {
		double *copy = (double *)palloc(sizeof(double)
			* rf_models[rf_model_count].feature_limit);

		memcpy(copy, right_means,
			sizeof(double)
				* rf_models[rf_model_count].feature_limit);
		rf_models[rf_model_count].right_branch_means = copy;
	}

	if (tree_count > 0 && trees != NULL) {
		GTree **tree_copy;

		tree_copy = (GTree **)palloc(sizeof(GTree *) * tree_count);
		for (i = 0; i < tree_count; i++)
			tree_copy[i] = trees[i];
		rf_models[rf_model_count].trees = tree_copy;
		rf_models[rf_model_count].tree_count = tree_count;

		if (tree_majority != NULL) {
			double *majority_copy
				= (double *)palloc(sizeof(double) * tree_count);

			for (i = 0; i < tree_count; i++)
				majority_copy[i] = tree_majority[i];
			rf_models[rf_model_count].tree_majority = majority_copy;
		}

		if (tree_majority_fraction != NULL) {
			double *fraction_copy
				= (double *)palloc(sizeof(double) * tree_count);

			for (i = 0; i < tree_count; i++)
				fraction_copy[i] = tree_majority_fraction[i];
			rf_models[rf_model_count].tree_majority_fraction
				= fraction_copy;
		}

		if (tree_second != NULL) {
			double *second_copy
				= (double *)palloc(sizeof(double) * tree_count);

			for (i = 0; i < tree_count; i++)
				second_copy[i] = tree_second[i];
			rf_models[rf_model_count].tree_second = second_copy;
		}

		if (tree_second_fraction != NULL) {
			double *second_fraction_copy
				= (double *)palloc(sizeof(double) * tree_count);

			for (i = 0; i < tree_count; i++)
				second_fraction_copy[i]
					= tree_second_fraction[i];
			rf_models[rf_model_count].tree_second_fraction
				= second_fraction_copy;
		}

		if (tree_oob_accuracy != NULL) {
			double *oob_copy
				= (double *)palloc(sizeof(double) * tree_count);

			for (i = 0; i < tree_count; i++)
				oob_copy[i] = tree_oob_accuracy[i];
			rf_models[rf_model_count].tree_oob_accuracy = oob_copy;
		}
	}

	rf_model_count++;

	MemoryContextSwitchTo(oldctx);
}

static bool rf_lookup_model(int32 model_id, RFModel **out) {
	int i;

	for (i = 0; i < rf_model_count; i++) {
		if (rf_models[i].model_id == model_id) {
			if (out)
				*out = &rf_models[i];
			return true;
		}
	}
	return false;
}

static int rf_count_classes(double *labels, int n_samples) {
	int max_class = -1;
	int i;

	if (n_samples <= 0)
		return 0;

	for (i = 0; i < n_samples; i++) {
		double val = labels[i];
		int as_int;

		if (!isfinite(val))
			continue;

		as_int = (int)rint(val);
		if (as_int < 0)
			continue;

		if (as_int > max_class)
			max_class = as_int;
	}

	return (max_class < 0) ? 0 : (max_class + 1);
}

Datum train_random_forest_classifier(PG_FUNCTION_ARGS) {
	text *table_name_text;
	text *feature_col_text;
	text *label_col_text;

	char *table_name;
	char *feature_col;
	char *label_col;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_label;

	StringInfoData query;

	int feature_dim = 0;
	int n_classes = 0;
	int majority_count = 0;
	int second_count = 0;
	int second_idx = -1;
	int *class_counts_tmp = NULL;
	int feature_sum_count = 0;
	int split_feature = -1;
	int *left_counts = NULL;
	int *right_counts = NULL;
	int left_majority_idx = -1;
	int right_majority_idx = -1;
	int left_total = 0;
	int right_total = 0;
	int majority_idx = -1;
	int feature_limit = 0;
	int best_feature = -1;
	int *left_feature_counts_vec = NULL;
	int *right_feature_counts_vec = NULL;
	int n_samples = 0;
	int split_pair_count = 0;
	int sample_count = 0;
	int *bootstrap_indices = NULL;
	int *feature_order = NULL;
	pg_prng_state rng;
	int32 model_id;

	double *labels = NULL;
	double majority_value = 0.0;
	double majority_fraction = 0.0;
	double gini_impurity = 0.0;
	double label_entropy = 0.0;
	double second_value = 0.0;
	double *feature_means_tmp = NULL;
	double *feature_vars_tmp = NULL;
	double *feature_importance_tmp = NULL;
	double *feature_sums = NULL;
	double *feature_sums_sq = NULL;
	double *class_feature_sums = NULL;
	int *class_feature_counts = NULL;
	double *left_feature_sums_vec = NULL;
	double *right_feature_sums_vec = NULL;
	double left_leaf_value = 0.0;
	double right_leaf_value = 0.0;
	double left_sum = 0.0;
	double right_sum = 0.0;
	double left_branch_fraction = 0.0;
	double right_branch_fraction = 0.0;
	double class_majority_mean = 0.0;
	double class_second_mean = 0.0;
	double class_mean_threshold = 0.0;
	double best_majority_mean = 0.0;
	double best_second_mean = 0.0;
	double best_score = -1.0;
	double max_deviation = 0.0;
	double max_split_deviation = 0.0;
	double split_threshold = 0.0;
	double second_fraction = 0.0;
	double *left_branch_means_vec = NULL;
	double *right_branch_means_vec = NULL;
	double forest_oob_accuracy = 0.0;
	double *tree_oob_accuracy = NULL;
	int oob_total_all = 0;
	int oob_correct_all = 0;
	float *stage_features = NULL;
	GTree **trees = NULL;
	double *tree_majorities = NULL;
	double *tree_majority_fractions = NULL;
	double *tree_seconds = NULL;
	double *tree_second_fractions = NULL;
	int tree_count = 0;
	int forest_trees_arg = RF_DEFAULT_TREES;
	int max_depth_arg = RF_MAX_DEPTH;
	int min_samples_arg = RF_MIN_SAMPLES;
	double best_split_impurity = DBL_MAX;
	double best_split_threshold = 0.0;

	bool branch_threshold_valid = false;
	bool class_mean_threshold_valid = false;
	bool best_score_valid = false;
	bool best_split_valid = false;

	GTree *primary_tree = NULL;
	RFSplitPair *split_pairs = NULL;

	if (PG_NARGS() < 3)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("random_forest: requires table, feature "
				       "column, and label column")));

	table_name_text = PG_GETARG_TEXT_PP(0);
	feature_col_text = PG_GETARG_TEXT_PP(1);
	label_col_text = PG_GETARG_TEXT_PP(2);

	if (PG_NARGS() > 3 && !PG_ARGISNULL(3)) {
		int32 arg_trees = PG_GETARG_INT32(3);

		if (arg_trees < 1)
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("random_forest: number of trees "
					       "must "
					       "be at least 1")));
		if (arg_trees > 1024)
			arg_trees = 1024;
		forest_trees_arg = arg_trees;
	}

	if (PG_NARGS() > 4 && !PG_ARGISNULL(4)) {
		int32 arg_depth = PG_GETARG_INT32(4);

		if (arg_depth < 1)
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("random_forest: max depth must "
					       "be at "
					       "least 1")));
		if (arg_depth > GTREE_MAX_DEPTH)
			arg_depth = GTREE_MAX_DEPTH;
		max_depth_arg = arg_depth;
	}

	if (PG_NARGS() > 5 && !PG_ARGISNULL(5)) {
		int32 arg_min_samples = PG_GETARG_INT32(5);

		if (arg_min_samples < 1)
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("random_forest: min samples "
					       "must be at "
					       "least 1")));
		if (arg_min_samples > 1000000)
			arg_min_samples = 1000000;
		min_samples_arg = arg_min_samples;
	}

	table_name = text_to_cstring(table_name_text);
	feature_col = text_to_cstring(feature_col_text);
	label_col = text_to_cstring(label_col_text);
	quoted_tbl = quote_identifier(table_name);
	quoted_feat = quote_identifier(feature_col);
	quoted_label = quote_identifier(label_col);

	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT vector_dims(%s) FROM %s WHERE %s IS NOT NULL LIMIT 1",
		quoted_feat, quoted_tbl, quoted_feat);

	if (SPI_connect() != SPI_OK_CONNECT) {
		pfree(table_name);
		pfree(feature_col);
		pfree(label_col);
		pfree(query.data);
		ereport(ERROR, (errmsg("random_forest: SPI_connect failed")));
	}

	if (SPI_execute(query.data, true, 1) == SPI_OK_SELECT
		&& SPI_processed > 0) {
		HeapTuple tup = SPI_tuptable->vals[0];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum dim_datum;
		bool dim_null;

		dim_datum = SPI_getbinval(tup, tupdesc, 1, &dim_null);
		if (!dim_null)
			feature_dim = DatumGetInt32(dim_datum);
	}

	resetStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, (%s)::float8 FROM %s WHERE %s IS NOT NULL "
		"AND %s IS NOT NULL",
		quoted_feat, quoted_label, quoted_tbl, quoted_feat,
		quoted_label);

	if (SPI_execute(query.data, true, 0) != SPI_OK_SELECT) {
		SPI_finish();
		pfree(table_name);
		pfree(feature_col);
		pfree(label_col);
		pfree(query.data);
		ereport(ERROR,
			(errmsg("random_forest: failed to fetch training "
				"data")));
	}

	n_samples = SPI_processed;
	if (n_samples > 0) {
		int i;

		labels = (double *)palloc(sizeof(double) * n_samples);
		/* Stage feature matrix and labels */
		if (feature_dim > 0 && n_samples > 0) {
			stage_features = (float *)palloc(
				sizeof(float) * feature_dim * n_samples);
			memset(stage_features, 0,
				sizeof(float) * feature_dim * n_samples);
		}

		if (feature_dim > 0)
			feature_importance_tmp = (double *)palloc0(
				sizeof(double) * feature_dim);

		if (feature_dim > 0) {
			int j;

			feature_order
				= (int *)palloc(sizeof(int) * feature_dim);
			for (j = 0; j < feature_dim; j++)
				feature_order[j] = j;
		}

		for (i = 0; i < n_samples; i++) {
			HeapTuple tup = SPI_tuptable->vals[i];
			TupleDesc tupdesc = SPI_tuptable->tupdesc;
			Datum feat_datum;
			Datum label_datum;
			bool feat_null;
			bool label_null;

			feat_datum = SPI_getbinval(tup, tupdesc, 1, &feat_null);
			label_datum
				= SPI_getbinval(tup, tupdesc, 2, &label_null);

			if (feat_null || label_null) {
				labels[i] = NAN;
				continue;
			}

			labels[i] = DatumGetFloat8(label_datum);

			if (stage_features != NULL) {
				Vector *vec;
				float *vec_data;
				int j;
				float *dest_row;

				vec = DatumGetVector(feat_datum);
				if (vec->dim != feature_dim) {
					labels[i] = NAN;
					continue;
				}

				vec_data = vec->data;
				dest_row = stage_features + (i * feature_dim);
				for (j = 0; j < feature_dim; j++)
					dest_row[j] = vec_data[j];
			}
		}

		sample_count
			= (int)rint(RF_BOOTSTRAP_FRACTION * (double)n_samples);
		if (sample_count <= 0)
			sample_count = n_samples;
		if (sample_count > n_samples)
			sample_count = n_samples;
		if (sample_count > 0) {
			if (!pg_prng_strong_seed(&rng))
				ereport(ERROR,
					(errmsg("random_forest: failed to seed "
						"PRNG")));
			bootstrap_indices
				= (int *)palloc(sizeof(int) * sample_count);
			for (i = 0; i < sample_count; i++)
				bootstrap_indices[i]
					= (int)pg_prng_uint64_range(&rng, 0,
						(uint64)(n_samples - 1));
		}

		if (feature_dim > 0 && sample_count > 0
			&& stage_features != NULL) {
			feature_sums = (double *)palloc0(
				sizeof(double) * feature_dim);
			feature_sums_sq = (double *)palloc0(
				sizeof(double) * feature_dim);

			for (i = 0; i < sample_count; i++) {
				int src = bootstrap_indices[i];
				float *row;
				int j;

				if (src < 0 || src >= n_samples)
					continue;
				if (!isfinite(labels[src]))
					continue;

				row = stage_features + (src * feature_dim);

				for (j = 0; j < feature_dim; j++) {
					double val = (double)row[j];
					feature_sums[j] += val;
					feature_sums_sq[j] += val * val;
				}
				feature_sum_count++;
			}
		}

		n_classes = rf_count_classes(labels, n_samples);

		if (n_classes > 0) {
			int *counts = (int *)palloc0(sizeof(int) * n_classes);
			int best_idx = 0;

			class_counts_tmp = counts;

			for (i = 0; i < sample_count; i++) {
				int src = bootstrap_indices[i];
				int idx;

				if (src < 0 || src >= n_samples)
					continue;
				idx = (int)rint(labels[src]);
				if (idx < 0 || idx >= n_classes)
					continue;
				counts[idx]++;
				if (counts[idx] > counts[best_idx]) {
					if (idx != best_idx) {
						second_idx = best_idx;
						second_count = counts[best_idx];
						second_value = (double)best_idx;
					}
					best_idx = idx;
				} else if (idx != best_idx
					&& counts[idx] > second_count) {
					second_idx = idx;
					second_count = counts[idx];
					second_value = (double)idx;
				}
			}

			majority_value = (double)best_idx;
			majority_count = counts[best_idx];
			majority_idx = best_idx;
			if (second_idx < 0 && n_classes > 1) {
				for (i = 0; i < n_classes; i++) {
					if (i == best_idx)
						continue;
					if (counts[i] >= second_count) {
						second_idx = i;
						second_count = counts[i];
						second_value = (double)i;
					}
				}
			}

			left_leaf_value = majority_value;
			right_leaf_value = (second_idx >= 0) ? second_value
							     : majority_value;

			if (sample_count > 0) {
				double sum_sq = 0.0;
				double entropy = 0.0;
				int c;
				double ln2 = log(2.0);

				for (c = 0; c < n_classes; c++) {
					double p = (double)counts[c]
						/ (double)sample_count;
					sum_sq += p * p;
					if (p > 0.0)
						entropy -= p * (log(p) / ln2);
				}
				gini_impurity = 1.0 - sum_sq;
				label_entropy = entropy;
			}

			if (class_counts_tmp) {
				StringInfoData histogram;
				initStringInfo(&histogram);
				appendStringInfo(&histogram, "[");
				for (i = 0; i < n_classes; i++) {
					if (i > 0)
						appendStringInfoString(
							&histogram, ", ");
					appendStringInfo(&histogram, "%d",
						class_counts_tmp[i]);
				}
				appendStringInfoChar(&histogram, ']');
				elog(DEBUG1,
					"random_forest: class histogram %s",
					histogram.data);
				pfree(histogram.data);
			}
		}

		if (feature_sums != NULL && feature_sum_count > 0) {
			int j;
			StringInfoData mean_log;
			StringInfoData var_log;

			feature_means_tmp = (double *)palloc(
				sizeof(double) * feature_dim);
			feature_vars_tmp = (double *)palloc(
				sizeof(double) * feature_dim);
			for (j = 0; j < feature_dim; j++) {
				double mean = feature_sums[j]
					/ (double)feature_sum_count;
				double mean_sq = feature_sums_sq[j]
					/ (double)feature_sum_count;
				double variance = mean_sq - (mean * mean);

				if (variance < 0.0)
					variance = 0.0;

				feature_means_tmp[j] = mean;
				feature_vars_tmp[j] = variance;
			}

			initStringInfo(&mean_log);
			appendStringInfo(&mean_log, "[");
			for (j = 0; j < feature_dim && j < 5; j++) {
				if (j > 0)
					appendStringInfoString(&mean_log, ", ");
				appendStringInfo(&mean_log, "%.3f",
					feature_means_tmp[j]);
			}
			if (feature_dim > 5)
				appendStringInfoString(&mean_log, ", ...");
			appendStringInfoChar(&mean_log, ']');
			elog(DEBUG1, "random_forest: feature means %s",
				mean_log.data);
			pfree(mean_log.data);

			initStringInfo(&var_log);
			appendStringInfo(&var_log, "[");
			for (j = 0; j < feature_dim && j < 5; j++) {
				if (j > 0)
					appendStringInfoString(&var_log, ", ");
				appendStringInfo(
					&var_log, "%.3f", feature_vars_tmp[j]);
			}
			if (feature_dim > 5)
				appendStringInfoString(&var_log, ", ...");
			appendStringInfoChar(&var_log, ']');
			elog(DEBUG1, "random_forest: feature variances %s",
				var_log.data);
			pfree(var_log.data);
		}

		if (feature_dim > 0 && sample_count > 0 && n_classes > 0
			&& stage_features != NULL) {
			feature_limit = Min(feature_dim, RF_STUB_MAX_FEATURES);
			if (feature_limit < 1)
				feature_limit = 1;

			class_feature_sums = (double *)palloc0(
				sizeof(double) * n_classes * feature_limit);
			class_feature_counts = (int *)palloc0(
				sizeof(int) * n_classes * feature_limit);

			for (i = 0; i < sample_count; i++) {
				int src = bootstrap_indices[i];
				int cls;
				int f;
				float *row;

				if (src < 0 || src >= n_samples)
					continue;
				if (!isfinite(labels[src]))
					continue;
				cls = (int)rint(labels[src]);
				if (cls < 0 || cls >= n_classes)
					continue;

				row = stage_features + (src * feature_dim);

				for (f = 0;
					f < feature_limit && f < feature_dim;
					f++) {
					double val = (double)row[f];
					class_feature_sums[cls * feature_limit
						+ f]
						+= val;
					class_feature_counts[cls * feature_limit
						+ f]++;
				}
			}

			if (majority_idx >= 0) {
				int f;

				for (f = 0; f < feature_limit; f++) {
					int idx = majority_idx * feature_limit
						+ f;
					if (class_feature_counts[idx] > 0) {
						double maj_mean
							= class_feature_sums
								  [idx]
							/ (double)
								class_feature_counts
									[idx];
						double sec_mean = 0.0;
						int sec_idx = -1;
						int sec_count = 0;

						if (second_idx >= 0) {
							sec_idx = second_idx
									* feature_limit
								+ f;
							sec_count = class_feature_counts
								[sec_idx];
						}

						if (sec_count > 0) {
							sec_mean
								= class_feature_sums
									  [sec_idx]
								/ (double)
									sec_count;

							if (fabs(maj_mean
								    - sec_mean)
								> best_score) {
								best_score = fabs(
									maj_mean
									- sec_mean);
								best_feature
									= f;
								best_majority_mean
									= maj_mean;
								best_second_mean
									= sec_mean;
								best_score_valid
									= true;
							}
						}
					}
				}
			}

			if (!best_score_valid && majority_idx >= 0) {
				int idx = majority_idx * feature_limit;
				if (class_feature_counts[idx] > 0)
					best_majority_mean
						= class_feature_sums[idx]
						/ (double)class_feature_counts
							[idx];
			}

			if (best_score_valid) {
				class_majority_mean = best_majority_mean;
				class_second_mean = best_second_mean;
				class_mean_threshold = 0.5
					* (class_majority_mean
						+ class_second_mean);
				class_mean_threshold_valid = true;
				split_feature = best_feature;
				elog(DEBUG1,
					"random_forest: best feature=%d "
					"majority_mean=%.3f second_mean=%.3f "
					"threshold=%.3f score=%.3f",
					split_feature, class_majority_mean,
					class_second_mean, class_mean_threshold,
					best_score);
			}
		}

		/* Refine threshold using sorted split candidates on the chosen
		 * feature */
		if (feature_dim > 0 && sample_count > 0 && n_classes > 0
			&& stage_features != NULL) {
			int sf_idx = (split_feature >= 0
					     && split_feature < feature_dim)
				? split_feature
				: 0;

			split_pairs = (RFSplitPair *)palloc(
				sizeof(RFSplitPair) * sample_count);
			split_pair_count = 0;

			for (i = 0; i < sample_count; i++) {
				int src = bootstrap_indices[i];
				int cls;
				float *row;

				if (src < 0 || src >= n_samples)
					continue;
				cls = (int)rint(labels[src]);
				if (!isfinite(labels[src]) || cls < 0
					|| cls >= n_classes)
					continue;

				row = stage_features + (src * feature_dim);
				if (sf_idx >= feature_dim)
					continue;

				split_pairs[split_pair_count].value
					= (double)row[sf_idx];
				split_pairs[split_pair_count].cls = cls;
				split_pair_count++;
			}

			if (split_pair_count > 1) {
				int *left_counts_tmp = (int *)palloc0(
					sizeof(int) * n_classes);
				int *right_counts_tmp = (int *)palloc0(
					sizeof(int) * n_classes);
				int right_total_eval = 0;
				int left_total_eval = 0;

				qsort(split_pairs, split_pair_count,
					sizeof(RFSplitPair), rf_split_pair_cmp);

				if (class_counts_tmp != NULL) {
					for (i = 0; i < n_classes; i++)
						right_counts_tmp[i]
							= class_counts_tmp[i];
				} else {
					for (i = 0; i < split_pair_count; i++)
						right_counts_tmp
							[split_pairs[i].cls]++;
				}

				for (i = 0; i < n_classes; i++)
					right_total_eval += right_counts_tmp[i];

				for (i = 0; i < split_pair_count - 1; i++) {
					int cls = split_pairs[i].cls;

					left_counts_tmp[cls]++;
					right_counts_tmp[cls]--;
					left_total_eval++;
					right_total_eval--;

					if (split_pairs[i].value
						== split_pairs[i + 1].value)
						continue;
					if (left_total_eval <= 0
						|| right_total_eval <= 0)
						continue;

					{
						double left_imp
							= rf_gini_from_counts(
								left_counts_tmp,
								n_classes,
								left_total_eval);
						double right_imp
							= rf_gini_from_counts(
								right_counts_tmp,
								n_classes,
								right_total_eval);
						double weighted
							= ((double)left_total_eval
								  / (double)
									  split_pair_count)
								* left_imp
							+ ((double)right_total_eval
								  / (double)
									  split_pair_count)
								* right_imp;

						if (weighted
							< best_split_impurity) {
							best_split_impurity
								= weighted;
							best_split_threshold
								= 0.5
								* (split_pairs[i]
										.value
									+ split_pairs[i
										+ 1]
										  .value);
							best_split_valid = true;
						}
					}
				}

				pfree(left_counts_tmp);
				pfree(right_counts_tmp);
			}

			if (best_split_valid) {
				class_mean_threshold = best_split_threshold;
				class_mean_threshold_valid = true;
				split_feature = sf_idx;
				elog(DEBUG1,
					"random_forest: refined threshold "
					"feature=%d threshold=%.3f "
					"impurity=%.6f",
					split_feature, class_mean_threshold,
					best_split_impurity);
			}

			if (split_pairs) {
				pfree(split_pairs);
				split_pairs = NULL;
			}
		}

		if (feature_dim > 0 && feature_means_tmp != NULL
			&& n_classes > 0 && stage_features != NULL
			&& sample_count > 0) {
			int sf = (split_feature >= 0
					 && split_feature < feature_dim)
				? split_feature
				: 0;
			double threshold = feature_means_tmp[sf];

			left_total = 0;
			right_total = 0;
			left_sum = 0.0;
			right_sum = 0.0;
			left_majority_idx = -1;
			right_majority_idx = -1;

			if (class_mean_threshold_valid)
				threshold = class_mean_threshold;

			left_counts = (int *)palloc0(sizeof(int) * n_classes);
			right_counts = (int *)palloc0(sizeof(int) * n_classes);
			if (feature_limit > 0) {
				left_feature_sums_vec = (double *)palloc0(
					sizeof(double) * feature_limit);
				right_feature_sums_vec = (double *)palloc0(
					sizeof(double) * feature_limit);
				left_feature_counts_vec = (int *)palloc0(
					sizeof(int) * feature_limit);
				right_feature_counts_vec = (int *)palloc0(
					sizeof(int) * feature_limit);
			}

			for (i = 0; i < sample_count; i++) {
				int src = bootstrap_indices[i];
				int cls;
				int f;
				float *row;
				double value;

				if (src < 0 || src >= n_samples)
					continue;
				if (!isfinite(labels[src]))
					continue;
				cls = (int)rint(labels[src]);
				if (cls < 0 || cls >= n_classes)
					continue;

				if (sf >= feature_dim)
					continue;

				row = stage_features + (src * feature_dim);
				value = (double)row[sf];

				if (value <= threshold) {
					left_counts[cls]++;
					left_total++;
					left_sum += value;
					if (feature_limit > 0) {
						for (f = 0; f < feature_limit
							&& f < feature_dim;
							f++) {
							left_feature_sums_vec[f]
								+= (double)
									row[f];
							left_feature_counts_vec
								[f]++;
						}
					}
				} else {
					right_counts[cls]++;
					right_total++;
					right_sum += value;
					if (feature_limit > 0) {
						for (f = 0; f < feature_limit
							&& f < feature_dim;
							f++) {
							right_feature_sums_vec
								[f]
								+= (double)
									row[f];
							right_feature_counts_vec
								[f]++;
						}
					}
				}
			}

			for (i = 0; i < n_classes; i++) {
				if (left_total > 0
					&& (left_majority_idx < 0
						|| left_counts[i] > left_counts
								[left_majority_idx]))
					left_majority_idx = i;
				if (right_total > 0
					&& (right_majority_idx < 0
						|| right_counts[i]
							> right_counts
								[right_majority_idx]))
					right_majority_idx = i;
			}

			if (left_majority_idx >= 0)
				left_leaf_value = (double)left_majority_idx;

			if (right_majority_idx >= 0) {
				right_leaf_value = (double)right_majority_idx;
				second_value = right_leaf_value;
				if (class_counts_tmp != NULL)
					second_fraction
						= ((double)class_counts_tmp
								  [right_majority_idx])
						/ (double)sample_count;
				else if (right_total > 0)
					second_fraction
						= ((double)right_counts
								  [right_majority_idx])
						/ (double)sample_count;
			}

			if (sample_count > 0) {
				if (left_total > 0)
					left_branch_fraction
						= ((double)left_total)
						/ (double)sample_count;
				if (right_total > 0)
					right_branch_fraction
						= ((double)right_total)
						/ (double)sample_count;
			}

			if (feature_limit > 0 && left_feature_sums_vec != NULL
				&& right_feature_sums_vec != NULL) {
				int f;

				left_branch_means_vec = (double *)palloc(
					sizeof(double) * feature_limit);
				right_branch_means_vec = (double *)palloc(
					sizeof(double) * feature_limit);

				for (f = 0; f < feature_limit; f++) {
					if (left_feature_counts_vec != NULL
						&& left_feature_counts_vec[f]
							> 0)
						left_branch_means_vec[f]
							= left_feature_sums_vec
								  [f]
							/ (double)
								left_feature_counts_vec
									[f];
					else if (feature_means_tmp != NULL
						&& f < feature_dim)
						left_branch_means_vec[f]
							= feature_means_tmp[f];
					else
						left_branch_means_vec[f] = 0.0;

					if (right_feature_counts_vec != NULL
						&& right_feature_counts_vec[f]
							> 0)
						right_branch_means_vec[f]
							= right_feature_sums_vec
								  [f]
							/ (double)
								right_feature_counts_vec
									[f];
					else if (feature_means_tmp != NULL
						&& f < feature_dim)
						right_branch_means_vec[f]
							= feature_means_tmp[f];
					else
						right_branch_means_vec[f]
							= left_branch_means_vec
								[f];
				}
			}

			if ((left_total == 0 || right_total == 0)
				&& feature_vars_tmp != NULL && sf < feature_dim
				&& feature_vars_tmp[sf] > 0.0) {
				double adjust;

				adjust = 0.5 * sqrt(feature_vars_tmp[sf]);
				if (right_total == 0) {
					threshold = feature_means_tmp[sf]
						+ adjust;
					branch_threshold_valid = true;
					if (right_branch_fraction <= 0.0)
						right_branch_fraction = 0.5;
					if (left_branch_fraction <= 0.0)
						left_branch_fraction = 1.0
							- right_branch_fraction;
					right_leaf_value = (second_idx >= 0)
						? second_value
						: majority_value;
				} else if (left_total == 0) {
					threshold = feature_means_tmp[sf]
						- adjust;
					branch_threshold_valid = true;
					if (left_branch_fraction <= 0.0)
						left_branch_fraction = 0.5;
					if (right_branch_fraction <= 0.0)
						right_branch_fraction = 1.0
							- left_branch_fraction;
					left_leaf_value = majority_value;
				}
			}

			if (left_total > 0 && right_total > 0) {
				double left_mean
					= left_sum / (double)left_total;
				double right_mean
					= right_sum / (double)right_total;

				threshold = 0.5 * (left_mean + right_mean);
				branch_threshold_valid = true;
			}

			if (left_branch_fraction <= 0.0
				&& right_branch_fraction <= 0.0)
				left_branch_fraction = majority_fraction;
			else if (left_branch_fraction <= 0.0
				&& right_branch_fraction > 0.0)
				left_branch_fraction
					= 1.0 - right_branch_fraction;
			else if (right_branch_fraction <= 0.0
				&& left_branch_fraction > 0.0)
				right_branch_fraction
					= 1.0 - left_branch_fraction;

			if (second_fraction <= 0.0
				&& right_branch_fraction > 0.0)
				second_fraction = right_branch_fraction;

			if (left_counts)
				pfree(left_counts);
			if (right_counts)
				pfree(right_counts);
			left_counts = NULL;
			right_counts = NULL;

			elog(DEBUG1,
				"random_forest: branch totals left=%d "
				"right=%d split=%.3f lf=%.3f rf=%.3f",
				left_total, right_total, threshold,
				left_branch_fraction, right_branch_fraction);
		}

		if (sample_count > 0 && majority_count > 0)
			majority_fraction = ((double)majority_count)
				/ (double)sample_count;
	}

	if (sample_count > 0 && second_count > 0 && second_fraction <= 0.0)
		second_fraction = ((double)second_count) / (double)sample_count;

	if (majority_count > 0) {
		MemoryContext oldctx;
		int forest_trees = forest_trees_arg;
		int t;

		if (forest_trees < 1)
			forest_trees = 1;
		if (forest_trees > n_samples)
			forest_trees = n_samples;

		if (forest_trees > 0) {
			trees = (GTree **)palloc0(
				sizeof(GTree *) * forest_trees);
			tree_majorities = (double *)palloc0(
				sizeof(double) * forest_trees);
			tree_majority_fractions = (double *)palloc0(
				sizeof(double) * forest_trees);
			tree_seconds = (double *)palloc0(
				sizeof(double) * forest_trees);
			tree_second_fractions = (double *)palloc0(
				sizeof(double) * forest_trees);
			tree_oob_accuracy = (double *)palloc0(
				sizeof(double) * forest_trees);
		}

		for (t = 0; t < forest_trees; t++) {
			GTree *tree;
			int node_idx;
			int left_idx;
			int right_idx;
			int tree_feature = split_feature;
			int feature_for_split = split_feature;
			double tree_threshold = split_threshold;
			double var0 = 0.0;
			int tree_majority_idx = -1;
			int tree_second_idx = -1;
			int tree_majority_count = 0;
			int tree_second_count = 0;
			double tree_majority_value = majority_value;
			double tree_second_value = second_value;
			double tree_majority_frac = majority_fraction;
			double tree_second_frac = second_fraction;
			int *tree_counts = NULL;
			int boot_samples = 0;
			int sample_target = n_samples;
			int j;
			int *tree_bootstrap = NULL;
			RFSplitPair *tree_pairs = NULL;
			int tree_pair_count = 0;
			double tree_best_impurity = DBL_MAX;
			double tree_best_threshold = split_threshold;
			int tree_best_feature = split_feature;
			bool tree_split_valid = false;
			int mtry = 0;
			int candidates = 0;
			int *left_tmp = NULL;
			int *right_tmp = NULL;
			int left_total_local = 0;
			int right_total_local = 0;
			int left_majority_local = -1;
			int right_majority_local = -1;
			int left_best_count = 0;
			int right_best_count = 0;
			double tree_left_value = left_leaf_value;
			double tree_right_value = right_leaf_value;
			double tree_left_fraction = left_branch_fraction;
			double tree_right_fraction = right_branch_fraction;
			bool *inbag = NULL;
			int oob_total_local = 0;
			int oob_correct_local = 0;
			int *left_indices_local = NULL;
			int *right_indices_local = NULL;
			int left_index_count = 0;
			int right_index_count = 0;

			if (n_classes > 0)
				tree_counts = (int *)palloc0(
					sizeof(int) * n_classes);

			if (n_samples > 0) {
				sample_target = (int)rint((double)n_samples
					* RF_BOOTSTRAP_FRACTION);
				if (sample_target < 1)
					sample_target = 1;
				if (sample_target > n_samples)
					sample_target = n_samples;
			}

			if (n_samples > 0)
				inbag = (bool *)palloc0(
					sizeof(bool) * n_samples);

			if (sample_target > 0 && n_samples > 0)
				tree_bootstrap = (int *)palloc(
					sizeof(int) * sample_target);

			for (j = 0; j < sample_target; j++) {
				int idx;

				idx = (int)pg_prng_uint64_range(
					&rng, 0, (uint64)(n_samples - 1));
				if (tree_bootstrap != NULL)
					tree_bootstrap[j] = idx;
				boot_samples++;
				if (tree_counts != NULL && labels != NULL) {
					int cls;

					if (!isfinite(labels[idx]))
						continue;
					cls = (int)rint(labels[idx]);
					if (cls < 0 || cls >= n_classes)
						continue;
					tree_counts[cls]++;
					if (tree_counts[cls]
						> tree_majority_count) {
						if (cls != tree_majority_idx) {
							tree_second_idx
								= tree_majority_idx;
							tree_second_count
								= tree_majority_count;
						}
						tree_majority_idx = cls;
						tree_majority_count
							= tree_counts[cls];
					} else if (cls != tree_majority_idx
						&& tree_counts[cls]
							> tree_second_count) {
						tree_second_idx = cls;
						tree_second_count
							= tree_counts[cls];
					}
				}
			}

			if (inbag != NULL && tree_bootstrap != NULL) {
				for (j = 0; j < boot_samples; j++) {
					int idx = tree_bootstrap[j];

					if (idx >= 0 && idx < n_samples)
						inbag[idx] = true;
				}
			}

			if (tree_majority_idx >= 0) {
				tree_majority_value = (double)tree_majority_idx;
				if (boot_samples > 0)
					tree_majority_frac
						= (double)tree_majority_count
						/ (double)boot_samples;
			}

			if (tree_second_idx < 0)
				tree_second_idx = second_idx;

			if (tree_second_idx >= 0) {
				tree_second_value = (double)tree_second_idx;
				if (boot_samples > 0 && tree_second_count > 0)
					tree_second_frac
						= (double)tree_second_count
						/ (double)boot_samples;
				else
					tree_second_frac = second_fraction;
			}

			if (feature_dim > 0 && stage_features != NULL
				&& labels != NULL && n_classes > 0
				&& tree_bootstrap != NULL) {
				int f;

				if (feature_order != NULL) {
					for (f = 0; f < feature_dim; f++)
						feature_order[f] = f;
				}

				mtry = (int)sqrt((double)feature_dim);
				if (mtry < 1)
					mtry = 1;
				if (mtry > feature_dim)
					mtry = feature_dim;
				candidates = mtry;

				if (feature_order != NULL && feature_dim > 0) {
					for (f = 0; f < candidates; f++) {
						int swap_idx;

						swap_idx = (int)
							pg_prng_uint64_range(
								&rng, (uint64)f,
								(uint64)(feature_dim
									- 1));
						if (swap_idx != f) {
							int tmp = feature_order
								[f];

							feature_order[f]
								= feature_order
									[swap_idx];
							feature_order[swap_idx]
								= tmp;
						}
					}
				} else
					candidates
						= Min(candidates, feature_dim);

				for (f = 0; f < candidates; f++) {
					int feature_idx = feature_order
						? feature_order[f]
						: f;
					int s;

					if (feature_idx < 0
						|| feature_idx >= feature_dim)
						continue;

					tree_pairs = (RFSplitPair *)palloc(
						sizeof(RFSplitPair)
						* boot_samples);
					tree_pair_count = 0;

					for (s = 0; s < boot_samples; s++) {
						int sample_idx
							= tree_bootstrap[s];
						float *row;
						double value;
						int cls;

						if (sample_idx < 0
							|| sample_idx
								>= n_samples)
							continue;
						if (!isfinite(
							    labels[sample_idx]))
							continue;

						cls = (int)rint(
							labels[sample_idx]);
						if (cls < 0 || cls >= n_classes)
							continue;

						row = stage_features
							+ (sample_idx
								* feature_dim);
						value = (double)
							row[feature_idx];

						tree_pairs[tree_pair_count]
							.value
							= value;
						tree_pairs[tree_pair_count].cls
							= cls;
						tree_pair_count++;
					}

					if (tree_pair_count > 1) {
						int *left_counts_tmp;
						int *right_counts_tmp;
						int left_total_eval = 0;
						int right_total_eval = 0;

						qsort(tree_pairs,
							tree_pair_count,
							sizeof(RFSplitPair),
							rf_split_pair_cmp);

						left_counts_tmp
							= (int *)palloc0(
								sizeof(int)
								* n_classes);
						right_counts_tmp
							= (int *)palloc0(
								sizeof(int)
								* n_classes);

						for (s = 0; s < tree_pair_count;
							s++)
							right_counts_tmp
								[tree_pairs[s].cls]++;

						right_total_eval
							= tree_pair_count;

						for (s = 0;
							s < tree_pair_count - 1;
							s++) {
							int cls_val
								= tree_pairs[s]
									  .cls;
							double left_imp;
							double right_imp;
							double weighted;
							double threshold_candidate;

							left_counts_tmp
								[cls_val]++;
							right_counts_tmp
								[cls_val]--;
							left_total_eval++;
							right_total_eval--;

							if (tree_pairs[s].value
								== tree_pairs[s
									+ 1]
									   .value)
								continue;
							if (left_total_eval <= 0
								|| right_total_eval
									<= 0)
								continue;

							left_imp = rf_gini_from_counts(
								left_counts_tmp,
								n_classes,
								left_total_eval);
							right_imp = rf_gini_from_counts(
								right_counts_tmp,
								n_classes,
								right_total_eval);
							threshold_candidate
								= 0.5
								* (tree_pairs[s].value
									+ tree_pairs[s
										+ 1]
										  .value);
							weighted
								= ((double)left_total_eval
									  / (double)
										  tree_pair_count)
									* left_imp
								+ ((double)right_total_eval
									  / (double)
										  tree_pair_count)
									* right_imp;

							if (weighted
								< tree_best_impurity) {
								tree_best_impurity
									= weighted;
								tree_best_threshold
									= threshold_candidate;
								tree_best_feature
									= feature_idx;
								tree_split_valid
									= true;
							}
						}

						pfree(left_counts_tmp);
						pfree(right_counts_tmp);
					}

					if (tree_pairs != NULL) {
						pfree(tree_pairs);
						tree_pairs = NULL;
					}
				}
			}

			if (tree_split_valid) {
				tree_feature = tree_best_feature;
				feature_for_split = tree_best_feature;
				tree_threshold = tree_best_threshold;
			} else
				feature_for_split = tree_feature;

			if (tree_bootstrap != NULL && feature_dim > 0
				&& n_classes > 0 && stage_features != NULL
				&& labels != NULL && tree_feature >= 0
				&& tree_feature < feature_dim) {
				left_tmp = (int *)palloc0(
					sizeof(int) * n_classes);
				right_tmp = (int *)palloc0(
					sizeof(int) * n_classes);

				left_index_count = 0;
				right_index_count = 0;
				if (boot_samples > 0) {
					if (left_indices_local == NULL)
						left_indices_local
							= (int *)palloc(
								sizeof(int)
								* boot_samples);
					if (right_indices_local == NULL)
						right_indices_local
							= (int *)palloc(
								sizeof(int)
								* boot_samples);
				}

				for (j = 0; j < boot_samples; j++) {
					int sample_idx = tree_bootstrap[j];
					float *row;
					double value;
					int cls;

					if (sample_idx < 0
						|| sample_idx >= n_samples)
						continue;
					if (!isfinite(labels[sample_idx]))
						continue;

					cls = (int)rint(labels[sample_idx]);
					if (cls < 0 || cls >= n_classes)
						continue;

					row = stage_features
						+ (sample_idx * feature_dim);
					value = (double)row[tree_feature];

					if (value <= tree_threshold) {
						left_tmp[cls]++;
						left_total_local++;
						if (left_indices_local != NULL)
							left_indices_local
								[left_index_count++]
								= sample_idx;
					} else {
						right_tmp[cls]++;
						right_total_local++;
						if (right_indices_local != NULL)
							right_indices_local
								[right_index_count++]
								= sample_idx;
					}
				}

				for (j = 0; j < n_classes; j++) {
					if (left_tmp[j] > left_best_count) {
						left_best_count = left_tmp[j];
						left_majority_local = j;
					}
					if (right_tmp[j] > right_best_count) {
						right_best_count = right_tmp[j];
						right_majority_local = j;
					}
				}

				if (left_majority_local >= 0)
					tree_left_value
						= (double)left_majority_local;
				if (right_majority_local >= 0) {
					tree_right_value
						= (double)right_majority_local;
					tree_second_value = tree_right_value;
				}

				if (boot_samples > 0) {
					tree_left_fraction
						= (double)left_total_local
						/ (double)boot_samples;
					tree_right_fraction
						= (double)right_total_local
						/ (double)boot_samples;
				}

				if (tree_second_frac <= 0.0
					&& tree_right_fraction > 0.0)
					tree_second_frac = tree_right_fraction;

				pfree(left_tmp);
				pfree(right_tmp);
			}

			feature_for_split = tree_feature;

			oldctx = MemoryContextSwitchTo(TopMemoryContext);
			tree = gtree_create("rf_model_tree", 4);
			MemoryContextSwitchTo(oldctx);

			if (tree == NULL) {
				if (tree_counts)
					pfree(tree_counts);
				if (tree_bootstrap)
					pfree(tree_bootstrap);
				continue;
			}

			if (feature_dim > 0 && feature_means_tmp != NULL) {
				if (feature_for_split < 0
					|| feature_for_split >= feature_dim)
					feature_for_split = 0;
				if (feature_vars_tmp != NULL
					&& feature_for_split < feature_dim)
					var0 = feature_vars_tmp
						[feature_for_split];

				if (!branch_threshold_valid)
					tree_threshold = feature_means_tmp
						[feature_for_split];

				node_idx = gtree_add_split(tree,
					feature_for_split, tree_threshold);

				if (left_indices_local != NULL
					&& left_index_count > 0)
					left_idx = rf_build_branch_tree(tree,
						stage_features, labels,
						feature_vars_tmp, feature_dim,
						n_classes, left_indices_local,
						left_index_count, 1,
						max_depth_arg, min_samples_arg,
						&rng, feature_order,
						feature_importance_tmp,
						&max_split_deviation);
				else
					left_idx = gtree_add_leaf(
						tree, tree_left_value);

				if (right_indices_local != NULL
					&& right_index_count > 0)
					right_idx = rf_build_branch_tree(tree,
						stage_features, labels,
						feature_vars_tmp, feature_dim,
						n_classes, right_indices_local,
						right_index_count, 1,
						max_depth_arg, min_samples_arg,
						&rng, feature_order,
						feature_importance_tmp,
						&max_split_deviation);
				else
					right_idx = gtree_add_leaf(
						tree, tree_right_value);

				gtree_set_child(tree, node_idx, left_idx, true);
				gtree_set_child(
					tree, node_idx, right_idx, false);
				gtree_set_root(tree, node_idx);

				if (var0 > 0.0) {
					double split_dev = fabs(tree_threshold)
						/ sqrt(var0);

					if (split_dev > max_split_deviation)
						max_split_deviation = split_dev;
				}

				if (tree_count == 0) {
					split_feature = feature_for_split;
					split_threshold = tree_threshold;
					left_leaf_value = tree_left_value;
					right_leaf_value = tree_right_value;
					left_branch_fraction
						= tree_left_fraction;
					right_branch_fraction
						= tree_right_fraction;
					branch_threshold_valid = true;
					second_value = tree_second_value;
					second_fraction = tree_second_frac;
				}

				elog(DEBUG1,
					"random_forest: tree %d split "
					"feature=%d "
					"threshold=%.3f left=%.3f right=%.3f",
					t + 1, feature_for_split,
					tree_threshold, tree_left_value,
					tree_right_value);
			} else {
				left_idx
					= gtree_add_leaf(tree, tree_left_value);
				gtree_set_root(tree, left_idx);
			}

			if (inbag != NULL && labels != NULL && n_samples > 0
				&& stage_features != NULL) {
				oob_total_local = 0;
				oob_correct_local = 0;

				for (j = 0; j < n_samples; j++) {
					int actual;
					int predicted;

					if (inbag[j])
						continue;
					if (!isfinite(labels[j]))
						continue;

					actual = (int)rint(labels[j]);
					if (actual < 0 || actual >= n_classes)
						continue;

					if (feature_dim > 0) {
						float *row = stage_features
							+ (j * feature_dim);
						double tree_pred;

						tree_pred = rf_tree_predict_row(
							tree, row, feature_dim);
						predicted
							= (int)rint(tree_pred);
					} else
						predicted = (int)rint(
							tree_left_value);

					oob_total_local++;
					if (predicted == actual)
						oob_correct_local++;
				}

				if (tree_oob_accuracy != NULL
					&& t < forest_trees) {
					if (oob_total_local > 0)
						tree_oob_accuracy[t]
							= (double)oob_correct_local
							/ (double)
								oob_total_local;
					else
						tree_oob_accuracy[t] = 0.0;
				}

				oob_total_all += oob_total_local;
				oob_correct_all += oob_correct_local;
			}

			gtree_validate(tree);

			if (primary_tree == NULL)
				primary_tree = tree;

			if (trees != NULL) {
				int idx = tree_count;

				if (idx < forest_trees) {
					trees[idx] = tree;
					tree_majorities[idx]
						= tree_majority_value;
					tree_majority_fractions[idx]
						= tree_majority_frac;
					tree_seconds[idx] = tree_second_value;
					tree_second_fractions[idx]
						= tree_second_frac;
					tree_count++;
				}
			}

			if (tree_counts)
				pfree(tree_counts);
			if (tree_bootstrap)
				pfree(tree_bootstrap);
			if (inbag)
				pfree(inbag);
			if (left_indices_local)
				pfree(left_indices_local);
			if (right_indices_local)
				pfree(right_indices_local);
		}
	}

	if (oob_total_all > 0)
		forest_oob_accuracy
			= (double)oob_correct_all / (double)oob_total_all;

	if (feature_importance_tmp != NULL && feature_dim > 0) {
		int i;
		double importance_total = 0.0;
		double top_import0 = 0.0;
		double top_import1 = 0.0;
		double top_import2 = 0.0;
		int top_index0 = -1;
		int top_index1 = -1;
		int top_index2 = -1;

		for (i = 0; i < feature_dim; i++) {
			double val = feature_importance_tmp[i];

			if (val < 0.0)
				val = 0.0;
			feature_importance_tmp[i] = val;
			importance_total += val;

			if (val > top_import0) {
				top_import2 = top_import1;
				top_index2 = top_index1;
				top_import1 = top_import0;
				top_index1 = top_index0;
				top_import0 = val;
				top_index0 = i;
			} else if (val > top_import1) {
				top_import2 = top_import1;
				top_index2 = top_index1;
				top_import1 = val;
				top_index1 = i;
			} else if (val > top_import2) {
				top_import2 = val;
				top_index2 = i;
			}
		}

		if (importance_total > 0.0) {
			for (i = 0; i < feature_dim; i++)
				feature_importance_tmp[i] /= importance_total;
			if (top_import0 > 0.0)
				top_import0 /= importance_total;
			if (top_import1 > 0.0)
				top_import1 /= importance_total;
			if (top_import2 > 0.0)
				top_import2 /= importance_total;
		}

		elog(DEBUG1,
			"random_forest: feature importance top=%d:%.3f %d:%.3f "
			"%d:%.3f",
			top_index0, top_import0, top_index1, top_import1,
			top_index2, top_import2);
	}

	model_id = rf_next_model_id++;
	if (feature_limit > 0 && left_branch_means_vec == NULL
		&& right_branch_means_vec == NULL)
		feature_limit = 0;

	rf_store_model(model_id, feature_dim, n_samples, n_classes,
		majority_value, majority_fraction, gini_impurity, label_entropy,
		class_counts_tmp, feature_means_tmp, feature_vars_tmp,
		feature_importance_tmp, primary_tree, split_feature,
		split_threshold, second_value, second_fraction, left_leaf_value,
		left_branch_fraction, right_leaf_value, right_branch_fraction,
		max_deviation, max_split_deviation, feature_limit,
		left_branch_means_vec, right_branch_means_vec, tree_count,
		trees, tree_majorities, tree_majority_fractions, tree_seconds,
		tree_second_fractions, tree_oob_accuracy, forest_oob_accuracy);

	elog(NOTICE,
		"train_random_forest_classifier: rows=%d, classes=%d, dim=%d, "
		"majority=%.3f, frac=%.3f, second=%.3f, sfrac=%.3f, gini=%.3f, "
		"entropy=%.3f, oob=%.3f",
		n_samples, n_classes, feature_dim, majority_value,
		majority_fraction, second_value, second_fraction, gini_impurity,
		label_entropy, forest_oob_accuracy);

	if (class_counts_tmp)
		pfree(class_counts_tmp);
	if (labels)
		pfree(labels);
	if (feature_means_tmp)
		pfree(feature_means_tmp);
	if (feature_vars_tmp)
		pfree(feature_vars_tmp);
	if (feature_importance_tmp)
		pfree(feature_importance_tmp);
	if (feature_sums)
		pfree(feature_sums);
	if (feature_sums_sq)
		pfree(feature_sums_sq);
	if (class_feature_sums)
		pfree(class_feature_sums);
	if (class_feature_counts)
		pfree(class_feature_counts);
	if (left_feature_sums_vec)
		pfree(left_feature_sums_vec);
	if (right_feature_sums_vec)
		pfree(right_feature_sums_vec);
	if (left_feature_counts_vec)
		pfree(left_feature_counts_vec);
	if (right_feature_counts_vec)
		pfree(right_feature_counts_vec);
	if (left_branch_means_vec)
		pfree(left_branch_means_vec);
	if (right_branch_means_vec)
		pfree(right_branch_means_vec);
	if (feature_order)
		pfree(feature_order);
	if (trees)
		pfree(trees);
	if (tree_majorities)
		pfree(tree_majorities);
	if (tree_majority_fractions)
		pfree(tree_majority_fractions);
	if (tree_seconds)
		pfree(tree_seconds);
	if (tree_second_fractions)
		pfree(tree_second_fractions);
	if (tree_oob_accuracy)
		pfree(tree_oob_accuracy);
	if (bootstrap_indices)
		pfree(bootstrap_indices);
	if (stage_features)
		pfree(stage_features);

	SPI_finish();

	if (table_name)
		pfree(table_name);
	if (feature_col)
		pfree(feature_col);
	if (label_col)
		pfree(label_col);
	if (query.data)
		pfree(query.data);

	PG_RETURN_INT32(model_id);
}

static double rf_tree_predict_single(const GTree *tree, const RFModel *model,
	const Vector *vec, double *left_dist, double *right_dist,
	int *leaf_out) {
	const GTreeNode *nodes;
	int idx;
	int steps = 0;
	int path_nodes[GTREE_MAX_DEPTH + 1];
	char path_dir[GTREE_MAX_DEPTH];
	int path_len = 0;
	int leaf_idx = -1;
	double result = 0.0;
	int i;

	if (leaf_out)
		*leaf_out = -1;

	if (left_dist)
		*left_dist = -1.0;
	if (right_dist)
		*right_dist = -1.0;

	if (tree == NULL)
		return (model != NULL) ? model->majority_value : 0.0;

	if (tree->root < 0 || tree->count <= 0)
		return (model != NULL) ? model->majority_value : 0.0;

	nodes = gtree_nodes(tree);
	idx = tree->root;

	while (idx >= 0 && idx < tree->count) {
		const GTreeNode *node = &nodes[idx];

		if (path_len <= GTREE_MAX_DEPTH)
			path_nodes[path_len] = idx;

		if (node->is_leaf) {
			leaf_idx = idx;
			break;
		}

		if (vec == NULL || node->feature_idx < 0
			|| node->feature_idx >= vec->dim) {
			elog(DEBUG1,
				"random_forest: path aborted at node %d "
				"(feature "
				"%d)",
				idx, node->feature_idx);
			if (model)
				return model->majority_value;
			return 0.0;
		}

		if (vec->data[node->feature_idx] <= node->threshold) {
			if (path_len < GTREE_MAX_DEPTH)
				path_dir[path_len] = 'L';
			idx = node->left;
		} else {
			if (path_len < GTREE_MAX_DEPTH)
				path_dir[path_len] = 'R';
			idx = node->right;
		}

		path_len++;

		if (++steps > GTREE_MAX_DEPTH)
			break;
	}

	if (leaf_idx >= 0 && leaf_idx < tree->count && nodes[leaf_idx].is_leaf)
		result = nodes[leaf_idx].value;
	else
		result = (model != NULL) ? model->majority_value : 0.0;

	if (path_len > GTREE_MAX_DEPTH)
		path_len = GTREE_MAX_DEPTH;

	if (leaf_idx >= 0 && path_len <= GTREE_MAX_DEPTH)
		path_nodes[path_len] = leaf_idx;

	{
		StringInfoData path_log;
		int edge_count = (leaf_idx >= 0) ? path_len : path_len - 1;

		initStringInfo(&path_log);
		appendStringInfo(&path_log, "[");
		for (i = 0; i <= edge_count && i <= GTREE_MAX_DEPTH; i++) {
			if (i > 0)
				appendStringInfoString(&path_log, ", ");
			appendStringInfo(&path_log, "%d", path_nodes[i]);
			if (i < edge_count && i < GTREE_MAX_DEPTH)
				appendStringInfo(&path_log, "%c", path_dir[i]);
		}
		appendStringInfoChar(&path_log, ']');
		elog(DEBUG1,
			"random_forest: gtree path len=%d leaf_idx=%d path=%s",
			(edge_count < 0) ? 0 : (edge_count + 1), leaf_idx,
			path_log.data);
		pfree(path_log.data);
	}

	if (left_dist != NULL && right_dist != NULL && vec != NULL
		&& model != NULL && model->feature_limit > 0
		&& model->left_branch_means != NULL
		&& model->right_branch_means != NULL) {
		int limit = model->feature_limit;
		int f;
		const float *vec_data;
		double lsum = 0.0;
		double rsum = 0.0;

		if (vec->dim < limit)
			limit = vec->dim;
		if (model->n_features < limit)
			limit = model->n_features;
		if (limit > RF_STUB_MAX_FEATURES)
			limit = RF_STUB_MAX_FEATURES;

		vec_data = vec->data;
		for (f = 0; f < limit; f++) {
			double val = (double)vec_data[f];
			double ldiff = val - model->left_branch_means[f];
			double rdiff = val - model->right_branch_means[f];

			lsum += ldiff * ldiff;
			rsum += rdiff * rdiff;
		}

		*left_dist = sqrt(lsum);
		*right_dist = sqrt(rsum);
	}

	if (leaf_out)
		*leaf_out = leaf_idx;

	return result;
}

Datum predict_random_forest(PG_FUNCTION_ARGS) {
	int32 model_id;
	RFModel *model;
	Vector *feature_vec = NULL;
	double result;
	double split_z = 0.0;
	bool split_z_valid = false;
	const char *branch_name = "majority";
	double branch_fraction = 0.0;
	double branch_value = 0.0;
	double left_mean_dist = -1.0;
	double right_mean_dist = -1.0;
	int mean_limit = 0;
	double vote_majority = 0.0;
	double best_vote_fraction = 0.0;
	double second_vote_value = 0.0;
	double second_vote_fraction = 0.0;
	int vote_count = 0;
	double vote_total_weight = 0.0;
	int i;
	double *vote_histogram = NULL;
	int vote_classes = 0;
	double fallback_value = 0.0;
	double fallback_fraction = 0.0;
	int top_feature_idx = -1;
	double top_feature_importance = 0.0;

	if (!PG_ARGISNULL(0))
		model_id = PG_GETARG_INT32(0);
	else
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("random_forest: model ID is required")));

	if (!PG_ARGISNULL(1))
		feature_vec = PG_GETARG_VECTOR_P(1);
	else
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("random_forest: feature vector is "
				       "required")));

	if (!rf_lookup_model(model_id, &model)) {
		if (!rf_load_model_from_catalog(model_id, &model))
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("random_forest: model %d not "
					       "found",
						model_id)));
	}

	fallback_value = model->second_value;
	fallback_fraction = model->second_fraction;
	branch_fraction = model->majority_fraction;
	branch_value = model->majority_value;

	if (model->n_features > 0 && feature_vec != NULL
		&& feature_vec->dim != model->n_features)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("random_forest: feature dimension "
				       "mismatch "
				       "(expected %d got %d)",
					model->n_features, feature_vec->dim)));

	if (model->feature_means != NULL && feature_vec != NULL) {
		float *vec_data = feature_vec->data;
		double dist = 0.0;
		int j;
		double max_z = 0.0;

		for (j = 0; j < model->n_features && j < feature_vec->dim;
			j++) {
			double diff
				= (double)vec_data[j] - model->feature_means[j];
			dist += diff * diff;
			if (model->feature_variances != NULL) {
				double var = model->feature_variances[j];
				double z;

				if (var <= 0.0)
					continue;
				z = fabs(diff) / sqrt(var);
				if (z > max_z)
					max_z = z;
			}
		}
		dist = sqrt(dist);
		if (model->feature_variances != NULL)
			elog(DEBUG1,
				"random_forest: feature L2 distance %.3f max-z "
				"%.3f",
				dist, max_z);
		else
			elog(DEBUG1,
				"random_forest: feature L2 distance to mean "
				"%.3f",
				dist);
		if (model->feature_variances != NULL
			&& model->second_fraction > 0.0 && max_z > 1.5)
			elog(DEBUG1,
				"random_forest: deviation %.3f exceeds "
				"threshold, "
				"considering second class %.3f",
				max_z, model->second_value);
		model->max_deviation = max_z;
	} else if (!PG_ARGISNULL(1))
		model->max_deviation = 0.0;

	if (feature_vec != NULL && model->split_feature >= 0) {
		int sf = model->split_feature;

		if (sf < feature_vec->dim) {
			double value = (double)feature_vec->data[sf];

			if (value <= model->split_threshold) {
				branch_name = "left";
				branch_fraction = model->left_branch_fraction;
				branch_value = model->left_branch_value;
			} else {
				branch_name = "right";
				branch_fraction = model->right_branch_fraction;
				branch_value = model->right_branch_value;
			}
			if (branch_fraction <= 0.0)
				branch_fraction = model->majority_fraction;

			if (model->feature_variances != NULL
				&& sf < model->n_features) {
				double var = model->feature_variances[sf];

				if (var > 0.0) {
					split_z = (value - model->split_threshold)
						/ sqrt(var);
					if (fabs(split_z)
						> model->max_split_deviation)
						model->max_split_deviation
							= fabs(split_z);
					split_z_valid = true;
				}
			}
		}
	}

	if (model->n_classes > 0) {
		vote_classes = model->n_classes;
		vote_histogram
			= (double *)palloc0(sizeof(double) * vote_classes);
	}

	if (model->tree_count > 0 && model->trees != NULL) {
		for (i = 0; i < model->tree_count; i++) {
			const GTree *tree = model->trees[i];
			double tree_left = -1.0;
			double tree_right = -1.0;
			int leaf_idx = -1;
			double tree_result;
			double vote_weight = 1.0;

			tree_result = rf_tree_predict_single(tree, model,
				feature_vec, &tree_left, &tree_right,
				&leaf_idx);

			if (model->tree_oob_accuracy != NULL
				&& i < model->tree_count) {
				vote_weight = model->tree_oob_accuracy[i];
				if (vote_weight <= 0.0)
					vote_weight = 1.0;
			}

			if (vote_histogram != NULL) {
				int cls = (int)rint(tree_result);

				if (cls >= 0 && cls < vote_classes) {
					vote_histogram[cls] += vote_weight;
					vote_total_weight += vote_weight;
				}
			}

			if (tree_left >= 0.0 && tree_right >= 0.0) {
				if (mean_limit <= 0) {
					left_mean_dist = tree_left;
					right_mean_dist = tree_right;
					mean_limit = model->feature_limit;
				}
			}

			vote_count++;
		}
	} else {
		result = rf_tree_predict_single(model->tree, model, feature_vec,
			&left_mean_dist, &right_mean_dist, NULL);
		vote_count = 1;
		if (vote_histogram != NULL) {
			int cls = (int)rint(result);

			if (cls >= 0 && cls < vote_classes) {
				vote_histogram[cls] += 1.0;
				vote_total_weight += 1.0;
			}
		}
	}

	if (vote_histogram != NULL && vote_total_weight > 0.0) {
		int best_idx = -1;
		int second_idx = -1;
		double best_weight = -1.0;
		double second_weight = -1.0;

		for (i = 0; i < vote_classes; i++) {
			double weight = vote_histogram[i];

			if (weight > best_weight) {
				if (best_idx >= 0) {
					second_idx = best_idx;
					second_weight = best_weight;
				}
				best_idx = i;
				best_weight = weight;
			} else if (weight > second_weight) {
				second_idx = i;
				second_weight = weight;
			}
		}

		if (best_idx >= 0 && best_weight > 0.0) {
			vote_majority = (double)best_idx;
			best_vote_fraction = best_weight / vote_total_weight;
			result = vote_majority;
			branch_name = "forest";
			branch_fraction = best_vote_fraction;
			branch_value = vote_majority;
		}

		if (second_idx >= 0 && second_weight > 0.0) {
			second_vote_value = (double)second_idx;
			second_vote_fraction
				= second_weight / vote_total_weight;
		}

		pfree(vote_histogram);
		vote_histogram = NULL;
	}

	if (best_vote_fraction <= 0.0)
		result = model->majority_value;

	if (second_vote_fraction > 0.0) {
		fallback_value = second_vote_value;
		fallback_fraction = second_vote_fraction;
	}

	if (mean_limit > 0 && left_mean_dist >= 0.0 && right_mean_dist >= 0.0) {
		elog(DEBUG1,
			"random_forest: branch mean distances left=%.3f "
			"right=%.3f limit=%d",
			left_mean_dist, right_mean_dist, mean_limit);

		if (right_mean_dist + 0.10 < left_mean_dist
			&& model->right_branch_fraction > 0.0) {
			result = model->right_branch_value;
			branch_name = "right-mean";
			branch_fraction = model->right_branch_fraction;
			branch_value = model->right_branch_value;
		} else if (left_mean_dist + 0.10 < right_mean_dist
			&& model->left_branch_fraction > 0.0) {
			result = model->left_branch_value;
			branch_name = "left-mean";
			branch_fraction = model->left_branch_fraction;
			branch_value = model->left_branch_value;
		}
	}

	if (model->feature_variances != NULL && fallback_fraction > 0.0
		&& !PG_ARGISNULL(1) && model->max_deviation > 2.0
		&& model->label_entropy > 0.1) {
		elog(DEBUG1,
			"random_forest: high deviation %.3f -> returning "
			"second "
			"class %.3f",
			model->max_deviation, fallback_value);
		result = fallback_value;
		branch_name = "fallback";
		branch_fraction = fallback_fraction;
		branch_value = fallback_value;
	}

	if (result != model->majority_value)
		elog(NOTICE,
			"random_forest: majority=%.3f frac=%.3f, fallback=%.3f "
			"frac=%.3f branch=%s bfrac=%.3f entropy=%.3f "
			"split_dev=%.3f",
			model->majority_value, model->majority_fraction, result,
			fallback_fraction, branch_name, branch_fraction,
			model->label_entropy, model->max_split_deviation);

	if (model->split_feature >= 0) {
		if (split_z_valid)
			elog(DEBUG1,
				"random_forest: split f=%d thr=%.3f left=%.3f "
				"lf=%.3f right=%.3f rf=%.3f z=%.3f",
				model->split_feature, model->split_threshold,
				model->left_branch_value,
				model->left_branch_fraction,
				model->right_branch_value,
				model->right_branch_fraction, split_z);
		else
			elog(DEBUG1,
				"random_forest: split f=%d thr=%.3f left=%.3f "
				"lf=%.3f right=%.3f rf=%.3f",
				model->split_feature, model->split_threshold,
				model->left_branch_value,
				model->left_branch_fraction,
				model->right_branch_value,
				model->right_branch_fraction);
	}

	if (model->feature_importance != NULL && model->n_features > 0) {
		for (i = 0; i < model->n_features; i++) {
			double val = model->feature_importance[i];

			if (val > top_feature_importance) {
				top_feature_importance = val;
				top_feature_idx = i;
			}
		}
	}

	elog(DEBUG1,
		"predict_random_forest: returning %.3f (branch %s leaf %.3f "
		"frac "
		"%.3f majority %.3f frac %.3f gini %.3f ldist %.3f rdist %.3f "
		"topfeat=%d topimp=%.3f)",
		result, branch_name, branch_value, branch_fraction,
		model->majority_value, model->majority_fraction,
		model->gini_impurity, left_mean_dist, right_mean_dist,
		top_feature_idx, top_feature_importance);
	PG_RETURN_FLOAT8(result);
}

Datum evaluate_random_forest(PG_FUNCTION_ARGS) {
	Datum result_datums[4];
	ArrayType *result_array;
	int32 model_id;
	RFModel *model;
	double accuracy;
	double error_rate;
	int top_feature_idx = -1;
	double top_feature_importance = 0.0;

	if (PG_NARGS() < 1 || PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("random_forest: model_id required")));

	model_id = PG_GETARG_INT32(0);

	if (!rf_lookup_model(model_id, &model))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("random_forest: model %d not found",
					model_id)));

	if (model->oob_accuracy > 0.0)
		accuracy = model->oob_accuracy;
	else
		accuracy = model->majority_fraction;
	error_rate = (accuracy > 1.0) ? 0.0 : (1.0 - accuracy);

	if (model->feature_importance != NULL && model->n_features > 0) {
		int i;

		for (i = 0; i < model->n_features; i++) {
			double val = model->feature_importance[i];

			if (val > top_feature_importance) {
				top_feature_importance = val;
				top_feature_idx = i;
			}
		}
	}

	elog(NOTICE,
		"evaluate_random_forest: samples=%d classes=%d accuracy=%.3f "
		"gini=%.3f entropy=%.3f oob=%.3f majority=%.3f topfeat=%d "
		"topimp=%.3f",
		model->n_samples, model->n_classes, accuracy,
		model->gini_impurity, model->label_entropy, model->oob_accuracy,
		model->majority_fraction, top_feature_idx,
		top_feature_importance);

	result_datums[0] = Float8GetDatum(accuracy);
	result_datums[1] = Float8GetDatum(error_rate);
	result_datums[2] = Float8GetDatum(model->gini_impurity);
	result_datums[3] = Float8GetDatum((double)model->n_classes);

	result_array = construct_array(
		result_datums, 4, FLOAT8OID, sizeof(float8), true, 'd');

	PG_RETURN_ARRAYTYPE_P(result_array);
}

static void rf_serialize_tree(StringInfo buf, const GTree *tree) {
	const GTreeNode *nodes;
	int i;

	if (tree == NULL) {
		pq_sendbyte(buf, 0);
		return;
	}

	pq_sendbyte(buf, 1);
	pq_sendint32(buf, tree->root);
	pq_sendint32(buf, tree->max_depth);
	pq_sendint32(buf, tree->count);

	nodes = gtree_nodes(tree);
	for (i = 0; i < tree->count; i++) {
		pq_sendint32(buf, nodes[i].feature_idx);
		pq_sendfloat8(buf, nodes[i].threshold);
		pq_sendint32(buf, nodes[i].left);
		pq_sendint32(buf, nodes[i].right);
		pq_sendbyte(buf, nodes[i].is_leaf ? 1 : 0);
		pq_sendfloat8(buf, nodes[i].value);
	}
}

static GTree *rf_deserialize_tree(StringInfo buf) {
	int flag = pq_getmsgbyte(buf);
	int count;
	int i;
	int root;
	int max_depth;
	GTree *tree;
	MemoryContext oldctx;

	if (flag == 0)
		return NULL;

	root = pq_getmsgint(buf, 4);
	max_depth = pq_getmsgint(buf, 4);
	count = pq_getmsgint(buf, 4);

	tree = gtree_create("rf_model_tree", Max(count, 4));
	oldctx = MemoryContextSwitchTo(tree->ctx);

	if (tree->nodes != NULL)
		pfree(tree->nodes);

	if (count > 0) {
		tree->nodes = (GTreeNode *)palloc(sizeof(GTreeNode) * count);
		for (i = 0; i < count; i++) {
			tree->nodes[i].feature_idx = pq_getmsgint(buf, 4);
			tree->nodes[i].threshold = pq_getmsgfloat8(buf);
			tree->nodes[i].left = pq_getmsgint(buf, 4);
			tree->nodes[i].right = pq_getmsgint(buf, 4);
			tree->nodes[i].is_leaf = pq_getmsgbyte(buf);
			tree->nodes[i].value = pq_getmsgfloat8(buf);
		}
	} else
		tree->nodes = NULL;

	tree->capacity = count;
	tree->count = count;
	tree->root = root;
	tree->max_depth = max_depth;

	MemoryContextSwitchTo(oldctx);
	return tree;
}

static void rf_write_int_array(StringInfo buf, const int *values, int count) {
	int i;

	if (values == NULL || count <= 0) {
		pq_sendbyte(buf, 0);
		return;
	}

	pq_sendbyte(buf, 1);
	pq_sendint32(buf, count);
	for (i = 0; i < count; i++)
		pq_sendint32(buf, values[i]);
}

static void rf_write_double_array(
	StringInfo buf, const double *values, int count) {
	int i;

	if (values == NULL || count <= 0) {
		pq_sendbyte(buf, 0);
		return;
	}

	pq_sendbyte(buf, 1);
	pq_sendint32(buf, count);
	for (i = 0; i < count; i++)
		pq_sendfloat8(buf, values[i]);
}

static int *rf_read_int_array(StringInfo buf, int expected_count) {
	int flag = pq_getmsgbyte(buf);
	int len;
	int *result;
	int i;

	if (flag == 0)
		return NULL;

	len = pq_getmsgint(buf, 4);
	if (expected_count >= 0 && len != expected_count)
		elog(ERROR,
			"random_forest: unexpected int array length %d "
			"(expected "
			"%d)",
			len, expected_count);

	result = (int *)palloc(sizeof(int) * len);
	for (i = 0; i < len; i++)
		result[i] = pq_getmsgint(buf, 4);

	return result;
}

static double *rf_read_double_array(StringInfo buf, int expected_count) {
	int flag = pq_getmsgbyte(buf);
	int len;
	double *result;
	int i;

	if (flag == 0)
		return NULL;

	len = pq_getmsgint(buf, 4);
	if (expected_count >= 0 && len != expected_count)
		elog(ERROR,
			"random_forest: unexpected double array length %d "
			"(expected %d)",
			len, expected_count);

	result = (double *)palloc(sizeof(double) * len);
	for (i = 0; i < len; i++)
		result[i] = pq_getmsgfloat8(buf);

	return result;
}

static bytea *rf_model_serialize(const RFModel *model) {
	StringInfoData buf;
	int i;

	pq_begintypsend(&buf);

	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendint32(&buf, model->n_classes);

	pq_sendfloat8(&buf, model->majority_value);
	pq_sendfloat8(&buf, model->majority_fraction);
	pq_sendfloat8(&buf, model->gini_impurity);
	pq_sendfloat8(&buf, model->label_entropy);
	pq_sendfloat8(&buf, model->max_deviation);
	pq_sendfloat8(&buf, model->max_split_deviation);
	pq_sendint32(&buf, model->split_feature);
	pq_sendfloat8(&buf, model->split_threshold);
	pq_sendfloat8(&buf, model->second_value);
	pq_sendfloat8(&buf, model->second_fraction);
	pq_sendfloat8(&buf, model->left_branch_value);
	pq_sendfloat8(&buf, model->left_branch_fraction);
	pq_sendfloat8(&buf, model->right_branch_value);
	pq_sendfloat8(&buf, model->right_branch_fraction);
	pq_sendint32(&buf, model->feature_limit);
	pq_sendfloat8(&buf, model->oob_accuracy);

	rf_write_int_array(&buf, model->class_counts, model->n_classes);
	rf_write_double_array(&buf, model->feature_means, model->n_features);
	rf_write_double_array(
		&buf, model->feature_variances, model->n_features);
	rf_write_double_array(
		&buf, model->feature_importance, model->n_features);
	rf_write_double_array(
		&buf, model->left_branch_means, model->feature_limit);
	rf_write_double_array(
		&buf, model->right_branch_means, model->feature_limit);

	pq_sendint32(&buf, model->tree_count);
	for (i = 0; i < model->tree_count; i++)
		rf_serialize_tree(&buf, model->trees[i]);

	rf_serialize_tree(&buf, model->tree);

	rf_write_double_array(&buf, model->tree_majority, model->tree_count);
	rf_write_double_array(
		&buf, model->tree_majority_fraction, model->tree_count);
	rf_write_double_array(&buf, model->tree_second, model->tree_count);
	rf_write_double_array(
		&buf, model->tree_second_fraction, model->tree_count);
	rf_write_double_array(
		&buf, model->tree_oob_accuracy, model->tree_count);

	return pq_endtypsend(&buf);
}

static RFModel *rf_model_deserialize(const bytea *data) {
	StringInfoData buf;
	RFModel *model;

	if (data == NULL)
		return NULL;

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.cursor = 0;

	model = (RFModel *)palloc0(sizeof(RFModel));

	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->n_classes = pq_getmsgint(&buf, 4);

	model->majority_value = pq_getmsgfloat8(&buf);
	model->majority_fraction = pq_getmsgfloat8(&buf);
	model->gini_impurity = pq_getmsgfloat8(&buf);
	model->label_entropy = pq_getmsgfloat8(&buf);
	model->max_deviation = pq_getmsgfloat8(&buf);
	model->max_split_deviation = pq_getmsgfloat8(&buf);
	model->split_feature = pq_getmsgint(&buf, 4);
	model->split_threshold = pq_getmsgfloat8(&buf);
	model->second_value = pq_getmsgfloat8(&buf);
	model->second_fraction = pq_getmsgfloat8(&buf);
	model->left_branch_value = pq_getmsgfloat8(&buf);
	model->left_branch_fraction = pq_getmsgfloat8(&buf);
	model->right_branch_value = pq_getmsgfloat8(&buf);
	model->right_branch_fraction = pq_getmsgfloat8(&buf);
	model->feature_limit = pq_getmsgint(&buf, 4);
	model->oob_accuracy = pq_getmsgfloat8(&buf);

	model->class_counts = rf_read_int_array(&buf, model->n_classes);
	model->feature_means = rf_read_double_array(&buf, model->n_features);
	model->feature_variances
		= rf_read_double_array(&buf, model->n_features);
	model->feature_importance
		= rf_read_double_array(&buf, model->n_features);
	model->left_branch_means
		= rf_read_double_array(&buf, model->feature_limit);
	model->right_branch_means
		= rf_read_double_array(&buf, model->feature_limit);

	model->tree_count = pq_getmsgint(&buf, 4);
	if (model->tree_count > 0) {
		int i;

		model->trees
			= (GTree **)palloc(sizeof(GTree *) * model->tree_count);
		for (i = 0; i < model->tree_count; i++)
			model->trees[i] = rf_deserialize_tree(&buf);
	} else
		model->trees = NULL;

	model->tree = rf_deserialize_tree(&buf);

	model->tree_majority = rf_read_double_array(&buf, model->tree_count);
	model->tree_majority_fraction
		= rf_read_double_array(&buf, model->tree_count);
	model->tree_second = rf_read_double_array(&buf, model->tree_count);
	model->tree_second_fraction
		= rf_read_double_array(&buf, model->tree_count);
	model->tree_oob_accuracy
		= rf_read_double_array(&buf, model->tree_count);

	return model;
}

static void rf_free_deserialized_model(RFModel *model) {
	if (model == NULL)
		return;

	if (model->class_counts)
		pfree(model->class_counts);
	if (model->feature_means)
		pfree(model->feature_means);
	if (model->feature_variances)
		pfree(model->feature_variances);
	if (model->feature_importance)
		pfree(model->feature_importance);
	if (model->left_branch_means)
		pfree(model->left_branch_means);
	if (model->right_branch_means)
		pfree(model->right_branch_means);
	if (model->tree_majority)
		pfree(model->tree_majority);
	if (model->tree_majority_fraction)
		pfree(model->tree_majority_fraction);
	if (model->tree_second)
		pfree(model->tree_second);
	if (model->tree_second_fraction)
		pfree(model->tree_second_fraction);
	if (model->tree_oob_accuracy)
		pfree(model->tree_oob_accuracy);
	if (model->trees)
		pfree(model->trees);

	pfree(model);
}

static bool rf_update_model_identifier(
	int32 internal_id, int32 new_id, RFModel **out_model) {
	int i;

	for (i = 0; i < rf_model_count; i++) {
		if (rf_models[i].model_id == internal_id) {
			rf_models[i].model_id = new_id;
			if (out_model)
				*out_model = &rf_models[i];
			return true;
		}
	}
	return false;
}

static bool rf_load_model_from_catalog(int32 model_id, RFModel **out) {
	StringInfoData sql;
	int ret;
	bool isnull;
	Datum datum;
	bytea *payload;
	RFModel *decoded;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR,
			"random_forest: SPI_connect failed while loading model "
			"%d",
			model_id);

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT model_data FROM neurondb.ml_models WHERE model_id = %d",
		model_id);

	ret = SPI_execute(sql.data, true, 1);
	pfree(sql.data);

	if (ret != SPI_OK_SELECT || SPI_processed == 0) {
		SPI_finish();
		return false;
	}

	datum = SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
	if (isnull) {
		SPI_finish();
		return false;
	}

	payload = DatumGetByteaP(datum);
	decoded = rf_model_deserialize(payload);

	pfree(payload);
	SPI_finish();

	if (decoded == NULL)
		return false;

	rf_store_model(model_id, decoded->n_features, decoded->n_samples,
		decoded->n_classes, decoded->majority_value,
		decoded->majority_fraction, decoded->gini_impurity,
		decoded->label_entropy, decoded->class_counts,
		decoded->feature_means, decoded->feature_variances,
		decoded->feature_importance, decoded->tree,
		decoded->split_feature, decoded->split_threshold,
		decoded->second_value, decoded->second_fraction,
		decoded->left_branch_value, decoded->left_branch_fraction,
		decoded->right_branch_value, decoded->right_branch_fraction,
		decoded->max_deviation, decoded->max_split_deviation,
		decoded->feature_limit, decoded->left_branch_means,
		decoded->right_branch_means, decoded->tree_count,
		decoded->trees, decoded->tree_majority,
		decoded->tree_majority_fraction, decoded->tree_second,
		decoded->tree_second_fraction, decoded->tree_oob_accuracy,
		decoded->oob_accuracy);

	rf_free_deserialized_model(decoded);

	if (out != NULL)
		return rf_lookup_model(model_id, out);

	return true;
}

static void rf_save_model_catalog(
	int32 catalog_id, const RFModel *model, const char *hyperparams_json) {
	bytea *payload;
	StringInfoData metrics_json;
	Oid argtypes[4] = { BYTEAOID, JSONBOID, JSONBOID, INT4OID };
	Datum values[4];
	char nulls[4] = { ' ', ' ', ' ', ' ' };
	int ret;

	payload = rf_model_serialize(model);

	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
		"{\"oob_accuracy\":%.6f,\"majority_fraction\":%.6f,"
		"\"gini\":%.6f}",
		model->oob_accuracy, model->majority_fraction,
		model->gini_impurity);

	values[0] = PointerGetDatum(payload);
	if (hyperparams_json != NULL)
		values[1] = DirectFunctionCall1(
			jsonb_in, CStringGetDatum((char *)hyperparams_json));
	else
		nulls[1] = 'n';
	values[2] = DirectFunctionCall1(
		jsonb_in, CStringGetDatum(metrics_json.data));
	values[3] = Int32GetDatum(catalog_id);

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR,
			"random_forest: SPI_connect failed while saving model "
			"%d",
			catalog_id);

	ret = SPI_execute_with_args("UPDATE neurondb.ml_models "
				    "SET model_data = $1, "
				    "    hyperparameters = CASE WHEN $2 IS "
				    "NULL THEN hyperparameters ELSE $2 END, "
				    "    metrics = $3, "
				    "    status = 'trained', "
				    "    updated_at = NOW() "
				    "WHERE model_id = $4",
		4, argtypes, values, nulls, false, 0);

	if (ret != SPI_OK_UPDATE)
		elog(ERROR,
			"random_forest: failed to update catalog entry for id "
			"%d",
			catalog_id);

	SPI_finish();

	pfree(metrics_json.data);
	pfree(DatumGetPointer(values[2]));
	if (nulls[1] == ' ')
		pfree(DatumGetPointer(values[1]));
	pfree(payload);
}

static void rf_promote_model_internal(
	int32 internal_id, int32 catalog_id, const char *hyperparams_json) {
	RFModel *model = NULL;

	if (!rf_update_model_identifier(internal_id, catalog_id, &model))
		elog(ERROR, "random_forest: internal model %d not found",
			internal_id);

	rf_save_model_catalog(catalog_id, model, hyperparams_json);
}

void rf_promote_model_to_catalog(
	int32 internal_id, int32 catalog_id, const char *hyperparams_json) {
	rf_promote_model_internal(internal_id, catalog_id, hyperparams_json);
}
