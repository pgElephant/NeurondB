
/*
 * ml_random_forest.c
 *    Refactored Random Forest implementation for classification (PostgreSQL C coding standard)
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

#include <math.h>
#include <string.h>

#include "neurondb.h"
#include "neurondb_ml.h"
#include "gtree.h"

PG_FUNCTION_INFO_V1(train_random_forest_classifier);
PG_FUNCTION_INFO_V1(predict_random_forest);
PG_FUNCTION_INFO_V1(evaluate_random_forest);

typedef struct RFStubModel
{
	int32		model_id;			/* model identifier */
	int			n_features;			/* number of features */
	int			n_samples;			/* number of samples */
	int			n_classes;			/* number of classes */
	double		majority_value;		/* most frequent class value */
	double		majority_fraction;	/* fraction of majority class */
	double		gini_impurity;		/* Gini impurity for root */
	int		   *class_counts;		/* class histogram */
	double	   *feature_means;		/* mean for each feature */
	double	   *feature_variances;	/* variance for each feature */
	GTree	   *tree;				/* tree structure pointer */
	int			split_feature;		/* feature index for split */
	double		split_threshold;	/* threshold used for split */
	double		second_value;		/* value for 2nd class by frequency */
	double		second_fraction;	/* fraction for 2nd class by frequency */
	double		label_entropy;		/* entropy of labels at root */
	double		max_deviation;		/* max deviation among features */
	double		max_split_deviation;/* max deviation at split */
	double		left_branch_value;	/* predicted value for left branch */
	double		left_branch_fraction;/* fraction for left branch */
	double		right_branch_value;	/* predicted value for right branch */
	double		right_branch_fraction;/* fraction for right branch */
} RFStubModel;


static RFStubModel *rf_stub_models = NULL;
static int rf_stub_model_count = 0;
static int32 rf_stub_next_model_id = 1;

static void
rf_stub_store_model(int32 model_id, int n_features, int n_samples, int n_classes,
	 double majority, double fraction, double gini, double entropy,
	 const int *class_counts,
	 const double *feature_means, const double *feature_variances, GTree *tree,
	 int split_feature, double split_threshold, double second_value,
	 double second_fraction, double left_value, double left_fraction,
	 double right_value, double right_fraction,
	 double max_deviation, double max_split_deviation)
{
	MemoryContext oldctx = MemoryContextSwitchTo(TopMemoryContext);
	int i;

	if (rf_stub_model_count == 0)
		rf_stub_models = (RFStubModel *) palloc(sizeof(RFStubModel));
	else
		rf_stub_models = (RFStubModel *) repalloc(rf_stub_models,
			sizeof(RFStubModel) * (rf_stub_model_count + 1));

	rf_stub_models[rf_stub_model_count].model_id = model_id;
	rf_stub_models[rf_stub_model_count].n_features = n_features;
	rf_stub_models[rf_stub_model_count].n_samples = n_samples;
	rf_stub_models[rf_stub_model_count].n_classes = n_classes;
	rf_stub_models[rf_stub_model_count].majority_value = majority;
	rf_stub_models[rf_stub_model_count].majority_fraction = fraction;
	rf_stub_models[rf_stub_model_count].gini_impurity = gini;
	rf_stub_models[rf_stub_model_count].label_entropy = entropy;

	rf_stub_models[rf_stub_model_count].class_counts = NULL;
	if (n_classes > 0 && class_counts != NULL)
	{
		int *copy = (int *) palloc(sizeof(int) * n_classes);

		memcpy(copy, class_counts, sizeof(int) * n_classes);
		rf_stub_models[rf_stub_model_count].class_counts = copy;
	}

	rf_stub_models[rf_stub_model_count].feature_means = NULL;
	if (n_features > 0 && feature_means != NULL)
	{
		double *means_copy = (double *) palloc(sizeof(double) * n_features);

		for (i = 0; i < n_features; i++)
			means_copy[i] = feature_means[i];
		rf_stub_models[rf_stub_model_count].feature_means = means_copy;
	}

	rf_stub_models[rf_stub_model_count].feature_variances = NULL;
	if (n_features > 0 && feature_variances != NULL)
	{
		double *vars_copy = (double *) palloc(sizeof(double) * n_features);

		for (i = 0; i < n_features; i++)
			vars_copy[i] = feature_variances[i];
		rf_stub_models[rf_stub_model_count].feature_variances = vars_copy;
	}

	rf_stub_models[rf_stub_model_count].tree = tree;
	rf_stub_models[rf_stub_model_count].split_feature = split_feature;
	rf_stub_models[rf_stub_model_count].split_threshold = split_threshold;
	rf_stub_models[rf_stub_model_count].second_value = second_value;
	rf_stub_models[rf_stub_model_count].second_fraction = second_fraction;
	rf_stub_models[rf_stub_model_count].left_branch_value = left_value;
	rf_stub_models[rf_stub_model_count].left_branch_fraction = left_fraction;
	rf_stub_models[rf_stub_model_count].right_branch_value = right_value;
	rf_stub_models[rf_stub_model_count].right_branch_fraction = right_fraction;
	rf_stub_models[rf_stub_model_count].max_deviation = max_deviation;
	rf_stub_models[rf_stub_model_count].max_split_deviation = max_split_deviation;

	rf_stub_model_count++;

	MemoryContextSwitchTo(oldctx);
}

static bool
rf_stub_lookup_model(int32 model_id, RFStubModel **out)
{
	int i;

	for (i = 0; i < rf_stub_model_count; i++)
	{
		if (rf_stub_models[i].model_id == model_id)
		{
			if (out)
				*out = &rf_stub_models[i];
			return true;
		}
	}
	return false;
}

static int
rf_count_classes(double *labels, int n_samples)
{
	int		max_class = -1;
	int		i;

	if (n_samples <= 0)
		return 0;

	for (i = 0; i < n_samples; i++)
	{
		double val = labels[i];
		int as_int;

		if (!isfinite(val))
			continue;

		as_int = (int) rint(val);
		if (as_int < 0)
			continue;

		if (as_int > max_class)
			max_class = as_int;
	}

	return (max_class < 0) ? 0 : (max_class + 1);
}

Datum
train_random_forest_classifier(PG_FUNCTION_ARGS)
{
	text		*table_name_text;
	text		*feature_col_text;
	text		*label_col_text;

	char		*table_name;
	char		*feature_col;
	char		*label_col;
	const char	*quoted_tbl;
	const char	*quoted_feat;
	const char	*quoted_label;

	StringInfoData	query;
	int		feature_dim		= 0;
	double	   *labels		= NULL;
	int		n_classes		= 0;
	double		majority_value		= 0.0;
	double		majority_fraction	= 0.0;
	double		gini_impurity		= 0.0;
	double		label_entropy		= 0.0;
	int		majority_count		= 0;
	int		second_count		= 0;
	int		second_idx		= -1;
	double		second_value		= 0.0;
	int	      *class_counts_tmp		= NULL;
	double   *feature_means_tmp	= NULL;
	double   *feature_vars_tmp	= NULL;
	double   *feature_sums		= NULL;
	double   *feature_sums_sq	= NULL;
double   *class_feature_sums0 = NULL;
int      *class_feature_counts0 = NULL;
	int	feature_sum_count	= 0;
	GTree   *stub_tree		= NULL;
	int	split_feature		= -1;
	double	split_threshold	= 0.0;
	double	second_fraction	= 0.0;
	double	max_deviation		= 0.0;
	double	max_split_deviation	= 0.0;
	int    *left_counts		= NULL;
	int    *right_counts	= NULL;
	int	left_majority_idx	= -1;
	int	right_majority_idx	= -1;
	int	left_total		= 0;
	int	right_total		= 0;
	double	left_leaf_value	= 0.0;
	double	right_leaf_value	= 0.0;
	double	left_sum		= 0.0;
	double	right_sum		= 0.0;
	bool	branch_threshold_valid = false;
	double	branch_threshold	= 0.0;
	double	left_branch_fraction	= 0.0;
	double	right_branch_fraction	= 0.0;
int	majority_idx		= -1;
double	class_majority_mean	= 0.0;
double	class_second_mean	= 0.0;
double	class_mean_threshold	= 0.0;
bool	class_mean_threshold_valid = false;
	int32	model_id;
	int	n_samples		= 0;

	if (PG_NARGS() < 3)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			errmsg("random_forest: requires table, feature column, and label column")));

	table_name_text = PG_GETARG_TEXT_PP(0);
	feature_col_text = PG_GETARG_TEXT_PP(1);
	label_col_text = PG_GETARG_TEXT_PP(2);

	table_name = text_to_cstring(table_name_text);
	feature_col = text_to_cstring(feature_col_text);
	label_col = text_to_cstring(label_col_text);
	quoted_tbl = quote_identifier(table_name);
	quoted_feat = quote_identifier(feature_col);
	quoted_label = quote_identifier(label_col);

	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT vector_dims(%s) FROM %s WHERE %s IS NOT NULL LIMIT 1",
		quoted_feat,
		quoted_tbl,
		quoted_feat);

	if (SPI_connect() != SPI_OK_CONNECT)
	{
		pfree(table_name);
		pfree(feature_col);
		pfree(label_col);
		pfree(query.data);
		ereport(ERROR, (errmsg("random_forest: SPI_connect failed")));
	}

	if (SPI_execute(query.data, true, 1) == SPI_OK_SELECT && SPI_processed > 0)
	{
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
		"SELECT %s, (%s)::float8 FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quoted_feat,
		quoted_label,
		quoted_tbl,
		quoted_feat,
		quoted_label);

	if (SPI_execute(query.data, true, 0) != SPI_OK_SELECT)
	{
		SPI_finish();
		pfree(table_name);
		pfree(feature_col);
		pfree(label_col);
		pfree(query.data);
		ereport(ERROR, (errmsg("random_forest: failed to fetch training data")));
	}

	n_samples = SPI_processed;
	if (n_samples > 0)
	{
		int i;

		labels = (double *) palloc(sizeof(double) * n_samples);
		if (feature_dim > 0)
		{
			feature_sums = (double *) palloc0(sizeof(double) * feature_dim);
			feature_sums_sq = (double *) palloc0(sizeof(double) * feature_dim);
		}

		for (i = 0; i < n_samples; i++)
		{
			HeapTuple tup = SPI_tuptable->vals[i];
			TupleDesc tupdesc = SPI_tuptable->tupdesc;
			Datum feat_datum;
			Datum label_datum;
			bool feat_null;
			bool label_null;

			feat_datum = SPI_getbinval(tup, tupdesc, 1, &feat_null);
			label_datum = SPI_getbinval(tup, tupdesc, 2, &label_null);

			if (feat_null || label_null)
			{
				labels[i] = NAN;
				continue;
			}

			labels[i] = DatumGetFloat8(label_datum);

			if (feature_sums != NULL)
			{
				Vector *vec;
				float  *vec_data;
				int j;

				vec = DatumGetVector(feat_datum);
				if (vec->dim != feature_dim)
					continue;

				vec_data = vec->data;
				for (j = 0; j < feature_dim; j++)
				{
					feature_sums[j] += (double) vec_data[j];
					feature_sums_sq[j] += ((double) vec_data[j]) * ((double) vec_data[j]);
				}
				feature_sum_count++;
			}
		}

		n_classes = rf_count_classes(labels, n_samples);

		if (n_classes > 0)
		{
			int *counts = (int *) palloc0(sizeof(int) * n_classes);
			int best_idx = 0;

			class_counts_tmp = counts;

			for (i = 0; i < n_samples; i++)
			{
				int idx = (int) rint(labels[i]);
				if (idx < 0 || idx >= n_classes)
					continue;
				counts[idx]++;
				if (counts[idx] > counts[best_idx])
				{
					if (idx != best_idx)
					{
						second_idx = best_idx;
						second_count = counts[best_idx];
						second_value = (double) best_idx;
					}
					best_idx = idx;
				}
				else if (idx != best_idx && counts[idx] > second_count)
				{
					second_idx = idx;
					second_count = counts[idx];
					second_value = (double) idx;
				}
			}

			majority_value = (double) best_idx;
			majority_count = counts[best_idx];
			majority_idx = best_idx;
			if (second_idx < 0 && n_classes > 1)
			{
				for (i = 0; i < n_classes; i++)
				{
					if (i == best_idx)
						continue;
					if (counts[i] >= second_count)
					{
						second_idx = i;
						second_count = counts[i];
						second_value = (double) i;
					}
				}
			}

			left_leaf_value = majority_value;
			right_leaf_value = (second_idx >= 0) ? second_value : majority_value;

			if (n_samples > 0)
			{
				double sum_sq = 0.0;
				double entropy = 0.0;
				int c;
				double ln2 = log(2.0);

				for (c = 0; c < n_classes; c++)
				{
					double p = (double) counts[c] / (double) n_samples;
					sum_sq += p * p;
					if (p > 0.0)
						entropy -= p * (log(p) / ln2);
				}
				gini_impurity = 1.0 - sum_sq;
				label_entropy = entropy;
			}

			if (class_counts_tmp)
			{
				StringInfoData histogram;
				initStringInfo(&histogram);
				appendStringInfo(&histogram, "[");
				for (i = 0; i < n_classes; i++)
				{
					if (i > 0)
						appendStringInfoString(&histogram, ", ");
					appendStringInfo(&histogram, "%d", class_counts_tmp[i]);
				}
				appendStringInfoChar(&histogram, ']');
				elog(DEBUG1, "random_forest stub: class histogram %s", histogram.data);
				pfree(histogram.data);
			}
		}

		if (feature_sums != NULL && feature_sum_count > 0)
		{
			int j;
			StringInfoData mean_log;
			StringInfoData var_log;

			feature_means_tmp = (double *) palloc(sizeof(double) * feature_dim);
			feature_vars_tmp = (double *) palloc(sizeof(double) * feature_dim);
			for (j = 0; j < feature_dim; j++)
			{
				double mean = feature_sums[j] / (double) feature_sum_count;
				double mean_sq = feature_sums_sq[j] / (double) feature_sum_count;
				double variance = mean_sq - (mean * mean);

				if (variance < 0.0)
					variance = 0.0;

				feature_means_tmp[j] = mean;
				feature_vars_tmp[j] = variance;
			}

			initStringInfo(&mean_log);
			appendStringInfo(&mean_log, "[");
			for (j = 0; j < feature_dim && j < 5; j++)
			{
				if (j > 0)
					appendStringInfoString(&mean_log, ", ");
				appendStringInfo(&mean_log, "%.3f", feature_means_tmp[j]);
			}
			if (feature_dim > 5)
				appendStringInfoString(&mean_log, ", ...");
			appendStringInfoChar(&mean_log, ']');
			elog(DEBUG1, "random_forest stub: feature means %s", mean_log.data);
			pfree(mean_log.data);

			initStringInfo(&var_log);
			appendStringInfo(&var_log, "[");
			for (j = 0; j < feature_dim && j < 5; j++)
			{
				if (j > 0)
					appendStringInfoString(&var_log, ", ");
				appendStringInfo(&var_log, "%.3f", feature_vars_tmp[j]);
			}
			if (feature_dim > 5)
				appendStringInfoString(&var_log, ", ...");
			appendStringInfoChar(&var_log, ']');
			elog(DEBUG1, "random_forest stub: feature variances %s", var_log.data);
			pfree(var_log.data);
		}

		if (feature_dim > 0 && feature_means_tmp != NULL && n_classes > 0)
		{
			int sf = 0;
			double threshold = feature_means_tmp[sf];
			TupleDesc tupdesc = SPI_tuptable->tupdesc;
			Vector *vec;
			float  *vec_data;
			Datum feat_datum;
			bool feat_null;

			left_total = 0;
			right_total = 0;
			left_sum = 0.0;
			right_sum = 0.0;
			left_majority_idx = -1;
			right_majority_idx = -1;

			left_counts = (int *) palloc0(sizeof(int) * n_classes);
			right_counts = (int *) palloc0(sizeof(int) * n_classes);

			for (i = 0; i < n_samples; i++)
			{
				int cls;

				if (!isfinite(labels[i]))
					continue;
				cls = (int) rint(labels[i]);
				if (cls < 0 || cls >= n_classes)
					continue;

				feat_datum = SPI_getbinval(SPI_tuptable->vals[i], tupdesc, 1, &feat_null);
				if (feat_null)
					continue;

				vec = DatumGetVector(feat_datum);
				if (vec->dim <= sf)
					continue;

				vec_data = vec->data;
				if ((double) vec_data[sf] <= threshold)
				{
					left_counts[cls]++;
					left_total++;
					left_sum += (double) vec_data[sf];
				}
				else
				{
					right_counts[cls]++;
					right_total++;
					right_sum += (double) vec_data[sf];
				}
			}

			for (i = 0; i < n_classes; i++)
			{
				if (left_total > 0 &&
				    (left_majority_idx < 0 || left_counts[i] > left_counts[left_majority_idx]))
					left_majority_idx = i;
				if (right_total > 0 &&
				    (right_majority_idx < 0 || right_counts[i] > right_counts[right_majority_idx]))
					right_majority_idx = i;
			}

			if (left_majority_idx >= 0)
				left_leaf_value = (double) left_majority_idx;

			if (right_majority_idx >= 0)
			{
				right_leaf_value = (double) right_majority_idx;
				second_value = right_leaf_value;
				if (class_counts_tmp != NULL)
					second_fraction = ((double) class_counts_tmp[right_majority_idx]) /
						(double) n_samples;
				else if (right_total > 0)
					second_fraction = ((double) right_counts[right_majority_idx]) /
						(double) n_samples;
			}

			if (n_samples > 0)
			{
				if (left_total > 0)
					left_branch_fraction = ((double) left_total) / (double) n_samples;
				if (right_total > 0)
					right_branch_fraction = ((double) right_total) / (double) n_samples;
			}

			if ((left_total == 0 || right_total == 0) &&
				feature_vars_tmp != NULL &&
				feature_vars_tmp[sf] > 0.0)
			{
				double adjust;

				adjust = 0.5 * sqrt(feature_vars_tmp[sf]);
				if (right_total == 0)
				{
					threshold = feature_means_tmp[sf] + adjust;
					branch_threshold = threshold;
					branch_threshold_valid = true;
					if (right_branch_fraction <= 0.0)
						right_branch_fraction = 0.5;
					if (left_branch_fraction <= 0.0)
						left_branch_fraction = 1.0 - right_branch_fraction;
					right_leaf_value = (second_idx >= 0) ? second_value : majority_value;
				}
				else if (left_total == 0)
				{
					threshold = feature_means_tmp[sf] - adjust;
					branch_threshold = threshold;
					branch_threshold_valid = true;
					if (left_branch_fraction <= 0.0)
						left_branch_fraction = 0.5;
					if (right_branch_fraction <= 0.0)
						right_branch_fraction = 1.0 - left_branch_fraction;
					left_leaf_value = majority_value;
				}
			}

			if (left_total > 0 && right_total > 0)
			{
				double left_mean = left_sum / (double) left_total;
				double right_mean = right_sum / (double) right_total;

				threshold = 0.5 * (left_mean + right_mean);
				branch_threshold = threshold;
				branch_threshold_valid = true;
			}

			if (left_branch_fraction <= 0.0 && right_branch_fraction <= 0.0)
				left_branch_fraction = majority_fraction;
			else if (left_branch_fraction <= 0.0 && right_branch_fraction > 0.0)
				left_branch_fraction = 1.0 - right_branch_fraction;
			else if (right_branch_fraction <= 0.0 && left_branch_fraction > 0.0)
				right_branch_fraction = 1.0 - left_branch_fraction;

			if (second_fraction <= 0.0 && right_branch_fraction > 0.0)
				second_fraction = right_branch_fraction;

			if (left_counts)
				pfree(left_counts);
			if (right_counts)
				pfree(right_counts);
			left_counts = NULL;
			right_counts = NULL;

			elog(DEBUG1,
				"random_forest stub: branch totals left=%d right=%d split=%.3f lf=%.3f rf=%.3f",
				left_total,
				right_total,
				threshold,
				left_branch_fraction,
				right_branch_fraction);
		}

		if (n_samples > 0 && majority_count > 0)
			majority_fraction = ((double) majority_count) / (double) n_samples;
	}

	if (n_samples > 0 && second_count > 0 && second_fraction <= 0.0)
		second_fraction = ((double) second_count) / (double) n_samples;

	if (majority_count > 0)
	{
		MemoryContext	oldctx;
		int		node_idx;
		int		left_idx;
		int		right_idx;

		oldctx = MemoryContextSwitchTo(TopMemoryContext);
		stub_tree = gtree_create("rf_stub_tree", 4);
		MemoryContextSwitchTo(oldctx);

		if (stub_tree != NULL)
		{
			if (feature_dim > 0 && feature_means_tmp != NULL)
			{
				double var0 = (feature_vars_tmp != NULL) ? feature_vars_tmp[0] : 0.0;
				double pivot;

				split_feature = 0;
				pivot = branch_threshold_valid ? branch_threshold : feature_means_tmp[0];
				split_threshold = pivot;
				node_idx = gtree_add_split(stub_tree, split_feature, split_threshold);
				left_idx = gtree_add_leaf(stub_tree, left_leaf_value);
				right_idx = gtree_add_leaf(stub_tree, right_leaf_value);
				gtree_set_child(stub_tree, node_idx, left_idx, true);
				gtree_set_child(stub_tree, node_idx, right_idx, false);
				gtree_set_root(stub_tree, node_idx);
				if (var0 > 0.0)
					max_split_deviation = fabs(split_threshold) / sqrt(var0);
				elog(DEBUG1,
					"random_forest stub: gtree split feature=%d threshold=%.3f left=%.3f right=%.3f",
					split_feature,
					split_threshold,
					left_leaf_value,
					right_leaf_value);
			}
			else
			{
				left_idx = gtree_add_leaf(stub_tree, left_leaf_value);
				gtree_set_root(stub_tree, left_idx);
			}
			gtree_validate(stub_tree);
		}
	}

	model_id = rf_stub_next_model_id++;
	rf_stub_store_model(model_id, feature_dim, n_samples, n_classes,
		majority_value, majority_fraction, gini_impurity, label_entropy, class_counts_tmp,
		feature_means_tmp, feature_vars_tmp, stub_tree,
		split_feature, split_threshold, second_value, second_fraction,
		left_leaf_value, left_branch_fraction, right_leaf_value, right_branch_fraction,
		max_deviation, max_split_deviation);

	elog(NOTICE,
		"train_random_forest_classifier(minimal stub): rows=%d, classes=%d, dim=%d, majority=%.3f, frac=%.3f, second=%.3f, sfrac=%.3f, gini=%.3f, entropy=%.3f",
		n_samples,
		n_classes,
		feature_dim,
		majority_value,
		majority_fraction,
		second_value,
		second_fraction,
		gini_impurity,
		label_entropy);

	if (class_counts_tmp)
		pfree(class_counts_tmp);
	if (labels)
		pfree(labels);
	if (feature_means_tmp)
		pfree(feature_means_tmp);
	if (feature_vars_tmp)
		pfree(feature_vars_tmp);
	if (feature_sums)
		pfree(feature_sums);
	if (feature_sums_sq)
		pfree(feature_sums_sq);

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

static double
rf_stub_tree_predict(const RFStubModel *model, const Vector *vec)
{
	const GTree		*tree;
	const GTreeNode *nodes;
	int		idx;
	int		steps = 0;
	int		path_nodes[GTREE_MAX_DEPTH + 1];
	char	path_dir[GTREE_MAX_DEPTH];
	int		path_len = 0;
	int		leaf_idx = -1;
	double	result = 0.0;
	int		i;

	if (!model)
		return 0.0;

	tree = model->tree;
	if (tree == NULL)
		return model->majority_value;

	if (tree->root < 0 || tree->count <= 0)
		return model->majority_value;

	nodes = gtree_nodes(tree);
	idx = tree->root;

	while (idx >= 0 && idx < tree->count)
	{
		const GTreeNode *node = &nodes[idx];

		if (path_len <= GTREE_MAX_DEPTH)
			path_nodes[path_len] = idx;

		if (node->is_leaf)
		{
			leaf_idx = idx;
			break;
		}

		if (vec == NULL || node->feature_idx < 0 || node->feature_idx >= vec->dim)
		{
			elog(DEBUG1,
				"random_forest stub: path aborted at node %d (feature %d)",
				idx,
				node->feature_idx);
			return model->majority_value;
		}

		if (vec->data[node->feature_idx] <= node->threshold)
		{
			if (path_len < GTREE_MAX_DEPTH)
				path_dir[path_len] = 'L';
			idx = node->left;
		}
		else
		{
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
		result = model->majority_value;

	if (path_len > GTREE_MAX_DEPTH)
		path_len = GTREE_MAX_DEPTH;

	if (leaf_idx >= 0 && path_len <= GTREE_MAX_DEPTH)
		path_nodes[path_len] = leaf_idx;

	{
		StringInfoData path_log;
		int edge_count = (leaf_idx >= 0) ? path_len : path_len - 1;

		initStringInfo(&path_log);
		appendStringInfo(&path_log, "[");
		for (i = 0; i <= edge_count && i <= GTREE_MAX_DEPTH; i++)
		{
			if (i > 0)
				appendStringInfoString(&path_log, ", ");
			appendStringInfo(&path_log, "%d", path_nodes[i]);
			if (i < edge_count && i < GTREE_MAX_DEPTH)
				appendStringInfo(&path_log, "%c", path_dir[i]);
		}
		appendStringInfoChar(&path_log, ']');
		elog(DEBUG1,
			"random_forest stub: gtree path len=%d leaf_idx=%d path=%s",
			(edge_count < 0) ? 0 : (edge_count + 1),
			leaf_idx,
			path_log.data);
		pfree(path_log.data);
	}

	return result;
}

Datum
predict_random_forest(PG_FUNCTION_ARGS)
{
	int32 model_id = PG_GETARG_INT32(0);
	RFStubModel *model;
	Vector *feature_vec = NULL;
	double result;
	double split_z = 0.0;
	bool split_z_valid = false;
	const char *branch_name = "majority";
	double branch_fraction = 0.0;
	double branch_value = 0.0;

	if (!rf_stub_lookup_model(model_id, &model))
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			errmsg("random_forest: model %d not found", model_id)));

	branch_fraction = model->majority_fraction;
	branch_value = model->majority_value;

	if (!PG_ARGISNULL(1))
		feature_vec = PG_GETARG_VECTOR_P(1);

	if (model->n_features > 0 && feature_vec != NULL && feature_vec->dim != model->n_features)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			errmsg("random_forest: feature dimension mismatch (expected %d got %d)",
			model->n_features, feature_vec->dim)));

	if (model->feature_means != NULL && feature_vec != NULL)
	{
		float *vec_data = feature_vec->data;
		double dist = 0.0;
		int j;
		double max_z = 0.0;

		for (j = 0; j < model->n_features && j < feature_vec->dim; j++)
		{
			double diff = (double) vec_data[j] - model->feature_means[j];
			dist += diff * diff;
			if (model->feature_variances != NULL)
			{
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
			elog(DEBUG1, "random_forest stub: feature L2 distance %.3f max-z %.3f",
				dist, max_z);
		else
			elog(DEBUG1, "random_forest stub: feature L2 distance to mean %.3f", dist);
		if (model->feature_variances != NULL && model->second_fraction > 0.0 && max_z > 1.5)
			elog(DEBUG1,
				"random_forest stub: deviation %.3f exceeds threshold, considering second class %.3f",
				max_z,
				model->second_value);
		model->max_deviation = max_z;
	}
	else if (!PG_ARGISNULL(1))
		model->max_deviation = 0.0;

	if (feature_vec != NULL && model->split_feature >= 0)
	{
		int sf = model->split_feature;

		if (sf < feature_vec->dim)
		{
			double value = (double) feature_vec->data[sf];

			if (value <= model->split_threshold)
			{
				branch_name = "left";
				branch_fraction = model->left_branch_fraction;
				branch_value = model->left_branch_value;
			}
			else
			{
				branch_name = "right";
				branch_fraction = model->right_branch_fraction;
				branch_value = model->right_branch_value;
			}
			if (branch_fraction <= 0.0)
				branch_fraction = model->majority_fraction;

			if (model->feature_variances != NULL &&
				sf < model->n_features)
			{
				double var = model->feature_variances[sf];

				if (var > 0.0)
				{
					split_z = (value - model->split_threshold) / sqrt(var);
					if (fabs(split_z) > model->max_split_deviation)
						model->max_split_deviation = fabs(split_z);
					split_z_valid = true;
				}
			}
		}
	}

	result = rf_stub_tree_predict(model, feature_vec);
	if (model->feature_variances != NULL && model->second_fraction > 0.0 && !PG_ARGISNULL(1) &&
		model->max_deviation > 2.0 && model->label_entropy > 0.1)
	{
		elog(DEBUG1,
			"random_forest stub: high deviation %.3f -> returning second class %.3f",
			model->max_deviation,
			model->second_value);
		result = model->second_value;
		branch_name = "fallback";
		branch_fraction = model->second_fraction;
		branch_value = model->second_value;
	}

	if (result != model->majority_value)
		elog(NOTICE,
			"random_forest stub: majority=%.3f frac=%.3f, fallback=%.3f frac=%.3f branch=%s bfrac=%.3f entropy=%.3f split_dev=%.3f",
			model->majority_value,
			model->majority_fraction,
			result,
			model->second_fraction,
			branch_name,
			branch_fraction,
			model->label_entropy,
			model->max_split_deviation);
 
	if (model->split_feature >= 0)
	{
		if (split_z_valid)
			elog(DEBUG1,
				"random_forest stub: split f=%d thr=%.3f left=%.3f lf=%.3f right=%.3f rf=%.3f z=%.3f",
				model->split_feature,
				model->split_threshold,
				model->left_branch_value,
				model->left_branch_fraction,
				model->right_branch_value,
				model->right_branch_fraction,
				split_z);
		else
			elog(DEBUG1,
				"random_forest stub: split f=%d thr=%.3f left=%.3f lf=%.3f right=%.3f rf=%.3f",
				model->split_feature,
				model->split_threshold,
				model->left_branch_value,
				model->left_branch_fraction,
				model->right_branch_value,
				model->right_branch_fraction);
	}

	elog(NOTICE, "predict_random_forest stub: returning %.3f (branch %s leaf %.3f frac %.3f majority %.3f frac %.3f gini %.3f)",
		result,
		branch_name,
		branch_value,
		branch_fraction,
		model->majority_value,
		model->majority_fraction,
		model->gini_impurity);
	PG_RETURN_FLOAT8(result);
}

Datum
evaluate_random_forest(PG_FUNCTION_ARGS)
{
	Datum result_datums[4];
	ArrayType *result_array;
	int32 model_id;
	RFStubModel *model;
	double accuracy;
	double error_rate;

	if (PG_NARGS() < 1 || PG_ARGISNULL(0))
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			errmsg("random_forest: model_id required")));

	model_id = PG_GETARG_INT32(0);

	if (!rf_stub_lookup_model(model_id, &model))
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			errmsg("random_forest: model %d not found", model_id)));

	accuracy = model->majority_fraction;
	error_rate = (accuracy > 1.0) ? 0.0 : (1.0 - accuracy);

	elog(NOTICE,
		"evaluate_random_forest stub: samples=%d classes=%d accuracy=%.3f gini=%.3f entropy=%.3f",
		model->n_samples,
		model->n_classes,
		accuracy,
		model->gini_impurity,
		model->label_entropy);

	result_datums[0] = Float8GetDatum(accuracy);
	result_datums[1] = Float8GetDatum(error_rate);
	result_datums[2] = Float8GetDatum(model->gini_impurity);
	result_datums[3] = Float8GetDatum((double) model->n_classes);

	result_array = construct_array(result_datums,
		4,
		FLOAT8OID,
		sizeof(float8),
		true,
		'd');

	PG_RETURN_ARRAYTYPE_P(result_array);
}

