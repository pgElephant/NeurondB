/*-------------------------------------------------------------------------
 *
 * ml_random_forest.c
 *    Random Forest implementation for classification and regression.
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
 #include "utils/memutils.h"
 #include "miscadmin.h"
 #include "access/htup_details.h"
 #include "utils/jsonb.h"
 #include "utils/elog.h"
 #include "utils/lsyscache.h"
 
 #include "neurondb.h"
 #include "neurondb_ml.h"
 #include "neurondb_gpu.h"
#include "ml_gpu_context.h"
#include "ml_gpu_buffer.h"
#include "ml_gpu_support.h"
 
 #include <math.h>
 #include <float.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <sys/time.h>
 #include <unistd.h>
 
 #include "common/pg_prng.h"
 
 /** Flat node for compact inference */
 typedef struct RFNodeFlat
 {
	 int32		feature_index;
	 uint32		_pad0;
	 double		threshold;
	 int32		left;
	 int32		right;
	 int32		is_leaf;
	 uint32		_pad1;
	 double		value;
 } RFNodeFlat;
 
 /** Flat tree header */
 typedef struct RFTreeFlat
 {
	 int32		offset_to_nodes;
	 int32		num_nodes;
	 int64		_pad2;
 } RFTreeFlat;
 
 /** Flat model header */
 typedef struct RFModelFlat
 {
	 int32		n_trees;
	 int32		max_depth;
	 int32		min_samples_split;
	 int32		max_features;
	 int32		n_classes;
	 int32		n_features;
	 int32		trees_offset;
	 int32		nodes_offset;
	 int64		_pad3;
 } RFModelFlat;
 
 /** RNG state */
 static pg_prng_state rf_rng_state;
 
 /** Decision tree node for training */
 typedef struct TreeNode
 {
	 int		feature_index;
	 double	threshold;
	 int		left;
	 int		right;
	 int		is_leaf;
	 double	value;
 } TreeNode;
 
 /** Training tree */
 typedef struct DecisionTree
 {
	 TreeNode   *nodes;
	 int		num_nodes;
	 int		allocated;
 } DecisionTree;
 
 static inline bool
 safe_add_mul_size(Size a, Size b, Size c, Size *out)
 {
	 Size	prod;
 
	 if (b != 0 && c > SIZE_MAX / b)
		 return false;
	 prod = b * c;
	 if (a > SIZE_MAX - prod)
		 return false;
	 *out = a + prod;
	 return true;
 }
 
 static void
 rf_rng_seed(void)
 {
	 struct timeval	tv;
	 uint64		seed;
 
	 gettimeofday(&tv, NULL);
	 seed = ((uint64) tv.tv_usec) ^ ((uint64) getpid() << 16);
	 pg_prng_seed(&rf_rng_state, seed);
 }
 
 static inline void
 rf_training_context_cleanup(MemoryContext *ctx_ptr,
				 MemoryContext oldcontext)
 {
	 MemoryContextSwitchTo(oldcontext);
	 if (ctx_ptr && *ctx_ptr)
	 {
		 MemoryContextDelete(*ctx_ptr);
		 *ctx_ptr = NULL;
	 }
 }
 
 /** Bootstrap sample indexes */
 static void
 rf_bootstrap_sample(int n_samples, int **indices_out)
 {
	 int    *indices;
	 int	i;
 
	 indices = (int *) palloc(sizeof(int) * n_samples);
	 for (i = 0; i < n_samples; i++)
	 {
		 CHECK_FOR_INTERRUPTS();
		 indices[i] =
			 (int) (pg_prng_uint32(&rf_rng_state) % (uint32) n_samples);
	 }
	 *indices_out = indices;
 }
 
 /** Random feature subset */
 static void
 rf_random_features(int n_features, int max_features, int **features_out)
 {
	 int    *all_features;
	 int    *features;
	 int	i;
	 int	j;
	 int	temp;
 
	 if (max_features <= 0 || max_features > n_features)
		 max_features = (int) sqrt((double) n_features);
 
	 features = (int *) palloc(sizeof(int) * max_features);
	 all_features = (int *) palloc(sizeof(int) * n_features);
 
	 for (i = 0; i < n_features; i++)
		 all_features[i] = i;
 
	 for (i = n_features - 1; i > 0; i--)
	 {
		 CHECK_FOR_INTERRUPTS();
		 j = (int) (pg_prng_uint32(&rf_rng_state) % (uint32) (i + 1));
		 temp = all_features[i];
		 all_features[i] = all_features[j];
		 all_features[j] = temp;
	 }
 
	 for (i = 0; i < max_features; i++)
		 features[i] = all_features[i];
 
	 pfree(all_features);
	 *features_out = features;
 }
 
 /** Count classes in labels */
 static int
 rf_count_classes(double *labels, int n)
 {
	 int	maxclass;
	 int	i;
 
	 maxclass = 0;
	 for (i = 0; i < n; i++)
	 {
		 int	asint;
 
		 asint = (int) labels[i];
		 if (asint > maxclass)
			 maxclass = asint;
	 }
	 return maxclass + 1;
 }
 
 /** Gini impurity */
 static double
 rf_gini(double *labels, int *indices, int n_indices, int n_classes)
 {
	 int	i;
	 int	c;
	 double *class_counts;
	 double	gini;
	 double	p;
 
	 if (n_indices == 0)
		 return 0.0;
 
	 class_counts = (double *) palloc0(sizeof(double) * n_classes);
	 gini = 1.0;
 
	 for (i = 0; i < n_indices; i++)
	 {
		 int	idx;
		 int	label_idx;
 
		 CHECK_FOR_INTERRUPTS();
		 idx = indices[i];
		 label_idx = (int) labels[idx];
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
 
 /** Choose best split for a node */
 static bool
 rf_find_best_split(float **X, double *y, int *indices, int n_indices,
			int n_features, int n_classes, int max_features,
			int *best_feature, double *best_threshold,
			double *best_gini, int **left_indices_out,
			int *n_left_out, int **right_indices_out,
			int *n_right_out, MLGpuContext *gpu_ctx)
 {
	 int    *features;
	 int	i;
	 int	f;
	 int	t;
	 double	min_gini;
	 bool	found_split;
	 MemoryContext cx;
	 MemoryContext loop_oldcx;
	 uint8_t *labels01;
	 bool	try_gpu;
 
	 features = NULL;
	 min_gini = 1.0;
	 found_split = false;
	 cx = NULL;
	 loop_oldcx = NULL;
	 labels01 = NULL;
	 try_gpu = false;

	 /* GPU assist precheck for binary labels */
	 if (gpu_ctx != NULL && ml_gpu_context_ready(gpu_ctx) && n_classes == 2 &&
	     ndb_gpu_kernel_enabled("rf_split"))
	 {
		 int k;
		 int mask = 0;
		 try_gpu = true;
		 labels01 = (uint8_t *) palloc(sizeof(uint8_t) * n_indices);
		 for (k = 0; k < n_indices; k++)
		 {
			 int yi = (int) y[indices[k]];
			 if (yi != 0 && yi != 1)
			 {
				 try_gpu = false;
				 break;
			 }
			 labels01[k] = (uint8_t) yi;
			 mask |= (1 << yi);
		 }
		 if (mask != 0x3)
			 try_gpu = false;
		 if (try_gpu)
			 cx = AllocSetContextCreate(CurrentMemoryContext,
						    "rf_gpu_split_ctx",
						    ALLOCSET_SMALL_SIZES);
		 else
		 {
			 pfree(labels01);
			 labels01 = NULL;
		 }
	 }
 
	 if (left_indices_out)
		 *left_indices_out = NULL;
	 if (right_indices_out)
		 *right_indices_out = NULL;
 
	 rf_random_features(n_features, max_features, &features);
 
	 for (f = 0; f < max_features; f++)
	 {
		 int	feature_idx;
		 double *feature_values;
		 int	j;
 
		 CHECK_FOR_INTERRUPTS();
 
		 feature_idx = features[f];

		/* GPU-assisted best split for binary labels */
		if (try_gpu && cx != NULL)
		{
			float  *featf;
			double gthr = 0.0;
			double ggini = 1.0;
			int lc = 0;
			int rc = 0;
			MLGpuBuffer feat_buf;

			loop_oldcx = MemoryContextSwitchTo(cx);
			featf = (float *) palloc(sizeof(float) * n_indices);
			for (i = 0; i < n_indices; i++)
			{
				float v = X[indices[i]][feature_idx];

				if (!isfinite(v))
				{
					MemoryContextSwitchTo(loop_oldcx);
					MemoryContextReset(cx);
					goto cpu_feature_fallback;
				}
				featf[i] = v;
			}
			ml_gpu_buffer_bind_host(&feat_buf, gpu_ctx,
						featf,
						sizeof(float) * n_indices,
						n_indices,
						MLGPU_DTYPE_FLOAT32);
			ml_gpu_buffer_invalidate_device(&feat_buf);
			if (neurondb_gpu_rf_best_split_binary(featf, labels01, n_indices,
								 &gthr, &ggini, &lc, &rc))
			{
				if (lc > 0 && rc > 0 && ggini < min_gini)
				{
					int *left;
					int *right;

					left = (int *) palloc(sizeof(int) * n_indices);
					right = (int *) palloc(sizeof(int) * n_indices);
					lc = 0;
					rc = 0;
					for (j = 0; j < n_indices; j++)
					{
						if (X[indices[j]][feature_idx] <= gthr)
							left[lc++] = indices[j];
						else
							right[rc++] = indices[j];
					}
					MemoryContextSwitchTo(loop_oldcx);
					ml_gpu_buffer_release(&feat_buf);
					if (lc == 0 || rc == 0)
					{
						MemoryContextReset(cx);
						goto cpu_feature_fallback;
					}
					if (best_feature)
						*best_feature = feature_idx;
					if (best_threshold)
						*best_threshold = gthr;
					if (best_gini)
						*best_gini = ggini;
					if (left_indices_out)
					{
						if (*left_indices_out)
							pfree(*left_indices_out);
						*left_indices_out = (int *) palloc(sizeof(int) * lc);
						memcpy(*left_indices_out, left,
							sizeof(int) * lc);
					}
					if (n_left_out)
						*n_left_out = lc;
					if (right_indices_out)
					{
						if (*right_indices_out)
							pfree(*right_indices_out);
						*right_indices_out = (int *) palloc(sizeof(int) * rc);
						memcpy(*right_indices_out, right,
							sizeof(int) * rc);
					}
					if (n_right_out)
						*n_right_out = rc;
					min_gini = ggini;
					found_split = true;
					MemoryContextReset(cx);
					continue;
				}
			}
			ml_gpu_buffer_release(&feat_buf);
			MemoryContextSwitchTo(loop_oldcx);
			MemoryContextReset(cx);
		}

cpu_feature_fallback:
		feature_values = (double *) palloc(sizeof(double) * n_indices);
 
		 for (i = 0; i < n_indices; i++)
			 feature_values[i] = X[indices[i]][feature_idx];
 
		 for (i = 0; i < n_indices - 1; i++)
		 {
			 for (t = i + 1; t < n_indices; t++)
			 {
				 if (feature_values[i] > feature_values[t])
				 {
					 double	tmp;
 
					 tmp = feature_values[i];
					 feature_values[i] = feature_values[t];
					 feature_values[t] = tmp;
				 }
			 }
		 }
 
		 for (i = 1; i < n_indices; i++)
		 {
			 double	threshold;
			 int    *left;
			 int    *right;
			 int	n_left;
			 int	n_right;
			 double	gini_left;
			 double	gini_right;
			 double	gini;
 
			 if (feature_values[i] == feature_values[i - 1])
				 continue;
 
			 threshold =
				 (feature_values[i - 1] + feature_values[i]) / 2.0;
			 left = (int *) palloc(sizeof(int) * n_indices);
			 right = (int *) palloc(sizeof(int) * n_indices);
			 n_left = 0;
			 n_right = 0;
 
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
 
			 gini_left = rf_gini(y, left, n_left, n_classes);
			 gini_right = rf_gini(y, right, n_right, n_classes);
 
			 gini = ((double) n_left / n_indices) * gini_left +
					((double) n_right / n_indices) * gini_right;
 
			 if (gini < min_gini)
			 {
				 if (left_indices_out && *left_indices_out)
					 pfree(*left_indices_out);
				 if (right_indices_out && *right_indices_out)
					 pfree(*right_indices_out);
 
				 if (best_feature)
					 *best_feature = feature_idx;
				 if (best_threshold)
					 *best_threshold = threshold;
				 if (best_gini)
					 *best_gini = gini;
 
				 if (left_indices_out)
				 {
					 *left_indices_out =
						 (int *) palloc(sizeof(int) *
								 n_left);
					 memcpy(*left_indices_out, left,
							sizeof(int) * n_left);
				 }
				 if (n_left_out)
					 *n_left_out = n_left;
 
				 if (right_indices_out)
				 {
					 *right_indices_out =
						 (int *) palloc(sizeof(int) *
								 n_right);
					 memcpy(*right_indices_out, right,
							sizeof(int) * n_right);
				 }
				 if (n_right_out)
					 *n_right_out = n_right;
 
				 min_gini = gini;
				 found_split = true;
			 }
			 pfree(left);
			 pfree(right);
		 }
		 pfree(feature_values);
	 }
	 if (features)
		 pfree(features);
	if (cx)
		MemoryContextDelete(cx);
	 if (labels01)
		 pfree(labels01);
 
	 return found_split;
 }
 
 /** Add node to training tree */
 static int
 rf_tree_add_node(DecisionTree *tree, int feature, double threshold,
		  int is_leaf, double value)
 {
	 int	new_alloc;
 
	 if (tree->num_nodes >= tree->allocated)
	 {
		 new_alloc = (tree->allocated == 0 ? 32 : tree->allocated * 2);
 
		 if (tree->nodes == NULL)
			 tree->nodes =
				 (TreeNode *) palloc(sizeof(TreeNode) *
							 new_alloc);
		 else
			 tree->nodes =
				 (TreeNode *) repalloc(tree->nodes,
							   sizeof(TreeNode) *
							   new_alloc);
 
		 tree->allocated = new_alloc;
	 }
 
	 tree->nodes[tree->num_nodes].feature_index = feature;
	 tree->nodes[tree->num_nodes].threshold = threshold;
	 tree->nodes[tree->num_nodes].is_leaf = is_leaf;
	 tree->nodes[tree->num_nodes].value = value;
	 tree->nodes[tree->num_nodes].left = -1;
	 tree->nodes[tree->num_nodes].right = -1;
 
	 return (tree->num_nodes++);
 }
 
 /** Recursively build a tree */
 static int
 rf_build_tree(DecisionTree *tree, float **X, double *y, int *indices,
		   int n_indices, int n_features, int n_classes, int depth,
		   int max_depth, int min_samples_split, int max_features,
		   MLGpuContext *gpu_ctx)
 {
	 int	i;
	 int	left_child;
	 int	right_child;
	 int    *left_indices;
	 int    *right_indices;
	 int	n_left;
	 int	n_right;
	 int	best_feature;
	 double	best_threshold;
	 double	best_gini;
	 int	node_id;
	 int    *class_counts;
	 double	majority;
	 int	max_count;
 
	 left_child = -1;
	 right_child = -1;
	 left_indices = NULL;
	 right_indices = NULL;
	 n_left = 0;
	 n_right = 0;
	 best_feature = -1;
	 best_threshold = 0.0;
	 best_gini = 0.0;
	 class_counts = NULL;
	 majority = -1.0;
	 max_count = 0;
 
	 class_counts =
		 (int *) palloc0(sizeof(int) * Max(n_classes, 2));
 
	 for (i = 0; i < n_indices; i++)
	 {
		 int	ylab;
 
		 CHECK_FOR_INTERRUPTS();
		 ylab = (int) y[indices[i]];
		 if (ylab < 0 || ylab >= n_classes)
			 continue;
		 class_counts[ylab]++;
	 }
 
	 for (i = 0; i < n_classes; i++)
	 {
		 if (class_counts[i] > max_count)
		 {
			 max_count = class_counts[i];
			 majority = (double) i;
		 }
	 }
 
	 if (depth >= max_depth ||
		 n_indices < min_samples_split ||
		 max_count == n_indices)
	 {
		 node_id = rf_tree_add_node(tree, -1, 0, 1, majority);
		 pfree(class_counts);
		 return node_id;
	 }
 
	 if (!rf_find_best_split(X, y, indices, n_indices, n_features,
				 n_classes, max_features, &best_feature,
				 &best_threshold, &best_gini, &left_indices,
				 &n_left, &right_indices, &n_right, gpu_ctx))
	 {
		 node_id = rf_tree_add_node(tree, -1, 0, 1, majority);
		 if (class_counts)
			 pfree(class_counts);
		 return node_id;
	 }
 
	 node_id = rf_tree_add_node(tree, best_feature, best_threshold, 0, 0.0);
 
	 left_child =
		 rf_build_tree(tree, X, y, left_indices, n_left, n_features,
				   n_classes, depth + 1, max_depth,
				   min_samples_split, max_features, gpu_ctx);
	 right_child =
		 rf_build_tree(tree, X, y, right_indices, n_right, n_features,
				   n_classes, depth + 1, max_depth,
				   min_samples_split, max_features, gpu_ctx);
 
	 tree->nodes[node_id].left = left_child;
	 tree->nodes[node_id].right = right_child;
 
	 if (left_indices)
		 pfree(left_indices);
	 if (right_indices)
		 pfree(right_indices);
	 if (class_counts)
		 pfree(class_counts);
 
	 return node_id;
 }
 
 /** Safe walk over flat tree */
 static double
 rf_tree_predict_flat_safe(const RFNodeFlat *nodes, int num_nodes,
			   int n_features, const float *x)
 {
	 int	curr;
	 int	num_steps;
	 int	max_tree_steps;
 
	 curr = 0;
	 num_steps = 0;
	 max_tree_steps = 2048;
 
	 while (curr >= 0 && curr < num_nodes && num_steps < max_tree_steps)
	 {
		 const RFNodeFlat *n;
		 int	left;
		 int	right;
 
		 n = &nodes[curr];
 
		 if (n->is_leaf)
			 return n->value;
 
		 if (n->feature_index < 0 || n->feature_index >= n_features)
			 elog(ERROR,
				  "tree feature_index %d out of range [0,%d]",
				  n->feature_index, n_features);
 
		 left = n->left;
		 right = n->right;
 
		 if (left < 0 || left >= num_nodes ||
			 right < 0 || right >= num_nodes)
			 elog(ERROR,
				  "invalid child pointers left=%d right=%d "
				  "num_nodes=%d",
				  left, right, num_nodes);
 
		 curr = (x[n->feature_index] <= n->threshold) ? left : right;
		 num_steps++;
	 }
	 elog(ERROR, "tree walk overflow or cycle detected");
	 pg_unreachable();
 }
 
 /** SQL: train_random_forest_classifier */
 PG_FUNCTION_INFO_V1(train_random_forest_classifier);
 Datum
 train_random_forest_classifier(PG_FUNCTION_ARGS)
 {
	 text	   *table_name;
	 text	   *feature_col;
	 text	   *label_col;
	 int		n_trees;
	 int		max_depth;
	 int		min_samples_split;
	 int		max_features;
	 char	   *tbl_str;
	 char	   *feat_str;
	 char	   *label_str;
	 int		ret;
	 int		nvec;
	 int		dim;
	 int		n_classes;
	 int		i;
	 int		total_nodes;
	 float	  **X;
	 double	   *y;
	 DecisionTree *trees;
	 RFTreeFlat *blob_trees;
	 RFNodeFlat *blob_nodes;
	 bytea	   *model_bytea;
	 int		tlen;
	 int		nodes_offset;
	 int		trees_offset;
	 int		model_id;
	 StringInfoData query;
	 StringInfoData insert_query;
	 bool		isnull;
	 Datum		result;
	 MemoryContext oldcontext;
	 MemoryContext traincx;
	 Size		tlen_size;
	 MLGpuContext *gpu_ctx;
	 MLGpuCallState gpu_state;
 
	 /* Parse arguments */
	 table_name = PG_GETARG_TEXT_PP(0);
	 feature_col = PG_GETARG_TEXT_PP(1);
	 label_col = PG_GETARG_TEXT_PP(2);
	 n_trees = PG_GETARG_INT32(3);
	 max_depth = PG_GETARG_INT32(4);
	 min_samples_split = PG_NARGS() > 5 ? PG_GETARG_INT32(5) : 2;
	 max_features = PG_NARGS() > 6 ? PG_GETARG_INT32(6) : 0;
	 result = (Datum) 0;

	 ml_gpu_call_begin(&gpu_state,
				"random_forest_train",
				"rf_split",
				(neurondb_gpu_enabled && !neurondb_gpu_fail_open));

	 if (neurondb_gpu_enabled && !ml_gpu_call_use_gpu(&gpu_state))
		 elog(NOTICE,
			 "neurondb: GPU enabled; random_forest training using CPU fallback");
 
	 tbl_str = text_to_cstring(table_name);
	 feat_str = text_to_cstring(feature_col);
	 label_str = text_to_cstring(label_col);
 
	 if (n_trees < 1)
		 ereport(ERROR,
			 (errmsg("n_trees must be at least 1")));
	 if (max_depth < 1)
		 ereport(ERROR, (errmsg("max_depth must be positive")));
	 if (min_samples_split < 2)
		 ereport(ERROR,
			 (errmsg("min_samples_split must be at least 2")));
	 if (max_features < 0)
		 ereport(ERROR, (errmsg("max_features cannot be negative")));
 
	 PG_TRY();
	 {
	 traincx = AllocSetContextCreate(CurrentMemoryContext,
					 "RF Training TempContext",
					 ALLOCSET_DEFAULT_SIZES);
	 oldcontext = MemoryContextSwitchTo(traincx);
 
	 rf_rng_seed();
 
	 ret = SPI_connect();
	 if (ret != SPI_OK_CONNECT)
		 ereport(ERROR,
			 (errmsg("SPI_connect failed: %d", ret)));
 
	 initStringInfo(&query);
	 appendStringInfo(&query,
			  "SELECT %s, %s FROM %s "
			  "WHERE %s IS NOT NULL AND %s IS NOT NULL",
			  feat_str, label_str, tbl_str, feat_str, label_str);
 
	 ret = SPI_execute(query.data, true, 0);
	 if (ret != SPI_OK_SELECT)
	 {
		 SPI_finish();
		 rf_training_context_cleanup(&traincx, oldcontext);
		 ereport(ERROR,
			 (errmsg("training query failed: %s", query.data)));
	 }
 
	 nvec = SPI_processed;
	 if (nvec < 2)
	 {
		 SPI_finish();
		 rf_training_context_cleanup(&traincx, oldcontext);
		 ereport(ERROR,
			 (errmsg("need at least 2 samples, have %d", nvec)));
	 }
 
	 X = (float **) palloc(sizeof(float *) * nvec);
	 y = (double *) palloc(sizeof(double) * nvec);
	 dim = 0;
 
	 for (i = 0; i < nvec; i++)
	 {
		 HeapTuple	tup;
		 TupleDesc	tupdesc;
		 Datum		feat_datum;
		 Datum		label_datum;
		 bool		feat_null;
		 bool		label_null;
		 Oid		label_type;
		 Vector	   *vec;
 
		 CHECK_FOR_INTERRUPTS();
 
		 tup = SPI_tuptable->vals[i];
		 tupdesc = SPI_tuptable->tupdesc;
 
		 feat_datum = SPI_getbinval(tup, tupdesc, 1, &feat_null);
		 if (feat_null)
		 {
			 SPI_finish();
			 rf_training_context_cleanup(&traincx, oldcontext);
			 ereport(ERROR,
				 (errmsg("null feature vector")));
		 }
		 vec = DatumGetVector(feat_datum);
 
		 if (i == 0)
			 dim = vec->dim;
		 else if (vec->dim != dim)
		 {
			 SPI_finish();
			 rf_training_context_cleanup(&traincx, oldcontext);
			 ereport(ERROR,
				 (errmsg("inconsistent feature dim %d vs %d",
					 vec->dim, dim)));
		 }
 
		 X[i] = (float *) palloc(sizeof(float) * dim);
		 memcpy(X[i], vec->data, sizeof(float) * dim);
 
		 label_datum = SPI_getbinval(tup, tupdesc, 2, &label_null);
		 if (label_null)
		 {
			 SPI_finish();
			 rf_training_context_cleanup(&traincx, oldcontext);
			 ereport(ERROR, (errmsg("null label")));
		 }
		 label_type = SPI_gettypeid(tupdesc, 2);
 
		 if (label_type == INT2OID || label_type == INT4OID)
			 y[i] = (double) DatumGetInt32(label_datum);
		 else if (label_type == INT8OID)
			 y[i] = (double) DatumGetInt64(label_datum);
		 else
			 y[i] = DatumGetFloat8(label_datum);
	 }
 
	 n_classes = rf_count_classes(y, nvec);
 
	 SPI_finish();
 
	 if (n_classes < 1 || dim < 1 || n_trees < 1)
	 {
		 rf_training_context_cleanup(&traincx, oldcontext);
		 ereport(ERROR,
			 (errmsg("illegal model config "
				 "classes=%d features=%d trees=%d",
				 n_classes, dim, n_trees)));
	 }
 
	 trees = (DecisionTree *) palloc0(sizeof(DecisionTree) * n_trees);
	 total_nodes = 0;
 
	 for (i = 0; i < n_trees; i++)
	 {
		 int	   *bootstrap_indices;
		 DecisionTree *tree;
 
		 CHECK_FOR_INTERRUPTS();
 
		 bootstrap_indices = NULL;
		 tree = &trees[i];
 
		 rf_bootstrap_sample(nvec, &bootstrap_indices);
 
		 tree->nodes = NULL;
		 tree->num_nodes = 0;
		 tree->allocated = 0;
 
		 (void) rf_build_tree(tree, X, y, bootstrap_indices, nvec,
					  dim, n_classes, 0, max_depth,
					  min_samples_split,
					  (max_features > 0)
						? max_features
						: (int) sqrt((double) dim),
					  ml_gpu_call_use_gpu(&gpu_state) ?
					  ml_gpu_call_context(&gpu_state) : NULL);
 
		 total_nodes += tree->num_nodes;
 
		 if (bootstrap_indices)
			 pfree(bootstrap_indices);
	 }
 
	 trees_offset = sizeof(RFModelFlat);
	 nodes_offset = trees_offset + n_trees * sizeof(RFTreeFlat);
 
	 if (!safe_add_mul_size((Size) nodes_offset,
					(Size) total_nodes,
					sizeof(RFNodeFlat),
					&tlen_size))
		 ereport(ERROR, (errmsg("model too large")));
	 if (tlen_size > MaxAllocSize - VARHDRSZ)
		 ereport(ERROR, (errmsg("model too large")));
	 tlen = (int) tlen_size;
 
	 model_bytea = (bytea *) palloc(tlen + VARHDRSZ);
	 SET_VARSIZE(model_bytea, tlen + VARHDRSZ);
 
	 {
		 RFModelFlat *rf_hdr;
		 int		node_cursor;
		 int		j;
 
		 rf_hdr = (RFModelFlat *) VARDATA(model_bytea);
		 node_cursor = 0;
 
		 rf_hdr->n_trees = n_trees;
		 rf_hdr->max_depth = max_depth;
		 rf_hdr->min_samples_split = min_samples_split;
		 rf_hdr->max_features =
			 (max_features > 0)
				 ? max_features
				 : (int) sqrt((double) dim);
		 rf_hdr->n_classes = n_classes;
		 rf_hdr->n_features = dim;
		 rf_hdr->trees_offset = trees_offset;
		 rf_hdr->nodes_offset = nodes_offset;
 
		 blob_trees = (RFTreeFlat *)
			 ((char *) rf_hdr + trees_offset);
		 blob_nodes = (RFNodeFlat *)
			 ((char *) rf_hdr + nodes_offset);
 
		 for (i = 0; i < n_trees; i++)
		 {
			 CHECK_FOR_INTERRUPTS();
 
			 if (node_cursor < 0 ||
				 node_cursor + trees[i].num_nodes > total_nodes)
			 {
				 rf_training_context_cleanup(&traincx,
								 oldcontext);
				 ereport(ERROR,
					 (errmsg("serialization overflow "
						 "tree=%d cursor=%d total=%d",
						 i, node_cursor,
						 total_nodes)));
			 }
 
			 blob_trees[i].offset_to_nodes = node_cursor;
			 blob_trees[i].num_nodes = trees[i].num_nodes;
			 blob_trees[i]._pad2 = 0;
 
			 for (j = 0; j < trees[i].num_nodes; j++)
			 {
				 blob_nodes[node_cursor].feature_index =
					 trees[i].nodes[j].feature_index;
				 blob_nodes[node_cursor]._pad0 = 0;
				 blob_nodes[node_cursor].threshold =
					 trees[i].nodes[j].threshold;
				 blob_nodes[node_cursor].left =
					 trees[i].nodes[j].left;
				 blob_nodes[node_cursor].right =
					 trees[i].nodes[j].right;
				 blob_nodes[node_cursor].is_leaf =
					 trees[i].nodes[j].is_leaf;
				 blob_nodes[node_cursor]._pad1 = 0;
				 blob_nodes[node_cursor].value =
					 trees[i].nodes[j].value;
				 node_cursor++;
			 }
		 }
	 }
 
	 if (trees)
	 {
		 for (i = 0; i < n_trees; i++)
			 if (trees[i].nodes)
				 pfree(trees[i].nodes);
		 pfree(trees);
	 }
 
	 ret = SPI_connect();
	 if (ret != SPI_OK_CONNECT)
	 {
		 rf_training_context_cleanup(&traincx, oldcontext);
		 ereport(ERROR,
			 (errmsg("SPI_connect failed: %d", ret)));
	 }
 
	 initStringInfo(&insert_query);
	 appendStringInfo(&insert_query,
			  "INSERT INTO neurondb.ml_models "
			  "(algorithm, training_table, training_column, "
			  " parameters, model_data, status) "
			  "VALUES ($1, $2, $3, $4, $5, $6) "
			  "RETURNING model_id");
 
	 {
		 Oid		argtypes[6];
		 Datum		values[6];
		 const char *nulls;
		 const char *algo_name;
		 char		parambuf[128];
		 Datum		params_jsonb;
 
		 argtypes[0] = TEXTOID;
		 argtypes[1] = TEXTOID;
		 argtypes[2] = TEXTOID;
		 argtypes[3] = JSONBOID;
		 argtypes[4] = BYTEAOID;
		 argtypes[5] = TEXTOID;
 
		 nulls = NULL;
		 algo_name = "random_forest";
 
		 values[0] = CStringGetTextDatum(algo_name);
		 values[1] = CStringGetTextDatum(tbl_str);
		 values[2] = CStringGetTextDatum(feat_str);
 
		 snprintf(parambuf, sizeof(parambuf),
			  "{\"n_trees\": %d, \"max_depth\": %d}",
			  n_trees, max_depth);
		 params_jsonb =
			 DirectFunctionCall1(jsonb_in,
						 CStringGetDatum(parambuf));
		 values[3] = params_jsonb;
		 values[4] = PointerGetDatum(model_bytea);
		 values[5] = CStringGetTextDatum("completed");
 
		 ret = SPI_execute_with_args(insert_query.data, 6,
						 argtypes, values, nulls,
						 false, 1);
 
		 if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
		 {
			 rf_training_context_cleanup(&traincx, oldcontext);
			 ereport(ERROR,
				 (errmsg("failed to insert model")));
		 }
		 model_id =
			 DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
							 SPI_tuptable->tupdesc,
							 1, &isnull));
	 }
	 SPI_finish();
 
	 result = Int32GetDatum(model_id);
 
	 rf_training_context_cleanup(&traincx, oldcontext);
	 }
	 PG_CATCH();
	 {
		 ml_gpu_call_end(&gpu_state);
		 PG_RE_THROW();
	 }
	 PG_END_TRY();

	 ml_gpu_call_end(&gpu_state);

	 PG_RETURN_DATUM(result);
 }
 
 /** SQL: predict_random_forest */
 PG_FUNCTION_INFO_V1(predict_random_forest);
 Datum
 predict_random_forest(PG_FUNCTION_ARGS)
 {
	 int		model_id;
	 Datum		features_datum;
	 Vector	   *features;
	 float	   *feature_vector;
	 int		ret;
	 bool		isnull;
	 bytea	   *model_bytea;
	 int		i;
	 int		t;
	 int		n_trees;
	 int		n_classes;
	 int		n_features;
	 double		prediction;
	 MLGpuCallState gpu_state;
	 bool		used_gpu;
	 char	   *gpu_errstr;
	 bool		spi_finished;

	 model_id = PG_GETARG_INT32(0);
	 features_datum = PG_GETARG_DATUM(1);

	 if (model_id < 0)
		 ereport(ERROR,
			 (errmsg("model_id must be non-negative")));

	 features = DatumGetVector(features_datum);
	 if (features == NULL)
		 PG_RETURN_NULL();

	 feature_vector = features->data;
	 prediction = 0.0;
	 used_gpu = false;
	 gpu_errstr = NULL;
	 spi_finished = false;

	 ml_gpu_call_begin(&gpu_state,
				"random_forest_predict",
				"rf_predict",
				(neurondb_gpu_enabled && !neurondb_gpu_fail_open));

	 PG_TRY();
	 {
	 if (SPI_connect() != SPI_OK_CONNECT)
		 ereport(ERROR,
			 (errmsg("SPI_connect failed for predict")));

	 {
		 const char *sql;
		 Oid		argtypes[1];
		 Datum		values[1];

		 sql = "SELECT model_data "
		   "FROM neurondb.ml_models WHERE model_id = $1";
		 argtypes[0] = INT4OID;
		 values[0] = Int32GetDatum(model_id);
		 ret = SPI_execute_with_args(sql, 1, argtypes,
					 values, NULL, true, 1);
	 }

	 if (ret != SPI_OK_SELECT || SPI_processed == 0)
	 {
		 SPI_finish();
		 spi_finished = true;
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("model %d not found", model_id)));
	 }

	 model_bytea = (bytea *) DatumGetPointer(
		 SPI_getbinval(SPI_tuptable->vals[0],
			   SPI_tuptable->tupdesc, 1, &isnull));

	 if (isnull || model_bytea == NULL)
	 {
		 SPI_finish();
		 spi_finished = true;
		 ereport(ERROR,
			 (errmsg("model data is NULL for id=%d", model_id)));
	 }
	 if (VARSIZE_ANY_EXHDR(model_bytea) <
		 (int32) sizeof(RFModelFlat))
	 {
		 SPI_finish();
		 spi_finished = true;
		 ereport(ERROR,
			 (errmsg("model blob too small for id=%d", model_id)));
	 }

	 {
		 const char	   *rf_data;
		 const RFModelFlat *rf;
		 const RFTreeFlat  *blob_trees;
		 const RFNodeFlat  *blob_nodes;
		 int		node_capacity;
		 Size		payload;
		 Size		trees_off;
		 Size		nodes_off;
		 Size		req_trees;

		 rf_data = VARDATA(model_bytea);
		 rf = (const RFModelFlat *) rf_data;

		 n_trees = rf->n_trees;
		 n_classes = rf->n_classes;
		 n_features = rf->n_features;

		 if (n_trees <= 0 || n_classes <= 0 || n_features <= 0)
		 {
			 SPI_finish();
			 spi_finished = true;
			 ereport(ERROR,
				 (errmsg("corrupt model id=%d", model_id)));
		 }

		 payload = (Size) VARSIZE_ANY_EXHDR(model_bytea);
		 trees_off = (Size) rf->trees_offset;
		 nodes_off = (Size) rf->nodes_offset;

		 if (trees_off > payload || nodes_off > payload)
		 {
			 SPI_finish();
			 spi_finished = true;
			 ereport(ERROR,
				 (errmsg("model offsets OOB for id=%d",
					 model_id)));
		 }
		 if (!safe_add_mul_size(trees_off, (Size) n_trees,
					sizeof(RFTreeFlat), &req_trees))
		 {
			 SPI_finish();
			 spi_finished = true;
			 ereport(ERROR,
				 (errmsg("tree header overflow id=%d",
					 model_id)));
		 }
		 if (payload < Max(req_trees, nodes_off))
		 {
			 SPI_finish();
			 spi_finished = true;
			 ereport(ERROR,
				 (errmsg("model bounds invalid id=%d",
					 model_id)));
		 }

		 if (features->dim != n_features)
		 {
			 SPI_finish();
			 spi_finished = true;
			 ereport(ERROR,
				 (errmsg("feature dim %d, need %d (id=%d)",
					 features->dim, n_features,
					 model_id)));
		 }

		 blob_trees = (const RFTreeFlat *) (rf_data + rf->trees_offset);
		 blob_nodes = (const RFNodeFlat *) (rf_data + rf->nodes_offset);
		 node_capacity =
			 (VARSIZE_ANY_EXHDR(model_bytea) - rf->nodes_offset) /
			 (int) sizeof(RFNodeFlat);
		 if (node_capacity <= 0)
		 {
			 SPI_finish();
			 spi_finished = true;
			 ereport(ERROR,
				 (errmsg("node array bounds invalid id=%d",
					 model_id)));
		 }

		 if (n_classes <= 0)
		 {
			 SPI_finish();
			 spi_finished = true;
			 ereport(ERROR, (errmsg("invalid n_classes")));
		 }

		 if (ml_gpu_call_use_gpu(&gpu_state))
		 {
			 int	gpu_class = -1;

			 if (neurondb_gpu_rf_predict(rf, blob_trees, blob_nodes,
					     node_capacity,
					     feature_vector,
					     n_features,
					     &gpu_class,
					     &gpu_errstr))
			 {
				 prediction = (double) gpu_class;
				 used_gpu = true;
			 }
			 else if (!neurondb_gpu_fail_open)
			 {
				 SPI_finish();
				 spi_finished = true;
				 if (gpu_errstr)
					 ereport(ERROR,
						 (errmsg("random_forest GPU predict failed: %s",
							 gpu_errstr)));
				 ereport(ERROR,
					 (errmsg("random_forest GPU predict failed")));
			 }
		 }

		 if (!used_gpu)
		 {
			 double	  *votes;

			 votes = (double *) palloc0(sizeof(double) *
					(Size) n_classes);

			 for (t = 0; t < n_trees; t++)
			 {
				 int	class_pred;

				 CHECK_FOR_INTERRUPTS();

				 if (blob_trees[t].offset_to_nodes < 0 ||
					 blob_trees[t].num_nodes <= 0 ||
					 blob_trees[t].offset_to_nodes +
					   blob_trees[t].num_nodes > node_capacity)
				 {
					 pfree(votes);
					 SPI_finish();
					 spi_finished = true;
					 ereport(ERROR,
						 (errmsg("tree OOB t=%d id=%d",
							 t, model_id)));
				 }

				 class_pred =
					 (int) rf_tree_predict_flat_safe(
					   blob_nodes +
					   blob_trees[t].offset_to_nodes,
					   blob_trees[t].num_nodes,
					   n_features,
					   feature_vector);

				 if (votes &&
					 class_pred >= 0 &&
					 class_pred < n_classes)
					 votes[class_pred] += 1.0;
			 }

			 {
				 double	maxv;
				 int	maxi;

				 maxv = votes[0];
				 maxi = 0;
				 for (i = 1; i < n_classes; i++)
				 {
					 if (votes[i] > maxv)
					 {
						 maxv = votes[i];
						 maxi = i;
					 }
				 }
				 prediction = (double) maxi;
			 }

			 pfree(votes);
		 }
	 }

	 SPI_finish();
	 spi_finished = true;
	 }
	 PG_CATCH();
	 {
		 if (!spi_finished)
		 {
			 SPI_finish();
			 spi_finished = true;
		 }
		 if (gpu_errstr)
			 pfree(gpu_errstr);
		 ml_gpu_call_end(&gpu_state);
		 PG_RE_THROW();
	 }
	 PG_END_TRY();

	 if (gpu_errstr)
		 pfree(gpu_errstr);
	 ml_gpu_call_end(&gpu_state);

	 PG_RETURN_FLOAT8(prediction);
}
 
 /** SQL: evaluate_random_forest */
 PG_FUNCTION_INFO_V1(evaluate_random_forest);
 Datum
 evaluate_random_forest(PG_FUNCTION_ARGS)
 {
	 text	   *table_name;
	 text	   *feature_col;
	 text	   *label_col;
	 int		model_id;
	 char	   *tbl_str;
	 char	   *feat_str;
	 char	   *label_str;
	 StringInfoData query;
	 int		ret;
	 int		nvec;
	 int		i;
	 Vector	   *vec;
	 Datum	   *result_datums;
	 ArrayType  *result_array;
	 float	  **X;
	 double	   *y;
	 double		acc;
	 double		prec;
	 double		recall;
	 double		f1;
	 int	   *label_values;
	 int		num_classes;
	 int		model_n_features;
	 bytea	   *model_bytea;
	 const RFModelFlat *rf_model;
	 const RFTreeFlat  *blob_trees;
	 const RFNodeFlat  *blob_nodes;
	 int		n_trees;
	 int		n_classes;
	 int		n_features;
	 int		node_capacity;
 
	 model_id = PG_GETARG_INT32(3);
	 table_name = PG_GETARG_TEXT_PP(0);
	 feature_col = PG_GETARG_TEXT_PP(1);
	 label_col = PG_GETARG_TEXT_PP(2);
 
	 if (model_id < 0)
		 ereport(ERROR, (errmsg("model_id must be non-negative")));
 
	 tbl_str = text_to_cstring(table_name);
	 feat_str = text_to_cstring(feature_col);
	 label_str = text_to_cstring(label_col);
 
	 if (SPI_connect() != SPI_OK_CONNECT)
		 ereport(ERROR,
			 (errmsg("SPI_connect failed for evaluate")));
 
	 {
		 const char *sql;
		 Oid		argtypes[1];
		 Datum		values[1];
 
		 sql = "SELECT model_data "
			   "FROM neurondb.ml_models WHERE model_id = $1";
		 argtypes[0] = INT4OID;
		 values[0] = Int32GetDatum(model_id);
		 ret = SPI_execute_with_args(sql, 1, argtypes,
						 values, NULL, true, 1);
	 }
	 if (ret != SPI_OK_SELECT || SPI_processed == 0)
	 {
		 SPI_finish();
		 ereport(ERROR,
			 (errmsg("model %d not found", model_id)));
	 }
 
	 model_bytea = (bytea *) DatumGetPointer(
		 SPI_getbinval(SPI_tuptable->vals[0],
				   SPI_tuptable->tupdesc, 1, NULL));
	 if (!model_bytea ||
		 VARSIZE_ANY_EXHDR(model_bytea) < sizeof(RFModelFlat))
	 {
		 SPI_finish();
		 ereport(ERROR,
			 (errmsg("model %d truncated or NULL", model_id)));
	 }
 
	 {
		 const char *rf_data;
		 Size	payload;
		 Size	trees_off;
		 Size	nodes_off;
		 Size	req_trees;
 
		 rf_data = VARDATA(model_bytea);
		 rf_model = (const RFModelFlat *) rf_data;
 
		 n_trees = rf_model->n_trees;
		 n_classes = rf_model->n_classes;
		 n_features = rf_model->n_features;
		 model_n_features = n_features;
 
		 if (n_trees <= 0 || n_classes <= 0 || n_features <= 0)
		 {
			 SPI_finish();
			 ereport(ERROR,
				 (errmsg("invalid model header id=%d",
					 model_id)));
		 }
 
		 payload = (Size) VARSIZE_ANY_EXHDR(model_bytea);
		 trees_off = (Size) rf_model->trees_offset;
		 nodes_off = (Size) rf_model->nodes_offset;
 
		 if (trees_off > payload || nodes_off > payload)
		 {
			 SPI_finish();
			 ereport(ERROR,
				 (errmsg("model offsets OOB id=%d",
					 model_id)));
		 }
		 if (!safe_add_mul_size(trees_off, (Size) n_trees,
						sizeof(RFTreeFlat), &req_trees))
		 {
			 SPI_finish();
			 ereport(ERROR,
				 (errmsg("tree header overflow id=%d",
					 model_id)));
		 }
		 if (payload < Max(req_trees, nodes_off))
		 {
			 SPI_finish();
			 ereport(ERROR,
				 (errmsg("model bounds invalid id=%d",
					 model_id)));
		 }
	 }
 
	 blob_trees = (const RFTreeFlat *)
		 ((VARDATA(model_bytea)) + rf_model->trees_offset);
	 blob_nodes = (const RFNodeFlat *)
		 ((VARDATA(model_bytea)) + rf_model->nodes_offset);
	 node_capacity =
		 ((VARSIZE_ANY_EXHDR(model_bytea) - rf_model->nodes_offset) /
		  (int) sizeof(RFNodeFlat));
	 if (node_capacity <= 0)
	 {
		 SPI_finish();
		 ereport(ERROR,
			 (errmsg("node array bounds invalid id=%d",
				 model_id)));
	 }
	 SPI_finish();
 
	 if (SPI_connect() != SPI_OK_CONNECT)
		 ereport(ERROR,
			 (errmsg("SPI_connect failed for data read")));
 
	 initStringInfo(&query);
	 appendStringInfo(&query,
			  "SELECT %s, %s FROM %s "
			  "WHERE %s IS NOT NULL AND %s IS NOT NULL",
			  feat_str, label_str, tbl_str, feat_str, label_str);
 
	 ret = SPI_execute(query.data, true, 0);
	 if (ret != SPI_OK_SELECT)
	 {
		 SPI_finish();
		 ereport(ERROR,
			 (errmsg("eval query failed: %s", query.data)));
	 }
 
	 nvec = SPI_processed;
	 X = (float **) palloc(sizeof(float *) * nvec);
	 y = (double *) palloc(sizeof(double) * nvec);
 
	 for (i = 0; i < nvec; i++)
	 {
		 HeapTuple	tup;
		 TupleDesc	tupdesc;
		 Datum		feat_datum;
		 Datum		label_datum;
		 bool		feat_null;
		 bool		label_null;
		 Oid		label_type;
 
		 tup = SPI_tuptable->vals[i];
		 tupdesc = SPI_tuptable->tupdesc;
 
		 feat_datum = SPI_getbinval(tup, tupdesc, 1, &feat_null);
		 if (feat_null)
			 goto eval_feature_null;
 
		 vec = DatumGetVector(feat_datum);
		 if (vec->dim != model_n_features)
		 {
			 int	j;
 
			 SPI_finish();
			 for (j = 0; j < i; j++)
				 if (X[j])
					 pfree(X[j]);
			 if (X)
				 pfree(X);
			 if (y)
				 pfree(y);
			 ereport(ERROR,
				 (errmsg("dim mismatch row %d has %d "
					 "expected %d (id=%d)",
					 i, vec->dim, model_n_features,
					 model_id)));
		 }
 
		 X[i] = (float *) palloc(sizeof(float) * model_n_features);
		 memcpy(X[i], vec->data, sizeof(float) * model_n_features);
 
		 label_datum = SPI_getbinval(tup, tupdesc, 2, &label_null);
		 if (label_null)
			 goto eval_label_null;
 
		 label_type = SPI_gettypeid(tupdesc, 2);
		 if (label_type == INT2OID || label_type == INT4OID)
			 y[i] = (double) DatumGetInt32(label_datum);
		 else if (label_type == INT8OID)
			 y[i] = (double) DatumGetInt64(label_datum);
		 else
			 y[i] = DatumGetFloat8(label_datum);
	 }
	 SPI_finish();
 
	 {
		 int	maxlbl;
		 int	minlbl;
		 int	count;
		 int	idx;
		 int	v;
		 int    *tmp;
 
		 maxlbl = 0;
		 minlbl = 0x7fffffff;
		 count = 0;
		 idx = 0;
		 tmp = NULL;
 
		 for (i = 0; i < nvec; i++)
		 {
			 v = (int) y[i];
			 if (v > maxlbl)
				 maxlbl = v;
			 if (v < minlbl)
				 minlbl = v;
		 }
		 num_classes = maxlbl - minlbl + 1;
		 label_values =
			 (int *) palloc0(sizeof(int) * num_classes);
		 for (i = 0; i < nvec; i++)
			 label_values[(int) y[i] - minlbl] = 1;
 
		 for (i = 0; i < num_classes; i++)
			 if (label_values[i])
				 count++;
		 num_classes = count;
 
		 if (num_classes > 0)
		 {
			 tmp = (int *) palloc(sizeof(int) * num_classes);
			 idx = 0;
			 for (i = 0; i < maxlbl - minlbl + 1; i++)
			 {
				 if (label_values[i])
					 tmp[idx++] = i + minlbl;
			 }
			 pfree(label_values);
			 label_values = tmp;
		 }
	 }
 
	 {
		 int    *confmat;
		 int	correct;
		 int	total;
		 int	c;
		 int	j;
 
		 confmat = (int *) palloc0(sizeof(int) *
					   num_classes * num_classes);
		 correct = 0;
		 total = 0;
 
		 for (i = 0; i < nvec; i++)
		 {
			 float   *x;
			 double	pred;
			 double	truth;
			 int	truth_cls;
			 int	pred_cls;
			 int	k;
			 double *votes;
			 int	t;
 
			 CHECK_FOR_INTERRUPTS();
 
			 x = X[i];
			 pred = 0.0;
			 truth = y[i];
			 truth_cls = -1;
			 pred_cls = -1;
 
			 votes = (n_classes > 0)
				 ? (double *) palloc0(sizeof(double) *
							   n_classes)
				 : NULL;
 
			 for (t = 0; t < n_trees; t++)
			 {
				 int	class_pred;
 
				 if (blob_trees[t].offset_to_nodes < 0 ||
					 blob_trees[t].num_nodes <= 0 ||
					 blob_trees[t].offset_to_nodes +
					   blob_trees[t].num_nodes >
					   node_capacity)
				 {
					 int	l;
 
					 if (votes)
						 pfree(votes);
					 for (l = 0; l < nvec; l++)
						 if (X[l])
							 pfree(X[l]);
					 if (X)
						 pfree(X);
					 if (y)
						 pfree(y);
					 if (label_values)
						 pfree(label_values);
					 if (confmat)
						 pfree(confmat);
					 ereport(ERROR,
						 (errmsg("tree bounds error "
							 "id=%d", model_id)));
				 }
 
				 class_pred =
					 (int) rf_tree_predict_flat_safe(
					   blob_nodes +
						 blob_trees[t].offset_to_nodes,
					   blob_trees[t].num_nodes,
					   model_n_features,
					   x);
 
				 if (votes &&
					 class_pred >= 0 &&
					 class_pred < n_classes)
					 votes[class_pred] += 1.0;
			 }
 
			 if (votes && n_classes > 0)
			 {
				 double	maxv;
				 int	maxi;
				 int	kk;
 
				 maxv = votes[0];
				 maxi = 0;
				 for (kk = 1; kk < n_classes; kk++)
				 {
					 if (votes[kk] > maxv)
					 {
						 maxv = votes[kk];
						 maxi = kk;
					 }
				 }
				 pred = (double) maxi;
			 }
			 if (votes)
				 pfree(votes);
 
			 for (k = 0; k < num_classes; k++)
			 {
				 if ((int) truth == label_values[k])
					 truth_cls = k;
				 if ((int) pred == label_values[k])
					 pred_cls = k;
			 }
			 if (truth_cls >= 0 && pred_cls >= 0)
				 confmat[truth_cls * num_classes +
					 pred_cls] += 1;
		 }
 
		 {
			 double	macro_prec;
			 double	macro_recall;
			 double	macro_f1;
 
			 macro_prec = 0.0;
			 macro_recall = 0.0;
			 macro_f1 = 0.0;
 
			 for (c = 0; c < num_classes; c++)
			 {
				 int	tp;
				 int	fn;
				 int	fp;
				 int	support;
 
				 tp = confmat[c * num_classes + c];
				 fn = 0;
				 fp = 0;
				 support = 0;
 
				 for (j = 0; j < num_classes; j++)
				 {
					 if (j != c)
					 {
						 fn += confmat[c *
								   num_classes +
								   j];
						 fp += confmat[j *
								   num_classes +
								   c];
					 }
					 support += confmat[c *
								num_classes + j];
				 }
 
				 {
					 double	prec_i;
					 double	recall_i;
					 double	f1_i;
 
					 prec_i = (tp + fp > 0)
						 ? ((double) tp / (tp + fp))
						 : 0.0;
					 recall_i = (tp + fn > 0)
						 ? ((double) tp / (tp + fn))
						 : 0.0;
					 f1_i = (prec_i + recall_i > 0)
						 ? (2.0 * prec_i * recall_i /
							(prec_i + recall_i))
						 : 0.0;
 
					 macro_prec += prec_i;
					 macro_recall += recall_i;
					 macro_f1 += f1_i;
				 }
 
				 correct += tp;
				 total += support;
			 }
 
			 acc = (total > 0) ?
				   ((double) correct / total) : 0.0;
			 prec = (num_classes > 0) ?
					(macro_prec / num_classes) : 0.0;
			 recall = (num_classes > 0) ?
				  (macro_recall / num_classes) : 0.0;
			 f1 = (num_classes > 0) ?
				  (macro_f1 / num_classes) : 0.0;
 
			 result_datums = (Datum *) palloc(sizeof(Datum) * 4);
			 result_datums[0] = Float8GetDatum(acc);
			 result_datums[1] = Float8GetDatum(prec);
			 result_datums[2] = Float8GetDatum(recall);
			 result_datums[3] = Float8GetDatum(f1);
 
			 result_array =
				 construct_array(result_datums, 4,
						 FLOAT8OID, 8,
						 FLOAT8PASSBYVAL, 'd');
 
			 for (i = 0; i < nvec; i++)
				 if (X[i])
					 pfree(X[i]);
			 if (X)
				 pfree(X);
			 if (y)
				 pfree(y);
			 if (label_values)
				 pfree(label_values);
			 if (confmat)
				 pfree(confmat);
			 if (result_datums)
				 pfree(result_datums);
 
			 PG_RETURN_ARRAYTYPE_P(result_array);
		 }
	 }
 
 eval_feature_null:
	 if (X)
	 {
		 int	k;
 
		 for (k = 0; k < nvec; k++)
			 if (X[k])
				 pfree(X[k]);
		 pfree(X);
	 }
	 if (y)
		 pfree(y);
	 ereport(ERROR,
		 (errmsg("null feature vector for model_id %d",
			 model_id)));
 
 eval_label_null:
	 if (X)
	 {
		 int	k;
 
		 for (k = 0; k < nvec; k++)
			 if (X[k])
				 pfree(X[k]);
		 pfree(X);
	 }
	 if (y)
		 pfree(y);
	 ereport(ERROR,
		 (errmsg("null label for model_id %d", model_id)));
 }
 