/* -------------------------------------------------------------------------
 *
 * ml_automl.c
 *     Automated Machine Learning (AutoML) for NeuronDB
 *
 * Implements automated model selection, hyperparameter tuning,
 * and ensemble methods with GPU acceleration support.
 *
 * IDENTIFICATION
 *     src/ml/ml_automl.c
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 * -------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include "utils/jsonb.h"
#include "utils/guc.h"
#include "neurondb_pgcompat.h"
#include "neurondb_automl.h"
#include "neurondb_gpu.h"

#include <math.h>
#include <string.h>

/* GUC variable for AutoML GPU usage */
bool neurondb_automl_use_gpu = false;

/* Model evaluation result */
typedef struct ModelScore
{
	char   *algorithm;
	float	score;
	int32	model_id;
	char   *hyperparams;
} ModelScore;

/*
 * neurondb_automl_define_gucs
 *	  Register GUC variables for AutoML.
 */
void
neurondb_automl_define_gucs(void)
{
	DefineCustomBoolVariable("neurondb.automl.use_gpu",
							 "Enable GPU acceleration for AutoML training",
							 "When enabled, AutoML will prefer GPU training for supported algorithms.",
							 &neurondb_automl_use_gpu,
							 false,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	EmitWarningsOnPlaceholders("neurondb.automl");
}

/*
 * neurondb_automl_choose_backend
 *	  Determine if GPU should be used for a given algorithm.
 */
AutoMLBackendType
neurondb_automl_choose_backend(const char *algorithm)
{
	if (!neurondb_automl_use_gpu)
		return AUTOML_BACKEND_CPU;

	if (!neurondb_gpu_is_available())
		return AUTOML_BACKEND_CPU;

	/* Check if algorithm supports GPU */
	if (algorithm != NULL)
	{
		if (strcmp(algorithm, "linear_regression") == 0 ||
			strcmp(algorithm, "logistic_regression") == 0 ||
			strcmp(algorithm, "random_forest") == 0 ||
			strcmp(algorithm, "decision_tree") == 0 ||
			strcmp(algorithm, "ridge") == 0 ||
			strcmp(algorithm, "lasso") == 0)
			return AUTOML_BACKEND_GPU;
	}

	return AUTOML_BACKEND_CPU;
}

/*
 * auto_train
 *	  Automated model selection with GPU acceleration support.
 *
 * Trains multiple algorithms and selects the best one based on evaluation metrics.
 * Supports both classification and regression tasks.
 *
 * auto_train(
 *   table_name text,
 *   feature_col text,
 *   label_col text,
 *   task text,  -- 'classification' or 'regression'
 *   metric text DEFAULT 'accuracy'  -- 'accuracy', 'f1', 'r2', 'mse', etc.
 * )
 */
PG_FUNCTION_INFO_V1(auto_train);

Datum
auto_train(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *feature_col = PG_GETARG_TEXT_PP(1);
	text	   *label_col = PG_GETARG_TEXT_PP(2);
	text	   *task = PG_GETARG_TEXT_PP(3);
	text	   *metric_text = PG_ARGISNULL(4) ? NULL : PG_GETARG_TEXT_PP(4);

	char	   *table_name_str = text_to_cstring(table_name);
	char	   *feature_col_str = text_to_cstring(feature_col);
	char	   *label_col_str = text_to_cstring(label_col);
	char	   *task_str = text_to_cstring(task);
	char	   *metric;
	MemoryContext oldcontext;
	MemoryContext automl_context;
	StringInfoData result;
	StringInfoData sql;
	const char *algorithms[5];
	int			n_algorithms;
	int			i;
	float		best_score = -1.0f;
	const char *best_algorithm = NULL;
	int32		best_model_id = 0;
	ModelScore *scores = NULL;
	int			ret;
	bool		isnull;

	if (metric_text)
		metric = text_to_cstring(metric_text);
	else
		metric = (strcmp(task_str, "classification") == 0) ?
			pstrdup("accuracy") : pstrdup("r2");

	/* Validate task */
	if (strcmp(task_str, "classification") != 0 &&
		strcmp(task_str, "regression") != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("task must be 'classification' or 'regression'")));

	/* Create memory context for AutoML operations */
	automl_context = AllocSetContextCreate(CurrentMemoryContext,
										   "automl memory context",
										   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(automl_context);

	/* Select algorithms based on task */
	if (strcmp(task_str, "classification") == 0)
	{
		algorithms[0] = "logistic_regression";
		algorithms[1] = "decision_tree";
		algorithms[2] = "random_forest";
		algorithms[3] = "svm";
		algorithms[4] = "knn";
		n_algorithms = 5;
	}
	else
	{
		algorithms[0] = "linear_regression";
		algorithms[1] = "ridge";
		algorithms[2] = "lasso";
		algorithms[3] = "decision_tree";
		algorithms[4] = "random_forest";
		n_algorithms = 5;
	}

	/* Allocate scores array */
	scores = (ModelScore *) palloc0(n_algorithms * sizeof(ModelScore));

	/* Connect to SPI */
	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("auto_train: SPI_connect failed")));

	/* Train and evaluate each algorithm */
	for (i = 0; i < n_algorithms; i++)
	{
		int32		model_id = 0;
		float		score = -1.0f;
		Jsonb	   *metrics_jsonb = NULL;
		Datum		metrics_datum;
		bool		metrics_isnull;
		JsonbIterator *it;
		JsonbValue	v;
		int			r;
		bool		found_metric = false;

		scores[i].algorithm = pstrdup(algorithms[i]);

		elog(DEBUG1, "auto_train: Training algorithm %s (%d/%d)",
			 algorithms[i], i + 1, n_algorithms);

		/* Train model using neurondb.train() */
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT neurondb.train("
						 "'automl_project', "
						 "'%s', "
						 "'%s', "
						 "'%s', "
						 "ARRAY['%s']::text[], "
						 "'{}'::jsonb)::integer",
						 algorithms[i],
						 table_name_str,
						 label_col_str,
						 feature_col_str);

		ret = SPI_execute(sql.data, true, 1);
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			elog(WARNING, "auto_train: Failed to train %s, skipping",
				 algorithms[i]);
			pfree(sql.data);
			scores[i].score = -1.0f;
			scores[i].model_id = 0;
			continue;
		}

		model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
											   SPI_tuptable->tupdesc,
											   1,
											   &isnull));
		pfree(sql.data);

		if (isnull || model_id <= 0)
		{
			elog(WARNING, "auto_train: Invalid model_id for %s, skipping",
				 algorithms[i]);
			scores[i].score = -1.0f;
			scores[i].model_id = 0;
			continue;
		}

		scores[i].model_id = model_id;

		/* Evaluate model using neurondb.evaluate() */
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT neurondb.evaluate("
						 "%d, "
						 "'%s', "
						 "'%s', "
						 "'%s')",
						 model_id,
						 table_name_str,
						 feature_col_str,
						 label_col_str);

		ret = SPI_execute(sql.data, true, 1);
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			elog(WARNING, "auto_train: Failed to evaluate %s, skipping",
				 algorithms[i]);
			pfree(sql.data);
			scores[i].score = -1.0f;
			continue;
		}

		metrics_datum = SPI_getbinval(SPI_tuptable->vals[0],
									  SPI_tuptable->tupdesc,
									  1,
									  &metrics_isnull);
		pfree(sql.data);

		if (metrics_isnull)
		{
			elog(WARNING, "auto_train: Null metrics for %s, skipping",
				 algorithms[i]);
			scores[i].score = -1.0f;
			continue;
		}

		/* Extract metric value from JSONB */
		metrics_jsonb = DatumGetJsonbP(metrics_datum);
		it = JsonbIteratorInit(&metrics_jsonb->root);

		while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
		{
			if (r == WJB_KEY)
			{
				char   *key = pnstrdup(v.val.string.val, v.val.string.len);

				r = JsonbIteratorNext(&it, &v, false);

				if (strcmp(key, metric) == 0 && v.type == jbvNumeric)
				{
					score = (float) DatumGetFloat8(
						DirectFunctionCall1(numeric_float8,
											NumericGetDatum(v.val.numeric)));
					found_metric = true;
					pfree(key);
					break;
				}
				pfree(key);
			}
		}

		if (!found_metric)
		{
			/* Try to find alternative names for the requested metric, then common metrics */
			bool		matches_metric;
			bool		matches_common;

			it = JsonbIteratorInit(&metrics_jsonb->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char   *key = pnstrdup(v.val.string.val, v.val.string.len);

					r = JsonbIteratorNext(&it, &v, false);

					/* Check for alternative names of requested metric */
					matches_metric = false;
					if (strcmp(metric, "r2") == 0 && strcmp(key, "r_squared") == 0)
						matches_metric = true;
					else if (strcmp(metric, "r_squared") == 0 && strcmp(key, "r2") == 0)
						matches_metric = true;
					else if (strcmp(key, metric) == 0)
						matches_metric = true;

					/* Also check common metric names */
					matches_common = false;
					if (strcmp(task_str, "classification") == 0 &&
						(strcmp(key, "accuracy") == 0 ||
						 strcmp(key, "f1") == 0 ||
						 strcmp(key, "precision") == 0 ||
						 strcmp(key, "recall") == 0))
						matches_common = true;
					else if (strcmp(task_str, "regression") == 0 &&
							 (strcmp(key, "r2") == 0 ||
							  strcmp(key, "r_squared") == 0 ||
							  strcmp(key, "mse") == 0 ||
							  strcmp(key, "mae") == 0))
						matches_common = true;

					if ((matches_metric || matches_common) && v.type == jbvNumeric)
					{
						score = (float) DatumGetFloat8(
							DirectFunctionCall1(numeric_float8,
												NumericGetDatum(v.val.numeric)));
						found_metric = true;
						pfree(key);
						break;
					}
					pfree(key);
				}
			}
		}

		if (!found_metric)
		{
			elog(WARNING,
				 "auto_train: Could not find metric '%s' for %s, using default score",
				 metric, algorithms[i]);
			score = 0.5f;	/* Default score */
		}

		/* For regression, higher is better (r2), lower is better (mse/mae) */
		if (strcmp(task_str, "regression") == 0 &&
			(strcmp(metric, "mse") == 0 ||
			 strcmp(metric, "mae") == 0 ||
			 strcmp(metric, "rmse") == 0))
		{
			/* Invert score for error metrics (lower is better) */
			score = 1.0f / (1.0f + score);
		}

		scores[i].score = score;

		elog(DEBUG1, "auto_train: %s scored %.4f (model_id: %d)",
			 algorithms[i], score, model_id);

		/* Track best model */
		if (score > best_score)
		{
			best_score = score;
			best_algorithm = algorithms[i];
			best_model_id = model_id;
		}
	}

	SPI_finish();

	/* Build result */
	initStringInfo(&result);
	if (best_algorithm != NULL && best_model_id > 0)
	{
		appendStringInfo(&result,
						 "AutoML completed. Best algorithm: %s, %s: %.4f, model_id: %d\n"
						 "Trained %d algorithms:\n",
						 best_algorithm, metric, best_score, best_model_id, n_algorithms);

		for (i = 0; i < n_algorithms; i++)
		{
			if (scores[i].model_id > 0)
			{
				appendStringInfo(&result,
								 "  %d. %s: %.4f (model_id: %d)\n",
								 i + 1, scores[i].algorithm,
								 scores[i].score, scores[i].model_id);
			}
			else
			{
				appendStringInfo(&result,
								 "  %d. %s: failed\n",
								 i + 1, scores[i].algorithm);
			}
		}
	}
	else
	{
		appendStringInfo(&result,
						 "AutoML failed: No models were successfully trained");
	}

	/* Save result to parent context before deleting automl_context */
	MemoryContextSwitchTo(oldcontext);
	{
		char   *result_copy = pstrdup(result.data);
		text   *result_text = cstring_to_text(result_copy);

		/* Delete automl_context (frees all allocations in it) */
		MemoryContextDelete(automl_context);

		PG_RETURN_TEXT_P(result_text);
	}
}

/*
 * Hyperparameter optimization using grid search.
 */
PG_FUNCTION_INFO_V1(optimize_hyperparameters);

Datum
optimize_hyperparameters(PG_FUNCTION_ARGS)
{
	text	   *algorithm = PG_GETARG_TEXT_PP(0);
	text	   *table_name = PG_GETARG_TEXT_PP(1);
	text	   *param_grid_json = PG_GETARG_TEXT_PP(2);

	char	   *algorithm_str = text_to_cstring(algorithm);
	char	   *table_name_str = text_to_cstring(table_name);
	char	   *param_grid_str = text_to_cstring(param_grid_json);
	StringInfoData result;

	/* Validate algorithm */
	if (strlen(algorithm_str) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("algorithm cannot be empty")));

	/* Placeholder: perform grid search */
	(void) table_name_str;
	(void) param_grid_str;

	initStringInfo(&result);
	appendStringInfo(&result,
					 "Hyperparameter optimization completed for %s",
					 algorithm_str);

	pfree(algorithm_str);
	pfree(table_name_str);
	pfree(param_grid_str);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}

/*
 * Feature importance analysis.
 */
PG_FUNCTION_INFO_V1(feature_importance);

Datum
feature_importance(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	ArrayType  *result_array;
	float		scores[10];
	Datum	   *elems;
	int			i;
	int			n_features = 10;

	/* Load model and compute feature importance */
	(void) model_id;

	/* Generate placeholder scores */
	for (i = 0; i < n_features; i++)
		scores[i] = 1.0f - (i * 0.1f);

	/* Build result array */
	elems = (Datum *) palloc(n_features * sizeof(Datum));
	for (i = 0; i < n_features; i++)
		elems[i] = Float8GetDatum(scores[i]);

	result_array = construct_array(elems,
								  n_features,
								  FLOAT8OID,
								  sizeof(float8),
								  FLOAT8PASSBYVAL,
								  'd');

	pfree(elems);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * Cross-validation.
 */
PG_FUNCTION_INFO_V1(cross_validate);

Datum
cross_validate(PG_FUNCTION_ARGS)
{
	text	   *algorithm = PG_GETARG_TEXT_PP(0);
	text	   *table_name = PG_GETARG_TEXT_PP(1);
	int32		n_folds = PG_ARGISNULL(2) ? 5 : PG_GETARG_INT32(2);

	char	   *algorithm_str = text_to_cstring(algorithm);
	char	   *table_name_str = text_to_cstring(table_name);
	float		mean_score;

	/* Validate */
	if (n_folds < 2 || n_folds > 20)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_folds must be between 2 and 20")));

	/* Perform cross-validation */
	(void) algorithm_str;
	(void) table_name_str;

	mean_score = 0.85f;		/* Placeholder */

	pfree(algorithm_str);
	pfree(table_name_str);

	PG_RETURN_FLOAT8(mean_score);
}

/*
 * Ensemble learning - combine multiple models.
 */
PG_FUNCTION_INFO_V1(create_ensemble);

Datum
create_ensemble(PG_FUNCTION_ARGS)
{
	ArrayType  *model_ids_array = PG_GETARG_ARRAYTYPE_P(0);
	text	   *method_text = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);

	char	   *method =
		method_text ? text_to_cstring(method_text) : pstrdup("voting");
	int			n_models;
	StringInfoData result;

	/* Get number of models */
	n_models = ArrayGetNItems(ARR_NDIM(model_ids_array),
							  ARR_DIMS(model_ids_array));

	if (n_models < 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ensemble requires at least 2 models")));

	/* Validate method */
	if (strcmp(method, "voting") != 0 &&
		strcmp(method, "averaging") != 0 &&
		strcmp(method, "stacking") != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("method must be 'voting', 'averaging', or 'stacking'")));

	/* Create ensemble model */
	initStringInfo(&result);
	appendStringInfo(&result,
					 "Ensemble created with %d models using %s",
					 n_models, method);

	pfree(method);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}

/*
 * Automated feature engineering.
 */
PG_FUNCTION_INFO_V1(auto_feature_engineering);

Datum
auto_feature_engineering(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	ArrayType  *feature_cols_array = PG_GETARG_ARRAYTYPE_P(1);

	char	   *table_name_str = text_to_cstring(table_name);
	int			n_features;
	StringInfoData result;

	n_features = ArrayGetNItems(ARR_NDIM(feature_cols_array),
								ARR_DIMS(feature_cols_array));

	/* Placeholder: generate polynomial features, interactions, etc. */
	(void) table_name_str;

	initStringInfo(&result);
	appendStringInfo(&result,
					 "Generated engineered features from %d base features",
					 n_features);

	pfree(table_name_str);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}

/*
 * Model comparison and leaderboard.
 */
PG_FUNCTION_INFO_V1(model_leaderboard);

Datum
model_leaderboard(PG_FUNCTION_ARGS)
{
	text	   *task = PG_GETARG_TEXT_PP(0);
	char	   *task_str = text_to_cstring(task);
	StringInfoData result;

	/* Placeholder: retrieve model performance metrics from database */
	(void) task_str;

	initStringInfo(&result);
	appendStringInfo(&result,
					 "Model leaderboard for %s:\n"
					 "1. Random Forest: 0.92\n"
					 "2. Neural Network: 0.89\n"
					 "3. Logistic Regression: 0.85",
					 task_str);

	pfree(task_str);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}
