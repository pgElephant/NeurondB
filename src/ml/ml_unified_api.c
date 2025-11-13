/*-------------------------------------------------------------------------
 *
 * ml_unified_api.c
 *    Unified SQL API for Machine Learning (PostgresML-compatible)
 *
 * Provides a simplified, unified interface for ML operations:
 * - neurondb.train() - Train any algorithm with one SQL call
 * - neurondb.predict() - Predict with trained model
 * - neurondb.deploy() - Deploy model to production
 * - neurondb.load_model() - Load external models
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_unified_api.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/timestamp.h"
#include "utils/memutils.h"
#include "utils/lsyscache.h"

#include <time.h>

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_gpu_bridge.h"

/* PG_MODULE_MAGIC is in neurondb.c only */

PG_FUNCTION_INFO_V1(neurondb_train);
PG_FUNCTION_INFO_V1(neurondb_predict);
PG_FUNCTION_INFO_V1(neurondb_deploy);
PG_FUNCTION_INFO_V1(neurondb_load_model);

/* Helper: clean up context and SPI */
static void
neurondb_cleanup(MemoryContext oldcontext,
	MemoryContext callcontext,
	bool finish_spi)
{
	if (finish_spi)
		SPI_finish();
	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(callcontext);
}

/* Helper: quote literal for SQL safely and free after use */
static char *
neurondb_quote_literal_cstr(const char *str)
{
	char *ret;
	text *txt = cstring_to_text(str);
	ret = TextDatumGetCString(
		DirectFunctionCall1(quote_literal, PointerGetDatum(txt)));
	pfree(txt);
	return ret;
}

/* Helper: determine model type from algorithm name */
static const char *
neurondb_get_model_type(const char *algorithm)
{
	if (algorithm == NULL)
		return "classification"; /* default */

	/* Classification algorithms */
	if (strcmp(algorithm, "logistic_regression") == 0 ||
		strcmp(algorithm, "random_forest") == 0 ||
		strcmp(algorithm, "svm") == 0 ||
		strcmp(algorithm, "decision_tree") == 0 ||
		strcmp(algorithm, "naive_bayes") == 0 ||
		strcmp(algorithm, "xgboost") == 0)
		return "classification";

	/* Regression algorithms */
	if (strcmp(algorithm, "linear_regression") == 0 ||
		strcmp(algorithm, "ridge") == 0 ||
		strcmp(algorithm, "lasso") == 0)
		return "regression";

	/* Clustering algorithms */
	if (strcmp(algorithm, "kmeans") == 0 ||
		strcmp(algorithm, "gmm") == 0 ||
		strcmp(algorithm, "minibatch_kmeans") == 0 ||
		strcmp(algorithm, "hierarchical") == 0)
		return "clustering";

	/* Dimensionality reduction */
	if (strcmp(algorithm, "pca") == 0 ||
		strcmp(algorithm, "opq") == 0)
		return "dimensionality_reduction";

	/* Default to classification */
	return "classification";
}

/* ----------
 * neurondb_train
 * Unified model training interface.
 * ----------
 */
Datum
neurondb_train(PG_FUNCTION_ARGS)
{
	text *project_name_text = PG_GETARG_TEXT_PP(0);
	text *algorithm_text = PG_GETARG_TEXT_PP(1);
	text *table_name_text = PG_GETARG_TEXT_PP(2);
	text *target_column_text = PG_GETARG_TEXT_PP(3);
	ArrayType *feature_columns_array =
		PG_ARGISNULL(4) ? NULL : PG_GETARG_ARRAYTYPE_P(4);
	Jsonb *hyperparams = PG_ARGISNULL(5) ? NULL : PG_GETARG_JSONB_P(5);
	const char **feature_names = NULL;
	int feature_name_count = 0;
	char *model_name = NULL;
	MLGpuTrainResult gpu_result;
	char *gpu_errmsg = NULL;

	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	StringInfoData feature_list;
	char *project_name;
	char *algorithm;
	char *table_name;
	char *target_column;
	int ret;
	int project_id = 0;
	int model_id = 0;
	bool isnull = false;
	int32 k_value = 5;
	int i;

	project_name = text_to_cstring(project_name_text);
	algorithm = text_to_cstring(algorithm_text);
	table_name = text_to_cstring(table_name_text);
	target_column = text_to_cstring(target_column_text);

	elog(DEBUG1,
		"neurondb.train: project=\"%s\", algorithm=\"%s\", "
		"table=\"%s\", "
		"target=\"%s\"",
		project_name,
		algorithm,
		table_name,
		target_column);

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
		"neurondb_train memory context",
		ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_connect failed")));

	/* Ensure ml_projects entry exists for the project */
	{
		const char *model_type = neurondb_get_model_type(algorithm);
		char *model_type_quoted = neurondb_quote_literal_cstr(model_type);

		initStringInfo(&sql);
		appendStringInfo(&sql,
			"INSERT INTO neurondb.ml_projects (project_name, model_type, "
			"description) "
			"VALUES (%s, %s, 'Auto-created by neurondb.train()') "
			"ON CONFLICT (project_name) DO UPDATE SET updated_at = "
			"CURRENT_TIMESTAMP "
			"RETURNING project_id",
			neurondb_quote_literal_cstr(project_name),
			model_type_quoted);
		pfree(model_type_quoted);
	}
	ret = SPI_execute(sql.data, false, 0);

	if ((ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING)
		|| SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("Failed to create/get project \"%s\"",
					project_name)));
	}
	project_id = DatumGetInt32(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	resetStringInfo(&sql);
	initStringInfo(&feature_list);

	/* Build feature list */
	if (feature_columns_array != NULL)
	{
		Oid elemtype = ARR_ELEMTYPE(feature_columns_array);
		int16 typlen;
		bool typbyval;
		char typalign;
		int ndims = ARR_NDIM(feature_columns_array);
		int *dims = ARR_DIMS(feature_columns_array);
		int nelems = ArrayGetNItems(ndims, dims);
		Datum *elem_values;
		bool *elem_nulls;

		get_typlenbyvalalign(elemtype, &typlen, &typbyval, &typalign);
		deconstruct_array(feature_columns_array,
			TEXTOID,
			typlen,
			typbyval,
			typalign,
			&elem_values,
			&elem_nulls,
			&nelems);

		for (i = 0; i < nelems; ++i)
		{
			if (!elem_nulls[i])
			{
				char *col = TextDatumGetCString(elem_values[i]);
				if (feature_list.len > 0)
					appendStringInfoString(
						&feature_list, ", ");
				appendStringInfoString(&feature_list, col);
				if (feature_names == NULL)
					feature_names = (const char **)palloc(
						sizeof(char *) * nelems);
				feature_names[feature_name_count++] =
					pstrdup(col);
				pfree(col);
			}
		}
		pfree(elem_values);
		pfree(elem_nulls);

		if (feature_list.len == 0)
			appendStringInfoString(&feature_list, "*");
	} else
	{
		appendStringInfoString(&feature_list, "*");
		feature_names = (const char **)palloc(sizeof(char *));
		feature_names[0] = pstrdup("*");
		feature_name_count = 1;
	}

	if (feature_name_count == 0)
	{
		feature_names = (const char **)palloc(sizeof(char *));
		feature_names[0] = pstrdup("*");
		feature_name_count = 1;
	}

	MemSet(&gpu_result, 0, sizeof(MLGpuTrainResult));
	model_name = psprintf("%s_%s", algorithm, project_name);
	if (ndb_gpu_try_train_model(algorithm,
		    project_name,
		    model_name,
		    table_name,
		    target_column,
		    feature_names,
		    feature_name_count,
		    hyperparams,
		    NULL,
		    NULL,
		    0,
		    0,
		    0,
		    &gpu_result,
		    &gpu_errmsg))
	{
		elog(DEBUG1,
			 "neurondb.train: GPU training succeeded, gpu_result.spec.metrics=%p",
			 (void *)gpu_result.spec.metrics);
		if (gpu_result.spec.metrics != NULL)
		{
			char *metrics_txt = DatumGetCString(
				DirectFunctionCall1(jsonb_out,
					JsonbPGetDatum(gpu_result.spec.metrics)));
			elog(DEBUG1,
				 "neurondb.train: GPU metrics content: %s",
				 metrics_txt);
			pfree(metrics_txt);
		}
		model_id = ml_catalog_register_model(&gpu_result.spec);
		gpu_result.model_id = model_id;
		elog(DEBUG1,
			"neurondb.train: GPU model_id=%d created "
			"successfully",
			model_id);
		ndb_gpu_free_train_result(&gpu_result);
		if (feature_names != NULL)
		{
			for (i = 0; i < feature_name_count; i++)
				pfree((void *)feature_names[i]);
			pfree(feature_names);
		}
		neurondb_cleanup(oldcontext, callcontext, true);
		pfree(model_name);
		PG_RETURN_INT32(model_id);
	}
	if (gpu_errmsg != NULL)
	{
		elog(WARNING,
			"neurondb: GPU training failed (%s), using CPU "
			"fallback",
			gpu_errmsg);
		pfree(gpu_errmsg);
	}
	ndb_gpu_free_train_result(&gpu_result);

	resetStringInfo(&sql);

	/* Algorithm branching */
	if (strcmp(algorithm, "linear_regression") == 0)
	{
		appendStringInfo(&sql,
			"SELECT train_linear_regression(%s, %s, %s)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_cstr(target_column));
	} else if (strcmp(algorithm, "logistic_regression") == 0)
	{
		int max_iters = 1000;
		double learning_rate = 0.01;
		double lambda = 0.001;

		if (hyperparams != NULL)
		{
			JsonbIterator *it;
			JsonbValue v;
			int r;

			it = JsonbIteratorInit(&hyperparams->root);
			while ((r = JsonbIteratorNext(&it, &v, false))
				!= WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val,
						v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "max_iters") == 0
						&& v.type == jbvNumeric)
						max_iters = DatumGetInt32(DirectFunctionCall1(
							numeric_int4,
							NumericGetDatum(
								v.val.numeric)));
					else if (strcmp(key, "learning_rate")
							== 0
						&& v.type == jbvNumeric)
						learning_rate = DatumGetFloat8(
							DirectFunctionCall1(
								numeric_float8,
								NumericGetDatum(
									v.val.numeric)));
					else if (strcmp(key, "lambda") == 0
						&& v.type == jbvNumeric)
						lambda = DatumGetFloat8(DirectFunctionCall1(
							numeric_float8,
							NumericGetDatum(
								v.val.numeric)));
					pfree(key);
				}
			}
		}
		appendStringInfo(&sql,
			"SELECT train_logistic_regression(%s, %s, %s, %d, "
			"%.6f, %.6f)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_cstr(target_column),
			max_iters,
			learning_rate,
			lambda);
	} else if (strcmp(algorithm, "linear_regression") == 0)
	{
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT train_linear_regression(%s, %s, %s)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_cstr(target_column));
	} else if (strcmp(algorithm, "svm") == 0)
	{
		double C = 1.0;
		int max_iters = 1000;
		if (hyperparams != NULL)
		{
			JsonbIterator *it;
			JsonbValue v;
			int r;

			it = JsonbIteratorInit(&hyperparams->root);
			while ((r = JsonbIteratorNext(&it, &v, false))
				!= WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val,
						v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "C") == 0
						&& v.type == jbvNumeric)
						C = DatumGetFloat8(DirectFunctionCall1(
							numeric_float8,
							NumericGetDatum(
								v.val.numeric)));
					else if (strcmp(key, "max_iters") == 0
						&& v.type == jbvNumeric)
						max_iters = DatumGetInt32(DirectFunctionCall1(
							numeric_int4,
							NumericGetDatum(
								v.val.numeric)));
					pfree(key);
				}
			}
		}
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT train_svm_classifier(%s, %s, %s, %.6f, %d)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_cstr(target_column),
			C,
			max_iters);
	} else if (strcmp(algorithm, "random_forest") == 0)
	{
		int n_trees = 10, max_depth = 10, min_samples = 100;
		if (hyperparams != NULL)
		{
			JsonbIterator *it;
			JsonbValue v;
			int r;

			it = JsonbIteratorInit(&hyperparams->root);
			while ((r = JsonbIteratorNext(&it, &v, false))
				!= WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val,
						v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "n_trees") == 0
						&& v.type == jbvNumeric)
						n_trees = DatumGetInt32(DirectFunctionCall1(
							numeric_int4,
							NumericGetDatum(
								v.val.numeric)));
					else if (strcmp(key, "max_depth") == 0
						&& v.type == jbvNumeric)
						max_depth = DatumGetInt32(DirectFunctionCall1(
							numeric_int4,
							NumericGetDatum(
								v.val.numeric)));
					else if ((strcmp(key, "min_samples")
								 == 0
							 || strcmp(key,
								    "min_"
								    "samples_"
								    "split")
								 == 0)
						&& v.type == jbvNumeric)
						min_samples = DatumGetInt32(DirectFunctionCall1(
							numeric_int4,
							NumericGetDatum(
								v.val.numeric)));
					pfree(key);
				}
			}
		}
		appendStringInfo(&sql,
			"SELECT train_random_forest_classifier(%s, "
			"%s, %s, %d, %d, %d)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_cstr(target_column),
			n_trees,
			max_depth,
			min_samples);
	} else if (strcmp(algorithm, "decision_tree") == 0)
	{
		int max_depth = 10, min_samples = 100;
		if (hyperparams != NULL)
		{
			JsonbIterator *it =
				JsonbIteratorInit(&hyperparams->root);
			JsonbValue v;
			int r;
			while ((r = JsonbIteratorNext(&it, &v, false))
				!= WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val,
						v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "max_depth") == 0
						&& v.type == jbvNumeric)
						max_depth = DatumGetInt32(DirectFunctionCall1(
							numeric_int4,
							NumericGetDatum(
								v.val.numeric)));
					else if (strcmp(key, "min_samples") == 0
						&& v.type == jbvNumeric)
						min_samples = DatumGetInt32(DirectFunctionCall1(
							numeric_int4,
							NumericGetDatum(
								v.val.numeric)));
					pfree(key);
				}
			}
		}
		appendStringInfo(&sql,
			"SELECT train_decision_tree_classifier(%s, "
			"%s, %s, %d, %d)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_cstr(target_column),
			max_depth,
			min_samples);
	} else if (strcmp(algorithm, "naive_bayes") == 0)
	{
		appendStringInfo(&sql,
			"SELECT train_naive_bayes_classifier(%s, %s, %s)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_cstr(target_column));
	} else if (strcmp(algorithm, "knn") == 0
		|| strcmp(algorithm, "knn_classifier") == 0)
	{
		if (hyperparams != NULL)
		{
			JsonbIterator *it =
				JsonbIteratorInit(&hyperparams->root);
			JsonbValue v;
			int r;
			while ((r = JsonbIteratorNext(&it, &v, false))
				!= WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val,
						v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "k") == 0
						&& v.type == jbvNumeric)
						k_value = DatumGetInt32(DirectFunctionCall1(
							numeric_int4,
							NumericGetDatum(
								v.val.numeric)));
					pfree(key);
				}
			}
		}
		{
			/* Use advisory lock and CTE to calculate version atomically */
			resetStringInfo(&sql);
			appendStringInfo(&sql,
				"SELECT pg_advisory_xact_lock(%d)",
				project_id);
			ret = SPI_execute(sql.data, false, 0);
			if (ret != SPI_OK_SELECT)
			{
				neurondb_cleanup(oldcontext, callcontext, true);
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("Failed to acquire advisory lock")));
			}

		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"WITH next_version AS ("
			"SELECT COALESCE(MAX(version), 0) + 1 AS v "
			"FROM neurondb.ml_models "
			"WHERE project_id = %d"
			") "
			"INSERT INTO neurondb.ml_models (project_id, "
			"version, algorithm, training_table, "
			"training_column, status, parameters) "
			"SELECT %d, v, 'knn'::neurondb.ml_algorithm_type, %s, %s, 'completed', "
			"'{\"k\": %d}'::jsonb "
			"FROM next_version "
			"RETURNING model_id",
			project_id,
			project_id,
				neurondb_quote_literal_cstr(table_name),
				neurondb_quote_literal_cstr(target_column),
				k_value);
		}
		ret = SPI_execute(sql.data, false, 0);
		if (ret == SPI_OK_INSERT_RETURNING && SPI_processed > 0)
		{
			model_id = DatumGetInt32(
				SPI_getbinval(SPI_tuptable->vals[0],
					SPI_tuptable->tupdesc,
					1,
					&isnull));
			neurondb_cleanup(oldcontext, callcontext, true);
			PG_RETURN_INT32(model_id);
		} else
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("failed to register knn "
					       "model")));
		}
	} else if (strcmp(algorithm, "ridge") == 0)
	{
		double alpha = 1.0;
		if (hyperparams != NULL)
		{
			JsonbIterator *it =
				JsonbIteratorInit(&hyperparams->root);
			JsonbValue v;
			int r;
			while ((r = JsonbIteratorNext(&it, &v, false))
				!= WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val,
						v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "alpha") == 0
						&& v.type == jbvNumeric)
						alpha = DatumGetFloat8(DirectFunctionCall1(
							numeric_float8,
							NumericGetDatum(
								v.val.numeric)));
					pfree(key);
				}
			}
		}
		appendStringInfo(&sql,
			"SELECT train_ridge_regression(%s, %s, %s, %f)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_cstr(target_column),
			alpha);
	} else if (strcmp(algorithm, "lasso") == 0)
	{
		double alpha = 1.0;
		if (hyperparams != NULL)
		{
			JsonbIterator *it =
				JsonbIteratorInit(&hyperparams->root);
			JsonbValue v;
			int r;
			while ((r = JsonbIteratorNext(&it, &v, false))
				!= WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val,
						v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "alpha") == 0
						&& v.type == jbvNumeric)
						alpha = DatumGetFloat8(DirectFunctionCall1(
							numeric_float8,
							NumericGetDatum(
								v.val.numeric)));
					pfree(key);
				}
			}
		}
		appendStringInfo(&sql,
			"SELECT train_lasso_regression(%s, %s, %s, %f)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_cstr(target_column),
			alpha);
	} else
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Unsupported algorithm: \"%s\"",
					algorithm),
				errhint("Supported algorithms: "
					"linear_regression, "
					"logistic_regression, random_forest, "
					"svm, "
					"decision_tree, naive_bayes, knn, "
					"ridge, "
					"lasso")));
	}

	ret = SPI_execute(sql.data, false, 0);

	if (strcmp(algorithm, "random_forest") == 0
		|| strcmp(algorithm, "logistic_regression") == 0
		|| strcmp(algorithm, "linear_regression") == 0
		|| strcmp(algorithm, "decision_tree") == 0
		|| strcmp(algorithm, "svm") == 0)
	{
		if (SPI_processed == 0)
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("%s training did not return a "
					       "model id",
						algorithm)));
		}
		model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc,
			1,
			&isnull));
		if (isnull)
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("%s training returned NULL "
					       "model id",
						algorithm)));
		}
	}

	if (ret < 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("Training failed for algorithm \"%s\"",
					algorithm)));
	}

	/* Step 4: Register model in ml_models table (if not already persisted) */
	{
		bool model_registered = false;

		/* If training function returned model_id > 0, it already registered */
		if (model_id > 0)
		{
			model_registered = true;
			elog(DEBUG1,
				"neurondb.train: model already registered by training function (model_id=%d)",
				model_id);
		}

		if (!model_registered)
		{
			/* Use advisory lock and CTE to calculate version atomically */
			/* Lock on project_id to serialize version calculation */
			resetStringInfo(&sql);
			appendStringInfo(&sql,
				"SELECT pg_advisory_xact_lock(%d)",
				project_id);
			ret = SPI_execute(sql.data, false, 0);
			if (ret != SPI_OK_SELECT)
			{
				neurondb_cleanup(oldcontext, callcontext, true);
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("Failed to acquire advisory lock")));
			}

			/* Now calculate version and insert atomically */
			resetStringInfo(&sql);
			appendStringInfo(&sql,
				"WITH next_version AS ("
				"SELECT COALESCE(MAX(version), 0) + 1 AS v "
				"FROM neurondb.ml_models "
				"WHERE project_id = %d"
				") "
				"INSERT INTO neurondb.ml_models (project_id, "
				"version, algorithm, training_table, "
				"training_column, "
				"status, "
				"parameters) "
				"SELECT %d, v, "
				"%s::neurondb.ml_algorithm_type, "
				"%s, %s, "
				"'completed', '{}'::jsonb "
				"FROM next_version "
				"RETURNING model_id",
				project_id,
				project_id,
				neurondb_quote_literal_cstr(algorithm),
				neurondb_quote_literal_cstr(table_name),
				neurondb_quote_literal_cstr(target_column));

			ret = SPI_execute(sql.data, false, 0);
			if (ret != SPI_OK_INSERT_RETURNING
				|| SPI_processed == 0)
			{
				neurondb_cleanup(oldcontext, callcontext, true);
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
						errmsg("Failed to register "
						       "model in catalog")));
			}

			model_id = DatumGetInt32(
				SPI_getbinval(SPI_tuptable->vals[0],
					SPI_tuptable->tupdesc,
					1,
					&isnull));
		}
	}

	if (model_name != NULL)
		pfree(model_name);

	neurondb_cleanup(oldcontext, callcontext, true);

	elog(DEBUG1,
		"neurondb.train: model_id=%d created successfully",
		model_id);

	PG_RETURN_INT32(model_id);
}

/* ----------
 * neurondb_predict
 * Unified prediction interface.
 * ----------
 */
Datum
neurondb_predict(PG_FUNCTION_ARGS)
{
	int32 model_id = PG_GETARG_INT32(0);
	ArrayType *features_array = PG_GETARG_ARRAYTYPE_P(1);

	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	StringInfoData features_str;
	int ret;
	bool isnull = false;
	char *algorithm = NULL;
	float8 prediction = 0.0;
	int ndims, nelems, i;
	int *dims;
	float8 *features;

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
		"neurondb_predict memory context",
		ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
	{
		neurondb_cleanup(oldcontext, callcontext, false);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_connect failed")));
	}

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT algorithm FROM neurondb.ml_models WHERE model_id = %d",
		model_id);
	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Model not found: %d", model_id)));
	}
	algorithm = TextDatumGetCString(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
	if (isnull)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Model algorithm is NULL for "
				       "model_id=%d",
					model_id)));
	}

	ndims = ARR_NDIM(features_array);
	dims = ARR_DIMS(features_array);
	nelems = ArrayGetNItems(ndims, dims);
	features = (float8 *)ARR_DATA_PTR(features_array);

	initStringInfo(&features_str);
	appendStringInfoString(&features_str, "ARRAY[");
	for (i = 0; i < nelems; ++i)
	{
		if (i > 0)
			appendStringInfoString(&features_str, ", ");
		appendStringInfo(&features_str, "%.6f", features[i]);
	}
	appendStringInfoString(&features_str, "]::real[]");

	resetStringInfo(&sql);

	/* Routing based on algorithm */
	if (strcmp(algorithm, "linear_regression") == 0)
		appendStringInfo(&sql,
			"SELECT predict_linear_regression(%d, %s)",
			model_id,
			features_str.data);
	else if (strcmp(algorithm, "logistic_regression") == 0)
		appendStringInfo(&sql,
			"SELECT predict_logistic_regression(%d, %s)",
			model_id,
			features_str.data);
	else if (strcmp(algorithm, "random_forest") == 0)
		appendStringInfo(&sql,
			"SELECT predict_random_forest(%d, %s)",
			model_id,
			features_str.data);
	else if (strcmp(algorithm, "linear_regression") == 0)
		appendStringInfo(&sql,
			"SELECT predict_linear_regression_model_id(%d, %s)",
			model_id,
			features_str.data);
	else if (strcmp(algorithm, "svm") == 0)
		appendStringInfo(&sql,
			"SELECT predict_svm_model_id(%d, %s)",
			model_id,
			features_str.data);
	else if (strcmp(algorithm, "decision_tree") == 0)
		appendStringInfo(&sql,
			"SELECT predict_decision_tree(%d, %s)",
			model_id,
			features_str.data);
	else if (strcmp(algorithm, "naive_bayes") == 0)
		appendStringInfo(&sql,
			"SELECT predict_naive_bayes(%d, %s)",
			model_id,
			features_str.data);
	else if (strcmp(algorithm, "ridge") == 0
		|| strcmp(algorithm, "lasso") == 0)
		appendStringInfo(&sql,
			"SELECT predict_regularized_regression(%d, %s)",
			model_id,
			features_str.data);
	else if (strcmp(algorithm, "knn") == 0
		|| strcmp(algorithm, "knn_classifier") == 0)
		appendStringInfo(&sql,
			"SELECT predict_knn(%d, %s)",
			model_id,
			features_str.data);
	else
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				errmsg("Unsupported algorithm for prediction: "
				       "\"%s\"",
					algorithm)));
	}

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("Prediction query did not return a "
				       "result")));
	}
	prediction = DatumGetFloat8(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	neurondb_cleanup(oldcontext, callcontext, true);

	elog(DEBUG1,
		"neurondb.predict: model_id=%d, algorithm=%s, prediction=%.6f",
		model_id,
		algorithm,
		prediction);

	PG_RETURN_FLOAT8(prediction);
}

/* ----------
 * neurondb_deploy
 * Deploy trained model for usage.
 * ----------
 */
Datum
neurondb_deploy(PG_FUNCTION_ARGS)
{
	int32 model_id = PG_GETARG_INT32(0);
	text *strategy_text = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);
	char *strategy;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	int ret;
	int deployment_id = 0;
	bool isnull = false;

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
		"neurondb_deploy memory context",
		ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	if (strategy_text)
		strategy = text_to_cstring(strategy_text);
	else
		strategy = pstrdup("replace");

	elog(DEBUG1,
		"neurondb.deploy: model_id=%d, strategy=%s",
		model_id,
		strategy);

	if (SPI_connect() != SPI_OK_CONNECT)
	{
		neurondb_cleanup(oldcontext, callcontext, false);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_connect failed")));
	}

	initStringInfo(&sql);
	appendStringInfoString(&sql,
		"CREATE TABLE IF NOT EXISTS neurondb.ml_deployments ("
		"deployment_id SERIAL PRIMARY KEY, "
		"model_id INTEGER NOT NULL REFERENCES "
		"neurondb.ml_models(model_id), "
		"deployment_name TEXT NOT NULL, "
		"strategy TEXT NOT NULL, "
		"status TEXT DEFAULT 'active', "
		"deployed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)");
	(void)SPI_execute(sql.data, false, 0);

	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"INSERT INTO neurondb.ml_deployments (model_id, "
		"deployment_name, strategy, status, deployed_at) "
		"VALUES (%d, %s, %s, 'active', CURRENT_TIMESTAMP) "
		"RETURNING deployment_id",
		model_id,
		neurondb_quote_literal_cstr(
			psprintf("deploy_%d_%ld", model_id, (long)time(NULL))),
		neurondb_quote_literal_cstr(strategy));

	ret = SPI_execute(sql.data, false, 0);
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("Failed to create deployment")));
	}

	deployment_id = DatumGetInt32(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	neurondb_cleanup(oldcontext, callcontext, true);

	elog(DEBUG1,
		"neurondb.deploy: deployment_id=%d created",
		deployment_id);

	PG_RETURN_INT32(deployment_id);
}

/* ----------
 * neurondb_load_model
 * Load an external ML model and register its metadata.
 * ----------
 */
Datum
neurondb_load_model(PG_FUNCTION_ARGS)
{
	text *project_name_text = PG_GETARG_TEXT_PP(0);
	text *model_path_text = PG_GETARG_TEXT_PP(1);
	text *model_format_text = PG_GETARG_TEXT_PP(2);

	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	char *project_name;
	char *model_path;
	char *model_format;
	int ret;
	int model_id = 0;
	int project_id = 0;
	bool isnull = false;

	project_name = text_to_cstring(project_name_text);
	model_path = text_to_cstring(model_path_text);
	model_format = text_to_cstring(model_format_text);

	elog(NOTICE,
		"neurondb.load_model: project='%s', path='%s', format='%s'",
		project_name,
		model_path,
		model_format);

	if (strcmp(model_format, "onnx") != 0
		&& strcmp(model_format, "tensorflow") != 0
		&& strcmp(model_format, "pytorch") != 0
		&& strcmp(model_format, "sklearn") != 0)
	{
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Unsupported model format: %s. "
				       "Supported: "
				       "onnx, tensorflow, pytorch, sklearn",
					model_format)));
	}

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
		"neurondb_load_model memory context",
		ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
	{
		neurondb_cleanup(oldcontext, callcontext, false);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_connect failed")));
	}

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"INSERT INTO neurondb.ml_projects (project_name, "
		"model_type, description) "
		"VALUES (%s, 'external', 'External model import') "
		"ON CONFLICT (project_name) DO UPDATE SET updated_at "
		"= CURRENT_TIMESTAMP "
		"RETURNING project_id",
		neurondb_quote_literal_cstr(project_name));

	ret = SPI_execute(sql.data, false, 0);
	if ((ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING)
		|| SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("Failed to create/get external project "
				       "\"%s\"",
					project_name)));
	}

	project_id = DatumGetInt32(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	{
		/* Use advisory lock and CTE to calculate version atomically */
		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT pg_advisory_xact_lock(%d)",
			project_id);
		ret = SPI_execute(sql.data, false, 0);
		if (ret != SPI_OK_SELECT)
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("Failed to acquire advisory lock")));
		}

		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"WITH next_version AS ("
			"SELECT COALESCE(MAX(version), 0) + 1 AS v "
			"FROM neurondb.ml_models "
			"WHERE project_id = %d"
			") "
			"INSERT INTO neurondb.ml_models (project_id, version, "
			"model_name, algorithm, training_table, training_column, "
			"status, metadata) "
			"SELECT %d, v, %s, %s, NULL, NULL, 'external', "
			"'{\"model_path\": %s, \"model_format\": %s}'::jsonb "
			"FROM next_version "
			"RETURNING model_id",
			project_id,
			project_id,
			neurondb_quote_literal_cstr(
				psprintf("%s_%ld", model_format, (long)time(NULL))),
			neurondb_quote_literal_cstr(model_format),
			neurondb_quote_literal_cstr(model_path),
			neurondb_quote_literal_cstr(model_format));
	}

	ret = SPI_execute(sql.data, false, 0);
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("Failed to register external model")));
	}
	model_id = DatumGetInt32(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	neurondb_cleanup(oldcontext, callcontext, true);

	elog(NOTICE, "neurondb.load_model: model_id=%d registered", model_id);

	PG_RETURN_INT32(model_id);
}
