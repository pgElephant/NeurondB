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
#include "ml_catalog.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_gpu.h"
#include "neurondb_cuda_nb.h"
#include "neurondb_cuda_gmm.h"
#include "neurondb_cuda_knn.h"

/* PG_MODULE_MAGIC is in neurondb.c only */

PG_FUNCTION_INFO_V1(neurondb_train);
PG_FUNCTION_INFO_V1(neurondb_predict);
PG_FUNCTION_INFO_V1(neurondb_deploy);
PG_FUNCTION_INFO_V1(neurondb_load_model);

/* Forward declarations */
static void neurondb_cleanup(MemoryContext oldcontext, MemoryContext callcontext, bool finish_spi);
static char *neurondb_quote_literal_cstr(const char *str);
static char *neurondb_quote_literal_or_null(const char *str);

/* Clean up function: restores context and optionally finishes SPI */
static void
neurondb_cleanup(MemoryContext oldcontext, MemoryContext callcontext, bool finish_spi)
{
	if (finish_spi)
		SPI_finish();
	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(callcontext);
}

/* Helper: Load training data from table into feature_matrix and label_vector */
static bool
neurondb_load_training_data(const char *table_name,
							const char *feature_list_str,
							const char *target_column,
							float **feature_matrix_out,
							double **label_vector_out,
							int *n_samples_out,
							int *feature_dim_out,
							int *class_count_out)
{
	StringInfoData sql;
	int ret;
	int n_samples = 0;
	int feature_dim = 0;
	int class_count = 0;
	float *feature_matrix = NULL;
	double *label_vector = NULL;
	TupleDesc tupdesc;
	HeapTuple tuple;
	bool isnull;
	Datum feat_datum;
	ArrayType *feat_arr = NULL;
	int i, j;
	Oid feature_type;

	if (feature_matrix_out)
		*feature_matrix_out = NULL;
	if (label_vector_out)
		*label_vector_out = NULL;
	if (n_samples_out)
		*n_samples_out = 0;
	if (feature_dim_out)
		*feature_dim_out = 0;
	if (class_count_out)
		*class_count_out = 0;

	/* Build query to select features and target */
	initStringInfo(&sql);
	if (target_column)
	{
		/* Make a copy of target_column before quoting to avoid memory corruption */
		char *target_copy = pstrdup(target_column);
		char *target_quoted = quote_identifier(target_copy);
		appendStringInfo(&sql, "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL", feature_list_str, target_quoted, table_name, feature_list_str, target_quoted);
		pfree(target_quoted);
		pfree(target_copy);
	}
	else
		appendStringInfo(&sql, "SELECT %s FROM %s WHERE %s IS NOT NULL", feature_list_str, table_name, feature_list_str);

	elog(DEBUG1, "neurondb_load_training_data: SQL=%s", sql.data);

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		elog(DEBUG1, "neurondb_load_training_data: SPI_execute failed, ret=%d", ret);
		pfree(sql.data);
		return false;
	}

	n_samples = SPI_processed;
	if (n_samples == 0)
	{
		pfree(sql.data);
		return false;
	}

	/* Determine feature dimension from first row */
	tupdesc = SPI_tuptable->tupdesc;
	tuple = SPI_tuptable->vals[0];
	feat_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
	if (isnull)
	{
		pfree(sql.data);
		return false;
	}

		feature_type = SPI_gettypeid(tupdesc, 1);
		if (feature_type == FLOAT4ARRAYOID || feature_type == FLOAT8ARRAYOID)
		{
			feat_arr = DatumGetArrayTypeP(feat_datum);
			feature_dim = ArrayGetNItems(ARR_NDIM(feat_arr), ARR_DIMS(feat_arr));
		}
		else
		{
			/* Try Vector type */
			Vector *vec = DatumGetVector(feat_datum);
			if (vec != NULL && vec->dim > 0)
			{
				feature_dim = vec->dim;
				feat_arr = NULL; /* Mark as Vector type, not array */
			}
			else
			{
				/* Assume single feature if not array or vector */
				feature_dim = 1;
				feat_arr = NULL;
			}
		}

	/* Allocate arrays */
	feature_matrix = (float *)palloc(sizeof(float) * (size_t)n_samples * (size_t)feature_dim);
	if (target_column)
		label_vector = (double *)palloc(sizeof(double) * (size_t)n_samples);

	/* Extract all data */
	for (i = 0; i < n_samples; i++)
	{
		HeapTuple current_tuple = SPI_tuptable->vals[i];
		bool isnull_feat, isnull_label;
		Datum featval, labelval;

		/* Features */
		featval = SPI_getbinval(current_tuple, tupdesc, 1, &isnull_feat);
		if (isnull_feat)
		{
			pfree(feature_matrix);
			if (label_vector)
				pfree(label_vector);
			pfree(sql.data);
			return false;
		}

		if (feat_arr)
		{
			ArrayType *curr_arr = DatumGetArrayTypeP(featval);
			if (ARR_NDIM(curr_arr) == 1)
			{
				int arr_len = ArrayGetNItems(ARR_NDIM(curr_arr), ARR_DIMS(curr_arr));
				if (arr_len != feature_dim)
				{
					pfree(feature_matrix);
					if (label_vector)
						pfree(label_vector);
					pfree(sql.data);
					return false;
				}
				if (feature_type == FLOAT8ARRAYOID)
				{
					float8 *fdat = (float8 *)ARR_DATA_PTR(curr_arr);
					for (j = 0; j < feature_dim; j++)
						feature_matrix[i * feature_dim + j] = (float)fdat[j];
				}
				else
				{
					float4 *fdat = (float4 *)ARR_DATA_PTR(curr_arr);
					for (j = 0; j < feature_dim; j++)
						feature_matrix[i * feature_dim + j] = fdat[j];
				}
			}
			else
			{
				pfree(feature_matrix);
				if (label_vector)
					pfree(label_vector);
				pfree(sql.data);
				return false;
			}
		}
		else if (feature_type == FLOAT8OID || feature_type == FLOAT4OID)
		{
			/* Scalar feature */
			if (feature_type == FLOAT8OID)
				feature_matrix[i * feature_dim] = (float)DatumGetFloat8(featval);
			else
				feature_matrix[i * feature_dim] = DatumGetFloat4(featval);
		}
		else
		{
			/* Try Vector type */
			Vector *vec = DatumGetVector(featval);
			if (vec != NULL && vec->dim == feature_dim)
			{
				for (j = 0; j < feature_dim; j++)
					feature_matrix[i * feature_dim + j] = vec->data[j];
			}
			else
			{
				pfree(feature_matrix);
				if (label_vector)
					pfree(label_vector);
				pfree(sql.data);
				return false;
			}
		}

		/* Labels (if target_column provided) */
		if (target_column)
		{
			labelval = SPI_getbinval(current_tuple, tupdesc, 2, &isnull_label);
			if (isnull_label)
			{
				pfree(feature_matrix);
				pfree(label_vector);
				pfree(sql.data);
				return false;
			}

			Oid label_type = SPI_gettypeid(tupdesc, 2);
			if (label_type == INT4OID)
				label_vector[i] = (double)DatumGetInt32(labelval);
			else if (label_type == INT8OID)
				label_vector[i] = (double)DatumGetInt64(labelval);
			else if (label_type == FLOAT4OID)
				label_vector[i] = (double)DatumGetFloat4(labelval);
			else if (label_type == FLOAT8OID)
				label_vector[i] = DatumGetFloat8(labelval);
			else
			{
				pfree(feature_matrix);
				pfree(label_vector);
				pfree(sql.data);
				return false;
			}
		}
	}

	/* Count unique classes for classification algorithms */
	if (target_column && label_vector)
	{
		int *seen_classes = (int *)palloc0(sizeof(int) * 256);
		int max_class = -1;
		for (i = 0; i < n_samples; i++)
		{
			int cls = (int)label_vector[i];
			if (cls >= 0 && cls < 256)
			{
				if (!seen_classes[cls])
				{
					seen_classes[cls] = 1;
					if (cls > max_class)
						max_class = cls;
				}
			}
		}
		class_count = max_class + 1;
		if (class_count == 0)
			class_count = 2; /* Default to binary classification */
		pfree(seen_classes);
	}

	pfree(sql.data);

	if (feature_matrix_out)
		*feature_matrix_out = feature_matrix;
	if (label_vector_out)
		*label_vector_out = label_vector;
	if (n_samples_out)
		*n_samples_out = n_samples;
	if (feature_dim_out)
		*feature_dim_out = feature_dim;
	if (class_count_out)
		*class_count_out = class_count;

	return true;
}

/* Return a quoted SQL literal, result must be pfree'd by caller */
static char *
neurondb_quote_literal_cstr(const char *str)
{
	char   *ret;
	text   *txt = cstring_to_text(str);

	ret = TextDatumGetCString(
		DirectFunctionCall1(quote_literal, PointerGetDatum(txt)));
	pfree(txt);

	return ret;
}

/* Helper: quote literal or return NULL for SQL */
static char *
neurondb_quote_literal_or_null(const char *str)
{
	if (str == NULL)
		return pstrdup("NULL");
	return neurondb_quote_literal_cstr(str);
}

/* Return canonical model type for a known ML algorithm */
static const char *
neurondb_get_model_type(const char *algorithm)
{
	if (algorithm == NULL)
		return "classification";
	if (strcmp(algorithm, "logistic_regression") == 0 ||
		strcmp(algorithm, "random_forest") == 0 ||
		strcmp(algorithm, "svm") == 0 ||
		strcmp(algorithm, "decision_tree") == 0 ||
		strcmp(algorithm, "naive_bayes") == 0 ||
		strcmp(algorithm, "xgboost") == 0)
		return "classification";
	if (strcmp(algorithm, "linear_regression") == 0 ||
		strcmp(algorithm, "ridge") == 0 ||
		strcmp(algorithm, "lasso") == 0)
		return "regression";
	if (strcmp(algorithm, "kmeans") == 0 ||
		strcmp(algorithm, "gmm") == 0 ||
		strcmp(algorithm, "minibatch_kmeans") == 0 ||
		strcmp(algorithm, "hierarchical") == 0)
		return "clustering";
	if (strcmp(algorithm, "pca") == 0 ||
		strcmp(algorithm, "opq") == 0)
		return "dimensionality_reduction";
	return "classification";
}

/* ----------
 * neurondb_train
 * Unified model training interface
 * ----------
 */
Datum
neurondb_train(PG_FUNCTION_ARGS)
{
	text	   *project_name_text;
	text	   *algorithm_text;
	text	   *table_name_text;
	text	   *target_column_text;
	ArrayType  *feature_columns_array;
	Jsonb	   *hyperparams;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	StringInfoData feature_list;
	const char **feature_names = NULL;
	int			feature_name_count = 0;
	char	   *model_name = NULL;
	MLGpuTrainResult gpu_result;
	char	   *gpu_errmsg = NULL;
	char	   *project_name;
	char	   *algorithm;
	char	   *table_name;
	char	   *target_column;
	int			ret;
	int			project_id = 0;
	int			model_id = 0;
	bool		isnull = false;
	int			i;

	/* Argument checking */
	if (PG_NARGS() != 6)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb.train: requires 6 arguments, got %d", PG_NARGS()),
				 errhint("Usage: neurondb.train(project_name, algorithm, table_name, label_col, feature_columns[], params)")));

	project_name_text = PG_ARGISNULL(0) ? NULL : PG_GETARG_TEXT_PP(0);
	algorithm_text = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);
	table_name_text = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);
	target_column_text = PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);
	feature_columns_array = PG_ARGISNULL(4) ? NULL : PG_GETARG_ARRAYTYPE_P(4);
	hyperparams = PG_ARGISNULL(5) ? NULL : PG_GETARG_JSONB_P(5);

	/* Required field checks */
	if (project_name_text == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb.train: project_name cannot be NULL")));
	if (algorithm_text == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb.train: algorithm cannot be NULL")));
	if (table_name_text == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb.train: table_name cannot be NULL")));

	project_name = text_to_cstring(project_name_text);
	algorithm = text_to_cstring(algorithm_text);
	table_name = text_to_cstring(table_name_text);

	/* target_column can be NULL for unsupervised algorithms (e.g., GMM, kmeans) */
	if (target_column_text == NULL)
	{
		/* Check if algorithm is unsupervised - allow NULL for these */
		if (strcmp(algorithm, "gmm") != 0 &&
			strcmp(algorithm, "kmeans") != 0 &&
			strcmp(algorithm, "minibatch_kmeans") != 0 &&
			strcmp(algorithm, "hierarchical") != 0)
		{
			ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb.train: target_column cannot be NULL for algorithm '%s'",
					algorithm)));
		}
		target_column = NULL;
	}
	else
	{
		target_column = text_to_cstring(target_column_text);
	}

	elog(DEBUG1, "neurondb.train: project=\"%s\", algorithm=\"%s\", table=\"%s\", target=\"%s\"",
		 project_name, algorithm, table_name, target_column ? target_column : "(NULL)");

	/* Determine backend based on GPU availability and GUC settings */
	{
		bool gpu_enabled_guc = false;
		bool gpu_available = neurondb_gpu_is_available();
		const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
		
		/* Check GUC setting */
		gpu_enabled_guc = (backend != NULL);
		
		elog(DEBUG1, "neurondb.train: GPU GUC enabled=%d, GPU hardware available=%d",
			gpu_enabled_guc, gpu_available);
		
		if (gpu_enabled_guc && !gpu_available)
		{
			ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("GPU training requested but GPU hardware not available"),
				 errhint("Disable GPU with: SET neurondb.gpu_enabled = off; or ensure CUDA is properly configured")));
		}
		
		/* Store the backend decision globally for this training session */
		/* We'll use this later to ensure consistent backend usage */
	}

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
									   "neurondb_train context",
									   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("SPI_connect failed")));

	/* Ensure ml_projects entry exists */
	{
		const char *model_type;
		char	   *model_type_quoted;

		model_type = neurondb_get_model_type(algorithm);
		model_type_quoted = neurondb_quote_literal_cstr(model_type);

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "INSERT INTO neurondb.ml_projects (project_name, model_type, description) "
						 "VALUES (%s, %s, 'Auto-created by neurondb.train()') "
						 "ON CONFLICT (project_name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP "
						 "RETURNING project_id",
						 neurondb_quote_literal_cstr(project_name),
						 model_type_quoted);
		pfree(model_type_quoted);
	}
	ret = SPI_execute(sql.data, false, 0);

	if ((ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING) || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Failed to create/get project \"%s\"", project_name)));
	}
	project_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
											 SPI_tuptable->tupdesc, 1, &isnull));

	resetStringInfo(&sql);
	initStringInfo(&feature_list);

	/* Build feature list (comma concatenated) */
	if (feature_columns_array != NULL)
	{
		Oid			elemtype;
		int16		typlen;
		bool		typbyval;
		char		typalign;
		int			ndims;
		int		   *dims;
		int			nelems;
		Datum	   *elem_values;
		bool	   *elem_nulls;

		elemtype = ARR_ELEMTYPE(feature_columns_array);
		ndims = ARR_NDIM(feature_columns_array);
		dims = ARR_DIMS(feature_columns_array);
		nelems = ArrayGetNItems(ndims, dims);
		get_typlenbyvalalign(elemtype, &typlen, &typbyval, &typalign);

		deconstruct_array(feature_columns_array, TEXTOID, typlen, typbyval, typalign, &elem_values, &elem_nulls, &nelems);
		for (i = 0; i < nelems; i++)
		{
			if (!elem_nulls[i])
			{
				char *col = TextDatumGetCString(elem_values[i]);
				if (feature_list.len > 0)
					appendStringInfoString(&feature_list, ", ");
				appendStringInfoString(&feature_list, col);
				if (feature_names == NULL)
					feature_names = (const char **)palloc(sizeof(char *) * nelems);
				feature_names[feature_name_count++] = pstrdup(col);
				pfree(col);
			}
		}
		pfree(elem_values);
		pfree(elem_nulls);

		if (feature_list.len == 0)
			appendStringInfoString(&feature_list, "*");
	}
	else
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
	elog(DEBUG1, "neurondb.train: Attempting GPU training via bridge for algorithm='%s'", algorithm);
	
	/* Wrap GPU bridge call in PG_TRY to catch crashes */
	PG_TRY();
	{
		if (ndb_gpu_try_train_model(algorithm, project_name, model_name, table_name, target_column,
									feature_names, feature_name_count, hyperparams,
									NULL, NULL, 0, 0, 0,
									&gpu_result, &gpu_errmsg))
		{
			elog(DEBUG1, "neurondb.train: GPU training via bridge succeeded");

			elog(DEBUG1, "neurondb.train: GPU training succeeded, gpu_result.spec.metrics=%p",
				 (void *)gpu_result.spec.metrics);

			if (gpu_result.spec.metrics != NULL)
			{
				char *metrics_txt = DatumGetCString(
						DirectFunctionCall1(jsonb_out, JsonbPGetDatum(gpu_result.spec.metrics)));
				elog(DEBUG1, "neurondb.train: GPU metrics content: %s", metrics_txt);
				pfree(metrics_txt);
			}
			model_id = ml_catalog_register_model(&gpu_result.spec);
			gpu_result.model_id = model_id;
			elog(DEBUG1, "neurondb.train: GPU model_id=%d created successfully", model_id);
			ndb_gpu_free_train_result(&gpu_result);

			if (feature_names)
			{
				for (i = 0; i < feature_name_count; i++)
					pfree((void *)feature_names[i]);
				pfree(feature_names);
			}
			neurondb_cleanup(oldcontext, callcontext, true);
			pfree(model_name);
			PG_RETURN_INT32(model_id);
		}
	}
	PG_CATCH();
	{
		elog(WARNING, "neurondb.train: GPU training via bridge crashed, falling back to CPU");
		FlushErrorState();
	}
	PG_END_TRY();
	
	/* Check if GPU was explicitly requested via GUC */
	{
		bool gpu_enabled_guc = (ndb_gpu_get_active_backend() != NULL);
		bool gpu_available = neurondb_gpu_is_available();
		
		if (gpu_enabled_guc && gpu_available)
		{
			/* GPU is enabled and available but training failed - ERROR, no fallback */
			if (gpu_errmsg)
			{
				/* ereport does longjmp, so we cleanup BEFORE calling it */
				StringInfoData errbuf;
				initStringInfo(&errbuf);
				appendStringInfoString(&errbuf, gpu_errmsg);
				
				ndb_gpu_free_train_result(&gpu_result);
				if (feature_names)
				{
					for (i = 0; i < feature_name_count; i++)
						pfree((void *)feature_names[i]);
					pfree(feature_names);
				}
				pfree(model_name);
				neurondb_cleanup(oldcontext, callcontext, true);
				
				ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("GPU training failed for algorithm '%s': %s", algorithm, errbuf.data),
					 errhint("GPU is enabled and available but training failed. Check if algorithm supports GPU or disable GPU with: SET neurondb.gpu_enabled = off;")));
			}
		}
	}
	
	elog(DEBUG1, "neurondb.train: GPU not requested or not available, using CPU training");
	ndb_gpu_free_train_result(&gpu_result);

	resetStringInfo(&sql);

	/* Algorithm specific codegen and SPI call */
	if (strcmp(algorithm, "linear_regression") == 0)
	{
		appendStringInfo(&sql,
						 "SELECT train_linear_regression(%s, %s, %s)",
						 neurondb_quote_literal_cstr(table_name),
						 neurondb_quote_literal_cstr(feature_list.data),
						 neurondb_quote_literal_or_null(target_column));
	}
	else if (strcmp(algorithm, "logistic_regression") == 0)
	{
		int max_iters = 1000;
		double learning_rate = 0.01;
		double lambda = 0.001;

		if (hyperparams)
		{
			JsonbIterator *it;
			JsonbValue	v;
			int			r;

			it = JsonbIteratorInit(&hyperparams->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val, v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "max_iters") == 0 && v.type == jbvNumeric)
						max_iters = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "learning_rate") == 0 && v.type == jbvNumeric)
						learning_rate = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "lambda") == 0 && v.type == jbvNumeric)
						lambda = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(v.val.numeric)));
					pfree(key);
				}
			}
		}
		appendStringInfo(&sql,
						 "SELECT train_logistic_regression(%s, %s, %s, %d, %.6f, %.6f)",
						 neurondb_quote_literal_cstr(table_name),
						 neurondb_quote_literal_cstr(feature_list.data),
						 neurondb_quote_literal_or_null(target_column),
						 max_iters, learning_rate, lambda);
	}
	else if (strcmp(algorithm, "svm") == 0)
	{
		double C = 1.0;
		int max_iters = 1000;
		if (hyperparams)
		{
			JsonbIterator *it;
			JsonbValue	v;
			int			r;

			it = JsonbIteratorInit(&hyperparams->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val, v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "C") == 0 && v.type == jbvNumeric)
						C = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "max_iters") == 0 && v.type == jbvNumeric)
						max_iters = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					pfree(key);
				}
			}
		}
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT train_svm_classifier(%s, %s, %s, %.6f, %d)",
						 neurondb_quote_literal_cstr(table_name),
						 neurondb_quote_literal_cstr(feature_list.data),
						 neurondb_quote_literal_or_null(target_column),
						 C, max_iters);
	}
	else if (strcmp(algorithm, "random_forest") == 0)
	{
		int n_trees = 10, max_depth = 10, min_samples = 100;
		if (hyperparams)
		{
			JsonbIterator *it;
			JsonbValue	v;
			int			r;
			it = JsonbIteratorInit(&hyperparams->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val, v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "n_trees") == 0 && v.type == jbvNumeric)
						n_trees = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "max_depth") == 0 && v.type == jbvNumeric)
						max_depth = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					else if ((strcmp(key, "min_samples") == 0 ||
							  strcmp(key, "min_samples_split") == 0) && v.type == jbvNumeric)
						min_samples = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					pfree(key);
				}
			}
		}
		appendStringInfo(&sql,
						 "SELECT train_random_forest_classifier(%s, %s, %s, %d, %d, %d)",
						 neurondb_quote_literal_cstr(table_name),
						 neurondb_quote_literal_cstr(feature_list.data),
						 neurondb_quote_literal_or_null(target_column),
						 n_trees, max_depth, min_samples);
	}
	else if (strcmp(algorithm, "decision_tree") == 0)
	{
		int max_depth = 10, min_samples = 100;
		if (hyperparams)
		{
			JsonbIterator *it = JsonbIteratorInit(&hyperparams->root);
			JsonbValue	v;
			int			r;
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val, v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "max_depth") == 0 && v.type == jbvNumeric)
						max_depth = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "min_samples") == 0 && v.type == jbvNumeric)
						min_samples = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					pfree(key);
				}
			}
		}
		appendStringInfo(&sql,
						 "SELECT train_decision_tree_classifier(%s, %s, %s, %d, %d)",
						 neurondb_quote_literal_cstr(table_name),
						 neurondb_quote_literal_cstr(feature_list.data),
						 neurondb_quote_literal_or_null(target_column),
						 max_depth, min_samples);
	}
	else if (strcmp(algorithm, "naive_bayes") == 0)
	{
		float *feature_matrix = NULL;
		double *label_vector = NULL;
		int n_samples = 0;
		int feature_dim = 0;
		int class_count = 0;
		bytea *gpu_model_data = NULL;
		Jsonb *gpu_metrics = NULL;
		char *gpu_errstr = NULL;
		const ndb_gpu_backend *backend = NULL;
		int gpu_rc = -1;
		bool gpu_trained = false;

		/* Load training data */
		elog(DEBUG1, "Naive Bayes: Before load_training_data - target='%s'", target_column ? target_column : "(null)");
		if (!neurondb_load_training_data(table_name, feature_list.data, target_column,
										 &feature_matrix, &label_vector,
										 &n_samples, &feature_dim, &class_count))
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Failed to load training data for Naive Bayes")));
		}
		elog(DEBUG1, "Naive Bayes: After load_training_data - target='%s'", target_column ? target_column : "(null)");

		if (n_samples < 10)
		{
			if (feature_matrix)
				pfree(feature_matrix);
			if (label_vector)
				pfree(label_vector);
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Naive Bayes: need at least 10 samples, got %d", n_samples)));
		}

		/* Try GPU training through proper bridge */
		elog(DEBUG1, "Naive Bayes: Attempting GPU training via bridge - samples=%d, dim=%d, classes=%d",
			 n_samples, feature_dim, class_count);
		
		MemSet(&gpu_result, 0, sizeof(MLGpuTrainResult));
		if (ndb_gpu_try_train_model("naive_bayes", project_name, model_name, table_name, target_column,
									feature_names, feature_name_count, hyperparams,
									feature_matrix, label_vector, n_samples, feature_dim, class_count,
									&gpu_result, &gpu_errstr) && gpu_result.spec.model_data != NULL)
		{
			elog(DEBUG1, "Naive Bayes: GPU training via bridge succeeded");
			gpu_trained = true;
			gpu_model_data = gpu_result.spec.model_data;
			gpu_metrics = gpu_result.spec.metrics;
		}
		else
		{
			elog(DEBUG1, "Naive Bayes: GPU training via bridge failed or not available");
			if (gpu_errstr)
			{
				elog(DEBUG1, "Naive Bayes: GPU error: %s", gpu_errstr);
				pfree(gpu_errstr);
				gpu_errstr = NULL;
			}
		}

		/* If GPU training failed, try CPU training */
		if (!gpu_trained)
		{
			/* Clean up allocated training data */
			if (feature_matrix)
				pfree(feature_matrix);
			if (label_vector)
				pfree(label_vector);
			
			/* Fall back to CPU training via SQL function */
			elog(DEBUG1, "Naive Bayes: CPU fallback - table='%s', features='%s', target='%s'",
				 table_name ? table_name : "(null)",
				 feature_list.data ? feature_list.data : "(null)",
				 target_column ? target_column : "(null)");
			resetStringInfo(&sql);
			appendStringInfo(&sql,
							 "SELECT train_naive_bayes_classifier_model_id(%s::text, %s::text, %s::text)",
							 neurondb_quote_literal_cstr(table_name),
							 neurondb_quote_literal_cstr(feature_list.data),
							 neurondb_quote_literal_or_null(target_column));
			elog(DEBUG1, "Naive Bayes: CPU fallback SQL = %s", sql.data);
			ret = SPI_execute(sql.data, true, 0);
			if (ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
													   SPI_tuptable->tupdesc, 1, &isnull));
				if (!isnull && model_id > 0)
				{
					elog(DEBUG1, "Naive Bayes: CPU training succeeded, model_id=%d", model_id);
					/* Update metrics to ensure storage='cpu' is set */
					resetStringInfo(&sql);
					appendStringInfo(&sql,
						"UPDATE neurondb.ml_models SET metrics = "
						"COALESCE(metrics, '{}'::jsonb) || '{\"storage\": \"cpu\"}'::jsonb "
						"WHERE model_id = %d",
						model_id);
					SPI_execute(sql.data, false, 0);
					goto train_complete;
				}
			}
			/* If CPU training failed, fall through to error */
		}
		else
		{
			/* Register GPU-trained model */
			MLCatalogModelSpec spec;
			Jsonb *final_metrics = NULL;

			resetStringInfo(&sql);
			appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
			ret = SPI_execute(sql.data, false, 0);
			if (ret != SPI_OK_SELECT)
			{
				if (feature_matrix)
					pfree(feature_matrix);
				if (label_vector)
					pfree(label_vector);
				if (gpu_model_data)
					pfree(gpu_model_data);
				if (gpu_metrics)
					pfree(gpu_metrics);
				if (gpu_errstr)
					pfree(gpu_errstr);
				neurondb_cleanup(oldcontext, callcontext, true);
				ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR),
								errmsg("Failed to acquire advisory lock")));
			}

			/* Build metrics JSONB */
			if (gpu_metrics)
			{
				final_metrics = gpu_metrics;
			}
			else
			{
				StringInfoData metrics_json;
				initStringInfo(&metrics_json);
				appendStringInfo(&metrics_json, "{\"storage\": \"gpu\", \"n_classes\": %d}", class_count);
				final_metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metrics_json.data)));
				pfree(metrics_json.data);
			}

			memset(&spec, 0, sizeof(MLCatalogModelSpec));
			spec.project_name = project_name;
			spec.algorithm = "naive_bayes";
			spec.training_table = table_name;
			spec.training_column = target_column;
			spec.model_data = gpu_model_data;
			spec.metrics = final_metrics;
			spec.num_samples = n_samples;
			spec.num_features = feature_dim;

			model_id = ml_catalog_register_model(&spec);

			/* Cleanup */
			if (feature_matrix)
				pfree(feature_matrix);
			if (label_vector)
				pfree(label_vector);
			if (gpu_errstr)
				pfree(gpu_errstr);

			if (model_id > 0)
			{
				neurondb_cleanup(oldcontext, callcontext, true);
				pfree(model_name);
				PG_RETURN_INT32(model_id);
			}
			else
			{
				neurondb_cleanup(oldcontext, callcontext, true);
				pfree(model_name);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("failed to register Naive Bayes model")));
			}
		}
	}
	else if (strcmp(algorithm, "knn") == 0 || strcmp(algorithm, "knn_classifier") == 0)
	{
		int k_value = 5;
		int task_type = 0; /* 0 = classification, 1 = regression */
		float *feature_matrix = NULL;
		double *label_vector = NULL;
		int n_samples = 0;
		int feature_dim = 0;
		int class_count = 0;
		bytea *gpu_model_data = NULL;
		Jsonb *gpu_metrics = NULL;
		char *gpu_errstr = NULL;
		const ndb_gpu_backend *backend = NULL;
		int gpu_rc = -1;
		bool gpu_trained = false;

		/* Parse hyperparameters */
		if (hyperparams)
		{
			JsonbIterator *it = JsonbIteratorInit(&hyperparams->root);
			JsonbValue	v;
			int			r;
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val, v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "k") == 0 && v.type == jbvNumeric)
						k_value = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "task_type") == 0 && v.type == jbvNumeric)
						task_type = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					pfree(key);
				}
			}
		}

		if (k_value <= 0)
			k_value = 5;
		if (k_value > 100)
			k_value = 100;

		/* Load training data */
		if (!neurondb_load_training_data(table_name, feature_list.data, target_column,
										 &feature_matrix, &label_vector,
										 &n_samples, &feature_dim, &class_count))
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Failed to load training data for KNN")));
		}

		if (n_samples < k_value)
		{
			if (feature_matrix)
				pfree(feature_matrix);
			if (label_vector)
				pfree(label_vector);
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("KNN: need at least %d samples, got %d", k_value, n_samples)));
		}

		/* Try GPU training through proper bridge */
		elog(DEBUG1, "KNN: Attempting GPU training via bridge - samples=%d, dim=%d, k=%d",
			 n_samples, feature_dim, k_value);
		
		MemSet(&gpu_result, 0, sizeof(MLGpuTrainResult));
		if (ndb_gpu_try_train_model("knn", project_name, model_name, table_name, target_column,
									feature_names, feature_name_count, hyperparams,
									feature_matrix, label_vector, n_samples, feature_dim, class_count,
									&gpu_result, &gpu_errstr) && gpu_result.spec.model_data != NULL)
		{
			elog(DEBUG1, "KNN: GPU training via bridge succeeded");
			gpu_trained = true;
			gpu_model_data = gpu_result.spec.model_data;
			gpu_metrics = gpu_result.spec.metrics;
		}
		else
		{
			elog(DEBUG1, "KNN: GPU training via bridge failed or not available");
			if (gpu_errstr)
			{
				elog(DEBUG1, "KNN: GPU error: %s", gpu_errstr);
				pfree(gpu_errstr);
				gpu_errstr = NULL;
			}
		}

		/* If GPU training failed, try CPU training */
		if (!gpu_trained)
		{
			/* Clean up allocated training data */
			if (feature_matrix)
				pfree(feature_matrix);
			if (label_vector)
				pfree(label_vector);
			
			/* Fall back to CPU training via SQL function */
			elog(DEBUG1, "KNN: CPU fallback - table='%s', features='%s', target='%s'",
				 table_name ? table_name : "(null)",
				 feature_list.data ? feature_list.data : "(null)",
				 target_column ? target_column : "(null)");
			resetStringInfo(&sql);
			appendStringInfo(&sql,
							 "SELECT train_knn_model_id(%s::text, %s::text, %s::text, %d)",
							 neurondb_quote_literal_cstr(table_name),
							 neurondb_quote_literal_cstr(feature_list.data),
							 neurondb_quote_literal_cstr(target_column),
							 k_value);
			elog(DEBUG1, "KNN: CPU fallback SQL = %s", sql.data);
			ret = SPI_execute(sql.data, true, 0);
			if (ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
													   SPI_tuptable->tupdesc, 1, &isnull));
				if (!isnull && model_id > 0)
				{
					elog(DEBUG1, "KNN: CPU training succeeded, model_id=%d", model_id);
					/* Update metrics to ensure storage='cpu' is set */
					resetStringInfo(&sql);
					appendStringInfo(&sql,
						"UPDATE neurondb.ml_models SET metrics = "
						"COALESCE(metrics, '{}'::jsonb) || '{\"storage\": \"cpu\"}'::jsonb "
						"WHERE model_id = %d",
						model_id);
					SPI_execute(sql.data, false, 0);
					goto train_complete;
				}
			}
			/* If CPU training failed, fall through to error */
		}
		else
		{
			/* Register GPU-trained model */
			MLCatalogModelSpec spec;
			StringInfoData metrics_json;
			Jsonb *final_metrics = NULL;

			resetStringInfo(&sql);
			appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
			ret = SPI_execute(sql.data, false, 0);
			if (ret != SPI_OK_SELECT)
			{
				if (feature_matrix)
					pfree(feature_matrix);
				if (label_vector)
					pfree(label_vector);
				if (gpu_model_data)
					pfree(gpu_model_data);
				if (gpu_metrics)
					pfree(gpu_metrics);
				if (gpu_errstr)
					pfree(gpu_errstr);
				neurondb_cleanup(oldcontext, callcontext, true);
				ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR),
								errmsg("Failed to acquire advisory lock")));
			}

			/* Build metrics JSONB */
			if (gpu_metrics)
			{
				final_metrics = gpu_metrics;
			}
			else
			{
				initStringInfo(&metrics_json);
				appendStringInfo(&metrics_json, "{\"k\": %d, \"task_type\": %d", k_value, task_type);
				if (gpu_trained)
					appendStringInfo(&metrics_json, ", \"storage\": \"gpu\"");
				appendStringInfoChar(&metrics_json, '}');
				final_metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metrics_json.data)));
				pfree(metrics_json.data);
			}

			memset(&spec, 0, sizeof(MLCatalogModelSpec));
			spec.project_name = project_name;
			spec.algorithm = "knn";
			spec.training_table = table_name;
			spec.training_column = target_column;
			spec.model_data = gpu_model_data;
			spec.metrics = final_metrics;
			spec.num_samples = n_samples;
			spec.num_features = feature_dim;

			model_id = ml_catalog_register_model(&spec);

			/* Cleanup GPU training data */
			if (feature_matrix)
				pfree(feature_matrix);
			if (label_vector)
				pfree(label_vector);
			if (gpu_errstr)
				pfree(gpu_errstr);

			if (model_id > 0)
			{
				neurondb_cleanup(oldcontext, callcontext, true);
				pfree(model_name);
				PG_RETURN_INT32(model_id);
			}
			else
			{
				neurondb_cleanup(oldcontext, callcontext, true);
				pfree(model_name);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("failed to register KNN model")));
			}
		}
	}
	else if (strcmp(algorithm, "ridge") == 0)
	{
		double alpha = 1.0;
		if (hyperparams)
		{
			JsonbIterator *it = JsonbIteratorInit(&hyperparams->root);
			JsonbValue	v;
			int			r;
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val, v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "alpha") == 0 && v.type == jbvNumeric)
						alpha = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(v.val.numeric)));
					pfree(key);
				}
			}
		}
		appendStringInfo(&sql,
						 "SELECT train_ridge_regression(%s, %s, %s, %f)",
						 neurondb_quote_literal_cstr(table_name),
						 neurondb_quote_literal_cstr(feature_list.data),
						 neurondb_quote_literal_or_null(target_column),
						 alpha);
	}
	else if (strcmp(algorithm, "lasso") == 0)
	{
		double alpha = 1.0;
		if (hyperparams)
		{
			JsonbIterator *it = JsonbIteratorInit(&hyperparams->root);
			JsonbValue	v;
			int			r;
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val, v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "alpha") == 0 && v.type == jbvNumeric)
						alpha = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(v.val.numeric)));
					pfree(key);
				}
			}
		}
		appendStringInfo(&sql,
						 "SELECT train_lasso_regression(%s, %s, %s, %f)",
						 neurondb_quote_literal_cstr(table_name),
						 neurondb_quote_literal_cstr(feature_list.data),
						 neurondb_quote_literal_or_null(target_column),
						 alpha);
	}
	else if (strcmp(algorithm, "gmm") == 0)
	{
		int k_value = 3;
		int max_iters = 100;
		float *feature_matrix = NULL;
		double *label_vector = NULL;
		int n_samples = 0;
		int feature_dim = 0;
		int class_count = 0;
		bytea *gpu_model_data = NULL;
		Jsonb *gpu_metrics = NULL;
		char *gpu_errstr = NULL;
		const ndb_gpu_backend *backend = NULL;
		int gpu_rc = -1;
		bool gpu_trained = false;

		/* Parse hyperparameters */
		if (hyperparams)
		{
			JsonbIterator *it = JsonbIteratorInit(&hyperparams->root);
			JsonbValue	v;
			int			r;
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char *key = pnstrdup(v.val.string.val, v.val.string.len);
					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "k") == 0 && v.type == jbvNumeric)
						k_value = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "max_iters") == 0 && v.type == jbvNumeric)
						max_iters = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					pfree(key);
				}
			}
		}

		/* Load training data */
		if (!neurondb_load_training_data(table_name, feature_list.data, NULL,
										 &feature_matrix, &label_vector,
										 &n_samples, &feature_dim, &class_count))
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Failed to load training data for GMM")));
		}

		if (n_samples < k_value)
		{
			if (feature_matrix)
				pfree(feature_matrix);
			if (label_vector)
				pfree(label_vector);
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("GMM: need at least %d samples, got %d", k_value, n_samples)));
		}

		/* Try GPU training through proper bridge */
		elog(DEBUG1, "GMM: Attempting GPU training via bridge - samples=%d, dim=%d, k=%d",
			 n_samples, feature_dim, k_value);
		
		MemSet(&gpu_result, 0, sizeof(MLGpuTrainResult));
		if (ndb_gpu_try_train_model("gmm", project_name, model_name, table_name, NULL,
									feature_names, feature_name_count, hyperparams,
									feature_matrix, NULL, n_samples, feature_dim, 0,
									&gpu_result, &gpu_errstr) && gpu_result.spec.model_data != NULL)
		{
			elog(DEBUG1, "GMM: GPU training via bridge succeeded");
			gpu_trained = true;
			gpu_model_data = gpu_result.spec.model_data;
			gpu_metrics = gpu_result.spec.metrics;
		}
		else
		{
			elog(DEBUG1, "GMM: GPU training via bridge failed or not available");
			if (gpu_errstr)
			{
				elog(DEBUG1, "GMM: GPU error: %s", gpu_errstr);
				pfree(gpu_errstr);
				gpu_errstr = NULL;
			}
		}

		/* If GPU training failed, try CPU training */
		if (!gpu_trained)
		{
			/* Clean up allocated training data */
			if (feature_matrix)
				pfree(feature_matrix);
			if (label_vector)
				pfree(label_vector);
			
			/* Fall back to CPU training via SQL function */
			elog(DEBUG1, "GMM: CPU fallback - table='%s', features='%s'",
				 table_name ? table_name : "(null)",
				 feature_list.data ? feature_list.data : "(null)");
			resetStringInfo(&sql);
			appendStringInfo(&sql,
							 "SELECT train_gmm_model_id(%s::text, %s::text, %d, %d)",
							 neurondb_quote_literal_cstr(table_name),
							 neurondb_quote_literal_cstr(feature_list.data),
							 k_value,
							 max_iters);
			elog(DEBUG1, "GMM: CPU fallback SQL = %s", sql.data);
			ret = SPI_execute(sql.data, true, 0);
			if (ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
													   SPI_tuptable->tupdesc, 1, &isnull));
				if (!isnull && model_id > 0)
				{
					elog(DEBUG1, "GMM: CPU training succeeded, model_id=%d", model_id);
					/* Update metrics to ensure storage='cpu' is set */
					resetStringInfo(&sql);
					appendStringInfo(&sql,
						"UPDATE neurondb.ml_models SET metrics = "
						"COALESCE(metrics, '{}'::jsonb) || '{\"storage\": \"cpu\"}'::jsonb "
						"WHERE model_id = %d",
						model_id);
					SPI_execute(sql.data, false, 0);
					goto train_complete;
				}
			}
			/* If CPU training failed, fall through to error */
		}
		else
		{
			/* Register GPU-trained model */
			MLCatalogModelSpec spec;
			StringInfoData metrics_json;
			Jsonb *final_metrics = NULL;

			resetStringInfo(&sql);
			appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
			ret = SPI_execute(sql.data, false, 0);
			if (ret != SPI_OK_SELECT)
			{
				if (feature_matrix)
					pfree(feature_matrix);
				if (label_vector)
					pfree(label_vector);
				if (gpu_model_data)
					pfree(gpu_model_data);
				if (gpu_metrics)
					pfree(gpu_metrics);
				if (gpu_errstr)
					pfree(gpu_errstr);
				neurondb_cleanup(oldcontext, callcontext, true);
				ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR),
								errmsg("Failed to acquire advisory lock")));
			}

			/* Build metrics JSONB */
			if (gpu_metrics)
			{
				final_metrics = gpu_metrics;
			}
			else
			{
				initStringInfo(&metrics_json);
				appendStringInfo(&metrics_json, "{\"k\": %d, \"max_iters\": %d", k_value, max_iters);
				if (gpu_trained)
					appendStringInfo(&metrics_json, ", \"storage\": \"gpu\"");
				appendStringInfoChar(&metrics_json, '}');
				final_metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metrics_json.data)));
				pfree(metrics_json.data);
			}

			memset(&spec, 0, sizeof(MLCatalogModelSpec));
			spec.project_name = project_name;
			spec.algorithm = "gmm";
			spec.training_table = table_name;
			spec.training_column = target_column;
			spec.model_data = gpu_model_data;
			spec.metrics = final_metrics;
			spec.num_samples = n_samples;
			spec.num_features = feature_dim;

			model_id = ml_catalog_register_model(&spec);

			/* Cleanup GPU training data */
			if (feature_matrix)
				pfree(feature_matrix);
			if (label_vector)
				pfree(label_vector);
			if (gpu_errstr)
				pfree(gpu_errstr);

			if (model_id > 0)
			{
				neurondb_cleanup(oldcontext, callcontext, true);
				pfree(model_name);
				PG_RETURN_INT32(model_id);
			}
			else
			{
				neurondb_cleanup(oldcontext, callcontext, true);
				pfree(model_name);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("failed to register GMM model")));
			}
		}
	}
	else
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		pfree(model_name);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Unsupported algorithm: \"%s\"", algorithm),
				 errhint("Supported algorithms: linear_regression, logistic_regression, random_forest, svm, decision_tree, naive_bayes, knn, ridge, lasso, gmm")));
	}

	if (strcmp(algorithm, "gmm") != 0)
	{
		ret = SPI_execute(sql.data, false, 0);
	}

	if (strcmp(algorithm, "random_forest") == 0 ||
		strcmp(algorithm, "logistic_regression") == 0 ||
		strcmp(algorithm, "linear_regression") == 0 ||
		strcmp(algorithm, "decision_tree") == 0 ||
		strcmp(algorithm, "svm") == 0)
	{
		if (SPI_processed == 0)
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("%s training did not return a model id", algorithm)));
		}
		model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
											   SPI_tuptable->tupdesc, 1, &isnull));
		if (isnull)
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("%s training returned NULL model id", algorithm)));
		}
	}

	if (ret < 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Training failed for algorithm \"%s\"", algorithm)));
	}

	{
		bool model_registered = false;
		if (model_id > 0)
		{
			model_registered = true;
			elog(DEBUG1, "neurondb.train: model already registered by training function (model_id=%d)", model_id);
		}
		if (!model_registered)
		{
			resetStringInfo(&sql);
			appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
			ret = SPI_execute(sql.data, false, 0);
			if (ret != SPI_OK_SELECT)
			{
				neurondb_cleanup(oldcontext, callcontext, true);
				ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to acquire advisory lock")));
			}

			resetStringInfo(&sql);
			appendStringInfo(&sql,
							 "WITH next_version AS (SELECT COALESCE(MAX(version), 0) + 1 AS v FROM neurondb.ml_models WHERE project_id = %d) "
							 "INSERT INTO neurondb.ml_models (project_id, version, algorithm, training_table, training_column, status, parameters) "
							 "SELECT %d, v, %s::neurondb.ml_algorithm_type, %s, %s, 'completed', '{}'::jsonb FROM next_version RETURNING model_id",
							 project_id,
							 project_id,
							 neurondb_quote_literal_cstr(algorithm),
							 neurondb_quote_literal_cstr(table_name),
							 neurondb_quote_literal_or_null(target_column));
			ret = SPI_execute(sql.data, false, 0);
			if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
			{
				neurondb_cleanup(oldcontext, callcontext, true);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("Failed to register model in catalog")));
			}
			model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
												   SPI_tuptable->tupdesc, 1, &isnull));
		}
	}

	/* Update metrics to store backend type for CPU-trained models */
	{
		bool gpu_enabled_guc = (ndb_gpu_get_active_backend() != NULL);
		bool gpu_available = neurondb_gpu_is_available();
		
		/* If GPU was not used (either disabled or not available), mark as CPU */
		if (!gpu_enabled_guc || !gpu_available)
		{
			resetStringInfo(&sql);
			appendStringInfo(&sql,
				"UPDATE neurondb.ml_models SET metrics = "
				"COALESCE(metrics, '{}'::jsonb) || '{\"storage\": \"cpu\"}'::jsonb "
				"WHERE model_id = %d",
				model_id);
			ret = SPI_execute(sql.data, false, 0);
			if (ret != SPI_OK_UPDATE)
			{
				elog(WARNING, "Failed to update metrics with storage type for model_id=%d", model_id);
			}
			else
			{
				elog(DEBUG1, "neurondb.train: Updated model_id=%d with storage='cpu'", model_id);
			}
		}
	}

train_complete:
	if (model_name)
		pfree(model_name);
	neurondb_cleanup(oldcontext, callcontext, true);
	elog(DEBUG1, "neurondb.train: model_id=%d created successfully", model_id);
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
	int32		model_id;
	ArrayType  *features_array;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	StringInfoData features_str;
	int			ret;
	bool		isnull = false;
	char	   *algorithm = NULL;
	float8		prediction = 0.0;
	int			ndims, nelems, i;
	int		   *dims;
	float8	   *features;
	float8	   *features_float = NULL; /* Allocated if conversion needed */

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb.predict: requires 2 arguments, got %d", PG_NARGS()),
				 errhint("Usage: neurondb.predict(model_id, features[])")));

	model_id = PG_GETARG_INT32(0);
	features_array = PG_GETARG_ARRAYTYPE_P(1);

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
									   "neurondb_predict context",
									   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
	{
		neurondb_cleanup(oldcontext, callcontext, false);
		ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("SPI_connect failed")));
	}

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT algorithm::text FROM neurondb.ml_models WHERE model_id = %d",
					 model_id);
	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Model not found: %d", model_id)));
	}
	algorithm = TextDatumGetCString(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
	if (isnull)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Model algorithm is NULL for model_id=%d", model_id)));
	}

	ndims = ARR_NDIM(features_array);
	if (ndims != 1)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("neurondb.predict: features must be 1-dimensional array, got %d dimensions", ndims)));
	}
	dims = ARR_DIMS(features_array);
	nelems = ArrayGetNItems(ndims, dims);
	if (nelems <= 0 || nelems > 100000)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("neurondb.predict: invalid feature count: %d (expected 1-100000)", nelems)));
	}
	
	/* Validate array element type */
	if (ARR_ELEMTYPE(features_array) != FLOAT8OID && ARR_ELEMTYPE(features_array) != FLOAT4OID)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("neurondb.predict: features array must be float8[] or float4[], got OID %u", ARR_ELEMTYPE(features_array))));
	}
	
	/* Extract features based on element type */
	if (ARR_ELEMTYPE(features_array) == FLOAT4OID)
	{
		/* Convert float4[] to float8[] */
		float4 *features_f4 = (float4 *) ARR_DATA_PTR(features_array);
		if (features_f4 == NULL)
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb.predict: features array data pointer is NULL")));
		}
		/* Allocate float8 array and convert */
		features_float = (float8 *)palloc(sizeof(float8) * nelems);
		for (i = 0; i < nelems; i++)
			features_float[i] = (float8)features_f4[i];
		features = features_float;
	}
	else
	{
		/* Already float8[] */
		features = (float8 *) ARR_DATA_PTR(features_array);
		if (features == NULL)
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb.predict: features array data pointer is NULL")));
		}
		features_float = NULL; /* Not allocated, don't free */
	}

	initStringInfo(&features_str);
	/* Build vector literal for algorithms that need it (NB, GMM) */
	if (strcmp(algorithm, "naive_bayes") == 0 || 
		strcmp(algorithm, "gmm") == 0)
	{
		/* Build vector literal: '[1.0, 2.0, ...]'::vector */
		appendStringInfoChar(&features_str, '\'');
		appendStringInfoChar(&features_str, '[');
		for (i = 0; i < nelems; i++)
		{
			if (i > 0)
				appendStringInfoString(&features_str, ", ");
			appendStringInfo(&features_str, "%.6f", features[i]);
		}
		appendStringInfoString(&features_str, "]'::vector");
	}
	else
	{
		/* Build array literal for other algorithms (including KNN) */
		appendStringInfoString(&features_str, "ARRAY[");
		for (i = 0; i < nelems; i++)
		{
			if (i > 0)
				appendStringInfoString(&features_str, ", ");
			appendStringInfo(&features_str, "%.6f", features[i]);
		}
		appendStringInfoString(&features_str, "]::real[]");
	}

	resetStringInfo(&sql);

	if (strcmp(algorithm, "linear_regression") == 0)
		appendStringInfo(&sql, "SELECT predict_linear_regression(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "logistic_regression") == 0)
		appendStringInfo(&sql, "SELECT predict_logistic_regression(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "random_forest") == 0)
		appendStringInfo(&sql, "SELECT predict_random_forest(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "svm") == 0)
		appendStringInfo(&sql, "SELECT predict_svm_model_id(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "decision_tree") == 0)
		appendStringInfo(&sql, "SELECT predict_decision_tree(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "naive_bayes") == 0)
	{
		bytea	   *model_data = NULL;
		Jsonb	   *metrics = NULL;
		bool		is_gpu = false;
		int			nb_class = 0;
		double		nb_probability = 0.0;
		float	   *features_float = NULL;
		int			feature_dim = nelems;
		char	   *errstr = NULL;
		int			rc;

		elog(DEBUG1, "neurondb_predict: Naive Bayes prediction starting for model_id=%d", model_id);

		if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Naive Bayes model %d not found", model_id)));
		}

		elog(DEBUG1, "neurondb_predict: Fetched model payload, model_data=%p, metrics=%p", (void*)model_data, (void*)metrics);

		if (model_data == NULL)
		{
			if (metrics)
				pfree(metrics);
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Naive Bayes model %d has no model data (model not trained)", model_id),
					 errhint("Naive Bayes training must be completed before prediction. The model may have been created without actual training.")));
		}

		elog(DEBUG1, "neurondb_predict: Parsing metrics to determine storage type");

		/* Default to CPU if no metrics or parsing fails */
		is_gpu = false;

		if (metrics != NULL)
		{
			JsonbIterator *it = NULL;
			JsonbValue	v;
			int			r;
			bool found_storage = false;
			
			PG_TRY();
			{
				it = JsonbIteratorInit(&metrics->root);
				while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
				{
					if (r == WJB_KEY && v.type == jbvString)
					{
						char *key = pnstrdup(v.val.string.val, v.val.string.len);
						r = JsonbIteratorNext(&it, &v, false);
						if (strcmp(key, "storage") == 0 && v.type == jbvString)
						{
							char *storage = pnstrdup(v.val.string.val, v.val.string.len);
							elog(DEBUG1, "neurondb_predict: Found storage='%s'", storage);
							found_storage = true;
							if (strcmp(storage, "gpu") == 0)
								is_gpu = true;
							pfree(storage);
						}
						pfree(key);
					}
				}
			}
			PG_CATCH();
			{
				/* If metrics parsing fails, assume CPU model */
				elog(DEBUG1, "neurondb_predict: Metrics parsing failed, assuming CPU model");
				is_gpu = false;
			}
			PG_END_TRY();
			
			if (!found_storage)
			{
				elog(DEBUG1, "neurondb_predict: No 'storage' field in metrics, assuming CPU model");
				is_gpu = false;
			}
		}
		else
		{
			elog(DEBUG1, "neurondb_predict: No metrics available, assuming CPU model");
			is_gpu = false;
		}

		elog(DEBUG1, "neurondb_predict: Model storage determined: is_gpu=%d", is_gpu);
		
		features_float = (float *) palloc(sizeof(float) * feature_dim);
		for (i = 0; i < feature_dim; i++)
			features_float[i] = (float) features[i];

		/* Use model's training backend (from catalog) regardless of current GPU state */
		{
			const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
			bool gpu_currently_enabled = (backend != NULL && neurondb_gpu_is_available());
			
			elog(DEBUG1, "neurondb_predict: NB model trained on %s, GPU currently %s",
				is_gpu ? "GPU" : "CPU",
				gpu_currently_enabled ? "enabled" : "disabled");
			
			if (is_gpu)
			{
				/* Model was trained on GPU - must use GPU prediction */
				if (!gpu_currently_enabled)
				{
					pfree(features_float);
					if (model_data)
						pfree(model_data);
					if (metrics)
						pfree(metrics);
					neurondb_cleanup(oldcontext, callcontext, true);
					ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("Naive Bayes: model %d was trained on GPU but GPU is not currently enabled", model_id),
						 errhint("Enable GPU with: SET neurondb.gpu_enabled = on; SELECT neurondb_gpu_enable();")));
				}
				
				/* Use GPU prediction */
				if (backend && backend->nb_predict && model_data != NULL)
				{
					rc = backend->nb_predict(model_data, features_float, feature_dim, &nb_class, &nb_probability, &errstr);
					if (rc == 0)
					{
						prediction = (double) nb_class;
						pfree(features_float);
						if (model_data)
							pfree(model_data);
						if (metrics)
							pfree(metrics);
						neurondb_cleanup(oldcontext, callcontext, true);
						elog(DEBUG1, "neurondb.predict: Naive Bayes GPU prediction succeeded");
						PG_RETURN_FLOAT8(prediction);
					}
					if (errstr)
						pfree(errstr);
				}
			}
			else
			{
				/* Model was trained on CPU - use CPU prediction (ignore current GPU state) */
				elog(DEBUG1, "neurondb_predict: NB model trained on CPU, using CPU prediction regardless of GPU state");
				/* Fall through to CPU prediction path below */
			}
		}
		appendStringInfo(&sql, "SELECT predict_naive_bayes_model_id(%d, %s)", model_id, features_str.data);
		pfree(features_float);
		if (model_data)
			pfree(model_data);
		if (metrics)
			pfree(metrics);
	}
	else if (strcmp(algorithm, "ridge") == 0 || strcmp(algorithm, "lasso") == 0)
		appendStringInfo(&sql, "SELECT predict_regularized_regression(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "knn") == 0 || strcmp(algorithm, "knn_classifier") == 0)
	{
		bytea	   *model_data = NULL;
		Jsonb	   *metrics = NULL;
		bool		is_gpu = false;
		double		knn_prediction = 0.0;
		float	   *features_float = NULL;
		int			feature_dim = nelems;
		char	   *errstr = NULL;
		int			rc;

		if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("KNN model %d not found", model_id)));
		}

		if (model_data == NULL)
		{
			if (metrics)
				pfree(metrics);
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("KNN model %d has no model data (model not trained)", model_id),
					 errhint("KNN training must be completed before prediction. The model may have been created without actual training.")));
		}

		if (metrics != NULL)
		{
			JsonbIterator *it = NULL;
			JsonbValue	v;
			int			r;
			PG_TRY();
			{
				it = JsonbIteratorInit(&metrics->root);
				while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
				{
					if (r == WJB_KEY && v.type == jbvString)
					{
						char *key = pnstrdup(v.val.string.val, v.val.string.len);
						r = JsonbIteratorNext(&it, &v, false);
						if (strcmp(key, "storage") == 0 && v.type == jbvString)
						{
							char *storage = pnstrdup(v.val.string.val, v.val.string.len);
							if (strcmp(storage, "gpu") == 0)
								is_gpu = true;
							pfree(storage);
						}
						pfree(key);
					}
				}
			}
			PG_CATCH();
			{
				/* If metrics parsing fails, assume CPU model */
				is_gpu = false;
			}
			PG_END_TRY();
		}
		features_float = (float *) palloc(sizeof(float) * feature_dim);
		for (i = 0; i < feature_dim; i++)
			features_float[i] = (float) features[i];

		/* Use model's training backend (from catalog) regardless of current GPU state */
		{
			const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
			bool gpu_currently_enabled = (backend != NULL && neurondb_gpu_is_available());
			
			elog(DEBUG1, "neurondb_predict: KNN model trained on %s, GPU currently %s",
				is_gpu ? "GPU" : "CPU",
				gpu_currently_enabled ? "enabled" : "disabled");
			
			if (is_gpu)
			{
				/* Model was trained on GPU - must use GPU prediction */
				if (!gpu_currently_enabled)
				{
					pfree(features_float);
					if (model_data)
						pfree(model_data);
					if (metrics)
						pfree(metrics);
					neurondb_cleanup(oldcontext, callcontext, true);
					ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("KNN: model %d was trained on GPU but GPU is not currently enabled", model_id),
						 errhint("Enable GPU with: SET neurondb.gpu_enabled = on; SELECT neurondb_gpu_enable();")));
				}
				
				/* Use GPU prediction */
				if (backend && backend->knn_predict && model_data != NULL)
				{
					rc = backend->knn_predict(model_data, features_float, feature_dim, &knn_prediction, &errstr);
					if (rc == 0)
					{
						prediction = knn_prediction;
						pfree(features_float);
						if (model_data)
							pfree(model_data);
						if (metrics)
							pfree(metrics);
						neurondb_cleanup(oldcontext, callcontext, true);
						elog(DEBUG1, "neurondb.predict: KNN GPU prediction succeeded");
						PG_RETURN_FLOAT8(prediction);
					}
					if (errstr)
						pfree(errstr);
				}
			}
			else
			{
				/* Model was trained on CPU - use CPU prediction (ignore current GPU state) */
				elog(DEBUG1, "neurondb_predict: KNN model trained on CPU, using CPU prediction regardless of GPU state");
				/* Fall through to CPU prediction path below */
			}
		}
		appendStringInfo(&sql, "SELECT predict_knn_model_id(%d, %s)", model_id, features_str.data);
		pfree(features_float);
		if (model_data)
			pfree(model_data);
		if (metrics)
			pfree(metrics);
	}
	else if (strcmp(algorithm, "gmm") == 0)
	{
		bytea	   *model_data = NULL;
		Jsonb	   *metrics = NULL;
		bool		is_gpu = false;
		int			gmm_cluster = 0;
		double		gmm_probability = 0.0;
		float	   *features_float = NULL;
		int			feature_dim = nelems;
		char	   *errstr = NULL;
		int			rc;

		if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
		{
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("GMM model %d not found", model_id)));
		}

		if (model_data == NULL)
		{
			if (metrics)
				pfree(metrics);
			neurondb_cleanup(oldcontext, callcontext, true);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("GMM model %d has no model data (model not trained)", model_id),
					 errhint("GMM training must be completed before prediction. The model may have been created without actual training.")));
		}

		if (metrics != NULL)
		{
			JsonbIterator *it = NULL;
			JsonbValue	v;
			int			r;
			PG_TRY();
			{
				it = JsonbIteratorInit(&metrics->root);
				while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
				{
					if (r == WJB_KEY && v.type == jbvString)
					{
						char *key = pnstrdup(v.val.string.val, v.val.string.len);
						r = JsonbIteratorNext(&it, &v, false);
						if (strcmp(key, "storage") == 0 && v.type == jbvString)
						{
							char *storage = pnstrdup(v.val.string.val, v.val.string.len);
							if (strcmp(storage, "gpu") == 0)
								is_gpu = true;
							pfree(storage);
						}
						pfree(key);
					}
				}
			}
			PG_CATCH();
			{
				/* If metrics parsing fails, assume CPU model */
				is_gpu = false;
			}
			PG_END_TRY();
		}
		features_float = (float *) palloc(sizeof(float) * feature_dim);
		for (i = 0; i < feature_dim; i++)
			features_float[i] = (float) features[i];

		/* Use model's training backend (from catalog) */
		{
			const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
			bool gpu_currently_enabled = (backend != NULL && neurondb_gpu_is_available());
			
			elog(DEBUG1, "neurondb_predict: GMM model trained on %s, GPU currently %s",
				is_gpu ? "GPU" : "CPU",
				gpu_currently_enabled ? "enabled" : "disabled");
			
			if (!is_gpu)
			{
				/* Use CPU prediction */
				elog(DEBUG1, "neurondb_predict: GMM model trained on CPU, using CPU prediction");
				resetStringInfo(&sql);
				appendStringInfo(&sql, "SELECT predict_gmm_model_id(%d, %s)", model_id, features_str.data);
				ret = SPI_execute(sql.data, true, 0);
				if (ret == SPI_OK_SELECT && SPI_processed > 0)
				{
					/* GMM returns integer cluster ID, convert to float8 */
					int32 cluster_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
																  SPI_tuptable->tupdesc, 1, &isnull));
					prediction = (float8)cluster_id;
					if (isnull)
					{
						pfree(features_float);
						if (model_data)
							pfree(model_data);
						if (metrics)
							pfree(metrics);
						neurondb_cleanup(oldcontext, callcontext, true);
						ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("GMM prediction returned NULL")));
					}
					pfree(features_float);
					if (model_data)
						pfree(model_data);
					if (metrics)
						pfree(metrics);
					neurondb_cleanup(oldcontext, callcontext, false);
					PG_RETURN_FLOAT8(prediction);
				}
				else
				{
					pfree(features_float);
					if (model_data)
						pfree(model_data);
					if (metrics)
						pfree(metrics);
					neurondb_cleanup(oldcontext, callcontext, true);
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("GMM CPU prediction failed")));
				}
			}
			
			if (is_gpu && !gpu_currently_enabled)
			{
				/* GPU model but GPU not enabled - ERROR */
				pfree(features_float);
				if (model_data)
					pfree(model_data);
				if (metrics)
					pfree(metrics);
				neurondb_cleanup(oldcontext, callcontext, true);
				ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("GMM: model %d was trained on GPU but GPU is not currently enabled", model_id),
					 errhint("Enable GPU with: SET neurondb.gpu_enabled = on; SELECT neurondb_gpu_enable();")));
			}
			
			/* Use GPU prediction */
			if (is_gpu && backend && backend->gmm_predict)
			{
				rc = backend->gmm_predict(model_data, features_float, feature_dim, &gmm_cluster, &gmm_probability, &errstr);
				if (rc == 0)
				{
					prediction = (double) gmm_cluster;
					pfree(features_float);
					if (model_data)
						pfree(model_data);
					if (metrics)
						pfree(metrics);
					neurondb_cleanup(oldcontext, callcontext, true);
					elog(DEBUG1, "neurondb.predict: GMM GPU prediction succeeded");
					PG_RETURN_FLOAT8(prediction);
				}
				if (errstr)
					pfree(errstr);
			}
		}
		/* GMM GPU prediction failed */
		pfree(features_float);
		if (model_data)
			pfree(model_data);
		if (metrics)
			pfree(metrics);
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("GMM GPU prediction failed"),
				 errhint("Ensure GPU is available and enabled.")));
	}
	else
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Unsupported algorithm for prediction: \"%s\"", algorithm)));
	}

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Prediction query did not return a result")));
	}
	prediction = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
	neurondb_cleanup(oldcontext, callcontext, true);

	elog(DEBUG1, "neurondb.predict: model_id=%d, algorithm=%s, prediction=%.6f", model_id, algorithm, prediction);
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
	int32		model_id;
	text	   *strategy_text;
	char	   *strategy;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	int			ret;
	int			deployment_id = 0;
	bool		isnull = false;

	if (PG_NARGS() < 1 || PG_NARGS() > 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb.deploy: requires 1-2 arguments, got %d", PG_NARGS()),
				 errhint("Usage: neurondb.deploy(model_id, [strategy])")));

	model_id = PG_GETARG_INT32(0);
	strategy_text = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
									   "neurondb_deploy context",
									   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	if (strategy_text)
		strategy = text_to_cstring(strategy_text);
	else
		strategy = pstrdup("replace");

	elog(DEBUG1, "neurondb.deploy: model_id=%d, strategy=%s", model_id, strategy);

	if (SPI_connect() != SPI_OK_CONNECT)
	{
		neurondb_cleanup(oldcontext, callcontext, false);
		ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("SPI_connect failed")));
	}

	initStringInfo(&sql);
	appendStringInfoString(&sql,
		"CREATE TABLE IF NOT EXISTS neurondb.ml_deployments ("
		"deployment_id SERIAL PRIMARY KEY, "
		"model_id INTEGER NOT NULL REFERENCES neurondb.ml_models(model_id), "
		"deployment_name TEXT NOT NULL, "
		"strategy TEXT NOT NULL, "
		"status TEXT DEFAULT 'active', "
		"deployed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)");
	(void) SPI_execute(sql.data, false, 0);

	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "INSERT INTO neurondb.ml_deployments (model_id, deployment_name, strategy, status, deployed_at) "
					 "VALUES (%d, %s, %s, 'active', CURRENT_TIMESTAMP) RETURNING deployment_id",
					 model_id,
					 neurondb_quote_literal_cstr(psprintf("deploy_%d_%ld", model_id, (long) time(NULL))),
					 neurondb_quote_literal_cstr(strategy));

	ret = SPI_execute(sql.data, false, 0);
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to create deployment")));
	}
	deployment_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	neurondb_cleanup(oldcontext, callcontext, true);

	elog(DEBUG1, "neurondb.deploy: deployment_id=%d created", deployment_id);

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
	text	   *project_name_text;
	text	   *model_path_text;
	text	   *model_format_text;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	char	   *project_name;
	char	   *model_path;
	char	   *model_format;
	int			ret;
	int			model_id = 0;
	int			project_id = 0;
	bool		isnull = false;

	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb.load_model: requires 3 arguments, got %d", PG_NARGS()),
				 errhint("Usage: neurondb.load_model(project_name, model_path, model_format)")));

	project_name_text = PG_GETARG_TEXT_PP(0);
	model_path_text = PG_GETARG_TEXT_PP(1);
	model_format_text = PG_GETARG_TEXT_PP(2);

	project_name = text_to_cstring(project_name_text);
	model_path = text_to_cstring(model_path_text);
	model_format = text_to_cstring(model_format_text);

	elog(NOTICE, "neurondb.load_model: project='%s', path='%s', format='%s'", project_name, model_path, model_format);

	if (strcmp(model_format, "onnx") != 0 &&
		strcmp(model_format, "tensorflow") != 0 &&
		strcmp(model_format, "pytorch") != 0 &&
		strcmp(model_format, "sklearn") != 0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Unsupported model format: %s. Supported: onnx, tensorflow, pytorch, sklearn", model_format)));
	}

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
									   "neurondb_load_model context",
									   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
	{
		neurondb_cleanup(oldcontext, callcontext, false);
		ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("SPI_connect failed")));
	}

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"INSERT INTO neurondb.ml_projects (project_name, model_type, description) "
		"VALUES (%s, 'external', 'External model import') "
		"ON CONFLICT (project_name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP "
		"RETURNING project_id",
		neurondb_quote_literal_cstr(project_name));

	ret = SPI_execute(sql.data, false, 0);
	if ((ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING) || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to create/get external project \"%s\"", project_name)));
	}
	project_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	resetStringInfo(&sql);
	appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
	ret = SPI_execute(sql.data, false, 0);
	if (ret != SPI_OK_SELECT)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to acquire advisory lock")));
	}

	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"WITH next_version AS (SELECT COALESCE(MAX(version), 0) + 1 AS v FROM neurondb.ml_models WHERE project_id = %d) "
		"INSERT INTO neurondb.ml_models (project_id, version, model_name, algorithm, training_table, training_column, status, metadata) "
		"SELECT %d, v, %s, %s, NULL, NULL, 'external', '{\"model_path\": %s, \"model_format\": %s}'::jsonb FROM next_version RETURNING model_id",
		project_id,
		project_id,
		neurondb_quote_literal_cstr(psprintf("%s_%ld", model_format, (long) time(NULL))),
		neurondb_quote_literal_cstr(model_format),
		neurondb_quote_literal_cstr(model_path),
		neurondb_quote_literal_cstr(model_format));

	ret = SPI_execute(sql.data, false, 0);
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true);
		ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to register external model")));
	}
	model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
	neurondb_cleanup(oldcontext, callcontext, true);

	elog(NOTICE, "neurondb.load_model: model_id=%d registered", model_id);
	PG_RETURN_INT32(model_id);
}
