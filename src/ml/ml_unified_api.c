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
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_safe_memory.h"
#include "ml_decision_tree_internal.h"

PG_FUNCTION_INFO_V1(neurondb_train);
PG_FUNCTION_INFO_V1(neurondb_predict);
PG_FUNCTION_INFO_V1(neurondb_deploy);
PG_FUNCTION_INFO_V1(neurondb_load_model);
PG_FUNCTION_INFO_V1(neurondb_evaluate);

/* Forward declarations */
static void neurondb_cleanup(MemoryContext oldcontext, MemoryContext callcontext, bool finish_spi, bool we_connected_spi);
static char *neurondb_quote_literal_cstr(const char *str);
static char *neurondb_quote_literal_or_null(const char *str);

/* Clean up function: restores context and optionally finishes SPI */
static void
neurondb_cleanup(MemoryContext oldcontext, MemoryContext callcontext, bool finish_spi, bool we_connected_spi)
{
	/* Only call SPI_finish() if we actually connected SPI ourselves */
	if (finish_spi && we_connected_spi)
		SPI_finish();
	
	/* Ensure we're in oldcontext before deleting callcontext */
	/* SPI_finish() might have changed CurrentMemoryContext */
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
	int valid_samples = 0; /* Track valid samples after skipping NULLs */
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
	/* Add LIMIT to prevent loading too much data into memory at once */
	/* Conservative limit: 200k samples to avoid MaxAllocSize errors */
	{
		int max_samples_limit = 200000;
		char *feature_where_clause = NULL;
		initStringInfo(&sql);
		
		/* Skip WHERE clause filtering - we'll skip NULL features during processing */
		feature_where_clause = NULL;
		
		if (target_column)
		{
			/* Make a copy of target_column before quoting to avoid memory corruption */
			char *target_copy = pstrdup(target_column);
			const char *target_quoted_const = quote_identifier(target_copy);
			char *target_quoted = (char *)target_quoted_const;
			/* Remove WHERE clause filtering - we'll skip NULLs in processing */
			appendStringInfo(&sql, "SELECT %s, %s FROM %s LIMIT %d", 
				feature_list_str, target_quoted, table_name, max_samples_limit);
			NDB_SAFE_PFREE_AND_NULL(target_quoted);
			NDB_SAFE_PFREE_AND_NULL(target_copy);
		}
		else
		{
			/* Remove WHERE clause filtering - we'll skip NULLs in processing */
			appendStringInfo(&sql, "SELECT %s FROM %s LIMIT %d", 
				feature_list_str, table_name, max_samples_limit);
		}
		
		/* Free the WHERE clause if it was allocated */
		if (feature_where_clause != NULL)
			NDB_SAFE_PFREE_AND_NULL(feature_where_clause);
	}


	ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		NDB_SAFE_PFREE_AND_NULL(sql.data);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb_load_training_data: failed to execute data query")));
		return false; /* not reached */
	}

	n_samples = SPI_processed;
	if (n_samples == 0)
	{
		NDB_SAFE_PFREE_AND_NULL(sql.data);
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb_load_training_data: no training data found")));
		return false; /* not reached */
	}
	
	/* Warn if we hit the limit */
	if (n_samples >= 200000)
	{
		elog(INFO,
			"neurondb_load_training_data: dataset has more than %d rows, "
			"limiting to %d samples to avoid memory allocation errors",
			200000, n_samples);
	}

	/* Determine feature dimension from first row */
	tupdesc = SPI_tuptable->tupdesc;
	tuple = SPI_tuptable->vals[0];
	feat_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
	if (isnull)
	{
		NDB_SAFE_PFREE_AND_NULL(sql.data);
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("neurondb_load_training_data: feature column contains NULL values")));
		return false; /* not reached */
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
	/* Check memory allocation size before palloc */
	{
		size_t feature_matrix_size = sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
		size_t label_vector_size = target_column ? sizeof(double) * (size_t)n_samples : 0;
		
		if (feature_matrix_size > MaxAllocSize)
		{
			NDB_SAFE_PFREE_AND_NULL(sql.data);
			ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("neurondb_load_training_data: feature matrix size (%zu bytes) exceeds MaxAllocSize (%zu bytes)",
					feature_matrix_size, (size_t)MaxAllocSize),
				 errhint("Reduce dataset size or feature dimension")));
		}
		
		if (label_vector_size > MaxAllocSize)
		{
			NDB_SAFE_PFREE_AND_NULL(sql.data);
			ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("neurondb_load_training_data: label vector size (%zu bytes) exceeds MaxAllocSize (%zu bytes)",
					label_vector_size, (size_t)MaxAllocSize),
				 errhint("Reduce dataset size")));
		}
	}
	
	feature_matrix = (float *)palloc(sizeof(float) * (size_t)n_samples * (size_t)feature_dim);
	if (target_column)
		label_vector = (double *)palloc(sizeof(double) * (size_t)n_samples);

	/* Extract all data - skip NULL features and continue */
	valid_samples = 0;
	for (i = 0; i < n_samples; i++)
	{
		HeapTuple current_tuple = SPI_tuptable->vals[i];
		bool isnull_feat, isnull_label;
		Datum featval, labelval;

		/* Features */
		featval = SPI_getbinval(current_tuple, tupdesc, 1, &isnull_feat);
		if (isnull_feat)
		{
			continue;
		}

		if (feat_arr)
		{
			ArrayType *curr_arr = DatumGetArrayTypeP(featval);
			if (ARR_NDIM(curr_arr) == 1)
			{
				int arr_len = ArrayGetNItems(ARR_NDIM(curr_arr), ARR_DIMS(curr_arr));
				if (arr_len != feature_dim)
				{
					NDB_SAFE_PFREE_AND_NULL(feature_matrix);
					if (label_vector)
						NDB_SAFE_PFREE_AND_NULL(label_vector);
					NDB_SAFE_PFREE_AND_NULL(sql.data);
					return false;
				}
			if (feature_type == FLOAT8ARRAYOID)
			{
				float8 *fdat = (float8 *)ARR_DATA_PTR(curr_arr);
				for (j = 0; j < feature_dim; j++)
					feature_matrix[valid_samples * feature_dim + j] = (float)fdat[j];
			}
			else
			{
				float4 *fdat = (float4 *)ARR_DATA_PTR(curr_arr);
				for (j = 0; j < feature_dim; j++)
					feature_matrix[valid_samples * feature_dim + j] = fdat[j];
			}
			}
			else
			{
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
				if (label_vector)
					NDB_SAFE_PFREE_AND_NULL(label_vector);
				NDB_SAFE_PFREE_AND_NULL(sql.data);
				return false;
			}
		}
		else if (feature_type == FLOAT8OID || feature_type == FLOAT4OID)
		{
			if (feature_type == FLOAT8OID)
				feature_matrix[valid_samples * feature_dim] = (float)DatumGetFloat8(featval);
			else
				feature_matrix[valid_samples * feature_dim] = DatumGetFloat4(featval);
		}
		else
		{
			/* Try Vector type */
			Vector *vec = DatumGetVector(featval);
			if (vec != NULL && vec->dim == feature_dim)
			{
				for (j = 0; j < feature_dim; j++)
					feature_matrix[valid_samples * feature_dim + j] = vec->data[j];
			}
			else
			{
				/* Skip invalid vector and continue */
				continue;
			}
		}

		/* Labels (if target_column provided) */
		if (target_column)
		{
			Oid label_type;

			labelval = SPI_getbinval(current_tuple, tupdesc, 2, &isnull_label);
			if (isnull_label)
			{
				continue;
			}

			label_type = SPI_gettypeid(tupdesc, 2);
			if (label_type == INT4OID)
				label_vector[valid_samples] = (double)DatumGetInt32(labelval);
			else if (label_type == INT8OID)
				label_vector[valid_samples] = (double)DatumGetInt64(labelval);
			else if (label_type == FLOAT4OID)
				label_vector[valid_samples] = (double)DatumGetFloat4(labelval);
			else if (label_type == FLOAT8OID)
				label_vector[valid_samples] = DatumGetFloat8(labelval);
			else
			{
				continue;
			}
		}
		
		valid_samples++;
	}
	
	n_samples = valid_samples;

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
		NDB_SAFE_PFREE_AND_NULL(seen_classes);
	}

	NDB_SAFE_PFREE_AND_NULL(sql.data);

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
	text   *txt;

	if (str == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb_quote_literal_cstr: cannot quote NULL string")));

	txt = cstring_to_text(str);

	ret = TextDatumGetCString(
		DirectFunctionCall1(quote_literal, PointerGetDatum(txt)));
	NDB_SAFE_PFREE_AND_NULL(txt);

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
	char	   *gpu_errmsg_ptr = NULL;
	char	   **gpu_errmsg = &gpu_errmsg_ptr;
	char	   *project_name;
	char	   *algorithm;
	char	   *table_name;
	char	   *target_column;
	int			ret;
	int			project_id = 0;
	int			model_id = 0;
	bool		isnull = false;
	int			i;
	bool		spi_was_connected;
	bool		we_connected_spi;

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

	elog(DEBUG1,
		"neurondb_train: training %s model on project=%s, table=%s, target=%s",
		algorithm, project_name, table_name, target_column ? target_column : "(NULL)");

	/* Determine backend based on GPU availability and GUC settings */
	{
		bool gpu_enabled_guc = neurondb_gpu_enabled;
		bool gpu_available;
		
		/* Trigger lazy GPU initialization if needed */
		ndb_gpu_init_if_needed();
		
		/* Now check if GPU is actually available after initialization */
		gpu_available = neurondb_gpu_is_available();

		elog(DEBUG1,
			"neurondb_train: GPU enabled=%d, available=%d",
			gpu_enabled_guc, gpu_available);
		
		if (gpu_enabled_guc && !gpu_available)
		{
			elog(WARNING,
				"neurondb.train: GPU training requested but GPU hardware not available, falling back to CPU training");
			elog(DEBUG1,
				"neurondb.train: To suppress this warning, disable GPU with: SET neurondb.gpu_enabled = off;");
		}
	}

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
									   "neurondb_train context",
									   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	/* Check if SPI is already connected (e.g., when called from SQL) */
	/* SPI_processed == -1 means SPI is not connected */
	spi_was_connected = (SPI_processed != -1);
	we_connected_spi = false;
	
	if (!spi_was_connected)
	{
		int spi_ret = SPI_connect();
		if (spi_ret != SPI_OK_CONNECT)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("SPI_connect failed with return code %d", spi_ret)));
		}
		we_connected_spi = true;
		elog(DEBUG1, "neurondb_train: SPI_connect() succeeded, SPI_processed=%ld", (long)SPI_processed);
	}
	else
	{
		elog(DEBUG1, "neurondb_train: SPI already connected, SPI_processed=%ld", (long)SPI_processed);
	}

	{
		const char *model_type;
		char	   *model_type_quoted;

		model_type = neurondb_get_model_type(algorithm);
		model_type_quoted = neurondb_quote_literal_cstr(model_type);

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "INSERT INTO neurondb.ml_projects (project_name, model_type, description) "
						 "VALUES (%s, %s::neurondb.ml_model_type, 'Auto-created by neurondb.train()') "
						 "ON CONFLICT (project_name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP "
						 "RETURNING project_id",
						 neurondb_quote_literal_cstr(project_name),
						 model_type_quoted);
		NDB_SAFE_PFREE_AND_NULL(model_type_quoted);
		
		/* Debug: log the query if it's suspicious */
		if (sql.data == NULL || strlen(sql.data) == 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: project creation query is NULL or empty")));
		}
		elog(DEBUG1, "neurondb: project creation query: %s", sql.data);
	}
	
	/* Verify SPI is still connected before executing - if not, try to reconnect */
	if (SPI_processed == -1)
	{
		elog(WARNING, "neurondb_train: SPI disconnected before project creation query, attempting to reconnect");
		if (SPI_connect() != SPI_OK_CONNECT)
		{
			NDB_SAFE_PFREE_AND_NULL(sql.data);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: SPI disconnected and reconnect failed")));
		}
		we_connected_spi = true; /* Mark that we connected it */
		elog(DEBUG1, "neurondb_train: Reconnected SPI, SPI_processed=%ld", (long)SPI_processed);
	}
	else
	{
		elog(DEBUG1, "neurondb_train: Before project creation query, SPI_processed=%ld", (long)SPI_processed);
	}
	ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE_IF_SELECT(ret);

	if ((ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING) || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Failed to create/get project \"%s\" (ret=%d, processed=%lu)",
					project_name, ret, (unsigned long)SPI_processed)));
	}

	/* Validate SPI_tuptable before access - we know ret is INSERT/UPDATE RETURNING */
	if (ret == SPI_OK_INSERT_RETURNING || ret == SPI_OK_UPDATE_RETURNING)
	{
		NDB_CHECK_SPI_TUPTABLE();
	}
	project_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
											 SPI_tuptable->tupdesc, 1, &isnull));

	/* Use safe free/reinit to handle potential memory context changes */
	NDB_SAFE_PFREE_AND_NULL(sql.data);
	initStringInfo(&sql);
	initStringInfo(&feature_list);

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
				NDB_SAFE_PFREE_AND_NULL(col);
			}
		}
		NDB_SAFE_PFREE_AND_NULL(elem_values);
		NDB_SAFE_PFREE_AND_NULL(elem_nulls);

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
	
	/* Wrap GPU bridge call in PG_TRY to catch crashes */
	PG_TRY();
	{
		if (ndb_gpu_try_train_model(algorithm, project_name, model_name, table_name, target_column,
									feature_names, feature_name_count, hyperparams,
									NULL, NULL, 0, 0, 0,
									&gpu_result, gpu_errmsg))
		{
			elog(DEBUG1,
				"neurondb_train: GPU training successful, metrics available: %p",
				(void *)gpu_result.spec.metrics);

			if (gpu_result.spec.metrics != NULL)
			{
				char *metrics_txt = DatumGetCString(
						DirectFunctionCall1(jsonb_out, JsonbPGetDatum(gpu_result.spec.metrics)));
				NDB_SAFE_PFREE_AND_NULL(metrics_txt);
			}
			model_id = ml_catalog_register_model(&gpu_result.spec);
			/* ml_catalog_register_model will throw ERROR if registration fails */
			gpu_result.model_id = model_id;
			ndb_gpu_free_train_result(&gpu_result);

			if (feature_names)
			{
				for (i = 0; i < feature_name_count; i++)
				{
					void *ptr = (void *)feature_names[i];
					ndb_safe_pfree(ptr);
					feature_names[i] = NULL;
				}
				NDB_SAFE_PFREE_AND_NULL(feature_names);
			}
			/* Free model_name before deleting callcontext */
			if (model_name)
			{
		MemoryContextSwitchTo(callcontext);
				NDB_SAFE_PFREE_AND_NULL(model_name);
				model_name = NULL;
			}
			MemoryContextSwitchTo(oldcontext);
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			PG_RETURN_INT32(model_id);
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
	}
	PG_END_TRY();
	
	/* Log GPU training failure if it occurred */
	if (gpu_result.spec.model_data == NULL && gpu_errmsg != NULL && *gpu_errmsg != NULL)
	{
		elog(DEBUG1,
			"neurondb: GPU training for %s failed: %s",
			algorithm ? algorithm : "unknown",
			*gpu_errmsg);
		if (*gpu_errmsg)
		{
			NDB_SAFE_PFREE_AND_NULL(*gpu_errmsg);
			*gpu_errmsg = NULL;
		}
	}
	else if (gpu_result.spec.model_data == NULL)
	{
		elog(DEBUG1,
			"neurondb: GPU training for %s failed: no error message (ndb_gpu_try_train_model returned false or model_data is NULL)",
			algorithm ? algorithm : "unknown");
	}
	
	/* Check if GPU was explicitly requested via GUC */
	{
		bool gpu_enabled_guc = neurondb_gpu_enabled;
		bool gpu_available = neurondb_gpu_is_available();
		
		if (gpu_enabled_guc && gpu_available)
		{
			/* GPU is enabled and available but training failed - ERROR, no fallback */
			if (gpu_errmsg && *gpu_errmsg)
			{
				/* ereport does longjmp, so we cleanup BEFORE calling it */
				StringInfoData errbuf;
				initStringInfo(&errbuf);
				appendStringInfoString(&errbuf, *gpu_errmsg);
				
				ndb_gpu_free_train_result(&gpu_result);
				if (feature_names)
				{
					for (i = 0; i < feature_name_count; i++)
					{
						void *ptr = (void *)feature_names[i];
						ndb_safe_pfree(ptr);
						feature_names[i] = NULL;
					}
					NDB_SAFE_PFREE_AND_NULL(feature_names);
				}
				/* Free model_name before deleting callcontext */
				if (model_name)
				{
 		MemoryContextSwitchTo(callcontext);
					NDB_SAFE_PFREE_AND_NULL(model_name);
					model_name = NULL;
				}
				MemoryContextSwitchTo(oldcontext);
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
				
				ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("GPU training failed for algorithm '%s': %s", algorithm, errbuf.data),
					 errhint("GPU is enabled and available but training failed. Check if algorithm supports GPU or disable GPU with: SET neurondb.gpu_enabled = off;")));
			}
		}
	}
	
	ndb_gpu_free_train_result(&gpu_result);

	/* Use safe free/reinit to handle potential memory context changes */
	NDB_SAFE_PFREE_AND_NULL(sql.data);
	initStringInfo(&sql);

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
		const char *feature_col;

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
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}
		/* Use first feature column for logistic regression */
		feature_col = (feature_name_count > 0 && feature_names != NULL) 
			? feature_names[0] : "features";
		appendStringInfo(&sql,
			"SELECT train_logistic_regression(%s, %s, %s, %d, %.6f, %.6f)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_col),
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
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}
		appendStringInfo(&sql,
			"SELECT train_svm_classifier(%s, %s, %s, %.6f, %d)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_or_null(target_column),
			C, max_iters);
	}
	else if (strcmp(algorithm, "random_forest") == 0)
	{
		int n_trees = 10, max_depth = 10, min_samples = 100, max_features = 0;
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
					else if (strcmp(key, "max_features") == 0 && v.type == jbvNumeric)
						max_features = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}
		appendStringInfo(&sql,
			"SELECT train_random_forest_classifier(%s, %s, %s, %d, %d, %d, %d)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_or_null(target_column),
			n_trees, max_depth, min_samples, max_features);
	}
	else if (strcmp(algorithm, "decision_tree") == 0)
	{
		float *feature_matrix = NULL;
		double *label_vector = NULL;
		int n_samples = 0;
		int feature_dim = 0;
		int class_count = 0;
		bytea *gpu_model_data = NULL;
		Jsonb *gpu_metrics = NULL;
		Jsonb *final_metrics = NULL;
		char *gpu_errstr_ptr = NULL;
		char **gpu_errstr = &gpu_errstr_ptr;
		bool gpu_trained = false;
		int max_depth;
		int min_samples;

		/* Load training data */
		if (!neurondb_load_training_data(table_name, feature_list.data, target_column,
										 &feature_matrix, &label_vector,
										 &n_samples, &feature_dim, &class_count))
		{
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Failed to load training data for Decision Tree")));
		}

		if (n_samples < 10)
		{
			if (feature_matrix)
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
			if (label_vector)
				NDB_SAFE_PFREE_AND_NULL(label_vector);
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Decision Tree: need at least 10 samples, got %d", n_samples)));
		}

		/* Try GPU training through proper bridge */
		MemSet(&gpu_result, 0, sizeof(MLGpuTrainResult));
		if (ndb_gpu_try_train_model("decision_tree", project_name, model_name, table_name, target_column,
									feature_names, feature_name_count, hyperparams,
									feature_matrix, label_vector, n_samples, feature_dim, class_count,
									&gpu_result, gpu_errstr) && gpu_result.spec.model_data != NULL)
		{
			gpu_trained = true;
			gpu_model_data = gpu_result.spec.model_data;
			gpu_metrics = gpu_result.spec.metrics;
		}
		else
		{
			/* Log the error before freeing it */
			if (gpu_errstr != NULL && *gpu_errstr != NULL)
			{
				elog(DEBUG1,
					"neurondb: GPU training for decision_tree failed: %s",
					*gpu_errstr);
				NDB_SAFE_PFREE_AND_NULL(*gpu_errstr);
				*gpu_errstr = NULL;
			}
			else
			{
				elog(DEBUG1,
					"neurondb: GPU training for decision_tree failed: no error message (ndb_gpu_try_train_model returned false or model_data is NULL)");
			}
		}

		/* If GPU training succeeded, register the model */
		if (gpu_trained)
		{
			MLCatalogModelSpec spec;
			
			/* Build metrics JSONB */
			if (gpu_metrics)
			{
				final_metrics = gpu_metrics;
			}
			else
			{
				/* Don't use DirectFunctionCall - it crashes in CUDA context.
				 * Build JSONB manually using JsonbBuilder API. */
				JsonbParseState *state = NULL;
				JsonbValue k, v;
				
				(void)pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);
				
				/* Add "storage": "gpu" */
				k.type = jbvString;
				k.val.string.len = strlen("storage");
				k.val.string.val = "storage";
				(void)pushJsonbValue(&state, WJB_KEY, &k);
				
				v.type = jbvString;
				v.val.string.len = strlen("gpu");
				v.val.string.val = "gpu";
				(void)pushJsonbValue(&state, WJB_VALUE, &v);
				
				/* Add "n_classes": class_count */
				k.type = jbvString;
				k.val.string.len = strlen("n_classes");
				k.val.string.val = "n_classes";
				(void)pushJsonbValue(&state, WJB_KEY, &k);
				
				v.type = jbvNumeric;
				v.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(class_count)));
				(void)pushJsonbValue(&state, WJB_VALUE, &v);
				
				final_metrics = JsonbValueToJsonb(pushJsonbValue(&state, WJB_END_OBJECT, NULL));
			}

			memset(&spec, 0, sizeof(MLCatalogModelSpec));
			spec.project_name = project_name;
			spec.algorithm = "decision_tree";
			spec.training_table = table_name;
			spec.training_column = target_column;
			spec.model_data = gpu_model_data;
			spec.metrics = final_metrics;
			spec.num_samples = n_samples;
			spec.num_features = feature_dim;

			model_id = ml_catalog_register_model(&spec);

			/* Cleanup */
			if (feature_matrix)
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
			if (label_vector)
				NDB_SAFE_PFREE_AND_NULL(label_vector);
			if (gpu_errstr && *gpu_errstr)
			{
				NDB_SAFE_PFREE_AND_NULL(*gpu_errstr);
				*gpu_errstr = NULL;
			}

			/* Free model_name before deleting callcontext */
			if (model_name)
			{
		MemoryContextSwitchTo(callcontext);
				NDB_SAFE_PFREE_AND_NULL(model_name);
				model_name = NULL;
			}
			MemoryContextSwitchTo(oldcontext);
			
			if (model_id > 0)
			{
				neurondb_cleanup(oldcontext, callcontext, false, we_connected_spi);
				PG_RETURN_INT32(model_id);
			}
			else
			{
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("Failed to register Decision Tree model in catalog")));
			}
		}

		/* If GPU training failed, fall back to CPU training */
		if (feature_matrix)
			NDB_SAFE_PFREE_AND_NULL(feature_matrix);
		if (label_vector)
			NDB_SAFE_PFREE_AND_NULL(label_vector);

		/* Fall back to CPU training via SQL function */
		if (neurondb_gpu_enabled)
			elog(INFO, "neurondb: Falling back to CPU training for decision_tree: table=%s, features=%s, target=%s",
				 table_name ? table_name : "(null)",
				 feature_list.data ? feature_list.data : "(null)",
				 target_column ? target_column : "(null)");
		
		max_depth = 10;
		min_samples = 100;
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
					NDB_SAFE_PFREE_AND_NULL(key);
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
		char *gpu_errstr_ptr = NULL;
		char **gpu_errstr = &gpu_errstr_ptr;
		bool gpu_trained = false;

		/* Load training data */
		if (!neurondb_load_training_data(table_name, feature_list.data, target_column,
										 &feature_matrix, &label_vector,
										 &n_samples, &feature_dim, &class_count))
		{
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Failed to load training data for Naive Bayes")));
		}

		if (n_samples < 10)
		{
			if (feature_matrix)
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
			if (label_vector)
				NDB_SAFE_PFREE_AND_NULL(label_vector);
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Naive Bayes: need at least 10 samples, got %d", n_samples)));
		}

		/* Try GPU training through proper bridge */
		MemSet(&gpu_result, 0, sizeof(MLGpuTrainResult));
		if (ndb_gpu_try_train_model("naive_bayes", project_name, model_name, table_name, target_column,
									feature_names, feature_name_count, hyperparams,
									feature_matrix, label_vector, n_samples, feature_dim, class_count,
									&gpu_result, gpu_errstr) && gpu_result.spec.model_data != NULL)
		{
			gpu_trained = true;
			gpu_model_data = gpu_result.spec.model_data;
			gpu_metrics = gpu_result.spec.metrics;
		}
		else
		{
			/* Log the error before freeing it */
			if (gpu_errstr != NULL && *gpu_errstr != NULL)
			{
				elog(DEBUG1,
					"neurondb: GPU training for naive_bayes failed: %s",
					*gpu_errstr);
				NDB_SAFE_PFREE_AND_NULL(*gpu_errstr);
				*gpu_errstr = NULL;
			}
			else
			{
				elog(DEBUG1,
					"neurondb: GPU training for naive_bayes failed: no error message (ndb_gpu_try_train_model returned false or model_data is NULL)");
			}
		}

		/* If GPU training failed, try CPU training */
		if (!gpu_trained)
		{
			/* Clean up allocated training data */
			if (feature_matrix)
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
			if (label_vector)
				NDB_SAFE_PFREE_AND_NULL(label_vector);

			/* Fall back to CPU training via SQL function */
			/* Only log as INFO if GPU was enabled but failed; use DEBUG1 if GPU was disabled */
			if (neurondb_gpu_enabled)
				elog(INFO, "neurondb: Falling back to CPU training for naive_bayes: table=%s, features=%s, target=%s",
					 table_name ? table_name : "(null)",
					 feature_list.data ? feature_list.data : "(null)",
					 target_column ? target_column : "(null)");
			else
				elog(DEBUG1, "neurondb: Using CPU training for naive_bayes (GPU disabled): table=%s, features=%s, target=%s",
					 table_name ? table_name : "(null)",
					 feature_list.data ? feature_list.data : "(null)",
					 target_column ? target_column : "(null)");
			/* Use safe free/reinit to handle potential memory context changes */
			NDB_SAFE_PFREE_AND_NULL(sql.data);
			initStringInfo(&sql);
			appendStringInfo(&sql,
				"SELECT train_naive_bayes_classifier_model_id(%s::text, %s::text, %s::text)",
				neurondb_quote_literal_cstr(table_name),
				neurondb_quote_literal_cstr(feature_list.data),
				neurondb_quote_literal_or_null(target_column));
			ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
			if (ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
													   SPI_tuptable->tupdesc, 1, &isnull));
				if (!isnull && model_id > 0)
				{
					/* Update metrics to ensure storage='cpu' is set */
					/* Use safe free/reinit to handle potential memory context changes */
					NDB_SAFE_PFREE_AND_NULL(sql.data);
					initStringInfo(&sql);
					appendStringInfo(&sql,
						"UPDATE neurondb.ml_models SET metrics = "
						"COALESCE(metrics, '{}'::jsonb) || '{\"storage\": \"cpu\"}'::jsonb "
						"WHERE model_id = %d",
						model_id);
					ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
					goto train_complete;
				}
			}
			/* If CPU training failed, report error */
			if (feature_matrix)
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
			if (label_vector)
				NDB_SAFE_PFREE_AND_NULL(label_vector);
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Failed to train Naive Bayes model: both GPU and CPU training failed")));
		}
		else
		{
			/* Register GPU-trained model */
			MLCatalogModelSpec spec;
			Jsonb *final_metrics = NULL;

		/* Use safe free/reinit to handle potential memory context changes */
		NDB_SAFE_PFREE_AND_NULL(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
		ret = ndb_spi_execute_safe(sql.data, false, 0);
		NDB_CHECK_SPI_TUPTABLE_IF_SELECT(ret);
		if (ret != SPI_OK_SELECT)
			{
				if (feature_matrix)
					NDB_SAFE_PFREE_AND_NULL(feature_matrix);
				if (label_vector)
					NDB_SAFE_PFREE_AND_NULL(label_vector);
				if (gpu_model_data)
					NDB_SAFE_PFREE_AND_NULL(gpu_model_data);
				if (gpu_metrics)
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
				if (gpu_errstr)
					NDB_SAFE_PFREE_AND_NULL(gpu_errstr);
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
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
			/* Don't use DirectFunctionCall - it crashes in CUDA context.
			 * Build JSONB manually using JsonbBuilder API. */
			JsonbParseState *state = NULL;
			JsonbValue k, v;
			
			(void)pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);
			
			/* Add "storage": "gpu" */
			k.type = jbvString;
			k.val.string.len = strlen("storage");
			k.val.string.val = "storage";
			(void)pushJsonbValue(&state, WJB_KEY, &k);
			
			v.type = jbvString;
			v.val.string.len = strlen("gpu");
			v.val.string.val = "gpu";
			(void)pushJsonbValue(&state, WJB_VALUE, &v);
			
			/* Add "n_classes": class_count */
			k.type = jbvString;
			k.val.string.len = strlen("n_classes");
			k.val.string.val = "n_classes";
			(void)pushJsonbValue(&state, WJB_KEY, &k);
			
			v.type = jbvNumeric;
			v.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(class_count)));
			(void)pushJsonbValue(&state, WJB_VALUE, &v);
			
			final_metrics = JsonbValueToJsonb(pushJsonbValue(&state, WJB_END_OBJECT, NULL));
		}			memset(&spec, 0, sizeof(MLCatalogModelSpec));
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
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
			if (label_vector)
				NDB_SAFE_PFREE_AND_NULL(label_vector);
			if (gpu_errstr && *gpu_errstr)
			{
				NDB_SAFE_PFREE_AND_NULL(*gpu_errstr);
				*gpu_errstr = NULL;
			}

			/* Free model_name before deleting callcontext */
			if (model_name)
			{
		MemoryContextSwitchTo(callcontext);
				NDB_SAFE_PFREE_AND_NULL(model_name);
				model_name = NULL;
			}
			MemoryContextSwitchTo(oldcontext);
			
			if (model_id > 0)
			{
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
				PG_RETURN_INT32(model_id);
			}
			else
			{
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("failed to register Naive Bayes model")));
			}
		}
	}
	else if (strcmp(algorithm, "knn") == 0 || strcmp(algorithm, "knn_classifier") == 0 || strcmp(algorithm, "knn_regressor") == 0)
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
		char *gpu_errstr_ptr = NULL;
		char **gpu_errstr = &gpu_errstr_ptr;
		bool gpu_trained = false;

		/* Set task type based on algorithm */
		if (strcmp(algorithm, "knn_regressor") == 0)
			task_type = 1;

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
					NDB_SAFE_PFREE_AND_NULL(key);
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
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Failed to load training data for KNN")));
		}

		if (n_samples < k_value)
		{
			if (feature_matrix)
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
			if (label_vector)
				NDB_SAFE_PFREE_AND_NULL(label_vector);
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("KNN: need at least %d samples, got %d", k_value, n_samples)));
		}

		/* Try GPU training through proper bridge */
		elog(DEBUG1, "neurondb: GPU training for knn: n_samples=%d, feature_dim=%d, k=%d",
			 n_samples, feature_dim, k_value);
		
		MemSet(&gpu_result, 0, sizeof(MLGpuTrainResult));
		if (ndb_gpu_try_train_model("knn", project_name, model_name, table_name, target_column,
									feature_names, feature_name_count, hyperparams,
									feature_matrix, label_vector, n_samples, feature_dim, class_count,
									&gpu_result, gpu_errstr) && gpu_result.spec.model_data != NULL)
		{
			gpu_trained = true;
			gpu_model_data = gpu_result.spec.model_data;
			gpu_metrics = gpu_result.spec.metrics;
		}
		else
		{
			/* Log the error before freeing it */
			if (gpu_errstr != NULL && *gpu_errstr != NULL)
			{
				elog(DEBUG1,
					"neurondb: GPU training for knn failed: %s",
					*gpu_errstr);
				NDB_SAFE_PFREE_AND_NULL(*gpu_errstr);
				*gpu_errstr = NULL;
			}
			else
			{
				elog(DEBUG1,
					"neurondb: GPU training for knn failed: no error message (ndb_gpu_try_train_model returned false or model_data is NULL)");
			}
		}

		/* If GPU training failed, try CPU training */
		if (!gpu_trained)
		{
			/* For KNN CPU training, we already have the data loaded, so create model directly */
			/* Only log as INFO if GPU was enabled but failed; use DEBUG1 if GPU was disabled */
			if (neurondb_gpu_enabled)
				elog(INFO, "neurondb: Falling back to CPU training for knn: table=%s, features=%s, target=%s",
					 table_name ? table_name : "(null)",
					 feature_list.data ? feature_list.data : "(null)",
					 target_column ? target_column : "(null)");
			else
				elog(DEBUG1, "neurondb: Using CPU training for knn (GPU disabled): table=%s, features=%s, target=%s",
					 table_name ? table_name : "(null)",
					 feature_list.data ? feature_list.data : "(null)",
					 target_column ? target_column : "(null)");

			elog(INFO, "neurondb: Starting KNN CPU training path");

			/* Create KNN model directly without calling train_knn_model_id via SQL */
			/* This avoids nested SPI issues */
			elog(DEBUG1, "neurondb: KNN CPU training - creating model directly");
			{
				bytea *model_data = NULL;
				Jsonb *metrics = NULL;
				MLCatalogModelSpec spec;
				StringInfoData model_buf;
				StringInfoData metrics_json;
				char *tbl_str = pstrdup(table_name);
				char *feat_str = pstrdup(feature_list.data);
				char *label_str = pstrdup(target_column);
				
				/* For KNN (lazy learner), store minimal model data: just k and dimensions */
				/* The actual training data stays in the table */
				initStringInfo(&model_buf);
				appendBinaryStringInfo(&model_buf, (char *)&k_value, sizeof(int));
				appendBinaryStringInfo(&model_buf, (char *)&n_samples, sizeof(int));
				appendBinaryStringInfo(&model_buf, (char *)&feature_dim, sizeof(int));
				
				/* Store table name, feature col, label col as strings for CPU prediction */
				appendBinaryStringInfo(&model_buf, tbl_str, strlen(tbl_str) + 1);
				appendBinaryStringInfo(&model_buf, feat_str, strlen(feat_str) + 1);
				appendBinaryStringInfo(&model_buf, label_str, strlen(label_str) + 1);
				
				{
					int total_size = VARHDRSZ + model_buf.len;
					model_data = (bytea *)palloc(total_size);
					SET_VARSIZE(model_data, total_size);
					memcpy(VARDATA(model_data), model_buf.data, model_buf.len);
					NDB_SAFE_PFREE_AND_NULL(model_buf.data);
				}
				
				/* Build metrics JSONB */
				initStringInfo(&metrics_json);
				appendStringInfo(&metrics_json, "{\"storage\": \"cpu\", \"k\": %d, \"n_samples\": %d, \"n_features\": %d}",
					k_value, n_samples, feature_dim);
				metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metrics_json.data)));
				NDB_SAFE_PFREE_AND_NULL(metrics_json.data);
				
				/* Store model in catalog */
				memset(&spec, 0, sizeof(MLCatalogModelSpec));
				spec.project_name = project_name;
				spec.algorithm = "knn";
				spec.training_table = tbl_str;
				spec.training_column = label_str;
				spec.model_data = model_data;
				spec.metrics = metrics;
				spec.num_samples = n_samples;
				spec.num_features = feature_dim;
				
				/* Directly register the model using existing SPI connection to avoid nested SPI */
				{
					StringInfoData insert_sql;
					StringInfoData algorithm_quoted;
					StringInfoData table_quoted;
					StringInfoData column_quoted;
					StringInfoData metrics_quoted;
					StringInfoData metrics_txt;
					int version = 1;
					int32 model_id_local = 0;
					int ret_local;

					elog(DEBUG1, "neurondb: KNN CPU training - registering model directly");

					/* Get project ID and version */
					/* Use safe free/reinit to handle potential memory context changes */
					NDB_SAFE_PFREE_AND_NULL(sql.data);
					initStringInfo(&sql);
					appendStringInfo(&sql,
						"INSERT INTO neurondb.ml_projects "
						"(project_name, model_type, description) "
						"VALUES (%s, 'classification', 'Auto-created by neurondb.train') "
						"ON CONFLICT (project_name) DO UPDATE "
						"  SET updated_at = NOW() "
						"RETURNING project_id",
						quote_literal_cstr(project_name));
				ret_local = ndb_spi_execute_safe(sql.data, false, 0);
				NDB_CHECK_SPI_TUPTABLE_IF_SELECT(ret_local);
				if (ret_local != SPI_OK_INSERT_RETURNING && ret_local != SPI_OK_UPDATE_RETURNING) {
						NDB_SAFE_PFREE_AND_NULL(tbl_str);
						NDB_SAFE_PFREE_AND_NULL(feat_str);
						NDB_SAFE_PFREE_AND_NULL(label_str);
						if (feature_matrix)
							NDB_SAFE_PFREE_AND_NULL(feature_matrix);
						if (label_vector)
							NDB_SAFE_PFREE_AND_NULL(label_vector);
						neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("Failed to train KNN model: failed to get/create project")));
					}
					project_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

					/* Get next version */
					/* Use safe free/reinit to handle potential memory context changes */
					NDB_SAFE_PFREE_AND_NULL(sql.data);
					initStringInfo(&sql);
					appendStringInfo(&sql,
						"SELECT COALESCE(MAX(version), 0) + 1 FROM neurondb.ml_models WHERE project_id = %d",
						project_id);
					ret_local = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
					if (ret_local == SPI_OK_SELECT && SPI_processed > 0) {
						version = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
					}

					/* Build quoted strings */
					initStringInfo(&algorithm_quoted);
					appendStringInfo(&algorithm_quoted, "'%s'", spec.algorithm);

					initStringInfo(&table_quoted);
					appendStringInfo(&table_quoted, "'%s'", spec.training_table);

					initStringInfo(&column_quoted);
					appendStringInfo(&column_quoted, "'%s'", spec.training_column);

					initStringInfo(&metrics_txt);
					appendStringInfo(&metrics_txt, "%s", DatumGetCString(DirectFunctionCall1(jsonb_out, JsonbPGetDatum(spec.metrics))));

					initStringInfo(&metrics_quoted);
					appendStringInfo(&metrics_quoted, "'%s'", metrics_txt.data);

					/* Insert the model */
					initStringInfo(&insert_sql);
					appendStringInfo(&insert_sql,
							"INSERT INTO neurondb.ml_models "
							"(project_id, version, algorithm, status, "
							"training_table, "
							"training_column, model_data, metrics, "
							"num_samples, num_features, "
							"completed_at) "
							"VALUES (%d, %d, %s::neurondb.ml_algorithm_type, 'completed', "
							"%s, %s, NULL, %s::jsonb, "
							"%d, %d, NOW()) "
							"RETURNING model_id",
							project_id,
							version,
							algorithm_quoted.data,
							table_quoted.data,
							column_quoted.data,
							metrics_quoted.data,
							spec.num_samples,
							spec.num_features);

					ret_local = ndb_spi_execute_safe(insert_sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
					if ((ret_local != SPI_OK_INSERT_RETURNING && ret_local != SPI_OK_UPDATE_RETURNING)
						|| SPI_processed == 0) {
						NDB_SAFE_PFREE_AND_NULL(tbl_str);
						NDB_SAFE_PFREE_AND_NULL(feat_str);
						NDB_SAFE_PFREE_AND_NULL(label_str);
						NDB_SAFE_PFREE_AND_NULL(algorithm_quoted.data);
						NDB_SAFE_PFREE_AND_NULL(table_quoted.data);
						NDB_SAFE_PFREE_AND_NULL(column_quoted.data);
						NDB_SAFE_PFREE_AND_NULL(metrics_txt.data);
						NDB_SAFE_PFREE_AND_NULL(metrics_quoted.data);
						NDB_SAFE_PFREE_AND_NULL(insert_sql.data);
						if (feature_matrix)
							NDB_SAFE_PFREE_AND_NULL(feature_matrix);
						if (label_vector)
							NDB_SAFE_PFREE_AND_NULL(label_vector);
						neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("Failed to train KNN model: failed to insert model")));
					}

					model_id_local = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

					/* Update model_data separately */
					/* Use safe free/reinit to handle potential memory context changes */
					NDB_SAFE_PFREE_AND_NULL(sql.data);
					initStringInfo(&sql);
					appendStringInfo(&sql,
						"UPDATE neurondb.ml_models SET model_data = $1 WHERE model_id = %d",
						model_id_local);

					{
						Oid argtypes[1] = {BYTEAOID};
						Datum values[1] = {PointerGetDatum(spec.model_data)};
						char nulls[1] = {' '};

						ret_local = SPI_execute_with_args(sql.data, 1, argtypes, values, nulls, false, 0);
						if (ret_local != SPI_OK_UPDATE) {
							NDB_SAFE_PFREE_AND_NULL(tbl_str);
							NDB_SAFE_PFREE_AND_NULL(feat_str);
							NDB_SAFE_PFREE_AND_NULL(label_str);
							NDB_SAFE_PFREE_AND_NULL(algorithm_quoted.data);
							NDB_SAFE_PFREE_AND_NULL(table_quoted.data);
							NDB_SAFE_PFREE_AND_NULL(column_quoted.data);
							NDB_SAFE_PFREE_AND_NULL(metrics_txt.data);
							NDB_SAFE_PFREE_AND_NULL(metrics_quoted.data);
							NDB_SAFE_PFREE_AND_NULL(insert_sql.data);
							if (feature_matrix)
								NDB_SAFE_PFREE_AND_NULL(feature_matrix);
							if (label_vector)
								NDB_SAFE_PFREE_AND_NULL(label_vector);
							neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
							ereport(ERROR,
									(errcode(ERRCODE_INTERNAL_ERROR),
									 errmsg("Failed to train KNN model: failed to update model_data")));
						}
					}

					/* Cleanup */
					NDB_SAFE_PFREE_AND_NULL(algorithm_quoted.data);
					NDB_SAFE_PFREE_AND_NULL(table_quoted.data);
					NDB_SAFE_PFREE_AND_NULL(column_quoted.data);
					NDB_SAFE_PFREE_AND_NULL(metrics_txt.data);
					NDB_SAFE_PFREE_AND_NULL(metrics_quoted.data);
					NDB_SAFE_PFREE_AND_NULL(insert_sql.data);
					model_id = model_id_local;
				}

				/* Cleanup */
				NDB_SAFE_PFREE_AND_NULL(tbl_str);
				NDB_SAFE_PFREE_AND_NULL(feat_str);
				NDB_SAFE_PFREE_AND_NULL(label_str);

				/* Clean up training data that was loaded but not used for KNN */
				if (feature_matrix)
					NDB_SAFE_PFREE_AND_NULL(feature_matrix);
				if (label_vector)
					NDB_SAFE_PFREE_AND_NULL(label_vector);

				if (model_id > 0)
				{
					goto train_complete;
				}
				else
				{
					neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("Failed to train KNN model: failed to register model in catalog")));
				}
			}
		}
		else
		{
			/* Register GPU-trained model */
			MLCatalogModelSpec spec;
			StringInfoData metrics_json;
			Jsonb *final_metrics = NULL;

		/* Use safe free/reinit to handle potential memory context changes */
		NDB_SAFE_PFREE_AND_NULL(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
		ret = ndb_spi_execute_safe(sql.data, false, 0);
		NDB_CHECK_SPI_TUPTABLE_IF_SELECT(ret);
		if (ret != SPI_OK_SELECT)
			{
				if (feature_matrix)
					NDB_SAFE_PFREE_AND_NULL(feature_matrix);
				if (label_vector)
					NDB_SAFE_PFREE_AND_NULL(label_vector);
				if (gpu_model_data)
					NDB_SAFE_PFREE_AND_NULL(gpu_model_data);
				if (gpu_metrics)
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
				if (gpu_errstr)
					NDB_SAFE_PFREE_AND_NULL(gpu_errstr);
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
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
				NDB_SAFE_PFREE_AND_NULL(metrics_json.data);
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
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
			if (label_vector)
				NDB_SAFE_PFREE_AND_NULL(label_vector);
			if (gpu_errstr && *gpu_errstr)
			{
				NDB_SAFE_PFREE_AND_NULL(*gpu_errstr);
				*gpu_errstr = NULL;
			}

			/* Free model_name before deleting callcontext */
			if (model_name)
			{
		MemoryContextSwitchTo(callcontext);
				NDB_SAFE_PFREE_AND_NULL(model_name);
				model_name = NULL;
			}
			MemoryContextSwitchTo(oldcontext);
			
			if (model_id > 0)
			{
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
				PG_RETURN_INT32(model_id);
			}
			else
			{
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
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
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}
		appendStringInfo(&sql,
			"SELECT train_ridge_regression(%s, %s, %s, %.6f)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_or_null(target_column),
			alpha);
	}
	else if (strcmp(algorithm, "lasso") == 0)
	{
		double alpha = 1.0;
		int max_iters = 1000;
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
					else if (strcmp(key, "max_iters") == 0 && v.type == jbvNumeric)
						max_iters = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}
		appendStringInfo(&sql,
			"SELECT train_lasso_regression(%s, %s, %s, %.6f, %d)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_or_null(target_column),
			alpha,
			max_iters);
	}
	else if (strcmp(algorithm, "rag") == 0 || strcmp(algorithm, "hybrid_search") == 0)
	{
		/* RAG and hybrid_search are special algorithms that don't require traditional training */
		/* They are handled by their respective modules, but we need to register a placeholder model */
		MLCatalogModelSpec spec;
		memset(&spec, 0, sizeof(spec));
		spec.algorithm = algorithm;
		spec.model_type = "embedding";  /* RAG and hybrid_search use embedding type */
		spec.training_table = table_name;
		spec.training_column = target_column;
		spec.parameters = hyperparams;
		spec.metrics = NULL;
		spec.model_data = NULL;
		spec.training_time_ms = 0;
		spec.num_samples = 0;
		spec.num_features = 0;
		
		model_id = ml_catalog_register_model(&spec);
		if (model_id <= 0)
		{
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Failed to register %s model", algorithm)));
		}
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		PG_RETURN_INT32(model_id);
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
		char *gpu_errstr_ptr = NULL;
		char **gpu_errstr = &gpu_errstr_ptr;
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
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}

		/* Load training data */
		if (!neurondb_load_training_data(table_name, feature_list.data, NULL,
										 &feature_matrix, &label_vector,
										 &n_samples, &feature_dim, &class_count))
		{
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Failed to load training data for GMM")));
		}

		if (n_samples < k_value)
		{
			if (feature_matrix)
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
			if (label_vector)
				NDB_SAFE_PFREE_AND_NULL(label_vector);
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("GMM: need at least %d samples, got %d", k_value, n_samples)));
		}

		/* Try GPU training through proper bridge */
		elog(DEBUG1, "neurondb: GPU training for knn: n_samples=%d, feature_dim=%d, k=%d",
			 n_samples, feature_dim, k_value);
		
		MemSet(&gpu_result, 0, sizeof(MLGpuTrainResult));
		if (ndb_gpu_try_train_model("gmm", project_name, model_name, table_name, NULL,
									feature_names, feature_name_count, hyperparams,
									feature_matrix, NULL, n_samples, feature_dim, 0,
									&gpu_result, gpu_errstr) && gpu_result.spec.model_data != NULL)
		{
			gpu_trained = true;
			gpu_model_data = gpu_result.spec.model_data;
			gpu_metrics = gpu_result.spec.metrics;
		}
		else
		{
			/* Log the error before freeing it */
			if (gpu_errstr != NULL && *gpu_errstr != NULL)
			{
				elog(DEBUG1,
					"neurondb: GPU training for gmm failed: %s",
					*gpu_errstr);
				NDB_SAFE_PFREE_AND_NULL(*gpu_errstr);
				*gpu_errstr = NULL;
			}
			else
			{
				elog(DEBUG1,
					"neurondb: GPU training for gmm failed: no error message (ndb_gpu_try_train_model returned false or model_data is NULL)");
			}
		}

		/* If GPU training failed, try CPU training */
		if (!gpu_trained)
		{
			/* Clean up allocated training data */
			if (feature_matrix)
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
			if (label_vector)
				NDB_SAFE_PFREE_AND_NULL(label_vector);
			
			/* Fall back to CPU training via SQL function */
			/* Only log as INFO if GPU was enabled but failed; use DEBUG1 if GPU was disabled */
			if (neurondb_gpu_enabled)
				elog(INFO, "neurondb: Falling back to CPU training for gmm: table=%s, features=%s",
					 table_name ? table_name : "(null)",
					 feature_list.data ? feature_list.data : "(null)");
			else
				elog(DEBUG1, "neurondb: Using CPU training for gmm (GPU disabled): table=%s, features=%s",
					 table_name ? table_name : "(null)",
					 feature_list.data ? feature_list.data : "(null)");
			/* Use safe free/reinit to handle potential memory context changes */
			NDB_SAFE_PFREE_AND_NULL(sql.data);
			initStringInfo(&sql);
			appendStringInfo(&sql,
							 	"SELECT train_gmm_model_id(%s::text, %s::text, %d, %d)",
							 neurondb_quote_literal_cstr(table_name),
							 neurondb_quote_literal_cstr(feature_list.data),
							 k_value,
							 max_iters);
			ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
			if (ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
													   SPI_tuptable->tupdesc, 1, &isnull));
				if (!isnull && model_id > 0)
				{
					/* Update metrics to ensure storage='cpu' is set */
					/* Use safe free/reinit to handle potential memory context changes */
					NDB_SAFE_PFREE_AND_NULL(sql.data);
					initStringInfo(&sql);
					appendStringInfo(&sql,
						"UPDATE neurondb.ml_models SET metrics = "
						"COALESCE(metrics, '{}'::jsonb) || '{\"storage\": \"cpu\"}'::jsonb "
						"WHERE model_id = %d",
						model_id);
					ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
					goto train_complete;
				}
			}
			/* If CPU training failed, report error */
			if (feature_matrix)
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
			if (label_vector)
				NDB_SAFE_PFREE_AND_NULL(label_vector);
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Failed to train GMM model: both GPU and CPU training failed")));
		}
		else
		{
			/* Register GPU-trained model */
			MLCatalogModelSpec spec;
			StringInfoData metrics_json;
			Jsonb *final_metrics = NULL;

		/* Use safe free/reinit to handle potential memory context changes */
		NDB_SAFE_PFREE_AND_NULL(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
		ret = ndb_spi_execute_safe(sql.data, false, 0);
		NDB_CHECK_SPI_TUPTABLE_IF_SELECT(ret);
		if (ret != SPI_OK_SELECT)
			{
				if (feature_matrix)
					NDB_SAFE_PFREE_AND_NULL(feature_matrix);
				if (label_vector)
					NDB_SAFE_PFREE_AND_NULL(label_vector);
				if (gpu_model_data)
					NDB_SAFE_PFREE_AND_NULL(gpu_model_data);
				if (gpu_metrics)
					NDB_SAFE_PFREE_AND_NULL(gpu_metrics);
				if (gpu_errstr)
					NDB_SAFE_PFREE_AND_NULL(gpu_errstr);
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
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
				NDB_SAFE_PFREE_AND_NULL(metrics_json.data);
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
				NDB_SAFE_PFREE_AND_NULL(feature_matrix);
			if (label_vector)
				NDB_SAFE_PFREE_AND_NULL(label_vector);
			if (gpu_errstr && *gpu_errstr)
			{
				NDB_SAFE_PFREE_AND_NULL(*gpu_errstr);
				*gpu_errstr = NULL;
			}

			/* Free model_name before deleting callcontext */
			if (model_name)
			{
		MemoryContextSwitchTo(callcontext);
				NDB_SAFE_PFREE_AND_NULL(model_name);
				model_name = NULL;
			}
			MemoryContextSwitchTo(oldcontext);
			
			if (model_id > 0)
			{
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
				PG_RETURN_INT32(model_id);
			}
			else
			{
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("failed to register GMM model")));
			}
		}
	}
	else if (strcmp(algorithm, "kmeans") == 0)
	{
		int k_value = 3;
		int max_iters = 100;
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
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}
		appendStringInfo(&sql,
			"SELECT train_kmeans_model_id(%s, %s, %d, %d)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			k_value,
			max_iters);
	}
	else if (strcmp(algorithm, "minibatch_kmeans") == 0)
	{
		int k_value = 3;
		int batch_size = 100;
		int max_iters = 100;
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
					else if (strcmp(key, "batch_size") == 0 && v.type == jbvNumeric)
						batch_size = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "max_iters") == 0 && v.type == jbvNumeric)
						max_iters = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}
		/* Suppress unused variable warning - batch_size may be used in future implementation */
		(void) batch_size;
		/* minibatch_kmeans uses cluster_minibatch_kmeans which returns array, not model_id */
		/* For now, we'll use train_kmeans_model_id as a fallback */
		appendStringInfo(&sql,
			"SELECT train_kmeans_model_id(%s, %s, %d, %d)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			k_value,
			max_iters);
	}
	else if (strcmp(algorithm, "hierarchical") == 0)
	{
		int n_clusters = 3;
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
					if (strcmp(key, "n_clusters") == 0 && v.type == jbvNumeric)
						n_clusters = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}
		/* hierarchical clustering - check if train_hierarchical_model_id exists */
		/* For now, use train_kmeans_model_id as placeholder */
		appendStringInfo(&sql,
			"SELECT train_kmeans_model_id(%s, %s, %d, 100)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			n_clusters);
	}
	else if (strcmp(algorithm, "pca") == 0)
	{
		/* PCA doesn't have a train_pca_model_id function - it's a transformation */
		/* Free model_name before deleting callcontext */
		if (model_name)
		{
			MemoryContextSwitchTo(callcontext);
			NDB_SAFE_PFREE_AND_NULL(model_name);
			model_name = NULL;
		}
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("PCA is not supported via neurondb.train - use pca_transform() function directly")));
	}
	else if (strcmp(algorithm, "opq") == 0)
	{
		int num_subspaces = 8;
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
					if (strcmp(key, "num_subspaces") == 0 && v.type == jbvNumeric)
						num_subspaces = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}
		/* Suppress unused variable warning - num_subspaces may be used in future implementation */
		(void) num_subspaces;
		/* OPQ uses train_opq_rotation which returns float8[], not model_id */
		/* Free model_name before deleting callcontext */
		if (model_name)
		{
			MemoryContextSwitchTo(callcontext);
			NDB_SAFE_PFREE_AND_NULL(model_name);
			model_name = NULL;
		}
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("OPQ is not supported via neurondb.train - use train_opq_rotation() function directly")));
	}
	else if (strcmp(algorithm, "xgboost") == 0)
	{
		int n_estimators = 100;
		int max_depth = 6;
		double learning_rate = 0.1;
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
					if (strcmp(key, "n_estimators") == 0 && v.type == jbvNumeric)
						n_estimators = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "max_depth") == 0 && v.type == jbvNumeric)
						max_depth = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "learning_rate") == 0 && v.type == jbvNumeric)
						learning_rate = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(v.val.numeric)));
					NDB_SAFE_PFREE_AND_NULL(key);
				}
			}
		}
		appendStringInfo(&sql,
			"SELECT train_xgboost_classifier(%s, %s, %s, %d, %d, %.6f)",
			neurondb_quote_literal_cstr(table_name),
			neurondb_quote_literal_cstr(feature_list.data),
			neurondb_quote_literal_or_null(target_column),
			n_estimators,
			max_depth,
			learning_rate);
	}
	else
	{
		/* Free model_name before deleting callcontext */
		if (model_name)
		{
			MemoryContextSwitchTo(callcontext);
			NDB_SAFE_PFREE_AND_NULL(model_name);
			model_name = NULL;
		}
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Unsupported algorithm: \"%s\"", algorithm),
				 errhint("Supported algorithms: linear_regression, logistic_regression, random_forest, svm, decision_tree, naive_bayes, knn, knn_classifier, knn_regressor, ridge, lasso, gmm, kmeans, minibatch_kmeans, hierarchical, xgboost")));
	}

	if (strcmp(algorithm, "gmm") != 0 &&
		strcmp(algorithm, "naive_bayes") != 0 &&
		strcmp(algorithm, "knn") != 0 &&
		strcmp(algorithm, "knn_classifier") != 0 &&
		strcmp(algorithm, "knn_regressor") != 0 &&
		strcmp(algorithm, "rag") != 0 &&
		strcmp(algorithm, "hybrid_search") != 0)
	{
		/* Execute training function via SQL - keep SPI connected */
		/* The training function will handle its own SPI connection internally */
		{
			MemoryContext oldcontext_nested = MemoryContextSwitchTo(callcontext);
			ret = ndb_spi_execute_safe(sql.data, false, 0);
			NDB_CHECK_SPI_TUPTABLE_IF_SELECT(ret);
			MemoryContextSwitchTo(oldcontext_nested);
		}

		/* Extract model_id from result before any SPI operations clear it */
		if (strcmp(algorithm, "logistic_regression") == 0 ||
			strcmp(algorithm, "linear_regression") == 0 ||
			strcmp(algorithm, "decision_tree") == 0 ||
			strcmp(algorithm, "svm") == 0 ||
			strcmp(algorithm, "random_forest") == 0 ||
			strcmp(algorithm, "knn_regressor") == 0 ||
			strcmp(algorithm, "ridge") == 0 ||
			strcmp(algorithm, "lasso") == 0 ||
			strcmp(algorithm, "kmeans") == 0 ||
			strcmp(algorithm, "minibatch_kmeans") == 0 ||
			strcmp(algorithm, "hierarchical") == 0 ||
			strcmp(algorithm, "xgboost") == 0)
		{
			if (ret == SPI_OK_SELECT && SPI_processed > 0 && SPI_tuptable != NULL &&
				SPI_tuptable->tupdesc != NULL && SPI_tuptable->vals != NULL &&
				SPI_tuptable->vals[0] != NULL)
			{
				/* Extract model_id from result */
				model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
													   SPI_tuptable->tupdesc, 1, &isnull));
				if (isnull || model_id <= 0)
				{
					neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("%s training returned invalid model id", algorithm)));
				}
			}
			else
			{
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("%s training did not return a model id (ret=%d, processed=%ld, tuptable=%p)",
							algorithm, ret, (long)SPI_processed, (void *)SPI_tuptable)));
			}
		}
	}
	else if (strcmp(algorithm, "logistic_regression") == 0 ||
		strcmp(algorithm, "linear_regression") == 0 ||
		strcmp(algorithm, "decision_tree") == 0 ||
		strcmp(algorithm, "svm") == 0 ||
		strcmp(algorithm, "random_forest") == 0 ||
		strcmp(algorithm, "knn_regressor") == 0 ||
		strcmp(algorithm, "ridge") == 0 ||
		strcmp(algorithm, "lasso") == 0 ||
		strcmp(algorithm, "kmeans") == 0 ||
		strcmp(algorithm, "minibatch_kmeans") == 0 ||
		strcmp(algorithm, "hierarchical") == 0 ||
		strcmp(algorithm, "xgboost") == 0)
	{
		/* This path is for when training was done via direct function calls, not SQL */
		if (SPI_processed == 0)
		{
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("%s training did not return a model id", algorithm)));
		}
		NDB_CHECK_SPI_TUPTABLE();
		model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
											   SPI_tuptable->tupdesc, 1, &isnull));
		if (isnull)
		{
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("%s training returned NULL model id", algorithm)));
		}
	}

	if (ret < 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Training failed for algorithm \"%s\"", algorithm)));
	}

	{
		bool model_registered = false;
		if (model_id > 0)
		{
			model_registered = true;
		}
		if (!model_registered)
		{
		/* Use safe free/reinit to handle potential memory context changes */
		NDB_SAFE_PFREE_AND_NULL(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
		ret = ndb_spi_execute_safe(sql.data, false, 0);
		NDB_CHECK_SPI_TUPTABLE_IF_SELECT(ret);
		if (ret != SPI_OK_SELECT)
			{
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
				ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to acquire advisory lock")));
			}

			/* Use safe free/reinit to handle potential memory context changes */
			NDB_SAFE_PFREE_AND_NULL(sql.data);
			initStringInfo(&sql);
			appendStringInfo(&sql,
							 "WITH next_version AS (SELECT COALESCE(MAX(version), 0) + 1 AS v FROM neurondb.ml_models WHERE project_id = %d) "
							 "INSERT INTO neurondb.ml_models (project_id, version, algorithm, training_table, training_column, status, parameters) "
							 "SELECT %d, v, %s::neurondb.ml_algorithm_type, %s, %s, 'completed', '{}'::jsonb FROM next_version RETURNING model_id",
							 project_id,
							 project_id,
							 neurondb_quote_literal_cstr(algorithm),
							 neurondb_quote_literal_cstr(table_name),
							 neurondb_quote_literal_or_null(target_column));
			ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
			if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
			{
				neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
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
		bool gpu_enabled_guc = neurondb_gpu_enabled;
		bool gpu_available = neurondb_gpu_is_available();
		
		/* If GPU was not used (either disabled or not available), mark as CPU */
		if (!gpu_enabled_guc || !gpu_available)
		{
			/* Use safe free/reinit to handle potential memory context changes */
			NDB_SAFE_PFREE_AND_NULL(sql.data);
			initStringInfo(&sql);
			appendStringInfo(&sql,
				"UPDATE neurondb.ml_models SET metrics = "
				"COALESCE(metrics, '{}'::jsonb) || '{\"storage\": \"cpu\"}'::jsonb "
				"WHERE model_id = %d",
				model_id);
			ret = ndb_spi_execute_safe(sql.data, false, 0);
			NDB_CHECK_SPI_TUPTABLE_IF_SELECT(ret);
			if (ret != SPI_OK_UPDATE)
			{
			}
			else
			{
			}
		}
	}

train_complete:
	/* Free model_name before deleting callcontext */
	/* model_name was allocated in callcontext (via psprintf), so must free it while context is still valid */
	/* We may be in oldcontext or callcontext at this point, so switch to callcontext to free model_name */
	if (model_name)
	{
		MemoryContextSwitchTo(callcontext);
		NDB_SAFE_PFREE_AND_NULL(model_name);
		model_name = NULL;
	}
	if (model_id <= 0 || model_id > 2147483647)
	{
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb.train: invalid model_id (%d) returned for algorithm '%s'",
					model_id, algorithm),
				 errhint("model_id must be a positive integer. This may indicate a memory corruption issue.")));
	}
	
	/* Now switch back to oldcontext before cleanup */
	MemoryContextSwitchTo(oldcontext);
	neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
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
	bool		spi_was_connected;
	bool		we_connected_spi;

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

	spi_was_connected = (SPI_processed != -1);
	we_connected_spi = false;

	if (!spi_was_connected)
	{
		if (SPI_connect() != SPI_OK_CONNECT)
		{
			neurondb_cleanup(oldcontext, callcontext, false, we_connected_spi);
			ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("SPI_connect failed")));
		}
		we_connected_spi = true;
	}

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT algorithm::text FROM neurondb.ml_models WHERE model_id = %d",
		model_id);
	elog(DEBUG1, "neurondb_predict: executing query: %s", sql.data);
	ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Model not found: %d", model_id)));
	}
	algorithm = TextDatumGetCString(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
	if (isnull)
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Model algorithm is NULL for model_id=%d", model_id)));
	}

	ndims = ARR_NDIM(features_array);
	if (ndims != 1)
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("neurondb.predict: features must be 1-dimensional array, got %d dimensions", ndims)));
	}
	dims = ARR_DIMS(features_array);
	nelems = ArrayGetNItems(ndims, dims);
	if (nelems <= 0 || nelems > 100000)
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("neurondb.predict: invalid feature count: %d (expected 1-100000)", nelems)));
	}
	
	/* Validate array element type */
	if (ARR_ELEMTYPE(features_array) != FLOAT8OID && ARR_ELEMTYPE(features_array) != FLOAT4OID)
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
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
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
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
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
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

	/* Use safe free/reinit to handle potential memory context changes */
	NDB_SAFE_PFREE_AND_NULL(sql.data);
	initStringInfo(&sql);

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


		if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
		{
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Naive Bayes model %d not found", model_id)));
		}


		if (model_data == NULL)
		{
			if (metrics)
				NDB_SAFE_PFREE_AND_NULL(metrics);
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Naive Bayes model %d has no model data (model not trained)", model_id),
					 errhint("Naive Bayes training must be completed before prediction. The model may have been created without actual training.")));
		}


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
							found_storage = true;
							if (strcmp(storage, "gpu") == 0)
								is_gpu = true;
							NDB_SAFE_PFREE_AND_NULL(storage);
						}
						NDB_SAFE_PFREE_AND_NULL(key);
					}
				}
			}
			PG_CATCH();
			{
				/* If metrics parsing fails, assume CPU model */
				is_gpu = false;
			}
			PG_END_TRY();
			
			if (!found_storage)
			{
				is_gpu = false;
			}
		}
		else
		{
			is_gpu = false;
		}

		
		features_float = (float *) palloc(sizeof(float) * feature_dim);
		for (i = 0; i < feature_dim; i++)
			features_float[i] = (float) features[i];

		/* Use model's training backend (from catalog) regardless of current GPU state */
		{
			const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
			bool gpu_currently_enabled = (backend != NULL && neurondb_gpu_is_available());
			
			elog(DEBUG1,
				"neurondb_predict: model trained on %s, GPU currently %s",
				is_gpu ? "GPU" : "CPU",
				gpu_currently_enabled ? "enabled" : "disabled");
			
			if (is_gpu)
			{
				/* Model was trained on GPU - must use GPU prediction */
				if (!gpu_currently_enabled)
				{
					NDB_SAFE_PFREE_AND_NULL(features_float);
					if (model_data)
						NDB_SAFE_PFREE_AND_NULL(model_data);
					if (metrics)
						NDB_SAFE_PFREE_AND_NULL(metrics);
					neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
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
						NDB_SAFE_PFREE_AND_NULL(features_float);
						if (model_data)
							NDB_SAFE_PFREE_AND_NULL(model_data);
						if (metrics)
							NDB_SAFE_PFREE_AND_NULL(metrics);
						neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
						PG_RETURN_FLOAT8(prediction);
					}
					if (errstr)
						NDB_SAFE_PFREE_AND_NULL(errstr);
				}
			}
			else
			{
				/* Model was trained on CPU - use CPU prediction (ignore current GPU state) */
				/* Fall through to CPU prediction path below */
			}
		}
		appendStringInfo(&sql, "SELECT predict_naive_bayes_model_id(%d, %s)", model_id, features_str.data);
		NDB_SAFE_PFREE_AND_NULL(features_float);
		if (model_data)
			NDB_SAFE_PFREE_AND_NULL(model_data);
		if (metrics)
			NDB_SAFE_PFREE_AND_NULL(metrics);
	}
	else if (strcmp(algorithm, "ridge") == 0 || strcmp(algorithm, "lasso") == 0)
		appendStringInfo(&sql, "SELECT predict_regularized_regression(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "knn") == 0 || strcmp(algorithm, "knn_classifier") == 0 || strcmp(algorithm, "knn_regressor") == 0)
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
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("KNN model %d not found", model_id)));
		}

		if (model_data == NULL)
		{
			if (metrics)
				NDB_SAFE_PFREE_AND_NULL(metrics);
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
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
							NDB_SAFE_PFREE_AND_NULL(storage);
						}
						NDB_SAFE_PFREE_AND_NULL(key);
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
			
			elog(DEBUG1,
				"neurondb_predict: model trained on %s, GPU currently %s",
				is_gpu ? "GPU" : "CPU",
				gpu_currently_enabled ? "enabled" : "disabled");
			
			if (is_gpu)
			{
				/* Model was trained on GPU - must use GPU prediction */
				if (!gpu_currently_enabled)
				{
					NDB_SAFE_PFREE_AND_NULL(features_float);
					if (model_data)
						NDB_SAFE_PFREE_AND_NULL(model_data);
					if (metrics)
						NDB_SAFE_PFREE_AND_NULL(metrics);
					neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
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
						NDB_SAFE_PFREE_AND_NULL(features_float);
						if (model_data)
							NDB_SAFE_PFREE_AND_NULL(model_data);
						if (metrics)
							NDB_SAFE_PFREE_AND_NULL(metrics);
						neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
						PG_RETURN_FLOAT8(prediction);
					}
					if (errstr)
						NDB_SAFE_PFREE_AND_NULL(errstr);
				}
			}
			else
			{
				/* Model was trained on CPU - use CPU prediction (ignore current GPU state) */
				/* Fall through to CPU prediction path below */
			}
		}
		appendStringInfo(&sql, "SELECT predict_knn_model_id(%d, %s)", model_id, features_str.data);
		NDB_SAFE_PFREE_AND_NULL(features_float);
		if (model_data)
			NDB_SAFE_PFREE_AND_NULL(model_data);
		if (metrics)
			NDB_SAFE_PFREE_AND_NULL(metrics);
	}
	else if (strcmp(algorithm, "kmeans") == 0)
		appendStringInfo(&sql, "SELECT predict_kmeans_model_id(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "minibatch_kmeans") == 0)
		appendStringInfo(&sql, "SELECT predict_kmeans_model_id(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "hierarchical") == 0)
		appendStringInfo(&sql, "SELECT predict_hierarchical_cluster(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "xgboost") == 0)
		appendStringInfo(&sql, "SELECT predict_xgboost(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "gmm") == 0)
		appendStringInfo(&sql, "SELECT predict_gmm_model_id(%d, %s)", model_id, features_str.data);
	else
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Unsupported algorithm for prediction: \"%s\"", algorithm)));
	}

	ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Prediction query did not return a result")));
	}
	prediction = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
	neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);

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
	bool		spi_was_connected;
	bool		we_connected_spi;

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

	spi_was_connected = (SPI_processed != -1);
	we_connected_spi = false;

	if (!spi_was_connected)
	{
		if (SPI_connect() != SPI_OK_CONNECT)
		{
			neurondb_cleanup(oldcontext, callcontext, false, we_connected_spi);
			ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("SPI_connect failed")));
		}
		we_connected_spi = true;
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
	(void) ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();

	/* Use safe free/reinit to handle potential memory context changes */
	NDB_SAFE_PFREE_AND_NULL(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "INSERT INTO neurondb.ml_deployments (model_id, deployment_name, strategy, status, deployed_at) "
					 "VALUES (%d, %s, %s, 'active', CURRENT_TIMESTAMP) RETURNING deployment_id",
					 model_id,
					 neurondb_quote_literal_cstr(psprintf("deploy_%d_%ld", model_id, (long) time(NULL))),
					 neurondb_quote_literal_cstr(strategy));

	ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to create deployment")));
	}
	deployment_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);


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
	bool		spi_was_connected;
	bool		we_connected_spi;

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


	if (strcmp(model_format, "onnx") != 0 &&
		strcmp(model_format, "tensorflow") != 0 &&
		strcmp(model_format, "pytorch") != 0 &&
		strcmp(model_format, "sklearn") != 0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Unsupported model format: %s. Supported: onnx, tensorflow, pytorch, sklearn", model_format)));
	}

	elog(DEBUG1, "neurondb_load_model context");
	callcontext = AllocSetContextCreate(CurrentMemoryContext, "neurondb_load_model",
									   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	spi_was_connected = (SPI_processed != -1);
	we_connected_spi = false;

	if (!spi_was_connected)
	{
		if (SPI_connect() != SPI_OK_CONNECT)
		{
			neurondb_cleanup(oldcontext, callcontext, false, we_connected_spi);
			ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("SPI_connect failed")));
		}
		we_connected_spi = true;
	}

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"INSERT INTO neurondb.ml_projects (project_name, model_type, description) "
		"VALUES (%s, 'external', 'External model import') "
		"ON CONFLICT (project_name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP "
		"RETURNING project_id",
		neurondb_quote_literal_cstr(project_name));

	ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if ((ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING) || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to create/get external project \"%s\"", project_name)));
	}
	project_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

		/* Use safe free/reinit to handle potential memory context changes */
		NDB_SAFE_PFREE_AND_NULL(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
		ret = ndb_spi_execute_safe(sql.data, false, 0);
		NDB_CHECK_SPI_TUPTABLE_IF_SELECT(ret);
		if (ret != SPI_OK_SELECT)
		{
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to acquire advisory lock")));
		}

	/* Use safe free/reinit to handle potential memory context changes */
	NDB_SAFE_PFREE_AND_NULL(sql.data);
	initStringInfo(&sql);
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

	ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to register external model")));
	}
	model_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
	neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);

	PG_RETURN_INT32(model_id);
}

/* ----------
 * neurondb_evaluate
 * Unified model evaluation interface
 * Dispatches to algorithm-specific evaluate functions
 * ----------
 */
Datum
neurondb_evaluate(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name_text;
	text	   *feature_col_text;
	text	   *label_col_text;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	int			ret;
	bool		isnull = false;
	char	   *algorithm = NULL;
	char	   *table_name;
	char	   *feature_col;
	char	   *label_col;
	Jsonb	   *result = NULL;
	bool		spi_was_connected;
	bool		we_connected_spi;

	if (PG_NARGS() != 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb.evaluate: requires 4 arguments, got %d", PG_NARGS()),
				 errhint("Usage: neurondb.evaluate(model_id, table_name, feature_col, label_col)")));

	/* NULL input validation - prevent crashes */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb.evaluate: model_id cannot be NULL")));
	
	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb.evaluate: table_name cannot be NULL")));
	
	if (PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb.evaluate: feature_col cannot be NULL")));
	
	/* label_col can be NULL for unsupervised algorithms (e.g., kmeans, gmm) */
	
	model_id = PG_GETARG_INT32(0);
	table_name_text = PG_GETARG_TEXT_PP(1);
	feature_col_text = PG_GETARG_TEXT_PP(2);
	label_col_text = PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);
	
	/* Additional validation after getting arguments */
	if (model_id <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb.evaluate: model_id must be positive, got %d", model_id)));
	
	/* Validate text pointers are not NULL after conversion */
	if (table_name_text == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb.evaluate: table_name is NULL after conversion")));
	
	if (feature_col_text == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb.evaluate: feature_col is NULL after conversion")));

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
									   "neurondb_evaluate context",
									   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	spi_was_connected = (SPI_processed != -1);
	we_connected_spi = false;

	if (!spi_was_connected)
	{
		if (SPI_connect() != SPI_OK_CONNECT)
		{
			neurondb_cleanup(oldcontext, callcontext, false, we_connected_spi);
			ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("SPI_connect failed")));
		}
		we_connected_spi = true;
	}

	/* Get algorithm from model_id */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT algorithm::text FROM neurondb.ml_models WHERE model_id = %d",
		model_id);
	elog(DEBUG1, "neurondb_evaluate: executing query: %s", sql.data);
	ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Model not found: %d", model_id)));
	}
	{
		char *temp_algorithm;
		temp_algorithm = TextDatumGetCString(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
		if (isnull)
		{
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Model algorithm is NULL for model_id=%d", model_id)));
		}
		/* Copy algorithm to callcontext to avoid corruption from subsequent SPI calls */
		algorithm = pstrdup(temp_algorithm);
		Assert(algorithm != NULL);
		Assert(strlen(algorithm) > 0);
		NDB_SAFE_PFREE_AND_NULL(temp_algorithm);
	}

	/* Note: Early model validation removed - dt_model_deserialize and dt_model_free are static */
	/* Model validation will be handled by the evaluation function itself */

	table_name = text_to_cstring(table_name_text);
	feature_col = text_to_cstring(feature_col_text);
	label_col = label_col_text ? text_to_cstring(label_col_text) : NULL;

	/* Assertions for crash tracking */
	Assert(table_name != NULL);
	Assert(feature_col != NULL);
	Assert(callcontext != NULL);

	/* Validate label_col for supervised algorithms */
	if (strcmp(algorithm, "linear_regression") == 0 ||
		strcmp(algorithm, "logistic_regression") == 0 ||
		strcmp(algorithm, "ridge") == 0 ||
		strcmp(algorithm, "lasso") == 0 ||
		strcmp(algorithm, "random_forest") == 0 ||
		strcmp(algorithm, "svm") == 0 ||
		strcmp(algorithm, "decision_tree") == 0 ||
		strcmp(algorithm, "naive_bayes") == 0 ||
		strcmp(algorithm, "knn") == 0 ||
		strcmp(algorithm, "knn_classifier") == 0 ||
		strcmp(algorithm, "knn_regressor") == 0 ||
		strcmp(algorithm, "xgboost") == 0)
	{
		if (label_col == NULL)
		{
			neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("neurondb.evaluate: label_col cannot be NULL for supervised algorithm '%s'", algorithm)));
		}
	}

	/* Dispatch to algorithm-specific evaluate function */
	/* Use safe free/reinit to handle potential memory context changes */
	NDB_SAFE_PFREE_AND_NULL(sql.data);
	initStringInfo(&sql);
	if (strcmp(algorithm, "linear_regression") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_linear_regression_by_model_id(%d, %s, %s, %s)",
			model_id, q_table_name, q_feature_col, q_label_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
		NDB_SAFE_PFREE_AND_NULL(q_label_col);
	}
	else if (strcmp(algorithm, "logistic_regression") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_logistic_regression_by_model_id(%d, %s, %s, %s)",
			model_id, q_table_name, q_feature_col, q_label_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
		NDB_SAFE_PFREE_AND_NULL(q_label_col);
	}
	else if (strcmp(algorithm, "ridge") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_ridge_regression_by_model_id(%d, %s, %s, %s)",
			model_id, q_table_name, q_feature_col, q_label_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
		NDB_SAFE_PFREE_AND_NULL(q_label_col);
	}
	else if (strcmp(algorithm, "lasso") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_lasso_regression_by_model_id(%d, %s, %s, %s)",
			model_id, q_table_name, q_feature_col, q_label_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
		NDB_SAFE_PFREE_AND_NULL(q_label_col);
	}
	else if (strcmp(algorithm, "random_forest") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_random_forest_by_model_id(%d, %s, %s, %s)",
			model_id, q_table_name, q_feature_col, q_label_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
		NDB_SAFE_PFREE_AND_NULL(q_label_col);
	}
	else if (strcmp(algorithm, "svm") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_svm_by_model_id(%d, %s, %s, %s)",
			model_id, q_table_name, q_feature_col, q_label_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
		NDB_SAFE_PFREE_AND_NULL(q_label_col);
	}
	else if (strcmp(algorithm, "decision_tree") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_decision_tree_by_model_id(%d, %s, %s, %s)",
			model_id, q_table_name, q_feature_col, q_label_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
		NDB_SAFE_PFREE_AND_NULL(q_label_col);
	}
	else if (strcmp(algorithm, "naive_bayes") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_naive_bayes_by_model_id(%d, %s, %s, %s)",
			model_id, q_table_name, q_feature_col, q_label_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
		NDB_SAFE_PFREE_AND_NULL(q_label_col);
	}
	else if (strcmp(algorithm, "knn") == 0 || strcmp(algorithm, "knn_classifier") == 0 || strcmp(algorithm, "knn_regressor") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_knn_by_model_id(%d, %s, %s, %s)",
			model_id, q_table_name, q_feature_col, q_label_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
		NDB_SAFE_PFREE_AND_NULL(q_label_col);
	}
	else if (strcmp(algorithm, "kmeans") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);

		appendStringInfo(&sql, "SELECT evaluate_kmeans_by_model_id(%d, %s, %s)",
			model_id, q_table_name, q_feature_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
	}
	else if (strcmp(algorithm, "gmm") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);

		appendStringInfo(&sql, "SELECT evaluate_gmm_by_model_id(%d, %s, %s)",
			model_id, q_table_name, q_feature_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
	}
	else if (strcmp(algorithm, "minibatch_kmeans") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);

		appendStringInfo(&sql, "SELECT evaluate_minibatch_kmeans_by_model_id(%d, %s, %s)",
			model_id, q_table_name, q_feature_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
	}
	else if (strcmp(algorithm, "hierarchical") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);

		appendStringInfo(&sql, "SELECT evaluate_hierarchical_by_model_id(%d, %s, %s)",
			model_id, q_table_name, q_feature_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
	}
	else if (strcmp(algorithm, "xgboost") == 0)
	{
		char *q_table_name = neurondb_quote_literal_cstr(table_name);
		char *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_xgboost_by_model_id(%d, %s, %s, %s)",
			model_id, q_table_name, q_feature_col, q_label_col);

		NDB_SAFE_PFREE_AND_NULL(q_table_name);
		NDB_SAFE_PFREE_AND_NULL(q_feature_col);
		NDB_SAFE_PFREE_AND_NULL(q_label_col);
	}
	else
	{
		neurondb_cleanup(oldcontext, callcontext, true, we_connected_spi);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("neurondb.evaluate: algorithm '%s' does not support evaluation", algorithm)));
	}

	elog(DEBUG1, "neurondb_evaluate: executing query: %s", sql.data);
	
	/* Assertions for crash tracking */
	Assert(sql.data != NULL);
	Assert(strlen(sql.data) > 0);
	Assert(algorithm != NULL);
	
	/* Wrap entire evaluation in error handler to prevent crashes */
	PG_TRY();
	{
		ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			/* Evaluation query failed - return error JSONB instead of crashing */
			MemoryContextSwitchTo(oldcontext);
			result = (Jsonb *) DatumGetJsonbP(DirectFunctionCall1(jsonb_in, 
				CStringGetDatum("{\"error\": \"evaluation query failed\"}")));
			SPI_finish();
			MemoryContextDelete(callcontext);
			PG_RETURN_JSONB_P(result);
		}

		/* Validate SPI_tuptable before access */
		NDB_CHECK_SPI_TUPTABLE();

		{
			Datum result_datum;
			bool result_isnull;
			Jsonb *temp_jsonb;
			
			/* Get JSONB from SPI result */
			result_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &result_isnull);
			if (result_isnull)
			{
				/* Evaluation returned NULL - return error JSONB instead of crashing */
				MemoryContextSwitchTo(oldcontext);
				result = (Jsonb *) DatumGetJsonbP(DirectFunctionCall1(jsonb_in, 
					CStringGetDatum("{\"error\": \"evaluation returned NULL\"}")));
				SPI_finish();
				MemoryContextDelete(callcontext);
				PG_RETURN_JSONB_P(result);
			}
			
			/* Get JSONB pointer from datum (still in SPI context) */
			/* Validate pointer is not NULL before dereferencing */
			if (DatumGetPointer(result_datum) == NULL)
			{
				MemoryContextSwitchTo(oldcontext);
				result = (Jsonb *) DatumGetJsonbP(DirectFunctionCall1(jsonb_in, 
					CStringGetDatum("{\"error\": \"invalid JSONB pointer\"}")));
				SPI_finish();
				MemoryContextDelete(callcontext);
				PG_RETURN_JSONB_P(result);
			}
			
			temp_jsonb = DatumGetJsonbP(result_datum);
			
			/* Validate JSONB structure before using it */
			if (temp_jsonb == NULL || VARSIZE(temp_jsonb) < sizeof(Jsonb))
			{
				MemoryContextSwitchTo(oldcontext);
				result = (Jsonb *) DatumGetJsonbP(DirectFunctionCall1(jsonb_in, 
					CStringGetDatum("{\"error\": \"invalid JSONB structure\"}")));
				SPI_finish();
				MemoryContextDelete(callcontext);
				PG_RETURN_JSONB_P(result);
			}
			
			/*
			 * Copy JSONB to caller's context before SPI_finish().
			 * This ensures the JSONB is valid after SPI context is cleaned up.
			 * SPI_finish() will delete the SPI memory context, so any pointers
			 * to data allocated in that context will become invalid.
			 */
			MemoryContextSwitchTo(oldcontext);
			result = (Jsonb *) PG_DETOAST_DATUM_COPY((Datum) temp_jsonb);
			
			if (result == NULL || VARSIZE(result) < sizeof(Jsonb))
			{
				if (result != NULL)
				{
					NDB_SAFE_PFREE_AND_NULL(result);
					result = NULL;
				}
				result = (Jsonb *) DatumGetJsonbP(DirectFunctionCall1(jsonb_in, 
					CStringGetDatum("{\"error\": \"JSONB copy validation failed\"}")));
			}
		}
	}
	PG_CATCH();
	{
		ErrorData *edata = NULL;
		char *error_msg;
		
		/*
		 * Switch out of ErrorContext before CopyErrorData().
		 * CopyErrorData() allocates memory and must NOT be called while
		 * in ErrorContext, as that context is only for error reporting
		 * and will be reset, causing memory leaks or corruption.
		 */
		MemoryContextSwitchTo(oldcontext);
		
		/* Suppress shadow warnings from nested PG_TRY blocks */
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wshadow=compatible-local"
		PG_TRY();
		{
			if (CurrentMemoryContext != ErrorContext)
			{
				edata = CopyErrorData();
				FlushErrorState();
			}
			else
			{
				FlushErrorState();
			}
		}
		PG_CATCH();
		{
			FlushErrorState();
		}
		PG_END_TRY();
		#pragma GCC diagnostic pop
		
		/* Create safe error message (escape quotes) */
		if (edata != NULL && edata->message != NULL)
			error_msg = pstrdup(edata->message);
		else
			error_msg = pstrdup("evaluation failed (GPU may be unavailable or model data missing)");
		
		/* Escape JSON special characters */
		{
			char *escaped = palloc(strlen(error_msg) * 2 + 1);
			char *p = escaped;
			const char *s = error_msg;
			while (*s)
			{
				if (*s == '"' || *s == '\\' || *s == '\n' || *s == '\r')
				{
					*p++ = '\\';
					if (*s == '\n')
						*p++ = 'n';
					else if (*s == '\r')
						*p++ = 'r';
					else
						*p++ = *s;
				}
				else
					*p++ = *s;
				s++;
			}
			*p = '\0';
			NDB_SAFE_PFREE_AND_NULL(error_msg);
			error_msg = NULL;
			error_msg = escaped;
		}
		
		/* Return error JSONB */
		{
			StringInfoData error_json;
			initStringInfo(&error_json);
			appendStringInfo(&error_json, "{\"error\": \"%s\"}", error_msg);
			result = (Jsonb *) DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
				CStringGetDatum(error_json.data)));
			NDB_SAFE_PFREE_AND_NULL(error_json.data);
			error_json.data = NULL;
			NDB_SAFE_PFREE_AND_NULL(error_msg);
			error_msg = NULL;

			/* Defensive check: ensure result is valid */
			if (result == NULL || VARSIZE(result) < sizeof(Jsonb))
			{
				result = (Jsonb *) DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
					CStringGetDatum("{\"error\": \"JSONB creation failed\"}")));
			}
		}
		
		/* Free error data if we copied it */
		if (edata != NULL)
			FreeErrorData(edata);
		
		SPI_finish();
		
		/*
		 * Clean up memory context. Must switch to oldcontext before
		 * deleting callcontext, as SPI_finish() or error handling may
		 * have changed CurrentMemoryContext. Attempting to delete a
		 * context while in that context will cause a crash.
		 */
		if (callcontext)
		{
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(callcontext);
		}
		
		/* Return error JSONB immediately - don't fall through to cleanup code */
		PG_RETURN_JSONB_P(result);
	}
	PG_END_TRY();

	/* Now safe to clean up SPI and delete callcontext */
	SPI_finish();
	
	/* Ensure we're in oldcontext before deleting callcontext */
	/* SPI_finish() might have changed CurrentMemoryContext */
	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(callcontext);

	/* oldcontext is current, result lives there */

	/* CRITICAL SAFETY: Never return invalid data that could crash PostgreSQL */
	elog(NOTICE, "neurondb_evaluate: final validation - result=%p", result);

	if (result == NULL)
	{
		/* Instead of returning invalid data, throw an error */
		elog(ERROR, "neurondb_evaluate: CRITICAL - result is NULL");
	}

	if (VARSIZE(result) < sizeof(Jsonb))
	{
		/* Instead of returning invalid data, throw an error */
		elog(ERROR, "neurondb_evaluate: CRITICAL - invalid JSONB structure (size %d < %d)",
			 VARSIZE(result), (int)sizeof(Jsonb));
	}

	/* Additional validation - check if result is valid */
	if (result == NULL)
	{
		elog(WARNING, "neurondb_evaluate: result is NULL");
		result = (Jsonb *) DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
			CStringGetDatum("{\"error\": \"invalid JSONB structure\"}")));
	}

	elog(NOTICE, "neurondb_evaluate: about to return JSONB result, size=%d", VARSIZE(result));

	/* EMERGENCY SAFETY: Ensure result is ALWAYS valid before returning */
	if (result == NULL || VARSIZE(result) < sizeof(Jsonb))
	{
		/* Create emergency valid JSONB to prevent PostgreSQL crash */
		elog(WARNING, "neurondb_evaluate: EMERGENCY - creating safe error JSONB");
		result = (Jsonb *) DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
			CStringGetDatum("{\"error\": \"critical evaluation failure\"}")));
	}

	PG_RETURN_JSONB_P(result);
}
