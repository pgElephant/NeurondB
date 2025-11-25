/*-------------------------------------------------------------------------
 *
 * ml_catalog.c
 *	  Helpers for interacting with neurondb.ml_models catalog.
 *
 * This module provides a comprehensive set of utilities for machine learning
 * model catalog management within the NeuronDB extension. The catalog system
 * serves as the central registry for all trained models, storing metadata,
 * serialized model data, training parameters, and performance metrics.
 *
 * The primary functions in this module handle the complete lifecycle of model
 * registration and retrieval. When a model is trained, ml_catalog_register_model
 * creates or updates entries in the neurondb.ml_projects and neurondb.ml_models
 * tables, ensuring proper versioning, locking, and data integrity. The function
 * validates all input parameters, handles project creation automatically if needed,
 * acquires advisory locks to prevent race conditions during version calculation,
 * and stores all model metadata including hyperparameters, training metrics, and
 * optional serialized model data.
 *
 * Model retrieval is handled by ml_catalog_fetch_model_payload, which safely
 * extracts model data, parameters, and metrics from the catalog. This function
 * uses PostgreSQL's SPI interface with proper error handling to ensure that
 * memory contexts are correctly managed and all resources are properly cleaned
 * up even in error conditions. All returned data is allocated in the caller's
 * memory context using detoasted copies, ensuring persistence after SPI operations
 * complete.
 *
 * The catalog system supports multiple model versions per project, allowing
 * incremental model updates and A/B testing scenarios. Version numbers are
 * automatically calculated using advisory locks to ensure atomicity when multiple
 * concurrent registrations occur for the same project. The system also handles
 * optional model data storage separately from metadata, allowing for efficient
 * storage of large serialized models using parameterized queries.
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/elog.h"

#include "ml_catalog.h"
#include "neurondb_validation.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

/*
 * ml_catalog_default_project
 *		Generate default project name from algorithm and training table.
 *
 * Returns a palloc'd string that must be freed by caller.
 */
static const char *
ml_catalog_default_project(const char *algorithm, const char *training_table)
{
	if (algorithm == NULL)
		return pstrdup("ml_default_project");
	if (training_table == NULL)
		return psprintf("%s_project", algorithm);

	return psprintf("%s_%s_project", algorithm, training_table);
}

/*
 * ml_catalog_register_model
 *		Register a new model in neurondb.ml_models catalog.
 *
 * Validates inputs, creates/updates project entry, acquires versioning lock,
 * calculates next version, and inserts model metadata. Returns the model_id.
 */
int32
ml_catalog_register_model(const MLCatalogModelSpec * spec)
{
	const char *project_name;
	const char *model_type;
	const char *training_table;
	const char *training_column;
	const char *algorithm;
	Jsonb	   *parameters;
	Jsonb	   *metrics;
	int32		project_id = 0;
	int32		version = 1;
	int32		model_id = 0;
	int32		ret;
	int			error_code = 0;
	StringInfoData sql = {0};
	StringInfoData insert_query = {0};
	StringInfoData select_query = {0};
	StringInfoData insert_sql = {0};
	MemoryContext oldcontext;
	NDB_DECLARE (NdbSpiSession *, spi_session);
	NDB_DECLARE (char *, algorithm_copy);
	NDB_DECLARE (char *, training_table_copy);
	NDB_DECLARE (char *, training_column_copy);
	NDB_DECLARE (char *, project_name_copy);
	NDB_DECLARE (char *, params_txt);
	NDB_DECLARE (char *, metrics_txt);
	NDB_DECLARE (char *, params_quoted);
	NDB_DECLARE (char *, metrics_quoted);
	NDB_DECLARE (char *, algorithm_quoted);
	NDB_DECLARE (char *, table_quoted);
	NDB_DECLARE (char *, column_quoted);

	Assert(spec != NULL);
	if (spec == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid spec parameter"),
				 errdetail("The spec parameter is NULL"),
				 errhint("Provide a valid MLCatalogModelSpec structure with required fields.")));

	algorithm = spec->algorithm;
	if (algorithm == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: algorithm must be provided"),
				 errdetail("The algorithm field in spec is NULL"),
				 errhint("Set spec->algorithm to a valid algorithm name (e.g., 'logistic_regression', 'random_forest').")));

	training_table = spec->training_table;
	if (training_table == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: training_table must be provided"),
				 errdetail("The training_table field in spec is NULL"),
				 errhint("Set spec->training_table to the name of the table used for training.")));

	training_column = spec->training_column;

	if (spec->model_type != NULL)
		model_type = spec->model_type;
	else
		model_type = "classification";

	if (spec->project_name != NULL)
		project_name = spec->project_name;
	else
		project_name = ml_catalog_default_project(algorithm, training_table);

	parameters = spec->parameters;
	metrics = spec->metrics;

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	if (oldcontext == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: CurrentMemoryContext is NULL"),
				 errdetail("Cannot proceed without a valid memory context"),
				 errhint("This is an internal error. Please report this issue.")));
	}

	/* Copy input strings to persist across SPI operations */
	if (algorithm != NULL)
	{
		algorithm_copy = pstrdup(algorithm);
		algorithm = algorithm_copy;
	}

	if (training_table != NULL)
	{
		training_table_copy = pstrdup(training_table);
		training_table = training_table_copy;
	}

	if (training_column != NULL)
	{
		training_column_copy = pstrdup(training_column);
		training_column = training_column_copy;
	}

	if (spec->project_name == NULL && project_name != NULL)
	{
		project_name_copy = pstrdup(project_name);
		{
			char	   *temp = (char *) project_name;

			NDB_FREE(temp);
		}
		project_name = project_name_copy;
	}
	else if (spec->project_name != NULL && project_name != NULL)
	{
		project_name_copy = pstrdup(project_name);
		project_name = project_name_copy;
	}

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);
	Assert(spi_session != NULL);

	if (metrics != NULL)
	{
		NDB_DECLARE (char *, meta_txt);

		meta_txt = DatumGetCString(
								   DirectFunctionCall1(jsonb_out, JsonbPGetDatum(metrics)));
		elog(DEBUG2, "neurondb: ml_catalog_register_model: metrics JSON: %s", meta_txt);
		NDB_FREE(meta_txt);
	}

	elog(INFO, "neurondb: ml_catalog_register_model: registering model for project '%s', algorithm '%s', type '%s'",
		 project_name, algorithm, model_type);

	/* Create or update project entry and retrieve project_id */
	ndb_spi_stringinfo_init(spi_session, &insert_query);
	ndb_spi_stringinfo_init(spi_session, &select_query);

	appendStringInfo(&insert_query,
					 "INSERT INTO neurondb.ml_projects "
					 "(project_name, model_type, description) "
					 "VALUES ('%s', '%s'::neurondb.ml_model_type, '%s') "
					 "ON CONFLICT (project_name) DO UPDATE "
					 "SET updated_at = NOW()",
					 project_name,
					 model_type,
					 "Auto-created by ml_catalog_register_model");

	appendStringInfo(&select_query,
					 "SELECT project_id FROM neurondb.ml_projects WHERE project_name = '%s'",
					 project_name);

	elog(DEBUG1, "neurondb: ml_catalog_register_model: executing project insert");
	ret = ndb_spi_execute(spi_session, insert_query.data, false, 0);
	if (ret < 0)
	{
		error_code = 1;
		goto error;
	}

	elog(DEBUG1, "neurondb: ml_catalog_register_model: retrieving project_id");
	ret = ndb_spi_execute(spi_session, select_query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		error_code = 2;
		goto error;
	}

	ndb_spi_stringinfo_free(spi_session, &insert_query);
	ndb_spi_stringinfo_free(spi_session, &select_query);

	/* Retrieve and validate project_id */
	if (!ndb_spi_get_int32(spi_session, 0, 1, &project_id))
	{
		error_code = 3;
		goto error;
	}

	Assert(project_id > 0);
	if (project_id <= 0)
	{
		error_code = 4;
		goto error;
	}

	elog(DEBUG1, "neurondb: ml_catalog_register_model: acquired project_id %d", project_id);

	/* Acquire advisory lock to prevent race conditions during version calculation */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT pg_advisory_xact_lock(%d)", project_id);

	ret = ndb_spi_execute(spi_session, sql.data, false, 0);
	if (ret != SPI_OK_SELECT)
	{
		error_code = 5;
		goto error;
	}

	/* Calculate next version number */
	ndb_spi_stringinfo_reset(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT COALESCE(MAX(version), 0) + 1 "
					 "FROM neurondb.ml_models WHERE project_id = %d",
					 project_id);

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
		version = 1;
	else
	{
		if (!ndb_spi_get_int32(spi_session, 0, 0, &version))
			version = 1;
	}

	ndb_spi_stringinfo_free(spi_session, &sql);
	elog(DEBUG1, "neurondb: ml_catalog_register_model: calculated version %d for project_id %d", version, project_id);

	/* Insert model row with quoted string literals for SQL safety */
	ndb_spi_stringinfo_init(spi_session, &insert_sql);

	/* Quote string literals to prevent SQL injection */
	{
		NDB_DECLARE (text *, algorithm_text);
		NDB_DECLARE (text *, quoted_algorithm);

		algorithm_text = cstring_to_text(algorithm);
		quoted_algorithm = DatumGetTextP(
											 DirectFunctionCall1(quote_literal,
																 PointerGetDatum(algorithm_text)));
		algorithm_quoted = text_to_cstring(quoted_algorithm);
		NDB_FREE(algorithm_text);
		NDB_FREE(quoted_algorithm);
	}
	{
		NDB_DECLARE (text *, table_text);
		NDB_DECLARE (text *, quoted_table);

		table_text = cstring_to_text(training_table);
		quoted_table = DatumGetTextP(
										 DirectFunctionCall1(quote_literal,
															 PointerGetDatum(table_text)));
		table_quoted = text_to_cstring(quoted_table);
		NDB_FREE(table_text);
		NDB_FREE(quoted_table);
	}
	column_quoted = NULL;
	if (training_column != NULL)
	{
		NDB_DECLARE (text *, column_text);
		NDB_DECLARE (text *, quoted_column);

		column_text = cstring_to_text(training_column);
		quoted_column = DatumGetTextP(
										  DirectFunctionCall1(quote_literal,
															  PointerGetDatum(column_text)));
		column_quoted = text_to_cstring(quoted_column);
		NDB_FREE(column_text);
		NDB_FREE(quoted_column);
	}

	params_txt = NULL;
	params_quoted = NULL;
	if (parameters != NULL)
	{
		NDB_DECLARE (text *, params_text);
		NDB_DECLARE (text *, quoted_params);

		params_txt = DatumGetCString(
										 DirectFunctionCall1(jsonb_out, JsonbPGetDatum(parameters)));
		params_text = cstring_to_text(params_txt);
		quoted_params = DatumGetTextP(
										  DirectFunctionCall1(quote_literal,
															  PointerGetDatum(params_text)));
		params_quoted = text_to_cstring(quoted_params);
		NDB_FREE(params_text);
		NDB_FREE(quoted_params);
	}

	metrics_txt = NULL;
	metrics_quoted = NULL;
	if (metrics != NULL)
	{
		NDB_DECLARE (text *, metrics_text);
		NDB_DECLARE (text *, quoted_metrics);

		metrics_txt = DatumGetCString(
										  DirectFunctionCall1(jsonb_out, JsonbPGetDatum(metrics)));
		metrics_text = cstring_to_text(metrics_txt);
		quoted_metrics = DatumGetTextP(
										   DirectFunctionCall1(quote_literal,
															   PointerGetDatum(metrics_text)));
		metrics_quoted = text_to_cstring(quoted_metrics);
		NDB_FREE(metrics_text);
		NDB_FREE(quoted_metrics);
	}

	{
		NDB_DECLARE (char *, time_ms_str);
		NDB_DECLARE (char *, num_samples_str);
		NDB_DECLARE (char *, num_features_str);

		time_ms_str = (spec->training_time_ms >= 0) ? psprintf("%d", spec->training_time_ms) : NULL;
		num_samples_str = (spec->num_samples > 0) ? psprintf("%d", spec->num_samples) : NULL;
		num_features_str = (spec->num_features > 0) ? psprintf("%d", spec->num_features) : NULL;

		appendStringInfo(&insert_sql,
						 "INSERT INTO neurondb.ml_models "
						 "(project_id, version, algorithm, status, "
						 "training_table, "
						 "training_column, parameters, model_data, metrics, "
						 "training_time_ms, num_samples, num_features, "
						 "completed_at) "
						 "VALUES (%d, %d, %s::neurondb.ml_algorithm_type, 'completed', "
						 "%s, %s, %s::jsonb, NULL, %s::jsonb, "
						 "%s, %s, %s, NOW()) "
						 "ON CONFLICT (project_id, version) DO UPDATE SET "
						 "algorithm = EXCLUDED.algorithm, "
						 "status = EXCLUDED.status, "
						 "training_table = EXCLUDED.training_table, "
						 "training_column = EXCLUDED.training_column, "
						 "parameters = EXCLUDED.parameters, "
						 "metrics = EXCLUDED.metrics, "
						 "training_time_ms = EXCLUDED.training_time_ms, "
						 "num_samples = EXCLUDED.num_samples, "
						 "num_features = EXCLUDED.num_features, "
						 "completed_at = EXCLUDED.completed_at "
						 "RETURNING model_id",
						 project_id,
						 version,
						 algorithm_quoted,
						 table_quoted,
						 column_quoted != NULL ? column_quoted : "NULL",
						 params_quoted != NULL ? params_quoted : "'{}'",
						 metrics_quoted != NULL ? metrics_quoted : "'{}'",
						 time_ms_str != NULL ? time_ms_str : "NULL",
						 num_samples_str != NULL ? num_samples_str : "NULL",
						 num_features_str != NULL ? num_features_str : "NULL");

		NDB_FREE(time_ms_str);
		NDB_FREE(num_samples_str);
		NDB_FREE(num_features_str);
	}

	elog(DEBUG1, "neurondb: ml_catalog_register_model: inserting model row");
	ret = ndb_spi_execute(spi_session, insert_sql.data, false, 0);

	if ((ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING) ||
		SPI_processed == 0)
	{
		error_code = 6;
		goto error;
	}

	if (!ndb_spi_get_int32(spi_session, 0, 0, &model_id))
	{
		error_code = 7;
		goto error;
	}

	Assert(model_id > 0);
	if (model_id <= 0)
	{
		error_code = 8;
		goto error;
	}

	elog(INFO, "neurondb: ml_catalog_register_model: successfully registered model_id %d for project_id %d, version %d",
		 model_id, project_id, version);

	/* Optionally update model_data using parameterized query */
	if (spec->model_data != NULL && model_id > 0)
	{
		Oid			argtypes[2];
		Datum		values[2];
		char		nulls[2];

		ndb_spi_stringinfo_reset(spi_session, &sql);
		appendStringInfo(&sql,
						 "UPDATE neurondb.ml_models SET model_data = $1 WHERE model_id = $2");

		argtypes[0] = BYTEAOID;
		argtypes[1] = INT4OID;
		values[0] = PointerGetDatum(spec->model_data);
		values[1] = Int32GetDatum(model_id);
		nulls[0] = ' ';
		nulls[1] = ' ';

		elog(DEBUG1, "neurondb: ml_catalog_register_model: updating model_data for model_id %d", model_id);
		ret = ndb_spi_execute_with_args(spi_session, sql.data, 2, argtypes, values, nulls, false, 0);
		if (ret != SPI_OK_UPDATE || SPI_processed != 1)
		{
			elog(WARNING,
				 "neurondb: ml_catalog_register_model: failed to update model_data for model_id %d (ret=%d, processed=%lu)",
				 model_id, ret, (unsigned long) SPI_processed);
		}
		ndb_spi_stringinfo_free(spi_session, &sql);
	}

	ndb_spi_stringinfo_free(spi_session, &insert_sql);
	NDB_FREE(algorithm_quoted);
	NDB_FREE(table_quoted);
	NDB_FREE(column_quoted);
	NDB_FREE(params_txt);
	NDB_FREE(params_quoted);
	NDB_FREE(metrics_txt);
	NDB_FREE(metrics_quoted);

	goto success;

error:
	if (insert_query.data != NULL)
		ndb_spi_stringinfo_free(spi_session, &insert_query);
	if (select_query.data != NULL)
		ndb_spi_stringinfo_free(spi_session, &select_query);
	if (insert_sql.data != NULL)
		ndb_spi_stringinfo_free(spi_session, &insert_sql);
	if (sql.data != NULL)
		ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_FREE(algorithm_quoted);
	NDB_FREE(table_quoted);
	NDB_FREE(column_quoted);
	NDB_FREE(params_txt);
	NDB_FREE(params_quoted);
	NDB_FREE(metrics_txt);
	NDB_FREE(metrics_quoted);
	NDB_SPI_SESSION_END(spi_session);
	NDB_FREE(project_name_copy);
	NDB_FREE(algorithm_copy);
	NDB_FREE(training_table_copy);
	NDB_FREE(training_column_copy);

	switch (error_code)
	{
		case 1:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: failed to insert project into catalog"),
					 errdetail("SPI execution returned error code %d. Project name: '%s', Model type: '%s'", ret, project_name, model_type),
					 errhint("Check database permissions and ensure the neurondb.ml_projects table exists and is accessible.")));
			break;
		case 2:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: failed to retrieve project_id from catalog"),
					 errdetail("SPI execution returned code %d (expected %d for SELECT). Project name: '%s'", ret, SPI_OK_SELECT, project_name),
					 errhint("Check database permissions and ensure the neurondb.ml_projects table exists and contains the project.")));
			break;
		case 3:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: failed to retrieve project_id from catalog"),
					 errdetail("No rows returned or invalid data type. Project name: '%s'", project_name),
					 errhint("Verify the project was successfully inserted and the catalog table structure is correct.")));
			break;
		case 4:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: invalid project_id returned from catalog"),
					 errdetail("Project ID is %d (must be > 0). Project name: '%s'", project_id, project_name),
					 errhint("This indicates a database integrity issue. Check the neurondb.ml_projects table.")));
			break;
		case 5:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: failed to acquire advisory lock for project"),
					 errdetail("SPI execution returned code %d (expected %d for SELECT). Project ID: %d, Project name: '%s'", ret, SPI_OK_SELECT, project_id, project_name),
					 errhint("This may indicate a database locking issue. Retry the operation.")));
			break;
		case 6:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: failed to insert/update model in catalog"),
					 errdetail("SPI execution returned code %d (expected %d or %d), processed %lu rows. Project ID: %d, Algorithm: '%s', Table: '%s'", ret, SPI_OK_INSERT_RETURNING, SPI_OK_UPDATE_RETURNING, (unsigned long) SPI_processed, project_id, algorithm, training_table),
					 errhint("Check database permissions, table constraints, and ensure the neurondb.ml_models table exists and is accessible.")));
			break;
		case 7:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: failed to retrieve model_id after insert/update"),
					 errdetail("No rows returned or invalid data type. Project ID: %d, Algorithm: '%s', Table: '%s'", project_id, algorithm, training_table),
					 errhint("Verify the model was successfully inserted and the catalog table structure is correct.")));
			break;
		case 8:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: invalid model_id returned from catalog"),
					 errdetail("Model ID is %d (must be > 0). Project ID: %d, Algorithm: '%s', Table: '%s'", model_id, project_id, algorithm, training_table),
					 errhint("This indicates a database integrity issue. Check the neurondb.ml_models table.")));
			break;
		default:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: unknown error occurred during model registration"),
					 errdetail("An unexpected error occurred during model registration"),
					 errhint("This is an internal error. Please report this issue.")));
			break;
	}

success:
	NDB_SPI_SESSION_END(spi_session);
	NDB_FREE(project_name_copy);
	NDB_FREE(algorithm_copy);
	NDB_FREE(training_table_copy);
	NDB_FREE(training_column_copy);

	return model_id;
}

/*
 * ml_catalog_fetch_model_payload
 *		Fetch model payload (model_data, parameters, metrics) from catalog.
 *
 * Retrieves serialized model data and metadata for the given model_id.
 * All returned data is allocated in caller's memory context.
 * Returns true if model found, false otherwise.
 */
bool
ml_catalog_fetch_model_payload(int32 model_id,
							   bytea * *model_data_out,
							   Jsonb * *parameters_out,
							   Jsonb * *metrics_out)
{
	int			ret;
	StringInfoData sql = {0};
	bool		found = false;
	MemoryContext caller_context;
	NDB_DECLARE (NdbSpiSession *, spi_session);

	if (model_data_out != NULL)
		*model_data_out = NULL;
	if (parameters_out != NULL)
		*parameters_out = NULL;
	if (metrics_out != NULL)
		*metrics_out = NULL;

	Assert(model_id > 0);
	if (model_id <= 0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid model_id"),
				 errdetail("Model ID must be greater than 0, got %d", model_id),
				 errhint("Provide a valid model_id from the neurondb.ml_models table.")));
	}

	caller_context = CurrentMemoryContext;
	Assert(caller_context != NULL);
	if (caller_context == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: CurrentMemoryContext is NULL"),
				 errdetail("Cannot proceed without a valid memory context"),
				 errhint("This is an internal error. Please report this issue.")));
	}

	NDB_SPI_SESSION_BEGIN(spi_session, caller_context);
	Assert(spi_session != NULL);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT model_data, parameters, metrics "
					 "FROM neurondb.ml_models WHERE model_id = %d",
					 model_id);

	PG_TRY();
	{
		elog(DEBUG1, "neurondb: ml_catalog_fetch_model_payload: fetching model_id %d", model_id);
		ret = ndb_spi_execute(spi_session, sql.data, true, 0);

		if (ret == SPI_OK_SELECT)
		{
			if (SPI_processed > 0)
			{
				found = true;

				elog(INFO, "neurondb: ml_catalog_fetch_model_payload: found model_id %d in catalog", model_id);

				if (model_data_out != NULL)
				{
					NDB_DECLARE (bytea *, copy);

					copy = ndb_spi_get_bytea(spi_session, 0, 1, caller_context);
					if (copy == NULL)
					{
						*model_data_out = NULL;
						elog(WARNING, "neurondb: ml_catalog_fetch_model_payload: model_data is NULL for model_id %d - model may not have been stored correctly", model_id);
					}
					else
					{
						*model_data_out = copy;
						elog(DEBUG2, "neurondb: ml_catalog_fetch_model_payload: loaded model_data for model_id %d (size=%u)",
							 model_id, VARSIZE(copy));
					}
				}

				if (parameters_out != NULL)
				{
					Jsonb	   *jsonb_val;

					jsonb_val = ndb_spi_get_jsonb(spi_session, 0, 2, caller_context);
					if (jsonb_val == NULL)
					{
						*parameters_out = NULL;
						elog(DEBUG1, "neurondb: ml_catalog_fetch_model_payload: parameters is NULL for model_id %d", model_id);
					}
					else
					{
						*parameters_out = jsonb_val;
					}
				}

				if (metrics_out != NULL)
				{
					Jsonb	   *jsonb_val;

					jsonb_val = ndb_spi_get_jsonb(spi_session, 0, 3, caller_context);
					if (jsonb_val == NULL)
					{
						*metrics_out = NULL;
						elog(DEBUG1, "neurondb: ml_catalog_fetch_model_payload: metrics is NULL for model_id %d", model_id);
					}
					else
					{
						*metrics_out = jsonb_val;
					}
				}
			}
			else
			{
				elog(DEBUG1, "neurondb: ml_catalog_fetch_model_payload: model_id %d not found in catalog", model_id);
			}
		}
		else
		{
			elog(WARNING, "neurondb: ml_catalog_fetch_model_payload: SPI execution returned code %d (expected %d) for model_id %d", ret, SPI_OK_SELECT, model_id);
		}

		ndb_spi_stringinfo_free(spi_session, &sql);
	}
	PG_CATCH();
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		PG_RE_THROW();
	}
	PG_END_TRY();

	NDB_SPI_SESSION_END(spi_session);

	return found;
}
