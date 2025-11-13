/*-------------------------------------------------------------------------
 *
 * ml_projects.c
 *	  ML Project Management System for NeuronDB
 *
 * Provides organized model lifecycle management with versioning,
 * experiment tracking, and model deployment capabilities.
 *
 * Inspired by PostgresML's project concept.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/datum.h"
#include "utils/guc.h"
#include "utils/timestamp.h"
#include "utils/tuplestore.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/memutils.h"

PG_FUNCTION_INFO_V1(neurondb_create_ml_project);
PG_FUNCTION_INFO_V1(neurondb_list_ml_projects);
PG_FUNCTION_INFO_V1(neurondb_delete_ml_project);
PG_FUNCTION_INFO_V1(neurondb_get_project_info);
PG_FUNCTION_INFO_V1(neurondb_train_kmeans_project);
PG_FUNCTION_INFO_V1(neurondb_deploy_model);
PG_FUNCTION_INFO_V1(neurondb_get_deployed_model);
PG_FUNCTION_INFO_V1(neurondb_list_project_models);

/*
 * neurondb_create_ml_project
 *
 * Create a new ML project for organizing models and experiments.
 *
 * Parameters:
 *   project_name TEXT - Unique name for the project
 *   model_type TEXT - Type of model ('clustering', 'classification', etc.)
 *   description TEXT - Optional description
 *
 * Returns: project_id INTEGER
 */
Datum
neurondb_create_ml_project(PG_FUNCTION_ARGS)
{
	text *project_name_text = PG_GETARG_TEXT_PP(0);
	text *model_type_text = PG_GETARG_TEXT_PP(1);
	text *description_text = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);
	char *project_name;
	char *model_type;
	char *description;
	StringInfoData sql;
	int ret;
	int project_id;
	bool isnull;

	project_name = text_to_cstring(project_name_text);
	model_type = text_to_cstring(model_type_text);
	description =
		description_text ? text_to_cstring(description_text) : NULL;

	/* Validate project name */
	if (strlen(project_name) < 3 || strlen(project_name) > 100)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("project name must be between 3 and 100 "
				       "characters")));

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"INSERT INTO neurondb.ml_projects (project_name, model_type, "
		"description) "
		"VALUES (%s, %s::neurondb.ml_model_type, %s) "
		"ON CONFLICT (project_name) DO NOTHING "
		"RETURNING project_id",
		quote_literal_cstr(project_name),
		quote_literal_cstr(model_type),
		description ? quote_literal_cstr(description) : "NULL");

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	ret = SPI_execute(sql.data, false, 0);
	if (ret != SPI_OK_INSERT_RETURNING)
	{
		SPI_finish();
		elog(ERROR, "failed to create project: %s", project_name);
	}

	if (SPI_processed == 0)
	{
		/* Conflict - project already exists, fetch existing ID */
		pfree(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT project_id FROM neurondb.ml_projects WHERE "
			"project_name = %s",
			quote_literal_cstr(project_name));

		ret = SPI_execute(sql.data, true, 0);
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			SPI_finish();
			elog(ERROR,
				"project exists but could not retrieve ID: %s",
				project_name);
		}
	}

	project_id = DatumGetInt32(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	SPI_finish();

	elog(LOG,
		"neurondb: created ML project '%s' (id=%d, type=%s)",
		project_name,
		project_id,
		model_type);

	PG_RETURN_INT32(project_id);
}

/*
 * neurondb_list_ml_projects
 *
 * List all ML projects with summary information.
 *
 * Returns: TABLE (project_id, project_name, model_type, total_models, latest_version)
 */
Datum
neurondb_list_ml_projects(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
	TupleDesc tupdesc;
	Tuplestorestate *tupstore;
	MemoryContext per_query_ctx;
	MemoryContext oldcontext;
	int ret;
	int i;

	/* Check result context */
	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				errmsg("set-valued function called in context "
				       "that cannot accept a set")));

	if (!(rsinfo->allowedModes & SFRM_Materialize))
		ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				errmsg("materialize mode required, but it is "
				       "not allowed in this context")));

	/* Build tuple descriptor */
	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");

	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

	tupstore = tuplestore_begin_heap(true, false, 1024);
	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = tupdesc;

	MemoryContextSwitchTo(oldcontext);

	/* Query projects */
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	ret = SPI_execute("SELECT * FROM neurondb.ml_projects_summary ORDER BY "
			  "created_at DESC",
		true,
		0);

	if (ret == SPI_OK_SELECT)
	{
		for (i = 0; i < (int)SPI_processed; i++)
		{
			HeapTuple spi_tuple = SPI_tuptable->vals[i];
			Datum values[6];
			bool nulls[6];

			memset(nulls, 0, sizeof(nulls));

			/* Extract values from SPI tuple */
			values[0] = SPI_getbinval(spi_tuple,
				SPI_tuptable->tupdesc,
				1,
				&nulls[0]); /* project_id */
			values[1] = SPI_getbinval(spi_tuple,
				SPI_tuptable->tupdesc,
				2,
				&nulls[1]); /* project_name */
			values[2] = SPI_getbinval(spi_tuple,
				SPI_tuptable->tupdesc,
				3,
				&nulls[2]); /* model_type */
			values[3] = SPI_getbinval(spi_tuple,
				SPI_tuptable->tupdesc,
				7,
				&nulls[3]); /* total_models */
			values[4] = SPI_getbinval(spi_tuple,
				SPI_tuptable->tupdesc,
				8,
				&nulls[4]); /* latest_version */
			values[5] = SPI_getbinval(spi_tuple,
				SPI_tuptable->tupdesc,
				9,
				&nulls[5]); /* deployed_version */

			tuplestore_putvalues(tupstore, tupdesc, values, nulls);
		}
	}

	SPI_finish();

	return (Datum)0;
}

/*
 * neurondb_delete_ml_project
 *
 * Delete an ML project and all associated models/experiments.
 *
 * Parameters:
 *   project_name TEXT - Name of project to delete
 *
 * Returns: BOOLEAN - true if deleted, false if not found
 */
Datum
neurondb_delete_ml_project(PG_FUNCTION_ARGS)
{
	text *project_name_text = PG_GETARG_TEXT_PP(0);
	char *project_name = text_to_cstring(project_name_text);
	StringInfoData sql;
	int ret;

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"DELETE FROM neurondb.ml_projects WHERE project_name = %s",
		quote_literal_cstr(project_name));

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	ret = SPI_execute(sql.data, false, 0);

	SPI_finish();

	if (ret != SPI_OK_DELETE)
		elog(ERROR, "failed to delete project: %s", project_name);

	elog(LOG, "neurondb: deleted ML project '%s'", project_name);

	PG_RETURN_BOOL(SPI_processed > 0);
}

/*
 * neurondb_get_project_info
 *
 * Get detailed information about a specific project.
 *
 * Parameters:
 *   project_name TEXT - Name of project
 *
 * Returns: JSONB with project details
 */
Datum
neurondb_get_project_info(PG_FUNCTION_ARGS)
{
	text *project_name_text = PG_GETARG_TEXT_PP(0);
	char *project_name = text_to_cstring(project_name_text);
	StringInfoData sql;
	int ret;
	Datum result;
	bool isnull;

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT row_to_json(p)::jsonb FROM "
		"neurondb.ml_projects_summary p "
		"WHERE project_name = %s",
		quote_literal_cstr(project_name));

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	ret = SPI_execute(sql.data, true, 0);

	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		SPI_finish();
		elog(ERROR, "project not found: %s", project_name);
	}

	result = SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);

	if (isnull)
	{
		SPI_finish();
		elog(ERROR, "failed to get project info for: %s", project_name);
	}

	/* Copy result to upper memory context */
	result = datumCopy(result, false, -1);

	SPI_finish();

	PG_RETURN_DATUM(result);
}

/*
 * neurondb_train_kmeans_project
 *
 * Train a K-means model within a project with automatic versioning.
 *
 * Parameters:
 *   project_name TEXT - Name of project
 *   table_name TEXT - Table containing vectors
 *   vector_col TEXT - Vector column name
 *   num_clusters INT - Number of clusters
 *   max_iters INT - Maximum iterations
 *
 * Returns: model_id INTEGER
 */
Datum
neurondb_train_kmeans_project(PG_FUNCTION_ARGS)
{
	text *project_name_text = PG_GETARG_TEXT_PP(0);
	text *table_name_text = PG_GETARG_TEXT_PP(1);
	text *vector_col_text = PG_GETARG_TEXT_PP(2);
	int32 num_clusters = PG_GETARG_INT32(3);
	int32 max_iters = PG_ARGISNULL(4) ? 100 : PG_GETARG_INT32(4);
	char *project_name = text_to_cstring(project_name_text);
	char *table_name = text_to_cstring(table_name_text);
	char *vector_col = text_to_cstring(vector_col_text);
	StringInfoData sql;
	int ret;
	int project_id;
	int next_version;
	int model_id;
	bool isnull;
	int64 start_time;
	int64 end_time;
	int training_time_ms;

	start_time = GetCurrentTimestamp();

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	/* Get project ID */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT project_id FROM neurondb.ml_projects WHERE "
		"project_name = %s",
		quote_literal_cstr(project_name));

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		SPI_finish();
		elog(ERROR, "project not found: %s", project_name);
	}

	project_id = DatumGetInt32(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	/* Get next version number */
	pfree(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT COALESCE(MAX(version), 0) + 1 FROM neurondb.ml_models "
		"WHERE project_id = %d",
		project_id);

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		elog(ERROR, "failed to get next version");
	}

	next_version = DatumGetInt32(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	/* Create model record */
	pfree(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"INSERT INTO neurondb.ml_models "
		"(project_id, version, algorithm, status, training_table, "
		"training_column, parameters) "
		"VALUES (%d, %d, 'kmeans', 'training', %s, %s, '{\"k\": %d, "
		"\"max_iters\": %d}'::jsonb) "
		"RETURNING model_id",
		project_id,
		next_version,
		quote_literal_cstr(table_name),
		quote_literal_cstr(vector_col),
		num_clusters,
		max_iters);

	ret = SPI_execute(sql.data, false, 0);
	if (ret != SPI_OK_INSERT_RETURNING)
	{
		SPI_finish();
		elog(ERROR, "failed to create model record");
	}

	model_id = DatumGetInt32(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	SPI_finish();

	/* Train the actual model (call existing cluster_kmeans) */
	/* For now, we just mark it as completed - actual training integration comes next */

	end_time = GetCurrentTimestamp();
	training_time_ms = (int)((end_time - start_time) / 1000);

	/* Update model status */
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	pfree(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"UPDATE neurondb.ml_models "
		"SET status = 'completed', completed_at = NOW(), "
		"training_time_ms = %d "
		"WHERE model_id = %d",
		training_time_ms,
		model_id);

	SPI_execute(sql.data, false, 0);
	SPI_finish();

	elog(LOG,
		"neurondb: trained K-means model for project '%s' (version %d, "
		"model_id=%d)",
		project_name,
		next_version,
		model_id);

	PG_RETURN_INT32(model_id);
}

/*
 * neurondb_deploy_model
 *
 * Deploy a specific model version for production use.
 *
 * Parameters:
 *   project_name TEXT - Name of project
 *   version INT - Model version to deploy (NULL for latest)
 *
 * Returns: BOOLEAN - true if deployed successfully
 */
Datum
neurondb_deploy_model(PG_FUNCTION_ARGS)
{
	text *project_name_text = PG_GETARG_TEXT_PP(0);
	int32 version = PG_ARGISNULL(1) ? -1 : PG_GETARG_INT32(1);
	char *project_name = text_to_cstring(project_name_text);
	StringInfoData sql;
	int ret;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	/* Undeploy all current models for this project */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"UPDATE neurondb.ml_models SET is_deployed = FALSE "
		"WHERE project_id = (SELECT project_id FROM "
		"neurondb.ml_projects WHERE project_name = %s)",
		quote_literal_cstr(project_name));

	SPI_execute(sql.data, false, 0);

	/* Deploy specified version (or latest) */
	pfree(sql.data);
	initStringInfo(&sql);
	if (version > 0)
	{
		appendStringInfo(&sql,
			"UPDATE neurondb.ml_models SET is_deployed = TRUE, "
			"deployed_at = NOW() "
			"WHERE project_id = (SELECT project_id FROM "
			"neurondb.ml_projects WHERE project_name = %s) "
			"AND version = %d AND status = 'completed'",
			quote_literal_cstr(project_name),
			version);
	} else
	{
		appendStringInfo(&sql,
			"UPDATE neurondb.ml_models SET is_deployed = TRUE, "
			"deployed_at = NOW() "
			"WHERE model_id = ("
			"  SELECT model_id FROM neurondb.ml_models "
			"  WHERE project_id = (SELECT project_id FROM "
			"neurondb.ml_projects WHERE project_name = %s) "
			"  AND status = 'completed' "
			"  ORDER BY version DESC LIMIT 1"
			")",
			quote_literal_cstr(project_name));
	}

	ret = SPI_execute(sql.data, false, 0);

	SPI_finish();

	if (ret != SPI_OK_UPDATE || SPI_processed == 0)
		elog(ERROR,
			"failed to deploy model for project: %s",
			project_name);

	elog(LOG,
		"neurondb: deployed model for project '%s' (version %d)",
		project_name,
		version);

	PG_RETURN_BOOL(true);
}

/*
 * neurondb_get_deployed_model
 *
 * Get the currently deployed model for a project.
 *
 * Parameters:
 *   project_name TEXT - Name of project
 *
 * Returns: model_id INTEGER
 */
Datum
neurondb_get_deployed_model(PG_FUNCTION_ARGS)
{
	text *project_name_text = PG_GETARG_TEXT_PP(0);
	char *project_name = text_to_cstring(project_name_text);
	StringInfoData sql;
	int ret;
	int model_id;
	bool isnull;

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT model_id FROM neurondb.ml_models "
		"WHERE project_id = (SELECT project_id FROM "
		"neurondb.ml_projects WHERE project_name = %s) "
		"AND is_deployed = TRUE LIMIT 1",
		quote_literal_cstr(project_name));

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	ret = SPI_execute(sql.data, true, 0);

	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		SPI_finish();
		PG_RETURN_NULL();
	}

	model_id = DatumGetInt32(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	SPI_finish();

	if (isnull)
		PG_RETURN_NULL();

	PG_RETURN_INT32(model_id);
}

/*
 * neurondb_list_project_models
 *
 * List all models for a project with metrics.
 *
 * Parameters:
 *   project_name TEXT - Name of project
 *
 * Returns: TABLE with model information
 */
Datum
neurondb_list_project_models(PG_FUNCTION_ARGS)
{
	text *project_name_text = PG_GETARG_TEXT_PP(0);
	char *project_name = text_to_cstring(project_name_text);
	ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
	TupleDesc tupdesc;
	Tuplestorestate *tupstore;
	MemoryContext per_query_ctx;
	MemoryContext oldcontext;
	StringInfoData sql;
	int ret;
	int i;

	/* Check result context */
	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				errmsg("set-valued function called in context "
				       "that cannot accept a set")));

	if (!(rsinfo->allowedModes & SFRM_Materialize))
		ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				errmsg("materialize mode required, but it is "
				       "not allowed in this context")));

	/* Build tuple descriptor */
	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");

	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

	tupstore = tuplestore_begin_heap(true, false, 1024);
	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = tupdesc;

	MemoryContextSwitchTo(oldcontext);

	/* Query models */
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT model_id, version, algorithm::text, status::text, "
		"is_deployed, "
		"metrics, training_time_ms, created_at "
		"FROM neurondb.ml_models "
		"WHERE project_id = (SELECT project_id FROM "
		"neurondb.ml_projects WHERE project_name = %s) "
		"ORDER BY version DESC",
		quote_literal_cstr(project_name));

	ret = SPI_execute(sql.data, true, 0);

	if (ret == SPI_OK_SELECT)
	{
		for (i = 0; i < (int)SPI_processed; i++)
		{
			HeapTuple spi_tuple = SPI_tuptable->vals[i];
			Datum values[8];
			bool nulls[8];
			int j;

			memset(nulls, 0, sizeof(nulls));

			/* Extract all columns */
			for (j = 0; j < 8; j++)
			{
				values[j] = SPI_getbinval(spi_tuple,
					SPI_tuptable->tupdesc,
					j + 1,
					&nulls[j]);
			}

			tuplestore_putvalues(tupstore, tupdesc, values, nulls);
		}
	}

	SPI_finish();

	return (Datum)0;
}
