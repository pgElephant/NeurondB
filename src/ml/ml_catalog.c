/*-------------------------------------------------------------------------
 *
 * ml_catalog.c
 *	  Helpers for interacting with neurondb.ml_models catalog.
 *
 * These utilities provide a reusable way for algorithm implementations to
 * register their trained models and retrieve serialized payloads.
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

#include "ml_catalog.h"
#include "neurondb_validation.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

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
 * Register a new model in neurondb.ml_models.
 * Returns the model_id of the inserted/updated row.
 */
int32
ml_catalog_register_model(const MLCatalogModelSpec *spec)
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
	StringInfoData sql;
	MemoryContext	oldcontext;
	NDB_DECLARE(NdbSpiSession *, spi_session);
	NDB_DECLARE(char *, algorithm_copy);
	NDB_DECLARE(char *, training_table_copy);
	NDB_DECLARE(char *, training_column_copy);
	NDB_DECLARE(char *, project_name_copy);

	if (spec == NULL)
		elog(ERROR, "ml_catalog_register_model: invalid spec (NULL)");

	algorithm = spec->algorithm;
	if (algorithm == NULL)
		elog(ERROR, "ml_catalog_register_model: algorithm must be provided");

	training_table = spec->training_table;
	if (training_table == NULL)
		elog(ERROR, "ml_catalog_register_model: training_table must be provided");

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

	/* Save current memory context */
	oldcontext = CurrentMemoryContext;

	/* Copy strings in parent context BEFORE SPI session begin */
	/* This ensures they're freed in the same context after SPI session ends */
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
			char *temp = (char *) project_name;
			NDB_FREE(temp);
		}
		project_name = project_name_copy;
	}
	else if (spec->project_name != NULL && project_name != NULL)
	{
		project_name_copy = pstrdup(project_name);
		project_name = project_name_copy;
	}

	/* Begin SPI session AFTER string copies are in parent context */
	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Debug: log metrics JSON if present */
	if (metrics != NULL)
	{
		NDB_DECLARE(char *, meta_txt);

		meta_txt = DatumGetCString(
					DirectFunctionCall1(jsonb_out, JsonbPGetDatum(metrics)));
		elog(DEBUG1, "neurondb: ml_catalog_store_metrics: stored metrics: %s", meta_txt);
		NDB_FREE(meta_txt);
	}

	elog(DEBUG1, "ml_catalog_register_model: project_name='%s', model_type='%s'",
		 project_name, model_type);

	/* Insert or update project entry, get project_id back */
	{
		StringInfoData insert_query;
		StringInfoData select_query;

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

		elog(DEBUG1, "ml_catalog_register_model: executing insert: %s", insert_query.data);
		ret = ndb_spi_execute(spi_session, insert_query.data, false, 0);
		if (ret < 0)
		{
			ndb_spi_stringinfo_free(spi_session, &insert_query);
			ndb_spi_stringinfo_free(spi_session, &select_query);
			NDB_SPI_SESSION_END(spi_session);
			elog(ERROR, "ml_catalog_register_model: INSERT failed with ret=%d", ret);
		}

		elog(DEBUG1, "ml_catalog_register_model: executing select: %s", select_query.data);
		ret = ndb_spi_execute(spi_session, select_query.data, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			ndb_spi_stringinfo_free(spi_session, &insert_query);
			ndb_spi_stringinfo_free(spi_session, &select_query);
			NDB_SPI_SESSION_END(spi_session);
			elog(ERROR, "ml_catalog_register_model: SELECT failed with ret=%d", ret);
		}

		ndb_spi_stringinfo_free(spi_session, &insert_query);
		ndb_spi_stringinfo_free(spi_session, &select_query);
	}

	if (!ndb_spi_get_int32(spi_session, 0, 1, &project_id))
		elog(ERROR, "ml_catalog_register_model: failed to get project_id or no rows returned after project upsert");

	if (project_id <= 0)
		elog(ERROR, "ml_catalog_register_model: invalid project_id %d", project_id);

	/* Lock for versioning */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT pg_advisory_xact_lock(%d)", project_id);
	elog(DEBUG1, "ml_catalog_register_model: acquiring advisory lock for project_id=%d with query: %s",
		 project_id, sql.data);

	ret = ndb_spi_execute(spi_session, sql.data, false, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		elog(ERROR,
			 "ml_catalog_register_model: failed to acquire advisory lock (ndb_spi_execute returned %d, project_id=%d)",
			 ret, project_id);
	}

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

	/* Insert model row */
	{
		StringInfoData insert_sql;
		NDB_DECLARE(char *, params_txt);
		NDB_DECLARE(char *, metrics_txt);
		NDB_DECLARE(char *, params_quoted);
		NDB_DECLARE(char *, metrics_quoted);
		NDB_DECLARE(char *, algorithm_quoted);
		NDB_DECLARE(char *, table_quoted);
		NDB_DECLARE(char *, column_quoted);

		ndb_spi_stringinfo_init(spi_session, &insert_sql);

		/* Quote string literals */
		{
			NDB_DECLARE(text *, algorithm_text);
			NDB_DECLARE(text *, quoted_algorithm);

			algorithm_text = cstring_to_text(algorithm);
			quoted_algorithm = DatumGetTextP(
				DirectFunctionCall1(quote_literal,
									PointerGetDatum(algorithm_text)));
			algorithm_quoted = text_to_cstring(quoted_algorithm);
			NDB_FREE(algorithm_text);
			NDB_FREE(quoted_algorithm);
		}
		{
			NDB_DECLARE(text *, table_text);
			NDB_DECLARE(text *, quoted_table);

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
			NDB_DECLARE(text *, column_text);
			NDB_DECLARE(text *, quoted_column);

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
			NDB_DECLARE(text *, params_text);
			NDB_DECLARE(text *, quoted_params);

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
			NDB_DECLARE(text *, metrics_text);
			NDB_DECLARE(text *, quoted_metrics);

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
			NDB_DECLARE(char *, time_ms_str);
			NDB_DECLARE(char *, num_samples_str);
			NDB_DECLARE(char *, num_features_str);

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

			/* Free temporary strings allocated in SPI context */
			NDB_FREE(time_ms_str);
			NDB_FREE(num_samples_str);
			NDB_FREE(num_features_str);
		}

		ret = ndb_spi_execute(spi_session, insert_sql.data, false, 0);

		if ((ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING) ||
			SPI_processed == 0)
		{
			ndb_spi_stringinfo_free(spi_session, &insert_sql);
			NDB_FREE(algorithm_quoted);
			NDB_FREE(table_quoted);
			if (column_quoted)
				NDB_FREE(column_quoted);
			if (params_txt)
				NDB_FREE(params_txt);
			if (params_quoted)
				NDB_FREE(params_quoted);
			if (metrics_txt)
				NDB_FREE(metrics_txt);
			if (metrics_quoted)
				NDB_FREE(metrics_quoted);
			NDB_SPI_SESSION_END(spi_session);
			elog(ERROR,
				 "ml_catalog_register_model: failed to insert/update ml_models row "
			     "(ret=%d, processed=%lu)", ret, (unsigned long)SPI_processed);
		}

		if (!ndb_spi_get_int32(spi_session, 0, 0, &model_id))
		{
			ndb_spi_stringinfo_free(spi_session, &insert_sql);
			NDB_FREE(algorithm_quoted);
			NDB_FREE(table_quoted);
			if (column_quoted)
				NDB_FREE(column_quoted);
			if (params_txt)
				NDB_FREE(params_txt);
			if (params_quoted)
				NDB_FREE(params_quoted);
			if (metrics_txt)
				NDB_FREE(metrics_txt);
			if (metrics_quoted)
				NDB_FREE(metrics_quoted);
			NDB_SPI_SESSION_END(spi_session);
			elog(ERROR, "ml_catalog_register_model: model_id NULL after insert/update");
		}
		if (model_id <= 0)
		{
			ndb_spi_stringinfo_free(spi_session, &insert_sql);
			NDB_FREE(algorithm_quoted);
			NDB_FREE(table_quoted);
			if (column_quoted)
				NDB_FREE(column_quoted);
			if (params_txt)
				NDB_FREE(params_txt);
			if (params_quoted)
				NDB_FREE(params_quoted);
			if (metrics_txt)
				NDB_FREE(metrics_txt);
			if (metrics_quoted)
				NDB_FREE(metrics_quoted);
			NDB_SPI_SESSION_END(spi_session);
			elog(ERROR,
				 "ml_catalog_register_model: model_id=%d is invalid (should be > 0)",
				 model_id);
		}

		if (spec->model_data != NULL && model_id > 0)
		{
			Oid		argtypes[2];
			Datum	values[2];
			char	nulls[2];

			ndb_spi_stringinfo_reset(spi_session, &sql);
			appendStringInfo(&sql,
							 "UPDATE neurondb.ml_models SET model_data = $1 WHERE model_id = $2");

			argtypes[0] = BYTEAOID;
			argtypes[1] = INT4OID;
			values[0] = PointerGetDatum(spec->model_data);
			values[1] = Int32GetDatum(model_id);
			nulls[0] = ' ';
			nulls[1] = ' ';

			ret = ndb_spi_execute_with_args(spi_session, sql.data, 2, argtypes, values, nulls, false, 0);
			if (ret != SPI_OK_UPDATE || SPI_processed != 1)
			{
				elog(WARNING,
					 "ml_catalog_register_model: failed to update model_data (ret=%d, processed=%lu)",
					 ret, (unsigned long)SPI_processed);
			}
			ndb_spi_stringinfo_free(spi_session, &sql);
		}

		ndb_spi_stringinfo_free(spi_session, &insert_sql);
		NDB_FREE(algorithm_quoted);
		NDB_FREE(table_quoted);
		if (column_quoted)
			NDB_FREE(column_quoted);
		if (params_txt)
			NDB_FREE(params_txt);
		if (params_quoted)
			NDB_FREE(params_quoted);
		if (metrics_txt)
			NDB_FREE(metrics_txt);
		if (metrics_quoted)
			NDB_FREE(metrics_quoted);
	}

	NDB_SPI_SESSION_END(spi_session);

	/* Free persistent strings. These are allocated in the current context. */
	if (project_name_copy)
		NDB_FREE(project_name_copy);
	if (algorithm_copy)
		NDB_FREE(algorithm_copy);
	if (training_table_copy)
		NDB_FREE(training_table_copy);
	if (training_column_copy)
		NDB_FREE(training_column_copy);

	return model_id;
}

/*
 * Fetch model payload from catalog.
 * Returns palloc'd detoasted copies in caller's memory context.
 * Caller is responsible for pfree'ing *model_data_out, *parameters_out, and *metrics_out if non-NULL.
 */
bool
ml_catalog_fetch_model_payload(int32 model_id,
							  bytea **model_data_out,
							  Jsonb **parameters_out,
							  Jsonb **metrics_out)
{
	int				ret;
	StringInfoData	sql;
	bool			found = false;
	MemoryContext	caller_context;
	NDB_DECLARE(NdbSpiSession *, spi_session);

	if (model_data_out != NULL)
		*model_data_out = NULL;
	if (parameters_out != NULL)
		*parameters_out = NULL;
	if (metrics_out != NULL)
		*metrics_out = NULL;

	caller_context = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, caller_context);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT model_data, parameters, metrics "
					 "FROM neurondb.ml_models WHERE model_id = %d",
					 model_id);

	PG_TRY();
	{
		ret = ndb_spi_execute(spi_session, sql.data, true, 0);

		if (ret == SPI_OK_SELECT)
		{
			if (SPI_processed > 0)
			{
				found = true;

				elog(DEBUG1, "ml_catalog_fetch_model_payload: found model_id %d in catalog", model_id);

				if (model_data_out != NULL)
				{
					NDB_DECLARE(bytea *, copy);

					copy = ndb_spi_get_bytea(spi_session, 0, 1, caller_context);
					if (copy == NULL)
					{
						*model_data_out = NULL;
						elog(WARNING, "ml_catalog_fetch_model_payload: model_data is NULL for model_id %d - model may not have been stored correctly", model_id);
					}
					else
					{
						*model_data_out = copy;
						elog(DEBUG1, "ml_catalog_fetch_model_payload: loaded model_data for model_id %d (size=%u)",
							 model_id, VARSIZE(copy));
					}
				}

				if (parameters_out != NULL)
				{
					Jsonb *jsonb_val;

					jsonb_val = ndb_spi_get_jsonb(spi_session, 0, 2, caller_context);
					if (jsonb_val == NULL)
					{
						*parameters_out = NULL;
						elog(WARNING, "ml_catalog_fetch_model_payload: failed to get parameters (model_id %d)", model_id);
					}
					else
					{
						*parameters_out = jsonb_val;
					}
				}

				if (metrics_out != NULL)
				{
					Jsonb *jsonb_val;

					jsonb_val = ndb_spi_get_jsonb(spi_session, 0, 3, caller_context);
					if (jsonb_val == NULL)
					{
						*metrics_out = NULL;
						elog(WARNING, "ml_catalog_fetch_model_payload: failed to get metrics (model_id %d)", model_id);
					}
					else
					{
						*metrics_out = jsonb_val;
					}
				}
			}
		}
		else
		{
			found = false;
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
