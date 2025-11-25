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
#include "neurondb_spi_safe.h"
#include "neurondb_safe_memory.h"
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
	StringInfoData	sql;
	bool		isnull = false;
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
		elog(ERROR,
			 "ml_catalog_register_model: training_table must be provided");

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

	/* Begin SPI session - handles connection state automatically */
	/* Check if SPI is already connected (e.g., when called from SQL) */
	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Make copies of string pointers AFTER SPI session begin to ensure they're in the correct context.
	 * This avoids use-after-free if they point to memory in a different context.
	 * This is especially important when called from nested SPI contexts or GPU contexts.
	 */
	/* Copy algorithm string */
	if (algorithm != NULL)
	{
		algorithm_copy = pstrdup(algorithm);
		algorithm = algorithm_copy;
	}

	/* Copy training_table string */
	if (training_table != NULL)
	{
		training_table_copy = pstrdup(training_table);
		training_table = training_table_copy;
	}

	/* Copy training_column string if provided */
	if (training_column != NULL)
	{
		training_column_copy = pstrdup(training_column);
		training_column = training_column_copy;
	}

	/* Copy project_name string AFTER SPI session begin to ensure it's in the correct context */
	if (spec->project_name == NULL && project_name != NULL)
	{
		/* project_name was allocated by ml_catalog_default_project in caller's context */
		project_name_copy = pstrdup(project_name);
		/* Free the original allocation - cast away const since we own it */
		ndb_safe_pfree((void *)project_name);
		project_name = project_name_copy;
	}
	else if (spec->project_name != NULL && project_name != NULL)
	{
		/* Make a copy even if it came from spec - ensure it's in current context */
		project_name_copy = pstrdup(project_name);
		project_name = project_name_copy;
	}

	/* Debug: verify metrics are received */
	if (metrics != NULL)
	{
		NDB_DECLARE(char *, meta_txt);
		meta_txt = DatumGetCString(DirectFunctionCall1(jsonb_out,
													   JsonbPGetDatum(metrics)));
		elog(DEBUG1,
			"neurondb: ml_catalog_store_metrics: stored metrics: %s",
			meta_txt);
		NDB_FREE(meta_txt);
	}

	/* Insert or update project entry */
	/* Debug logging */
	elog(DEBUG1, "ml_catalog_register_model: project_name='%s', model_type='%s'", project_name, model_type);

	/* Use separate INSERT and SELECT to avoid SPI issues with RETURNING */
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
		if (ret != SPI_OK_SELECT || SPI_processed == 0 || SPI_tuptable == NULL)
		{
			ndb_spi_stringinfo_free(spi_session, &insert_query);
			ndb_spi_stringinfo_free(spi_session, &select_query);
			NDB_SPI_SESSION_END(spi_session);
			elog(ERROR, "ml_catalog_register_model: SELECT failed with ret=%d, processed=%lu, tuptable=%p",
				 ret, (unsigned long)SPI_processed, SPI_tuptable);
		}

		ndb_spi_stringinfo_free(spi_session, &insert_query);
		ndb_spi_stringinfo_free(spi_session, &select_query);
	}

	if (SPI_processed == 0)
		elog(ERROR, "ml_catalog_register_model: SPI_processed is 0, no rows affected");

	if (SPI_tuptable == NULL)
		elog(ERROR, "ml_catalog_register_model: SPI_tuptable is NULL");

	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0 ||
		SPI_tuptable == NULL)
		elog(ERROR, "ml_catalog_register_model: failed to upsert project");

	project_id = DatumGetInt32(SPI_getbinval(
								SPI_tuptable->vals[0],
								SPI_tuptable->tupdesc, 1, &isnull));
	if (isnull)
		elog(ERROR, "ml_catalog_register_model: project_id NULL after upsert");
	isnull = false;

	/* Determine next model version for project */
	/* Use advisory lock to serialize version calculation, same as neurondb.train */
	/* Note: pg_advisory_xact_lock blocks until lock is acquired, returns void */
	if (project_id <= 0)
		elog(ERROR, "ml_catalog_register_model: invalid project_id %d", project_id);
	
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
	elog(DEBUG1, "ml_catalog_register_model: acquiring advisory lock for project_id=%d with query: %s", project_id, sql.data);
	ret = ndb_spi_execute(spi_session, sql.data, false, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		elog(ERROR,
			"ml_catalog_register_model: failed to acquire advisory lock (ndb_spi_execute returned %d, project_id=%d)",
			ret, project_id);
	}

	/* Now calculate version atomically - reuse sql buffer */
	ndb_spi_stringinfo_reset(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT COALESCE(MAX(version), 0) + 1 "
					 "FROM neurondb.ml_models WHERE project_id = %d",
					 project_id);

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT || SPI_processed == 0 || SPI_tuptable == NULL)
		version = 1;
	else
	{
		if (!ndb_spi_get_int32(spi_session, 0, 0, &version))
			version = 1;
	}

	ndb_spi_stringinfo_free(spi_session, &sql);

	/* Insert model row - construct SQL directly with JSONB text embedded */
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

		/* Serialize JSONB to text for embedding in SQL */
		if (parameters != NULL)
		{
			params_txt = DatumGetCString(
				DirectFunctionCall1(jsonb_out,
					JsonbPGetDatum(parameters)));
			{
				NDB_DECLARE(text *, params_text);
				NDB_DECLARE(text *, quoted_params);
				params_text = cstring_to_text(params_txt);
				quoted_params = DatumGetTextP(
					DirectFunctionCall1(quote_literal,
						PointerGetDatum(params_text)));
				params_quoted = text_to_cstring(quoted_params);
				NDB_FREE(params_text);
				NDB_FREE(quoted_params);
			}
		}

		if (metrics != NULL)
		{
			metrics_txt = DatumGetCString(
				DirectFunctionCall1(jsonb_out,
					JsonbPGetDatum(metrics)));
			{
				NDB_DECLARE(text *, metrics_text);
				NDB_DECLARE(text *, quoted_metrics);
				metrics_text = cstring_to_text(metrics_txt);
				quoted_metrics = DatumGetTextP(
					DirectFunctionCall1(quote_literal,
						PointerGetDatum(metrics_text)));
				metrics_quoted = text_to_cstring(quoted_metrics);
				NDB_FREE(metrics_text);
				NDB_FREE(quoted_metrics);
			}
		}

	/* Construct INSERT SQL with JSONB text embedded */
	/* Use ON CONFLICT to update existing row if it exists */
	/* model_data (bytea) is set to NULL initially and updated separately if provided */
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
			spec->training_time_ms >= 0 ? psprintf("%d", spec->training_time_ms) : "NULL",
			spec->num_samples > 0 ? psprintf("%d", spec->num_samples) : "NULL",
			spec->num_features > 0 ? psprintf("%d", spec->num_features) : "NULL");

		ret = ndb_spi_execute(spi_session, insert_sql.data, false, 0);

		/* ON CONFLICT DO UPDATE returns SPI_OK_UPDATE_RETURNING, not SPI_OK_INSERT_RETURNING */
		if ((ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING)
			|| SPI_processed == 0)
		{
			ndb_spi_stringinfo_free(spi_session, &insert_sql);
			NDB_FREE(algorithm_quoted);
			NDB_FREE(table_quoted);
			NDB_FREE(column_quoted);
			NDB_FREE(params_txt);
			NDB_FREE(params_quoted);
			NDB_FREE(metrics_txt);
			NDB_FREE(metrics_quoted);
			NDB_SPI_SESSION_END(spi_session);
			elog(ERROR,
				"ml_catalog_register_model: failed to insert/update ml_models row "
				"(ret=%d, processed=%lu)",
				 ret, (unsigned long) SPI_processed);
		}

		{
			if (!ndb_spi_get_int32(spi_session, 0, 0, &model_id))
			{
				ndb_spi_stringinfo_free(spi_session, &insert_sql);
				NDB_FREE(algorithm_quoted);
				NDB_FREE(table_quoted);
				NDB_FREE(column_quoted);
				NDB_FREE(params_txt);
				NDB_FREE(params_quoted);
				NDB_FREE(metrics_txt);
				NDB_FREE(metrics_quoted);
				NDB_SPI_SESSION_END(spi_session);
				elog(ERROR,
					 "ml_catalog_register_model: model_id NULL after insert/update");
			}
			if (model_id <= 0)
			{
				ndb_spi_stringinfo_free(spi_session, &insert_sql);
				NDB_FREE(algorithm_quoted);
				NDB_FREE(table_quoted);
				NDB_FREE(column_quoted);
				NDB_FREE(params_txt);
				NDB_FREE(params_quoted);
				NDB_FREE(metrics_txt);
				NDB_FREE(metrics_quoted);
				NDB_SPI_SESSION_END(spi_session);
				elog(ERROR,
					"ml_catalog_register_model: model_id=%d is invalid (should be > 0)",
					model_id);
			}
		}

		if (spec->model_data != NULL && model_id > 0)
		{
			Oid argtypes[2];
			Datum values[2];
			char nulls[2];

			/* Reuse sql buffer for UPDATE */
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

		/* Cleanup */
		ndb_spi_stringinfo_free(spi_session, &insert_sql);
		NDB_FREE(algorithm_quoted);
		NDB_FREE(table_quoted);
		NDB_FREE(column_quoted);
		NDB_FREE(params_txt);
		NDB_FREE(params_quoted);
		NDB_FREE(metrics_txt);
		NDB_FREE(metrics_quoted);
	}

	/* Note: values[] array contains Datums, not pointers to free */
	/* Only free the model_data bytea if it was provided and we're responsible for it */
	/* But spec->model_data is owned by caller, so don't free it here */

	/* End SPI session - handles cleanup automatically */
	NDB_SPI_SESSION_END(spi_session);

	/* Free the copied strings we allocated in this function - use safe operations */
	/* These are allocated in current context, not SPI context, so must be freed */
	NDB_FREE(project_name_copy);
	NDB_FREE(algorithm_copy);
	NDB_FREE(training_table_copy);
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
	int			ret;
	StringInfoData	sql;
	bool		found = false;
	MemoryContext	oldcontext;
	MemoryContext	caller_context;
	NDB_DECLARE(NdbSpiSession *, spi_session);

	if (model_data_out != NULL)
		*model_data_out = NULL;
	if (parameters_out != NULL)
		*parameters_out = NULL;
	if (metrics_out != NULL)
		*metrics_out = NULL;

	/* Save caller's memory context */
	caller_context = CurrentMemoryContext;

	/* Begin SPI session */
	NDB_SPI_SESSION_BEGIN(spi_session, caller_context);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT model_data, parameters, metrics "
					 "FROM neurondb.ml_models WHERE model_id = %d",
					 model_id);

	/* Use read-only query - no explicit snapshot needed */
	/* Wrap in PG_TRY to ensure SPI session is ended even on error */
	PG_TRY();
	{
		ret = ndb_spi_execute(spi_session, sql.data, true, 0);
		
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			Datum		datum;
			bool		isnull;

			found = true;
			
			elog(DEBUG1, "ml_catalog_fetch_model_payload: found model_id %d in catalog", model_id);

			/* Switch to caller's context for detoasted copies */
			oldcontext = MemoryContextSwitchTo(caller_context);

			if (model_data_out != NULL)
			{
				Datum		datum;
				bool		isnull;
				NDB_DECLARE(bytea *, copy);
				NDB_DECLARE(bytea *, temp_copy);

				datum = SPI_getbinval(SPI_tuptable->vals[0],
									  SPI_tuptable->tupdesc, 1, &isnull);
				if (!isnull)
				{
					temp_copy = DatumGetByteaP(datum);
					if (temp_copy == NULL)
					{
						*model_data_out = NULL;
						elog(WARNING, "ml_catalog_fetch_model_payload: DatumGetByteaP returned NULL for model_id %d", model_id);
					}
					else
					{
						copy = (bytea *) PG_DETOAST_DATUM_COPY((Datum) temp_copy);
						if (copy == NULL)
						{
							*model_data_out = NULL;
							elog(WARNING, "ml_catalog_fetch_model_payload: PG_DETOAST_DATUM_COPY returned NULL for model_id %d", model_id);
						}
						else
						{
							*model_data_out = copy;
							elog(DEBUG1, "ml_catalog_fetch_model_payload: loaded model_data for model_id %d (size=%u)",
								 model_id, VARSIZE(copy));
						}
					}
				}
				else
				{
					*model_data_out = NULL;
					elog(WARNING, "ml_catalog_fetch_model_payload: model_data is NULL for model_id %d - model may not have been stored correctly", model_id);
				}
			}

			if (parameters_out != NULL)
			{
				Datum		datum;
				bool		isnull;
				Jsonb	   *jsonb_val;

				jsonb_val = ndb_spi_get_jsonb(spi_session, 0, 1, caller_context);
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
				Jsonb	   *jsonb_val;

				jsonb_val = ndb_spi_get_jsonb(spi_session, 0, 2, caller_context);
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

			MemoryContextSwitchTo(oldcontext);
		}
		else
		{
			/* Unexpected state */
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

	/* End SPI session - handles cleanup automatically */
	NDB_SPI_SESSION_END(spi_session);

	return found;
}
