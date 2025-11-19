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
	Oid		argtypes[3];
	Datum		values[3];
	char		nulls[3];
	bool		isnull = false;

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

	/* Debug: verify metrics are received */
	if (metrics != NULL)
	{
		char   *meta_txt;

		meta_txt = DatumGetCString(DirectFunctionCall1(jsonb_out,
													   JsonbPGetDatum(metrics)));
		elog(DEBUG1,
			"neurondb: ml_catalog_store_metrics: stored metrics: %s",
			meta_txt);
		pfree(meta_txt);
	}
	else
	{
	}

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "ml_catalog_register_model: SPI_connect failed");

	/* Insert or update project entry */
	memset(nulls, ' ', sizeof(nulls));
	argtypes[0] = TEXTOID;
	argtypes[1] = TEXTOID;
	argtypes[2] = TEXTOID;

	values[0] = CStringGetTextDatum(project_name);
	values[1] = CStringGetTextDatum(model_type);
	values[2] = CStringGetTextDatum(
		"Auto-created by ml_catalog_register_model");

	ret = SPI_execute_with_args(
		"INSERT INTO neurondb.ml_projects "
		" (project_name, model_type, description) "
		"VALUES ($1, $2::neurondb.ml_model_type, $3) "
		"ON CONFLICT (project_name) DO UPDATE "
		"  SET updated_at = NOW() "
		"RETURNING project_id",
		3,
		argtypes,
		values,
		nulls,
		false,
		1);

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
	
	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
	elog(DEBUG1, "ml_catalog_register_model: acquiring advisory lock for project_id=%d with query: %s", project_id, sql.data);
	ret = SPI_execute(sql.data, false, 0);
	if (ret != SPI_OK_SELECT)
	{
		pfree(sql.data);
		SPI_finish();
		elog(ERROR,
			"ml_catalog_register_model: failed to acquire advisory lock (SPI_execute returned %d, project_id=%d)",
			ret, project_id);
	}

	/* Now calculate version atomically - reuse sql buffer */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT COALESCE(MAX(version), 0) + 1 "
					 "FROM neurondb.ml_models WHERE project_id = %d",
					 project_id);

	ret = SPI_execute(sql.data, true, 1);
	if (ret != SPI_OK_SELECT || SPI_processed == 0 || SPI_tuptable == NULL)
		version = 1;
	else
	{
		version = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
											  SPI_tuptable->tupdesc, 1,
											  &isnull));
		if (isnull)
			version = 1;
	}

	pfree(sql.data);

	/* Insert model row - construct SQL directly with JSONB text embedded */
	{
		StringInfoData insert_sql;
		char *params_txt = NULL;
		char *metrics_txt = NULL;
		char *params_quoted = NULL;
		char *metrics_quoted = NULL;
		char *algorithm_quoted;
		char *table_quoted;
		char *column_quoted = NULL;

		initStringInfo(&insert_sql);

		/* Quote string literals */
		{
			text *algorithm_text = cstring_to_text(algorithm);
			text *quoted_algorithm = DatumGetTextP(
				DirectFunctionCall1(quote_literal,
					PointerGetDatum(algorithm_text)));
			algorithm_quoted = text_to_cstring(quoted_algorithm);
			pfree(algorithm_text);
			pfree(quoted_algorithm);
		}
		{
			text *table_text = cstring_to_text(training_table);
			text *quoted_table = DatumGetTextP(
				DirectFunctionCall1(quote_literal,
					PointerGetDatum(table_text)));
			table_quoted = text_to_cstring(quoted_table);
			pfree(table_text);
			pfree(quoted_table);
		}
		if (training_column != NULL)
		{
			text *column_text = cstring_to_text(training_column);
			text *quoted_column = DatumGetTextP(
				DirectFunctionCall1(quote_literal,
					PointerGetDatum(column_text)));
			column_quoted = text_to_cstring(quoted_column);
			pfree(column_text);
			pfree(quoted_column);
		}

		/* Serialize JSONB to text for embedding in SQL */
		if (parameters != NULL)
		{
			params_txt = DatumGetCString(
				DirectFunctionCall1(jsonb_out,
					JsonbPGetDatum(parameters)));
			{
				text *params_text = cstring_to_text(params_txt);
				text *quoted_params = DatumGetTextP(
					DirectFunctionCall1(quote_literal,
						PointerGetDatum(params_text)));
				params_quoted = text_to_cstring(quoted_params);
				pfree(params_text);
				pfree(quoted_params);
			}
		}

		if (metrics != NULL)
		{
		metrics_txt = DatumGetCString(
			DirectFunctionCall1(jsonb_out,
				JsonbPGetDatum(metrics)));
		{
				text *metrics_text = cstring_to_text(metrics_txt);
				text *quoted_metrics = DatumGetTextP(
					DirectFunctionCall1(quote_literal,
						PointerGetDatum(metrics_text)));
				metrics_quoted = text_to_cstring(quoted_metrics);
				pfree(metrics_text);
				pfree(quoted_metrics);
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

		ret = SPI_execute(insert_sql.data, false, 1);

		/* ON CONFLICT DO UPDATE returns SPI_OK_UPDATE_RETURNING, not SPI_OK_INSERT_RETURNING */
		if ((ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING)
			|| SPI_processed == 0)
		{
			pfree(insert_sql.data);
			if (algorithm_quoted) pfree(algorithm_quoted);
			if (table_quoted) pfree(table_quoted);
			if (column_quoted) pfree(column_quoted);
			if (params_txt) pfree(params_txt);
			if (params_quoted) pfree(params_quoted);
			if (metrics_txt) pfree(metrics_txt);
			if (metrics_quoted) pfree(metrics_quoted);
			elog(ERROR,
				"ml_catalog_register_model: failed to insert/update ml_models row "
				"(ret=%d, processed=%lu)",
				 ret, (unsigned long) SPI_processed);
		}

		{
			bool isnull_result = false;

			model_id = DatumGetInt32(
				SPI_getbinval(SPI_tuptable->vals[0],
							  SPI_tuptable->tupdesc, 1,
							  &isnull_result));
		if (isnull_result)
				elog(ERROR,
					 "ml_catalog_register_model: model_id NULL after insert/update");
			if (model_id <= 0)
				elog(ERROR,
					"ml_catalog_register_model: model_id=%d is invalid (should be > 0)",
					model_id);
		}

		if (spec->model_data != NULL && model_id > 0)
		{
			Oid argtypes[2];
			Datum values[2];
			char nulls[2];

			resetStringInfo(&sql);
			appendStringInfo(&sql,
				"UPDATE neurondb.ml_models SET model_data = $1 WHERE model_id = $2");

			argtypes[0] = BYTEAOID;
			argtypes[1] = INT4OID;
			values[0] = PointerGetDatum(spec->model_data);
			values[1] = Int32GetDatum(model_id);
			nulls[0] = ' ';
			nulls[1] = ' ';

			ret = SPI_execute_with_args(sql.data, 2, argtypes, values, nulls, false, 0);
			if (ret != SPI_OK_UPDATE || SPI_processed != 1)
			{
				elog(WARNING,
					"ml_catalog_register_model: failed to update model_data (ret=%d, processed=%lu)",
					ret, (unsigned long)SPI_processed);
			}
		}

		/* Cleanup */
		pfree(insert_sql.data);
		if (algorithm_quoted) pfree(algorithm_quoted);
		if (table_quoted) pfree(table_quoted);
		if (column_quoted) pfree(column_quoted);
		if (params_txt) pfree(params_txt);
		if (params_quoted) pfree(params_quoted);
		if (metrics_txt) pfree(metrics_txt);
		if (metrics_quoted) 		pfree(metrics_quoted);
	}

	/* Note: values[] array contains Datums, not pointers to free */
	/* Only free the model_data bytea if it was provided and we're responsible for it */
	/* But spec->model_data is owned by caller, so don't free it here */

	SPI_finish();

	if (spec->project_name == NULL)
		pfree((void *) project_name);

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

	if (model_data_out != NULL)
		*model_data_out = NULL;
	if (parameters_out != NULL)
		*parameters_out = NULL;
	if (metrics_out != NULL)
		*metrics_out = NULL;

	/* Save caller's memory context */
	caller_context = CurrentMemoryContext;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "ml_catalog_fetch_model_payload: SPI_connect failed");

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT model_data, parameters, metrics "
					 "FROM neurondb.ml_models WHERE model_id = %d",
					 model_id);

	/* Use SPI_EXEC_REPEAT_OK to allow seeing uncommitted changes in same transaction */
	ret = SPI_execute(sql.data, true, 1);
	if (ret != SPI_OK_SELECT)
	{
		pfree(sql.data);
		SPI_finish();
		return false;
	}

	if (SPI_processed == 0)
	{
		elog(DEBUG1, "ml_catalog_fetch_model_payload: model_id %d not found in catalog", model_id);
		pfree(sql.data);
		SPI_finish();
		return false;
	}
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
			datum = SPI_getbinval(SPI_tuptable->vals[0],
								  SPI_tuptable->tupdesc, 1, &isnull);
			if (!isnull)
			{
				bytea *copy;

				copy = DatumGetByteaP(datum);
				copy = (bytea *) PG_DETOAST_DATUM_COPY((Datum) copy);
				*model_data_out = copy;
				elog(DEBUG1, "ml_catalog_fetch_model_payload: loaded model_data for model_id %d (size=%u)",
					 model_id, copy ? VARSIZE(copy) : 0);
			}
			else
			{
				*model_data_out = NULL;
				elog(WARNING, "ml_catalog_fetch_model_payload: model_data is NULL for model_id %d - model may not have been stored correctly", model_id);
			}
		}

		if (parameters_out != NULL)
		{
			datum = SPI_getbinval(SPI_tuptable->vals[0],
								  SPI_tuptable->tupdesc, 2, &isnull);
			if (!isnull)
			{
				Jsonb *jsonb_val;

				jsonb_val = DatumGetJsonbP(datum);
				jsonb_val = (Jsonb *) PG_DETOAST_DATUM_COPY((Datum) jsonb_val);
				*parameters_out = jsonb_val;
			}
		}

		if (metrics_out != NULL)
		{
			datum = SPI_getbinval(SPI_tuptable->vals[0],
								  SPI_tuptable->tupdesc, 3, &isnull);
			if (!isnull)
			{
				Jsonb *jsonb_val = NULL;

				PG_TRY();
				{
					jsonb_val = DatumGetJsonbP(datum);
					jsonb_val = (Jsonb *) PG_DETOAST_DATUM_COPY((Datum) jsonb_val);
					*metrics_out = jsonb_val;
				}
				PG_CATCH();
				{
					*metrics_out = NULL;
				}
				PG_END_TRY();
			}
		}

		MemoryContextSwitchTo(oldcontext);
	}

	pfree(sql.data);
	SPI_finish();

	return found;
}
