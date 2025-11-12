/*-------------------------------------------------------------------------
 *
 * ml_catalog.c
 *    Helpers for interacting with neurondb.ml_models catalog.
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
	Jsonb *parameters;
	Jsonb *metrics;
	bytea *model_data;
	int32 project_id = 0;
	int32 version = 1;
	int32 model_id = 0;
	int32 ret;
	StringInfoData sql;
	Oid argtypes[3];
	Datum values[3];
	char nulls[3];
	bool isnull = false;

	if (spec == NULL)
		elog(ERROR, "ml_catalog_register_model: invalid spec (NULL)");

	algorithm = spec->algorithm;
	if (algorithm == NULL)
		elog(ERROR,
			"ml_catalog_register_model: algorithm must be "
			"provided");

	training_table = spec->training_table;
	if (training_table == NULL)
		elog(ERROR,
			"ml_catalog_register_model: training_table must be "
			"provided");

	training_column = spec->training_column;
	model_type = (spec->model_type != NULL) ? spec->model_type
						: "classification";
	project_name = (spec->project_name != NULL)
		? spec->project_name
		: ml_catalog_default_project(algorithm, training_table);
	parameters = spec->parameters;
	metrics = spec->metrics;
	model_data = spec->model_data;

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

	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0
		|| SPI_tuptable == NULL)
		elog(ERROR,
			"ml_catalog_register_model: failed to upsert project");

	project_id = DatumGetInt32(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
	if (isnull)
		elog(ERROR,
			"ml_catalog_register_model: project_id NULL after "
			"upsert");
	isnull = false;

	/* Determine next model version for project */
	initStringInfo(&sql);
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
			SPI_tuptable->tupdesc,
			1,
			&isnull));
		if (isnull)
			version = 1;
	}

	pfree(sql.data);

	/* Insert model row */
	{
		Oid insert_argtypes[11];
		Datum insert_values[11];
		char insert_nulls[11];

		memset(insert_nulls, ' ', sizeof(insert_nulls));

		insert_argtypes[0] = INT4OID; /* project_id */
		insert_argtypes[1] = INT4OID; /* version */
		insert_argtypes[2] = TEXTOID; /* algorithm */
		insert_argtypes[3] = TEXTOID; /* training_table */
		insert_argtypes[4] = TEXTOID; /* training_column */
		insert_argtypes[5] = JSONBOID; /* parameters */
		insert_argtypes[6] = BYTEAOID; /* model_data */
		insert_argtypes[7] = JSONBOID; /* metrics */
		insert_argtypes[8] = INT4OID; /* training_time_ms */
		insert_argtypes[9] = INT4OID; /* num_samples */
		insert_argtypes[10] = INT4OID; /* num_features */

		insert_values[0] = Int32GetDatum(project_id);
		insert_values[1] = Int32GetDatum(version);
		insert_values[2] = CStringGetTextDatum(algorithm);
		insert_values[3] = CStringGetTextDatum(training_table);
		if (training_column != NULL)
			insert_values[4] = CStringGetTextDatum(training_column);
		else
			insert_nulls[4] = 'n';

		if (parameters != NULL)
			insert_values[5] = PointerGetDatum(parameters);
		else
			insert_nulls[5] = 'n';

		if (model_data != NULL)
			insert_values[6] = PointerGetDatum(model_data);
		else
			insert_nulls[6] = 'n';

		if (metrics != NULL)
			insert_values[7] = PointerGetDatum(metrics);
		else
			insert_nulls[7] = 'n';

		if (spec->training_time_ms >= 0)
			insert_values[8] =
				Int32GetDatum(spec->training_time_ms);
		else
			insert_nulls[8] = 'n';

		if (spec->num_samples > 0)
			insert_values[9] = Int32GetDatum(spec->num_samples);
		else
			insert_nulls[9] = 'n';

		if (spec->num_features > 0)
			insert_values[10] = Int32GetDatum(spec->num_features);
		else
			insert_nulls[10] = 'n';

		ret = SPI_execute_with_args(
			"INSERT INTO neurondb.ml_models "
			"(project_id, version, algorithm, status, "
			"training_table, "
			"training_column, parameters, model_data, metrics, "
			"training_time_ms, num_samples, num_features, "
			"completed_at) "
			"VALUES ($1, $2, $3::neurondb.ml_algorithm_type, "
			"'completed', "
			"$4, $5, COALESCE($6, '{}'::jsonb), $7, COALESCE($8, "
			"'{}'::jsonb), "
			"$9, $10, $11, NOW()) "
			"RETURNING model_id",
			11,
			insert_argtypes,
			insert_values,
			insert_nulls,
			false,
			1);

		if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
			elog(ERROR,
				"ml_catalog_register_model: failed to insert "
				"ml_models row");

		{
			bool isnull_result = false;
			model_id = DatumGetInt32(
				SPI_getbinval(SPI_tuptable->vals[0],
					SPI_tuptable->tupdesc,
					1,
					&isnull_result));
			if (isnull_result)
				elog(ERROR,
					"ml_catalog_register_model: model_id "
					"NULL "
					"after insert");
		}

		pfree(DatumGetPointer(insert_values[2]));
		pfree(DatumGetPointer(insert_values[3]));
		if (training_column != NULL)
			pfree(DatumGetPointer(insert_values[4]));
	}

	pfree(DatumGetPointer(values[0]));
	pfree(DatumGetPointer(values[1]));
	pfree(DatumGetPointer(values[2]));

	SPI_finish();

	if (spec->project_name == NULL)
		pfree((void *)project_name);

	return model_id;
}

bool
ml_catalog_fetch_model_payload(int32 model_id,
	bytea **model_data_out,
	Jsonb **parameters_out,
	Jsonb **metrics_out)
{
	int ret;
	StringInfoData sql;
	bool found = false;

	if (model_data_out != NULL)
		*model_data_out = NULL;
	if (parameters_out != NULL)
		*parameters_out = NULL;
	if (metrics_out != NULL)
		*metrics_out = NULL;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR,
			"ml_catalog_fetch_model_payload: SPI_connect failed");

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT model_data, parameters, metrics "
		"FROM neurondb.ml_models WHERE model_id = %d",
		model_id);

	ret = SPI_execute(sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		Datum datum;
		bool isnull;

		found = true;

		if (model_data_out != NULL)
		{
			datum = SPI_getbinval(SPI_tuptable->vals[0],
				SPI_tuptable->tupdesc,
				1,
				&isnull);
			if (!isnull)
			{
				bytea *copy = DatumGetByteaP(datum);

				copy = (bytea *)PG_DETOAST_DATUM_COPY(
					(Datum)copy);
				*model_data_out = copy;
			}
		}

		if (parameters_out != NULL)
		{
			datum = SPI_getbinval(SPI_tuptable->vals[0],
				SPI_tuptable->tupdesc,
				2,
				&isnull);
			if (!isnull)
			{
				Datum jsonb_datum = PG_DETOAST_DATUM_COPY(datum);
				Jsonb *jsonb_val = (Jsonb *)DatumGetPointer(jsonb_datum);

				*parameters_out = jsonb_val;
			}
		}

		if (metrics_out != NULL)
		{
			datum = SPI_getbinval(SPI_tuptable->vals[0],
				SPI_tuptable->tupdesc,
				3,
				&isnull);
			if (!isnull)
			{
				Jsonb *jsonb_val = NULL;

				PG_TRY();
				{
					Datum jsonb_datum = PG_DETOAST_DATUM_COPY(datum);

					jsonb_val = (Jsonb *)DatumGetPointer(jsonb_datum);
					*metrics_out = jsonb_val;
				}
				PG_CATCH();
				{
					/* If jsonb is invalid, set to NULL */
					*metrics_out = NULL;
				}
				PG_END_TRY();
			}
		}
	}

	pfree(sql.data);
	SPI_finish();

	return found;
}
