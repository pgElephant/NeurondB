/*-------------------------------------------------------------------------
 *
 * ml_feature_store.c
 *    Feature Store for NeuronDB
 *
 * Provides feature management, versioning, and serving:
 * - neurondb.create_feature_store() - Initialize feature store
 * - neurondb.register_feature() - Register a feature
 * - neurondb.get_features() - Retrieve features for given entities/feature list
 * - neurondb.feature_engineering() - Apply transformations from pipeline
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_feature_store.c
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/timestamp.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "lib/stringinfo.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_spi_safe.h"

static char *entity_key_for_store(int32 store_id);
static char *entity_table_for_store(int32 store_id);
static void get_feature_definitions(int32 store_id,
	char ***feat_names,
	char ***feat_types,
	char ***feat_transforms,
	int *n_features);

/*
 * neurondb.create_feature_store()
 */
PG_FUNCTION_INFO_V1(neurondb_create_feature_store);

Datum
neurondb_create_feature_store(PG_FUNCTION_ARGS)
{
	text *store_name_text = PG_GETARG_TEXT_PP(0);
	text *entity_table_text = PG_GETARG_TEXT_PP(1);
	text *entity_key_text = PG_GETARG_TEXT_PP(2);
	text *description_text = PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);

	char *store_name = text_to_cstring(store_name_text);
	char *entity_table = text_to_cstring(entity_table_text);
	char *entity_key = text_to_cstring(entity_key_text);
	char *description =
		description_text ? text_to_cstring(description_text) : NULL;

	StringInfoData sql;
	int ret;
	bool isnull;
	int store_id;

	elog(DEBUG1,
		"neurondb.create_feature_store: name='%s', entity_table='%s', entity_key='%s'",
		store_name,
		entity_table,
		entity_key);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("could not connect to SPI manager")));

	/* Insert feature store definition */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"INSERT INTO neurondb.feature_stores "
		"(store_name, entity_table, entity_key, description, "
		"created_at, updated_at) "
		"VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
		"RETURNING store_id",
		quote_literal_cstr(store_name),
		quote_literal_cstr(entity_table),
		quote_literal_cstr(entity_key),
		description ? quote_literal_cstr(description) : "NULL");

	ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed != 1)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("failed to create feature store")));

	store_id = DatumGetInt32(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	SPI_finish();

	NDB_SAFE_PFREE_AND_NULL(store_name);
	NDB_SAFE_PFREE_AND_NULL(entity_table);
	NDB_SAFE_PFREE_AND_NULL(entity_key);
	if (description)
		NDB_SAFE_PFREE_AND_NULL(description);

	PG_RETURN_INT32(store_id);
}

/*
 * neurondb.register_feature()
 */
PG_FUNCTION_INFO_V1(neurondb_register_feature);

Datum
neurondb_register_feature(PG_FUNCTION_ARGS)
{
	int32 store_id = PG_GETARG_INT32(0);
	text *feature_name_text = PG_GETARG_TEXT_PP(1);
	text *feature_type_text = PG_GETARG_TEXT_PP(2);
	text *transformation_text =
		PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);

	char *feature_name;
	char *feature_type;
	char *transformation;
	StringInfoData sql;
	int ret;
	int feature_id;
	bool isnull;

	feature_name = text_to_cstring(feature_name_text);
	feature_type = text_to_cstring(feature_type_text);
	transformation = transformation_text
		? text_to_cstring(transformation_text)
		: NULL;

	elog(DEBUG1,
		"neurondb.register_feature: store=%d, feature='%s', type=%s, transformation='%s'",
		store_id,
		feature_name,
		feature_type,
		transformation ? transformation : "none");

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("could not connect to SPI manager")));

	initStringInfo(&sql);
	if (transformation)
	{
		appendStringInfo(&sql,
			"INSERT INTO neurondb.feature_definitions ("
			" store_id, feature_name, feature_type, "
			"transformation, created_at) "
			"VALUES (%d, %s, %s, %s, CURRENT_TIMESTAMP) "
			"RETURNING feature_id",
			store_id,
			quote_literal_cstr(feature_name),
			quote_literal_cstr(feature_type),
			quote_literal_cstr(transformation));
	} else
	{
		appendStringInfo(&sql,
			"INSERT INTO neurondb.feature_definitions ("
			" store_id, feature_name, feature_type, created_at) "
			"VALUES (%d, %s, %s, CURRENT_TIMESTAMP) "
			"RETURNING feature_id",
			store_id,
			quote_literal_cstr(feature_name),
			quote_literal_cstr(feature_type));
	}

	ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed != 1)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("failed to register feature")));

	feature_id = DatumGetInt32(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	SPI_finish();

	NDB_SAFE_PFREE_AND_NULL(feature_name);
	NDB_SAFE_PFREE_AND_NULL(feature_type);
	if (transformation)
		NDB_SAFE_PFREE_AND_NULL(transformation);

	PG_RETURN_INT32(feature_id);
}

/*
 * neurondb.get_features()
 * Retrieves features for a given list of entity keys and list of feature names.
 *
 * Arguments:
 *    0: int4 store_id
 *    1: text[] entity_ids
 *    2: text[] feature_names (optional, may be NULL for all)
 *
 * Returns:
 *    jsonb: {"entity_id": {"feature1": value1, ...}, ...}
 */
PG_FUNCTION_INFO_V1(neurondb_get_features);

Datum
neurondb_get_features(PG_FUNCTION_ARGS)
{
	int32 store_id = PG_GETARG_INT32(0);
	ArrayType *entities_array = PG_GETARG_ARRAYTYPE_P(1);
	ArrayType *features_array =
		PG_ARGISNULL(2) ? NULL : PG_GETARG_ARRAYTYPE_P(2);

	Oid entity_elemtype, feat_elemtype;
	int16 entity_typlen, feat_typlen;
	bool entity_typbyval, feat_typbyval;
	char entity_typalign, feat_typalign;
	int num_entities, num_features = 0;
	int i, j, ndefs;
	Datum *entity_datums = NULL;
	bool *entity_nulls = NULL;
	Datum *feat_datums = NULL;
	bool *feat_nulls = NULL;

	char *entity_key_col = NULL;
	char *entity_table = NULL;
	char **feature_names = NULL;
	char **feature_types = NULL;
	char **feature_transforms = NULL;
	MemoryContext oldcontext, fncontext;
	StringInfoData sql;
	int spi_ret;

	JsonbParseState *ps_top = NULL;
	JsonbValue *o = NULL;
	JsonbValue ent_key_jbv, feature_obj_value;
	Jsonb *result_jsonb;

	/* Validate array arguments */
	if (ARR_NDIM(entities_array) != 1 || ARR_HASNULL(entities_array))
		ereport(ERROR,
			(errmsg("entity_ids must be a one-dimensional, not "
				"null array")));
	entity_elemtype = ARR_ELEMTYPE(entities_array);
	get_typlenbyvalalign(entity_elemtype,
		&entity_typlen,
		&entity_typbyval,
		&entity_typalign);
	deconstruct_array(entities_array,
		entity_elemtype,
		entity_typlen,
		entity_typbyval,
		entity_typalign,
		&entity_datums,
		&entity_nulls,
		&num_entities);

	/* Features */
	if (features_array)
	{
		if (ARR_NDIM(features_array) != 1
			|| ARR_HASNULL(features_array))
			ereport(ERROR,
				(errmsg("feature_names must be a "
					"one-dimensional, not null array")));
		feat_elemtype = ARR_ELEMTYPE(features_array);
		get_typlenbyvalalign(feat_elemtype,
			&feat_typlen,
			&feat_typbyval,
			&feat_typalign);
		deconstruct_array(features_array,
			feat_elemtype,
			feat_typlen,
			feat_typbyval,
			feat_typalign,
			&feat_datums,
			&feat_nulls,
			&num_features);

		feature_names = (char **)palloc(sizeof(char *) * num_features);
		for (i = 0; i < num_features; ++i)
			feature_names[i] = TextDatumGetCString(feat_datums[i]);
	}

	/* Look up entity table name and entity key for this store */
	entity_key_col = entity_key_for_store(store_id);
	entity_table = entity_table_for_store(store_id);

	/* Find all feature definitions for this store */
	get_feature_definitions(store_id,
		&feature_names,
		&feature_types,
		&feature_transforms,
		&ndefs);

	/* Create a context for our output assembly */
	fncontext = AllocSetContextCreate(CurrentMemoryContext,
		"neurondb_get_features_context",
		ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(fncontext);

	/* Build SQL to select requested features for each entity */
	for (i = 0; i < num_entities; ++i)
	{
		char *eid = TextDatumGetCString(entity_datums[i]);
		initStringInfo(&sql);

		appendStringInfo(
			&sql, "SELECT %s", quote_identifier(entity_key_col));
		for (j = 0; j < ndefs; ++j)
		{
			appendStringInfo(&sql, ", %s", quote_identifier(feature_names[j]));
		}
		appendStringInfo(&sql,
			" FROM %s WHERE %s = %s",
			quote_identifier(entity_table),
			quote_identifier(entity_key_col),
			quote_literal_cstr(eid));

		/* Execute for this entity */
		if (SPI_connect() != SPI_OK_CONNECT)
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("could not connect to SPI "
					       "manager")));

		spi_ret = ndb_spi_execute_safe(sql.data, true, 1);
	NDB_CHECK_SPI_TUPTABLE();

		if (spi_ret == SPI_OK_SELECT && SPI_processed == 1)
		{
			HeapTuple tuple = SPI_tuptable->vals[0];
			TupleDesc tupdesc = SPI_tuptable->tupdesc;
			int nf = ndefs;
			JsonbPair *obj_pairs =
				(JsonbPair *)palloc(sizeof(JsonbPair) * nf);
			bool isnull;
			for (j = 0; j < ndefs; ++j)
			{
				Datum feat_val = SPI_getbinval(
					tuple, tupdesc, j + 2, &isnull);
				obj_pairs[j].key.type = jbvString;
				obj_pairs[j].key.val.string.len =
					strlen(feature_names[j]);
				obj_pairs[j].key.val.string.val =
					feature_names[j];
				if (isnull)
				{
					obj_pairs[j].value.type = jbvNull;
				} else
				{
					/* Numeric and text only */
					if (strcmp(feature_types[j], "float8")
						== 0)
					{
						obj_pairs[j].value.type =
							jbvNumeric;
						obj_pairs[j]
							.value.val
							.numeric = DatumGetNumeric(
							DirectFunctionCall1(
								float8_numeric,
								feat_val));
					} else if (strcmp(feature_types[j],
							   "int4")
						== 0)
					{
						obj_pairs[j].value.type =
							jbvNumeric;
						obj_pairs[j]
							.value.val
							.numeric = DatumGetNumeric(
							DirectFunctionCall1(
								int4_numeric,
								feat_val));
					} else
					{
						char *sval =
							TextDatumGetCString(
								feat_val);
						obj_pairs[j].value.type =
							jbvString;
						obj_pairs[j]
							.value.val.string.val =
							sval;
						obj_pairs[j]
							.value.val.string.len =
							strlen(sval);
					}
				}
			}
			/* Now build {"entity_id": {feature1: v1, ...}} */
			ent_key_jbv.type = jbvString;
			ent_key_jbv.val.string.val = eid;
			ent_key_jbv.val.string.len = strlen(eid);

			feature_obj_value.type = jbvObject;
			feature_obj_value.val.object.nPairs = nf;
			feature_obj_value.val.object.pairs = obj_pairs;

			pushJsonbValue(&ps_top, WJB_BEGIN_OBJECT, NULL);
			pushJsonbValue(&ps_top, WJB_KEY, &ent_key_jbv);
			pushJsonbValue(&ps_top, WJB_VALUE, &feature_obj_value);
			o = pushJsonbValue(&ps_top, WJB_END_OBJECT, NULL);
		} else
		{
			/* entity row not found: return nulls for all features */
			int nf = ndefs;
			JsonbPair *obj_pairs =
				(JsonbPair *)palloc(sizeof(JsonbPair) * nf);
			for (j = 0; j < ndefs; ++j)
			{
				obj_pairs[j].key.type = jbvString;
				obj_pairs[j].key.val.string.len =
					strlen(feature_names[j]);
				obj_pairs[j].key.val.string.val =
					feature_names[j];
				obj_pairs[j].value.type = jbvNull;
			}
			ent_key_jbv.type = jbvString;
			ent_key_jbv.val.string.val = eid;
			ent_key_jbv.val.string.len = strlen(eid);

			feature_obj_value.type = jbvObject;
			feature_obj_value.val.object.nPairs = nf;
			feature_obj_value.val.object.pairs = obj_pairs;

			pushJsonbValue(&ps_top, WJB_BEGIN_OBJECT, NULL);
			pushJsonbValue(&ps_top, WJB_KEY, &ent_key_jbv);
			pushJsonbValue(&ps_top, WJB_VALUE, &feature_obj_value);
			o = pushJsonbValue(&ps_top, WJB_END_OBJECT, NULL);
		}

		SPI_finish();
		NDB_SAFE_PFREE_AND_NULL(eid);
		NDB_SAFE_PFREE_AND_NULL(sql.data);
	}

	/* Top level: assemble all entity objects into one result object */
	if (!o)
		result_jsonb = DatumGetJsonbP(
			DirectFunctionCall1(jsonb_in, CStringGetDatum("{}")));
	else
		result_jsonb = JsonbValueToJsonb(o);

	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(fncontext);

	if (entity_key_col)
		NDB_SAFE_PFREE_AND_NULL(entity_key_col);
	if (entity_table)
		NDB_SAFE_PFREE_AND_NULL(entity_table);
	if (feature_names)
	{
		for (j = 0; j < ndefs; ++j)
			if (feature_names[j])
				NDB_SAFE_PFREE_AND_NULL(feature_names[j]);
		NDB_SAFE_PFREE_AND_NULL(feature_names);
	}
	if (feature_types)
	{
		for (j = 0; j < ndefs; ++j)
			if (feature_types[j])
				NDB_SAFE_PFREE_AND_NULL(feature_types[j]);
		NDB_SAFE_PFREE_AND_NULL(feature_types);
	}
	if (feature_transforms)
	{
		for (j = 0; j < ndefs; ++j)
			if (feature_transforms[j])
				NDB_SAFE_PFREE_AND_NULL(feature_transforms[j]);
		NDB_SAFE_PFREE_AND_NULL(feature_transforms);
	}
	NDB_SAFE_PFREE_AND_NULL(entity_datums);

	PG_RETURN_JSONB_P(result_jsonb);
}

/*
 * neurondb.feature_engineering()
 * For every entity, applies feature transformations as defined in feature_definitions table.
 * Populates features in the entity_table.
 * Arguments:
 *    0 : store_id (int4)
 *    1 : JSONB pipeline (for future, not used)
 *    2 : source_table (text, table to source transformation data from)
 * Returns:
 *    int4: number of rows updated
 */
PG_FUNCTION_INFO_V1(neurondb_feature_engineering);

Datum
neurondb_feature_engineering(PG_FUNCTION_ARGS)
{
	int32 store_id = PG_GETARG_INT32(0);
	text *source_table_text;
	char *source_table;
	char *entity_key = entity_key_for_store(store_id);
	char *entity_table = entity_table_for_store(store_id);
	char **feature_names = NULL;
	char **feature_types = NULL;
	char **feature_transforms = NULL;
	int ndefs, i, ret, updated = 0;
	StringInfoData sql;

	(void)PG_GETARG_JSONB_P(1); /* unused for now */
	source_table_text = PG_GETARG_TEXT_PP(2);
	source_table = text_to_cstring(source_table_text);

	get_feature_definitions(store_id,
		&feature_names,
		&feature_types,
		&feature_transforms,
		&ndefs);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("could not connect to SPI manager")));

	/*
     * Apply each transformation to each feature.
     * For simplicity, we evaluate SQL expressions for each transformation,
     * and update the features in the entity_table via an UPDATE ... FROM ... statement.
     * Transformation is assumed to be a SQL expression or NULL (in which case, nothing is done).
     */
	for (i = 0; i < ndefs; ++i)
	{
		if (feature_transforms[i] == NULL
			|| strlen(feature_transforms[i]) == 0)
			continue;
		/* Build UPDATE SQL */
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"UPDATE %s dst SET %s = src.val FROM "
			"(SELECT %s AS %s, %s AS val FROM %s) src "
			"WHERE dst.%s = src.%s",
			quote_identifier(entity_table),
			quote_identifier(feature_names[i]),
			quote_identifier(entity_key),
			quote_identifier(entity_key),
			feature_transforms[i],
			quote_identifier(source_table),
			quote_identifier(entity_key),
			quote_identifier(entity_key));
		ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_UPDATE)
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("Failed to update feature '%s'",
						feature_names[i])));
		updated += SPI_processed;
		NDB_SAFE_PFREE_AND_NULL(sql.data);
	}

	SPI_finish();

	NDB_SAFE_PFREE_AND_NULL(source_table);
	if (entity_key)
		NDB_SAFE_PFREE_AND_NULL(entity_key);
	if (entity_table)
		NDB_SAFE_PFREE_AND_NULL(entity_table);
	if (feature_names)
	{
		for (i = 0; i < ndefs; ++i)
			if (feature_names[i])
				NDB_SAFE_PFREE_AND_NULL(feature_names[i]);
		NDB_SAFE_PFREE_AND_NULL(feature_names);
	}
	if (feature_types)
	{
		for (i = 0; i < ndefs; ++i)
			if (feature_types[i])
				NDB_SAFE_PFREE_AND_NULL(feature_types[i]);
		NDB_SAFE_PFREE_AND_NULL(feature_types);
	}
	if (feature_transforms)
	{
		for (i = 0; i < ndefs; ++i)
			if (feature_transforms[i])
				NDB_SAFE_PFREE_AND_NULL(feature_transforms[i]);
		NDB_SAFE_PFREE_AND_NULL(feature_transforms);
	}

	PG_RETURN_INT32(updated);
}

/*
 * Helper: get entity key column name for a store
 */
static char *
entity_key_for_store(int32 store_id)
{
	StringInfoData sql;
	int ret;
	bool isnull;
	char *entity_key;

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT entity_key FROM neurondb.feature_stores WHERE store_id = %d",
		store_id);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("could not connect to SPI manager in "
				       "helper")));

	ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT || SPI_processed != 1)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("feature store %d not found",
					store_id)));
	}

	entity_key = TextDatumGetCString(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	SPI_finish();
	return entity_key;
}

/*
 * Helper: get entity table name for a store
 */
static char *
entity_table_for_store(int32 store_id)
{
	StringInfoData sql;
	int ret;
	bool isnull;
	char *entity_table;

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT entity_table FROM neurondb.feature_stores WHERE store_id = %d",
		store_id);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("could not connect to SPI manager in "
				       "helper")));

	ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT || SPI_processed != 1)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("feature store %d not found",
					store_id)));
	}

	entity_table = TextDatumGetCString(SPI_getbinval(
		SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	SPI_finish();
	return entity_table;
}

/*
 * Helper: get all features for a store (names, types, optional transforms)
 */
static void
get_feature_definitions(int32 store_id,
	char ***out_feat_names,
	char ***out_feat_types,
	char ***out_feat_transforms,
	int *out_n_features)
{
	StringInfoData sql;
	int ret, i, nfeat;
	bool isnull;
	char **fnames;
	char **ftypes;
	char **ftransforms;

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT feature_name, feature_type, transformation "
		"FROM neurondb.feature_definitions WHERE store_id = %d ORDER BY feature_id",
		store_id);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("could not connect to SPI manager in "
				       "feature_definitions helper")));

	ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("could not retrieve feature definitions for store %d",
					store_id)));
	}

	nfeat = SPI_processed;
	fnames = (char **)palloc0(sizeof(char *) * nfeat);
	ftypes = (char **)palloc0(sizeof(char *) * nfeat);
	ftransforms = (char **)palloc0(sizeof(char *) * nfeat);

	for (i = 0; i < nfeat; ++i)
	{
		fnames[i] =
			TextDatumGetCString(SPI_getbinval(SPI_tuptable->vals[i],
				SPI_tuptable->tupdesc,
				1,
				&isnull));
		ftypes[i] =
			TextDatumGetCString(SPI_getbinval(SPI_tuptable->vals[i],
				SPI_tuptable->tupdesc,
				2,
				&isnull));
		if (SPI_getbinval(SPI_tuptable->vals[i],
			    SPI_tuptable->tupdesc,
			    3,
			    &isnull)
			&& isnull)
			ftransforms[i] = NULL;
		else
			ftransforms[i] = TextDatumGetCString(
				SPI_getbinval(SPI_tuptable->vals[i],
					SPI_tuptable->tupdesc,
					3,
					&isnull));
	}

	SPI_finish();
	*out_feat_names = fnames;
	*out_feat_types = ftypes;
	*out_feat_transforms = ftransforms;
	*out_n_features = nfeat;
}
