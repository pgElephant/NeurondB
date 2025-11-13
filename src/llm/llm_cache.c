/*-------------------------------------------------------------------------
 *
 * llm_cache.c
 *		LLM response cache with TTL and key-based lookup
 *
 * This cache is backed by the neurondb.neurondb_llm_cache table.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/llm/llm_cache.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "executor/spi.h"
#include "miscadmin.h"
#include "access/xact.h"
#include "lib/stringinfo.h"
#include "neurondb_llm.h"

/*
 * Returns true if a non-stale cache entry is found for 'key' (and text value, if non-NULL), false otherwise.
 * If found and out_text is non-NULL, sets *out_text to a freshly allocated string with the "text" field's value.
 *
 * Note: All variable declarations appear before executable statements for pre-C99 compatibility.
 */
bool
ndb_llm_cache_lookup(const char *key, int max_age_seconds, char **out_text)
{
	bool hit = false;
	char *result_text = NULL;
	StringInfoData buf;
	int exec_res;
	bool isnull;
	Datum d;

	/* All declarations above */

	if (SPI_connect() != SPI_OK_CONNECT)
		return false;

	initStringInfo(&buf);
	appendStringInfo(&buf,
		"SELECT value->>'text' FROM neurondb.neurondb_llm_cache "
		"WHERE key = %s AND now() - created_at < make_interval(secs => %d)",
		quote_literal_cstr(key),
		max_age_seconds > 0 ? max_age_seconds : 600);

	exec_res = SPI_execute(buf.data, true, 1);

	if (exec_res == SPI_OK_SELECT && SPI_processed > 0)
	{
		d = SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc,
			1,
			&isnull);
		if (!isnull)
		{
			result_text = TextDatumGetCString(d);
			if (out_text)
				*out_text = result_text;
			else
				pfree(result_text);
			hit = true;
		}
	}
	SPI_finish();
	return hit;
}

/*
 * Store a text result in the cache under the given key, replacing any existing entry.
 * Variable declarations have been moved before any executable statements for pre-C99 compatibility.
 */
void
ndb_llm_cache_store(const char *key, const char *text)
{
	Oid argtypes[2];
	Datum values[2];
	StringInfoData val;

	/* All declarations above */
	argtypes[0] = TEXTOID;
	argtypes[1] = JSONBOID;

	if (SPI_connect() != SPI_OK_CONNECT)
		return;

	initStringInfo(&val);

	/* Construct JSON with a single 'text' field. */
	appendStringInfo(&val, "{\"text\":%s}", quote_literal_cstr(text));
	values[0] = CStringGetTextDatum(key);
	/* JSONBOID column; supply as text datum, will be cast to jsonb in SQL. */
	values[1] = CStringGetTextDatum(val.data);

	SPI_execute_with_args("INSERT INTO neurondb.neurondb_llm_cache(key, "
			      "value, created_at) "
			      "VALUES($1, $2::jsonb, now()) "
			      "ON CONFLICT (key) DO UPDATE SET "
			      "value=EXCLUDED.value, created_at=now()",
		2,
		argtypes,
		values,
		NULL,
		false,
		0);

	SPI_finish();
}

/*
 * Example test function:
 *   SELECT ndb_llm_cache_test('somekey', 'testvalue', 100);
 *
 * Useful for verifying the detailed implementation.
 * All variable declarations now appear at the top for standards compliance.
 */
PG_FUNCTION_INFO_V1(ndb_llm_cache_test);
Datum
ndb_llm_cache_test(PG_FUNCTION_ARGS)
{
	text *key_in;
	text *val_in;
	int32 max_age;
	char *key;
	char *val;
	char *result;
	bool hit;
	text *out;

	/* All declarations above */
	result = NULL;

	key_in = PG_GETARG_TEXT_PP(0);
	val_in = PG_GETARG_TEXT_PP(1);
	max_age = PG_GETARG_INT32(2);

	key = text_to_cstring(key_in);
	val = text_to_cstring(val_in);

	ndb_llm_cache_store(key, val);

	hit = ndb_llm_cache_lookup(key, max_age, &result);

	if (hit && result)
	{
		out = cstring_to_text(result);
		pfree(result);
		PG_RETURN_TEXT_P(out);
	} else
		PG_RETURN_NULL();
}
