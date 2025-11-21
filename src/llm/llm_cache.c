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
#include "neurondb.h"
#include "utils/timestamp.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "funcapi.h"
#include "utils/array.h"
#include "parser/parse_type.h"
#include "parser/parse_func.h"
#include "utils/lsyscache.h"
#include "catalog/pg_proc.h"
#include "nodes/makefuncs.h"

/*
 * ndb_quote_json_cstr
 *    Quote a C string for JSON (returns JSON string with quotes and escaping)
 */
static char *
ndb_quote_json_cstr(const char *str)
{
	StringInfoData buf;
	const char *p;
	char *result;

	if (str == NULL)
		return pstrdup("null");

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '"');

	for (p = str; *p; p++)
	{
		switch (*p)
		{
			case '"':
				appendStringInfoString(&buf, "\\\"");
				break;
			case '\\':
				appendStringInfoString(&buf, "\\\\");
				break;
			case '\b':
				appendStringInfoString(&buf, "\\b");
				break;
			case '\f':
				appendStringInfoString(&buf, "\\f");
				break;
			case '\n':
				appendStringInfoString(&buf, "\\n");
				break;
			case '\r':
				appendStringInfoString(&buf, "\\r");
				break;
			case '\t':
				appendStringInfoString(&buf, "\\t");
				break;
			default:
				if ((unsigned char) *p < 0x20)
				{
					appendStringInfo(&buf, "\\u%04x", (unsigned char) *p);
				} else
				{
					appendStringInfoChar(&buf, *p);
				}
				break;
		}
	}

	appendStringInfoChar(&buf, '"');
	result = pstrdup(buf.data);
	pfree(buf.data);
	return result;
}

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
		ndb_quote_json_cstr(key),
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
	appendStringInfo(&val, "{\"text\":%s}", ndb_quote_json_cstr(text));
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

/*
 * ndb_llm_cache_evict_lru
 *    Evict least recently used cache entries when cache exceeds max_size.
 *    Returns number of entries evicted.
 */
static int
ndb_llm_cache_evict_lru(int max_size)
{
	int			spi_ret;
	int			current_count = 0;
	int			evicted = 0;
	StringInfoData sql;

	if (SPI_connect() != SPI_OK_CONNECT)
		return 0;

	/* Get current cache size */
	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT COUNT(*) FROM neurondb.neurondb_llm_cache");
	spi_ret = SPI_execute(sql.data, true, 1);
	if (spi_ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		d;

		d = SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc,
			1,
			&isnull);
		if (!isnull)
			current_count = DatumGetInt32(d);
	}
	pfree(sql.data);

	if (current_count <= max_size)
	{
		SPI_finish();
		return 0;
	}

	/* Evict oldest entries */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"DELETE FROM neurondb.neurondb_llm_cache "
		"WHERE key IN ("
		"  SELECT key FROM neurondb.neurondb_llm_cache "
		"  ORDER BY created_at ASC "
		"  LIMIT %d"
		")",
		current_count - max_size);
	spi_ret = SPI_execute(sql.data, true, 0);
	if (spi_ret == SPI_OK_DELETE)
		evicted = current_count - max_size;
	pfree(sql.data);

	SPI_finish();
	return evicted;
}

/*
 * ndb_llm_cache_store_with_eviction
 *    Store cache entry with automatic eviction if cache exceeds max_size.
 */
void
ndb_llm_cache_store_with_eviction(const char *key, const char *text, int max_size)
{
	ndb_llm_cache_store(key, text);
	if (max_size > 0)
		ndb_llm_cache_evict_lru(max_size);
}

PG_FUNCTION_INFO_V1(ndb_llm_cache_stats);
/*
 * ndb_llm_cache_stats
 *    Return cache statistics as JSONB.
 */
Datum
ndb_llm_cache_stats(PG_FUNCTION_ARGS)
{
	Jsonb	   *result = NULL;
	StringInfoData jsonbuf;
	StringInfoData sql;
	int			spi_ret;
	int			total_entries = 0;
	int			stale_entries = 0;
	int			ttl_seconds = 0;
	TimestampTz oldest_entry = 0;
	TimestampTz newest_entry = 0;

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
			 errmsg("ndb_llm_cache_stats: SPI_connect failed")));

	ttl_seconds = neurondb_llm_cache_ttl;

	/* Get total entries */
	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT COUNT(*) FROM neurondb.neurondb_llm_cache");
	spi_ret = SPI_execute(sql.data, true, 1);
	if (spi_ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		d;

		d = SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc,
			1,
			&isnull);
		if (!isnull)
			total_entries = DatumGetInt32(d);
	}
	pfree(sql.data);

	/* Get stale entries */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT COUNT(*) FROM neurondb.neurondb_llm_cache "
		"WHERE now() - created_at >= make_interval(secs => %d)",
		ttl_seconds);
	spi_ret = SPI_execute(sql.data, true, 1);
	if (spi_ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		d;

		d = SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc,
			1,
			&isnull);
		if (!isnull)
			stale_entries = DatumGetInt32(d);
	}
	pfree(sql.data);

	/* Get oldest and newest entry timestamps */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT MIN(created_at), MAX(created_at) FROM neurondb.neurondb_llm_cache");
	spi_ret = SPI_execute(sql.data, true, 1);
	if (spi_ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull1;
		bool		isnull2;
		Datum		d1;
		Datum		d2;

		d1 = SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc,
			1,
			&isnull1);
		d2 = SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc,
			2,
			&isnull2);
		if (!isnull1)
			oldest_entry = DatumGetTimestampTz(d1);
		if (!isnull2)
			newest_entry = DatumGetTimestampTz(d2);
	}
	pfree(sql.data);

	SPI_finish();

	/* Build JSON result */
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"total_entries\":%d,"
		"\"stale_entries\":%d,"
		"\"valid_entries\":%d,"
		"\"ttl_seconds\":%d",
		total_entries,
		stale_entries,
		total_entries - stale_entries,
		ttl_seconds);
	if (oldest_entry != 0)
	{
		char	   *oldest_str = NULL;

		oldest_str = timestamptz_to_str(oldest_entry);
		appendStringInfo(&jsonbuf, ",\"oldest_entry\":%s", ndb_quote_json_cstr(oldest_str));
		pfree(oldest_str);
	}
	if (newest_entry != 0)
	{
		char	   *newest_str = NULL;

		newest_str = timestamptz_to_str(newest_entry);
		appendStringInfo(&jsonbuf, ",\"newest_entry\":%s", ndb_quote_json_cstr(newest_str));
		pfree(newest_str);
	}
	appendStringInfoChar(&jsonbuf, '}');

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(jsonbuf.data)));
	pfree(jsonbuf.data);

	PG_RETURN_JSONB_P(result);
}

PG_FUNCTION_INFO_V1(ndb_llm_cache_clear);
/*
 * ndb_llm_cache_clear
 *    Clear all cache entries or entries matching a pattern.
 *
 * pattern: TEXT (optional, NULL clears all)
 * Returns: INTEGER (number of entries cleared)
 */
Datum
ndb_llm_cache_clear(PG_FUNCTION_ARGS)
{
	text	   *pattern = NULL;
	char	   *pattern_str = NULL;
	StringInfoData sql;
	int			spi_ret;
	int			deleted = 0;

	if (PG_ARGISNULL(0))
		pattern = NULL;
	else
		pattern = PG_GETARG_TEXT_PP(0);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
			 errmsg("ndb_llm_cache_clear: SPI_connect failed")));

	initStringInfo(&sql);
	if (pattern == NULL)
	{
		appendStringInfo(&sql, "DELETE FROM neurondb.neurondb_llm_cache");
	} else
	{
		pattern_str = text_to_cstring(pattern);
		appendStringInfo(&sql,
			"DELETE FROM neurondb.neurondb_llm_cache "
			"WHERE key LIKE %s",
			ndb_quote_json_cstr(pattern_str));
		pfree(pattern_str);
	}

	spi_ret = SPI_execute(sql.data, true, 0);
	if (spi_ret == SPI_OK_DELETE)
	{
		/* Get number of deleted rows */
		deleted = SPI_processed;
	}

	pfree(sql.data);
	SPI_finish();

	PG_RETURN_INT32(deleted);
}

PG_FUNCTION_INFO_V1(ndb_llm_cache_evict_stale);
/*
 * ndb_llm_cache_evict_stale
 *    Evict stale cache entries (older than TTL).
 *
 * Returns: INTEGER (number of entries evicted)
 */
Datum
ndb_llm_cache_evict_stale(PG_FUNCTION_ARGS)
{
	StringInfoData sql;
	int			spi_ret;
	int			deleted = 0;
	int			ttl_seconds = 0;

	ttl_seconds = neurondb_llm_cache_ttl;

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
			 errmsg("ndb_llm_cache_evict_stale: SPI_connect failed")));

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"DELETE FROM neurondb.neurondb_llm_cache "
		"WHERE now() - created_at >= make_interval(secs => %d)",
		ttl_seconds);

	spi_ret = SPI_execute(sql.data, true, 0);
	if (spi_ret == SPI_OK_DELETE)
		deleted = SPI_processed;

	pfree(sql.data);
	SPI_finish();

	PG_RETURN_INT32(deleted);
}

PG_FUNCTION_INFO_V1(ndb_llm_cache_evict_size);
/*
 * ndb_llm_cache_evict_size
 *    Evict cache entries to maintain max_size limit (LRU eviction).
 *
 * max_size: INTEGER
 * Returns: INTEGER (number of entries evicted)
 */
Datum
ndb_llm_cache_evict_size(PG_FUNCTION_ARGS)
{
	int32		max_size;
	int			evicted = 0;

	max_size = PG_GETARG_INT32(0);
	if (max_size < 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("ndb_llm_cache_evict_size: max_size must be non-negative")));

	evicted = ndb_llm_cache_evict_lru(max_size);

	PG_RETURN_INT32(evicted);
}

PG_FUNCTION_INFO_V1(ndb_llm_cache_warm);
/*
 * ndb_llm_cache_warm
 *    Pre-populate cache with embeddings for given texts and model.
 *
 * texts: TEXT[]
 * model: TEXT (optional)
 * Returns: INTEGER (number of entries cached)
 */
Datum
ndb_llm_cache_warm(PG_FUNCTION_ARGS)
{
	ArrayType  *texts_array;
	Datum	   *text_datums = NULL;
	bool	   *text_nulls = NULL;
	int			nitems = 0;
	int			i;
	int			cached = 0;
	text	   *model_text = NULL;
	char	   *model_str = NULL;

	texts_array = PG_GETARG_ARRAYTYPE_P(0);
	if (PG_ARGISNULL(1))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(1);

	deconstruct_array(texts_array,
		TEXTOID,
		-1,
		false,
		'i',
		&text_datums,
		&text_nulls,
		&nitems);

	if (model_text != NULL)
		model_str = text_to_cstring(model_text);
	else
		model_str = pstrdup("sentence-transformers/all-MiniLM-L6-v2");

	for (i = 0; i < nitems; i++)
	{
		if (!text_nulls[i])
		{
			text	   *input_text = (text *) DatumGetPointer(text_datums[i]);
			char	   *input_str = NULL;
			char	   *cache_key = NULL;
			StringInfoData key_buf;
			Vector	   *embedding = NULL;
			StringInfoData cache_val;
			int			d;
			const char *p;
			uint32		hashval;

			input_str = text_to_cstring(input_text);

			/* Generate cache key */
			initStringInfo(&key_buf);
			appendStringInfo(&key_buf, "embed:");

			hashval = 0;
			for (p = input_str; *p; p++)
				hashval = (hashval * 31) + *p;
			appendStringInfo(&key_buf, "%u:", (unsigned int) hashval);
			appendStringInfoString(&key_buf, model_str);
			
			/* BULLETPROOF: Copy cache_key to current context before SPI calls */
			/* ndb_llm_cache_lookup/store use SPI which may change memory context */
			cache_key = pstrdup(key_buf.data);
			pfree(key_buf.data);  /* Free original StringInfo data */

			/* Check if already cached */
			{
				char	   *cached_text = NULL;
				bool		hit = false;

				hit = ndb_llm_cache_lookup(cache_key, neurondb_llm_cache_ttl, &cached_text);
				if (hit && cached_text != NULL)
				{
					pfree(cached_text);
					pfree(cache_key);
					pfree(input_str);
					continue;
				}
			}

			/* Generate embedding via FunctionCall */
			{
				Oid			embed_text_oid;
				FmgrInfo	flinfo;
				List	   *funcname;
				Oid			argtypes[2];

				funcname = list_make1(makeString("embed_text"));
				if (model_text != NULL)
				{
					argtypes[0] = TEXTOID;
					argtypes[1] = TEXTOID;
					embed_text_oid = LookupFuncName(funcname, 2, argtypes, false);
					if (OidIsValid(embed_text_oid))
					{
						fmgr_info(embed_text_oid, &flinfo);
						embedding = (Vector *) DatumGetPointer(
							FunctionCall2(&flinfo,
								PointerGetDatum(input_text),
								PointerGetDatum(model_text)));
					}
				}
				if (embedding == NULL)
				{
					argtypes[0] = TEXTOID;
					embed_text_oid = LookupFuncName(funcname, 1, argtypes, false);
					if (OidIsValid(embed_text_oid))
					{
						fmgr_info(embed_text_oid, &flinfo);
						embedding = (Vector *) DatumGetPointer(
							FunctionCall1(&flinfo,
								PointerGetDatum(input_text)));
					}
				}
			}

			if (embedding != NULL)
			{
				/* Store in cache */
				initStringInfo(&cache_val);
				appendStringInfo(&cache_val, "{%d", embedding->dim);
				for (d = 0; d < embedding->dim; d++)
					appendStringInfo(&cache_val, ",%.7f", embedding->data[d]);
				appendStringInfoChar(&cache_val, '}');

				ndb_llm_cache_store(cache_key, cache_val.data);
				cached++;

				pfree(cache_val.data);
			}

			pfree(cache_key);
			pfree(input_str);
		}
	}

	if (text_datums)
		pfree(text_datums);
	if (text_nulls)
		pfree(text_nulls);
	if (model_str)
		pfree(model_str);

	PG_RETURN_INT32(cached);
}
