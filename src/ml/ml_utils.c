/*-------------------------------------------------------------------------
 *
 * ml_utils.c
 *    Common utility functions for ML operations
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/ml/ml_utils.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"

#include "neurondb.h"
#include "neurondb_ml.h"

/*
 * fetch_vectors_from_table
 *    Extract all vectors from a table column as float arrays via SPI.
 *    Returns a pointer to an allocated array of float pointers
 *    (each is a vector), sets *out_count to the number returned and
 *    *out_dim to inferred vector dimension.
 */
float **
neurondb_fetch_vectors_from_table(const char *table,
	const char *col,
	int *out_count,
	int *out_dim)
{
	StringInfoData sql;
	int ret;
	int i, d;
	float **result;
	MemoryContext oldcontext;
	MemoryContext caller_context;

	/* Assert: Internal invariants */
	Assert(table != NULL);
	Assert(col != NULL);
	Assert(out_count != NULL);
	Assert(out_dim != NULL);

	/* Defensive: Check NULL inputs */
	if (table == NULL || col == NULL || out_count == NULL ||
		out_dim == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("neurondb: fetch_vectors_from_table "
				       "NULL pointer argument")));

	/* Defensive: Validate string lengths */
	if (strlen(table) == 0 || strlen(table) > NAMEDATALEN ||
		strlen(col) == 0 || strlen(col) > NAMEDATALEN)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: fetch_vectors_from_table "
				       "invalid table or column name length")));

	/* Save the caller's context before SPI operations */
	caller_context = CurrentMemoryContext;

	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT %s FROM %s", col, table);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: fetch_vectors_from_table "
				       "SPI_connect failed")));

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		pfree(sql.data);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: fetch_vectors_from_table "
				       "SPI_execute failed: %s", sql.data)));
	}

	/* Defensive: Validate SPI_tuptable */
	if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
		SPI_tuptable->tupdesc == NULL)
	{
		pfree(sql.data);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: fetch_vectors_from_table "
				       "SPI_tuptable is invalid")));
	}

	*out_count = SPI_processed;
	if (*out_count == 0)
	{
		pfree(sql.data);
		SPI_finish();
		*out_dim = 0;
		return NULL;
	}

	/* Defensive: Validate tuple */
	if (SPI_tuptable->vals[0] == NULL)
	{
		pfree(sql.data);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: fetch_vectors_from_table "
				       "NULL tuple in first row")));
	}

	/* Get dimension from first vector */
	{
		bool isnull;
		Datum first_datum;
		Vector *first_vec;

		first_datum = SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc,
			1,
			&isnull);
		if (isnull)
		{
			pfree(sql.data);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					errmsg("neurondb: fetch_vectors_from_table "
					       "NULL vector in first row")));
		}

		first_vec = DatumGetVector(first_datum);

		/* Defensive: Validate vector */
		if (first_vec == NULL)
		{
			pfree(sql.data);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					errmsg("neurondb: fetch_vectors_from_table "
					       "NULL vector pointer")));
		}

		/* Defensive: Validate dimension */
		if (first_vec->dim <= 0 || first_vec->dim > 65536)
		{
			pfree(sql.data);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: fetch_vectors_from_table "
					       "invalid vector dimension: %d",
					       first_vec->dim)));
		}

		*out_dim = first_vec->dim;
	}

	/* Switch to caller's context to allocate result that survives
	 * SPI_finish() */
	oldcontext = MemoryContextSwitchTo(caller_context);

	result = (float **)palloc0(sizeof(float *) * (*out_count));

	/* Defensive: Validate allocation */
	if (result == NULL)
	{
		MemoryContextSwitchTo(oldcontext);
		pfree(sql.data);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("neurondb: fetch_vectors_from_table "
				       "failed to allocate result array")));
	}

	for (i = 0; i < *out_count; i++)
	{
		bool isnull;
		Datum vec_datum;
		Vector *vec;

		/* Defensive: Validate tuple */
		if (SPI_tuptable->vals[i] == NULL)
		{
			elog(WARNING,
			     "neurondb: fetch_vectors_from_table NULL tuple at "
			     "index %d, skipping", i);
			continue;
		}

		vec_datum = SPI_getbinval(SPI_tuptable->vals[i],
			SPI_tuptable->tupdesc,
			1,
			&isnull);
		if (isnull)
		{
			elog(WARNING,
			     "neurondb: fetch_vectors_from_table NULL vector at "
			     "index %d, skipping", i);
			continue;
		}

		vec = DatumGetVector(vec_datum);

		/* Defensive: Validate vector */
		if (vec == NULL)
		{
			elog(WARNING,
			     "neurondb: fetch_vectors_from_table NULL vector "
			     "pointer at index %d, skipping", i);
			continue;
		}

		/* Defensive: Verify dimension consistency */
		if (vec->dim != *out_dim)
		{
			MemoryContextSwitchTo(oldcontext);
			pfree(sql.data);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: fetch_vectors_from_table "
					       "inconsistent vector dimension at "
					       "row %d: expected %d, got %d",
					       i, *out_dim, vec->dim)));
		}

		/* Copy vector data */
		result[i] = (float *)palloc(sizeof(float) * (*out_dim));

		/* Defensive: Validate allocation */
		if (result[i] == NULL)
		{
			MemoryContextSwitchTo(oldcontext);
			pfree(sql.data);
			SPI_finish();
			ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
					errmsg("neurondb: fetch_vectors_from_table "
					       "failed to allocate vector %d", i)));
		}

		for (d = 0; d < *out_dim; d++)
		{
			/* Defensive: Check for NaN/Inf */
			if (isnan(vec->data[d]) || isinf(vec->data[d]))
			{
				elog(WARNING,
				     "neurondb: fetch_vectors_from_table NaN or "
				     "Infinity at vector %d, dimension %d", i, d);
			}
			result[i][d] = vec->data[d];
		}
	}

	/* Switch back to SPI context before finishing */
	MemoryContextSwitchTo(oldcontext);
	pfree(sql.data);
	SPI_finish();

	elog(DEBUG1,
	     "neurondb: fetch_vectors_from_table fetched %d vectors of "
	     "dimension %d", *out_count, *out_dim);

	return result;
}
