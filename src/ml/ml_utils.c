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
 *    Returns a pointer to an allocated array of float pointers (each is a vector),
 *    sets *out_count to the number returned and *out_dim to inferred vector dimension.
 */
float **
neurondb_fetch_vectors_from_table(const char *table, const char *col, int *out_count, int *out_dim)
{
	StringInfoData	sql;
	int				ret;
	int				i, d;
	float			**result;
	MemoryContext	oldcontext;
	MemoryContext	caller_context;

	/* Save the caller's context before SPI operations */
	caller_context = CurrentMemoryContext;

	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT %s FROM %s", col, table);

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		elog(ERROR, "SPI_execute failed: %s", sql.data);
	}

	*out_count = SPI_processed;
	if (*out_count == 0)
	{
		SPI_finish();
		*out_dim = 0;
		return NULL;
	}

	/* Get dimension from first vector */
	{
		bool		isnull;
		Datum		first_datum;
		Vector	   *first_vec;

		first_datum = SPI_getbinval(SPI_tuptable->vals[0],
									SPI_tuptable->tupdesc,
									1, &isnull);
		if (isnull)
		{
			SPI_finish();
			elog(ERROR, "NULL vector in first row");
		}
		
		first_vec = DatumGetVector(first_datum);
		*out_dim = first_vec->dim;
	}

	/* Switch to caller's context to allocate result that survives SPI_finish() */
	oldcontext = MemoryContextSwitchTo(caller_context);
	
	result = (float **) palloc0(sizeof(float *) * (*out_count));

	elog(DEBUG1, "neurondb: copying %d vectors of dimension %d", *out_count, *out_dim);

	for (i = 0; i < *out_count; i++)
	{
		bool		isnull;
		Datum		vec_datum;
		Vector	   *vec;

		vec_datum = SPI_getbinval(SPI_tuptable->vals[i],
								  SPI_tuptable->tupdesc,
								  1, &isnull);
		if (isnull)
		{
			SPI_finish();
			elog(ERROR, "NULL vector at row %d", i);
		}

		vec = DatumGetVector(vec_datum);
		
		/* Verify dimension consistency */
		if (vec->dim != *out_dim)
		{
			SPI_finish();
			elog(ERROR, "Inconsistent vector dimension at row %d: expected %d, got %d",
				 i, *out_dim, vec->dim);
		}
		
		/* Copy vector data */
		result[i] = (float *) palloc(sizeof(float) * (*out_dim));
		for (d = 0; d < *out_dim; d++)
			result[i][d] = vec->data[d];
			
		if (i % 1000 == 0 && i > 0)
			elog(DEBUG1, "neurondb: copied %d vectors", i);
	}
	
	elog(DEBUG1, "neurondb: all %d vectors copied successfully", *out_count);

	/* Switch back to SPI context before finishing */
	MemoryContextSwitchTo(oldcontext);
	SPI_finish();
	
	return result;
}

