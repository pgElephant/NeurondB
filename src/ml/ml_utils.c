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

	/* Assume all vectors have same dimension, derive from first result. */
	{
		ArrayType  *first_vec;
		int			ndims;
		const int  *dims;

		first_vec = DatumGetArrayTypeP(SPI_getbinval(SPI_tuptable->vals[0],
													 SPI_tuptable->tupdesc,
													 1, NULL));
		ndims = ARR_NDIM(first_vec);
		dims = ARR_DIMS(first_vec);

		if (ndims != 1)
		{
			SPI_finish();
			elog(ERROR, "Expected 1-D array for vector column");
		}
		*out_dim = dims[0];
	}

	result = (float **) palloc0(sizeof(float *) * (*out_count));

	for (i = 0; i < *out_count; i++)
	{
		bool		isnull;
		ArrayType  *vec;
		float4	   *vals;

		vec = DatumGetArrayTypeP(SPI_getbinval(SPI_tuptable->vals[i],
											   SPI_tuptable->tupdesc,
											   1, &isnull));
		if (isnull)
		{
			SPI_finish();
			elog(ERROR, "NULL vector at row %d", i);
		}

		vals = (float4 *) ARR_DATA_PTR(vec);
		result[i] = (float *) palloc(sizeof(float) * (*out_dim));
		for (d = 0; d < *out_dim; d++)
			result[i][d] = vals[d];
	}

	SPI_finish();
	return result;
}

