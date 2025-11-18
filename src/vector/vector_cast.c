/*-------------------------------------------------------------------------
 *
 * vector_cast.c
 *		Vector type casting and conversion functions
 *
 * Implements comprehensive type casting between vectors, arrays, and
 * quantized formats (FP16, INT8, sparse).
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *	  contrib/neurondb/vector_cast.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include <math.h>

/*
 * array_to_vector_float4
 *
 * Convert PostgreSQL float4 array to vector.
 */
PG_FUNCTION_INFO_V1(array_to_vector_float4);
Datum
array_to_vector_float4(PG_FUNCTION_ARGS)
{
	ArrayType *arr;
	Vector *result;
	float4 *data;
	int dim;
	int i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("array_to_vector_float4 requires 1 argument, got %d",
					PG_NARGS())));

	arr = PG_GETARG_ARRAYTYPE_P(0);

	if (arr == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("array must not be NULL")));

	if (ARR_NDIM(arr) != 1)
		ereport(ERROR,
			(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				errmsg("array must be one-dimensional")));

	dim = ARR_DIMS(arr)[0];
	if (dim <= 0 || dim > VECTOR_MAX_DIM)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid array dimension: %d (max: %d)",
					dim,
					VECTOR_MAX_DIM)));

	if (ARR_HASNULL(arr))
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("array must not contain NULL values")));

	data = (float4 *)ARR_DATA_PTR(arr);
	if (data == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid array data")));

	result = new_vector(dim);
	if (result == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("out of memory")));
	for (i = 0; i < dim; i++)
		result->data[i] = data[i];

	PG_RETURN_VECTOR_P(result);
}

/*
 * array_to_vector_float8
 *
 * Convert PostgreSQL float8 array to vector.
 */
PG_FUNCTION_INFO_V1(array_to_vector_float8);
Datum
array_to_vector_float8(PG_FUNCTION_ARGS)
{
	ArrayType *arr;
	Vector *result;
	float8 *data;
	int dim;
	int i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("array_to_vector_float8 requires 1 argument, got %d",
					PG_NARGS())));

	arr = PG_GETARG_ARRAYTYPE_P(0);

	if (arr == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("array must not be NULL")));

	if (ARR_NDIM(arr) != 1)
		ereport(ERROR,
			(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				errmsg("array must be one-dimensional")));

	dim = ARR_DIMS(arr)[0];
	if (dim <= 0 || dim > VECTOR_MAX_DIM)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid array dimension: %d (max: %d)",
					dim,
					VECTOR_MAX_DIM)));

	if (ARR_HASNULL(arr))
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("array must not contain NULL values")));

	data = (float8 *)ARR_DATA_PTR(arr);
	if (data == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid array data")));

	result = new_vector(dim);
	if (result == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("out of memory")));
	for (i = 0; i < dim; i++)
		result->data[i] = (float4)data[i];

	PG_RETURN_VECTOR_P(result);
}

/*
 * array_to_vector_integer
 *
 * Convert PostgreSQL integer array to vector.
 */
PG_FUNCTION_INFO_V1(array_to_vector_integer);
Datum
array_to_vector_integer(PG_FUNCTION_ARGS)
{
	ArrayType *arr;
	Vector *result;
	int32 *data;
	int dim;
	int i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("array_to_vector_integer requires 1 argument, got %d",
					PG_NARGS())));

	arr = PG_GETARG_ARRAYTYPE_P(0);

	if (arr == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("array must not be NULL")));

	if (ARR_NDIM(arr) != 1)
		ereport(ERROR,
			(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				errmsg("array must be one-dimensional")));

	dim = ARR_DIMS(arr)[0];
	if (dim <= 0 || dim > VECTOR_MAX_DIM)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid array dimension: %d (max: %d)",
					dim,
					VECTOR_MAX_DIM)));

	if (ARR_HASNULL(arr))
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("array must not contain NULL values")));

	data = (int32 *)ARR_DATA_PTR(arr);
	if (data == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid array data")));

	result = new_vector(dim);
	if (result == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("out of memory")));
	for (i = 0; i < dim; i++)
		result->data[i] = (float4)data[i];

	PG_RETURN_VECTOR_P(result);
}

/*
 * vector_to_array_float4
 *
 * Convert vector to PostgreSQL float4 array.
 */
PG_FUNCTION_INFO_V1(vector_to_array_float4);
Datum
vector_to_array_float4(PG_FUNCTION_ARGS)
{
	Vector *vec;
	ArrayType *result;
	Datum *elems;
	bool *nulls;
	int i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("vector_to_array_float4 requires 1 argument, got %d",
					PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);

	if (vec == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("vector must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid vector dimension: %d",
					vec->dim)));

	elems = (Datum *)palloc(sizeof(Datum) * vec->dim);
	nulls = (bool *)palloc(sizeof(bool) * vec->dim);
	if (elems == NULL || nulls == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("out of memory")));

	for (i = 0; i < vec->dim; i++)
	{
		elems[i] = Float4GetDatum(vec->data[i]);
		nulls[i] = false;
	}

	result = construct_array(elems,
		vec->dim,
		FLOAT4OID,
		sizeof(float4),
		true,
		'i');
	pfree(elems);
	pfree(nulls);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * vector_to_array_float8
 *
 * Convert vector to PostgreSQL float8 array.
 */
PG_FUNCTION_INFO_V1(vector_to_array_float8);
Datum
vector_to_array_float8(PG_FUNCTION_ARGS)
{
	Vector *vec;
	ArrayType *result;
	Datum *elems;
	bool *nulls;
	int i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("vector_to_array_float8 requires 1 argument, got %d",
					PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);

	if (vec == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("vector must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid vector dimension: %d",
					vec->dim)));

	elems = (Datum *)palloc(sizeof(Datum) * vec->dim);
	nulls = (bool *)palloc(sizeof(bool) * vec->dim);
	if (elems == NULL || nulls == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("out of memory")));

	for (i = 0; i < vec->dim; i++)
	{
		elems[i] = Float8GetDatum((float8)vec->data[i]);
		nulls[i] = false;
	}

	result = construct_array(elems,
		vec->dim,
		FLOAT8OID,
		sizeof(float8),
		true,
		'd');
	pfree(elems);
	pfree(nulls);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * vector_cast_dimension
 *
 * Change vector dimension by truncating or padding with zeros.
 */
PG_FUNCTION_INFO_V1(vector_cast_dimension);
Datum
vector_cast_dimension(PG_FUNCTION_ARGS)
{
	Vector *vec;
	int32 new_dim;
	Vector *result;
	int i;

	if (PG_NARGS() != 2)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("vector_cast_dimension requires 2 arguments, got %d",
					PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);
	new_dim = PG_GETARG_INT32(1);

	if (vec == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("vector must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid vector dimension: %d",
					vec->dim)));

	if (new_dim <= 0 || new_dim > VECTOR_MAX_DIM)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid dimension: %d (max: %d)",
					new_dim,
					VECTOR_MAX_DIM)));

	result = new_vector(new_dim);
	if (result == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("out of memory")));

	if (new_dim <= vec->dim)
	{
		/* Truncate */
		for (i = 0; i < new_dim; i++)
			result->data[i] = vec->data[i];
	}
	else
	{
		/* Pad with zeros */
		for (i = 0; i < vec->dim; i++)
			result->data[i] = vec->data[i];
		for (i = vec->dim; i < new_dim; i++)
			result->data[i] = 0.0f;
	}

	PG_RETURN_VECTOR_P(result);
}

