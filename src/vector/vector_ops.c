/*-------------------------------------------------------------------------
 *
 * vector_ops.c
 *		Extended vector operations and transformations
 *
 * This file implements advanced vector manipulation functions including
 * element access, slicing, concatenation, element-wise operations
 * (absolute value, square, square root, power), Hadamard product,
 * statistical functions (mean, variance, stddev, min, max), vector
 * comparison, clipping, standardization, and normalization methods.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  contrib/neurondb/vector_ops.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include <math.h>
#include <stdint.h>

/*
 * Vector element access
 */
PG_FUNCTION_INFO_V1(vector_get);
Datum
vector_get(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	int32		idx;

	CHECK_NARGS(2);
	v = PG_GETARG_VECTOR_P(0);
	idx = PG_GETARG_INT32(1);

	/* Defensive: Check NULL vector */
	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot get element from NULL vector")));

	/* Defensive: Validate vector dimension */
	if (v->dim <= 0 || v->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension %d", v->dim)));

	/* Defensive: Validate vector size matches dimension */
	if (VARSIZE_ANY(v) < (int) offsetof(Vector, data) + (int) (sizeof(float4) * v->dim))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("vector size %d does not match dimension %d",
						VARSIZE_ANY(v), v->dim)));

	/* Defensive: Check index bounds */
	if (idx < 0 || idx >= v->dim)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("index %d out of bounds for vector of dimension %d",
						idx, v->dim)));

	/* Defensive: Validate data pointer is accessible */
	if (v->data == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("vector data pointer is NULL")));

	PG_RETURN_FLOAT4(v->data[idx]);
}

/*
 * Vector element update
 */
PG_FUNCTION_INFO_V1(vector_set);
Datum
vector_set(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	int32		idx;
	float4		val;
	Vector	   *result;

	CHECK_NARGS(3);
	v = PG_GETARG_VECTOR_P(0);
	idx = PG_GETARG_INT32(1);
	val = PG_GETARG_FLOAT4(2);

	/* Defensive: Check NULL vector */
	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot set element in NULL vector")));

	/* Defensive: Validate vector dimension */
	if (v->dim <= 0 || v->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension %d", v->dim)));

	/* Defensive: Check index bounds */
	if (idx < 0 || idx >= v->dim)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("index %d out of bounds for vector of dimension %d",
						idx, v->dim)));

	/* Defensive: Validate value is finite */
	if (isnan(val) || isinf(val))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cannot set vector element to NaN or Infinity")));

	result = copy_vector(v);
	result->data[idx] = val;

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector slice (extract subvector)
 */
PG_FUNCTION_INFO_V1(vector_slice);
Datum
vector_slice(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	int32		start;
	int32		end;
	Vector	   *result;
	int			new_dim;

	CHECK_NARGS(3);
	v = PG_GETARG_VECTOR_P(0);
	start = PG_GETARG_INT32(1);
	end = PG_GETARG_INT32(2);

	/* Defensive: Check NULL vector */
	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot slice NULL vector")));

	/* Defensive: Validate vector dimension */
	if (v->dim <= 0 || v->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension %d", v->dim)));

	/* Defensive: Validate slice bounds */
	if (start < 0 || start >= v->dim)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("slice start index %d out of bounds for vector of dimension %d",
						start, v->dim)));

	if (end < start || end > v->dim)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("slice end index %d invalid (start=%d, dim=%d)",
						end, start, v->dim)));

	new_dim = end - start;

	/* Defensive: Validate new dimension */
	if (new_dim <= 0 || new_dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("slice would create invalid dimension %d", new_dim)));

	/* Defensive: Check for overflow in pointer arithmetic */
	if (start > v->dim - new_dim)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("slice bounds would cause buffer overflow")));

	result = new_vector(new_dim);
	memcpy(result->data, v->data + start, sizeof(float4) * new_dim);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector append
 */
PG_FUNCTION_INFO_V1(vector_append);
Datum
vector_append(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	float4		val;
	Vector	   *result;
	int			new_dim;

	CHECK_NARGS(2);
	v = PG_GETARG_VECTOR_P(0);
	val = PG_GETARG_FLOAT4(1);

	/* Defensive: Check NULL vector */
	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot append to NULL vector")));

	/* Defensive: Validate vector dimension */
	if (v->dim <= 0 || v->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension %d", v->dim)));

	/* Defensive: Check for overflow */
	new_dim = v->dim + 1;
	if (new_dim > VECTOR_MAX_DIM || new_dim <= v->dim)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("appending would exceed maximum vector dimension %d",
						VECTOR_MAX_DIM)));

	/* Defensive: Validate value is finite */
	if (isnan(val) || isinf(val))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cannot append NaN or Infinity to vector")));

	result = new_vector(new_dim);
	memcpy(result->data, v->data, sizeof(float4) * v->dim);
	result->data[v->dim] = val;

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector prepend
 */
PG_FUNCTION_INFO_V1(vector_prepend);
Datum
vector_prepend(PG_FUNCTION_ARGS)
{
	float4		val;
	Vector	   *v;
	Vector	   *result;
	int			new_dim;

	CHECK_NARGS(2);
	val = PG_GETARG_FLOAT4(0);
	v = PG_GETARG_VECTOR_P(1);

	/* Defensive: Check NULL vector */
	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot prepend to NULL vector")));

	/* Defensive: Validate vector dimension */
	if (v->dim <= 0 || v->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension %d", v->dim)));

	/* Defensive: Check for overflow */
	new_dim = v->dim + 1;
	if (new_dim > VECTOR_MAX_DIM || new_dim <= v->dim)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("prepending would exceed maximum vector dimension %d",
						VECTOR_MAX_DIM)));

	/* Defensive: Validate value is finite */
	if (isnan(val) || isinf(val))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cannot prepend NaN or Infinity to vector")));

	result = new_vector(new_dim);
	result->data[0] = val;
	memcpy(result->data + 1, v->data, sizeof(float4) * v->dim);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise absolute value
 */
PG_FUNCTION_INFO_V1(vector_abs);
Datum
vector_abs(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	Vector	   *result;
	int			i;

	CHECK_NARGS(1);
	v = PG_GETARG_VECTOR_P(0);

	/* Defensive: Check NULL vector */
	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute absolute value of NULL vector")));

	/* Defensive: Validate vector dimension */
	if (v->dim <= 0 || v->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension %d", v->dim)));

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		/* Defensive: Check for NaN/Inf before fabs */
		if (isnan(v->data[i]) || isinf(v->data[i]))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("cannot compute absolute value of NaN or Infinity at index %d", i)));

		result->data[i] = fabs(v->data[i]);

		/* Defensive: Validate result */
		if (isnan(result->data[i]) || isinf(result->data[i]))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("absolute value calculation resulted in NaN or Infinity at index %d", i)));
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise square
 */
PG_FUNCTION_INFO_V1(vector_square);
Datum
vector_square(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	Vector	   *result;
	double		val;
	double		squared;
	int			i;

	CHECK_NARGS(1);
	v = PG_GETARG_VECTOR_P(0);

	/* Defensive: Check NULL vector */
	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot square NULL vector")));

	/* Defensive: Validate vector dimension */
	if (v->dim <= 0 || v->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension %d", v->dim)));

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		val = (double) v->data[i];
		squared = val * val;

		/* Defensive: Check for overflow in squaring */
		if (isinf(squared) || isnan(squared))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("squaring resulted in overflow at index %d (value=%.15e)",
							i, val)));

		result->data[i] = (float4) squared;

		/* Defensive: Validate result */
		if (isnan(result->data[i]) || isinf(result->data[i]))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("squaring resulted in NaN or Infinity at index %d", i)));
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise square root
 */
PG_FUNCTION_INFO_V1(vector_sqrt);
Datum
vector_sqrt(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	Vector	   *result;
	double		val;
	double		sqrt_val;
	int			i;

	CHECK_NARGS(1);
	v = PG_GETARG_VECTOR_P(0);

	/* Defensive: Check NULL vector */
	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute square root of NULL vector")));

	/* Defensive: Validate vector dimension */
	if (v->dim <= 0 || v->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension %d", v->dim)));

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		val = (double) v->data[i];

		/* Defensive: Check for NaN/Inf */
		if (isnan(val) || isinf(val))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("cannot take square root of NaN or Infinity at index %d", i)));

		/* Defensive: Check for negative values */
		if (val < 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("cannot take square root of negative number at index %d (value=%.15e)",
							i, val)));

		sqrt_val = sqrt(val);

		/* Defensive: Validate sqrt result */
		if (isnan(sqrt_val) || isinf(sqrt_val))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("square root calculation resulted in NaN or Infinity at index %d", i)));

		result->data[i] = (float4) sqrt_val;

		/* Defensive: Validate final result */
		if (isnan(result->data[i]) || isinf(result->data[i]))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("square root resulted in NaN or Infinity at index %d", i)));
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise power
 */
PG_FUNCTION_INFO_V1(vector_pow);
Datum
vector_pow(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	float8		exp;
	Vector	   *result;
	double		base;
	double		powered;
	int			i;

	CHECK_NARGS(2);
	v = PG_GETARG_VECTOR_P(0);
	exp = PG_GETARG_FLOAT8(1);

	/* Defensive: Check NULL vector */
	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute power of NULL vector")));

	/* Defensive: Validate vector dimension */
	if (v->dim <= 0 || v->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension %d", v->dim)));

	/* Defensive: Validate exponent */
	if (isnan(exp) || isinf(exp))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("exponent cannot be NaN or Infinity")));

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		base = (double) v->data[i];
		powered = pow(base, exp);

		/* Defensive: Check for overflow/underflow in pow */
		if (isinf(powered) || isnan(powered))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("power calculation resulted in overflow or NaN at index %d (base=%.15e, exp=%.15e)",
							i, base, exp)));

		result->data[i] = (float4) powered;

		/* Defensive: Validate result */
		if (isnan(result->data[i]) || isinf(result->data[i]))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("power resulted in NaN or Infinity at index %d", i)));
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise multiplication (Hadamard product)
 */
PG_FUNCTION_INFO_V1(vector_hadamard);
Datum
vector_hadamard(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	Vector	   *result;
	double		val_a;
	double		val_b;
	double		product;
	int			i;

	CHECK_NARGS(2);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);

	/* Defensive: Check NULL vectors */
	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute Hadamard product with NULL vectors")));

	/* Defensive: Validate vector dimensions */
	if (a->dim <= 0 || a->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector A dimension %d", a->dim)));

	if (b->dim <= 0 || b->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector B dimension %d", b->dim)));

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match: %d vs %d",
						a->dim, b->dim)));

	result = new_vector(a->dim);
	for (i = 0; i < a->dim; i++)
	{
		val_a = (double) a->data[i];
		val_b = (double) b->data[i];
		product = val_a * val_b;

		/* Defensive: Check for overflow in multiplication */
		if (isinf(product) || isnan(product))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("Hadamard product: overflow in multiplication at index %d (a=%.15e, b=%.15e)",
							i, val_a, val_b)));

		result->data[i] = (float4) product;

		/* Defensive: Validate result */
		if (isnan(result->data[i]) || isinf(result->data[i]))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("Hadamard product resulted in NaN or Infinity at index %d", i)));
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise division
 */
PG_FUNCTION_INFO_V1(vector_divide);
Datum
vector_divide(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	Vector	   *result;
	double		val_a;
	double		val_b;
	double		quotient;
	int			i;

	CHECK_NARGS(2);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);

	/* Defensive: Check NULL vectors */
	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot divide NULL vectors")));

	/* Defensive: Validate vector dimensions */
	if (a->dim <= 0 || a->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector A dimension %d", a->dim)));

	if (b->dim <= 0 || b->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector B dimension %d", b->dim)));

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match: %d vs %d",
						a->dim, b->dim)));

	result = new_vector(a->dim);
	for (i = 0; i < a->dim; i++)
	{
		val_a = (double) a->data[i];
		val_b = (double) b->data[i];

		/* Defensive: Check for division by zero */
		if (val_b == 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_DIVISION_BY_ZERO),
					 errmsg("division by zero at index %d", i)));

		/* Defensive: Check for NaN/Inf in operands */
		if (isnan(val_a) || isinf(val_a) || isnan(val_b) || isinf(val_b))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("cannot divide NaN or Infinity at index %d", i)));

		quotient = val_a / val_b;

		/* Defensive: Check for overflow in division */
		if (isinf(quotient) || isnan(quotient))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("division resulted in overflow or NaN at index %d (a=%.15e, b=%.15e)",
							i, val_a, val_b)));

		result->data[i] = (float4) quotient;

		/* Defensive: Validate result */
		if (isnan(result->data[i]) || isinf(result->data[i]))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("division resulted in NaN or Infinity at index %d", i)));
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector statistics
 */
PG_FUNCTION_INFO_V1(vector_mean);
Datum
vector_mean(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	double		sum = 0.0;
	int			i;

	CHECK_NARGS(1);
	v = PG_GETARG_VECTOR_P(0);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute mean of NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("cannot compute mean of vector with dimension %d",
						v->dim)));

	for (i = 0; i < v->dim; i++)
		sum += v->data[i];

	PG_RETURN_FLOAT8(sum / v->dim);
}

PG_FUNCTION_INFO_V1(vector_variance);
Datum
vector_variance(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	double		mean = 0.0;
	double		variance = 0.0;
	double		diff;
	int			i;

	CHECK_NARGS(1);
	v = PG_GETARG_VECTOR_P(0);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute variance of NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("cannot compute variance of vector with dimension %d",
						v->dim)));

	for (i = 0; i < v->dim; i++)
		mean += v->data[i];
	mean /= v->dim;

	for (i = 0; i < v->dim; i++)
	{
		diff = v->data[i] - mean;
		variance += diff * diff;
	}

	PG_RETURN_FLOAT8(variance / v->dim);
}

PG_FUNCTION_INFO_V1(vector_stddev);
Datum
vector_stddev(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	double		mean = 0.0;
	double		variance = 0.0;
	double		diff;
	int			i;

	CHECK_NARGS(1);
	v = PG_GETARG_VECTOR_P(0);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute standard deviation of NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("cannot compute standard deviation of vector with dimension %d",
						v->dim)));

	for (i = 0; i < v->dim; i++)
		mean += v->data[i];
	mean /= v->dim;

	for (i = 0; i < v->dim; i++)
	{
		diff = v->data[i] - mean;
		variance += diff * diff;
	}

	PG_RETURN_FLOAT8(sqrt(variance / v->dim));
}

PG_FUNCTION_INFO_V1(vector_min);
Datum
vector_min(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	float4		min_val;
	int			i;

	CHECK_NARGS(1);
	v = PG_GETARG_VECTOR_P(0);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot find minimum of NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cannot find minimum of empty vector")));

	min_val = v->data[0];
	for (i = 1; i < v->dim; i++)
		if (v->data[i] < min_val)
			min_val = v->data[i];

	PG_RETURN_FLOAT4(min_val);
}

PG_FUNCTION_INFO_V1(vector_max);
Datum
vector_max(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	float4		max_val;
	int			i;

	CHECK_NARGS(1);
	v = PG_GETARG_VECTOR_P(0);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot find maximum of NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cannot find maximum of empty vector")));

	max_val = v->data[0];
	for (i = 1; i < v->dim; i++)
		if (v->data[i] > max_val)
			max_val = v->data[i];

	PG_RETURN_FLOAT4(max_val);
}

PG_FUNCTION_INFO_V1(vector_sum);
Datum
vector_sum(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	double		sum = 0.0;
	double		c = 0.0;
	double		val;
	double		y;
	double		t;
	int			i;

	CHECK_NARGS(1);
	v = PG_GETARG_VECTOR_P(0);

	/* Defensive: Check NULL vector */
	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute sum of NULL vector")));

	/* Defensive: Validate vector dimension */
	if (v->dim <= 0 || v->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension %d", v->dim)));

	/* Defensive: Use Kahan summation for numerical stability */
	for (i = 0; i < v->dim; i++)
	{
		val = (double) v->data[i];

		/* Defensive: Check for NaN/Inf */
		if (isnan(val) || isinf(val))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("cannot sum vector containing NaN or Infinity at index %d", i)));

		/* Kahan summation */
		y = val - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;

		/* Defensive: Check for overflow in accumulation */
		if (isinf(sum) || isnan(sum))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("sum calculation resulted in overflow at index %d", i)));
	}

	/* Defensive: Validate result */
	if (isnan(sum) || isinf(sum))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("vector sum resulted in NaN or Infinity")));

	PG_RETURN_FLOAT8(sum);
}

/*
 * Vector comparison
 */
PG_FUNCTION_INFO_V1(vector_eq);
Datum
vector_eq(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	int			i;

	CHECK_NARGS(2);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);

	/* Handle NULL vectors */
	if (a == NULL && b == NULL)
		PG_RETURN_BOOL(true);
	if (a == NULL || b == NULL)
		PG_RETURN_BOOL(false);

	if (a->dim != b->dim)
		PG_RETURN_BOOL(false);

	/* Use epsilon comparison for float equality */
	for (i = 0; i < a->dim; i++)
	{
		if (isnan(a->data[i]) || isnan(b->data[i]))
		{
			/* NaN != NaN */
			if (isnan(a->data[i]) != isnan(b->data[i]))
				PG_RETURN_BOOL(false);
		}
		else if (fabs(a->data[i] - b->data[i]) > 1e-6)
			PG_RETURN_BOOL(false);
	}

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(vector_ne);
Datum
vector_ne(PG_FUNCTION_ARGS)
{
	Datum		result;

	CHECK_NARGS(2);
	result = DirectFunctionCall2(vector_eq, PG_GETARG_DATUM(0),
								  PG_GETARG_DATUM(1));
	return result ? BoolGetDatum(false) : BoolGetDatum(true);
}

/*
 * vector_hash(vector) -> uint32
 * Hash function for vector type to support hash joins and hash-based operations
 */
PG_FUNCTION_INFO_V1(vector_hash);
Datum
vector_hash(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	uint32		hash = 5381;
	int32		tmp;
	int			stride;
	int			i;

	CHECK_NARGS(1);
	v = PG_GETARG_VECTOR_P(0);

	if (v == NULL)
		PG_RETURN_UINT32(0);

	/* Hash dimension first */
	hash = ((hash << 5) + hash) + (uint32) v->dim;

	/* Hash vector data (use first 16 elements for performance) */
	for (i = 0; i < v->dim && i < 16; i++)
	{
		/* Convert float to int for hashing */
		tmp = (int32) (v->data[i] * 1000000.0f);
		hash = ((hash << 5) + hash) + (uint32) tmp;
	}

	/* If vector is longer, hash remaining elements with stride */
	if (v->dim > 16)
	{
		stride = v->dim / 16;
		for (i = 16; i < v->dim; i += stride)
		{
			tmp = (int32) (v->data[i] * 1000000.0f);
			hash = ((hash << 5) + hash) + (uint32) tmp;
		}
	}

	PG_RETURN_UINT32(hash);
}

/*
 * Vector clipping (element-wise min/max)
 */
PG_FUNCTION_INFO_V1(vector_clip);
Datum
vector_clip(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	float4		min_val;
	float4		max_val;
	Vector	   *result;
	int			i;

	CHECK_NARGS(3);
	v = PG_GETARG_VECTOR_P(0);
	min_val = PG_GETARG_FLOAT4(1);
	max_val = PG_GETARG_FLOAT4(2);

	if (min_val > max_val)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("min_val must be <= max_val")));

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		if (v->data[i] < min_val)
			result->data[i] = min_val;
		else if (v->data[i] > max_val)
			result->data[i] = max_val;
		else
			result->data[i] = v->data[i];
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector standardization (z-score normalization)
 */
PG_FUNCTION_INFO_V1(vector_standardize);
Datum
vector_standardize(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	Vector	   *result;
	double		mean = 0.0;
	double		stddev = 0.0;
	double		diff;
	int			i;

	CHECK_NARGS(1);
	v = PG_GETARG_VECTOR_P(0);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot standardize NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("cannot standardize vector with dimension %d",
						v->dim)));

	/* Calculate mean */
	for (i = 0; i < v->dim; i++)
		mean += v->data[i];
	mean /= v->dim;

	/* Calculate standard deviation */
	for (i = 0; i < v->dim; i++)
	{
		diff = v->data[i] - mean;
		stddev += diff * diff;
	}
	stddev = sqrt(stddev / v->dim);

	/* Standardize */
	result = new_vector(v->dim);
	if (stddev > 0.0)
	{
		for (i = 0; i < v->dim; i++)
			result->data[i] = (v->data[i] - mean) / stddev;
	}
	else
	{
		/* All values are the same - set to zero */
		memset(result->data, 0, sizeof(float4) * v->dim);
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector min-max normalization
 */
PG_FUNCTION_INFO_V1(vector_minmax_normalize);
Datum
vector_minmax_normalize(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	Vector	   *result;
	float4		min_val;
	float4		max_val;
	float4		range;
	int			i;

	CHECK_NARGS(1);
	v = PG_GETARG_VECTOR_P(0);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot normalize NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cannot normalize empty vector")));

	/* Find min and max */
	min_val = max_val = v->data[0];
	for (i = 1; i < v->dim; i++)
	{
		if (v->data[i] < min_val)
			min_val = v->data[i];
		if (v->data[i] > max_val)
			max_val = v->data[i];
	}

	range = max_val - min_val;
	result = new_vector(v->dim);

	if (range > 0.0)
	{
		for (i = 0; i < v->dim; i++)
			result->data[i] = (v->data[i] - min_val) / range;
	}
	else
	{
		/* All values are the same */
		for (i = 0; i < v->dim; i++)
			result->data[i] = 0.5;	/* Middle of [0,1] range */
	}

	PG_RETURN_VECTOR_P(result);
}
