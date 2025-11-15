/*-------------------------------------------------------------------------
 *
 * vector_sparse.c
 *	  Sparse vector operations optimized for sparse data
 *
 * This file implements distance metrics and arithmetic operations for
 * sparse vectors (vecmap type). All operations are optimized to only
 * iterate over non-zero entries, providing O(nnz) complexity instead
 * of O(dim) for dense vectors.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  contrib/neurondb/vector_sparse.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include <math.h>
#include <float.h>

/* Helper: Check dimensions match */
static inline void
check_vecmap_dimensions(const VectorMap *a, const VectorMap *b)
{
	/* Defensive: Check NULL inputs */
	if (a == NULL || b == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("cannot compute distance with NULL vectors")));

	/* Defensive: Validate dimensions */
	if (a->total_dim <= 0 || a->total_dim > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid vector A dimension %d", a->total_dim)));

	if (b->total_dim <= 0 || b->total_dim > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid vector B dimension %d", b->total_dim)));

	if (a->total_dim != b->total_dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("sparse vector dimensions must match: %d vs %d",
					a->total_dim,
					b->total_dim)));
}

/*
 * Sparse L2 distance: optimized for sparse vectors
 * Uses merge-like algorithm to only process non-zero entries
 */
PG_FUNCTION_INFO_V1(vecmap_l2_distance);
Datum
vecmap_l2_distance(PG_FUNCTION_ARGS)
{
	VectorMap *a;
	VectorMap *b;
	int32 *a_indices;
	float4 *a_values;
	int32 *b_indices;
	float4 *b_values;
	double sum = 0.0;
	double c = 0.0; /* Kahan summation correction */
	int i;
	int j;
	int32 idx_a;
	int32 idx_b;
	double diff;
	double y;
	double t;

	CHECK_NARGS(2);
	a = (VectorMap *)PG_GETARG_POINTER(0);
	b = (VectorMap *)PG_GETARG_POINTER(1);

	/* Defensive: Check NULL inputs */
	if (a == NULL || b == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("cannot compute distance with NULL vectors")));

	check_vecmap_dimensions(a, b);

	/* Defensive: Validate nnz */
	if (a->nnz < 0 || a->nnz > 1000000 || b->nnz < 0 || b->nnz > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid nnz values: %d, %d", a->nnz, b->nnz)));

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	/* Merge-like algorithm: process both sparse vectors simultaneously */
	i = 0;
	j = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			/* Only b has value at this index */
			idx_b = b_indices[j];
			diff = 0.0 - (double)b_values[j];
			y = (diff * diff) - c;
			t = sum + y;
			c = (t - sum) - y;
			sum = t;
			j++;
		}
		else if (j >= b->nnz)
		{
			/* Only a has value at this index */
			idx_a = a_indices[i];
			diff = (double)a_values[i] - 0.0;
			y = (diff * diff) - c;
			t = sum + y;
			c = (t - sum) - y;
			sum = t;
			i++;
		}
		else
		{
			idx_a = a_indices[i];
			idx_b = b_indices[j];

			if (idx_a < idx_b)
			{
				/* Only a has value at this index */
				diff = (double)a_values[i] - 0.0;
				y = (diff * diff) - c;
				t = sum + y;
				c = (t - sum) - y;
				sum = t;
				i++;
			}
			else if (idx_a > idx_b)
			{
				/* Only b has value at this index */
				diff = 0.0 - (double)b_values[j];
				y = (diff * diff) - c;
				t = sum + y;
				c = (t - sum) - y;
				sum = t;
				j++;
			}
			else
			{
				/* Both have values at this index */
				diff = (double)a_values[i] - (double)b_values[j];
				y = (diff * diff) - c;
				t = sum + y;
				c = (t - sum) - y;
				sum = t;
				i++;
				j++;
			}
		}
	}

	PG_RETURN_FLOAT4((float4)sqrt(sum));
}

/*
 * Sparse inner product: optimized for sparse vectors
 */
PG_FUNCTION_INFO_V1(vecmap_inner_product);
Datum
vecmap_inner_product(PG_FUNCTION_ARGS)
{
	VectorMap *a;
	VectorMap *b;
	int32 *a_indices;
	float4 *a_values;
	int32 *b_indices;
	float4 *b_values;
	double sum = 0.0;
	int i;
	int j;

	CHECK_NARGS(2);
	a = (VectorMap *)PG_GETARG_POINTER(0);
	b = (VectorMap *)PG_GETARG_POINTER(1);

	/* Defensive: Check NULL inputs */
	if (a == NULL || b == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("cannot compute inner product with NULL vectors")));

	check_vecmap_dimensions(a, b);

	/* Defensive: Validate nnz */
	if (a->nnz < 0 || a->nnz > 1000000 || b->nnz < 0 || b->nnz > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid nnz values: %d, %d", a->nnz, b->nnz)));

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	/* Merge-like algorithm: only multiply when both have non-zero values */
	i = 0;
	j = 0;

	while (i < a->nnz && j < b->nnz)
	{
		if (a_indices[i] < b_indices[j])
		{
			i++;
		}
		else if (a_indices[i] > b_indices[j])
		{
			j++;
		}
		else
		{
			/* Both have values at this index */
			sum += (double)a_values[i] * (double)b_values[j];
			i++;
			j++;
		}
	}

	PG_RETURN_FLOAT4((float4)sum);
}

/*
 * Sparse cosine distance: optimized for sparse vectors
 */
PG_FUNCTION_INFO_V1(vecmap_cosine_distance);
Datum
vecmap_cosine_distance(PG_FUNCTION_ARGS)
{
	VectorMap *a;
	VectorMap *b;
	int32 *a_indices;
	float4 *a_values;
	int32 *b_indices;
	float4 *b_values;
	double dot = 0.0;
	double norm_a = 0.0;
	double norm_b = 0.0;
	int i;
	int j;
	int32 idx_a;
	int32 idx_b;

	CHECK_NARGS(2);
	a = (VectorMap *)PG_GETARG_POINTER(0);
	b = (VectorMap *)PG_GETARG_POINTER(1);

	/* Defensive: Check NULL inputs */
	if (a == NULL || b == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("cannot compute cosine distance with NULL vectors")));

	check_vecmap_dimensions(a, b);

	/* Defensive: Validate nnz */
	if (a->nnz < 0 || a->nnz > 1000000 || b->nnz < 0 || b->nnz > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid nnz values: %d, %d", a->nnz, b->nnz)));

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	/* Compute dot product and norms simultaneously */
	i = 0;
	j = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			idx_b = b_indices[j];
			norm_b += (double)b_values[j] * (double)b_values[j];
			j++;
		}
		else if (j >= b->nnz)
		{
			idx_a = a_indices[i];
			norm_a += (double)a_values[i] * (double)a_values[i];
			i++;
		}
		else
		{
			idx_a = a_indices[i];
			idx_b = b_indices[j];

			if (idx_a < idx_b)
			{
				norm_a += (double)a_values[i] * (double)a_values[i];
				i++;
			}
			else if (idx_a > idx_b)
			{
				norm_b += (double)b_values[j] * (double)b_values[j];
				j++;
			}
			else
			{
				/* Both have values at this index */
				double va = (double)a_values[i];
				double vb = (double)b_values[j];
				dot += va * vb;
				norm_a += va * va;
				norm_b += vb * vb;
				i++;
				j++;
			}
		}
	}

	/* Handle zero norms */
	if (norm_a == 0.0 || norm_b == 0.0)
		PG_RETURN_FLOAT4(1.0f);

	PG_RETURN_FLOAT4((float4)(1.0 - (dot / (sqrt(norm_a) * sqrt(norm_b)))));
}

/*
 * Sparse L1 (Manhattan) distance: optimized for sparse vectors
 */
PG_FUNCTION_INFO_V1(vecmap_l1_distance);
Datum
vecmap_l1_distance(PG_FUNCTION_ARGS)
{
	VectorMap *a;
	VectorMap *b;
	int32 *a_indices;
	float4 *a_values;
	int32 *b_indices;
	float4 *b_values;
	double sum = 0.0;
	int i;
	int j;

	CHECK_NARGS(2);
	a = (VectorMap *)PG_GETARG_POINTER(0);
	b = (VectorMap *)PG_GETARG_POINTER(1);

	/* Defensive: Check NULL inputs */
	if (a == NULL || b == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("cannot compute L1 distance with NULL vectors")));

	check_vecmap_dimensions(a, b);

	/* Defensive: Validate nnz */
	if (a->nnz < 0 || a->nnz > 1000000 || b->nnz < 0 || b->nnz > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid nnz values: %d, %d", a->nnz, b->nnz)));

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	/* Merge-like algorithm */
	i = 0;
	j = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			/* Only b has value */
			sum += fabs((double)b_values[j]);
			j++;
		}
		else if (j >= b->nnz)
		{
			/* Only a has value */
			sum += fabs((double)a_values[i]);
			i++;
		}
		else
		{
			if (a_indices[i] < b_indices[j])
			{
				sum += fabs((double)a_values[i]);
				i++;
			}
			else if (a_indices[i] > b_indices[j])
			{
				sum += fabs((double)b_values[j]);
				j++;
			}
			else
			{
				/* Both have values */
				sum += fabs((double)a_values[i] - (double)b_values[j]);
				i++;
				j++;
			}
		}
	}

	PG_RETURN_FLOAT4((float4)sum);
}

/*
 * Sparse vector addition: creates new sparse vector
 */
PG_FUNCTION_INFO_V1(vecmap_add);
Datum
vecmap_add(PG_FUNCTION_ARGS)
{
	VectorMap *a;
	VectorMap *b;
	int32 *a_indices;
	float4 *a_values;
	int32 *b_indices;
	float4 *b_values;
	VectorMap *result;
	int32 *result_indices;
	float4 *result_values;
	int32 max_nnz;
	int32 result_nnz;
	int i;
	int j;
	int k;
	int size;

	CHECK_NARGS(2);
	a = (VectorMap *)PG_GETARG_POINTER(0);
	b = (VectorMap *)PG_GETARG_POINTER(1);

	/* Defensive: Check NULL inputs */
	if (a == NULL || b == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("cannot add NULL vectors")));

	check_vecmap_dimensions(a, b);

	/* Defensive: Validate nnz */
	if (a->nnz < 0 || a->nnz > 1000000 || b->nnz < 0 || b->nnz > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid nnz values: %d, %d", a->nnz, b->nnz)));

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	/* Maximum possible nnz is sum of both (if no overlap) */
	max_nnz = a->nnz + b->nnz;

	/* Defensive: Check for overflow */
	if (max_nnz < 0 || max_nnz > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				errmsg("result nnz %d exceeds maximum", max_nnz)));

	result_indices = (int32 *)palloc(sizeof(int32) * max_nnz);
	result_values = (float4 *)palloc(sizeof(float4) * max_nnz);

	/* Defensive: Validate allocation */
	if (result_indices == NULL || result_values == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate vecmap arrays")));

	/* Merge-like addition */
	i = 0;
	j = 0;
	k = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			result_indices[k] = b_indices[j];
			result_values[k] = b_values[j];
			j++;
			k++;
		}
		else if (j >= b->nnz)
		{
			result_indices[k] = a_indices[i];
			result_values[k] = a_values[i];
			i++;
			k++;
		}
		else
		{
			if (a_indices[i] < b_indices[j])
			{
				result_indices[k] = a_indices[i];
				result_values[k] = a_values[i];
				i++;
				k++;
			}
			else if (a_indices[i] > b_indices[j])
			{
				result_indices[k] = b_indices[j];
				result_values[k] = b_values[j];
				j++;
				k++;
			}
			else
			{
				/* Both have values - add them */
				float4 sum_val = a_values[i] + b_values[j];
				if (fabs(sum_val) > 1e-10f) /* Skip near-zero results */
				{
					result_indices[k] = a_indices[i];
					result_values[k] = sum_val;
					k++;
				}
				i++;
				j++;
			}
		}
	}

	result_nnz = k;

	/* Allocate result */
	size = sizeof(VectorMap) + sizeof(int32) * result_nnz +
		sizeof(float4) * result_nnz;

	/* Defensive: Validate size calculation */
	if (size < (int)sizeof(VectorMap) || size > 100000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("vecmap size %d out of valid range", size)));

	result = (VectorMap *)palloc0(size);

	/* Defensive: Validate allocation */
	if (result == NULL)
	{
		pfree(result_indices);
		pfree(result_values);
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate vecmap")));
	}

	SET_VARSIZE(result, size);

	result->total_dim = a->total_dim;
	result->nnz = result_nnz;

	memcpy(VECMAP_INDICES(result), result_indices, sizeof(int32) * result_nnz);
	memcpy(VECMAP_VALUES(result), result_values, sizeof(float4) * result_nnz);

	pfree(result_indices);
	pfree(result_values);

	PG_RETURN_POINTER(result);
}

/*
 * Sparse vector subtraction: creates new sparse vector
 */
PG_FUNCTION_INFO_V1(vecmap_sub);
Datum
vecmap_sub(PG_FUNCTION_ARGS)
{
	VectorMap *a;
	VectorMap *b;
	int32 *a_indices;
	float4 *a_values;
	int32 *b_indices;
	float4 *b_values;
	VectorMap *result;
	int32 *result_indices;
	float4 *result_values;
	int32 max_nnz;
	int32 result_nnz;
	int i;
	int j;
	int k;
	int size;

	CHECK_NARGS(2);
	a = (VectorMap *)PG_GETARG_POINTER(0);
	b = (VectorMap *)PG_GETARG_POINTER(1);

	/* Defensive: Check NULL inputs */
	if (a == NULL || b == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("cannot subtract NULL vectors")));

	check_vecmap_dimensions(a, b);

	/* Defensive: Validate nnz */
	if (a->nnz < 0 || a->nnz > 1000000 || b->nnz < 0 || b->nnz > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid nnz values: %d, %d", a->nnz, b->nnz)));

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	max_nnz = a->nnz + b->nnz;

	/* Defensive: Check for overflow */
	if (max_nnz < 0 || max_nnz > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				errmsg("result nnz %d exceeds maximum", max_nnz)));

	result_indices = (int32 *)palloc(sizeof(int32) * max_nnz);
	result_values = (float4 *)palloc(sizeof(float4) * max_nnz);

	/* Defensive: Validate allocation */
	if (result_indices == NULL || result_values == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
				errmsg("failed to allocate vecmap arrays")));

	/* Merge-like subtraction */
	i = 0;
	j = 0;
	k = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			result_indices[k] = b_indices[j];
			result_values[k] = -b_values[j]; /* Negate b */
			j++;
			k++;
		}
		else if (j >= b->nnz)
		{
			result_indices[k] = a_indices[i];
			result_values[k] = a_values[i];
			i++;
			k++;
		}
		else
		{
			if (a_indices[i] < b_indices[j])
			{
				result_indices[k] = a_indices[i];
				result_values[k] = a_values[i];
				i++;
				k++;
			}
			else if (a_indices[i] > b_indices[j])
			{
				result_indices[k] = b_indices[j];
				result_values[k] = -b_values[j];
				j++;
				k++;
			}
			else
			{
				/* Both have values - subtract */
				float4 diff_val = a_values[i] - b_values[j];
				if (fabs(diff_val) > 1e-10f)
				{
					result_indices[k] = a_indices[i];
					result_values[k] = diff_val;
					k++;
				}
				i++;
				j++;
			}
		}
	}

	result_nnz = k;

	size = sizeof(VectorMap) + sizeof(int32) * result_nnz +
		sizeof(float4) * result_nnz;
	result = (VectorMap *)palloc0(size);
	SET_VARSIZE(result, size);

	result->total_dim = a->total_dim;
	result->nnz = result_nnz;

	memcpy(VECMAP_INDICES(result), result_indices, sizeof(int32) * result_nnz);
	memcpy(VECMAP_VALUES(result), result_values, sizeof(float4) * result_nnz);

	pfree(result_indices);
	pfree(result_values);

	PG_RETURN_POINTER(result);
}

/*
 * Sparse vector scalar multiplication: creates new sparse vector
 */
PG_FUNCTION_INFO_V1(vecmap_mul_scalar);
Datum
vecmap_mul_scalar(PG_FUNCTION_ARGS)
{
	VectorMap *a;
	VectorMap *result;
	float4 scalar;
	int32 *a_indices;
	float4 *a_values;
	int32 *result_indices;
	float4 *result_values;
	int i;
	int size;

	CHECK_NARGS(2);
	a = (VectorMap *)PG_GETARG_POINTER(0);
	scalar = PG_GETARG_FLOAT4(1);

	/* Defensive: Check NULL input */
	if (a == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("cannot multiply NULL vector")));

	/* Defensive: Check for NaN/Inf scalar */
	if (isnan(scalar) || isinf(scalar))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("scalar cannot be NaN or Infinity")));

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);

	/* Defensive: Validate pointers */
	if (a_indices == NULL || a_values == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_CORRUPTED),
				errmsg("vecmap has NULL indices or values")));

	/* Defensive: Validate nnz */
	if (a->nnz < 0 || a->nnz > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid nnz value: %d", a->nnz)));

	/* Count non-zero results */
	result_nnz = 0;
	for (i = 0; i < a->nnz; i++)
	{
		if (fabs(a_values[i] * scalar) > 1e-10f)
			result_nnz++;
	}

	size = sizeof(VectorMap) + sizeof(int32) * result_nnz +
		sizeof(float4) * result_nnz;
	result = (VectorMap *)palloc0(size);
	SET_VARSIZE(result, size);

	result->total_dim = a->total_dim;
	result->nnz = result_nnz;

	result_indices = VECMAP_INDICES(result);
	result_values = VECMAP_VALUES(result);

	{
		int k = 0;
		for (i = 0; i < a->nnz; i++)
		{
			float4 scaled = a_values[i] * scalar;
			if (fabs(scaled) > 1e-10f)
			{
				result_indices[k] = a_indices[i];
				result_values[k] = scaled;
				k++;
			}
		}
	}

	PG_RETURN_POINTER(result);
}

/*
 * Sparse vector norm (L2): compute ||v||
 */
PG_FUNCTION_INFO_V1(vecmap_norm);
Datum
vecmap_norm(PG_FUNCTION_ARGS)
{
	VectorMap *a;
	float4 *values;
	double sum = 0.0;
	int i;

	CHECK_NARGS(1);
	a = (VectorMap *)PG_GETARG_POINTER(0);

	/* Defensive: Check NULL input */
	if (a == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("cannot compute norm of NULL vector")));

	/* Defensive: Validate nnz */
	if (a->nnz < 0 || a->nnz > 1000000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid nnz value: %d", a->nnz)));

	values = VECMAP_VALUES(a);

	/* Defensive: Validate pointer */
	if (values == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_CORRUPTED),
				errmsg("vecmap has NULL values")));

	for (i = 0; i < a->nnz; i++)
		sum += (double)values[i] * (double)values[i];

	PG_RETURN_FLOAT4((float4)sqrt(sum));
}

