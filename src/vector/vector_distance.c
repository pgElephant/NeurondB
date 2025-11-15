/*-------------------------------------------------------------------------
 *
 * distance.c
 *		Distance and similarity metric functions for vectors
 *
 * This file implements comprehensive distance metrics including L2
 * (Euclidean), cosine, inner product, L1 (Manhattan), Hamming,
 * Chebyshev, and Minkowski distances. All functions use Kahan
 * summation for numerical stability where appropriate.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  contrib/neurondb/distance.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include <math.h>
#include <float.h>

static inline void
check_dimensions(const Vector *a, const Vector *b)
{
	/* Defensive: Check NULL pointers first */
	if (a == NULL || b == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("cannot compute distance with NULL vectors")));

	/* Defensive: Validate vector structure integrity */
	if (a->dim <= 0 || a->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid vector dimension %d (must be between 1 and %d)",
					a->dim,
					VECTOR_MAX_DIM)));

	if (b->dim <= 0 || b->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("invalid vector dimension %d (must be between 1 and %d)",
					b->dim,
					VECTOR_MAX_DIM)));

	/* Defensive: Check dimension match */
	if (a->dim != b->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("vector dimensions must match: %d vs %d",
					a->dim,
					b->dim)));

	/* Defensive: Validate vector size matches dimension */
	if (VARSIZE_ANY(a) < (int)offsetof(Vector, data) + (int)(sizeof(float4) * a->dim))
		ereport(ERROR,
			(errcode(ERRCODE_DATA_CORRUPTED),
				errmsg("vector A size %d does not match dimension %d",
					VARSIZE_ANY(a),
					a->dim)));

	if (VARSIZE_ANY(b) < (int)offsetof(Vector, data) + (int)(sizeof(float4) * b->dim))
		ereport(ERROR,
			(errcode(ERRCODE_DATA_CORRUPTED),
				errmsg("vector B size %d does not match dimension %d",
					VARSIZE_ANY(b),
					b->dim)));

	/* Defensive: Check for NaN/Inf in vectors */
	{
		int i;
		for (i = 0; i < a->dim; i++)
		{
			if (isnan(a->data[i]) || isinf(a->data[i]))
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("vector A contains NaN or Infinity at index %d", i)));
		}
		for (i = 0; i < b->dim; i++)
		{
			if (isnan(b->data[i]) || isinf(b->data[i]))
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("vector B contains NaN or Infinity at index %d", i)));
		}
	}
}

/* L2 (Euclidean) distance, uses Kahan summation for numerical stability */
float4
l2_distance(Vector *a, Vector *b)
{
	double		sum = 0.0;
	double		c = 0.0;
	double		diff;
	double		squared_diff;
	double		y;
	double		t;
	int			i;
	float4		result;
	double		max_squared_diff = 0.0;

	check_dimensions(a, b);

	/* Defensive: Use Kahan summation with overflow protection */
	for (i = 0; i < a->dim; i++)
	{
		diff = (double) a->data[i] - (double) b->data[i];
		squared_diff = diff * diff;

		/* Defensive: Check for overflow in squaring */
		if (isinf(squared_diff) || isnan(squared_diff))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("L2 distance: overflow in squared difference at index %d", i)));

		/* Track maximum for overflow detection */
		if (squared_diff > max_squared_diff)
			max_squared_diff = squared_diff;

		/* Kahan summation */
		y = squared_diff - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;

		/* Defensive: Check for overflow in accumulation */
		if (isinf(sum) || isnan(sum))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("L2 distance: overflow in accumulation at index %d", i)));
	}

	/* Defensive: Validate sum before sqrt */
	if (sum < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("L2 distance: negative sum (numerical error)")));

	result = (float4) sqrt(sum);

	/* Defensive: Check for overflow/underflow in result */
	if (isnan(result) || isinf(result))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("L2 distance calculation resulted in NaN or Infinity (sum=%.15e)",
						sum)));

	/* Defensive: Validate result is finite and non-negative */
	if (result < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("L2 distance: negative result (numerical error)")));

	return result;
}

PG_FUNCTION_INFO_V1(vector_l2_distance);
Datum
vector_l2_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;

	CHECK_NARGS(2);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);
	PG_RETURN_FLOAT4(l2_distance(a, b));
}

/* Inner product distance, negative for correct ordering for similarity */
float4
inner_product_distance(Vector *a, Vector *b)
{
	double		sum = 0.0;
	double		c = 0.0;
	double		product;
	double		y;
	double		t;
	int			i;
	float4		result;

	check_dimensions(a, b);

	/* Defensive: Use Kahan summation for numerical stability */
	for (i = 0; i < a->dim; i++)
	{
		product = (double) a->data[i] * (double) b->data[i];

		/* Defensive: Check for overflow in multiplication */
		if (isinf(product) || isnan(product))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("inner product: overflow in multiplication at index %d", i)));

		/* Kahan summation */
		y = product - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;

		/* Defensive: Check for overflow in accumulation */
		if (isinf(sum) || isnan(sum))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("inner product: overflow in accumulation at index %d", i)));
	}

	result = (float4) (-sum);

	/* Defensive: Validate result */
	if (isnan(result) || isinf(result))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("inner product calculation resulted in NaN or Infinity")));

	return result;
}

PG_FUNCTION_INFO_V1(vector_inner_product);
Datum
vector_inner_product(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;

	CHECK_NARGS(2);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);
	PG_RETURN_FLOAT4(inner_product_distance(a, b));
}

/* Cosine distance: 1.0 - (dot(a,b) / (||a||*||b||)), returns 1.0 if norm is zero */
float4
cosine_distance(Vector *a, Vector *b)
{
	double		dot = 0.0;
	double		norm_a = 0.0;
	double		norm_b = 0.0;
	double		dot_c = 0.0;
	double		norm_a_c = 0.0;
	double		norm_b_c = 0.0;
	double		va;
	double		vb;
	double		product;
	double		va_sq;
	double		vb_sq;
	double		y;
	double		t;
	double		sqrt_norm_a;
	double		sqrt_norm_b;
	double		denominator;
	double		cosine_sim;
	int			i;
	float4		result;

	check_dimensions(a, b);

	/* Defensive: Compute dot(a, b), ||a||^2, ||b||^2 with Kahan summation */
	for (i = 0; i < a->dim; i++)
	{
		va = (double) a->data[i];
		vb = (double) b->data[i];
		product = va * vb;
		va_sq = va * va;
		vb_sq = vb * vb;

		/* Defensive: Check for overflow in multiplication */
		if (isinf(product) || isnan(product) || isinf(va_sq) || isnan(va_sq) ||
			isinf(vb_sq) || isnan(vb_sq))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("cosine distance: overflow in computation at index %d", i)));

		/* Kahan summation for dot product */
		y = product - dot_c;
		t = dot + y;
		dot_c = (t - dot) - y;
		dot = t;

		/* Kahan summation for norm_a */
		y = va_sq - norm_a_c;
		t = norm_a + y;
		norm_a_c = (t - norm_a) - y;
		norm_a = t;

		/* Kahan summation for norm_b */
		y = vb_sq - norm_b_c;
		t = norm_b + y;
		norm_b_c = (t - norm_b) - y;
		norm_b = t;

		/* Defensive: Check for overflow in accumulation */
		if (isinf(dot) || isnan(dot) || isinf(norm_a) || isnan(norm_a) ||
			isinf(norm_b) || isnan(norm_b))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("cosine distance: overflow in accumulation at index %d", i)));
	}

	/* Defensive: Validate norms are non-negative */
	if (norm_a < 0.0 || norm_b < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("cosine distance: negative norm (numerical error)")));

	/* Handle zero norms to prevent divide by zero */
	if (norm_a == 0.0 || norm_b == 0.0)
		return 1.0;

	/* Defensive: Compute square roots with validation */
	sqrt_norm_a = sqrt(norm_a);
	sqrt_norm_b = sqrt(norm_b);

	if (isnan(sqrt_norm_a) || isinf(sqrt_norm_a) || isnan(sqrt_norm_b) ||
		isinf(sqrt_norm_b))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("cosine distance: invalid square root (norm_a=%.15e, norm_b=%.15e)",
						norm_a, norm_b)));

	/* Defensive: Check denominator */
	denominator = sqrt_norm_a * sqrt_norm_b;
	if (denominator == 0.0 || isnan(denominator) || isinf(denominator))
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("cosine distance: zero or invalid denominator")));

	cosine_sim = dot / denominator;

	/* Defensive: Validate cosine similarity is in valid range [-1, 1] */
	if (cosine_sim < -1.0)
		cosine_sim = -1.0;
	else if (cosine_sim > 1.0)
		cosine_sim = 1.0;

	result = (float4) (1.0 - cosine_sim);

	/* Defensive: Validate result */
	if (isnan(result) || isinf(result))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("cosine distance calculation resulted in NaN or Infinity (dot=%.15e, denominator=%.15e)",
						dot, denominator)));

	/* Defensive: Result should be in [0, 2] range */
	if (result < 0.0 || result > 2.0)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("cosine distance: result %.15e out of valid range [0, 2]",
						result)));

	return result;
}

PG_FUNCTION_INFO_V1(vector_cosine_distance);
Datum
vector_cosine_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;

	CHECK_NARGS(2);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);
	PG_RETURN_FLOAT4(cosine_distance(a, b));
}

/* L1 (Manhattan) distance, standard implementation */
float4
l1_distance(Vector *a, Vector *b)
{
	double		sum = 0.0;
	double		c = 0.0;
	double		diff;
	double		abs_diff;
	double		y;
	double		t;
	int			i;
	float4		result;

	check_dimensions(a, b);

	/* Defensive: Use Kahan summation for numerical stability */
	for (i = 0; i < a->dim; i++)
	{
		diff = (double) a->data[i] - (double) b->data[i];
		abs_diff = fabs(diff);

		/* Defensive: Check for overflow */
		if (isinf(abs_diff) || isnan(abs_diff))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("L1 distance: overflow in absolute difference at index %d", i)));

		/* Kahan summation */
		y = abs_diff - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;

		/* Defensive: Check for overflow in accumulation */
		if (isinf(sum) || isnan(sum))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("L1 distance: overflow in accumulation at index %d", i)));
	}

	result = (float4) sum;

	/* Defensive: Validate result */
	if (isnan(result) || isinf(result))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("L1 distance calculation resulted in NaN or Infinity")));

	/* Defensive: Result should be non-negative */
	if (result < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("L1 distance: negative result (numerical error)")));

	return result;
}

PG_FUNCTION_INFO_V1(vector_l1_distance);
Datum
vector_l1_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;

	CHECK_NARGS(2);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);
	PG_RETURN_FLOAT4(l1_distance(a, b));
}

/* Hamming distance: counts differing coordinates */
PG_FUNCTION_INFO_V1(vector_hamming_distance);
Datum
vector_hamming_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	int			count = 0;
	int			i;

	CHECK_NARGS(2);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		/* Using "!=" direct numerical for float equality */
		if (a->data[i] != b->data[i])
			count++;
	}

	PG_RETURN_INT32(count);
}

/* Chebyshev distance: maximum coordinate-wise difference */
PG_FUNCTION_INFO_V1(vector_chebyshev_distance);
Datum
vector_chebyshev_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	double		max_diff = 0.0;
	double		diff;
	int			i;
	bool		found_valid = false;

	CHECK_NARGS(2);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);

	check_dimensions(a, b);

	/* Defensive: Find maximum with validation */
	for (i = 0; i < a->dim; i++)
	{
		diff = fabs((double) a->data[i] - (double) b->data[i]);

		/* Defensive: Check for overflow */
		if (isinf(diff) || isnan(diff))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("Chebyshev distance: overflow in absolute difference at index %d", i)));

		if (!found_valid || diff > max_diff)
		{
			max_diff = diff;
			found_valid = true;
		}
	}

	/* Defensive: Validate we found at least one valid value */
	if (!found_valid)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Chebyshev distance: no valid differences found")));

	/* Defensive: Validate result */
	if (isnan(max_diff) || isinf(max_diff))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("Chebyshev distance calculation resulted in NaN or Infinity")));

	/* Defensive: Result should be non-negative */
	if (max_diff < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("Chebyshev distance: negative result (numerical error)")));

	PG_RETURN_FLOAT8(max_diff);
}

/* Minkowski distance, p-norm (generalized distance): sum(|a_i-b_i|^p)^(1/p) */
PG_FUNCTION_INFO_V1(vector_minkowski_distance);
Datum
vector_minkowski_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	float8		p;
	double		sum = 0.0;
	double		partial = 0.0;
	double		c = 0.0;
	double		diff;
	double		powered;
	double		y;
	double		t;
	double		max_diff = 0.0;
	double		this_diff;
	double		result;
	int			i;

	CHECK_NARGS(3);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);
	p = PG_GETARG_FLOAT8(2);

	check_dimensions(a, b);

	if (p <= 0 || isnan(p) || isinf(p))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("p must be positive and finite")));

	if (p == 1.0)
	{
		/* Shortcut: L1 distance (Manhattan) */
		for (i = 0; i < a->dim; i++)
			sum += fabs((double) a->data[i] - (double) b->data[i]);
		PG_RETURN_FLOAT8(sum);
	}
	else if (p == 2.0)
	{
		/* Shortcut: L2 distance */
		c = 0.0;
		for (i = 0; i < a->dim; i++)
		{
			diff = (double) a->data[i] - (double) b->data[i];
			y = (diff * diff) - c;
			t = partial + y;
			c = (t - partial) - y;
			partial = t;
		}
		PG_RETURN_FLOAT8(sqrt(partial));
	}
	else if (p == INFINITY || p > 1e10)
	{
		/* Large p treated as Chebyshev */
		max_diff = 0.0;
		for (i = 0; i < a->dim; i++)
		{
			this_diff = fabs((double) a->data[i] - (double) b->data[i]);
			if (this_diff > max_diff)
				max_diff = this_diff;
		}
		PG_RETURN_FLOAT8(max_diff);
	}
	else
	{
		/* General Minkowski sum */
		c = 0.0;
		for (i = 0; i < a->dim; i++)
		{
			diff = fabs((double) a->data[i] - (double) b->data[i]);
			powered = pow(diff, p);

			/* Defensive: Check for overflow in pow */
			if (isinf(powered) || isnan(powered))
				ereport(ERROR,
						(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
						 errmsg("Minkowski distance: overflow in pow at index %d (diff=%.15e, p=%.15e)",
								i, diff, p)));

			/* Kahan summation */
			y = powered - c;
			t = sum + y;
			c = (t - sum) - y;
			sum = t;

			/* Defensive: Check for overflow in accumulation */
			if (isinf(sum) || isnan(sum))
				ereport(ERROR,
						(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
						 errmsg("Minkowski distance: overflow in accumulation at index %d", i)));
		}

		/* Defensive: Validate sum before final pow */
		if (sum < 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("Minkowski distance: negative sum (numerical error)")));

		if (sum == 0.0)
			PG_RETURN_FLOAT8(0.0);

		result = pow(sum, 1.0 / p);

		/* Defensive: Validate final result */
		if (isnan(result) || isinf(result))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("Minkowski distance: final result is NaN or Infinity (sum=%.15e, p=%.15e)",
							sum, p)));

		if (result < 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("Minkowski distance: negative result (numerical error)")));

		PG_RETURN_FLOAT8(result);
	}
}

/*
 * Squared Euclidean distance: L2^2 (faster, no sqrt)
 * Useful when only relative distances matter
 */
PG_FUNCTION_INFO_V1(vector_squared_l2_distance);
Datum
vector_squared_l2_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	double		sum = 0.0;
	double		c = 0.0;
	double		diff;
	double		squared_diff;
	double		y;
	double		t;
	int			i;

	CHECK_NARGS(2);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		diff = (double) a->data[i] - (double) b->data[i];
		squared_diff = diff * diff;

		/* Defensive: Check for overflow in squaring */
		if (isinf(squared_diff) || isnan(squared_diff))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("squared L2 distance: overflow in squared difference at index %d", i)));

		/* Kahan summation */
		y = squared_diff - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;

		/* Defensive: Check for overflow in accumulation */
		if (isinf(sum) || isnan(sum))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("squared L2 distance: overflow in accumulation at index %d", i)));
	}

	/* Defensive: Validate result */
	if (isnan(sum) || isinf(sum))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("squared L2 distance calculation resulted in NaN or Infinity")));

	if (sum < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("squared L2 distance: negative result (numerical error)")));

	PG_RETURN_FLOAT8(sum);
}

/*
 * Jaccard distance: 1 - Jaccard similarity
 * Jaccard similarity = |A ∩ B| / |A ∪ B|
 * For vectors, treats non-zero values as set membership
 */
PG_FUNCTION_INFO_V1(vector_jaccard_distance);
Datum
vector_jaccard_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	int			intersection = 0;
	int			union_count = 0;
	int			i;
	double		jaccard_sim;
	bool		a_nonzero;
	bool		b_nonzero;

	CHECK_NARGS(2);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		a_nonzero = (fabs((double) a->data[i]) > 1e-10);
		b_nonzero = (fabs((double) b->data[i]) > 1e-10);

		if (a_nonzero && b_nonzero)
			intersection++;
		if (a_nonzero || b_nonzero)
			union_count++;
	}

	if (union_count == 0)
	{
		/* Both vectors are zero */
		PG_RETURN_FLOAT8(0.0);
	}

	jaccard_sim = (double) intersection / (double) union_count;
	PG_RETURN_FLOAT8(1.0 - jaccard_sim);
}

/*
 * Dice distance: 1 - Dice coefficient
 * Dice coefficient = 2|A ∩ B| / (|A| + |B|)
 * For vectors, treats non-zero values as set membership
 */
PG_FUNCTION_INFO_V1(vector_dice_distance);
Datum
vector_dice_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	int			intersection = 0;
	int			a_count = 0;
	int			b_count = 0;
	int			i;
	double		dice_coeff;
	bool		a_nonzero;
	bool		b_nonzero;

	CHECK_NARGS(2);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		a_nonzero = (fabs((double) a->data[i]) > 1e-10);
		b_nonzero = (fabs((double) b->data[i]) > 1e-10);

		if (a_nonzero && b_nonzero)
			intersection++;
		if (a_nonzero)
			a_count++;
		if (b_nonzero)
			b_count++;
	}

	if (a_count == 0 && b_count == 0)
	{
		/* Both vectors are zero */
		PG_RETURN_FLOAT8(0.0);
	}

	if (a_count == 0 || b_count == 0)
	{
		/* No overlap */
		PG_RETURN_FLOAT8(1.0);
	}

	dice_coeff = (2.0 * (double) intersection) / ((double) a_count + (double) b_count);
	PG_RETURN_FLOAT8(1.0 - dice_coeff);
}

/*
 * Mahalanobis distance: sqrt((a-b)^T * S^(-1) * (a-b))
 * Where S is the covariance matrix
 * Requires pre-computed covariance matrix (passed as vector of inverse diagonal)
 * Simplified version: assumes diagonal covariance matrix
 */
PG_FUNCTION_INFO_V1(vector_mahalanobis_distance);
Datum
vector_mahalanobis_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	Vector	   *covariance_inv;
	double		sum = 0.0;
	double		c = 0.0;
	double		diff;
	double		inv_var;
	double		diff_sq;
	double		term;
	double		y;
	double		t;
	double		result;
	int			i;

	CHECK_NARGS_RANGE(2, 3);
	a = PG_GETARG_VECTOR_P(0);
	b = PG_GETARG_VECTOR_P(1);
	covariance_inv = PG_ARGISNULL(2) ? NULL : PG_GETARG_VECTOR_P(2);

	check_dimensions(a, b);

	if (covariance_inv == NULL)
	{
		/* No covariance matrix: fall back to L2 distance */
		PG_RETURN_FLOAT4(l2_distance(a, b));
	}

	if (covariance_inv->dim != a->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("covariance matrix dimension must match vector dimension: %d vs %d",
						covariance_inv->dim, a->dim)));

	/* Defensive: Validate covariance matrix structure */
	if (covariance_inv->dim <= 0 || covariance_inv->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid covariance matrix dimension %d",
						covariance_inv->dim)));

	/* Defensive: Compute Mahalanobis distance with diagonal covariance */
	c = 0.0;
	for (i = 0; i < a->dim; i++)
	{
		diff = (double) a->data[i] - (double) b->data[i];
		inv_var = (double) covariance_inv->data[i];

		/* Defensive: Validate covariance inverse */
		if (inv_var <= 0.0 || isnan(inv_var) || isinf(inv_var))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("covariance inverse must be positive and finite at index %d (value=%.15e)",
							i, inv_var)));

		/* Defensive: Check for overflow in multiplication */
		diff_sq = diff * diff;
		if (isinf(diff_sq) || isnan(diff_sq))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("Mahalanobis distance: overflow in squared difference at index %d", i)));

		term = diff_sq * inv_var;
		if (isinf(term) || isnan(term))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("Mahalanobis distance: overflow in term at index %d", i)));

		/* Kahan summation */
		y = term - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;

		/* Defensive: Check for overflow in accumulation */
		if (isinf(sum) || isnan(sum))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("Mahalanobis distance: overflow in accumulation at index %d", i)));
	}

	/* Defensive: Validate sum before sqrt */
	if (sum < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("Mahalanobis distance: negative sum (numerical error)")));

	result = sqrt(sum);

	/* Defensive: Validate final result */
	if (isnan(result) || isinf(result))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("Mahalanobis distance: final result is NaN or Infinity (sum=%.15e)",
						sum)));

	if (result < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("Mahalanobis distance: negative result (numerical error)")));

	PG_RETURN_FLOAT8(result);
}
