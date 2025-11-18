/*-------------------------------------------------------------------------
 *
 * opclass.c
 *		Operator classes and families for vector distance operations
 *
 * Defines operator classes for:
 * - L2 distance (Euclidean)
 * - Cosine distance
 * - Inner product
 *
 * Each operator class supports both HNSW and IVF access methods.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/index/opclass.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "catalog/pg_am.h"
#include "catalog/pg_operator.h"
#include "catalog/pg_opclass.h"
#include "catalog/pg_opfamily.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include <math.h>

/*
 * Distance function implementations for operator support
 */

/*
 * vector_l2_distance(vector, vector) -> float4
 *
 * Computes Euclidean (L2) distance between two vectors.
 * Used by the <-> operator.
 */
PG_FUNCTION_INFO_V1(vector_l2_distance_op);

Datum
vector_l2_distance_op(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	float4 result;
	extern float4 l2_distance_simd(Vector *a, Vector *b);

	if (a->dim != b->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must "
				       "match: %d vs %d",
					a->dim,
					b->dim)));

	/* Use SIMD-optimized version */
	result = l2_distance_simd(a, b);
	PG_RETURN_FLOAT4(result);
}

/*
 * vector_cosine_distance(vector, vector) -> float4
 *
 * Computes cosine distance (1 - cosine_similarity).
 * Used by the <=> operator.
 */
PG_FUNCTION_INFO_V1(vector_cosine_distance_op);

Datum
vector_cosine_distance_op(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	float4 result;
	extern float4 cosine_distance_simd(Vector *a, Vector *b);

	if (a->dim != b->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must "
				       "match")));

	/* Use SIMD-optimized version */
	result = cosine_distance_simd(a, b);
	PG_RETURN_FLOAT4(result);
}

/*
 * vector_inner_product_distance(vector, vector) -> float4
 *
 * Computes negative inner product (for maximum inner product search).
 * Used by the <#> operator.
 */
PG_FUNCTION_INFO_V1(vector_inner_product_distance_op);

Datum
vector_inner_product_distance_op(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	float4 result;
	extern float4 inner_product_simd(Vector *a, Vector *b);

	if (a->dim != b->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must "
				       "match")));

	/* Use SIMD-optimized version */
	result = inner_product_simd(a, b);
	PG_RETURN_FLOAT4(result);
}

/*
 * Comparison functions for ordering
 *
 * These are required by the operator class system to enable
 * ORDER BY clauses with distance operators.
 */

/*
 * vector_l2_less(vector, vector, vector) -> bool
 *
 * Returns true if distance(a, query) < distance(b, query)
 */
PG_FUNCTION_INFO_V1(vector_l2_less);

Datum
vector_l2_less(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	double sum_a = 0.0, sum_b = 0.0;
	int i;

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must "
				       "match")));

	for (i = 0; i < query->dim; i++)
	{
		double diff_a = a->data[i] - query->data[i];
		double diff_b = b->data[i] - query->data[i];
		sum_a += diff_a * diff_a;
		sum_b += diff_b * diff_b;
	}

	dist_a = (float4)sqrt(sum_a);
	dist_b = (float4)sqrt(sum_b);

	PG_RETURN_BOOL(dist_a < dist_b);
}

/*
 * vector_l2_less_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_l2_less_equal);

Datum
vector_l2_less_equal(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	double sum_a = 0.0, sum_b = 0.0;
	int i;

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must "
				       "match")));

	for (i = 0; i < query->dim; i++)
	{
		double diff_a = a->data[i] - query->data[i];
		double diff_b = b->data[i] - query->data[i];
		sum_a += diff_a * diff_a;
		sum_b += diff_b * diff_b;
	}

	dist_a = (float4)sqrt(sum_a);
	dist_b = (float4)sqrt(sum_b);

	PG_RETURN_BOOL(dist_a <= dist_b);
}

/*
 * vector_l2_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_l2_equal);

Datum
vector_l2_equal(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	double sum_a = 0.0, sum_b = 0.0;
	int i;

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must "
				       "match")));

	for (i = 0; i < query->dim; i++)
	{
		double diff_a = a->data[i] - query->data[i];
		double diff_b = b->data[i] - query->data[i];
		sum_a += diff_a * diff_a;
		sum_b += diff_b * diff_b;
	}

	dist_a = (float4)sqrt(sum_a);
	dist_b = (float4)sqrt(sum_b);

	/* Use epsilon for float comparison */
	PG_RETURN_BOOL(fabs(dist_a - dist_b) < 1e-6);
}

/*
 * vector_l2_greater(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_l2_greater);

Datum
vector_l2_greater(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	double sum_a = 0.0, sum_b = 0.0;
	int i;

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must "
				       "match")));

	for (i = 0; i < query->dim; i++)
	{
		double diff_a = a->data[i] - query->data[i];
		double diff_b = b->data[i] - query->data[i];
		sum_a += diff_a * diff_a;
		sum_b += diff_b * diff_b;
	}

	dist_a = (float4)sqrt(sum_a);
	dist_b = (float4)sqrt(sum_b);

	PG_RETURN_BOOL(dist_a > dist_b);
}

/*
 * vector_l2_greater_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_l2_greater_equal);

Datum
vector_l2_greater_equal(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	double sum_a = 0.0, sum_b = 0.0;
	int i;

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must "
				       "match")));

	for (i = 0; i < query->dim; i++)
	{
		double diff_a = a->data[i] - query->data[i];
		double diff_b = b->data[i] - query->data[i];
		sum_a += diff_a * diff_a;
		sum_b += diff_b * diff_b;
	}

	dist_a = (float4)sqrt(sum_a);
	dist_b = (float4)sqrt(sum_b);

	PG_RETURN_BOOL(dist_a >= dist_b);
}

/*
 * Similar comparison functions would be needed for cosine and inner product.
 * Omitted for brevity - follow same pattern.
 */

/*
 * Distance support function for index AM
 *
 * This function is called by index AMs to compute distance during search.
 */
PG_FUNCTION_INFO_V1(vector_distance_support);

Datum
vector_distance_support(PG_FUNCTION_ARGS)
{
	/* This would return information about distance function properties */
	/* For now, stub */
	PG_RETURN_VOID();
}

/*
 * Operator ordering function
 *
 * Returns ordering information for the distance operator.
 */
PG_FUNCTION_INFO_V1(vector_order_support);

Datum
vector_order_support(PG_FUNCTION_ARGS)
{
	/* Provides ordering semantics to the planner */
	PG_RETURN_VOID();
}

/*
 * Helper: Register operator class in catalog (called at extension install)
 *
 * This would typically be done via SQL DDL in neurondb--1.0.sql:
 *
 * CREATE OPERATOR CLASS vector_l2_ops
 *     DEFAULT FOR TYPE vector USING hnsw AS
 *     OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
 *     FUNCTION 1 vector_l2_distance_op(vector, vector);
 *
 * CREATE OPERATOR CLASS vector_cosine_ops
 *     FOR TYPE vector USING hnsw AS
 *     OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
 *     FUNCTION 1 vector_cosine_distance_op(vector, vector);
 *
 * CREATE OPERATOR CLASS vector_ip_ops
 *     FOR TYPE vector USING hnsw AS
 *     OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
 *     FUNCTION 1 vector_inner_product_distance_op(vector, vector);
 */

/*
 * vector_cosine_less(vector, vector, vector) -> bool
 *
 * For cosine distance ordering: lower distance = better match
 */
PG_FUNCTION_INFO_V1(vector_cosine_less);

Datum
vector_cosine_less(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	extern float4 cosine_distance_simd(Vector *a, Vector *b);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must "
				       "match: %d vs %d vs %d",
					a->dim, b->dim, query->dim)));

	/* Use SIMD-optimized version */
	dist_a = cosine_distance_simd(a, query);
	dist_b = cosine_distance_simd(b, query);

	PG_RETURN_BOOL(dist_a < dist_b);
}

/*
 * vector_cosine_less_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_cosine_less_equal);

Datum
vector_cosine_less_equal(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	extern float4 cosine_distance_simd(Vector *a, Vector *b);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must match")));

	dist_a = cosine_distance_simd(a, query);
	dist_b = cosine_distance_simd(b, query);

	PG_RETURN_BOOL(dist_a <= dist_b);
}

/*
 * vector_cosine_greater(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_cosine_greater);

Datum
vector_cosine_greater(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	extern float4 cosine_distance_simd(Vector *a, Vector *b);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must match")));

	dist_a = cosine_distance_simd(a, query);
	dist_b = cosine_distance_simd(b, query);

	PG_RETURN_BOOL(dist_a > dist_b);
}

/*
 * vector_cosine_greater_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_cosine_greater_equal);

Datum
vector_cosine_greater_equal(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	extern float4 cosine_distance_simd(Vector *a, Vector *b);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must match")));

	dist_a = cosine_distance_simd(a, query);
	dist_b = cosine_distance_simd(b, query);

	PG_RETURN_BOOL(dist_a >= dist_b);
}

/*
 * vector_inner_product_less(vector, vector, vector) -> bool
 *
 * For inner product: higher value = better match, so we want to minimize
 * the negative inner product distance
 */
PG_FUNCTION_INFO_V1(vector_inner_product_less);

Datum
vector_inner_product_less(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	extern float4 inner_product_distance_simd(Vector *a, Vector *b);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must match")));

	dist_a = inner_product_distance_simd(a, query);
	dist_b = inner_product_distance_simd(b, query);

	PG_RETURN_BOOL(dist_a < dist_b);
}

/*
 * vector_inner_product_less_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_inner_product_less_equal);

Datum
vector_inner_product_less_equal(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	extern float4 inner_product_distance_simd(Vector *a, Vector *b);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must match")));

	dist_a = inner_product_distance_simd(a, query);
	dist_b = inner_product_distance_simd(b, query);

	PG_RETURN_BOOL(dist_a <= dist_b);
}

/*
 * vector_inner_product_greater(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_inner_product_greater);

Datum
vector_inner_product_greater(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	extern float4 inner_product_distance_simd(Vector *a, Vector *b);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must match")));

	dist_a = inner_product_distance_simd(a, query);
	dist_b = inner_product_distance_simd(b, query);

	PG_RETURN_BOOL(dist_a > dist_b);
}

/*
 * vector_inner_product_greater_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_inner_product_greater_equal);

Datum
vector_inner_product_greater_equal(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *query = PG_GETARG_VECTOR_P(2);
	float4 dist_a, dist_b;
	extern float4 inner_product_distance_simd(Vector *a, Vector *b);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("neurondb: vector dimensions must match")));

	dist_a = inner_product_distance_simd(a, query);
	dist_b = inner_product_distance_simd(b, query);

	PG_RETURN_BOOL(dist_a >= dist_b);
}

/*
 * Utility function to check if operator class exists
 */
PG_FUNCTION_INFO_V1(neurondb_has_opclass);

Datum
neurondb_has_opclass(PG_FUNCTION_ARGS)
{
	text *opclass_name = PG_GETARG_TEXT_PP(0);
	char *name = text_to_cstring(opclass_name);
	bool exists = false;

	/* TODO: Query pg_opclass to check if operator class exists */
	/* For now, return true as stub */
	exists = true;

	elog(DEBUG1,
		"neurondb: Checking for operator class '%s': %s",
		name,
		exists ? "found" : "not found");

	PG_RETURN_BOOL(exists);
}
