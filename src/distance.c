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
 * Copyright (c) 2024-2025, NeuronDB Development Group
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

static inline void
check_dimensions(const Vector *a, const Vector *b)
{
    if (a->dim != b->dim)
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("vector dimensions must match: %d vs %d", a->dim, b->dim)));
}

/* L2 (Euclidean) distance */
float4
l2_distance(Vector *a, Vector *b)
{
    double sum = 0.0, c = 0.0;
    int i;

    check_dimensions(a, b);

    /* Kahan summation for numerical stability */
    for (i = 0; i < a->dim; i++)
    {
        double diff = (double) a->data[i] - (double) b->data[i];
        double y = diff * diff - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return (float4) sqrt(sum);
}

PG_FUNCTION_INFO_V1(vector_l2_distance);
Datum
vector_l2_distance(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    PG_RETURN_FLOAT4(l2_distance(a, b));
}

/* Inner product distance */
float4
inner_product_distance(Vector *a, Vector *b)
{
    double sum = 0.0;
    int i;

    check_dimensions(a, b);

    for (i = 0; i < a->dim; i++)
        sum += (double) a->data[i] * (double) b->data[i];

    return (float4) -sum;  /* Negative for ascending order */
}

PG_FUNCTION_INFO_V1(vector_inner_product);
Datum
vector_inner_product(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    PG_RETURN_FLOAT4(inner_product_distance(a, b));
}

/* Cosine distance */
float4
cosine_distance(Vector *a, Vector *b)
{
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    int i;

    check_dimensions(a, b);

    for (i = 0; i < a->dim; i++)
    {
        double va = (double) a->data[i];
        double vb = (double) b->data[i];
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    if (norm_a == 0.0 || norm_b == 0.0)
        return 1.0;

    return (float4) (1.0 - (dot / (sqrt(norm_a) * sqrt(norm_b))));
}

PG_FUNCTION_INFO_V1(vector_cosine_distance);
Datum
vector_cosine_distance(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    PG_RETURN_FLOAT4(cosine_distance(a, b));
}

/* L1 (Manhattan) distance */
float4
l1_distance(Vector *a, Vector *b)
{
    double sum = 0.0;
    int i;

    check_dimensions(a, b);

    for (i = 0; i < a->dim; i++)
        sum += fabs((double) a->data[i] - (double) b->data[i]);

    return (float4) sum;
}

PG_FUNCTION_INFO_V1(vector_l1_distance);
Datum
vector_l1_distance(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    PG_RETURN_FLOAT4(l1_distance(a, b));
}

/* Additional distance metrics... */
PG_FUNCTION_INFO_V1(vector_hamming_distance);
Datum
vector_hamming_distance(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    int count = 0, i;

    check_dimensions(a, b);

    for (i = 0; i < a->dim; i++)
        if (a->data[i] != b->data[i])
            count++;

    PG_RETURN_INT32(count);
}

PG_FUNCTION_INFO_V1(vector_chebyshev_distance);
Datum
vector_chebyshev_distance(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    double max_diff = 0.0;
    int i;

    check_dimensions(a, b);

    for (i = 0; i < a->dim; i++)
    {
        double diff = fabs((double) a->data[i] - (double) b->data[i]);
        if (diff > max_diff)
            max_diff = diff;
    }

    PG_RETURN_FLOAT8(max_diff);
}

PG_FUNCTION_INFO_V1(vector_minkowski_distance);
Datum
vector_minkowski_distance(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    float8  p = PG_GETARG_FLOAT8(2);
    double sum = 0.0;
    int i;

    check_dimensions(a, b);

    if (p <= 0)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("p must be positive")));

    for (i = 0; i < a->dim; i++)
        sum += pow(fabs((double) a->data[i] - (double) b->data[i]), p);

    PG_RETURN_FLOAT8(pow(sum, 1.0 / p));
}
