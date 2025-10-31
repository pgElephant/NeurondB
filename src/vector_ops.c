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

/*
 * Vector element access
 */
PG_FUNCTION_INFO_V1(vector_get);
Datum
vector_get(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    int32   idx = PG_GETARG_INT32(1);
    
    if (idx < 0 || idx >= v->dim)
        ereport(ERROR,
                (errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
                 errmsg("index %d out of bounds for vector of dimension %d", idx, v->dim)));
    
    PG_RETURN_FLOAT4(v->data[idx]);
}

/*
 * Vector element update
 */
PG_FUNCTION_INFO_V1(vector_set);
Datum
vector_set(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    int32   idx = PG_GETARG_INT32(1);
    float4  val = PG_GETARG_FLOAT4(2);
    Vector *result;
    
    if (idx < 0 || idx >= v->dim)
        ereport(ERROR,
                (errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
                 errmsg("index %d out of bounds", idx)));
    
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
    Vector *v = PG_GETARG_VECTOR_P(0);
    int32   start = PG_GETARG_INT32(1);
    int32   end = PG_GETARG_INT32(2);
    Vector *result;
    int     new_dim;
    
    if (start < 0 || start >= v->dim || end < start || end > v->dim)
        ereport(ERROR,
                (errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
                 errmsg("invalid slice bounds")));
    
    new_dim = end - start;
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
    Vector *v = PG_GETARG_VECTOR_P(0);
    float4  val = PG_GETARG_FLOAT4(1);
    Vector *result;
    
    result = new_vector(v->dim + 1);
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
    float4  val = PG_GETARG_FLOAT4(0);
    Vector *v = PG_GETARG_VECTOR_P(1);
    Vector *result;
    
    result = new_vector(v->dim + 1);
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
    Vector *v = PG_GETARG_VECTOR_P(0);
    Vector *result;
    int     i;
    
    result = new_vector(v->dim);
    for (i = 0; i < v->dim; i++)
        result->data[i] = fabs(v->data[i]);
    
    PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise square
 */
PG_FUNCTION_INFO_V1(vector_square);
Datum
vector_square(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    Vector *result;
    int     i;
    
    result = new_vector(v->dim);
    for (i = 0; i < v->dim; i++)
        result->data[i] = v->data[i] * v->data[i];
    
    PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise square root
 */
PG_FUNCTION_INFO_V1(vector_sqrt);
Datum
vector_sqrt(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    Vector *result;
    int     i;
    
    result = new_vector(v->dim);
    for (i = 0; i < v->dim; i++)
    {
        if (v->data[i] < 0)
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("cannot take square root of negative number")));
        result->data[i] = sqrt(v->data[i]);
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
    Vector *v = PG_GETARG_VECTOR_P(0);
    float8  exp = PG_GETARG_FLOAT8(1);
    Vector *result;
    int     i;
    
    result = new_vector(v->dim);
    for (i = 0; i < v->dim; i++)
        result->data[i] = pow(v->data[i], exp);
    
    PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise multiplication (Hadamard product)
 */
PG_FUNCTION_INFO_V1(vector_hadamard);
Datum
vector_hadamard(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    Vector *result;
    int     i;
    
    if (a->dim != b->dim)
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("vector dimensions must match")));
    
    result = new_vector(a->dim);
    for (i = 0; i < a->dim; i++)
        result->data[i] = a->data[i] * b->data[i];
    
    PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise division
 */
PG_FUNCTION_INFO_V1(vector_divide);
Datum
vector_divide(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    Vector *result;
    int     i;
    
    if (a->dim != b->dim)
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("vector dimensions must match")));
    
    result = new_vector(a->dim);
    for (i = 0; i < a->dim; i++)
    {
        if (b->data[i] == 0.0)
            ereport(ERROR,
                    (errcode(ERRCODE_DIVISION_BY_ZERO),
                     errmsg("division by zero")));
        result->data[i] = a->data[i] / b->data[i];
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
    Vector *v = PG_GETARG_VECTOR_P(0);
    double  sum = 0.0;
    int     i;
    
    for (i = 0; i < v->dim; i++)
        sum += v->data[i];
    
    PG_RETURN_FLOAT8(sum / v->dim);
}

PG_FUNCTION_INFO_V1(vector_variance);
Datum
vector_variance(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    double  mean = 0.0, variance = 0.0;
    int     i;
    
    for (i = 0; i < v->dim; i++)
        mean += v->data[i];
    mean /= v->dim;
    
    for (i = 0; i < v->dim; i++)
    {
        double diff = v->data[i] - mean;
        variance += diff * diff;
    }
    
    PG_RETURN_FLOAT8(variance / v->dim);
}

PG_FUNCTION_INFO_V1(vector_stddev);
Datum
vector_stddev(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    double  mean = 0.0, variance = 0.0;
    int     i;
    
    for (i = 0; i < v->dim; i++)
        mean += v->data[i];
    mean /= v->dim;
    
    for (i = 0; i < v->dim; i++)
    {
        double diff = v->data[i] - mean;
        variance += diff * diff;
    }
    
    PG_RETURN_FLOAT8(sqrt(variance / v->dim));
}

PG_FUNCTION_INFO_V1(vector_min);
Datum
vector_min(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    float4  min_val = v->data[0];
    int     i;
    
    for (i = 1; i < v->dim; i++)
        if (v->data[i] < min_val)
            min_val = v->data[i];
    
    PG_RETURN_FLOAT4(min_val);
}

PG_FUNCTION_INFO_V1(vector_max);
Datum
vector_max(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    float4  max_val = v->data[0];
    int     i;
    
    for (i = 1; i < v->dim; i++)
        if (v->data[i] > max_val)
            max_val = v->data[i];
    
    PG_RETURN_FLOAT4(max_val);
}

PG_FUNCTION_INFO_V1(vector_sum);
Datum
vector_sum(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    double  sum = 0.0;
    int     i;
    
    for (i = 0; i < v->dim; i++)
        sum += v->data[i];
    
    PG_RETURN_FLOAT8(sum);
}

/*
 * Vector comparison
 */
PG_FUNCTION_INFO_V1(vector_eq);
Datum
vector_eq(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    int     i;
    
    if (a->dim != b->dim)
        PG_RETURN_BOOL(false);
    
    for (i = 0; i < a->dim; i++)
        if (a->data[i] != b->data[i])
            PG_RETURN_BOOL(false);
    
    PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(vector_ne);
Datum
vector_ne(PG_FUNCTION_ARGS)
{
    return DirectFunctionCall2(vector_eq,
                               PG_GETARG_DATUM(0),
                               PG_GETARG_DATUM(1)) ? 
           BoolGetDatum(false) : BoolGetDatum(true);
}

/*
 * Vector clipping (element-wise min/max)
 */
PG_FUNCTION_INFO_V1(vector_clip);
Datum
vector_clip(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    float4  min_val = PG_GETARG_FLOAT4(1);
    float4  max_val = PG_GETARG_FLOAT4(2);
    Vector *result;
    int     i;
    
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
    Vector *v = PG_GETARG_VECTOR_P(0);
    Vector *result;
    double  mean = 0.0, stddev = 0.0;
    int     i;
    
    /* Calculate mean */
    for (i = 0; i < v->dim; i++)
        mean += v->data[i];
    mean /= v->dim;
    
    /* Calculate standard deviation */
    for (i = 0; i < v->dim; i++)
    {
        double diff = v->data[i] - mean;
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
        /* All values are the same */
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
    Vector *v = PG_GETARG_VECTOR_P(0);
    Vector *result;
    float4  min_val = v->data[0], max_val = v->data[0];
    float4  range;
    int     i;
    
    /* Find min and max */
    for (i = 1; i < v->dim; i++)
    {
        if (v->data[i] < min_val) min_val = v->data[i];
        if (v->data[i] > max_val) max_val = v->data[i];
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
            result->data[i] = 0.5;  /* Middle of [0,1] range */
    }
    
    PG_RETURN_VECTOR_P(result);
}

