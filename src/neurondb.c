/*-------------------------------------------------------------------------
 *
 * neurondb.c
 *		Core implementation of NeurondB vector type and operations
 *
 * This file contains the main entry point and shared vector utilities
 * including type I/O functions, vector construction, normalization,
 * arithmetic operations, and array conversions.
 *
 * Copyright (c) 2024-2025, NeuronDB Development Group
 *
 * IDENTIFICATION
 *	  contrib/neurondb/neurondb.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include <math.h>

PG_MODULE_MAGIC;

/* ========== Vector Construction ========== */

Vector *
new_vector(int dim)
{
    Vector *result;
    int     size;

    if (dim < 1)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("vector dimension must be at least 1")));

    if (dim > VECTOR_MAX_DIM)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("vector dimension cannot exceed %d", VECTOR_MAX_DIM)));

    size = VECTOR_SIZE(dim);
    result = (Vector *) palloc0(size);
    SET_VARSIZE(result, size);
    result->dim = dim;

    return result;
}

Vector *
copy_vector(Vector *vector)
{
    Vector *result;
    int     size = VARSIZE_ANY(vector);

    result = (Vector *) palloc(size);
    memcpy(result, vector, size);
    return result;
}

/* ========== Vector I/O Functions ========== */

Vector *
vector_in_internal(char *str, int *out_dim, bool check)
{
    char       *ptr = str;
    float4     *data;
    int         dim = 0;
    int         capacity = 16;
    Vector     *result;
    char       *endptr;

    /* Skip whitespace */
    while (isspace((unsigned char) *ptr))
        ptr++;

    /* Check for opening bracket */
    if (*ptr == '[' || *ptr == '{')
        ptr++;

    /* Allocate initial buffer */
    data = (float4 *) palloc(sizeof(float4) * capacity);

    /* Parse numbers */
    while (*ptr && *ptr != ']' && *ptr != '}')
    {
        /* Skip whitespace and commas */
        while (isspace((unsigned char) *ptr) || *ptr == ',')
            ptr++;

        if (*ptr == ']' || *ptr == '}' || *ptr == '\0')
            break;

        /* Resize if needed */
        if (dim >= capacity)
        {
            capacity *= 2;
            data = (float4 *) repalloc(data, sizeof(float4) * capacity);
        }

        /* Parse float */
        data[dim] = strtof(ptr, &endptr);
        if (ptr == endptr)
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                     errmsg("invalid input syntax for type vector: \"%s\"", str)));

        if (check && (isinf(data[dim]) || isnan(data[dim])))
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                     errmsg("vector values cannot be NaN or Infinity")));

        ptr = endptr;
        dim++;
    }

    if (dim == 0)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                 errmsg("vector must have at least 1 dimension")));

    /* Create result */
    result = new_vector(dim);
    memcpy(result->data, data, sizeof(float4) * dim);
    pfree(data);

    if (out_dim)
        *out_dim = dim;

    return result;
}

char *
vector_out_internal(Vector *vector)
{
    StringInfoData buf;
    int i;

    initStringInfo(&buf);
    appendStringInfoChar(&buf, '[');

    for (i = 0; i < vector->dim; i++)
    {
        if (i > 0)
            appendStringInfoChar(&buf, ',');
        appendStringInfo(&buf, "%g", vector->data[i]);
    }

    appendStringInfoChar(&buf, ']');
    return buf.data;
}

/* ========== SQL-Callable Functions ========== */

PG_FUNCTION_INFO_V1(vector_in);
Datum
vector_in(PG_FUNCTION_ARGS)
{
    char   *str = PG_GETARG_CSTRING(0);
    Vector *result = vector_in_internal(str, NULL, true);
    PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_out);
Datum
vector_out(PG_FUNCTION_ARGS)
{
    Vector *vector = PG_GETARG_VECTOR_P(0);
    char   *result = vector_out_internal(vector);
    PG_RETURN_CSTRING(result);
}

PG_FUNCTION_INFO_V1(vector_recv);
Datum
vector_recv(PG_FUNCTION_ARGS)
{
    StringInfo  buf = (StringInfo) PG_GETARG_POINTER(0);
    Vector     *result;
    int16       dim;
    int         i;

    dim = pq_getmsgint(buf, sizeof(int16));
    result = new_vector(dim);

    for (i = 0; i < dim; i++)
        result->data[i] = pq_getmsgfloat4(buf);

    PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_send);
Datum
vector_send(PG_FUNCTION_ARGS)
{
    Vector         *vec = PG_GETARG_VECTOR_P(0);
    StringInfoData  buf;
    int             i;

    pq_begintypsend(&buf);
    pq_sendint(&buf, vec->dim, sizeof(int16));

    for (i = 0; i < vec->dim; i++)
        pq_sendfloat4(&buf, vec->data[i]);

    PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/* Vector dimensions */
PG_FUNCTION_INFO_V1(vector_dims);
Datum
vector_dims(PG_FUNCTION_ARGS)
{
    Vector *vector = PG_GETARG_VECTOR_P(0);
    PG_RETURN_INT32(vector->dim);
}

/* Vector norm (L2) */
PG_FUNCTION_INFO_V1(vector_norm);
Datum
vector_norm(PG_FUNCTION_ARGS)
{
    Vector *vector = PG_GETARG_VECTOR_P(0);
    double  sum = 0.0;
    int     i;

    for (i = 0; i < vector->dim; i++)
        sum += (double) vector->data[i] * (double) vector->data[i];

    PG_RETURN_FLOAT8(sqrt(sum));
}

/* Normalize vector */
void
normalize_vector(Vector *v)
{
    double norm = 0.0;
    int i;

    for (i = 0; i < v->dim; i++)
        norm += (double) v->data[i] * (double) v->data[i];

    if (norm > 0.0)
    {
        norm = sqrt(norm);
        for (i = 0; i < v->dim; i++)
            v->data[i] /= norm;
    }
}

Vector *
normalize_vector_new(Vector *v)
{
    Vector *result = copy_vector(v);
    normalize_vector(result);
    return result;
}

PG_FUNCTION_INFO_V1(vector_normalize);
Datum
vector_normalize(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    Vector *result = normalize_vector_new(v);
    PG_RETURN_VECTOR_P(result);
}

/* Vector concatenation */
PG_FUNCTION_INFO_V1(vector_concat);
Datum
vector_concat(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    Vector *result;
    int     new_dim = a->dim + b->dim;

    result = new_vector(new_dim);
    memcpy(result->data, a->data, sizeof(float4) * a->dim);
    memcpy(result->data + a->dim, b->data, sizeof(float4) * b->dim);

    PG_RETURN_VECTOR_P(result);
}

/* Vector addition */
PG_FUNCTION_INFO_V1(vector_add);
Datum
vector_add(PG_FUNCTION_ARGS)
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
        result->data[i] = a->data[i] + b->data[i];

    PG_RETURN_VECTOR_P(result);
}

/* Vector subtraction */
PG_FUNCTION_INFO_V1(vector_sub);
Datum
vector_sub(PG_FUNCTION_ARGS)
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
        result->data[i] = a->data[i] - b->data[i];

    PG_RETURN_VECTOR_P(result);
}

/* Vector scalar multiplication */
PG_FUNCTION_INFO_V1(vector_mul);
Datum
vector_mul(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    float8  scalar = PG_GETARG_FLOAT8(1);
    Vector *result;
    int     i;

    result = new_vector(v->dim);
    for (i = 0; i < v->dim; i++)
        result->data[i] = v->data[i] * scalar;

    PG_RETURN_VECTOR_P(result);
}

/* Array to vector conversion */
PG_FUNCTION_INFO_V1(array_to_vector);
Datum
array_to_vector(PG_FUNCTION_ARGS)
{
    ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
    Vector     *result;
    int16       typlen;
    bool        typbyval;
    char        typalign;
    Datum      *elems;
    bool       *nulls;
    int         nelems;
    int         i;

    if (ARR_NDIM(array) != 1)
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("array must be one-dimensional")));

    get_typlenbyvalalign(ARR_ELEMTYPE(array), &typlen, &typbyval, &typalign);
    deconstruct_array(array, ARR_ELEMTYPE(array),
                      typlen, typbyval, typalign,
                      &elems, &nulls, &nelems);

    result = new_vector(nelems);

    for (i = 0; i < nelems; i++)
    {
        if (nulls[i])
            ereport(ERROR,
                    (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
                     errmsg("array must not contain nulls")));

        result->data[i] = DatumGetFloat4(elems[i]);
    }

    PG_RETURN_VECTOR_P(result);
}

/* Vector to array conversion */
PG_FUNCTION_INFO_V1(vector_to_array);
Datum
vector_to_array(PG_FUNCTION_ARGS)
{
    Vector     *vec = PG_GETARG_VECTOR_P(0);
    Datum      *elems;
    ArrayType  *result;
    int         i;

    elems = (Datum *) palloc(sizeof(Datum) * vec->dim);

    for (i = 0; i < vec->dim; i++)
        elems[i] = Float4GetDatum(vec->data[i]);

    result = construct_array(elems, vec->dim, FLOAT4OID,
                            sizeof(float4), true, 'i');

    pfree(elems);

    PG_RETURN_ARRAYTYPE_P(result);
}
