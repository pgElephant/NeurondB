/*-------------------------------------------------------------------------
 *
 * quantization.c
 *		Vector quantization for memory efficiency and performance
 *
 * This file implements multiple quantization schemes including INT8
 * quantization (8x compression), float16 quantization (2x compression),
 * and binary quantization (32x compression). Includes optimized
 * Hamming distance for binary vectors.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  contrib/neurondb/quantization.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include <math.h>

/* INT8 Quantization */
VectorI8 *
quantize_vector_i8(Vector *v)
{
    VectorI8   *result;
    int         size;
    float4      max_abs = 0.0;
    float4      scale;
    int         i;

    for (i = 0; i < v->dim; i++)
    {
        float4 abs_val = fabs(v->data[i]);
        if (abs_val > max_abs)
            max_abs = abs_val;
    }

    size = offsetof(VectorI8, data) + sizeof(int8) * v->dim;
    result = (VectorI8 *) palloc0(size);
    SET_VARSIZE(result, size);
    result->dim = v->dim;

    if (max_abs == 0.0)
        return result;

    scale = 127.0 / max_abs;

    for (i = 0; i < v->dim; i++)
        result->data[i] = (int8) round(v->data[i] * scale);

    return result;
}

PG_FUNCTION_INFO_V1(vector_to_int8);
Datum
vector_to_int8(PG_FUNCTION_ARGS)
{
    Vector     *v = PG_GETARG_VECTOR_P(0);
    VectorI8   *result = quantize_vector_i8(v);
    PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(int8_to_vector);
Datum
int8_to_vector(PG_FUNCTION_ARGS)
{
    VectorI8   *v8 = (VectorI8 *) PG_GETARG_POINTER(0);
    Vector     *result;
    int8        max_abs = 0;
    float4      scale;
    int         i;

    for (i = 0; i < v8->dim; i++)
    {
        int8 abs_val = abs(v8->data[i]);
        if (abs_val > max_abs)
            max_abs = abs_val;
    }

    result = new_vector(v8->dim);

    if (max_abs == 0)
        PG_RETURN_VECTOR_P(result);

    scale = 1.0 / 127.0;

    for (i = 0; i < v8->dim; i++)
        result->data[i] = v8->data[i] * scale;

    PG_RETURN_VECTOR_P(result);
}

/* Binary quantization */
PG_FUNCTION_INFO_V1(vector_to_binary);
Datum
vector_to_binary(PG_FUNCTION_ARGS)
{
    Vector         *v = PG_GETARG_VECTOR_P(0);
    VectorBinary   *result;
    int             nbytes = (v->dim + 7) / 8;
    int             size;
    int             i, byte_idx, bit_idx;

    size = offsetof(VectorBinary, data) + nbytes;
    result = (VectorBinary *) palloc0(size);
    SET_VARSIZE(result, size);
    result->dim = v->dim;

    for (i = 0; i < v->dim; i++)
    {
        if (v->data[i] > 0.0)
        {
            byte_idx = i / 8;
            bit_idx = i % 8;
            result->data[byte_idx] |= (1 << bit_idx);
        }
    }

    PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(binary_hamming_distance);
Datum
binary_hamming_distance(PG_FUNCTION_ARGS)
{
    VectorBinary   *a = (VectorBinary *) PG_GETARG_POINTER(0);
    VectorBinary   *b = (VectorBinary *) PG_GETARG_POINTER(1);
    int             count = 0;
    int             nbytes = (a->dim + 7) / 8;
    int             i;

    if (a->dim != b->dim)
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("binary vector dimensions must match")));

    for (i = 0; i < nbytes; i++)
    {
        uint8 xor_val = a->data[i] ^ b->data[i];
        while (xor_val)
        {
            xor_val &= xor_val - 1;
            count++;
        }
    }

    PG_RETURN_INT32(count);
}
