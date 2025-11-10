/*-------------------------------------------------------------------------
 *
 * quantization.c
 *	  Vector quantization for memory efficiency and performance
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
#include <string.h>

/* PG_MODULE_MAGIC already defined in neurondb.c */

/* Helper function: float32 -> int8 quantization (max-abs scaling) */
VectorI8 *
quantize_vector_i8(Vector *v)
{
	VectorI8   *result;
	int			size;
	float4		max_abs = 0.0f;
	float4		scale;
	int			i;

	/* Find maximum absolute value for scaling */
	for (i = 0; i < v->dim; i++)
	{
		float4		abs_val = fabsf(v->data[i]);
		if (abs_val > max_abs)
			max_abs = abs_val;
	}

	size = offsetof(VectorI8, data) + sizeof(int8) * v->dim;
	result = (VectorI8 *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = v->dim;

	if (max_abs == 0.0f)
		return result;

	scale = 127.0f / max_abs;

	for (i = 0; i < v->dim; i++)
	{
		float4		val = v->data[i] * scale;

		if (val > 127.0f)
			val = 127.0f;
		if (val < -128.0f)
			val = -128.0f;
		result->data[i] = (int8) rintf(val);	/* round to nearest int8 */
	}

	return result;
}

/*
 * SQL interface: float32 vector -> int8 quantized vector
 */
PG_FUNCTION_INFO_V1(vector_to_int8);
Datum
vector_to_int8(PG_FUNCTION_ARGS)
{
	Vector	   *v = PG_GETARG_VECTOR_P(0);
	VectorI8   *result;

	result = quantize_vector_i8(v);
	PG_RETURN_POINTER(result);
}

/*
 * SQL interface: int8 quantized vector -> float32 vector (dequantize)
 */
PG_FUNCTION_INFO_V1(int8_to_vector);
Datum
int8_to_vector(PG_FUNCTION_ARGS)
{
	VectorI8   *v8 = (VectorI8 *) PG_GETARG_POINTER(0);
	Vector	   *result;
	int			i;

	result = new_vector(v8->dim);

	/* Dequantize using scaling convention: max quantization maps to 127.0f */
	for (i = 0; i < v8->dim; i++)
		result->data[i] = ((float4) v8->data[i]) / 127.0f;

	PG_RETURN_VECTOR_P(result);
}

/*
 * Half-precision quantization: float16 (simulated as uint16)
 */

static uint16
float_to_fp16(float f)
{
	/* Simple IEEE-754 conversion (round to nearest) */
	uint32		u;
	uint16		sign;
	uint32		mantissa;
	int16		exp;

	memcpy(&u, &f, sizeof(uint32));
	sign = (u >> 16) & 0x8000;
	mantissa = u & 0x7fffff;
	exp = ((u >> 23) & 0xff) - 127 + 15; /* bias change */

	if (exp <= 0)
	{
		/* flush to zero */
		return sign;
	}
	else if (exp >= 31)
	{
		/* inf/NaN */
		return sign | 0x7c00;
	}
	else
	{
		return sign | (exp << 10) | (mantissa >> 13);
	}
}

static float
fp16_to_float(uint16 h)
{
	uint32		sign = (h & 0x8000) << 16;
	uint32		exp = (h & 0x7c00) >> 10;
	uint32		mantissa = h & 0x03ff;
	uint32		f;

	if (exp == 0)
	{
		if (mantissa == 0)
			f = sign;
		else
		{
			/* subnormal */
			uint32		m = mantissa;
			uint32		exponent;

			exp = 1;

			while ((m & 0x0400) == 0)
			{
				m <<= 1;
				exp--;
			}
			m &= 0x03ff;
			exponent = 127 - 15 - (10 - exp);
			f = sign | (exponent << 23) | (m << 13);
		}
	}
	else if (exp == 0x1f)
	{
		/* inf/NaN */
		f = sign | 0x7f800000 | (mantissa << 13);
	}
	else
	{
		uint32		exponent = exp + 127 - 15;

		f = sign | (exponent << 23) | (mantissa << 13);
	}

	{
		float		ret;

		memcpy(&ret, &f, 4);
		return ret;
	}
}

VectorF16 *
quantize_vector_f16(Vector *v)
{
	VectorF16  *result;
	int			size;
	int			i;

	size = offsetof(VectorF16, data) + sizeof(uint16) * v->dim;
	result = (VectorF16 *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = v->dim;

	for (i = 0; i < v->dim; i++)
		result->data[i] = float_to_fp16(v->data[i]);

	return result;
}

PG_FUNCTION_INFO_V1(vector_to_float16);
Datum
vector_to_float16(PG_FUNCTION_ARGS)
{
	Vector	   *v = PG_GETARG_VECTOR_P(0);
	VectorF16  *result;

	result = quantize_vector_f16(v);
	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(float16_to_vector);
Datum
float16_to_vector(PG_FUNCTION_ARGS)
{
	VectorF16  *vf16 = (VectorF16 *) PG_GETARG_POINTER(0);
	Vector	   *result;
	int			i;

	result = new_vector(vf16->dim);
	for (i = 0; i < vf16->dim; i++)
		result->data[i] = fp16_to_float(vf16->data[i]);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Binary quantization: maps each component to a bit (sign > 0)
 */
PG_FUNCTION_INFO_V1(vector_to_binary);
Datum
vector_to_binary(PG_FUNCTION_ARGS)
{
	Vector		   *v = PG_GETARG_VECTOR_P(0);
	VectorBinary   *result;
	int				nbytes;
	int				size;
	int				i;
	int				byte_idx;
	int				bit_idx;

	nbytes = (v->dim + 7) / 8;
	size = offsetof(VectorBinary, data) + nbytes;
	result = (VectorBinary *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = v->dim;
	memset(result->data, 0, nbytes);

	for (i = 0; i < v->dim; i++)
	{
		if (v->data[i] > 0.0f)
		{
			byte_idx = i / 8;
			bit_idx = i % 8;
			result->data[byte_idx] |= (1 << bit_idx);
		}
	}

	PG_RETURN_POINTER(result);
}

/*
 * Binary to float32: sign decoding (+1.0 or -1.0)
 */
PG_FUNCTION_INFO_V1(binary_to_vector);
Datum
binary_to_vector(PG_FUNCTION_ARGS)
{
	VectorBinary *vb = (VectorBinary *) PG_GETARG_POINTER(0);
	Vector		 *result;
	int			  i;
	int			  byte_idx;
	int			  bit_idx;

	result = new_vector(vb->dim);

	for (i = 0; i < vb->dim; i++)
	{
		byte_idx = i / 8;
		bit_idx = i % 8;

		result->data[i] = (vb->data[byte_idx] & (1 << bit_idx))
			? 1.0f
			: -1.0f;
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Hamming distance between two binary vectors
 */
PG_FUNCTION_INFO_V1(binary_hamming_distance);
Datum
binary_hamming_distance(PG_FUNCTION_ARGS)
{
	VectorBinary   *a = (VectorBinary *) PG_GETARG_POINTER(0);
	VectorBinary   *b = (VectorBinary *) PG_GETARG_POINTER(1);
	int				count = 0;
	int				nbytes;
	int				i;

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("binary vector dimensions must match")));

	nbytes = (a->dim + 7) / 8;

	for (i = 0; i < nbytes; i++)
	{
		uint8 xor_val = a->data[i] ^ b->data[i];
#if defined(__GNUC__) && (__GNUC__ >= 4)
		count += __builtin_popcount(xor_val);
#else
		/* fallback: Kernighan’s algorithm */
		while (xor_val)
		{
			xor_val &= xor_val - 1;
			count++;
		}
#endif
	}

	PG_RETURN_INT32(count);
}

/*
 * Utility: Dynamic quantization selector
 */
PG_FUNCTION_INFO_V1(dynamic_quantize_vector);
Datum
dynamic_quantize_vector(PG_FUNCTION_ARGS)
{
	Vector  *v = PG_GETARG_VECTOR_P(0);
	float8	memory_pressure = PG_GETARG_FLOAT8(1);
	float8	recall_target = PG_GETARG_FLOAT8(2);

	if ((memory_pressure > 0.8) || (recall_target < 0.85))
		PG_RETURN_POINTER(quantize_vector_i8(v));
	else if ((memory_pressure > 0.6) || (recall_target < 0.90))
		PG_RETURN_POINTER(quantize_vector_f16(v));
	else
		PG_RETURN_POINTER(v);
}
