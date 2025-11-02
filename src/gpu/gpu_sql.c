/*
 * gpu_sql.c
 * PostgreSQL SQL-callable wrappers for NeurondB GPU-accelerated vector
 * operations. Implements robust CPU fallback logic, strict error checking,
 * and correct Postgres resource management throughout.
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/guc.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"

#include "neurondb_config.h"
#include "neurondb_gpu.h"
#include "neurondb.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/*
 * ndb_gpu_can_run
 * Check if GPU support and specific kernel are enabled and available.
 * All conditions required for this to return true; safe for fallback.
 */
static inline bool
ndb_gpu_can_run(const char *kernel_name)
{
	/* Check if global GPU support is enabled. */
	if (!neurondb_gpu_enabled)
		return false;
	/* Check if the specified kernel is available. */
	if (!ndb_gpu_kernel_enabled(kernel_name))
		return false;
	/* If needed, initialize device/system. */
	ndb_gpu_init_if_needed();
	/* Final status: device is available and ready. */
	return neurondb_gpu_is_available();
}

/*
 * vector_l2_distance_gpu
 * SQL-callable interface: Compute L2 distance (Euclidean norm) between
 * two vectors. Uses GPU if available, otherwise falls back to CPU.
 */
PG_FUNCTION_INFO_V1(vector_l2_distance_gpu);
Datum
vector_l2_distance_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	float4		result = -1.0f;
	extern float4 l2_distance(Vector *a, Vector *b);

	if (ndb_gpu_can_run("l2") && a->dim == b->dim)
	{
		result = neurondb_gpu_l2_distance(a->data, b->data, a->dim);
		if (result >= 0.0f && !isnan(result))
			PG_RETURN_FLOAT4(result);
		/* Otherwise, fall through to CPU. */
	}

	/* CPU fallback. */
	PG_RETURN_FLOAT4(l2_distance(a, b));
}

/*
 * vector_cosine_distance_gpu
 * SQL-callable interface: Compute cosine distance (1 - cosine similarity)
 * between two vectors. Prefers GPU, falls back to CPU when needed.
 */
PG_FUNCTION_INFO_V1(vector_cosine_distance_gpu);
Datum
vector_cosine_distance_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	float4		result = -1.0f;
	extern float4 cosine_distance(Vector *a, Vector *b);

	if (ndb_gpu_can_run("cosine") && a->dim == b->dim)
	{
		result = neurondb_gpu_cosine_distance(a->data, b->data, a->dim);
		if (result >= 0.0f && !isnan(result))
			PG_RETURN_FLOAT4(result);
	}

	PG_RETURN_FLOAT4(cosine_distance(a, b));
}

/*
 * vector_inner_product_gpu
 * SQL-callable interface: Compute -dot(a, b) as a distance metric.
 * Uses GPU if able, else falls back to CPU.
 */
PG_FUNCTION_INFO_V1(vector_inner_product_gpu);
Datum
vector_inner_product_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	float4		result = -1.0f;
	extern float4 inner_product_distance(Vector *a, Vector *b);

	if (ndb_gpu_can_run("ip") && a->dim == b->dim)
	{
		result = neurondb_gpu_inner_product(a->data, b->data, a->dim);
		if (result >= 0.0f && !isnan(result))
			PG_RETURN_FLOAT4(result);
	}

	PG_RETURN_FLOAT4(inner_product_distance(a, b));
}

/*
 * vector_to_int8_gpu
 * SQL-callable: Quantizes a float32 vector to int8, returns bytea of
 * length count. GPU preferred, CPU fallback when needed.
 */
PG_FUNCTION_INFO_V1(vector_to_int8_gpu);
Datum
vector_to_int8_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *v = PG_GETARG_VECTOR_P(0);
	int			count = v->dim;
	bytea	   *out;

	out = (bytea *) palloc(VARHDRSZ + count);
	SET_VARSIZE(out, VARHDRSZ + count);

	if (ndb_gpu_can_run("quantize"))
	{
		neurondb_gpu_quantize_int8(v->data, (int8 *) VARDATA(out), count);
	}
	else
	{
		int			i;
		float		maxv = 0.0f;
		float		scale;
		int8	   *quantized = (int8 *) VARDATA(out);

		for (i = 0; i < count; i++)
		{
			float	val = fabsf(v->data[i]);
			if (val > maxv)
				maxv = val;
		}

		scale = (maxv > 0.0f) ? (127.0f / maxv) : 1.0f;

		for (i = 0; i < count; i++)
		{
			float	scaled = v->data[i] * scale;

			if (scaled > 127.0f)
				scaled = 127.0f;
			else if (scaled < -128.0f)
				scaled = -128.0f;

			quantized[i] = (int8) lrintf(scaled);
		}
	}

	PG_RETURN_BYTEA_P(out);
}

/*
 * vector_to_fp16_gpu
 * SQL-callable: Quantizes float32 vector to packed IEEE 754 half-precision,
 * two bytes per value, returned in a bytea.
 */
PG_FUNCTION_INFO_V1(vector_to_fp16_gpu);
Datum
vector_to_fp16_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *v = PG_GETARG_VECTOR_P(0);
	int			count = v->dim;
	int			out_bytes = count * 2;	/* 2 bytes per fp16 */
	bytea	   *out;

	out = (bytea *) palloc(VARHDRSZ + out_bytes);
	SET_VARSIZE(out, VARHDRSZ + out_bytes);

	if (ndb_gpu_can_run("quantize"))
	{
		neurondb_gpu_quantize_fp16(v->data, (void *) VARDATA(out), count);
	}
	else
	{
		int			i;
		uint16	   *dst = (uint16 *) VARDATA(out);

		for (i = 0; i < count; i++)
		{
			float		f = v->data[i];
			union { float f; uint32 u; } in;
			uint32		f32;
			uint32		sign;
			int32		exp;
			uint32		mant;
			uint16		out_fp16;

			in.f = f;
			f32 = in.u;

			sign = (f32 >> 31) & 0x1;
			exp = ((f32 >> 23) & 0xFF) - 127;
			mant = f32 & 0x7FFFFF;
			out_fp16 = 0;

			if ((f32 & 0x7FFFFFFF) == 0)
			{
				/* Signed zero */
				out_fp16 = sign << 15;
			}
			else if ((f32 & 0x7F800000) == 0x7F800000)
			{
				/* NaN or Inf */
				if ((f32 & 0x007FFFFF) == 0)
				{
					out_fp16 = (sign << 15) | (0x1F << 10);
				}
				else
				{
					out_fp16 = (sign << 15) | (0x1F << 10) | ((mant >> 13) ? (mant >> 13) : 1);
				}
			}
			else if (exp > 15)
			{
				/* Overflow => Inf */
				out_fp16 = (sign << 15) | (0x1F << 10);
			}
			else if (exp >= -14)
			{
				uint32	new_exp = exp + 15;
				uint32	mant_fp16 = mant >> 13;

				/* Round-to-nearest/evens. */
				if (((mant >> 12) & 1) && (((mant & 0xFFF) > 0) || (mant_fp16 & 1)))
				{
					mant_fp16 += 1;
					if (mant_fp16 == 0x400)
					{
						mant_fp16 = 0;
						new_exp++;
						if (new_exp == 0x1F)
						{
							out_fp16 = (sign << 15) | (0x1F << 10);
							dst[i] = out_fp16;
							continue;
						}
					}
				}
				out_fp16 = (sign << 15) | ((new_exp & 0x1F) << 10) | (mant_fp16 & 0x3FF);
			}
			else if (exp >= -24)
			{
				uint32	shift = (uint32) (-14 - exp);
				uint32	mantissa;
				if (shift > 24)
					shift = 24;
				mantissa = (mant | 0x800000) >> (shift + 13);

				if (((mant | 0x800000) >> (shift + 12)) & 1)
					mantissa += 1;
				out_fp16 = (sign << 15) | mantissa;
			}
			else
			{
				out_fp16 = (sign << 15);
			}

			dst[i] = out_fp16;
		}
	}

	PG_RETURN_BYTEA_P(out);
}

/*
 * vector_to_binary_gpu
 * SQL-callable: Convert float32 vector to packed bitstring:
 * 1 bit for each value: set if > 0.0f.
 */
PG_FUNCTION_INFO_V1(vector_to_binary_gpu);
Datum
vector_to_binary_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *v = PG_GETARG_VECTOR_P(0);
	int			count = v->dim;
	int			out_bytes = (count + 7) / 8;
	bytea	   *out;

	out = (bytea *) palloc(VARHDRSZ + out_bytes);
	SET_VARSIZE(out, VARHDRSZ + out_bytes);

	/* Clear all bits (avoid garbage bits in final bytes). */
	memset(VARDATA(out), 0, out_bytes);

	if (ndb_gpu_can_run("quantize"))
	{
		neurondb_gpu_quantize_binary(v->data, (uint8 *) VARDATA(out), count);
	}
	else
	{
		int			i;
		uint8	   *dst = (uint8 *) VARDATA(out);

		for (i = 0; i < count; i++)
		{
			/* Set bit if positive. */
			if (v->data[i] > 0.0f)
				dst[i >> 3] |= (1u << (i & 7));
		}
	}

	PG_RETURN_BYTEA_P(out);
}

/*
 * hnsw_knn_search_gpu
 * Placeholder: Not implemented. Throws error on use.
 */
PG_FUNCTION_INFO_V1(hnsw_knn_search_gpu);
Datum
hnsw_knn_search_gpu(PG_FUNCTION_ARGS)
{
	(void) PG_GETARG_VECTOR_P(0);
	(void) PG_GETARG_INT32(1);
	if (PG_NARGS() > 2)
		(void) PG_GETARG_INT32(2);

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("hnsw_knn_search_gpu is not implemented in this build")));
	PG_RETURN_NULL();
}

/*
 * ivf_knn_search_gpu
 * Placeholder: Not implemented. Throws error on use.
 */
PG_FUNCTION_INFO_V1(ivf_knn_search_gpu);
Datum
ivf_knn_search_gpu(PG_FUNCTION_ARGS)
{
	(void) PG_GETARG_VECTOR_P(0);
	(void) PG_GETARG_INT32(1);
	if (PG_NARGS() > 2)
		(void) PG_GETARG_INT32(2);

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("ivf_knn_search_gpu is not implemented in this build")));
	PG_RETURN_NULL();
}
