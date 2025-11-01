#include "postgres.h"
#include "fmgr.h"
#include "utils/guc.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"

#include "neurondb_config.h"
#include "neurondb_gpu.h"
#include "neurondb.h"


/* Helper: check if kernel is enabled and GPU usable */
static inline bool
ndb_gpu_can_run(const char *kernel_name)
{
	if (!neurondb_gpu_enabled)
		return false;
	if (!ndb_gpu_kernel_enabled(kernel_name))
		return false;
	ndb_gpu_init_if_needed();
	return neurondb_gpu_is_available();
}

/* vector_l2_distance_gpu(a, b) RETURNS real */
PG_FUNCTION_INFO_V1(vector_l2_distance_gpu);
Datum
vector_l2_distance_gpu(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	float4 result;

	if (ndb_gpu_can_run("l2") && a->dim == b->dim)
	{
		result = neurondb_gpu_l2_distance(a->data, b->data, a->dim);
		if (result >= 0.0f)
			PG_RETURN_FLOAT4(result);
	}

	/* CPU fallback */
	extern float4 l2_distance(Vector *a, Vector *b);
	PG_RETURN_FLOAT4(l2_distance(a, b));
}

/* vector_cosine_distance_gpu(a, b) */
PG_FUNCTION_INFO_V1(vector_cosine_distance_gpu);
Datum
vector_cosine_distance_gpu(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	float4 result;

	if (ndb_gpu_can_run("cosine") && a->dim == b->dim)
	{
		result = neurondb_gpu_cosine_distance(a->data, b->data, a->dim);
		if (result >= 0.0f)
			PG_RETURN_FLOAT4(result);
	}

	extern float4 cosine_distance(Vector *a, Vector *b);
	PG_RETURN_FLOAT4(cosine_distance(a, b));
}

/* vector_inner_product_gpu(a, b) */
PG_FUNCTION_INFO_V1(vector_inner_product_gpu);
Datum
vector_inner_product_gpu(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	float4 result;

	if (ndb_gpu_can_run("ip") && a->dim == b->dim)
	{
		result = neurondb_gpu_inner_product(a->data, b->data, a->dim);
		if (result >= 0.0f)
			PG_RETURN_FLOAT4(result);
	}

	extern float4 inner_product_distance(Vector *a, Vector *b);
	PG_RETURN_FLOAT4(inner_product_distance(a, b));
}

/* Quantization wrappers - return bytea buffers */
PG_FUNCTION_INFO_V1(vector_to_int8_gpu);
Datum
vector_to_int8_gpu(PG_FUNCTION_ARGS)
{
	Vector *v = PG_GETARG_VECTOR_P(0);
	int count = v->dim;
	bytea *out = (bytea *) palloc(VARHDRSZ + count);
	SET_VARSIZE(out, VARHDRSZ + count);

	if (ndb_gpu_can_run("quantize"))
		neurondb_gpu_quantize_int8(v->data, (int8 *) VARDATA(out), count);
	else
	{
		/* CPU fallback: simple scaling to INT8 */
		int i; float maxv = 0.0f;
		for (i = 0; i < count; i++)
			if (fabsf(v->data[i]) > maxv) maxv = fabsf(v->data[i]);
		float scale = (maxv > 0.0f) ? 127.0f / maxv : 1.0f;
		for (i = 0; i < count; i++)
		{
			float val = v->data[i] * scale;
			if (val > 127.0f) val = 127.0f;
			if (val < -128.0f) val = -128.0f;
			((int8 *) VARDATA(out))[i] = (int8) val;
		}
	}
	PG_RETURN_BYTEA_P(out);
}

PG_FUNCTION_INFO_V1(vector_to_fp16_gpu);
Datum
vector_to_fp16_gpu(PG_FUNCTION_ARGS)
{
	Vector *v = PG_GETARG_VECTOR_P(0);
	int count = v->dim;
	int out_bytes = count * 2; /* half-precision */
	bytea *out = (bytea *) palloc(VARHDRSZ + out_bytes);
	SET_VARSIZE(out, VARHDRSZ + out_bytes);

	if (ndb_gpu_can_run("quantize"))
		neurondb_gpu_quantize_fp16(v->data, (void *) VARDATA(out), count);
	else
	{
		/* CPU fallback: naive fp32 to fp16 conversion (approx) */
		int i; uint16 *dst = (uint16 *) VARDATA(out);
		for (i = 0; i < count; i++)
		{
			float f = v->data[i];
			/* very rough float32 -> float16 conversion */
			union { float f; uint32 u; } in; in.f = f;
			uint32 sign = (in.u >> 31) & 0x1;
			int32  exp  = ((in.u >> 23) & 0xFF) - 127 + 15;
			uint32 mant = (in.u >> 13) & 0x3FF;
			if (exp <= 0) { exp = 0; mant = 0; }
			if (exp >= 31) { exp = 31; mant = 0; }
			dst[i] = (uint16)((sign << 15) | ((exp & 0x1F) << 10) | (mant & 0x3FF));
		}
	}
	PG_RETURN_BYTEA_P(out);
}

PG_FUNCTION_INFO_V1(vector_to_binary_gpu);
Datum
vector_to_binary_gpu(PG_FUNCTION_ARGS)
{
	Vector *v = PG_GETARG_VECTOR_P(0);
	int count = v->dim;
	int out_bytes = (count + 7) / 8;
	bytea *out = (bytea *) palloc(VARHDRSZ + out_bytes);
	SET_VARSIZE(out, VARHDRSZ + out_bytes);
	memset(VARDATA(out), 0, out_bytes);

	if (ndb_gpu_can_run("quantize"))
		neurondb_gpu_quantize_binary(v->data, (uint8 *) VARDATA(out), count);
	else
	{
		int i; uint8 *dst = (uint8 *) VARDATA(out);
		for (i = 0; i < count; i++)
		{
			if (v->data[i] > 0.0f)
				dst[i >> 3] |= (1u << (i & 7));
		}
	}
	PG_RETURN_BYTEA_P(out);
}

/* ANN search GPU entry points with CPU fallback to existing functions */

PG_FUNCTION_INFO_V1(hnsw_knn_search_gpu);
Datum
hnsw_knn_search_gpu(PG_FUNCTION_ARGS)
{
    (void) PG_GETARG_VECTOR_P(0);
    (void) PG_GETARG_INT32(1);
    if (PG_NARGS() > 2)
        (void) PG_GETARG_INT32(2);
    ereport(ERROR, (errmsg("hnsw_knn_search_gpu is not implemented in this build")));
    PG_RETURN_NULL();
}

PG_FUNCTION_INFO_V1(ivf_knn_search_gpu);
Datum
ivf_knn_search_gpu(PG_FUNCTION_ARGS)
{
    (void) PG_GETARG_VECTOR_P(0);
    (void) PG_GETARG_INT32(1);
    if (PG_NARGS() > 2)
        (void) PG_GETARG_INT32(2);
    ereport(ERROR, (errmsg("ivf_knn_search_gpu is not implemented in this build")));
    PG_RETURN_NULL();
}
