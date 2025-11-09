/*-------------------------------------------------------------------------
 *
 * gpu_quantization.c
 *     Host-side quantization helpers used by GPU entry points.
 *
 * For now these implementations execute on the CPU so that the extension
 * continues to build without requiring CUDA/HIP kernels to be compiled.
 * Once proper GPU kernels are restored we can swap these functions to
 * dispatch to device code.
 *
 *-------------------------------------------------------------------------*/

#include "postgres.h"
#include "fmgr.h"

#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static uint16_t
float_to_half_bits(float value)
{
    union { float f; uint32_t u; } v = { .f = value };
    uint32_t sign = (v.u >> 16) & 0x8000;
    uint32_t mant = v.u & 0x7fffff;
    int32_t  exp  = ((int32_t)(v.u >> 23) & 0xff) - 127 + 15;

    if (exp <= 0)
    {
        if (exp < -10)
            return (uint16_t) sign;
        mant |= 0x800000;
        uint32_t shifted = mant >> (1 - exp);
        return (uint16_t)(sign | ((shifted + 0x1000) >> 13));
    }
    else if (exp >= 31)
    {
        return (uint16_t)(sign | 0x7c00);
    }

    return (uint16_t)(sign | (exp << 10) | ((mant + 0x1000) >> 13));
}

static inline int8_t
quantize_int8_value(float value, float scale)
{
    float scaled = roundf(value * scale);
    if (scaled > 127.0f)
        scaled = 127.0f;
    else if (scaled < -128.0f)
        scaled = -128.0f;
    return (int8_t) scaled;
}

void
neurondb_gpu_quantize_fp16(const float *input, void *output, int count)
{
    const ndb_gpu_backend *backend;

    if (input == NULL || output == NULL || count <= 0)
        return;

    if (neurondb_gpu_is_available())
    {
        backend = ndb_gpu_get_active_backend();
        if (backend && backend->launch_quant_fp16)
        {
            if (backend->launch_quant_fp16(input, output, count, NULL) == 0)
                return;
            elog(DEBUG1, "neurondb: GPU quantize_fp16 failed, using CPU fallback");
        }
    }

    uint16_t *dst = (uint16_t *) output;
    for (int i = 0; i < count; i++)
        dst[i] = float_to_half_bits(input[i]);
}

void
neurondb_gpu_quantize_int8(const float *input, int8 *output, int count)
{
    const ndb_gpu_backend *backend;

    if (input == NULL || output == NULL || count <= 0)
        return;

    float max_abs = 0.0f;
    for (int i = 0; i < count; i++)
    {
        float abs_val = fabsf(input[i]);
        if (abs_val > max_abs)
            max_abs = abs_val;
    }

    float scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 1.0f;

    if (neurondb_gpu_is_available())
    {
        backend = ndb_gpu_get_active_backend();
        if (backend && backend->launch_quant_int8)
        {
            if (backend->launch_quant_int8(input, (int8_t *) output, count, scale, NULL) == 0)
                return;
            elog(DEBUG1, "neurondb: GPU int8 quantization failed; using CPU fallback");
        }
    }

    for (int i = 0; i < count; i++)
        output[i] = quantize_int8_value(input[i], scale);
}

void
neurondb_gpu_quantize_binary(const float *input, uint8 *output, int count)
{
    const ndb_gpu_backend *backend;

    if (input == NULL || output == NULL || count <= 0)
        return;

    if (neurondb_gpu_is_available())
    {
        backend = ndb_gpu_get_active_backend();
        if (backend && backend->launch_quant_binary)
        {
            if (backend->launch_quant_binary(input, (uint8_t *) output, count, NULL) == 0)
                return;
            elog(DEBUG1, "neurondb: GPU binary quantization failed; using CPU fallback");
        }
    }

    int out_bytes = (count + 7) / 8;
    memset(output, 0, out_bytes);

    for (int idx = 0; idx < count; idx++)
    {
        if (input[idx] > 0.0f)
            output[idx / 8] |= (1u << (idx % 8));
    }
}

