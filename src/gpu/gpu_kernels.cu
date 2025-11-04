/*-------------------------------------------------------------------------
 *
 * gpu_kernels.cu
 *    CUDA/HIP kernels for NeurondB GPU acceleration
 *
 * This file provides detailed, robust, and high-performance implementations
 * for GPU-accelerated computation, including distance metrics (L2, cosine,
 * inner product), quantization to binary, int8, fp16 formats, and KMeans
 * clustering steps (assignment and centroid update). All kernels leverage
 * architecture-specific intrinsics where available, explicit restrict
 * qualifiers for load/store efficiency, and are designed for maximally
 * predictable numerical stability, thread safety, and correctness.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/gpu_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#ifdef NDB_GPU_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#endif

#ifdef NDB_GPU_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

#include <float.h>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// L2 distance kernel: Each thread computes the L2 norm (Euclidean distance)
// between a query vector and a matrix row, storing the result in out[row].
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void l2_distance_kernel(const float * __restrict__ query,
                        const float * __restrict__ matrix,
                        float * __restrict__ out,
                        int rows,
                        int dim)
{
    // Calculate the global row index this thread is responsible for.
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (__builtin_expect(row < rows, 1)) // Branch prediction for performance
    {
        // Pointer to the start of the matrix row being compared.
        const float *vec = matrix + row * dim;

        // Use double for accumulation to reduce numerical drift in large dims.
        double acc = 0.0;
        for (int d = 0; d < dim; ++d)
        {
            float diff = query[d] - vec[d];
            acc += (double)diff * (double)diff;
        }
        // sqrtf is fine for float-out; casting from double accumulator.
        out[row] = sqrtf((float)acc);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Cosine distance kernel: Each thread computes
// cosine distance = 1 - dot(query, vec) / (||query|| * ||vec||)
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void cosine_distance_kernel(const float * __restrict__ query,
                            const float * __restrict__ matrix,
                            float * __restrict__ out,
                            int rows,
                            int dim)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (__builtin_expect(row < rows, 1)) 
    {
        const float *vec = matrix + row * dim;
        double dot = 0.0, n_q = 0.0, n_v = 0.0;

        // Loop unrolling for higher dimensions.
        int d = 0;
        for (; d <= dim - 4; d += 4)
        {
            float q0 = query[d  ], v0 = vec[d  ];
            float q1 = query[d+1], v1 = vec[d+1];
            float q2 = query[d+2], v2 = vec[d+2];
            float q3 = query[d+3], v3 = vec[d+3];
            dot += (double)q0 * v0 + (double)q1 * v1 + (double)q2 * v2 + (double)q3 * v3;
            n_q += (double)q0*q0 + (double)q1*q1 + (double)q2*q2 + (double)q3*q3;
            n_v += (double)v0*v0 + (double)v1*v1 + (double)v2*v2 + (double)v3*v3;
        }
        for (; d < dim; ++d) {
            float q = query[d], v = vec[d];
            dot += (double)q * v;
            n_q += (double)q * q;
            n_v += (double)v * v;
        }
        n_q = sqrt(n_q);
        n_v = sqrt(n_v);
        float cosine_sim = 0.0f;
        if (n_q > 1e-10 && n_v > 1e-10)
            cosine_sim = (float)(dot / (n_q * n_v));
        // Clamp result to the representable range due to precision issues.
        cosine_sim = fminf(fmaxf(cosine_sim, -1.0f), 1.0f);
        out[row] = 1.0f - cosine_sim;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Inner product kernel: Each thread computes -dot(query, vec)
// so that larger dot means smaller 'distance' for similarity search.
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void inner_product_kernel(const float * __restrict__ query,
                          const float * __restrict__ matrix,
                          float * __restrict__ out,
                          int rows,
                          int dim)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (__builtin_expect(row < rows, 1)) 
    {
        const float *vec = matrix + row * dim;
        double acc = 0.0;

        int d = 0;
        for (; d <= dim - 4; d += 4) // loop unroll
            acc += (double)query[d  ] * vec[d  ] +
                   (double)query[d+1] * vec[d+1] +
                   (double)query[d+2] * vec[d+2] +
                   (double)query[d+3] * vec[d+3];
        for (; d < dim; ++d)
            acc += (double)query[d] * vec[d];
        // Negative as per convention to turn similarity into distance.
        out[row] = (float)(-acc);
    }
}

////////////////////////////////////////////////////////////////////////////////
// FP16 quantization kernel: Convert float32 to float16 (IEEE 754 16-bit)
// using hardware intrinsics for CUDA/HIP (__float2half).
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void quantize_fp16_kernel(const float * __restrict__ input,
                          __half * __restrict__ output,
                          int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (__builtin_expect(idx < count, 1)) {
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_HIP)
        output[idx] = __float2half(input[idx]);
#else
        // Fallback: pack as unsigned 16-bit value (not recommended, unreachable)
        union { float f; unsigned u32; } u;
        u.f = input[idx];
        output[idx] = static_cast<__half>(u.u32 >> 16);
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////
// INT8 quantization kernel: Convert float32 to int8. Each value is multiplied
// by `scale`, clamped to [-128, 127], and then cast to int8.
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void quantize_int8_kernel(const float * __restrict__ input,
                          signed char * __restrict__ output,
                          int count,
                          float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (__builtin_expect(idx < count, 1)) {
        // Multiply by scale factor
        float val = input[idx] * scale;
        // Explicit clamp to int8 range (note int8 is [-128,127])
        if (val > 127.0f)
            val = 127.0f;
        else if (val < -128.0f)
            val = -128.0f;
        output[idx] = (signed char) __float2int_rn(val);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Binary quantization kernel: Each 8 input floats are packed into 1 byte.
// If input[i] > 0, that output bit is set to 1, else 0.
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void quantize_binary_kernel(const float * __restrict__ input,
                            unsigned char * __restrict__ output,
                            int count)
{
    // Each thread writes one output byte, representing 8 quantized bits.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int byte_count = (count + 7) / 8; // Total output bytes required.

    if (idx < byte_count)
    {
        unsigned char byte = 0;
        int start = idx * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            int in_idx = start + i;
            if (in_idx < count && input[in_idx] > 0.0f)
                byte |= (1 << i);
        }
        output[idx] = byte;
    }
}

////////////////////////////////////////////////////////////////////////////////
// KMeans assignment kernel: Each thread compares a vector in `vectors`
// with all centroids, computes squared L2 distance, and assigns it to
// its closest centroid in `assignments`.
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void kmeans_assign_kernel(const float * __restrict__ vectors,
                          const float * __restrict__ centroids,
                          int * __restrict__ assignments,
                          int num_vectors,
                          int num_centroids,
                          int dim)
{
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (__builtin_expect(vec_idx < num_vectors, 1))
    {
        const float *vec = vectors + vec_idx * dim;
        float min_dist = FLT_MAX;
        int best_centroid = 0;

        for (int c = 0; c < num_centroids; c++)
        {
            const float *centroid = centroids + c * dim;
            double dist = 0.0;
            int d = 0;
            for (; d <= dim - 4; d += 4)
            {
                float diff0 = vec[d  ] - centroid[d  ];
                float diff1 = vec[d+1] - centroid[d+1];
                float diff2 = vec[d+2] - centroid[d+2];
                float diff3 = vec[d+3] - centroid[d+3];
                dist += (double)diff0*diff0 + (double)diff1*diff1 +
                        (double)diff2*diff2 + (double)diff3*diff3;
            }
            for (; d < dim; ++d)
            {
                float diff = vec[d] - centroid[d];
                dist += (double)diff * diff;
            }
            if (dist < min_dist)
            {
                min_dist = (float)dist;
                best_centroid = c;
            }
        }
        assignments[vec_idx] = best_centroid;
    }
}

////////////////////////////////////////////////////////////////////////////////
// KMeans update kernel: Each warp of threads (1 block per centroid, 1 thread
// per dimension) accumulates sums for the centroid, then computes the new
// centroid coordinate, and updates member count if d==0.
// Expects centroids, counts to be zero-filled prior to kernel launch.
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void kmeans_update_kernel(const float * __restrict__ vectors,
                          const int * __restrict__ assignments,
                          float * __restrict__ centroids,
                          int * __restrict__ counts,
                          int num_vectors,
                          int num_centroids,
                          int dim)
{
    int c = blockIdx.x;   // Each block handles one centroid.
    int d = threadIdx.x;  // Each thread handles one dimension.

    if (c < num_centroids && d < dim)
    {
        // Accumulate sum of dimension d for centroid c.
        double sum = 0.0;
        int cnt = 0;
        for (int v = 0; v < num_vectors; ++v)
        {
            if (__builtin_expect(assignments[v] == c, 0.1f)) // usually sparse
            {
                sum += (double)vectors[v * dim + d];
                if (d == 0) ++cnt;
            }
        }
        if (cnt > 0)
        {
            centroids[c * dim + d] = (float)(sum / (double)cnt);
            if (d == 0)
                counts[c] = cnt;
        }
        else
        {
            // If centroid got no assignments, leave centroid unchanged (user must handle empty clusters)
        }
    }
}
