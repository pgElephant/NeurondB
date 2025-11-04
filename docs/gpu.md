````markdown
# GPU Acceleration

NeuronDB includes GPU-aware components initialized at server startup when compiled and configured accordingly.

## Overview

- GPU config is initialized when `shared_preload_libraries` loads the extension.
- In current builds, GPU acceleration covers vector distance and quantization helpers with automatic CPU fallback.

## Configuration

- Add `neurondb` to `shared_preload_libraries` and restart Postgres. This ensures GPU GUCs are registered and shared memory is allocated.
- Tune via GUCs (can be changed at runtime per session):

```sql
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_device = 0;          -- pick device
SET neurondb.gpu_batch_size = 8192;   -- throughput vs. latency
SET neurondb.gpu_fail_open = on;      -- fallback to CPU on error
```

Example postgresql.conf:

```conf
shared_preload_libraries = 'neurondb'
# neurondb.gpu_enabled = off
# neurondb.gpu_device = 0
# neurondb.gpu_batch_size = 8192
# neurondb.gpu_streams = 2
# neurondb.gpu_memory_pool_mb = 512
# neurondb.gpu_fail_open = on
# neurondb.gpu_kernels = 'l2,cosine,ip'
# neurondb.gpu_timeout_ms = 30000
```

## SQL usage

- Distances (GPU with CPU fallback): `vector_l2_distance_gpu`, `vector_cosine_distance_gpu`, `vector_inner_product_gpu`
- Quantization: `vector_to_int8_gpu`, `vector_to_fp16_gpu`, `vector_to_binary_gpu`

```sql
SELECT vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);
SELECT octet_length(vector_to_int8_gpu('[0.1,0.2,0.3]'::vector));
```

## Verification

- After restart, check Postgres logs for neurondb GPU initialization.
- Compare timings of distance/quantization operations vs CPU equivalents on your data.

## Notes

- Ensure compatible CUDA (or ROCm) drivers/toolkit are installed when building with GPU support.
- GPU clustering functions are planned and not yet available; use CPU `cluster_kmeans` / `cluster_minibatch_kmeans` in the meantime.

````
