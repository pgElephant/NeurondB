# NeurondB GPU Acceleration

## Overview

NeurondB provides optional GPU acceleration for compute-intensive vector operations using NVIDIA CUDA or AMD ROCm. GPU support is **optional** and automatically falls back to CPU if unavailable.

## Features

### GPU-Accelerated Operations

| Operation | CUDA | ROCm | Fallback |
|-----------|------|------|----------|
| L2 Distance | ✓ (cuBLAS) | ✓ (rocBLAS) | CPU |
| Cosine Distance | ✓ (cuBLAS) | ✓ (rocBLAS) | CPU |
| Inner Product | ✓ (cuBLAS) | ✓ (rocBLAS) | CPU |
| Batch Distance Matrix | ✓ (GEMM) | ✓ (GEMM) | CPU |
| INT8 Quantization | ✓ (Custom kernel) | ✓ (Custom kernel) | CPU |
| FP16 Quantization | ✓ (Custom kernel) | ✓ (Custom kernel) | CPU |
| Binary Quantization | ✓ (Custom kernel) | ✓ (Custom kernel) | CPU |
| K-Means Clustering | ✓ (cuML optional) | ✓ (Custom) | CPU |
| HNSW k-NN Search | ✓ (Batch distance) | ✓ (Batch distance) | CPU |
| IVF k-NN Search | ✓ (Batch distance) | ✓ (Batch distance) | CPU |
| ONNX Inference | ✓ (CUDA EP) | Partial | CPU |

## Building with GPU Support

### Prerequisites

**NVIDIA GPU (CUDA):**
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6

# Rocky Linux
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install -y cuda-toolkit-12-6
```

**AMD GPU (ROCm):**
```bash
# Ubuntu
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install -y --usecase=rocm

# Rocky Linux
sudo dnf install -y rocm-hip-sdk rocm-libs
```

### Build Commands

**CPU-Only (Default):**
```bash
./build.sh
```

**With GPU Support:**
```bash
./build.sh --with-gpu
```

**Custom GPU Paths:**
```bash
./build.sh --with-gpu --cuda-path /opt/cuda --onnx-path /usr/local
```

**Or with make directly:**
```bash
make CUDA_PATH=/usr/local/cuda ROCM_PATH=/opt/rocm
sudo make install
```

## Configuration

### GUC Variables

Add to `postgresql.conf`:

```ini
# Load NeurondB extension
shared_preload_libraries = 'neurondb'

# GPU Configuration (all optional)
neurondb.gpu_enabled = off                    # Enable GPU (default: off)
neurondb.gpu_device = 0                       # GPU device ID (default: 0)
neurondb.gpu_batch_size = 8192                # Batch size (default: 8192)
neurondb.gpu_streams = 2                      # CUDA/HIP streams (default: 2)
neurondb.gpu_memory_pool_mb = 512             # Memory pool MB (default: 512)
neurondb.gpu_fail_open = on                   # Fallback to CPU on error (default: on)
neurondb.gpu_kernels = 'l2,cosine,ip'         # Enabled kernels (default: all)
neurondb.gpu_timeout_ms = 30000               # Kernel timeout ms (default: 30000)
```

### Runtime Control

```sql
-- Enable GPU for session
SET neurondb.gpu_enabled = on;

-- Or enable programmatically
SELECT neurondb_gpu_enable(true);

-- Check GPU info
SELECT * FROM neurondb_gpu_info();

-- Monitor GPU usage
SELECT * FROM pg_stat_neurondb_gpu;
```

## Usage Examples

### GPU Distance Operations

```sql
-- Enable GPU
SET neurondb.gpu_enabled = on;

-- GPU-accelerated distance (explicit)
SELECT vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);

-- GPU-accelerated k-NN search
SELECT * FROM hnsw_knn_search_gpu('[0.1, 0.2, 0.3]'::vector, 10, 128);

-- Batch quantization
SELECT vector_to_int8_gpu(embedding) FROM documents WHERE id < 1000;
```

### GPU Clustering

```sql
-- K-Means clustering on GPU
SELECT * FROM cluster_kmeans_gpu('embeddings', 'vec', 5, 100);
```

### Performance Monitoring

```sql
-- Check GPU stats
SELECT 
    queries_executed,
    fallback_count,
    avg_latency_ms,
    (total_gpu_time_ms / NULLIF(queries_executed, 0)) as avg_gpu_ms
FROM neurondb_gpu_stats();

-- Reset stats
SELECT neurondb_gpu_stats_reset();
```

## Architecture

### Per-Backend Initialization

- GPU context initialized **once per PostgreSQL backend process**
- No shared GPU contexts across backends (PostgreSQL process model compliant)
- Lazy initialization on first GPU function call
- Automatic cleanup on backend exit

### CPU Fallback Strategy

```
If GPU disabled        → Always use CPU
If GPU enabled:
  ├─ GPU init fails    → Log WARNING, use CPU, set gpu_disabled flag
  ├─ GPU OOM          → Log WARNING, reduce batch size, retry, then CPU
  ├─ Kernel timeout   → Abort kernel, use CPU for that operation
  └─ Device error     → Log WARNING, use CPU, increment fallback_count
```

### Error Handling

**Fail-Open Mode (`neurondb.gpu_fail_open = on`, default):**
- Log `WARNING` on GPU errors
- Fallback to CPU automatically
- Record in `fallback_count` statistic

**Fail-Closed Mode (`neurondb.gpu_fail_open = off`):**
- Raise `ERROR` on GPU initialization failure
- Do not fallback to CPU
- Used when GPU is required for SLA

## Performance

### Expected Performance (NVIDIA A100)

| Operation | Dataset | Performance | CPU Baseline |
|-----------|---------|-------------|--------------|
| L2 Distance Batch | 1M vectors, dim=768 | ~10ms | ~200ms |
| HNSW k-NN Search | 1M vectors, k=10, ef=64 | ~8ms | ~50ms |
| INT8 Quantization | 1M vectors, dim=768 | ~5ms | ~80ms |
| K-Means (10 iters) | 100K vectors, k=5, dim=768 | ~150ms | ~3000ms |

### Optimization Tips

1. **Batch Operations**: GPU performs best with batch sizes ≥ 4096
2. **Warm-up**: First query initializes GPU (adds ~100-500ms latency)
3. **Memory Pool**: Increase `gpu_memory_pool_mb` for large datasets
4. **Streams**: Use 2-4 streams for copy/compute overlap

## Troubleshooting

### GPU Not Detected

```sql
SELECT * FROM neurondb_gpu_info();
-- Returns empty or device_id = -1
```

**Solutions:**
1. Check CUDA/ROCm installation: `nvidia-smi` or `rocm-smi`
2. Verify library paths: `ldconfig -p | grep cuda`
3. Rebuild with `--with-gpu` flag
4. Check build log for GPU detection messages

### High Fallback Count

```sql
SELECT fallback_count FROM neurondb_gpu_stats();
```

**Possible Causes:**
- GPU OOM (increase `gpu_memory_pool_mb` or reduce `gpu_batch_size`)
- Device busy (reduce concurrent backends)
- Driver issues (check `dmesg` for GPU errors)

### Poor Performance

1. Check batch size: `SHOW neurondb.gpu_batch_size;`
2. Monitor GPU utilization: `nvidia-smi` or `rocm-smi`
3. Check if GPU is actually being used: `SELECT * FROM pg_stat_neurondb_gpu;`
4. Verify kernel is enabled: `SHOW neurondb.gpu_kernels;`

## Limitations

1. **Single-threaded**: Each backend has one GPU context (PostgreSQL design)
2. **No SPI in GPU paths**: Do not call GPU functions from within SPI loops
3. **No long locks**: Avoid GPU operations while holding table locks
4. **Batch size limits**: Maximum batch size limited by GPU memory

## Development

### Testing GPU Code

```bash
# Run regression tests
./build.sh --with-gpu --test

# Run specific GPU tests
make installcheck REGRESS=09_gpu_features

# Run TAP tests
prove t/010_gpu_features.pl
```

### Debugging

Enable verbose GPU logging:
```sql
SET client_min_messages = DEBUG1;
SET neurondb.gpu_enabled = on;
-- GPU init messages will appear in logs
```

## Security

- No `SECURITY DEFINER` on GPU functions
- GPU memory isolated per backend process
- Resource limits enforced via GUCs
- Timeout protection via `gpu_timeout_ms`

## References

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [cuBLAS Library](https://docs.nvidia.com/cuda/cublas/)
- [ONNX Runtime CUDA Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

