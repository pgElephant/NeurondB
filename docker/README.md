# NeuronDB Docker Images

This directory contains reproducible Docker builds for NeuronDB that mirror the
packaging approach used by PostgresML. Two variants are provided:

- `docker/Dockerfile` – CPU-focused image based on `postgres:17-bookworm`
- `docker/Dockerfile.gpu` – CUDA-enabled image built from
  `nvidia/cuda:<version>-devel-ubuntu22.04`

Both images bundle ONNX Runtime, pre-enable the NeuronDB extension, and expose a
ready-to-use PostgreSQL server.

## Quick Start

### CPU Image

```bash
cd Neurondb/docker
docker compose build neurondb         # or: docker build -f docker/Dockerfile ..
docker compose up neurondb
```

Connect:

```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c '\dx neurondb'
```

### GPU Image

Requirements: NVIDIA driver ≥ 535, `nvidia-container-toolkit`, Docker Compose v2.

```bash
docker compose --profile gpu build neurondb-gpu \
  --build-arg ENABLE_RAPIDS=1   # optional cuML/XGBoost GPU support
docker compose --profile gpu up neurondb-gpu
```

To verify GPU visibility:

```bash
docker exec -it neurondb-gpu nvidia-smi
psql "postgresql://neurondb:neurondb@localhost:5434/neurondb" \
  -c "SELECT neurondb.gpu_device_info();"
```

## Build Arguments

| Argument        | Default | Description |
|-----------------|---------|-------------|
| `PG_MAJOR`      | `17`    | PostgreSQL major version (16, 17 or 18) |
| `ONNX_VERSION`  | `1.17.0`| ONNX Runtime release to embed |
| `CUDA_VERSION`  | `12.4.1`| (GPU) CUDA base image tag |
| `ENABLE_RAPIDS` | `0`     | (GPU) Install RAPIDS cuML / XGBoost GPU wheels |

When `ENABLE_RAPIDS=1`, the build will automatically detect and pass the cuML &
XGBoost include/library paths into `make`, enabling GPU-accelerated AutoML
pipelines.

## Configuration

During `initdb`, the container appends the following defaults to
`postgresql.conf`:

```conf
shared_preload_libraries = 'neurondb'
neurondb.gpu_enabled = off
neurondb.automl.use_gpu = off
```

Change them at runtime with `ALTER SYSTEM` or by editing
`$PGDATA/postgresql.conf`.

An initialization SQL script (`20_create_neurondb.sql`) ensures that the
extension is available in the default database on first boot.

## Lifecycle

- Images are multi-stage, so the final layer only includes the compiled module,
  SQL extension files, ONNX Runtime, and optional CUDA/cuML runtimes.
- `docker/docker-compose.yml` exposes ports `5433` (CPU) and `5434` (GPU) with
  persistent volumes `neurondb-data` / `neurondb-gpu-data`.
- Health checks rely on `pg_isready` for straightforward orchestration.

## Notes

- The GPU image sets `runtime: nvidia` and requests one GPU via Compose – adjust
  for your environment as needed.
- Ensure build hosts have adequate disk space (~4 GB) while compiling the CUDA
  toolchain and RAPIDS libraries.
- Generated images inherit the upstream PostgreSQL entrypoint; standard
  environment variables (`POSTGRES_USER`, `POSTGRES_PASSWORD`, etc.) continue to
  work unchanged.

