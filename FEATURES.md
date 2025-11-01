# NeurondB Features

A production-grade AI/Vector extension for PostgreSQL focused on performance, safety, and operational excellence.

## Supported PostgreSQL Versions

- 16.x (supported)
- 17.x (supported)
- 18.x (supported)

Note: Builds fail fast on unsupported versions.

## Core Capabilities

### Vector Types
- float32 dense vectors (`vector`)
- float16 packed vectors (`VectorF16`)
- int8 quantized vectors (`VectorI8`)
- binary vectors (`VectorBinary`)
- packed/sparse/advanced types for specialized storage (`types_core.c`)

### Distance Metrics
- L2 (Euclidean)
- Cosine distance
- Inner product (negative dot for similarity ordering)
- L1 (Manhattan)
- Chebyshev
- Minkowski (general p-norm)

### Indexing & Search
- HNSW index with tunable parameters (M, ef_build, ef_search)
- IVF index with centroid probing (nlist, nprobe)
- Hybrid search: combine vector similarity + FTS + metadata filters
- Multi-vector use-cases (support in hybrid layer)

### Quantization & Compression
- FP16 (2x memory reduction)
- INT8 (8x memory reduction)
- Binary packing (up to 32x)
- CPU paths always available; GPU acceleration optional

### Analytics
- K-Means clustering (GPU/CPU)
- DBSCAN (CPU; GPU hooks present)
- PCA (CPU; GPU hooks present)

### ML / Inference
- ONNX Runtime integration (CPU baseline; GPU provider optional)
- Embedding/reranking entry points via SQL API

## GPU Acceleration (Optional)

GPU support is off by default and automatically falls back to CPU when unavailable or disabled.

- Backends: CUDA (NVIDIA), ROCm (AMD)
- Initialization: lazy per-backend; no shared GPU contexts across backends
- Memory: device buffer pool with pinned host buffers
- Parallelism: multi-stream copy/compute overlap
- Error policy: fail-open by default (warnings + fallback)

### GPU GUCs
- `neurondb.gpu_enabled` (boolean, default off)
- `neurondb.gpu_device` (int, default 0)
- `neurondb.gpu_batch_size` (int, default 8192)
- `neurondb.gpu_streams` (int, default 2)
- `neurondb.gpu_memory_pool_mb` (int, default 512)
- `neurondb.gpu_fail_open` (boolean, default on)
- `neurondb.gpu_kernels` (text list, default: l2,cosine,ip)
- `neurondb.gpu_timeout_ms` (int, default 30000)

### GPU SQL API
- Control/Info:
  - `neurondb_gpu_enable(boolean)`
  - `neurondb_gpu_info()`
  - `neurondb_gpu_stats()` / `neurondb_gpu_stats_reset()`
  - View: `pg_stat_neurondb_gpu`
- Distance (explicit GPU overrides):
  - `vector_l2_distance_gpu(vector, vector) returns real`
  - `vector_cosine_distance_gpu(vector, vector) returns real`
  - `vector_inner_product_gpu(vector, vector) returns real`
- Quantization:
  - `vector_to_int8_gpu(vector) returns bytea`
  - `vector_to_fp16_gpu(vector) returns bytea`
  - `vector_to_binary_gpu(vector) returns bytea`
- ANN entry points (stubs for now; explicit error if called):
  - `hnsw_knn_search_gpu(vector, int, int default 100)`
  - `ivf_knn_search_gpu(vector, int, int default 10)`

## Background Workers

Three safe-by-default workers for async work; all controlled via GUCs and registered with `shared_preload_libraries`.

- `neuranq` (Queue executor): job queue with SKIP LOCKED; rate limits, retries, poison job handling
- `neuranmon` (Auto-tuner): adjusts search params from SLOs; rotates caches; records recall@k
- `neurandefrag` (Index care): compacts graphs, re-levels, prunes tombstones, schedules rebuilds

Operational notes:
- Dynamic shared memory for queues/stats; protected by LWLocks
- Per-tenant lanes with QPS/token/cost budgets
- Structured JSON logging with `neurondb:` prefix

## Observability & Admin

- `pg_stat_neurondb` views (core stats)
- `pg_stat_neurondb_gpu` view (device_id, backend_pid, queries, batches, avg_batch_rows, avg_latency_ms, fallback_count, oom_count, last_error)
- Worker heartbeats and watchdog views (tuner/queue/defrag)

## Security & Governance

- CPU/GPU functions run as normal user functions (no SECURITY DEFINER)
- Designed to work with RLS and tenant isolation patterns
- No long-held locks in GPU paths; no SPI during GPU kernels
- Configurable fail-open/fail-closed behavior for GPU

## SQL Highlights (non-exhaustive)

Distance:
- `vector_l2_distance(a, b)`
- `vector_cosine_distance(a, b)`
- `vector_inner_product(a, b)`

Hybrid & Rerank:
- `hybrid_search(table, vec, query_text, filters, weight, limit)`
- `rerank_cross_encoder(query, docs[], model, top_n)`

Quantization:
- `quantize_vector_i8(vector)` / `quantize_vector_f16(vector)` / `quantize_vector_binary(vector)`
- GPU variants as above

Analytics:
- `cluster_kmeans(table, col, k, iters)`
- `cluster_kmeans_gpu(table, col, k, iters)`

Admin/Stats:
- `neurondb_gpu_enable(bool)`
- `neurondb_gpu_info()` / `neurondb_gpu_stats()` / `neurondb_gpu_stats_reset()`

## Configuration Summary

General:
- Worker enable/disable and tunables
- Index creation parameters (HNSW/IVF)
- Hybrid weights and thresholds

GPU (optional): see GPU GUCs above

## Build & Compatibility

- PostgreSQL 16/17/18 only
- CPU-only by default
- Optional GPU build via `./build.sh --with-gpu` (auto-detects CUDA/ROCm)
- No hard-coded paths; uses `pg_config`

## Roadmap (Highlights)

- Planner integration for GPU paths (CustomScan & cost-based selection)
- GPU ANN paths for HNSW/IVF when candidates exceed thresholds
- Extended analytics (UMAP, t-SNE) and advanced quantization

# NeuronDB Features

**Status:** Development - Core functionality implemented, advanced features in progress

See [TODO.md](TODO.md) for detailed implementation roadmap.

---

## Core Vector Types

### Basic Types (✅ Implemented)
- **vectorf32**: Float32 vectors (full precision) - I/O, operations, storage
- **vectorf16**: Float16 quantized vectors (2x compression) - I/O, basic ops
- **vectori8**: INT8 quantized vectors (8x compression) - I/O, basic ops
- **vectorbin**: Binary vectors (32x compression) - I/O, basic ops

### Advanced Types (🔄 Partial)
- **vectorp**: Packed SIMD vector with validation metadata - structure defined
- **vecmap**: Sparse high-dimensional maps (>10K dimensions) - structure defined
- **rtext**: Retrievable text with token offsets - structure defined
- **vgraph**: Compact graph storage - structure defined

---

## Distance Metrics (✅ Implemented)

All distance functions have basic C implementations:
- L2 Distance (Euclidean)
- Cosine Distance
- Inner Product
- L1 Distance (Manhattan)
- Hamming Distance (binary)
- Jaccard Distance
- Minkowski Distance
- Chebyshev Distance

**Note:** SIMD optimizations (AVX2, AVX-512, NEON) are planned.

---

## Vector Operations (✅ Implemented)

- Addition, Subtraction, Multiplication
- Normalization
- Concatenation
- Slicing
- Dot Product
- Magnitude
- Comparison operators

---

## Indexing Methods (🔄 Partial)

### HNSW (Hierarchical Navigable Small World)
- ⏳ Full `IndexAmRoutine` implementation (planned)
- ⏳ Layer construction and neighbor selection (planned)
- 🔄 Multi-tenant aware indexes (structure exists)
- 🔄 Configurable `ef_construction` and `ef_search` (planned)

### IVF (Inverted File Index)
- ⏳ Full `IndexAmRoutine` implementation (planned)
- ⏳ Cluster training with k-means (planned)
- ⏳ Quantization support (planned)

### Hybrid Indexes (🔄 Partial)
- 🔄 Fused ANN + GIN full-text search (CustomScan planned)
- 🔄 Temporal vector indexes (structure exists)
- 🔄 Consistent query indexes (CQ-HNSW) (structure exists)
- 🔄 Rerank-ready indexes (structure exists)

---

## Machine Learning Integration (🔄 Partial)

- 🔄 **Model Inference**: HTTP-based model calls via libcurl (basic implementation)
- 🔄 **LLM Integration**: Token counting and cost tracking (tables defined)
- 🔄 **Embedding Generation**: Text to vector conversion (basic implementation)
- 🔄 **Hybrid Search**: Combined vector + full-text search (SPI-based)
- 🔄 **Reranking**: Cross-encoder, LLM, ColBERT algorithms (basic structure)

---

## Analytics (🔄 Partial)

- 🔄 K-means clustering (basic SPI-based implementation)
- 🔄 DBSCAN clustering (basic SPI-based implementation)
- 🔄 Outlier/anomaly detection (basic implementation)
- 🔄 Topic modeling (basic implementation)
- 🔄 Embedding quality metrics (basic implementation)

---

## Data Management (🔄 Partial)

- 🔄 Vector time-travel (MVCC-aware structure)
- 🔄 Cold-tier compression (basic implementation)
- 🔄 Vector-aware VACUUM (planned)
- 🔄 Index rebalance API (basic structure)

---

## Multi-Tenant & Governance (🔄 Partial)

- 🔄 Tenant-scoped background workers (structure exists, needs BGW registration)
- ✅ Usage metering (`pg_stat_neurondb`) (shared memory stats implemented)
- 🔄 Policy engine (basic SPI-based implementation)
- 🔄 Audit logging (table structure with HMAC support)

---

## Security & Privacy (🔄 Partial)

- 🔄 Vector encryption (AES-GCM via OpenSSL, basic implementation)
- 🔄 Differential privacy for embeddings (noise addition structure)
- ⏳ Row-level security for vectors (PostgreSQL RLS policies planned)
- 🔄 Signed results (HMAC-SHA256 via OpenSSL)

---

## Performance Features (🔄 Partial)

- 🔄 **ANN Buffer**: In-memory cache for hot centroids (shared memory structure)
- 🔄 **WAL Compression**: Delta encoding for vector updates (placeholder hooks)
- ⏳ **Parallel Execution**: Multi-worker kNN search (CustomScan planned)
- 🔄 **Predictive Prefetching**: Smart entry point loading (basic implementation)

---

## Adaptive Intelligence (🔄 Partial)

- 🔄 Auto-routing planner hook (structure exists, needs `planner_hook` registration)
- 🔄 Self-learning query optimizer (query fingerprint tracking)
- 🔄 Dynamic precision scaling (basic implementation)
- 🔄 Predictive prefetcher (pattern tracking structure)

---

## Distributed Features (🔄 Partial)

- 🔄 Shard-aware ANN execution (basic routing logic)
- 🔄 Cross-node recall guarantees (merge logic structure)
- 🔄 Vector load balancer (latency tracking)
- 🔄 Async index synchronization (planned)

---

## Observability (✅ Implemented)

- ✅ `pg_stat_neurondb`: Native statistics view with shared memory counters
- ✅ Query latency tracking (histogram structure)
- ✅ Recall@K tracking (sampling structure)
- ✅ Cache hit rates (counter tracking)
- 🔄 Model cost tracking (basic accounting)

---

## Developer Tools (🔄 Partial)

- 🔄 Planner extension API (structure exists)
- ⏳ Logical replication plugin (planned)
- ⏳ Foreign data wrapper for vectors (planned)
- ⏳ SQL-based unit test framework (planned)

---

## Configuration (🔄 Partial)

- 🔄 `SHOW VECTOR CONFIG`: View all settings (basic implementation)
- 🔄 `SET VECTOR CONFIG`: Modify parameters (basic implementation)
- 🔄 Runtime tuning without restart (GUC-based)
- ✅ Per-session configuration (PostgreSQL GUCs)

---

## RAG (Retrieval Augmented Generation) (🔄 Partial)

- 🔄 `retrieve()`: ANN search with metadata filters (SPI-based)
- 🔄 `answer()`: Generate answer from retrieved docs (HTTP to LLM)
- 🔄 `plan()`: Multi-step RAG execution plan (basic planner)
- 🔄 `guardrails()`: Content safety and policy checks (basic rules)

---

## Compatibility (✅ Implemented)

- ✅ PostgreSQL 15, 16, 17, 18
- ✅ Linux (Ubuntu, Rocky, Debian)
- ✅ macOS (Intel and Apple Silicon)
- ✅ Standard PostgreSQL extension architecture (PGXS)
- ✅ Zero compilation warnings
- ✅ Zero compilation errors
- ✅ 100% PostgreSQL C coding standards

---

## Build Status (✅ Complete)

- ✅ 28 C source files (all compile cleanly)
- ✅ 8 header files (proper structure)
- ✅ Regression test suite (3 tests: types, operations, distance)
- ✅ TAP test coverage (3 tests: basic, distance, indexing)
- ✅ GitHub Actions CI/CD (9 platform combinations)
- ✅ Multi-platform artifacts (Ubuntu, macOS, Rocky Linux)

---

## Extension-Safe Architecture

NeuronDB is built as a pure PostgreSQL extension with no core modifications:

### What We Use
- ✅ Custom data types via `CREATE TYPE`
- ✅ SQL functions via `CREATE FUNCTION`
- ✅ Shared memory via `RequestAddinShmemSpace`
- ✅ Background workers via `RegisterBackgroundWorker`
- ✅ Hooks (`planner_hook`, `set_rel_pathlist_hook`, etc.)
- ✅ SPI for database interaction
- ✅ External libraries (libcurl, OpenSSL, zlib)

### What We Cannot Do (Core-Only)
- ❌ New WAL resource managers
- ❌ Custom WAL compression formats
- ❌ Checkpoint scheduling changes
- ❌ New wait event types
- ❌ Kernel-level TDE
- ❌ Built-in autovacuum logic changes

### Workarounds Implemented
- ✅ Index-aware checkpoint behavior (periodic flush in worker)
- ✅ Crash-resumable index build (layer checkpoints)
- ✅ Application-level encryption (at type layer)
- ✅ Transparent decrypt in function calls

---

## Implementation Phases

### Phase 1: Core Foundation (Current)
- ✅ Basic vector types and operations
- ✅ Distance metrics
- ✅ Build system and testing
- ✅ CI/CD pipeline
- 🔄 Shared memory statistics
- ⏳ Real IndexAM for HNSW/IVF

### Phase 2: Advanced Features (Next)
- ⏳ CustomScan for hybrid ANN+FTS
- ⏳ Parallel kNN execution
- ⏳ Background workers (job queue, tuning)
- ⏳ SIMD distance kernels
- ⏳ Logical decoding plugin

### Phase 3: Production Hardening (Future)
- ⏳ Tenant quotas and governance
- ⏳ Audit logging with tamper detection
- ⏳ Prometheus exporter
- ⏳ Advanced EXPLAIN support
- ⏳ Recall@K monitoring

### Phase 4: Advanced Integrations (Future)
- ⏳ GPU offload (CUDA/ROCm)
- ⏳ FDW for external vector stores
- ⏳ FDW to external KMS
- ⏳ SQL unit test framework

---

## Documentation

- [README.md](README.md) - Quick start and installation
- [TODO.md](TODO.md) - Detailed implementation roadmap
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [SECURITY.md](SECURITY.md) - Security policy
- [LICENSE](LICENSE) - PostgreSQL License

---

## Contact

For questions, bug reports, or feature requests:
**admin@pgelephant.com**

---

**Legend:**
- ✅ **Implemented**: Fully functional and tested
- 🔄 **Partial**: Structure exists, needs completion
- ⏳ **Planned**: Design complete, implementation pending
