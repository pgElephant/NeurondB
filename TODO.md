# NeuronDB Implementation TODO

This document tracks features that need full implementation to match the safe "extension-only" feature set for NeuronDB.

## Status Legend
- ✅ **Implemented**: Fully functional
- 🔄 **Partial**: Basic structure exists, needs completion
- ⏳ **Planned**: Not yet started

---

## Data & Math

### Vector Types
- ✅ vectorf32, vectorf16, vectori8, vectorbin (basic I/O implemented)
- 🔄 vectorp (structure defined, needs SIMD validation)
- 🔄 vecmap (structure defined, needs sparse ops)
- 🔄 rtext (structure defined, needs token offset tracking)
- 🔄 vgraph (structure defined, needs graph algorithms)

### Distance Functions
- ✅ L2, Cosine, Inner Product, Manhattan, Hamming, Jaccard, Minkowski, Chebyshev (basic implementations)
- ⏳ SIMD optimizations (AVX2, AVX-512, NEON) for all distance kernels
- ⏳ Vectorized batch distance computation

### Quantization
- 🔄 Quantize/dequantize functions (stubs exist, need proper algorithms)
- ⏳ Product Quantization (PQ)
- ⏳ Scalar Quantization (SQ)
- ⏳ Binary quantization with learned thresholds

---

## Indexing

### Access Methods
- ⏳ **HNSW IndexAM**: Real `IndexAmRoutine` with handlers
  - Need: `ambuild`, `aminsert`, `amrescan`, `amgettuple`, `amendscan`
  - Need: Layer construction, neighbor selection, pruning
  - Need: ef_construction and ef_search parameter handling
  
- ⏳ **IVF IndexAM**: Real `IndexAmRoutine` with handlers
  - Need: Cluster training (k-means)
  - Need: Centroid storage and search
  - Need: Posting list management
  
- ⏳ **Tenant-aware index options**: Parse and enforce tenant_id in index
  
- ⏳ **Temporal decay scoring**: Time-weighted distance in index scan
  
- ⏳ **Consistent top-k (CQ-HNSW)**: Snapshot-based deterministic search

### Hybrid Search
- ⏳ **CustomScan node** for ANN + FTS fusion
  - Need: Register custom scan provider
  - Need: Plan merging logic (vector + GIN results)
  - Need: Score fusion (weighted combination)
  - Need: Efficient result deduplication

---

## Planner and Executor

### Cost Estimation
- ⏳ **planner_hook**: Inject ANN costs into query planning
  - Need: Cost model for index scan (I/O + CPU)
  - Need: Selectivity estimation for kNN
  - Need: Compare ANN vs sequential scan costs
  
- ⏳ **set_rel_pathlist_hook**: Add ANN paths to relation
  - Need: Create IndexPath for ANN indexes
  - Need: Parameterize by k, distance metric
  - Need: Cost estimation per path

### Execution
- ⏳ **CustomScan for distance+filter fusion**
  - Need: Push down filters to index scan
  - Need: Avoid fetching + filtering separately
  
- ⏳ **Parallel kNN via CustomScan workers**
  - Need: Partition search space across workers
  - Need: Merge partial top-k results
  - Need: Deterministic global top-k ordering

### Caching
- 🔄 **Plan cache for ANN entry points** in shared memory
  - Need: Shared memory segment allocation
  - Need: LRU eviction policy
  - Need: Concurrent access with spinlocks

---

## Runtime and Jobs

### Background Workers
- 🔄 **Background worker for job queue**
  - Need: Register worker via `shared_preload_libraries`
  - Need: Job table with SKIP LOCKED pattern
  - Need: Process embedding generation, index tuning, defragmentation
  
- 🔄 **Auto-tuning worker**
  - Need: Monitor query performance
  - Need: Adjust ef_search, beam_size dynamically
  - Need: Store tuning history

- 🔄 **Index defragmentation worker**
  - Need: Detect bloated index layers
  - Need: Incremental layer rebuild
  - Need: Non-blocking defragmentation

### Model Integration
- 🔄 **HTTP calls to model sidecar via libcurl**
  - Need: Connection pooling
  - Need: Retry logic with exponential backoff
  - Need: Circuit breaker pattern
  - Need: Async request handling

- 🔄 **LLM token and cost accounting**
  - Need: Track tokens per request
  - Need: Cost calculation (input + output tokens)
  - Need: Per-user/tenant aggregation
  - Need: Budget enforcement

### Triggers and Jobs
- ⏳ **Embedding generation triggers**
  - Need: Trigger on INSERT/UPDATE of text columns
  - Need: Enqueue embedding job
  - Need: Async processing with SKIP LOCKED
  
- ⏳ **SKIP LOCKED job queue pattern**
  - Need: Job status table (pending, running, done, failed)
  - Need: Worker claims jobs atomically
  - Need: Retry failed jobs with backoff

---

## RAG in SQL

### SQL API
- 🔄 **retrieve(query, top_k, filters)**: ANN search with metadata filters
- 🔄 **answer(query, context, model)**: Generate answer from retrieved docs
- 🔄 **plan(query)**: Multi-step RAG execution plan
- 🔄 **guardrails(text, rules)**: Content safety and policy checks

### Caching
- ⏳ **Semantic cache via ANN on prompts**
  - Need: Store (prompt_embedding, response) pairs
  - Need: Search for similar prompts
  - Need: Return cached response if similarity > threshold
  
- ⏳ **TTL-based cache expiration**
  - Need: Timestamp column
  - Need: Background worker for cleanup
  - Need: LRU eviction when cache full

---

## Governance

### Tenant Quotas
- 🔄 **Quota enforcement in index AM**
  - Need: Track index size per tenant
  - Need: Reject insert if quota exceeded
  - Need: Quota table (tenant_id, max_vectors, max_storage)

- 🔄 **Quota enforcement in functions**
  - Need: Check before embedding generation
  - Need: Check before expensive operations

### Policy Engine
- 🔄 **Policy checks in SQL functions**
  - Need: Policy table (tenant_id, resource, action, allowed)
  - Need: Check before queries, inserts, model calls
  - Need: Audit policy violations

### Audit Logging
- 🔄 **Immutable audit table with HMAC**
  - Need: Log all ANN queries with vector hash
  - Need: Sign with HMAC-SHA256
  - Need: Tamper detection on read
  - Need: Partition by time for performance

### Row-Level Security
- ⏳ **RLS policies for vector data**
  - Need: Enable RLS on vector tables
  - Need: Policy: users see only their tenant's data
  - Need: Policy: admins see all data

---

## Security

### Encryption
- 🔄 **Application-level AES-GCM encryption**
  - Need: Encrypt vectors before storage
  - Need: Decrypt on read
  - Need: Store IV and auth tag
  - Need: Key derivation from tenant key

- 🔄 **Differential privacy noise**
  - Need: Add Laplace/Gaussian noise to embeddings
  - Need: Epsilon/delta privacy budget tracking
  - Need: Per-user privacy accounting

### Key Management
- ⏳ **FDW to external KMS** (AWS KMS, Vault)
  - Need: Foreign table for key retrieval
  - Need: Cache keys in session memory
  - Need: Rotate keys periodically

- ⏳ **SGX/SEV via UDFs**
  - Need: UDF that calls enclave library
  - Need: Decrypt inside enclave
  - Need: Return plaintext only to trusted caller

---

## Replication and Streaming

### Logical Decoding
- ⏳ **Logical decoding plugin for embeddings**
  - Need: Register output plugin
  - Need: Decode INSERT/UPDATE of vector columns
  - Need: Emit ANN index hints (entry points, links)
  - Need: Consumer applies to replica

### Foreign Data Wrappers
- ⏳ **FDW for external vector stores** (FAISS, Milvus, Weaviate)
  - Need: Implement `GetForeignRelSize`, `GetForeignPaths`, `IterateForeignScan`
  - Need: Push down kNN to external store
  - Need: Fetch results as tuples

---

## Observability

### Statistics View
- 🔄 **pg_stat_neurondb backed by shared stats**
  - Need: Shared memory segment for counters
  - Need: Atomic increments for query count, latency
  - Need: Per-index statistics
  - Need: Per-user statistics

### Metrics
- 🔄 **Latency histograms**
  - Need: Bucket latencies (p50, p95, p99)
  - Need: Per-query-type histograms
  - Need: Sliding window aggregation

- 🔄 **Recall@K sampling**
  - Need: Randomly sample queries
  - Need: Compute exact top-k for comparison
  - Need: Calculate recall = |ANN ∩ exact| / k

### Prometheus Exporter
- ⏳ **Background worker HTTP server**
  - Need: Listen on configurable port
  - Need: Serve /metrics endpoint
  - Need: Expose query_count, avg_latency, recall, cache_hit_rate
  - Need: Scrape by Prometheus

### EXPLAIN Support
- ⏳ **ExplainProperty for ANN nodes**
  - Need: Output: index used, ef_search, nodes visited, distance computed
  - Need: Output: cache hits, prefetch count
  - Need: Output: recall estimate if available

---

## GPU and SIMD

### SIMD Kernels
- ⏳ **AVX2 distance kernels** (x86-64)
- ⏳ **AVX-512 distance kernels** (newer x86-64)
- ⏳ **NEON distance kernels** (ARM64)
- ⏳ **Runtime CPU detection and dispatch**

### GPU Offload
- ⏳ **CUDA UDFs for batch distance** (NVIDIA)
  - Need: Copy vectors to GPU
  - Need: Launch kernel for distance matrix
  - Need: Copy results back
  
- ⏳ **ROCm UDFs for batch distance** (AMD)
  - Need: Similar to CUDA
  
- ⏳ **Background worker offload**
  - Need: Job queue for GPU tasks
  - Need: Worker with GPU context
  - Need: Batch multiple requests

---

## Developer Experience

### Planner Extension API
- ⏳ **Register custom distance metrics at runtime**
  - Need: Table of (metric_name, function_oid)
  - Need: Load from extension table
  - Need: Use in index scan

### Testing
- ⏳ **SQL unit test framework**
  - Need: `assert_equals(expected, actual)` function
  - Need: `assert_vector_near(v1, v2, epsilon)` function
  - Need: Test suite runner
  - Need: TAP integration

- ⏳ **pgbench helpers**
  - Need: Generate random vectors
  - Need: kNN query templates
  - Need: Benchmark recall vs latency

### Configuration
- 🔄 **SHOW VECTOR CONFIG** (basic implementation exists)
  - Need: Read all GUCs
  - Need: Format as table
  
- 🔄 **SET VECTOR CONFIG** (basic implementation exists)
  - Need: Validate parameter names
  - Need: Apply to session or index

---

## Distributed Features

### Shard-Aware Query Routing
- 🔄 **SQL function to route to shard**
  - Need: Partition map (shard_id, key_range, host)
  - Need: Hash or range partition logic
  - Need: Execute remote query via dblink or postgres_fdw

### Deterministic Top-K Merge
- ⏳ **Executor node for cross-shard merge**
  - Need: Fetch top-k from each shard
  - Need: Global merge sort by distance
  - Need: Return global top-k
  - Need: Early termination if possible

---

## Extension Limitations (Cannot Implement)

These features are **not possible** as a PostgreSQL extension and are documented for transparency:

### Core Engine Restrictions
- ❌ **New WAL resource managers**: Only core can add these
- ❌ **Custom WAL compression formats**: Only core can change WAL format
- ❌ **Checkpoint scheduling**: Only core controls checkpointing
- ❌ **New wait event types**: Core-only feature
- ❌ **Kernel-level TDE**: Requires core modification
- ❌ **Autovacuum logic changes**: Limited to hooks only

### Workarounds Available
- ✅ **Index-aware checkpoint behavior**: Use periodic flush in background worker
- ✅ **Crash-resumable index build**: Store partial state, detect and resume
- ✅ **HOT-friendly updates**: Careful tuple layout design
- ✅ **Transparent decrypt in shared memory**: Decrypt at function call boundaries

---

## Implementation Priority

### Phase 1: Core Foundation (High Priority)
1. **Real HNSW IndexAM** with full `IndexAmRoutine`
2. **Real IVF IndexAM** with full `IndexAmRoutine`
3. **planner_hook and set_rel_pathlist_hook** for cost-based ANN
4. **Shared memory statistics** for `pg_stat_neurondb`
5. **SIMD distance kernels** (AVX2, NEON)

### Phase 2: Advanced Features (Medium Priority)
6. **CustomScan for ANN+FTS hybrid**
7. **Parallel kNN via CustomScan workers**
8. **Background workers** (job queue, tuning, defragmentation)
9. **Logical decoding plugin** for replication
10. **FDW for external vector stores**

### Phase 3: Production Hardening (Medium Priority)
11. **Tenant quotas and governance**
12. **Audit logging with HMAC**
13. **Prometheus exporter**
14. **EXPLAIN support for ANN**
15. **Recall@K sampling and monitoring**

### Phase 4: Advanced Integrations (Lower Priority)
16. **Semantic cache for RAG**
17. **GPU offload (CUDA/ROCm)**
18. **FDW to external KMS**
19. **Logical replication of embeddings**
20. **SQL unit test framework**

---

## Notes

- All features must be **extension-only** (no core modifications)
- Use **PostgreSQL hooks** wherever possible (planner, executor, utility)
- Use **shared memory** for cross-backend state (with proper locking)
- Use **background workers** for async tasks (preload via `shared_preload_libraries`)
- Document any **limitations** due to extension boundaries

---

## Contact

For questions about implementation priorities or architecture decisions:
**admin@pgelephant.com**

