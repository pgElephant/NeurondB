# NeurondB Source Directory Structure

## Organization Philosophy
Clean separation of concerns with dedicated subdirectories for each major subsystem.

## Directory Layout

```
src/
├── core/                    # Core vector types and operations
│   ├── neurondb.c          # Main extension entry point
│   ├── types_core.c        # Vector type I/O and basic ops
│   ├── distance.c          # Distance metrics
│   ├── operators.c         # SQL operators
│   └── vector_ops.c        # Vector arithmetic
│
├── index/                   # Index access methods
│   ├── hnsw_am.c           # HNSW IndexAM (complete)
│   ├── ivf_am.c            # IVF IndexAM (complete)
│   ├── hybrid_am.c         # Hybrid fused index
│   ├── index_validator.c   # Index validation functions
│   ├── index_cache.c       # Entrypoint caching
│   └── opclass.c           # Operator classes/families
│
├── scan/                    # Scan nodes and execution
│   ├── hnsw_scan.c         # HNSW scan logic
│   ├── ivf_scan.c          # IVF scan logic  
│   ├── hybrid_customscan.c # CustomScan for hybrid
│   ├── scan_rls.c          # RLS integration
│   └── scan_quota.c        # Quota enforcement
│
├── worker/                  # Background workers
│   ├── worker_queue.c      # neuranq - job queue
│   ├── worker_tuner.c      # neuranmon - auto-tuner
│   ├── worker_defrag.c     # neurandefrag - maintenance
│   ├── worker_llm.c        # neuranllm - LLM jobs
│   └── worker_init.c       # Worker registration
│
├── llm/                     # LLM integration
│   ├── llm_runtime.c       # Main LLM runtime
│   ├── llm_cache.c         # Response cache
│   ├── llm_jobs.c          # Job management
│   └── hf_http.c           # Hugging Face HTTP client
│
├── gpu/                     # GPU acceleration
│   ├── gpu_core.c          # GPU initialization
│   ├── gpu_distance.c      # GPU distance kernels
│   ├── gpu_batch.c         # Batch operations
│   ├── gpu_quantization.c  # GPU quantization
│   ├── gpu_clustering.c    # GPU clustering
│   └── gpu_kernels.cu      # CUDA/ROCm kernels
│
├── types/                   # Data type operations
│   ├── quantization.c      # Vector quantization
│   ├── aggregates.c        # Aggregate functions
│   └── casts.c             # Type casting
│
├── ml/                      # Machine learning
│   ├── ml_inference.c      # Model inference
│   ├── analytics.c         # Clustering, PCA, etc.
│   └── model_runtime.c     # Model management
│
├── metrics/                 # Observability
│   ├── pg_stat_neurondb.c  # Statistics view
│   ├── prometheus.c        # Prometheus exporter
│   └── slow_log.c          # Slow query logging
│
├── search/                  # Search algorithms
│   ├── hybrid_search.c     # Hybrid search logic
│   ├── rerank.c            # Reranking
│   └── temporal.c          # Temporal scoring
│
├── storage/                 # Storage and memory
│   ├── ann_buffer.c        # In-memory ANN buffer
│   ├── vector_wal.c        # WAL compression
│   └── buffer.c            # Buffer management
│
├── tenant/                  # Multi-tenancy
│   ├── multi_tenant.c      # Tenant management
│   ├── quota.c             # Quota enforcement
│   └── rls.c               # RLS helpers
│
├── planner/                 # Query planning
│   ├── planner.c           # Planner hooks
│   └── cost.c              # Cost estimation
│
└── util/                    # Utilities
    ├── config.c            # Configuration
    ├── security.c          # Security helpers
    ├── hooks.c             # Developer hooks
    └── distributed.c       # Distributed features
```

## File Naming Conventions

- **Index AM files:** `<name>_am.c` (e.g., `hnsw_am.c`)
- **Scan files:** `<name>_scan.c` (e.g., `hnsw_scan.c`)
- **Worker files:** `worker_<name>.c` (e.g., `worker_queue.c`)
- **Feature files:** Descriptive names (e.g., `prometheus.c`, `validator.c`)

## Migration Plan

Phase 1: Move existing files to proper locations
Phase 2: Rename files to follow conventions
Phase 3: Update Makefile with new paths
Phase 4: Complete missing implementations
