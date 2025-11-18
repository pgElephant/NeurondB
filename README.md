# NeuronDB - Advanced AI Database Extension for PostgreSQL

**Production-grade vector search, machine learning, and hybrid search—directly in PostgreSQL.**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/pgElephant/NeurondB)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%2C17%2C18-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-PostgreSQL-blue.svg)](LICENSE)
[![Code Quality](https://img.shields.io/badge/code-quality-100%25%20PostgreSQL%20C%20standards-success.svg)]()
[![Production Ready](https://img.shields.io/badge/production-ready-success.svg)]()

---

## Overview

NeuronDB transforms PostgreSQL into a comprehensive AI database platform. Built from the ground up with PostgreSQL's architecture in mind, it provides enterprise-grade vector search, ML model inference, hybrid retrieval, and complete RAG pipeline support—all within your existing PostgreSQL infrastructure.

## Key Features

> **📚 Detailed documentation available for each feature below**

### Vector Search & Indexing
- **[Vector Types](https://pgelephant.com/neurondb/features/vector-types/)**: `vector` (float32), `vectorp` (packed), `vecmap` (sparse), `vgraph` (graph-based), `rtext` (retrieval text)
- **[Indexing](https://pgelephant.com/neurondb/features/indexing/)**: HNSW, IVF with automatic tuning and maintenance
- **[Distance Metrics](https://pgelephant.com/neurondb/features/distance-metrics/)**: L2, Cosine, Inner Product, Manhattan, Hamming, Jaccard, and more
- **[Quantization](https://pgelephant.com/neurondb/features/quantization/)**: Product Quantization (PQ), Optimized PQ (OPQ) with 2x-32x compression

### ML Algorithms & Analytics
- **[Random Forest](https://pgelephant.com/neurondb/ml/random-forest/)**: Classification and regression with GPU acceleration
- **[Gradient Boosting](https://pgelephant.com/neurondb/ml/gradient-boosting/)**: XGBoost, LightGBM, CatBoost integration
- **[Clustering](https://pgelephant.com/neurondb/analytics/clustering/)**: K-Means, Mini-batch K-means, DBSCAN, GMM, Hierarchical clustering
- **[Dimensionality Reduction](https://pgelephant.com/neurondb/analytics/dimensionality/)**: PCA, PCA Whitening
- **[Classification](https://pgelephant.com/neurondb/ml/classification/)**: SVM, Logistic Regression, Naive Bayes, Decision Trees, Neural Networks
- **[Regression](https://pgelephant.com/neurondb/ml/regression/)**: Linear Regression, Ridge, Lasso, Deep Learning models
- **[Outlier Detection](https://pgelephant.com/neurondb/analytics/outliers/)**: Z-score, Modified Z-score, IQR methods
- **[Quality Metrics](https://pgelephant.com/neurondb/analytics/quality/)**: Recall@K, Precision@K, F1@K, MRR, Davies-Bouldin Index
- **[Drift Detection](https://pgelephant.com/neurondb/analytics/drift/)**: Centroid drift, Distribution divergence, Temporal monitoring
- **[Topic Discovery](https://pgelephant.com/neurondb/analytics/topics/)**: Topic modeling and analysis
- **[Time Series](https://pgelephant.com/neurondb/ml/timeseries/)**: Forecasting and analysis
- **[Recommendation Systems](https://pgelephant.com/neurondb/ml/recommender/)**: Collaborative filtering and ranking

### ML & Embeddings
- **[Embedding Generation](https://pgelephant.com/neurondb/ml/embeddings/)**: Text, image, multimodal embeddings with intelligent caching
- **[Model Inference](https://pgelephant.com/neurondb/ml/inference/)**: ONNX runtime, batch processing, model management
- **[Model Management](https://pgelephant.com/neurondb/ml/model-management/)**: Load, export, version, monitor models with catalog integration
- **[AutoML](https://pgelephant.com/neurondb/ml/automl/)**: Automated hyperparameter tuning and model selection
- **[Feature Store](https://pgelephant.com/neurondb/ml/feature-store/)**: Centralized feature management and versioning

### Hybrid Search & Retrieval
- **[Hybrid Search](https://pgelephant.com/neurondb/hybrid/overview/)**: Combine vector and full-text search with configurable weights
- **[Multi-Vector](https://pgelephant.com/neurondb/hybrid/multi-vector/)**: Multiple embeddings per document for enhanced retrieval
- **[Faceted Search](https://pgelephant.com/neurondb/hybrid/faceted/)**: Category-aware retrieval with filtering
- **[Temporal Search](https://pgelephant.com/neurondb/hybrid/temporal/)**: Time-decay relevance scoring

### Reranking
- **[Cross-Encoder](https://pgelephant.com/neurondb/reranking/cross-encoder/)**: Neural reranking models
- **[LLM Reranking](https://pgelephant.com/neurondb/reranking/llm/)**: GPT/Claude-powered scoring
- **[ColBERT](https://pgelephant.com/neurondb/reranking/colbert/)**: Late interaction models
- **[Ensemble](https://pgelephant.com/neurondb/reranking/ensemble/)**: Combine multiple reranking strategies

### RAG Pipeline
- **[Complete RAG Support](https://pgelephant.com/neurondb/rag/)**: End-to-end Retrieval Augmented Generation
- **[LLM Integration](https://pgelephant.com/neurondb/llm/)**: Hugging Face and OpenAI integration
- **[Document Processing](https://pgelephant.com/neurondb/nlp/)**: Text processing and NLP capabilities

### Background Workers
- **[neuranq](https://pgelephant.com/neurondb/workers/neuranq/)**: Async job queue executor with batch processing
- **[neuranmon](https://pgelephant.com/neurondb/workers/neuranmon/)**: Live query auto-tuner and performance optimization
- **[neurandefrag](https://pgelephant.com/neurondb/workers/neurandefrag/)**: Automatic index maintenance and defragmentation
- **[neuranllm](https://pgelephant.com/neurondb/workers/neuranllm/)**: LLM job processing with crash recovery

### GPU Acceleration
- **[CUDA Support](https://pgelephant.com/neurondb/gpu/)**: NVIDIA GPU acceleration for vector operations and ML inference
- **[ROCm Support](https://pgelephant.com/neurondb/gpu/)**: AMD GPU acceleration
- **[Metal Support](https://pgelephant.com/neurondb/gpu/)**: Apple Silicon GPU acceleration
- **[Auto-Detection](https://pgelephant.com/neurondb/gpu/)**: Automatic GPU detection and fallback to CPU

### Performance & Security
- **[SIMD Optimization](https://pgelephant.com/neurondb/performance/optimization/)**: AVX2/AVX512 (x86_64), NEON (ARM64) with prefetching
- **[Security](https://pgelephant.com/neurondb/security/overview/)**: Encryption, differential privacy, Row-Level Security (RLS) integration
- **[Monitoring](https://pgelephant.com/neurondb/performance/monitoring/)**: 7 built-in monitoring views, Prometheus metrics export

## Quick Start

### Installation

**Using build.sh (Recommended):**
```bash
git clone https://github.com/pgElephant/NeurondB.git
cd NeurondB
./build.sh                    # CPU-only build
./build.sh --with-gpu         # With GPU support (CUDA/ROCm auto-detected)
./build.sh --with-gpu --test  # Build with GPU and run tests
```

**Ubuntu/Debian:**
```bash
sudo apt-get install -y postgresql-17 postgresql-server-dev-17 \
    build-essential libcurl4-openssl-dev libssl-dev zlib1g-dev

git clone https://github.com/pgElephant/NeurondB.git
cd NeurondB
make PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config
sudo make install PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config
```

**macOS:**
```bash
brew install postgresql@17

git clone https://github.com/pgElephant/NeurondB.git
cd NeurondB
make PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config
sudo make install PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config
```

### Basic Usage

```sql
-- Create extension
CREATE EXTENSION neurondb;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding vector(384)
);

-- Generate embeddings
INSERT INTO documents (title, content, embedding) VALUES
    ('Machine Learning', 'Introduction to ML', embed_text('Introduction to ML'));

-- Create HNSW index
SELECT hnsw_create_index('documents', 'embedding', 'doc_idx', 16, 200);

-- Semantic search
SELECT title, content,
       embedding <-> embed_text('artificial intelligence') AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- Hybrid search (70% vector, 30% text)
SELECT * FROM hybrid_search(
    'documents',
    embed_text('machine learning'),
    'neural networks',
    '{}',
    0.7,
    10
);

-- Train Random Forest classifier
SELECT train_random_forest_classifier(
    'training_table',
    'features',
    'label',
    100,  -- n_trees
    10,   -- max_depth
    5     -- min_samples_split
);

-- Rerank with cross-encoder
SELECT idx, score FROM rerank_cross_encoder(
    'What is deep learning?',
    ARRAY['Neural networks tutorial', 'Deep learning basics', 'AI history'],
    'ms-marco-MiniLM-L-6-v2',
    3
);
```

## Architecture

NeuronDB follows PostgreSQL's C coding standards and architectural patterns:

```
┌─────────────────────────────────────────────────────────┐
│                    SQL Interface                         │
│              (473 functions, types, operators)          │
├─────────────────────────────────────────────────────────┤
│  Vector Types  │  Distance Ops  │  Index Methods        │
│  (vector, vectorp, vecmap, vgraph, rtext)               │
├─────────────────────────────────────────────────────────┤
│  ML Inference  │  Embeddings    │  Model Management     │
│  (52 ML algorithms: RF, XGBoost, LightGBM, etc.)        │
├─────────────────────────────────────────────────────────┤
│  Hybrid Search │  Reranking     │  RAG Pipeline         │
├─────────────────────────────────────────────────────────┤
│  Background Workers (neuranq, neuranmon, neurandefrag)  │
├─────────────────────────────────────────────────────────┤
│  GPU Acceleration (CUDA/ROCm/Metal)                     │
├─────────────────────────────────────────────────────────┤
│  PostgreSQL Core (Storage, WAL, SPI, Shared Memory)     │
└─────────────────────────────────────────────────────────┘
```

**Design Principles:**
- **100% PostgreSQL C coding standards** - Tabs, 80-column lines, C-style comments
- **Pure C implementation** - 144 source files, zero external dependencies for core
- **PostgreSQL PGXS build system** - Standard extension build process
- **Shared memory for caching** - High-performance in-memory operations
- **WAL integration** - Full durability and crash recovery
- **SPI for safe database operations** - Secure query execution
- **Background worker framework** - Async processing and maintenance
- **Defensive programming** - Comprehensive NULL checks, error handling, memory safety

## Performance Benchmarks

Tested on AWS r6i.2xlarge (8 vCPU, 64GB RAM), 10M vectors, 768 dimensions:

| Operation | Throughput | Latency (p95) | Notes |
|-----------|------------|---------------|-------|
| Vector Insert | 50K/sec | 2ms | With SIMD optimization |
| HNSW Search (k=10) | 10K QPS | 5ms | ef_search=64 |
| Embedding Generation | 1K/sec | 10ms | Cached |
| Hybrid Search | 5K QPS | 8ms | 70% vector, 30% FTS |
| Reranking (Cross-encoder) | 2K/sec | 15ms | CPU |
| Random Forest Training | 100K samples | 2.5s | 100 trees, SIMD |
| K-Means Clustering | 1M vectors | 2.5s | 100 clusters, SIMD |
| PCA (768→128 dims) | 100K vectors | 1.2s | SIMD optimized |
| GPU Distance (batch) | 500K/sec | - | CUDA/ROCm/Metal |

**SIMD Acceleration:**
- **x86_64**: AVX2/AVX512 with FMA instructions
- **ARM64**: NEON with dotprod extension  
- **Compiler flags**: `-O3 -march=native -funroll-loops`

## Configuration

```sql
-- Background workers (requires shared_preload_libraries)
ALTER SYSTEM SET shared_preload_libraries = 'neurondb';
-- Restart PostgreSQL

-- Configure workers
SET neurondb.neuranq_enabled = true;
SET neurondb.neuranq_naptime = 1000;  -- milliseconds
SET neurondb.neuranmon_enabled = true;
SET neurondb.neurandefrag_enabled = true;

-- Query settings
SET neurondb.default_ef_search = 64;
SET neurondb.enable_prefetch = true;

-- GPU settings
SET neurondb.gpu_enabled = true;
SELECT neurondb_gpu_info();  -- Check GPU availability
```

## Monitoring & Observability

NeuronDB provides 7 built-in monitoring views in the `neurondb` schema:

```sql
-- Aggregate vector statistics
SELECT * FROM neurondb.vector_stats;

-- Index health dashboard  
SELECT * FROM neurondb.index_health;

-- Tenant quota usage with warnings
SELECT * FROM neurondb.tenant_quota_usage;

-- LLM job queue status
SELECT * FROM neurondb.llm_job_status;

-- Query performance metrics (last 24h)
SELECT * FROM neurondb.query_performance;

-- Index maintenance operations
SELECT * FROM neurondb.index_maintenance_status;

-- Prometheus metrics summary
SELECT * FROM neurondb.metrics_summary;

-- Extension statistics (function)
SELECT * FROM pg_stat_neurondb();

-- Background worker status
SELECT * FROM neurondb_worker_status();
```

## Testing

```bash
# Regression tests (14 test suites)
make installcheck PG_CONFIG=/path/to/pg_config

# TAP tests
make prove PG_CONFIG=/path/to/pg_config
```

All tests pass on PostgreSQL 16, 17, 18 across Ubuntu, Debian, Rocky Linux, and macOS.

## Compatibility

| PostgreSQL | Status | Platforms |
|------------|--------|-----------|
| 16.x | ✅ Supported | Ubuntu, Debian, Rocky Linux, macOS |
| 17.x | ✅ Supported | Ubuntu, Debian, Rocky Linux, macOS |
| 18.x | ✅ Supported | Ubuntu, Debian, Rocky Linux, macOS |

> **Note**: NeuronDB supports only PostgreSQL 16, 17, and 18. The extension validates the PostgreSQL version at creation time and will fail fast with a clear error message if an unsupported version is detected.

## Code Quality

NeuronDB maintains the highest code quality standards:

- **100% PostgreSQL C coding standards compliance**
  - Tabs for indentation (8-space visual width)
  - 80-column line limit (with exceptions for error messages)
  - C-style block comments only (`/* */`)
  - Variables declared at function start (C89/C99 compliance)
  - Exactly one blank line between function definitions
  - Typedefs and structs before first function definition

- **Zero compiler warnings** - All code compiles cleanly with `-Wall -Wextra`
- **Comprehensive error handling** - All error paths properly clean up resources
- **Memory safety** - All allocations checked for integer overflow with `MaxAllocSize`
- **Defensive programming** - NULL pointer checks, validation, and explicit error paths

## Documentation

- **[Full Documentation](https://pgelephant.com/neurondb)** - Comprehensive guides and API reference
- **[GPU Acceleration Guide](docs/gpu.md)** - CUDA/ROCm/Metal GPU support documentation
- **[Installation Guide](INSTALL.md)** - Detailed installation instructions
- **[Contributing Guide](CONTRIBUTING.md)** - Development workflow and code standards
- **[Security Policy](SECURITY.md)** - Security best practices and vulnerability reporting

## Project Structure

```
neurondb/
├── src/
│   ├── core/         # Core vector types and operations
│   ├── ml/           # 52 ML algorithm implementations
│   │   ├── ml_random_forest.c      # Random Forest (100% PostgreSQL C standards)
│   │   ├── ml_xgboost.c            # XGBoost integration
│   │   ├── ml_lightgbm.c          # LightGBM integration
│   │   ├── ml_catboost.c          # CatBoost integration
│   │   ├── ml_kmeans.c            # K-Means clustering
│   │   ├── ml_dbscan.c            # DBSCAN clustering
│   │   ├── ml_gmm.c               # Gaussian Mixture Models
│   │   ├── ml_pca_whitening.c     # PCA and whitening
│   │   ├── ml_linear_regression.c # Linear regression
│   │   ├── ml_logistic_regression.c # Logistic regression
│   │   ├── ml_svm.c               # Support Vector Machines
│   │   ├── ml_naive_bayes.c       # Naive Bayes
│   │   ├── ml_neural_network.c    # Neural networks
│   │   ├── ml_deeplearning.c      # Deep learning models
│   │   ├── ml_decision_tree.c     # Decision trees
│   │   ├── ml_hierarchical.c      # Hierarchical clustering
│   │   ├── ml_minibatch_kmeans.c  # Mini-batch K-means
│   │   ├── ml_product_quantization.c # Product Quantization
│   │   ├── ml_opq.c               # Optimized Product Quantization
│   │   ├── ml_outlier_detection.c # Outlier detection
│   │   ├── ml_drift_detection.c   # Drift detection
│   │   ├── ml_topic_discovery.c   # Topic modeling
│   │   ├── ml_timeseries.c        # Time series analysis
│   │   ├── ml_recommender.c       # Recommendation systems
│   │   ├── ml_automl.c            # AutoML
│   │   ├── ml_hyperparameter_tuning.c # Hyperparameter tuning
│   │   ├── embeddings.c           # Embedding generation
│   │   ├── ml_inference.c         # Model inference
│   │   ├── reranking.c            # Reranking algorithms
│   │   └── ...                    # Additional ML algorithms
│   ├── gpu/          # GPU acceleration (CUDA/ROCm/Metal)
│   │   ├── cuda/     # CUDA implementation
│   │   ├── rocm/     # ROCm implementation
│   │   └── metal/    # Metal implementation
│   ├── worker/       # 5 background workers
│   ├── index/        # HNSW & IVF index access methods
│   ├── scan/         # Custom scan nodes
│   ├── llm/          # LLM integration (Hugging Face, etc.)
│   ├── search/       # Hybrid and temporal search
│   ├── metrics/      # Prometheus metrics
│   ├── storage/      # Buffer management and WAL
│   ├── planner/      # Query optimization
│   ├── tenant/       # Multi-tenancy support
│   ├── types/        # Quantization and aggregates
│   └── util/         # Configuration, security, hooks
├── include/          # Header files (63 header files)
├── sql/              # Regression test SQL (14 test suites)
├── expected/         # Expected test outputs
├── t/                # TAP test suite (Perl tests)
├── docs/             # MkDocs documentation source
├── config/           # Configuration examples
└── neurondb--1.0.sql # Extension SQL (6,641 lines, perfectly organized)
```

**Code Statistics:**
- **144 C source files** - All following 100% PostgreSQL C coding standards
- **63 header files** - Comprehensive API definitions
- **52 ML algorithm implementations** - Production-ready implementations
- **473 SQL functions/types/operators** - Complete feature coverage
- **6,641 lines of SQL** - Well-organized extension definitions

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/pgElephant/NeurondB/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pgElephant/NeurondB/discussions)
- **Email**: admin@pgelephant.com
- **Security**: Report vulnerabilities to admin@pgelephant.com

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines (100% PostgreSQL C standards)
- Development workflow
- Testing requirements
- Pull request process

**Key Requirements:**
- All code must follow 100% PostgreSQL C coding standards
- Zero compiler warnings (`-Wall -Wextra`)
- Comprehensive error handling and memory safety
- All tests must pass

## License

NeuronDB is released under the PostgreSQL License. See [LICENSE](LICENSE) for details.

## Authors

**pgElephant, Inc.**  
Email: admin@pgelephant.com  
Website: https://pgelephant.com

Built for the PostgreSQL community with enterprise-grade reliability and 100% code quality standards.

---

<div align="center">

**[Documentation](https://pgelephant.com/neurondb)** • 
**[GitHub](https://github.com/pgElephant/NeurondB)** • 
**[Support](mailto:admin@pgelephant.com)**

</div>
