# NeuronDB - Advanced AI Database Extension for PostgreSQL

**Production-grade vector search, machine learning, and hybrid search—directly in PostgreSQL.**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/pgElephant/NeurondB)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%2C17%2C18-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code Quality](https://img.shields.io/badge/code-quality-100%25%20PostgreSQL%20C%20standards-success.svg)]()
[![Production Ready](https://img.shields.io/badge/production-ready-success.svg)]()

---

## Overview

NeuronDB transforms PostgreSQL into a comprehensive AI database platform. Built from the ground up with PostgreSQL's architecture in mind, it provides enterprise-grade vector search, ML model inference, hybrid retrieval, and complete RAG pipeline support—all within your existing PostgreSQL infrastructure.

## Key Features

> **📚 Detailed documentation available for each feature below**

### Vector Search & Indexing
- **[Vector Types](https://www.pgelephant.com/docs/neurondb/features/vector-types)**: `vector` (float32), `vectorp` (packed), `vecmap` (sparse), `vgraph` (graph-based), `rtext` (retrieval text)
- **[Indexing](https://www.pgelephant.com/docs/neurondb/features/indexing)**: HNSW, IVF with automatic tuning and maintenance
- **[Distance Metrics](https://www.pgelephant.com/docs/neurondb/features/distance-metrics)**: L2, Cosine, Inner Product, Manhattan, Hamming, Jaccard, and more
- **[Quantization](https://www.pgelephant.com/docs/neurondb/features/quantization)**: Product Quantization (PQ), Optimized PQ (OPQ) with 2x-32x compression

### ML Algorithms & Analytics
- **[Random Forest](https://www.pgelephant.com/docs/neurondb/ml/random-forest/)**: Classification and regression with GPU acceleration
- **[Gradient Boosting](https://www.pgelephant.com/docs/neurondb/ml/gradient-boosting/)**: XGBoost, LightGBM, CatBoost integration
- **[Clustering](https://www.pgelephant.com/docs/neurondb/analytics/clustering/)**: K-Means, Mini-batch K-means, DBSCAN, GMM, Hierarchical clustering
- **[Dimensionality Reduction](https://www.pgelephant.com/docs/neurondb/analytics/dimensionality/)**: PCA, PCA Whitening
- **[Classification](https://www.pgelephant.com/docs/neurondb/ml/classification/)**: SVM, Logistic Regression, Naive Bayes, Decision Trees, Neural Networks
- **[Regression](https://www.pgelephant.com/docs/neurondb/ml/regression/)**: Linear Regression, Ridge, Lasso, Deep Learning models
- **[Outlier Detection](https://www.pgelephant.com/docs/neurondb/analytics/outliers/)**: Z-score, Modified Z-score, IQR methods
- **[Quality Metrics](https://www.pgelephant.com/docs/neurondb/analytics/quality/)**: Recall@K, Precision@K, F1@K, MRR, Davies-Bouldin Index
- **[Drift Detection](https://www.pgelephant.com/docs/neurondb/analytics/drift/)**: Centroid drift, Distribution divergence, Temporal monitoring
- **[Topic Discovery](https://www.pgelephant.com/docs/neurondb/analytics/topics/)**: Topic modeling and analysis
- **[Time Series](https://www.pgelephant.com/docs/neurondb/ml/timeseries/)**: Forecasting and analysis
- **[Recommendation Systems](https://www.pgelephant.com/docs/neurondb/ml/recommender/)**: Collaborative filtering and ranking

### ML & Embeddings
- **[Embedding Generation](https://www.pgelephant.com/docs/neurondb/ml/embeddings/)**: Text, image, multimodal embeddings with intelligent caching
- **[Model Inference](https://www.pgelephant.com/docs/neurondb/ml/inference/)**: ONNX runtime, batch processing, model management
- **[Model Management](https://www.pgelephant.com/docs/neurondb/ml/model-management/)**: Load, export, version, monitor models with catalog integration
- **[AutoML](https://www.pgelephant.com/docs/neurondb/ml/automl/)**: Automated hyperparameter tuning and model selection
- **[Feature Store](https://www.pgelephant.com/docs/neurondb/ml/feature-store/)**: Centralized feature management and versioning

### Hybrid Search & Retrieval
- **[Hybrid Search](https://www.pgelephant.com/docs/neurondb/hybrid/overview/)**: Combine vector and full-text search with configurable weights
- **[Multi-Vector](https://www.pgelephant.com/docs/neurondb/hybrid/multi-vector/)**: Multiple embeddings per document for enhanced retrieval
- **[Faceted Search](https://www.pgelephant.com/docs/neurondb/hybrid/faceted/)**: Category-aware retrieval with filtering
- **[Temporal Search](https://www.pgelephant.com/docs/neurondb/hybrid/temporal/)**: Time-decay relevance scoring

### Reranking
- **[Cross-Encoder](https://www.pgelephant.com/docs/neurondb/reranking/cross-encoder/)**: Neural reranking models
- **[LLM Reranking](https://www.pgelephant.com/docs/neurondb/reranking/llm/)**: GPT/Claude-powered scoring
- **[ColBERT](https://www.pgelephant.com/docs/neurondb/reranking/colbert/)**: Late interaction models
- **[Ensemble](https://www.pgelephant.com/docs/neurondb/reranking/ensemble/)**: Combine multiple reranking strategies

### RAG Pipeline
- **[Complete RAG Support](https://www.pgelephant.com/docs/neurondb/rag/)**: End-to-end Retrieval Augmented Generation
- **[LLM Integration](https://www.pgelephant.com/docs/neurondb/llm/)**: Hugging Face and OpenAI integration
- **[Document Processing](https://www.pgelephant.com/docs/neurondb/nlp/)**: Text processing and NLP capabilities

### Background Workers
- **[neuranq](https://www.pgelephant.com/docs/neurondb/workers/neuranq/)**: Async job queue executor with batch processing
- **[neuranmon](https://www.pgelephant.com/docs/neurondb/workers/neuranmon/)**: Live query auto-tuner and performance optimization
- **[neurandefrag](https://www.pgelephant.com/docs/neurondb/workers/neurandefrag/)**: Automatic index maintenance and defragmentation
- **[neuranllm](https://www.pgelephant.com/docs/neurondb/workers/neuranllm/)**: LLM job processing with crash recovery

### GPU Acceleration
- **[CUDA Support](https://www.pgelephant.com/docs/neurondb/gpu/)**: NVIDIA GPU acceleration for vector operations and ML inference
- **[ROCm Support](https://www.pgelephant.com/docs/neurondb/gpu/)**: AMD GPU acceleration
- **[Metal Support](https://www.pgelephant.com/docs/neurondb/gpu/)**: Apple Silicon GPU acceleration
- **[Auto-Detection](https://www.pgelephant.com/docs/neurondb/gpu/)**: Automatic GPU detection and fallback to CPU

### Performance & Security
- **[SIMD Optimization](https://www.pgelephant.com/docs/neurondb/performance/optimization/)**: AVX2/AVX512 (x86_64), NEON (ARM64) with prefetching
- **[Security](https://www.pgelephant.com/docs/neurondb/security/overview/)**: Encryption, differential privacy, Row-Level Security (RLS) integration
- **[Monitoring](https://www.pgelephant.com/docs/neurondb/performance/monitoring/)**: 7 built-in monitoring views, Prometheus metrics export

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

## Documentation

- **[Getting Started](https://www.pgelephant.com/docs/neurondb/getting-started)** - Installation and quick start guide
- **[Full Documentation](https://www.pgelephant.com/docs/neurondb)** - Comprehensive guides and API reference
- **[GPU Acceleration Guide](docs/gpu.md)** - CUDA/ROCm/Metal GPU support documentation
- **[Installation Guide](INSTALL.md)** - Detailed installation instructions
- **[Contributing Guide](CONTRIBUTING.md)** - Development workflow and code standards
- **[Security Policy](SECURITY.md)** - Security best practices and vulnerability reporting

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

NeuronDB is released under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

**pgElephant, Inc.**  
Email: admin@pgelephant.com  
Website: https://pgelephant.com

Built for the PostgreSQL community with enterprise-grade reliability and 100% code quality standards.

---

<div align="center">

**[Documentation](https://www.pgelephant.com/docs/neurondb/getting-started)** • 
**[GitHub](https://github.com/pgElephant/NeurondB)** • 
**[Support](mailto:admin@pgelephant.com)**

</div>
