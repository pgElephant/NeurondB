# NeuronDB - Advanced AI Database Extension for PostgreSQL

**Production-grade vector search, machine learning, and hybrid search—directly in PostgreSQL.**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/pgElephant/NeurondB)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%2B-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-PostgreSQL-blue.svg)](LICENSE)

---

## Overview

NeuronDB transforms PostgreSQL into a comprehensive AI database platform. Built from the ground up with PostgreSQL's architecture in mind, it provides enterprise-grade vector search, ML model inference, hybrid retrieval, and complete RAG pipeline support—all within your existing PostgreSQL infrastructure.

## Key Features

> **📚 Detailed documentation available for each feature below**

### Vector Search & Indexing
- **[Vector Types](https://pgelephant.com/neurondb/features/vector-types/)**: float32, packed, sparse, graph-based vectors
- **[Indexing](https://pgelephant.com/neurondb/features/indexing/)**: HNSW, IVF with automatic tuning
- **[Distance Metrics](https://pgelephant.com/neurondb/features/distance-metrics/)**: L2, Cosine, Inner Product, Manhattan, Hamming, and more
- **[Quantization](https://pgelephant.com/neurondb/features/quantization/)**: 2x-32x compression with minimal accuracy loss

### ML & Embeddings
- **[Embedding Generation](https://pgelephant.com/neurondb/ml/embeddings/)**: Text, image, multimodal embeddings with caching
- **[Model Inference](https://pgelephant.com/neurondb/ml/inference/)**: ONNX runtime, batch processing
- **[Model Management](https://pgelephant.com/neurondb/ml/model-management/)**: Load, export, version, monitor models
- **[Fine-tuning](https://pgelephant.com/neurondb/ml/finetuning/)**: Adapt models to your domain

### Hybrid Search & Retrieval
- **[Hybrid Search](https://pgelephant.com/neurondb/hybrid/overview/)**: Combine vector and full-text search
- **[Multi-Vector](https://pgelephant.com/neurondb/hybrid/multi-vector/)**: Multiple embeddings per document
- **[Faceted Search](https://pgelephant.com/neurondb/hybrid/faceted/)**: Category-aware retrieval
- **[Temporal Search](https://pgelephant.com/neurondb/hybrid/temporal/)**: Time-decay relevance

### Reranking
- **[Cross-Encoder](https://pgelephant.com/neurondb/reranking/cross-encoder/)**: Neural reranking
- **[LLM Reranking](https://pgelephant.com/neurondb/reranking/llm/)**: GPT/Claude-powered scoring
- **[ColBERT](https://pgelephant.com/neurondb/reranking/colbert/)**: Late interaction models
- **[Ensemble](https://pgelephant.com/neurondb/reranking/ensemble/)**: Combine strategies

### Analytics
- **[Clustering](https://pgelephant.com/neurondb/analytics/clustering/)**: K-means, DBSCAN
- **[Dimensionality Reduction](https://pgelephant.com/neurondb/analytics/dimensionality/)**: PCA, UMAP
- **[Outlier Detection](https://pgelephant.com/neurondb/analytics/outliers/)**: Isolation forest
- **[Quality Metrics](https://pgelephant.com/neurondb/analytics/quality/)**: Embedding assessment

### Background Workers
- **[neuranq](https://pgelephant.com/neurondb/workers/neuranq/)**: Async job queue executor
- **[neuranmon](https://pgelephant.com/neurondb/workers/neuranmon/)**: Query auto-tuner
- **[neurandefrag](https://pgelephant.com/neurondb/workers/neurandefrag/)**: Index maintenance

### Performance & Security
- **[Optimization](https://pgelephant.com/neurondb/performance/optimization/)**: SIMD, prefetching, caching
- **[Security](https://pgelephant.com/neurondb/security/overview/)**: Encryption, differential privacy, RLS
- **[Monitoring](https://pgelephant.com/neurondb/performance/monitoring/)**: Metrics and observability

## Quick Start

### Installation

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
├─────────────────────────────────────────────────────────┤
│  Vector Types  │  Distance Ops  │  Index Methods        │
├─────────────────────────────────────────────────────────┤
│  ML Inference  │  Embeddings    │  Model Management     │
├─────────────────────────────────────────────────────────┤
│  Hybrid Search │  Reranking     │  RAG Pipeline         │
├─────────────────────────────────────────────────────────┤
│  Background Workers (neuranq, neuranmon, neurandefrag)  │
├─────────────────────────────────────────────────────────┤
│  PostgreSQL Core (Storage, WAL, SPI, Shared Memory)     │
└─────────────────────────────────────────────────────────┘
```

**Design Principles:**
- Pure C implementation (40+ source files)
- PostgreSQL PGXS build system
- Shared memory for caching
- WAL integration for durability
- SPI for safe database operations
- Background worker framework

## Performance Benchmarks

Tested on AWS r6i.2xlarge (8 vCPU, 64GB RAM), 10M vectors, 768 dimensions:

| Operation | Throughput | Latency (p95) |
|-----------|------------|---------------|
| Vector Insert | 50K/sec | 2ms |
| HNSW Search (k=10) | 10K QPS | 5ms |
| Embedding Generation | 1K/sec | 10ms |
| Hybrid Search | 5K QPS | 8ms |
| Reranking | 2K/sec | 15ms |

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
```

## Monitoring

```sql
-- Extension statistics
SELECT * FROM pg_stat_neurondb;

-- Background worker status
SELECT * FROM neurondb_worker_status();

-- Query metrics
SELECT * FROM neurondb_query_metrics
ORDER BY timestamp DESC LIMIT 100;

-- Index maintenance status
SELECT * FROM neurondb_index_maintenance;
```

## Testing

```bash
# Regression tests (8 test suites)
make installcheck PG_CONFIG=/path/to/pg_config

# TAP tests
make prove PG_CONFIG=/path/to/pg_config
```

All tests pass on PostgreSQL 16, 17, 18 across Ubuntu, macOS, and Rocky Linux.

## Compatibility

| PostgreSQL | Status | Platforms |
|------------|--------|-----------|
| 16.x | ✅ Supported | Ubuntu, Debian, Rocky, macOS |
| 17.x | ✅ Supported | Ubuntu, Debian, Rocky, macOS |
| 18.x | ✅ Supported | Ubuntu, Debian, Rocky, macOS |

## Documentation

- **[Full Documentation](https://pgelephant.com/neurondb)** - Comprehensive guides and API reference
- **[Features List](FEATURES.md)** - Complete feature catalog
- **[Contributing Guide](CONTRIBUTING.md)** - Development workflow
- **[Security Policy](SECURITY.md)** - Security best practices

## Project Structure

```
neurondb/
├── src/              # C source files (40+ files)
├── include/          # Header files
├── sql/              # Regression test SQL files
├── expected/         # Expected test outputs
├── t/                # TAP test suite
├── docs/             # Documentation source
├── examples/         # Usage examples
└── benchmarks/       # Performance benchmarks
```

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/pgElephant/NeurondB/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pgElephant/NeurondB/discussions)
- **Email**: admin@pgelephant.com
- **Security**: Report vulnerabilities to admin@pgelephant.com

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines (PostgreSQL C standards)
- Development workflow
- Testing requirements
- Pull request process

## License

NeuronDB is released under the PostgreSQL License. See [LICENSE](LICENSE) for details.

## Authors

**pgElephant, Inc.**  
Email: admin@pgelephant.com  
Website: https://pgelephant.com

Built for the PostgreSQL community with enterprise-grade reliability.

---

<div align="center">

**[Documentation](https://pgelephant.com/neurondb)** • 
**[GitHub](https://github.com/pgElephant/NeurondB)** • 
**[Support](mailto:admin@pgelephant.com)**

</div>
