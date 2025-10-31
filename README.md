# NeuronDB - Advanced AI Database Extension for PostgreSQL

NeuronDB is a production-ready PostgreSQL extension that brings comprehensive AI and vector search capabilities directly into your database.

## Features

- **Multiple Vector Types**: float32, float16, int8, binary - choose the right precision for your use case
- **Advanced Indexing**: HNSW, IVF, Hybrid (vector + full-text), Temporal, Multi-tenant
- **Distance Metrics**: L2, Cosine, Inner Product, Manhattan, Hamming, Jaccard, Minkowski, Chebyshev
- **ML Integration**: Model inference, LLM calls, embedding generation, cost tracking
- **Hybrid Search**: Combine vector similarity with full-text search and metadata filters
- **Reranking**: Cross-encoder, LLM-based, ColBERT algorithms
- **Analytics**: K-means, DBSCAN, outlier detection, topic modeling, drift detection
- **Performance**: ANN buffer caching, WAL compression, parallel execution, prefetching
- **Multi-Tenant**: Tenant-scoped workers, usage metering, policy engine, audit logging
- **Security**: Vector encryption, differential privacy, row-level security, signed results
- **Observability**: Native `pg_stat_neurondb` view, query metrics, recall tracking

## Quick Start

```sql
-- Create extension
CREATE EXTENSION neurondb;

-- Create a table with vectors
CREATE TABLE documents (
    id serial PRIMARY KEY,
    content text,
    embedding vectorf32
);

-- Insert some vectors
INSERT INTO documents (content, embedding) VALUES
    ('AI and machine learning', '[0.1, 0.2, 0.3]'::vectorf32),
    ('Database systems', '[0.4, 0.5, 0.6]'::vectorf32),
    ('Web development', '[0.7, 0.8, 0.9]'::vectorf32);

-- Create HNSW index for fast similarity search
CREATE INDEX ON documents USING hnsw (embedding);

-- Find similar documents
SELECT content, l2_distance(embedding, '[0.15, 0.25, 0.35]'::vectorf32) AS distance
FROM documents
ORDER BY embedding <-> '[0.15, 0.25, 0.35]'::vectorf32
LIMIT 5;

-- Hybrid search (vector + full-text)
SELECT * FROM hybrid_search(
    'documents', 
    'embedding',
    '[0.1, 0.2, 0.3]'::vectorf32,
    'machine learning',
    0.7  -- 70% vector weight, 30% text weight
) LIMIT 10;
```

## Installation

### Prerequisites

- PostgreSQL 15, 16, 17, or 18
- GCC or Clang compiler
- libcurl (for model runtime features)
- OpenSSL (for encryption features)

### Build from Source

```bash
# Clone repository
git clone https://github.com/YOUR_ORG/NeurondB.git
cd NeurondB

# Build
make PG_CONFIG=/path/to/pg_config

# Install
sudo make install PG_CONFIG=/path/to/pg_config

# Create extension in your database
psql -d your_database -c "CREATE EXTENSION neurondb;"
```

### Running Tests

```bash
# Regression tests
make installcheck PG_CONFIG=/path/to/pg_config

# TAP tests
make prove PG_CONFIG=/path/to/pg_config
```

## Configuration

```sql
-- View all configuration
SELECT * FROM show_vector_config();

-- Set parameters
SELECT set_vector_config('ef_search', '100');
SELECT set_vector_config('ef_construction', '200');

-- Get specific parameter
SELECT get_vector_config('ef_search');
```

## Monitoring

```sql
-- View statistics
SELECT * FROM pg_stat_neurondb();

-- Reset statistics
SELECT pg_neurondb_stat_reset();

-- ANN buffer stats
SELECT neurondb_ann_buffer_get_stats();

-- WAL compression stats
SELECT vector_wal_get_stats();
```

## Architecture

NeuronDB is built with PostgreSQL's extension architecture and follows all PostgreSQL C coding standards:

- **Pure C Implementation**: 28 source files, 100% PostgreSQL compatible
- **Zero Dependencies**: Only requires PostgreSQL, libcurl, OpenSSL
- **PGXS Build System**: Standard PostgreSQL extension build
- **Shared Memory**: Efficient caching using PostgreSQL shared buffers
- **WAL Integration**: Full crash recovery and replication support

## Performance

- **Sub-millisecond queries**: HNSW index provides fast approximate nearest neighbor search
- **Horizontal scaling**: Shard-aware execution across multiple nodes
- **Memory efficient**: Quantization reduces storage by 2-32x
- **Replication friendly**: WAL compression reduces bandwidth by 2-5x

## Documentation

- [Features](FEATURES.md) - Complete feature list
- [Contributing](CONTRIBUTING.md) - How to contribute
- [Security](SECURITY.md) - Security policy and best practices
- [License](LICENSE) - MIT License

## Compatibility

| PostgreSQL Version | Status |
|--------------------|--------|
| 15                 | ✅ Supported |
| 16                 | ✅ Supported |
| 17                 | ✅ Supported |
| 18                 | ✅ Supported |

| Platform | Status |
|----------|--------|
| Linux (Ubuntu, Debian, Rocky) | ✅ Tested |
| macOS (Intel, Apple Silicon) | ✅ Tested |

## Support

- GitHub Issues: [Report bugs](https://github.com/YOUR_ORG/NeurondB/issues)
- Discussions: [Ask questions](https://github.com/YOUR_ORG/NeurondB/discussions)
- Security: security@neurondb.org

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits

NeuronDB is developed by the NeuronDB Development Group.

Built with ❤️ for the PostgreSQL community.
