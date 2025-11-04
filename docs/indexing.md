# Indexing and Distance Metrics

NeuronDB provides powerful indexing and distance computation for vector search, supporting multiple distance functions and approximate nearest neighbor (ANN) algorithms.

## Distance Metrics

NeuronDB supports several distance metrics for vector similarity, each optimized for different use cases:

### L2 (Euclidean) Distance

The straight-line distance between two points. Lower values mean more similar vectors. Commonly used for embeddings that represent spatial relationships.

```sql
-- L2 distance operator
SELECT embedding <-> '[0.1, 0.2, 0.3]'::vector
FROM documents
ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector
LIMIT 10;

-- GPU-accelerated L2 distance (when GPU enabled)
SELECT id, vector_l2_distance_gpu(embedding, '[0.1, 0.2, 0.3]'::vector) AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

### Cosine Distance

Measures the angle between vectors, normalized to [0, 2]. Lower values indicate more similar directions. Ideal for text embeddings and normalized vectors where magnitude is less important than direction.

```sql
-- Cosine distance operator
SELECT embedding <=> '[0.1, 0.2, 0.3]'::vector
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, 0.3]'::vector
LIMIT 10;

-- GPU-accelerated cosine distance
SELECT id, vector_cosine_distance_gpu(embedding, '[0.1, 0.2, 0.3]'::vector) AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

### Inner Product (Dot Product)

Computes the dot product of two vectors. Higher values indicate greater similarity. Use with normalized vectors for maximum inner product search (MIPS).

```sql
-- Inner product operator (negative for ordering, higher is more similar)
SELECT embedding <#> '[0.1, 0.2, 0.3]'::vector
FROM documents
ORDER BY embedding <#> '[0.1, 0.2, 0.3]'::vector
LIMIT 10;

-- GPU-accelerated inner product
SELECT id, vector_inner_product_gpu(embedding, '[0.1, 0.2, 0.3]'::vector) AS score
FROM documents
ORDER BY score DESC
LIMIT 10;
```

## Distance Operators Summary

| Operator | Distance Type | Ordering | Use Case |
|----------|---------------|----------|----------|
| `<->` | L2 (Euclidean) | ASC (lower = more similar) | Spatial data, general embeddings |
| `<=>` | Cosine | ASC (lower = more similar) | Text embeddings, normalized vectors |
| `<#>` | Inner Product | ASC (higher raw value = more similar) | MIPS, normalized embeddings |

## Vector Indexes

NeuronDB implements advanced approximate nearest neighbor (ANN) index types for fast similarity search:

### HNSW (Hierarchical Navigable Small World)

A graph-based index offering excellent recall and speed. HNSW builds a multi-layer graph structure for efficient navigation to nearest neighbors. Recommended for most use cases.

```sql
-- Create HNSW index with L2 distance
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops);

-- Create HNSW index with cosine distance
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Create HNSW index with inner product
CREATE INDEX ON documents USING hnsw (embedding vector_ip_ops);

-- Custom parameters (m = max connections per layer, ef_construction = search width during build)
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- Runtime tuning: increase ef_search for better recall at query time
SET hnsw.ef_search = 100;

SELECT * FROM documents
ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector
LIMIT 10;
```

### IVF (Inverted File)

Partitions the vector space into clusters (centroids). At query time, only the nearest clusters are searched. Best for very large datasets where index build time and memory are critical.

```sql
-- Create IVF index with 100 lists (adjust based on dataset size)
CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- IVF with cosine distance
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Runtime tuning: probes controls how many lists to search (higher = better recall, slower)
SET ivfflat.probes = 10;

SELECT * FROM documents
ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector
LIMIT 10;
```

## Index Selection Guidelines

**Use HNSW when:**
- You need high recall and fast query performance
- Index build time and memory usage are acceptable
- Dataset size is small to medium (< 10M vectors)
- You want the best general-purpose ANN index

**Use IVF when:**
- Dataset is very large (> 10M vectors)
- Index build time and memory are constrained
- You can tolerate slightly lower recall for faster build
- Data distribution has natural clusters

**No index (exact search) when:**
- Dataset is very small (< 10k vectors)
- You require 100% recall (no approximation)
- Vectors are frequently updated (avoid index rebuild overhead)

## Operator Classes

Operator classes define which distance metric an index uses. Choose the operator class matching your query pattern:

| Operator Class | Distance Metric | Index Support |
|----------------|-----------------|---------------|
| `vector_l2_ops` | L2 (Euclidean) | HNSW, IVF |
| `vector_cosine_ops` | Cosine | HNSW, IVF |
| `vector_ip_ops` | Inner Product | HNSW, IVF |

**Important:** The operator class in your index must match the distance operator in your query. For example, an index with `vector_l2_ops` will be used for queries with `<->`, but not for `<=>` (cosine) queries.

## Index Tuning Parameters

### HNSW Parameters

- **m** (default: 16): Max connections per layer. Higher m = better recall, larger index. Typical range: 12–64.
- **ef_construction** (default: 64): Search width during index build. Higher = better index quality, slower build. Typical range: 64–500.
- **hnsw.ef_search** (runtime GUC, default: 40): Search width at query time. Higher = better recall, slower queries. Typical range: 40–400.

```sql
-- Build-time tuning
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops)
WITH (m = 32, ef_construction = 128);

-- Runtime tuning for higher recall
SET hnsw.ef_search = 200;
```

### IVF Parameters

- **lists** (build-time): Number of clusters. Rule of thumb: `sqrt(num_rows)` for datasets > 1M. Typical range: 100–10,000.
- **ivfflat.probes** (runtime GUC, default: 1): Number of clusters to search at query time. Higher = better recall, slower queries. Typical range: 1–100.

```sql
-- Build-time tuning (for 10M rows, sqrt(10M) ~ 3162)
CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops)
WITH (lists = 3000);

-- Runtime tuning for better recall
SET ivfflat.probes = 20;
```

## GPU-Accelerated Distance Functions

When GPU acceleration is enabled, NeuronDB provides GPU-accelerated distance computation functions for batch operations. See the [GPU Acceleration](gpu.md) documentation for configuration details.

```sql
-- Enable GPU acceleration (session-level)
SET neurondb.gpu_enabled = true;
SET neurondb.gpu_device = 0;
SET neurondb.gpu_batch_size = 1000;

-- GPU distance functions
SELECT id, 
       vector_l2_distance_gpu(embedding, query_vec) AS l2_dist,
       vector_cosine_distance_gpu(embedding, query_vec) AS cos_dist,
       vector_inner_product_gpu(embedding, query_vec) AS ip_score
FROM documents, (SELECT '[0.1, 0.2, 0.3]'::vector AS query_vec) q
ORDER BY l2_dist
LIMIT 100;
```

## Best Practices

### 1. Index Selection
Start with HNSW for most workloads. Switch to IVF if your dataset exceeds 10M vectors or index build time becomes prohibitive.

### 2. Distance Metric
Use cosine distance for text embeddings from models like OpenAI, Cohere, or Sentence Transformers. Use L2 for image embeddings or when embedding model documentation recommends Euclidean distance.

### 3. Normalization
For cosine similarity, normalizing vectors before storage can improve performance. L2-normalized vectors make cosine distance equivalent to L2 distance (with a scaling factor).

### 4. Index Maintenance
Indexes are automatically updated on INSERT/UPDATE/DELETE, but frequent updates can degrade quality. For bulk loads, consider building the index after data ingestion with `CREATE INDEX`.

### 5. Query Tuning
Adjust runtime GUCs (`hnsw.ef_search`, `ivfflat.probes`) to balance recall and latency. Monitor query performance and index scan statistics with `EXPLAIN ANALYZE`.

## Example: Complete Indexing Workflow

```sql
-- 1. Create table with vector column
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(1536)  -- OpenAI ada-002 dimension
);

-- 2. Insert embeddings (bulk load or incremental)
INSERT INTO documents (content, embedding)
SELECT content, embedding FROM external_source;

-- 3. Create HNSW index for cosine similarity (text embeddings)
CREATE INDEX documents_embedding_idx ON documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. Tune runtime parameters for better recall
SET hnsw.ef_search = 100;

-- 5. Perform similarity search
SELECT id, content, embedding <=> '[0.1, 0.2, ...]'::vector AS distance
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- 6. Monitor index usage
EXPLAIN ANALYZE
SELECT id FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

## Next Steps

- [GPU Acceleration](gpu.md): Learn about GPU-accelerated distance computation
- [Hybrid Search](hybrid.md): Combine vector and full-text search
- [ML Analytics](analytics.md): Use clustering and outlier detection
- [Performance](performance.md): Optimize indexing and query performance
