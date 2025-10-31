# Vector Types

## Understanding Vectors in Databases

### What is a Vector?

A **vector** is a mathematical object represented as an array of numbers. In the context of AI and machine learning, vectors are used to represent data (text, images, audio) in a numerical format that computers can process and compare.

**Example Vector:**
```
[0.234, -0.891, 0.456, 0.123, -0.678]
```

This is a 5-dimensional vector where each number represents a feature or characteristic of the data.

### Why Store Vectors in Databases?

Traditional databases store structured data (numbers, text, dates). Modern AI applications need to store and search **unstructured data** like:

- **Text documents**: News articles, emails, customer reviews
- **Images**: Product photos, user uploads, medical scans
- **Audio**: Voice recordings, music files
- **Videos**: Security footage, user content

Vectors enable:
- **Semantic Search**: Find similar items based on meaning, not just keywords
- **Recommendation Systems**: Suggest related products or content
- **Anomaly Detection**: Identify unusual patterns in data
- **Classification**: Automatically categorize content

### How Vector Search Works

1. **Convert data to vectors** (embeddings)
2. **Store vectors** in database
3. **Search by similarity** using distance metrics
4. **Return nearest neighbors** (most similar items)

**Visual Example:**
```
Query: "laptop computers"
   ↓
Convert to vector: [0.8, 0.2, 0.1, ...]
   ↓
Find similar vectors in database:
   • "notebook PCs"     distance: 0.15 ✅ Very similar
   • "tablets"          distance: 0.45 ✅ Somewhat similar  
   • "bicycles"         distance: 2.30 ❌ Not similar
```

## NeuronDB Vector Types

NeuronDB provides multiple vector types optimized for different use cases, offering a balance between **accuracy**, **storage efficiency**, and **query speed**.

### 1. vector (Standard Precision)

The primary vector type using **32-bit floating-point** numbers (float32).

```sql
-- Create column with vector type
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    data vector(384)  -- 384-dimensional vector
);

-- Insert vector
INSERT INTO embeddings (data) 
VALUES ('[0.1, 0.2, 0.3, ...]'::vector);

-- Query vector
SELECT data FROM embeddings WHERE id = 1;
```

**Specifications:**
- **Precision**: 32-bit floating-point (float32)
- **Range**: ±1.175e-38 to ±3.402e+38
- **Storage**: 4 bytes × dimensions
- **Accuracy**: High (7 decimal digits precision)

**Use Cases:**
- General-purpose embeddings
- Research and development
- High-accuracy requirements
- When storage space is not a primary concern

**Example:**
```sql
-- Store 768-dimensional BERT embeddings
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    embedding vector(768)  -- 768 × 4 bytes = 3KB per vector
);

-- Calculate storage: 1 million documents
-- Storage = 1,000,000 × 3KB = 3GB
```

### 2. vectorp (Packed/Compressed)

Packed vector format with **optimized storage** layout for faster memory access.

```sql
-- Create packed vector
SELECT vectorp_in('[0.1, 0.2, 0.3, 0.4]')::text;

-- Get dimensions
SELECT vectorp_dims(vectorp_in('[1.0, 2.0, 3.0, 4.0]'));
-- Returns: 4
```

**Specifications:**
- **Layout**: Optimized binary format
- **Cache-friendly**: Aligned for CPU cache lines
- **SIMD-optimized**: Faster vector operations
- **Storage**: ~Same as vector, but better memory layout

**Use Cases:**
- High-performance applications
- Batch processing
- When query speed is critical
- Real-time similarity search

**Performance Comparison:**
```
Operation: Calculate 10,000 distances
- vector type:  45ms
- vectorp type: 28ms  (38% faster)
```

### 3. vecmap (Sparse Vectors)

Sparse vector representation storing only **non-zero values**, ideal for high-dimensional data with many zeros.

```sql
-- Create sparse vector
-- Format: {dim:total, nnz:count, indices:[...], values:[...]}
SELECT vecmap_in('{
    "dim": 10000,
    "nnz": 3,
    "indices": [0, 5, 9999],
    "values": [1.5, 2.3, 0.8]
}')::text;
```

**Specifications:**
- **Format**: Dictionary of keys (DOK)
- **Storage**: Only non-zero elements
- **Compression**: Up to 100x for very sparse data
- **Operations**: Optimized sparse math

**Use Cases:**
- NLP (TF-IDF vectors, BoW)
- Collaborative filtering
- Graph embeddings
- High-dimensional sparse data

**Storage Comparison:**
```
Example: 100,000-dimensional vector with 50 non-zero values

Dense (vector):
- Storage: 100,000 × 4 bytes = 400KB

Sparse (vecmap):
- Storage: (50 indices + 50 values) × 4 bytes = 400 bytes
- Compression: 1000x smaller!
```

### 4. vgraph (Graph-Based Vectors)

Specialized type for graph structures, storing vectors along with connectivity information.

```sql
-- Create graph vector
SELECT vgraph_in('{
    "nodes": 5,
    "edges": [
        [0, 1], [1, 2], [2, 3], [3, 4]
    ]
}')::text;
```

**Specifications:**
- **Structure**: Nodes + edges + vectors
- **Traversal**: Optimized for graph operations
- **Algorithms**: PageRank, community detection
- **Storage**: Adjacency list format

**Use Cases:**
- Knowledge graphs
- Social networks
- Recommendation graphs
- Citation networks

**Example Application:**
```sql
-- Store product recommendation graph
CREATE TABLE product_graph (
    product_id INT PRIMARY KEY,
    graph_data vgraph
);

-- Graph structure:
-- Product A → Related products [B, C, D]
-- Each edge has a similarity score
```

## Vector Type Comparison

| Type | Storage | Speed | Precision | Best For |
|------|---------|-------|-----------|----------|
| `vector` | 1x | Fast | High | General use, research |
| `vectorp` | 1x | Fastest | High | Production, high-throughput |
| `vecmap` | 0.01-0.1x | Medium | High | Sparse data, NLP |
| `vgraph` | 1.5x | Medium | High | Graph algorithms |

## Choosing the Right Type

### Decision Tree

```
Start
  ↓
Is your data sparse (>90% zeros)?
  YES → Use vecmap
  NO  → Continue
    ↓
  Do you need graph structure?
    YES → Use vgraph
    NO  → Continue
      ↓
    Is query speed critical?
      YES → Use vectorp
      NO  → Use vector (default)
```

### Real-World Scenarios

**Scenario 1: E-commerce Product Search**
- **Data**: Product descriptions (text embeddings)
- **Dimensions**: 384 (all-MiniLM-L6-v2)
- **Volume**: 1 million products
- **Query Rate**: 1000 QPS
- **Recommendation**: `vectorp` for speed

```sql
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name TEXT,
    description TEXT,
    embedding vectorp  -- Optimized for fast search
);
```

**Scenario 2: Document Archive**
- **Data**: Research papers (TF-IDF vectors)
- **Dimensions**: 50,000 (vocabulary size)
- **Sparsity**: 99.5% zeros (only 250 non-zero)
- **Volume**: 10 million documents
- **Recommendation**: `vecmap` for storage efficiency

```sql
CREATE TABLE papers (
    paper_id INT PRIMARY KEY,
    title TEXT,
    tfidf_vector vecmap  -- Sparse representation
);

-- Storage savings:
-- Dense: 10M × 50K × 4 bytes = 2TB
-- Sparse: 10M × 250 × 8 bytes = 20GB (100x smaller)
```

**Scenario 3: Social Network**
- **Data**: User connections and interests
- **Dimensions**: 128 (user embedding)
- **Structure**: Graph (friends, followers)
- **Volume**: 100 million users
- **Recommendation**: `vgraph` for relationships

```sql
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    network_data vgraph  -- Includes connections
);
```

## Working with Vector Types

### Type Conversion

```sql
-- Convert between types
SELECT 
    '[1.0, 2.0, 3.0]'::vector::text,           -- vector to text
    array_to_vector(ARRAY[1.0, 2.0, 3.0]),     -- array to vector
    vector_to_array('[1.0, 2.0, 3.0]'::vector) -- vector to array
;
```

### Vector Properties

```sql
-- Get vector dimensions
SELECT vector_dims('[1.0, 2.0, 3.0]'::vector);
-- Returns: 3

-- Get vector norm (magnitude)
SELECT vector_norm('[3.0, 4.0]'::vector);
-- Returns: 5.0 (sqrt(3² + 4²))

-- Normalize vector to unit length
SELECT vector_normalize('[3.0, 4.0]'::vector);
-- Returns: [0.6, 0.8]
```

### Vector Arithmetic

```sql
-- Add vectors
SELECT 
    '[1.0, 2.0]'::vector + '[3.0, 4.0]'::vector;
-- Returns: [4.0, 6.0]

-- Subtract vectors
SELECT 
    '[5.0, 7.0]'::vector - '[2.0, 3.0]'::vector;
-- Returns: [3.0, 4.0]

-- Scalar multiplication
SELECT 
    '[1.0, 2.0, 3.0]'::vector * 2.5;
-- Returns: [2.5, 5.0, 7.5]

-- Concatenate vectors
SELECT vector_concat(
    '[1.0, 2.0]'::vector,
    '[3.0, 4.0]'::vector
);
-- Returns: [1.0, 2.0, 3.0, 4.0]
```

## Performance Optimization

### Storage Optimization

```sql
-- Compress table with vectors
ALTER TABLE embeddings SET (
    toast_tuple_target = 128
);

-- Use tablespace for SSD storage
CREATE TABLESPACE fast_storage 
LOCATION '/mnt/ssd';

ALTER TABLE embeddings 
SET TABLESPACE fast_storage;
```

### Query Optimization

```sql
-- Create appropriate index
CREATE INDEX ON embeddings 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

-- Use index hints
SET enable_seqscan = OFF;  -- Force index usage
```

## Best Practices

### 1. Consistent Dimensions

```sql
-- ❌ Bad: Varying dimensions
INSERT INTO embeddings (data) VALUES
    ('[1.0, 2.0]'::vector),      -- 2D
    ('[1.0, 2.0, 3.0]'::vector); -- 3D (Error!)

-- ✅ Good: Fixed dimensions
CREATE TABLE embeddings (
    data vector(384),  -- Always 384 dimensions
    CHECK (vector_dims(data) = 384)
);
```

### 2. Normalization

```sql
-- Normalize vectors before storage
INSERT INTO embeddings (data)
VALUES (
    vector_normalize('[0.5, 0.3, 0.8]'::vector)
);

-- Benefits:
-- - Cosine distance = dot product (faster)
-- - Consistent magnitude for all vectors
-- - Better index performance
```

### 3. Batch Operations

```sql
-- Process vectors in batches
UPDATE embeddings
SET normalized_data = vector_normalize(data)
WHERE id BETWEEN 1 AND 10000;  -- Batch of 10K
```

## Next Steps

- [Distance Metrics](distance-metrics.md) - Learn how to compare vectors
- [Indexing](indexing.md) - Speed up vector search
- [Quantization](quantization.md) - Compress vectors further
- [Embeddings](../ml/embeddings.md) - Generate vectors from data

