# Quick Start Guide

Get up and running with NeuronDB in 5 minutes.

## Prerequisites

- PostgreSQL 16, 17, or 18 installed
- NeuronDB extension installed (see [Installation Guide](installation.md))

## Step 1: Create Extension

```sql
-- Connect to your database
psql -d mydb

-- Create NeuronDB extension
CREATE EXTENSION neurondb;

-- Verify installation
SELECT extversion FROM pg_extension WHERE extname = 'neurondb';
```

## Step 2: Create Your First Vector Table

```sql
-- Create a table for storing documents with embeddings
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT now()
);
```

## Step 3: Generate and Store Embeddings

```sql
-- Insert articles with automatic embedding generation
INSERT INTO articles (title, content, embedding)
VALUES (
    'Introduction to Machine Learning',
    'Machine learning is a subset of artificial intelligence...',
    embed_text('Machine learning is a subset of artificial intelligence...')
);

INSERT INTO articles (title, content, embedding)
VALUES (
    'Deep Learning Fundamentals',
    'Deep learning uses neural networks with multiple layers...',
    embed_text('Deep learning uses neural networks with multiple layers...')
);

INSERT INTO articles (title, content, embedding)
VALUES (
    'Natural Language Processing',
    'NLP enables computers to understand human language...',
    embed_text('NLP enables computers to understand human language...')
);
```

## Step 4: Create Index for Fast Search

```sql
-- Create HNSW index for similarity search
SELECT hnsw_create_index(
    'articles',          -- table name
    'embedding',         -- column name
    'articles_idx',      -- index name
    16,                  -- m (connections per layer)
    200                  -- ef_construction
);
```

## Step 5: Perform Semantic Search

```sql
-- Find articles similar to a query
SELECT 
    id,
    title,
    embedding <-> embed_text('artificial intelligence') AS distance
FROM articles
ORDER BY distance
LIMIT 5;
```

## Next Steps

- **[Vector Types](../features/vector-types.md)** - Learn about different vector formats
- **[Embeddings](../ml/embeddings.md)** - Deep dive into embedding generation
- **[Hybrid Search](../hybrid/overview.md)** - Combine semantic and keyword search
- **[Configuration](configuration.md)** - Optimize NeuronDB for your workload

## Common Operations

### Update Existing Embeddings

```sql
-- Regenerate embeddings for all articles
UPDATE articles
SET embedding = embed_text(content)
WHERE embedding IS NULL;
```

### Hybrid Search (Vector + Text)

```sql
-- Combine semantic and keyword search
SELECT * FROM hybrid_search(
    'articles',
    embed_text('machine learning'),
    'neural networks',
    '{}',
    0.7,  -- 70% vector weight
    10    -- top 10 results
);
```

### Check Statistics

```sql
-- View extension statistics
SELECT * FROM pg_stat_neurondb;
```
