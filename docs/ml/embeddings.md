# Embedding Generation

## What Are Embeddings?

**Embeddings** are dense vector representations of data (text, images, audio) that capture semantic meaning in a high-dimensional space. Unlike traditional keyword-based representations, embeddings encode contextual relationships, allowing machines to understand similarity and meaning.

### Why Embeddings Matter

Traditional databases store and search data based on exact matches or simple patterns. Embeddings enable **semantic search**: finding results based on *meaning* rather than exact text matches.

**Example:**
- Search query: "machine learning algorithms"
- Traditional search matches: documents containing those exact words
- Semantic search matches: documents about "neural networks", "deep learning models", "AI training methods" (related concepts)

### How Embeddings Work

1. **Input Data**: Text like "artificial intelligence"
2. **Model Processing**: Neural network transforms input into numbers
3. **Vector Output**: Array of floating-point numbers, e.g., `[0.234, -0.891, 0.456, ...]`
4. **Semantic Space**: Similar concepts have similar vectors (measured by distance metrics)

**Visual Representation:**
```
Text "cat" → [0.8, 0.2, 0.1, ...]     ┐
Text "dog" → [0.7, 0.3, 0.15, ...]    ├─ Close together (both animals)
Text "car" → [-0.3, 0.9, -0.5, ...]   ┘  Far apart (different concept)
```

### Embedding Dimensions

Embeddings typically have **384, 768, or 1536 dimensions** depending on the model:

- **384-dim**: Fast, efficient for most applications (all-MiniLM-L6-v2)
- **768-dim**: Balanced performance and accuracy (BERT-base)
- **1536-dim**: High accuracy for complex tasks (OpenAI text-embedding-ada-002)

## NeuronDB Embedding Capabilities

NeuronDB provides **built-in embedding generation** directly in PostgreSQL, eliminating the need for external API calls or separate services.

### Supported Embedding Types

1. **Text Embeddings**: Natural language processing
2. **Image Embeddings**: Computer vision applications
3. **Multimodal Embeddings**: Combined text and image
4. **Custom Models**: Load your own ONNX models

## Text Embeddings

### Basic Text Embedding

Generate embeddings from text using the default model:

```sql
-- Generate single embedding
SELECT embed_text('artificial intelligence');

-- Result: vector(384) containing the embedding
-- [0.234, -0.891, 0.456, ..., 0.123]
```

### Specify Model

Choose different embedding models for your use case:

```sql
-- Fast, efficient model (384 dimensions)
SELECT embed_text(
    'machine learning algorithms',
    'all-MiniLM-L6-v2'
);

-- Higher quality model (768 dimensions)
SELECT embed_text(
    'machine learning algorithms',
    'all-mpnet-base-v2'
);
```

### Batch Text Embedding

Process multiple texts efficiently:

```sql
-- Embed multiple texts at once
SELECT embed_text_batch(
    ARRAY[
        'artificial intelligence',
        'machine learning',
        'deep learning',
        'neural networks'
    ],
    'all-MiniLM-L6-v2'
);

-- Returns: array of vectors
```

**Performance Tip:** Batch processing is **3-5x faster** than individual calls due to GPU/CPU parallelization.

## Practical Examples

### Example 1: Store Document Embeddings

```sql
-- Create table with documents
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Insert document and generate embedding
INSERT INTO documents (title, content, embedding)
VALUES (
    'Introduction to AI',
    'Artificial intelligence is the simulation of human intelligence...',
    embed_text('Artificial intelligence is the simulation of human intelligence...')
);

-- Batch insert with embeddings
INSERT INTO documents (title, content, embedding)
SELECT 
    title,
    content,
    embed_text(content, 'all-MiniLM-L6-v2')
FROM imported_documents;
```

### Example 2: Semantic Search

```sql
-- Find similar documents using embeddings
SELECT 
    id,
    title,
    content,
    embedding <-> embed_text('machine learning basics') AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- Explain results:
-- Distance 0.1 = Very similar (nearly identical meaning)
-- Distance 0.5 = Moderately similar (related topics)
-- Distance 1.0+ = Dissimilar (different topics)
```

### Example 3: Update Existing Embeddings

```sql
-- Add embedding column to existing table
ALTER TABLE articles ADD COLUMN embedding vector(384);

-- Generate embeddings for existing data
UPDATE articles
SET embedding = embed_text(content, 'all-MiniLM-L6-v2')
WHERE embedding IS NULL;

-- Create index for fast similarity search
CREATE INDEX ON articles USING hnsw (embedding);
```

## Image Embeddings

### What Are Image Embeddings?

Image embeddings convert visual information into vectors, enabling:

- **Reverse image search**: Find similar images
- **Image classification**: Categorize images automatically
- **Multimodal search**: Search images using text descriptions

### Generate Image Embeddings

```sql
-- Embed image from binary data
SELECT embed_image(
    pg_read_binary_file('/path/to/image.jpg'),
    'clip'  -- CLIP model (text + image)
);

-- Store image embeddings
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    image_data BYTEA,
    embedding vector(512),
    created_at TIMESTAMPTZ DEFAULT now()
);

INSERT INTO images (filename, image_data, embedding)
VALUES (
    'product_photo.jpg',
    pg_read_binary_file('product_photo.jpg'),
    embed_image(pg_read_binary_file('product_photo.jpg'))
);
```

### Image Similarity Search

```sql
-- Find similar images
SELECT 
    filename,
    embedding <-> (
        SELECT embedding FROM images WHERE filename = 'query_image.jpg'
    ) AS similarity
FROM images
ORDER BY similarity
LIMIT 10;
```

## Multimodal Embeddings

### Combined Text and Image

CLIP (Contrastive Language-Image Pre-training) models create a shared embedding space for text and images, enabling **cross-modal search**.

```sql
-- Embed text and image together
SELECT embed_multimodal(
    'A red sports car on a mountain road',  -- Text description
    pg_read_binary_file('car_photo.jpg'),   -- Image data
    'clip'                                   -- Model
);

-- Search images using text
SELECT 
    filename,
    embedding <-> embed_text('sunset over ocean', 'clip') AS relevance
FROM images
ORDER BY relevance
LIMIT 10;
```

**Use Cases:**
- E-commerce: "Find blue dresses similar to this image"
- Content moderation: Match text policies to image content
- Digital asset management: Search photos by description

## Embedding Cache

### Why Caching Matters

Generating embeddings is computationally expensive. NeuronDB automatically caches embeddings to avoid redundant calculations.

### How Caching Works

```sql
-- First call: Generates embedding and caches it
SELECT embed_cached('artificial intelligence', 'all-MiniLM-L6-v2');
-- Execution time: 50ms

-- Second call: Returns cached embedding
SELECT embed_cached('artificial intelligence', 'all-MiniLM-L6-v2');
-- Execution time: <1ms (50x faster!)
```

### Cache Management

```sql
-- View cache statistics
SELECT 
    cache_key,
    model_name,
    created_at,
    access_count
FROM neurondb_embedding_cache
ORDER BY access_count DESC
LIMIT 20;

-- Clear old cache entries
DELETE FROM neurondb_embedding_cache
WHERE last_accessed < now() - interval '30 days'
  AND access_count < 5;

-- Manual cache insertion
INSERT INTO neurondb_embedding_cache (cache_key, embedding, model_name)
VALUES (
    'frequently_used_text',
    embed_text('frequently used text'),
    'all-MiniLM-L6-v2'
);
```

## Model Configuration

### List Available Models

```sql
-- View all loaded models
SELECT list_models();

-- Returns JSON with model details:
-- {
--   "all-MiniLM-L6-v2": {"dimensions": 384, "type": "sentence-transformer"},
--   "clip": {"dimensions": 512, "type": "multimodal"}
-- }
```

### Configure Model Parameters

```sql
-- Configure embedding model settings
SELECT configure_embedding_model(
    'all-MiniLM-L6-v2',
    '{
        "batch_size": 32,
        "normalize": true,
        "device": "cpu"
    }'::jsonb
);
```

## Best Practices

### 1. Choose the Right Model

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | ⚡⚡⚡ | ⭐⭐ | General purpose, fast |
| all-mpnet-base-v2 | 768 | ⚡⚡ | ⭐⭐⭐ | Higher quality search |
| CLIP | 512 | ⚡⚡ | ⭐⭐⭐ | Multimodal (text+image) |

### 2. Batch Processing

Always use batch functions for multiple texts:

```sql
-- ❌ Slow: Individual calls
DO $$
DECLARE
    doc RECORD;
BEGIN
    FOR doc IN SELECT * FROM documents LOOP
        UPDATE documents 
        SET embedding = embed_text(doc.content)
        WHERE id = doc.id;
    END LOOP;
END $$;

-- ✅ Fast: Batch processing
UPDATE documents
SET embedding = batch_embedding.emb
FROM (
    SELECT 
        id,
        unnest(embed_text_batch(array_agg(content))) AS emb
    FROM documents
    GROUP BY id % 100  -- Process in batches of 100
) batch_embedding
WHERE documents.id = batch_embedding.id;
```

### 3. Index Your Embeddings

Always create indexes for similarity search:

```sql
-- Create HNSW index for fast ANN search
CREATE INDEX idx_docs_embedding ON documents 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

-- Query uses index automatically
SELECT * FROM documents
ORDER BY embedding <-> embed_text('query')
LIMIT 10;
```

### 4. Monitor Cache Hit Rates

```sql
-- Check cache efficiency
SELECT 
    COUNT(*) as total_cached,
    SUM(access_count) as total_accesses,
    AVG(access_count) as avg_reuse
FROM neurondb_embedding_cache;

-- Cache hit rate should be > 50% for optimal performance
```

## Advanced Topics

### Custom Models

Load your own ONNX models:

```sql
-- Load custom model
SELECT load_model(
    'my-custom-model',
    '/path/to/model.onnx',
    'onnx'
);

-- Use custom model
SELECT embed_text('test text', 'my-custom-model');
```

### Fine-tuning

**Note:** The `finetune_model()` function is for fine-tuning ML models (classification, regression, etc.), not embedding models. Embedding models are typically pre-trained and used as-is, or fine-tuned using external tools before being loaded into NeuronDB.

For embedding model customization, consider:
- Using domain-specific pre-trained models (e.g., `sentence-transformers/all-mpnet-base-v2` for general text, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for multilingual)
- Fine-tuning embedding models externally using frameworks like Hugging Face Transformers, then loading the ONNX model into NeuronDB
- Using model configuration (see `configure_embedding_model()`) to adjust runtime parameters like batch size and device selection

## Next Steps

- [Hybrid Search](../hybrid/overview.md) - Combine embeddings with full-text search
- [Model Inference](inference.md) - Run custom ML models
- [Reranking](../reranking/cross-encoder.md) - Improve search quality
- [RAG Pipeline](../rag/overview.md) - Build retrieval-augmented generation systems

