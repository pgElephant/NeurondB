-- NeurondB: Complete RAG Pipeline Example

CREATE EXTENSION IF NOT EXISTS neurondb;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 1. Create knowledge base schema
CREATE TABLE knowledge_base (
    id bigserial PRIMARY KEY,
    title text NOT NULL,
    content text NOT NULL,
    source text,
    metadata jsonb DEFAULT '{}',
    embedding vector(384),
    fts tsvector GENERATED ALWAYS AS (
        to_tsvector('english', title || ' ' || content)
    ) STORED,
    created_at timestamp DEFAULT now(),
    updated_at timestamp DEFAULT now()
);

-- 2. Create indexes
CREATE INDEX kb_embedding_idx ON knowledge_base USING ivfflat (embedding);
CREATE INDEX kb_fts_idx ON knowledge_base USING gin(fts);
CREATE INDEX kb_metadata_idx ON knowledge_base USING gin(metadata);
CREATE INDEX kb_created_idx ON knowledge_base(created_at DESC);

-- 3. Insert sample documents
INSERT INTO knowledge_base (title, content, source, metadata) VALUES
('Introduction to AI', 
 'Artificial Intelligence is the simulation of human intelligence by machines...',
 'docs/ai-intro.md',
 '{"category": "AI", "difficulty": "beginner"}'),
 
('Machine Learning Basics',
 'Machine Learning is a subset of AI that enables systems to learn from data...',
 'docs/ml-basics.md',
 '{"category": "ML", "difficulty": "beginner"}'),
 
('Deep Learning Networks',
 'Deep Learning uses neural networks with multiple layers to learn complex patterns...',
 'docs/dl-networks.md',
 '{"category": "DL", "difficulty": "advanced"}');

-- 4. Generate embeddings (in production, use real embedding model)
-- Placeholder: In real usage, call embed_text() or embed_text_batch()
UPDATE knowledge_base
SET embedding = embed_text(title || ' ' || content, 'all-MiniLM-L6-v2')
WHERE embedding IS NULL;

-- 5. Simple vector search
WITH query AS (
    SELECT embed_text('What is machine learning?', 'all-MiniLM-L6-v2') as query_vec
)
SELECT 
    id,
    title,
    embedding <-> query_vec as distance
FROM knowledge_base, query
ORDER BY distance
LIMIT 5;

-- 6. Hybrid search (vector + FTS + metadata)
SELECT *
FROM hybrid_search(
    'knowledge_base',
    embed_text('deep learning neural networks', 'all-MiniLM-L6-v2'),
    'neural networks',
    '{"category": "DL"}'::text,
    0.7,  -- 70% vector weight, 30% FTS weight
    10
);

-- 7. Two-stage retrieval with reranking
WITH candidates AS (
    -- Stage 1: Fast approximate search (100 candidates)
    SELECT id, title, content
    FROM knowledge_base
    ORDER BY embedding <-> embed_text('explain AI', 'all-MiniLM-L6-v2')
    LIMIT 100
)
-- Stage 2: Precise reranking (top 10)
SELECT 
    c.id,
    c.title,
    r.score
FROM candidates c,
LATERAL rerank_cross_encoder(
    'explain AI to a beginner',
    ARRAY[c.content],
    'ms-marco-MiniLM-L-6-v2',
    10
) r
ORDER BY r.score DESC
LIMIT 10;

-- 8. Temporal-aware search (boost recent documents)
SELECT *
FROM temporal_vector_search(
    'knowledge_base',
    embed_text('AI trends', 'all-MiniLM-L6-v2'),
    'created_at',
    0.01,  -- Exponential decay rate
    10
);

-- 9. Diverse results (Maximal Marginal Relevance)
SELECT *
FROM diverse_vector_search(
    'knowledge_base',
    embed_text('machine learning', 'all-MiniLM-L6-v2'),
    0.5,  -- 50% relevance, 50% diversity
    10
);

-- 10. Multi-vector search (ColBERT-style)
WITH query_tokens AS (
    SELECT unnest(ARRAY[
        embed_text('what', 'all-MiniLM-L6-v2'),
        embed_text('is', 'all-MiniLM-L6-v2'),
        embed_text('deep learning', 'all-MiniLM-L6-v2')
    ]) as token_vec
)
SELECT *
FROM multi_vector_search(
    'knowledge_base',
    ARRAY(SELECT token_vec FROM query_tokens),
    'max',  -- max similarity across tokens
    10
);

-- 11. Analyze embeddings
SELECT compute_embedding_quality('knowledge_base', 'embedding');

-- 12. Find similar documents (content-based recommendation)
WITH target AS (
    SELECT embedding FROM knowledge_base WHERE id = 1
)
SELECT 
    kb.id,
    kb.title,
    kb.embedding <-> t.embedding as similarity
FROM knowledge_base kb, target t
WHERE kb.id != 1
ORDER BY similarity
LIMIT 5;

-- 13. Clustering analysis
SELECT 
    kb.title,
    c.cluster
FROM knowledge_base kb
JOIN cluster_kmeans('knowledge_base', 'embedding', 3, 100) c
    ON kb.id = c.id;

-- 14. Outlier detection
SELECT 
    kb.title,
    o.outlier_score
FROM knowledge_base kb
JOIN detect_outliers('knowledge_base', 'embedding', 'isolation_forest', 0.1) o
    ON kb.id = o.id
ORDER BY o.outlier_score DESC;

-- 15. Topic discovery
SELECT * FROM discover_topics('knowledge_base', 'embedding', 5);

-- 16. Performance monitoring
SELECT 
    COUNT(*) as total_docs,
    AVG(vector_dims(embedding)) as avg_dims,
    AVG(vector_norm(embedding)) as avg_norm,
    COUNT(*) FILTER (WHERE embedding IS NOT NULL) as embedded_docs
FROM knowledge_base;

