-- ============================================================================
-- Test 002: Text Embeddings with HuggingFace Models via ONNX
-- ============================================================================
-- Demonstrates: ONNX Runtime embedding generation, vector similarity
-- ============================================================================

\echo 'Testing HuggingFace text embeddings via ONNX Runtime...'

-- Check ONNX Runtime status
\echo ''
\echo 'ONNX Runtime Status:'
SELECT * FROM neurondb.onnx_runtime_status;

-- Test embedding generation
\echo ''
\echo 'Generating embeddings with all-MiniLM-L6-v2 (384-dim)...'

CREATE TABLE IF NOT EXISTS hf_embeddings_test (
    text_id SERIAL PRIMARY KEY,
    text_content TEXT NOT NULL,
    embedding VECTOR(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample texts
INSERT INTO hf_embeddings_test (text_content) VALUES
    ('PostgreSQL is a powerful open source database'),
    ('Machine learning enables computers to learn from data'),
    ('Vector databases store high-dimensional embeddings'),
    ('ONNX Runtime provides fast model inference'),
    ('HuggingFace offers thousands of pretrained models');

\echo 'Sample texts inserted'
\echo ''

-- Generate embeddings using ONNX
\timing on
UPDATE hf_embeddings_test
SET embedding = neurondb_hf_embedding('all-MiniLM-L6-v2', text_content);
\timing off

\echo ''
\echo 'Embeddings generated! Verifying...'

SELECT 
    text_id,
    text_content,
    vector_dims(embedding) AS embedding_dim,
    (vector_to_array(embedding))[1:5] AS first_5_dims
FROM hf_embeddings_test
WHERE embedding IS NOT NULL
LIMIT 3;

-- Test semantic similarity
\echo ''
\echo 'Testing semantic similarity search...'

WITH query_embedding AS (
    SELECT neurondb_hf_embedding('all-MiniLM-L6-v2', 
           'database systems') AS emb
)
SELECT 
    text_id,
    text_content,
    1 - (embedding <=> qe.emb) AS similarity
FROM hf_embeddings_test, query_embedding qe
ORDER BY embedding <=> qe.emb
LIMIT 3;

\echo ''
\echo 'Text embeddings test complete!'


