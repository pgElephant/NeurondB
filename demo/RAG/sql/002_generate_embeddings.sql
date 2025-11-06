-- ============================================================================
-- Test 002: Generate Vector Embeddings
-- ============================================================================
-- Demonstrates: Text embedding generation, vector storage
-- ============================================================================

\echo 'Generating embeddings for all document chunks...'

-- Update chunks with embeddings using NeuronDB embedding functions
-- Using sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)

UPDATE document_chunks
SET embedding = neurondb_generate_embedding('sentence-transformers/all-MiniLM-L6-v2', chunk_text)
WHERE embedding IS NULL;

\echo 'Verifying embeddings...'

SELECT 
    COUNT(*) AS chunks_with_embeddings,
    COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS embedded_chunks,
    COUNT(*) FILTER (WHERE embedding IS NULL) AS missing_embeddings
FROM document_chunks;

\echo ''
\echo 'Sample embedding vectors (first 10 dimensions):'
SELECT 
    chunk_id,
    left(chunk_text, 60) || '...' AS text_preview,
    (embedding::text::float[])[1:10] AS embedding_preview
FROM document_chunks
LIMIT 3;

\echo ''
\echo 'Embedding statistics:'
SELECT 
    MIN(array_length((embedding::text::float[]), 1)) AS min_dimensions,
    MAX(array_length((embedding::text::float[]), 1)) AS max_dimensions,
    AVG(array_length((embedding::text::float[]), 1)) AS avg_dimensions
FROM document_chunks
WHERE embedding IS NOT NULL;

