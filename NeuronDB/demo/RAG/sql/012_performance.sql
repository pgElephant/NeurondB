-- ============================================================================
-- Test 012: RAG Performance Benchmarks
-- ============================================================================
-- Demonstrates: Query performance, index effectiveness
-- ============================================================================

\echo 'Testing RAG performance...'

\timing on

\echo ''
\echo 'Benchmark 1: Vector search performance'
EXPLAIN ANALYZE
SELECT dc.chunk_id, dc.chunk_text
FROM document_chunks dc
ORDER BY dc.embedding <=> neurondb_generate_embedding('sentence-transformers/all-MiniLM-L6-v2'::text,
    'database performance'
)
LIMIT 10;

\echo ''
\echo 'Benchmark 2: Hybrid search performance'
EXPLAIN ANALYZE
WITH vector_results AS (
    SELECT chunk_id, embedding <=> neurondb_generate_embedding('sentence-transformers/all-MiniLM-L6-v2'::text,
        'machine learning'
    ) AS dist
    FROM document_chunks
    ORDER BY dist
    LIMIT 20
)
SELECT COUNT(*) FROM vector_results;

\timing off
\echo ''
\echo 'Performance benchmarks complete!'


