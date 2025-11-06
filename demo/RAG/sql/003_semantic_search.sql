-- ============================================================================
-- Test 003: Semantic Search with Vector Similarity
-- ============================================================================
-- Demonstrates: Vector similarity search, cosine distance, top-k retrieval
-- ============================================================================

\echo 'Testing semantic search with various queries...'

-- Query 1: Search for indexing information
\echo ''
\echo 'Query 1: "How do database indexes work?"'
WITH query_embedding AS (
    SELECT neurondb_generate_embedding('sentence-transformers/all-MiniLM-L6-v2'::text, 
        'How do database indexes work?'
    ) AS embedding
)
SELECT 
    dc.chunk_id,
    d.title,
    dc.chunk_text,
    1 - (dc.embedding <=> qe.embedding) AS similarity_score,
    RANK() OVER (ORDER BY dc.embedding <=> qe.embedding) AS rank
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN query_embedding qe
ORDER BY dc.embedding <=> qe.embedding
LIMIT 5;

-- Query 2: Search for RAG information
\echo ''
\echo 'Query 2: "What is retrieval augmented generation?"'
WITH query_embedding AS (
    SELECT neurondb_generate_embedding('sentence-transformers/all-MiniLM-L6-v2'::text, 
        'What is retrieval augmented generation?'
    ) AS embedding
)
SELECT 
    dc.chunk_id,
    d.title,
    dc.chunk_text,
    1 - (dc.embedding <=> qe.embedding) AS similarity_score
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN query_embedding qe
ORDER BY dc.embedding <=> qe.embedding
LIMIT 5;

-- Query 3: Search for ML best practices
\echo ''
\echo 'Query 3: "machine learning model training tips"'
WITH query_embedding AS (
    SELECT neurondb_generate_embedding('sentence-transformers/all-MiniLM-L6-v2'::text, 
        'machine learning model training tips'
    ) AS embedding
)
SELECT 
    dc.chunk_id,
    d.title,
    dc.chunk_text,
    1 - (dc.embedding <=> qe.embedding) AS similarity_score
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN query_embedding qe
ORDER BY dc.embedding <=> qe.embedding
LIMIT 5;

-- Query 4: Test semantic understanding (synonyms)
\echo ''
\echo 'Query 4: "vector similarity search" (testing semantic understanding)'
WITH query_embedding AS (
    SELECT neurondb_generate_embedding('sentence-transformers/all-MiniLM-L6-v2'::text, 
        'vector similarity search'
    ) AS embedding
)
SELECT 
    dc.chunk_id,
    d.title,
    left(dc.chunk_text, 100) || '...' AS chunk_preview,
    ROUND((1 - (dc.embedding <=> qe.embedding))::numeric, 4) AS similarity
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN query_embedding qe
ORDER BY dc.embedding <=> qe.embedding
LIMIT 5;

\echo ''
\echo 'Semantic search performance test:'
\timing on
WITH query_embedding AS (
    SELECT neurondb_generate_embedding('sentence-transformers/all-MiniLM-L6-v2'::text, 
        'database scalability strategies'
    ) AS embedding
)
SELECT COUNT(*) AS results_found
FROM (
    SELECT dc.chunk_id
    FROM document_chunks dc
    CROSS JOIN query_embedding qe
    ORDER BY dc.embedding <=> qe.embedding
    LIMIT 10
) results;
\timing off

