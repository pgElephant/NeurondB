\timing on
\pset footer off

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'hf_rerank';
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

-- Test Hugging Face reranking with GPU acceleration
\echo 'Testing Hugging Face reranking with GPU acceleration...'

-- Create test documents
DROP TABLE IF EXISTS hf_rerank_test;
CREATE TEMP TABLE hf_rerank_test (
	doc_id SERIAL PRIMARY KEY,
	document TEXT NOT NULL
);

-- Insert sample documents
INSERT INTO hf_rerank_test (document) VALUES
	('PostgreSQL is an advanced open source relational database management system'),
	('Machine learning algorithms learn patterns from data to make predictions'),
	('Vector databases are specialized databases for storing and querying high-dimensional vectors'),
	('GPU computing accelerates parallel processing tasks significantly'),
	('Database systems store and manage structured data efficiently');

\echo 'Sample documents inserted'

-- Test reranking
\echo ''
\echo 'Test 1: Single query reranking'
SELECT 
	doc_id,
	LEFT(document, 60) AS document_preview,
	ROUND(score::numeric, 4) AS relevance_score
FROM hf_rerank_test,
	LATERAL ndb_llm_rerank(
		'What is a database?',
		ARRAY[document],
		'all-MiniLM-L6-v2',
		5
	) AS t
ORDER BY score DESC;

-- Test batch reranking
\echo ''
\echo 'Test 2: Batch reranking with multiple queries'
WITH all_docs AS (
	SELECT array_agg(document ORDER BY doc_id) AS doc_array
	FROM hf_rerank_test
),
query_list AS (
	SELECT ARRAY[
		'database systems',
		'machine learning',
		'vector storage'
	] AS queries
)
SELECT 
	query_idx + 1 AS query_num,
	doc_idx + 1 AS doc_num,
	ROUND(score::numeric, 4) AS relevance_score
FROM query_list,
	LATERAL ndb_llm_rerank_batch(
		queries,
		ARRAY[
			(SELECT doc_array FROM all_docs),
			(SELECT doc_array FROM all_docs),
			(SELECT doc_array FROM all_docs)
		],
		'all-MiniLM-L6-v2',
		3
	) AS t
ORDER BY query_idx, score DESC;

-- Test top-k retrieval
\echo ''
\echo 'Test 3: Top-3 most relevant documents'
WITH ranked AS (
	SELECT 
		doc_id,
		document,
		score
	FROM hf_rerank_test,
		LATERAL ndb_llm_rerank(
			'What are vector databases used for?',
			ARRAY[document],
			'all-MiniLM-L6-v2',
			3
		) AS t
	ORDER BY score DESC
	LIMIT 3
)
SELECT 
	doc_id,
	LEFT(document, 70) AS document_preview,
	ROUND(score::numeric, 4) AS relevance_score
FROM ranked;

-- Cleanup
DROP TABLE IF EXISTS hf_rerank_test;

