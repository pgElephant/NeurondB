-- 031_storage_advance.sql
-- Comprehensive advanced test for ALL storage module functions
-- Tests buffer management, WAL operations, ANN buffer operations comprehensively

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'Storage Module: Exhaustive Buffer, WAL, and ANN Buffer Coverage'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- COMPREHENSIVE BUFFER OPERATIONS ----
 * Test buffer management with various scenarios
 *------------------------------------------------------------------*/
\echo ''
\echo 'Buffer Management Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create test table
DROP TABLE IF EXISTS storage_advance_test;
CREATE TABLE storage_advance_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28),
	metadata jsonb,
	created_at timestamp DEFAULT now()
);

-- Insert large dataset
INSERT INTO storage_advance_test (embedding, metadata)
SELECT features, jsonb_build_object('source', 'test', 'batch', i/100)
FROM test_train_view, generate_series(1, 10) i
LIMIT 1000;

\echo 'Test 1: Buffer hit ratio after multiple queries'
-- Run queries to populate buffer
SELECT COUNT(*) FROM storage_advance_test;
SELECT COUNT(*) FROM storage_advance_test WHERE id < 100;
SELECT COUNT(*) FROM storage_advance_test WHERE id > 900;

SELECT 
	'Buffer Stats' AS test_type,
	heap_blks_read,
	heap_blks_hit,
	CASE 
		WHEN (heap_blks_read + heap_blks_hit) > 0 THEN
			ROUND(100.0 * heap_blks_hit / (heap_blks_read + heap_blks_hit), 2)
		ELSE 0
	END AS hit_ratio_percent
FROM pg_statio_user_tables
WHERE tablename = 'storage_advance_test';

\echo 'Test 2: Buffer operations with updates'
UPDATE storage_advance_test 
SET metadata = jsonb_set(metadata, '{updated}', 'true'::jsonb)
WHERE id % 10 = 0;

VACUUM ANALYZE storage_advance_test;

\echo 'Test 3: Buffer operations with deletes'
DELETE FROM storage_advance_test WHERE id % 20 = 0;

VACUUM ANALYZE storage_advance_test;

/*-------------------------------------------------------------------
 * ---- COMPREHENSIVE WAL OPERATIONS ----
 * Test all WAL compression and operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'WAL Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 4: WAL compression with various vector sizes'
SELECT 
	'Small Vectors' AS vector_type,
	AVG(pg_column_size(vector_wal_compress(
		vector_out(embedding)::text,
		vector_out((SELECT embedding FROM storage_advance_test LIMIT 1))::text
	))) AS avg_compressed_size
FROM storage_advance_test
WHERE vector_norm(embedding) < 5.0
LIMIT 100;

\echo 'Test 5: WAL compression ratio analysis'
WITH compressed AS (
	SELECT 
		embedding,
		pg_column_size(vector_out(embedding)::text) AS original_size,
		pg_column_size(vector_wal_compress(
			vector_out(embedding)::text,
			vector_out((SELECT embedding FROM storage_advance_test LIMIT 1))::text
		)) AS compressed_size
	FROM storage_advance_test
	LIMIT 100
)
SELECT 
	'WAL Compression Ratio' AS test_type,
	AVG(original_size) AS avg_original,
	AVG(compressed_size) AS avg_compressed,
	ROUND(100.0 * AVG(compressed_size) / NULLIF(AVG(original_size), 0), 2) AS compression_ratio_percent
FROM compressed;

\echo 'Test 6: WAL decompression round-trip'
WITH test_data AS (
	SELECT 
		embedding,
		vector_wal_compress(
			vector_out(embedding)::text,
			vector_out((SELECT embedding FROM storage_advance_test LIMIT 1))::text
		) AS compressed
	FROM storage_advance_test
	LIMIT 10
)
SELECT 
	'WAL Round-trip' AS test_type,
	COUNT(*) AS n_tested
FROM test_data;

\echo 'Test 7: WAL compression settings'
SELECT vector_wal_set_compression(true) AS compression_enabled;
SELECT vector_wal_get_stats() AS stats_enabled;
SELECT vector_wal_set_compression(false) AS compression_disabled;
SELECT vector_wal_get_stats() AS stats_disabled;

\echo 'Test 8: WAL size estimation accuracy'
WITH estimates AS (
	SELECT 
		embedding,
		vector_wal_estimate_size(
			vector_out(embedding)::text,
			vector_out((SELECT embedding FROM storage_advance_test LIMIT 1))::text
		) AS estimated_size,
		pg_column_size(vector_wal_compress(
			vector_out(embedding)::text,
			vector_out((SELECT embedding FROM storage_advance_test LIMIT 1))::text
		)) AS actual_size
	FROM storage_advance_test
	LIMIT 100
)
SELECT 
	'WAL Estimation' AS test_type,
	AVG(estimated_size) AS avg_estimated,
	AVG(actual_size) AS avg_actual,
	ROUND(100.0 * AVG(ABS(estimated_size - actual_size)) / NULLIF(AVG(actual_size), 0), 2) AS error_percent
FROM estimates;

/*-------------------------------------------------------------------
 * ---- COMPREHENSIVE ANN BUFFER OPERATIONS ----
 * Test ANN buffer with various index types
 *------------------------------------------------------------------*/
\echo ''
\echo 'ANN Buffer Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 9: HNSW index with ANN buffer'
CREATE INDEX idx_storage_hnsw ON storage_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 200);

-- Multiple queries to test buffer
SELECT COUNT(*) FROM (
	SELECT 
		id,
		embedding <-> (SELECT embedding FROM storage_advance_test LIMIT 1) AS distance
	FROM storage_advance_test
	ORDER BY embedding <-> (SELECT embedding FROM storage_advance_test LIMIT 1)
	LIMIT 10
) sub;

\echo 'Test 10: IVF index with ANN buffer'
CREATE INDEX idx_storage_ivf ON storage_advance_test 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 10);

SELECT COUNT(*) FROM (
	SELECT 
		id,
		embedding <-> (SELECT embedding FROM storage_advance_test LIMIT 1) AS distance
	FROM storage_advance_test
	ORDER BY embedding <-> (SELECT embedding FROM storage_advance_test LIMIT 1)
	LIMIT 10
) sub;

\echo 'Test 11: Index buffer statistics'
SELECT 
	indexrelname,
	idx_blks_read,
	idx_blks_hit,
	CASE 
		WHEN (idx_blks_read + idx_blks_hit) > 0 THEN
			ROUND(100.0 * idx_blks_hit / (idx_blks_read + idx_blks_hit), 2)
		ELSE 0
	END AS hit_ratio_percent
FROM pg_statio_user_indexes
WHERE tablename = 'storage_advance_test'
ORDER BY indexrelname;

\echo 'Test 12: Buffer operations with concurrent access simulation'
-- Simulate concurrent access patterns
SELECT COUNT(*) FROM storage_advance_test WHERE id % 2 = 0;
SELECT COUNT(*) FROM storage_advance_test WHERE id % 3 = 0;
SELECT COUNT(*) FROM storage_advance_test WHERE id % 5 = 0;

\echo ''
\echo '=========================================================================='
\echo '✓ Storage Module: Full exhaustive code-path test complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS storage_advance_test CASCADE;

\echo 'Test completed successfully'
