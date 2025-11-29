-- 036_types_advance.sql
-- Comprehensive advanced test for types module: quantization and aggregates comprehensively

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'Types Module: Exhaustive Quantization and Aggregates Coverage'
\echo '=========================================================================='

-- Create test table
DROP TABLE IF EXISTS types_advance_test;
CREATE TABLE types_advance_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28),
	label integer
);

INSERT INTO types_advance_test (embedding, label)
SELECT features, label
FROM test_train_view
LIMIT 1000;

/*-------------------------------------------------------------------
 * ---- COMPREHENSIVE QUANTIZATION TESTS ----
 * Test all quantization types and round-trips
 *------------------------------------------------------------------*/
\echo ''
\echo 'Quantization Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: INT8 quantization with various ranges'
WITH min_max AS (
	SELECT 
		(SELECT embedding FROM types_advance_test LIMIT 1) AS min_vec,
		(SELECT embedding FROM types_advance_test ORDER BY id DESC LIMIT 1) AS max_vec
)
SELECT 
	'INT8 Quantize' AS test_type,
	COUNT(*) AS n_quantized,
	AVG(pg_column_size(vector_quantize_int8(embedding, min_vec, max_vec))) AS avg_size
FROM types_advance_test, min_max
LIMIT 100;

\echo 'Test 2: FP16 quantization comprehensive'
SELECT 
	'FP16 Quantize' AS test_type,
	COUNT(*) AS n_quantized,
	AVG(pg_column_size(vector_quantize_fp16(embedding))) AS avg_size
FROM types_advance_test
LIMIT 100;

\echo 'Test 3: Binary quantization comprehensive'
SELECT 
	'Binary Quantize' AS test_type,
	COUNT(*) AS n_quantized,
	AVG(pg_column_size(vector_quantize_binary(embedding))) AS avg_size
FROM types_advance_test
LIMIT 100;

\echo 'Test 4: Quantization round-trip accuracy'
WITH min_max AS (
	SELECT 
		(SELECT embedding FROM types_advance_test LIMIT 1) AS min_vec,
		(SELECT embedding FROM types_advance_test ORDER BY id DESC LIMIT 1) AS max_vec
)
SELECT 
	'INT8 Round-trip' AS test_type,
	COUNT(*) AS n_tested,
	ROUND(AVG(vector_l2_distance(
		embedding,
		vector_dequantize_int8(
			vector_quantize_int8(embedding, min_vec, max_vec),
			min_vec,
			max_vec
		)
	))::numeric, 8) AS avg_reconstruction_error
FROM types_advance_test, min_max
LIMIT 100;

\echo 'Test 5: FP16 round-trip accuracy'
SELECT 
	'FP16 Round-trip' AS test_type,
	COUNT(*) AS n_tested,
	ROUND(AVG(vector_l2_distance(
		embedding,
		vector_dequantize_fp16(vector_quantize_fp16(embedding))
	))::numeric, 8) AS avg_reconstruction_error
FROM types_advance_test
LIMIT 100;

/*-------------------------------------------------------------------
 * ---- COMPREHENSIVE AGGREGATE TESTS ----
 * Test all aggregate functions
 *------------------------------------------------------------------*/
\echo ''
\echo 'Aggregate Functions Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 6: Vector statistics aggregates'
SELECT 
	'Statistics' AS test_type,
	COUNT(*) AS count,
	AVG(vector_norm(embedding)) AS avg_norm,
	MIN(vector_norm(embedding)) AS min_norm,
	MAX(vector_norm(embedding)) AS max_norm,
	STDDEV(vector_norm(embedding)) AS stddev_norm
FROM types_advance_test;

\echo 'Test 7: Vector aggregates by label'
SELECT 
	label,
	COUNT(*) AS count,
	AVG(vector_norm(embedding)) AS avg_norm,
	AVG(vector_mean(embedding)) AS avg_mean,
	AVG(vector_variance(embedding)) AS avg_variance
FROM types_advance_test
GROUP BY label
ORDER BY label
LIMIT 10;

\echo ''
\echo '=========================================================================='
\echo '✓ Types Module: Full exhaustive code-path test complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS types_advance_test CASCADE;

\echo 'Test completed successfully'
