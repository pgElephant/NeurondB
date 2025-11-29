-- 036_types_basic.sql
-- Basic test for types module: quantization and aggregates

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Types Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- QUANTIZATION TYPES ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Quantization Types Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: INT8 quantization'
SELECT 
	vector_quantize_int8(
		vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector,
		vector '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector,
		vector '[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]'::vector
	) AS int8_quantized;

\echo 'Test 2: FP16 quantization'
SELECT vector_quantize_fp16(vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector) AS fp16_quantized;

\echo 'Test 3: Binary quantization'
SELECT vector_quantize_binary(vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector) AS binary_quantized;

/*-------------------------------------------------------------------
 * ---- AGGREGATE FUNCTIONS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Aggregate Functions Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 4: Vector aggregates (if available)'
DROP TABLE IF EXISTS types_test_table;
CREATE TABLE types_test_table (
	id SERIAL PRIMARY KEY,
	embedding vector(28)
);

INSERT INTO types_test_table (embedding)
SELECT features FROM test_train_view LIMIT 100;

-- Test aggregates
SELECT 
	COUNT(*) AS count,
	AVG(vector_norm(embedding)) AS avg_norm,
	MIN(vector_norm(embedding)) AS min_norm,
	MAX(vector_norm(embedding)) AS max_norm
FROM types_test_table;

\echo ''
\echo '=========================================================================='
\echo '✓ Types Module: Basic tests complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS types_test_table CASCADE;

\echo 'Test completed successfully'
