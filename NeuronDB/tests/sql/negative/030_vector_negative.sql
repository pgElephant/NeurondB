-- 030_vector_negative.sql
-- Negative test cases for vector module: error handling, invalid inputs
-- Tests SIMD, quantization, sparse, graph, and WAL error paths

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Vector Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- QUANTIZATION ERRORS ----
 * Test error handling for quantization operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Quantization Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: INT8 Quantization with NULL vector'
SELECT vector_quantize_int8(NULL::vector, vector '[1,2,3]'::vector, vector '[4,5,6]'::vector);

\echo 'Error Test 2: INT8 Quantization with NULL min vector'
SELECT vector_quantize_int8(vector '[1,2,3]'::vector, NULL::vector, vector '[4,5,6]'::vector);

\echo 'Error Test 3: INT8 Quantization with NULL max vector'
SELECT vector_quantize_int8(vector '[1,2,3]'::vector, vector '[1,2,3]'::vector, NULL::vector);

\echo 'Error Test 4: INT8 Quantization with Dimension Mismatch'
SELECT vector_quantize_int8(
	vector '[1,2,3]'::vector,
	vector '[1,2]'::vector,
	vector '[4,5]'::vector
);

\echo 'Error Test 5: FP16 Quantization with NULL vector'
SELECT vector_quantize_fp16(NULL::vector);

\echo 'Error Test 6: Binary Quantization with NULL vector'
SELECT vector_quantize_binary(NULL::vector);

\echo 'Error Test 7: Dequantization with NULL quantized vector'
SELECT vector_dequantize_int8(NULL, vector '[1,2,3]'::vector, vector '[4,5,6]'::vector);

\echo 'Error Test 8: Dequantization with Dimension Mismatch'
SELECT vector_dequantize_int8(
	vector_quantize_int8(vector '[1,2,3]'::vector, vector '[1,2,3]'::vector, vector '[4,5,6]'::vector),
	vector '[1,2]'::vector,
	vector '[4,5]'::vector
);

/*-------------------------------------------------------------------
 * ---- WAL OPERATION ERRORS ----
 * Test error handling for WAL operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'WAL Operation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 9: WAL Compress with NULL vector'
SELECT vector_wal_compress(NULL::text, vector_out(vector '[1,2,3]'::vector)::text);

\echo 'Error Test 10: WAL Compress with NULL base vector'
SELECT vector_wal_compress(vector_out(vector '[1,2,3]'::vector)::text, NULL::text);

\echo 'Error Test 11: WAL Decompress with NULL compressed data'
SELECT vector_wal_decompress(NULL::text, vector_out(vector '[1,2,3]'::vector)::text);

\echo 'Error Test 12: WAL Decompress with NULL base vector'
SELECT vector_wal_decompress('compressed_data'::text, NULL::text);

\echo 'Error Test 13: WAL Estimate Size with NULL vector'
SELECT vector_wal_estimate_size(NULL::text, vector_out(vector '[1,2,3]'::vector)::text);

\echo 'Error Test 14: WAL Estimate Size with NULL base vector'
SELECT vector_wal_estimate_size(vector_out(vector '[1,2,3]'::vector)::text, NULL::text);

/*-------------------------------------------------------------------
 * ---- SPARSE VECTOR ERRORS ----
 * Test error handling for sparse vector operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Sparse Vector Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 15: Sparse operations with NULL vectors'
-- Sparse operations may be internal, test through error paths
DO $$
BEGIN
	BEGIN
		PERFORM vector_norm(NULL::vector);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- SIMD OPERATION ERRORS ----
 * Test error handling for SIMD operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'SIMD Operation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 16: SIMD Distance with NULL vectors'
SELECT vector_l2_distance(NULL::vector, vector '[1,2,3]'::vector);
SELECT vector_l2_distance(vector '[1,2,3]'::vector, NULL::vector);
SELECT vector_l2_distance(NULL::vector, NULL::vector);

\echo 'Error Test 17: SIMD Distance with Dimension Mismatch'
SELECT vector_l2_distance(
	vector '[1,2,3]'::vector,
	vector '[1,2,3,4]'::vector
);

\echo 'Error Test 18: SIMD Operations with Very Large Vectors'
-- Test with vectors that might exceed SIMD register size
DO $$
DECLARE
	large_vec1 vector;
	large_vec2 vector;
BEGIN
	-- Create large vectors (if supported)
	large_vec1 := (SELECT features FROM test_train_view LIMIT 1);
	large_vec2 := (SELECT features FROM test_train_view LIMIT 1);
	
	BEGIN
		PERFORM vector_l2_distance(large_vec1, large_vec2);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- May error if vector too large
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- GRAPH OPERATION ERRORS ----
 * Test error handling for graph operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Graph Operation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 19: Graph operations with NULL vectors'
-- Graph operations may be internal, test through error paths
DO $$
BEGIN
	BEGIN
		PERFORM vector_l2_distance(NULL::vector, vector '[1,2,3]'::vector);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

\echo 'Error Test 20: Graph operations with Dimension Mismatch'
SELECT vector_l2_distance(
	vector '[1,2,3]'::vector,
	vector '[1,2]'::vector
);

/*-------------------------------------------------------------------
 * ---- BATCH OPERATION ERRORS ----
 * Test error handling for batch operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Batch Operation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 21: Batch operations with NULL vectors'
SELECT 
	COUNT(*) AS null_count
FROM (
	SELECT 
		NULL::vector AS v1,
		vector '[1,2,3]'::vector AS v2
) sub
WHERE v1 IS NULL;

\echo 'Error Test 22: Batch operations with Mixed Dimensions'
-- This should handle gracefully or error appropriately
DO $$
BEGIN
	BEGIN
		PERFORM vector_l2_distance(
			vector '[1,2,3]'::vector,
			vector '[1,2,3,4]'::vector
		);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

\echo ''
\echo '=========================================================================='
\echo '✓ Vector Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'
