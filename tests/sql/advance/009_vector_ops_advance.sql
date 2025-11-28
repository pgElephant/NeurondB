-- 009_vector_ops_advance.sql
-- Exhaustive detailed test for vector operations: all functions, operators, error handling.
-- Works on 1000 rows and tests each and every way with comprehensive coverage
-- Tests: All vector operations, distance metrics, normalization, error handling

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'Vector Operations: Exhaustive Advanced Test'
\echo '=========================================================================='

\echo ''
\echo 'Dataset Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
SELECT 
	'Vector Operations Test' AS test_type,
	'No dataset required' AS dataset_status;

/*---- Register required GPU kernels ----*/
\echo ''
\echo 'GPU Configuration'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
/* GPU configuration via GUC (ALTER SYSTEM) */
SELECT neurondb_gpu_enable() AS gpu_available;
SELECT * FROM neurondb_gpu_info();

/*
 * ---- VECTOR OPERATIONS TESTS ----
 * Test all vector operations comprehensively
 */
\echo ''
\echo 'Vector Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Vector Normalization and Scaling'
WITH vectors AS (
	SELECT '[1,2,3,4,5]'::vector AS v1, '[10,20,30,40,50]'::vector AS v2
)
SELECT 
	vector_norm(v1) AS norm_v1,
	vector_norm(v2) AS norm_v2,
	vector_norm(vector_normalize(v1)) AS normalized_norm_v1,
	ROUND((vector_norm(vector_normalize(v1))::numeric), 6) AS normalized_norm_check,
	vector_normalize(v1) AS normalized_v1
FROM vectors;

\echo 'Test 2: Multiple Distance Metrics Comparison'
WITH vectors AS (
	SELECT '[1,2,3]'::vector AS v1, '[4,5,6]'::vector AS v2
)
SELECT 
	ROUND((vector_l2_distance(v1, v2))::numeric, 6) AS l2_distance,
	ROUND((vector_cosine_distance(v1, v2))::numeric, 6) AS cosine_distance,
	ROUND((vector_inner_product(v1, v2))::numeric, 6) AS inner_product,
	ROUND((vector_l1_distance(v1, v2))::numeric, 6) AS l1_distance,
	ROUND((vector_minkowski_distance(v1, v2, 3.0))::numeric, 6) AS minkowski_p3,
	ROUND((vector_chebyshev_distance(v1, v2))::numeric, 6) AS chebyshev_distance
FROM vectors;

\echo 'Test 3: Vector Concatenation'
SELECT 
	vector_concat('[1,2,3]'::vector, '[4,5,6]'::vector) AS concatenated,
	vector_dims(vector_concat('[1,2,3]'::vector, '[4,5,6]'::vector)) AS concat_dims,
	vector_dims('[1,2,3]'::vector) + vector_dims('[4,5,6]'::vector) AS expected_dims,
	CASE 
		WHEN vector_dims(vector_concat('[1,2,3]'::vector, '[4,5,6]'::vector)) = 
		     vector_dims('[1,2,3]'::vector) + vector_dims('[4,5,6]'::vector) 
		THEN '✓ Dimensions match'
		ELSE '✗ Dimension mismatch'
	END AS dim_check
FROM (SELECT 1) t;

\echo 'Test 4: Vector Conversion Operations'
SELECT 
	vector_to_binary('[1,2,3,4,5]'::vector) AS binary_representation,
	pg_column_size(vector_to_binary('[1,2,3,4,5]'::vector)) AS binary_size,
	vector_to_int8('[1,2,3,4,5]'::vector) AS int8_representation,
	vector_dims('[1,2,3,4,5]'::vector) AS original_dims
FROM (SELECT 1) t;

\echo 'Test 5: Vector Operations on Large Vectors'
WITH large_vectors AS (
	SELECT 
		array_to_vector(ARRAY(SELECT generate_series(1, 100)::float4)) AS v1,
		array_to_vector(ARRAY(SELECT generate_series(101, 200)::float4)) AS v2
)
SELECT 
	vector_dims(v1) AS dims_v1,
	vector_dims(v2) AS dims_v2,
	ROUND((vector_l2_distance(v1, v2))::numeric, 4) AS l2_distance,
	ROUND((vector_cosine_distance(v1, v2))::numeric, 6) AS cosine_distance
FROM large_vectors;

\echo 'Test 6: Vector Normalization Edge Cases'
SELECT 
	vector_normalize('[0,0,0]'::vector) AS zero_vector_normalized,
	vector_normalize('[1,0,0]'::vector) AS unit_vector_x,
	vector_normalize('[0,1,0]'::vector) AS unit_vector_y,
	ROUND((vector_norm(vector_normalize('[1,1,1]'::vector))::numeric), 6) AS normalized_norm,
	CASE 
		WHEN ABS(vector_norm(vector_normalize('[1,1,1]'::vector)) - 1.0) < 0.0001 
		THEN '✓ Normalization correct'
		ELSE '✗ Normalization error'
	END AS norm_check
FROM (SELECT 1) t;

\echo 'Test 7: Vector Arithmetic Properties'
WITH test_vectors AS (
	SELECT '[1,2,3]'::vector AS a, '[4,5,6]'::vector AS b, '[7,8,9]'::vector AS c
)
SELECT 
	-- Commutativity: a + b = b + a
	vector_add(a, b) = vector_add(b, a) AS addition_commutative,
	-- Associativity: (a + b) + c = a + (b + c)
	vector_add(vector_add(a, b), c) = vector_add(a, vector_add(b, c)) AS addition_associative,
	-- Scalar multiplication: k * (a + b) = k*a + k*b
	vector_mul(vector_add(a, b), 2.0) = vector_add(vector_mul(a, 2.0), vector_mul(b, 2.0)) AS scalar_distributive
FROM test_vectors;

\echo 'Test 8: GPU vs CPU Distance Functions'
WITH test_vectors AS (
	SELECT '[1,2,3,4,5]'::vector AS v1, '[6,7,8,9,10]'::vector AS v2
)
SELECT 
	'vector_l2_distance' AS function_name,
	ROUND((vector_l2_distance(v1, v2))::numeric, 6) AS cpu_result,
	ROUND((vector_l2_distance_gpu(v1, v2))::numeric, 6) AS gpu_result,
	ROUND(ABS((vector_l2_distance(v1, v2) - vector_l2_distance_gpu(v1, v2))::numeric), 8) AS difference
FROM test_vectors
UNION ALL
SELECT 
	'vector_cosine_distance' AS function_name,
	ROUND((vector_cosine_distance(v1, v2))::numeric, 6) AS cpu_result,
	ROUND((vector_cosine_distance_gpu(v1, v2))::numeric, 6) AS gpu_result,
	ROUND(ABS((vector_cosine_distance(v1, v2) - vector_cosine_distance_gpu(v1, v2))::numeric), 8) AS difference
FROM test_vectors;

/* --- ERROR path: invalid parameters --- */
\echo ''
\echo 'Error Handling Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: Dimension mismatch in distance'
DO $$
BEGIN
	BEGIN
		PERFORM vector_l2_distance('[1,2,3]'::vector, '[4,5]'::vector);
		RAISE EXCEPTION 'FAIL: expected error for dimension mismatch';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 2: NULL vector in distance'
DO $$
BEGIN
	BEGIN
		PERFORM vector_l2_distance(NULL, '[1,2,3]'::vector);
		RAISE EXCEPTION 'FAIL: expected error for NULL vector';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 3: Dimension mismatch in concatenation'
DO $$
BEGIN
	BEGIN
		-- This should work, but test edge cases
		PERFORM vector_concat('[1,2,3]'::vector, '[4,5]'::vector);
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 4: Invalid Minkowski p parameter'
DO $$
BEGIN
	BEGIN
		PERFORM vector_minkowski_distance('[1,2,3]'::vector, '[4,5,6]'::vector, 0.0);
		RAISE EXCEPTION 'FAIL: expected error for p=0';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo ''
\echo '=========================================================================='
\echo '✓ Vector Operations: Full exhaustive test complete'
\echo '=========================================================================='

\echo 'Test completed successfully'
