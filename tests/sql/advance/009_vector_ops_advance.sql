\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Vector Operations - Advanced Features Test'
\echo '=========================================================================='

\echo ''
\echo 'Test 1: Vector Normalization and Scaling'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH vectors AS (
	SELECT '[1,2,3,4,5]'::vector AS v1, '[10,20,30,40,50]'::vector AS v2
)
SELECT 
	vector_norm(v1) AS norm_v1,
	vector_norm(v2) AS norm_v2,
	vector_norm(vector_normalize(v1)) AS normalized_norm_v1,
	vector_normalize(v1) AS normalized_v1
FROM vectors;

\echo ''
\echo 'Test 2: Multiple Distance Metrics Comparison'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH vectors AS (
	SELECT '[1,2,3]'::vector AS v1, '[4,5,6]'::vector AS v2
)
SELECT 
	vector_l2_distance(v1, v2) AS l2_distance,
	vector_cosine_distance(v1, v2) AS cosine_distance,
	vector_inner_product(v1, v2) AS inner_product,
	vector_l1_distance(v1, v2) AS l1_distance,
	vector_minkowski_distance(v1, v2, 3.0) AS minkowski_p3,
	vector_chebyshev_distance(v1, v2) AS chebyshev_distance
FROM vectors;

\echo ''
\echo 'Test 3: Vector Concatenation'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_concat('[1,2,3]'::vector, '[4,5,6]'::vector) AS concatenated,
	vector_dims(vector_concat('[1,2,3]'::vector, '[4,5,6]'::vector)) AS concat_dims,
	vector_dims('[1,2,3]'::vector) + vector_dims('[4,5,6]'::vector) AS expected_dims;

\echo ''
\echo 'Test 4: Vector Conversion Operations'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_to_binary('[1,2,3,4,5]'::vector) AS binary_representation,
	pg_column_size(vector_to_binary('[1,2,3,4,5]'::vector)) AS binary_size,
	vector_to_int8('[1,2,3,4,5]'::vector) AS int8_representation;

\echo ''
\echo 'Test 5: Vector Operations on Large Vectors'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH large_vectors AS (
	SELECT 
		array_to_vector(ARRAY(SELECT generate_series(1, 100)::float4)) AS v1,
		array_to_vector(ARRAY(SELECT generate_series(101, 200)::float4)) AS v2
)
SELECT 
	vector_dims(v1) AS dims_v1,
	vector_dims(v2) AS dims_v2,
	vector_l2_distance(v1, v2) AS l2_distance,
	vector_cosine_distance(v1, v2) AS cosine_distance
FROM large_vectors;

\echo ''
\echo 'Test 6: Vector Normalization Edge Cases'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_normalize('[0,0,0]'::vector) AS zero_vector_normalized,
	vector_normalize('[1,0,0]'::vector) AS unit_vector_x,
	vector_normalize('[0,1,0]'::vector) AS unit_vector_y,
	vector_norm(vector_normalize('[1,1,1]'::vector)) AS normalized_norm;

\echo ''
\echo 'Test 7: Vector Arithmetic Properties'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

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

\echo ''
\echo 'Advanced Vector Operations Test Complete!'
\echo '=========================================================================='

