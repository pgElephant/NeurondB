\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo '=========================================================================='

-- Test 1: Vector Creation and Basic Properties
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_dims('[1,2,3,4,5]'::vector) AS dims,
	vector_norm('[1,2,3,4,5]'::vector) AS norm,
	vector_normalize('[1,2,3,4,5]'::vector) AS normalized;

-- Test 2: Vector Arithmetic Operations
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_add('[1,2,3]'::vector, '[4,5,6]'::vector) AS addition,
	vector_sub('[4,5,6]'::vector, '[1,2,3]'::vector) AS subtraction,
	vector_mul('[1,2,3]'::vector, 2.0) AS scalar_multiplication;

-- Test 3: Vector Distance Functions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH vectors AS (
	SELECT '[1,2,3]'::vector AS v1, '[4,5,6]'::vector AS v2
)
SELECT 
	vector_l2_distance(v1, v2) AS l2_distance,
	vector_cosine_distance(v1, v2) AS cosine_distance,
	vector_inner_product(v1, v2) AS inner_product,
	vector_l1_distance(v1, v2) AS l1_distance,
	vector_minkowski_distance(v1, v2, 3.0) AS minkowski_distance
FROM vectors;

-- Test 4: Vector Concatenation
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_concat('[1,2,3]'::vector, '[4,5,6]'::vector) AS concatenated,
	vector_dims(vector_concat('[1,2,3]'::vector, '[4,5,6]'::vector)) AS concat_dims;

-- Test 5: Vector Conversion
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_to_binary('[1,2,3,4,5]'::vector) AS binary_representation,
	pg_column_size(vector_to_binary('[1,2,3,4,5]'::vector)) AS binary_size;

\echo ''
\echo '=========================================================================='

\echo 'Test completed successfully'
