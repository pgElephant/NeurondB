-- 030_core_advance.sql
-- Comprehensive advanced test for ALL core module functions
-- Tests every operator, distance function, and code path in core module
-- Works on 1000 rows and tests each and every way with comprehensive coverage

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'Core Module: Exhaustive Operator and Distance Function Coverage'
\echo '=========================================================================='

-- Setup: Create test vectors
DROP TABLE IF EXISTS core_test_vectors;
CREATE TEMP TABLE core_test_vectors (
	id SERIAL PRIMARY KEY,
	v1 vector,
	v2 vector,
	v3 vector,
	label integer
);

-- Insert test vectors with various dimensions and values
INSERT INTO core_test_vectors (v1, v2, v3, label)
SELECT 
	features AS v1,
	(SELECT features FROM test_train_view ORDER BY random() LIMIT 1) AS v2,
	(SELECT features FROM test_train_view ORDER BY random() LIMIT 1) AS v3,
	label
FROM test_train_view
WHERE features IS NOT NULL
LIMIT 1000;

\echo ''
\echo 'Dataset Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
SELECT 
	COUNT(*)::bigint AS vector_count,
	(SELECT vector_dims(v1) FROM core_test_vectors LIMIT 1) AS feature_dim
FROM core_test_vectors;

/*-------------------------------------------------------------------
 * ---- COMPARISON OPERATORS ----
 * Test all comparison operators: =, <>, <, <=, >, >=
 *------------------------------------------------------------------*/
\echo ''
\echo 'Comparison Operators Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Equality operator (=)'
SELECT 
	'Equality' AS test_type,
	COUNT(*) FILTER (WHERE v1 = v1) AS equal_self,
	COUNT(*) FILTER (WHERE v1 = v2) AS equal_other,
	COUNT(*) FILTER (WHERE v1 <> v1) AS not_equal_self
FROM core_test_vectors;

\echo 'Test 2: Inequality operator (<>)'
SELECT 
	'Inequality' AS test_type,
	COUNT(*) FILTER (WHERE v1 <> v2) AS not_equal_count,
	COUNT(*) FILTER (WHERE v1 <> v1) AS not_equal_self
FROM core_test_vectors;

\echo 'Test 3: Less than operator (<)'
SELECT 
	'Less Than' AS test_type,
	COUNT(*) FILTER (WHERE v1 < v2) AS less_than_count,
	COUNT(*) FILTER (WHERE v1 < v1) AS less_than_self
FROM core_test_vectors;

\echo 'Test 4: Less than or equal operator (<=)'
SELECT 
	'Less Equal' AS test_type,
	COUNT(*) FILTER (WHERE v1 <= v2) AS less_equal_count,
	COUNT(*) FILTER (WHERE v1 <= v1) AS less_equal_self
FROM core_test_vectors;

\echo 'Test 5: Greater than operator (>)'
SELECT 
	'Greater Than' AS test_type,
	COUNT(*) FILTER (WHERE v1 > v2) AS greater_than_count,
	COUNT(*) FILTER (WHERE v1 > v1) AS greater_than_self
FROM core_test_vectors;

\echo 'Test 6: Greater than or equal operator (>=)'
SELECT 
	'Greater Equal' AS test_type,
	COUNT(*) FILTER (WHERE v1 >= v2) AS greater_equal_count,
	COUNT(*) FILTER (WHERE v1 >= v1) AS greater_equal_self
FROM core_test_vectors;

\echo 'Test 7: Comparison with identical vectors'
SELECT 
	'Identical Vectors' AS test_type,
	COUNT(*) FILTER (WHERE v1 = v1) AS all_equal_self,
	COUNT(*) FILTER (WHERE v1 <> v1) AS none_not_equal_self
FROM core_test_vectors;

\echo 'Test 8: Comparison with zero vectors'
SELECT 
	'Zero Vectors' AS test_type,
	(vector '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector = 
	 vector '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector) AS zero_equal,
	(vector '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector < 
	 vector '[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]'::vector) AS zero_less_than_one;

/*-------------------------------------------------------------------
 * ---- DISTANCE FUNCTIONS ----
 * Test all distance metrics: L2, cosine, inner product, L1, Hamming, Chebyshev, Minkowski
 *------------------------------------------------------------------*/
\echo ''
\echo 'Distance Functions Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: L2 (Euclidean) Distance'
SELECT 
	'L2 Distance' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(vector_l2_distance(v1, v2))::numeric, 6) AS avg_distance,
	ROUND(MIN(vector_l2_distance(v1, v2))::numeric, 6) AS min_distance,
	ROUND(MAX(vector_l2_distance(v1, v2))::numeric, 6) AS max_distance,
	ROUND(STDDEV(vector_l2_distance(v1, v2))::numeric, 6) AS stddev_distance
FROM core_test_vectors
LIMIT 100;

\echo 'Test 2: Cosine Distance'
SELECT 
	'Cosine Distance' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(vector_cosine_distance(v1, v2))::numeric, 6) AS avg_distance,
	ROUND(MIN(vector_cosine_distance(v1, v2))::numeric, 6) AS min_distance,
	ROUND(MAX(vector_cosine_distance(v1, v2))::numeric, 6) AS max_distance
FROM core_test_vectors
LIMIT 100;

\echo 'Test 3: Inner Product'
SELECT 
	'Inner Product' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(vector_inner_product(v1, v2))::numeric, 6) AS avg_product,
	ROUND(MIN(vector_inner_product(v1, v2))::numeric, 6) AS min_product,
	ROUND(MAX(vector_inner_product(v1, v2))::numeric, 6) AS max_product
FROM core_test_vectors
LIMIT 100;

\echo 'Test 4: L1 (Manhattan) Distance'
SELECT 
	'L1 Distance' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(vector_l1_distance(v1, v2))::numeric, 6) AS avg_distance,
	ROUND(MIN(vector_l1_distance(v1, v2))::numeric, 6) AS min_distance,
	ROUND(MAX(vector_l1_distance(v1, v2))::numeric, 6) AS max_distance
FROM core_test_vectors
LIMIT 100;

\echo 'Test 5: Hamming Distance'
SELECT 
	'Hamming Distance' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(vector_hamming_distance(v1, v2))::numeric, 2) AS avg_distance,
	MIN(vector_hamming_distance(v1, v2)) AS min_distance,
	MAX(vector_hamming_distance(v1, v2)) AS max_distance
FROM core_test_vectors
LIMIT 100;

\echo 'Test 6: Chebyshev Distance'
SELECT 
	'Chebyshev Distance' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(vector_chebyshev_distance(v1, v2))::numeric, 6) AS avg_distance,
	ROUND(MIN(vector_chebyshev_distance(v1, v2))::numeric, 6) AS min_distance,
	ROUND(MAX(vector_chebyshev_distance(v1, v2))::numeric, 6) AS max_distance
FROM core_test_vectors
LIMIT 100;

\echo 'Test 7: Minkowski Distance (p=1, equivalent to L1)'
SELECT 
	'Minkowski p=1' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(vector_minkowski_distance(v1, v2, 1.0))::numeric, 6) AS avg_distance
FROM core_test_vectors
LIMIT 100;

\echo 'Test 8: Minkowski Distance (p=2, equivalent to L2)'
SELECT 
	'Minkowski p=2' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(vector_minkowski_distance(v1, v2, 2.0))::numeric, 6) AS avg_distance
FROM core_test_vectors
LIMIT 100;

\echo 'Test 9: Minkowski Distance (p=3)'
SELECT 
	'Minkowski p=3' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(vector_minkowski_distance(v1, v2, 3.0))::numeric, 6) AS avg_distance
FROM core_test_vectors
LIMIT 100;

\echo 'Test 10: Minkowski Distance (p=infinity, equivalent to Chebyshev)'
SELECT 
	'Minkowski p=inf' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(vector_minkowski_distance(v1, v2, 1000.0))::numeric, 6) AS avg_distance
FROM core_test_vectors
LIMIT 100;

/*-------------------------------------------------------------------
 * ---- DISTANCE OPERATORS ----
 * Test distance operators: <-> (L2), <=> (cosine), <#> (inner product)
 *------------------------------------------------------------------*/
\echo ''
\echo 'Distance Operators Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: L2 Distance Operator (<->)'
SELECT 
	'L2 Operator' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(v1 <-> v2)::numeric, 6) AS avg_distance
FROM core_test_vectors
LIMIT 100;

\echo 'Test 2: Cosine Distance Operator (<=>)'
SELECT 
	'Cosine Operator' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(v1 <=> v2)::numeric, 6) AS avg_distance
FROM core_test_vectors
LIMIT 100;

\echo 'Test 3: Inner Product Operator (<#>)'
SELECT 
	'Inner Product Operator' AS test_type,
	COUNT(*) AS n_computations,
	ROUND(AVG(v1 <#> v2)::numeric, 6) AS avg_product
FROM core_test_vectors
LIMIT 100;

/*-------------------------------------------------------------------
 * ---- EDGE CASES ----
 * Test edge cases: zero vectors, identical vectors, orthogonal vectors
 *------------------------------------------------------------------*/
\echo ''
\echo 'Edge Cases Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Zero vector distances'
SELECT 
	'Zero Vectors' AS test_type,
	vector_l2_distance(vector '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector,
	                   vector '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector) AS l2_zero_zero,
	vector_cosine_distance(vector '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector,
	                      vector '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector) AS cosine_zero_zero;

\echo 'Test 2: Identical vector distances'
SELECT 
	'Identical Vectors' AS test_type,
	vector_l2_distance(v1, v1) AS l2_identical,
	vector_cosine_distance(v1, v1) AS cosine_identical,
	vector_l1_distance(v1, v1) AS l1_identical
FROM core_test_vectors
LIMIT 10;

\echo 'Test 3: Unit vectors'
SELECT 
	'Unit Vectors' AS test_type,
	vector_l2_distance(
		vector_normalize(vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector),
		vector_normalize(vector '[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]'::vector)
	) AS l2_normalized;

\echo 'Test 4: Orthogonal vectors (dot product should be ~0)'
SELECT 
	'Orthogonal Vectors' AS test_type,
	ABS(vector_inner_product(
		vector '[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector,
		vector '[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector
	)) AS inner_product_orthogonal;

\echo 'Test 5: Very large vectors'
SELECT 
	'Large Vectors' AS test_type,
	vector_l2_distance(
		(SELECT v1 FROM core_test_vectors ORDER BY vector_norm(v1) DESC LIMIT 1),
		(SELECT v2 FROM core_test_vectors ORDER BY vector_norm(v2) DESC LIMIT 1)
	) AS l2_large;

\echo 'Test 6: Very small vectors'
SELECT 
	'Small Vectors' AS test_type,
	vector_l2_distance(
		(SELECT v1 FROM core_test_vectors ORDER BY vector_norm(v1) ASC LIMIT 1),
		(SELECT v2 FROM core_test_vectors ORDER BY vector_norm(v2) ASC LIMIT 1)
	) AS l2_small;

/*-------------------------------------------------------------------
 * ---- BATCH OPERATIONS ----
 * Test batch distance computations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Batch Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Batch L2 distance computation'
SELECT 
	'Batch L2' AS test_type,
	COUNT(*) AS batch_size,
	ROUND(AVG(vector_l2_distance(v1, v2))::numeric, 6) AS avg_distance
FROM core_test_vectors;

\echo 'Test 2: Batch cosine distance computation'
SELECT 
	'Batch Cosine' AS test_type,
	COUNT(*) AS batch_size,
	ROUND(AVG(vector_cosine_distance(v1, v2))::numeric, 6) AS avg_distance
FROM core_test_vectors;

\echo 'Test 3: Batch inner product computation'
SELECT 
	'Batch Inner Product' AS test_type,
	COUNT(*) AS batch_size,
	ROUND(AVG(vector_inner_product(v1, v2))::numeric, 6) AS avg_product
FROM core_test_vectors;

\echo ''
\echo '=========================================================================='
\echo '✓ Core Module: Full exhaustive code-path test complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




