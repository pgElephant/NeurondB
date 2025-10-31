/*
 * Test 03: Distance Metrics
 * Tests all distance calculation functions
 */

\echo '=== Test 03: Distance Metrics ==='

-- Test 3.1: L2 Distance (Euclidean)
SELECT '3.1: L2 Distance'::text AS test;
SELECT l2_distance('[1.0, 2.0, 3.0]'::vectorf32, '[4.0, 5.0, 6.0]'::vectorf32) AS distance;
SELECT l2_distance('[0, 0]'::vectorf32, '[3, 4]'::vectorf32) AS distance; -- Should be 5.0

-- Test 3.2: Cosine Distance
SELECT '3.2: Cosine Distance'::text AS test;
SELECT cosine_distance('[1.0, 0, 0]'::vectorf32, '[1.0, 0, 0]'::vectorf32) AS distance; -- Should be 0
SELECT cosine_distance('[1.0, 0]'::vectorf32, '[0, 1.0]'::vectorf32) AS distance; -- Should be 1

-- Test 3.3: Inner Product Distance
SELECT '3.3: Inner Product'::text AS test;
SELECT inner_product('[1.0, 2.0, 3.0]'::vectorf32, '[4.0, 5.0, 6.0]'::vectorf32) AS distance;
SELECT inner_product('[1, 0]'::vectorf32, '[0, 1]'::vectorf32) AS distance; -- Should be 0

-- Test 3.4: Hamming Distance
SELECT '3.4: Hamming Distance'::text AS test;
SELECT hamming_distance('1010'::vectorbin, '1100'::vectorbin) AS distance;
SELECT hamming_distance('111'::vectorbin, '000'::vectorbin) AS distance; -- Should be 3

-- Test 3.5: Jaccard Distance
SELECT '3.5: Jaccard Distance'::text AS test;
SELECT jaccard_distance('1010'::vectorbin, '1100'::vectorbin) AS distance;

-- Test 3.6: Manhattan Distance (L1)
SELECT '3.6: Manhattan Distance'::text AS test;
SELECT l1_distance('[1.0, 2.0]'::vectorf32, '[4.0, 6.0]'::vectorf32) AS distance; -- Should be 7

-- Test 3.7: Minkowski Distance
SELECT '3.7: Minkowski Distance'::text AS test;
SELECT minkowski_distance('[1.0, 2.0]'::vectorf32, '[4.0, 6.0]'::vectorf32, 3) AS distance;

-- Test 3.8: Chebyshev Distance
SELECT '3.8: Chebyshev Distance'::text AS test;
SELECT chebyshev_distance('[1.0, 2.0, 3.0]'::vectorf32, '[4.0, 5.0, 6.0]'::vectorf32) AS distance; -- Should be 3

-- Test 3.9: Distance with quantized vectors
SELECT '3.9: Quantized distance'::text AS test;
SELECT l2_distance('[1, 2, 3]'::vectori8, '[4, 5, 6]'::vectori8) AS distance;

-- Test 3.10: Batch distance calculations
SELECT '3.10: Batch distances'::text AS test;
CREATE TEMP TABLE vectors_for_distance (
    id int,
    vec vectorf32
);
INSERT INTO vectors_for_distance VALUES
    (1, '[1.0, 0, 0]'::vectorf32),
    (2, '[0, 1.0, 0]'::vectorf32),
    (3, '[0, 0, 1.0]'::vectorf32);

SELECT v1.id, v2.id, l2_distance(v1.vec, v2.vec) AS distance
FROM vectors_for_distance v1
CROSS JOIN vectors_for_distance v2
WHERE v1.id < v2.id
ORDER BY v1.id, v2.id;

DROP TABLE vectors_for_distance;

