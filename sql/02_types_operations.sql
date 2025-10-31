/*
 * Test 02: Vector Type Operations
 * Tests vector arithmetic, normalization, and transformations
 */

\echo '=== Test 02: Vector Type Operations ==='

-- Test 2.1: Vector addition
SELECT '2.1: Vector addition'::text AS test;
SELECT vector_add('[1.0, 2.0, 3.0]'::vectorf32, '[1.0, 1.0, 1.0]'::vectorf32) AS result;

-- Test 2.2: Vector subtraction
SELECT '2.2: Vector subtraction'::text AS test;
SELECT vector_sub('[5.0, 4.0, 3.0]'::vectorf32, '[1.0, 1.0, 1.0]'::vectorf32) AS result;

-- Test 2.3: Vector scalar multiplication
SELECT '2.3: Scalar multiplication'::text AS test;
SELECT vector_mul('[1.0, 2.0, 3.0]'::vectorf32, 2.0) AS result;

-- Test 2.4: Vector normalization
SELECT '2.4: Normalization'::text AS test;
SELECT vector_normalize('[3.0, 4.0]'::vectorf32) AS result; -- Should be [0.6, 0.8]

-- Test 2.5: Vector concatenation
SELECT '2.5: Concatenation'::text AS test;
SELECT vector_concat('[1.0, 2.0]'::vectorf32, '[3.0, 4.0]'::vectorf32) AS result;

-- Test 2.6: Vector slice
SELECT '2.6: Slice'::text AS test;
SELECT vector_slice('[1.0, 2.0, 3.0, 4.0, 5.0]'::vectorf32, 2, 4) AS result;

-- Test 2.7: Vector dot product
SELECT '2.7: Dot product'::text AS test;
SELECT vector_dot('[1.0, 2.0, 3.0]'::vectorf32, '[4.0, 5.0, 6.0]'::vectorf32) AS result;

-- Test 2.8: Vector magnitude
SELECT '2.8: Magnitude'::text AS test;
SELECT vector_magnitude('[3.0, 4.0]'::vectorf32) AS result; -- Should be 5.0

-- Test 2.9: Vector comparison
SELECT '2.9: Comparison'::text AS test;
SELECT '[1.0, 2.0, 3.0]'::vectorf32 = '[1.0, 2.0, 3.0]'::vectorf32 AS equal;
SELECT '[1.0, 2.0, 3.0]'::vectorf32 <> '[1.0, 2.0, 4.0]'::vectorf32 AS not_equal;

-- Test 2.10: Batch operations
SELECT '2.10: Batch operations'::text AS test;
CREATE TEMP TABLE vec_ops_test (
    id int,
    v1 vectorf32,
    v2 vectorf32
);
INSERT INTO vec_ops_test VALUES
    (1, '[1.0, 0, 0]'::vectorf32, '[0, 1.0, 0]'::vectorf32),
    (2, '[2.0, 0, 0]'::vectorf32, '[0, 2.0, 0]'::vectorf32);

SELECT id, vector_add(v1, v2) AS sum FROM vec_ops_test ORDER BY id;

DROP TABLE vec_ops_test;

