/*
 * Test 01: Basic Vector Types
 * Tests creation and basic operations on neurondb vector types
 */

\echo '=== Test 01: Basic Vector Types ==='

-- Load extension
CREATE EXTENSION IF NOT EXISTS neurondb;

-- Test 1.1: Create vectorf32 type
SELECT '1.1: Create vectorf32'::text AS test;
SELECT '[1.0, 2.0, 3.0]'::vectorf32 AS result;
SELECT '[0.1, 0.2, 0.3, 0.4, 0.5]'::vectorf32 AS result;

-- Test 1.2: Create vectorf16 type
SELECT '1.2: Create vectorf16'::text AS test;
SELECT '[1, 2, 3]'::vectorf16 AS result;

-- Test 1.3: Create vectori8 type
SELECT '1.3: Create vectori8'::text AS test;
SELECT '[1, 2, 3, 4, 5]'::vectori8 AS result;

-- Test 1.4: Create vectorbin type
SELECT '1.4: Create vectorbin'::text AS test;
SELECT '101010'::vectorbin AS result;

-- Test 1.5: Vector dimensions
SELECT '1.5: Vector dimensions'::text AS test;
SELECT vector_dims('[1.0, 2.0, 3.0]'::vectorf32) AS dims;
SELECT vector_dims('[1, 2, 3, 4, 5]'::vectori8) AS dims;

-- Test 1.6: Create table with vectors
SELECT '1.6: Create table with vectors'::text AS test;
CREATE TABLE test_vectors (
    id serial PRIMARY KEY,
    vec_f32 vectorf32,
    vec_f16 vectorf16,
    vec_i8 vectori8,
    vec_bin vectorbin
);

-- Test 1.7: Insert vectors
SELECT '1.7: Insert vectors'::text AS test;
INSERT INTO test_vectors (vec_f32, vec_f16, vec_i8, vec_bin) VALUES
    ('[1.0, 2.0, 3.0]'::vectorf32, '[1, 2, 3]'::vectorf16, '[1, 2, 3]'::vectori8, '101'::vectorbin),
    ('[4.0, 5.0, 6.0]'::vectorf32, '[4, 5, 6]'::vectorf16, '[4, 5, 6]'::vectori8, '110'::vectorbin);

-- Test 1.8: Query vectors
SELECT '1.8: Query vectors'::text AS test;
SELECT id, vec_f32, vec_f16, vec_i8, vec_bin FROM test_vectors ORDER BY id;

-- Test 1.9: Vector type casts
SELECT '1.9: Vector type casts'::text AS test;
SELECT vectorf32_to_vectorf16('[1.0, 2.0, 3.0]'::vectorf32) AS result;
SELECT vectorf32_to_vectori8('[1.0, 2.0, 3.0]'::vectorf32) AS result;

-- Test 1.10: Error handling - invalid dimensions
SELECT '1.10: Error handling'::text AS test;
\set ON_ERROR_STOP 0
SELECT '[]'::vectorf32;
SELECT '[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]'::vectorf32 WHERE 1=0; -- Should not error
\set ON_ERROR_STOP 1

-- Cleanup
DROP TABLE test_vectors;

