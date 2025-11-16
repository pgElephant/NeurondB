\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Vector Operations - Negative Test Cases (Error Handling)'
\echo '=========================================================================='

\echo ''
\echo 'Test 1: NULL Vector Operations'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_dims(NULL::vector);
SELECT vector_norm(NULL::vector);
SELECT vector_add(NULL::vector, '[1,2,3]'::vector);
SELECT vector_add('[1,2,3]'::vector, NULL::vector);
SELECT vector_l2_distance(NULL::vector, '[1,2,3]'::vector);
SELECT vector_l2_distance('[1,2,3]'::vector, NULL::vector);

\echo ''
\echo 'Test 2: Dimension Mismatch'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_add('[1,2,3]'::vector, '[4,5]'::vector);
SELECT vector_sub('[1,2,3]'::vector, '[4,5,6,7]'::vector);
SELECT vector_l2_distance('[1,2,3]'::vector, '[4,5]'::vector);
SELECT vector_cosine_distance('[1,2]'::vector, '[4,5,6]'::vector);

\echo ''
\echo 'Test 3: Empty Vector'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_dims('[]'::vector);
SELECT vector_norm('[]'::vector);
SELECT vector_normalize('[]'::vector);

\echo ''
\echo 'Test 4: Invalid Vector Format'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT '[1,2,abc]'::vector;
SELECT '[1,2,3,4,5'::vector;
SELECT '1,2,3,4,5]'::vector;

\echo ''
\echo 'Test 5: Zero Division in Normalization'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_normalize('[0,0,0]'::vector);
SELECT vector_cosine_distance('[0,0,0]'::vector, '[1,2,3]'::vector);

\echo ''
\echo 'Test 6: Invalid Scalar Operations'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_mul('[1,2,3]'::vector, NULL::float8);
SELECT vector_mul('[1,2,3]'::vector, 'NaN'::float8);
SELECT vector_mul('[1,2,3]'::vector, 'Infinity'::float8);

\echo ''
\echo 'Test 7: Invalid Distance Parameters'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_minkowski_distance('[1,2,3]'::vector, '[4,5,6]'::vector, 0.0);
SELECT vector_minkowski_distance('[1,2,3]'::vector, '[4,5,6]'::vector, -1.0);
SELECT vector_minkowski_distance('[1,2,3]'::vector, '[4,5,6]'::vector, NULL::float8);

\echo ''
\echo 'Test 8: Vector Concatenation Edge Cases'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_concat(NULL::vector, '[1,2,3]'::vector);
SELECT vector_concat('[1,2,3]'::vector, NULL::vector);
SELECT vector_concat('[]'::vector, '[1,2,3]'::vector);
SELECT vector_concat('[1,2,3]'::vector, '[]'::vector);

\echo ''
\echo 'Test 9: Conversion Operations with NULL'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_to_binary(NULL::vector);
SELECT vector_to_int8(NULL::vector);
SELECT int8_to_vector(NULL::bytea);

\echo ''
\echo 'Test 10: Invalid Binary Data'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT int8_to_vector('\x00'::bytea);
SELECT int8_to_vector('invalid_binary_data'::bytea);

\echo ''
\echo 'Negative Vector Operations Test Complete!'
\echo '=========================================================================='

