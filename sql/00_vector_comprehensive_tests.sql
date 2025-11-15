/*-------------------------------------------------------------------------
 *
 * 00_vector_comprehensive_tests.sql
 *    Comprehensive test suite for all vector functions
 *
 * Tests include:
 *   - Basic functionality tests (happy path)
 *   - Advanced/edge case tests
 *   - Negative tests (error conditions, invalid inputs)
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 *-------------------------------------------------------------------------
 */

\set ON_ERROR_STOP on

-- ============================================================================
-- SECTION 1: BASIC TESTS - Happy Path
-- ============================================================================

\echo '============================================================================'
\echo 'SECTION 1: BASIC TESTS - Happy Path'
\echo '============================================================================'

-- Test 1.1: Vector creation and I/O
\echo ''
\echo 'Test 1.1: Vector creation and I/O'
SELECT '[1,2,3]'::vector AS v1;
SELECT '[1.5,2.5,3.5]'::vector AS v2;
SELECT '[0,0,0]'::vector AS zero_vector;
SELECT '[1,1,1,1,1]'::vector AS ones_vector;

-- Test 1.2: Vector dimensions
\echo ''
\echo 'Test 1.2: Vector dimensions'
SELECT vector_dims('[1,2,3]'::vector) = 3 AS test_dims;
SELECT vector_dims('[1,2,3,4,5]'::vector) = 5 AS test_dims_5;

-- Test 1.3: Vector norm
\echo ''
\echo 'Test 1.3: Vector norm'
SELECT abs(vector_norm('[3,4]'::vector) - 5.0) < 0.0001 AS test_norm_3_4;
SELECT abs(vector_norm('[1,1,1]'::vector) - sqrt(3.0)) < 0.0001 AS test_norm_1_1_1;
SELECT vector_norm('[0,0,0]'::vector) = 0.0 AS test_norm_zero;

-- Test 1.4: Vector addition
\echo ''
\echo 'Test 1.4: Vector addition'
SELECT '[1,2,3]'::vector + '[4,5,6]'::vector = '[5,7,9]'::vector AS test_add;
SELECT '[0,0]'::vector + '[1,1]'::vector = '[1,1]'::vector AS test_add_zero;

-- Test 1.5: Vector subtraction
\echo ''
\echo 'Test 1.5: Vector subtraction'
SELECT '[5,7,9]'::vector - '[1,2,3]'::vector = '[4,5,6]'::vector AS test_sub;
SELECT '[1,1]'::vector - '[1,1]'::vector = '[0,0]'::vector AS test_sub_zero;

-- Test 1.6: Vector scalar multiplication
\echo ''
\echo 'Test 1.6: Vector scalar multiplication'
SELECT '[1,2,3]'::vector * 2.0 = '[2,4,6]'::vector AS test_mul_scalar;
SELECT '[1,1,1]'::vector * 0.0 = '[0,0,0]'::vector AS test_mul_zero;

-- Test 1.7: Vector concatenation
\echo ''
\echo 'Test 1.7: Vector concatenation'
SELECT vector_concat('[1,2]'::vector, '[3,4]'::vector) = '[1,2,3,4]'::vector AS test_concat;

-- Test 1.8: Vector normalization
\echo ''
\echo 'Test 1.8: Vector normalization'
SELECT abs(vector_norm(vector_normalize('[3,4]'::vector)) - 1.0) < 0.0001 AS test_normalize;

-- Test 1.9: Distance metrics - L2
\echo ''
\echo 'Test 1.9: Distance metrics - L2'
SELECT abs(vector_l2_distance('[0,0]'::vector, '[3,4]'::vector) - 5.0) < 0.0001 AS test_l2;
SELECT vector_l2_distance('[1,1]'::vector, '[1,1]'::vector) = 0.0 AS test_l2_same;

-- Test 1.10: Distance metrics - Cosine
\echo ''
\echo 'Test 1.10: Distance metrics - Cosine'
SELECT abs(vector_cosine_distance('[1,0]'::vector, '[0,1]'::vector) - 1.0) < 0.0001 AS test_cosine_orthogonal;
SELECT vector_cosine_distance('[1,1]'::vector, '[2,2]'::vector) = 0.0 AS test_cosine_parallel;

-- Test 1.11: Distance metrics - Inner Product
\echo ''
\echo 'Test 1.11: Distance metrics - Inner Product'
SELECT vector_inner_product('[1,2]'::vector, '[3,4]'::vector) = -11.0 AS test_inner_product;

-- Test 1.12: Vector operations - get/set
\echo ''
\echo 'Test 1.12: Vector operations - get/set'
SELECT vector_get('[1,2,3]'::vector, 0) = 1.0 AS test_get_0;
SELECT vector_get('[1,2,3]'::vector, 1) = 2.0 AS test_get_1;
SELECT vector_set('[1,2,3]'::vector, 1, 99.0) = '[1,99,3]'::vector AS test_set;

-- Test 1.13: Vector operations - slice
\echo ''
\echo 'Test 1.13: Vector operations - slice'
SELECT vector_slice('[1,2,3,4,5]'::vector, 1, 4) = '[2,3,4]'::vector AS test_slice;

-- Test 1.14: Vector operations - statistics
\echo ''
\echo 'Test 1.14: Vector operations - statistics'
SELECT abs(vector_mean('[1,2,3,4,5]'::vector) - 3.0) < 0.0001 AS test_mean;
SELECT abs(vector_sum('[1,2,3]'::vector) - 6.0) < 0.0001 AS test_sum;
SELECT vector_min('[5,2,8,1,9]'::vector) = 1.0 AS test_min;
SELECT vector_max('[5,2,8,1,9]'::vector) = 9.0 AS test_max;

-- Test 1.15: Vector comparison operators
\echo ''
\echo 'Test 1.15: Vector comparison operators'
SELECT '[1,2,3]'::vector = '[1,2,3]'::vector AS test_eq_true;
SELECT '[1,2,3]'::vector <> '[1,2,4]'::vector AS test_ne_true;
SELECT '[1,2,3]'::vector = '[1,2,4]'::vector AS test_eq_false;

-- ============================================================================
-- SECTION 2: ADVANCED TESTS - Edge Cases
-- ============================================================================

\echo ''
\echo '============================================================================'
\echo 'SECTION 2: ADVANCED TESTS - Edge Cases'
\echo '============================================================================'

-- Test 2.1: Large vectors
\echo ''
\echo 'Test 2.1: Large vectors'
SELECT vector_dims(('[' || string_agg(i::text, ',') || ']')::vector) = 100 AS test_large_vector
FROM (SELECT i FROM generate_series(1, 100) i) t;

-- Test 2.2: High-dimensional vectors
\echo ''
\echo 'Test 2.2: High-dimensional vectors'
SELECT vector_dims(('[' || string_agg('1.0', ',') || ']')::vector) = 1000 AS test_high_dim
FROM (SELECT 1 FROM generate_series(1, 1000)) t;

-- Test 2.3: Very small values
\echo ''
\echo 'Test 2.3: Very small values'
SELECT vector_norm('[0.000001,0.000001]'::vector) > 0.0 AS test_small_values;

-- Test 2.4: Very large values
\echo ''
\echo 'Test 2.4: Very large values'
SELECT vector_norm('[1000000,1000000]'::vector) > 0.0 AS test_large_values;

-- Test 2.5: Normalized vectors maintain unit length
\echo ''
\echo 'Test 2.5: Normalized vectors maintain unit length'
SELECT abs(vector_norm(vector_normalize('[100,200,300]'::vector)) - 1.0) < 0.0001 AS test_normalize_large;

-- Test 2.6: Distance metrics - L1 (Manhattan)
\echo ''
\echo 'Test 2.6: Distance metrics - L1 (Manhattan)'
SELECT abs(vector_l1_distance('[0,0]'::vector, '[3,4]'::vector) - 7.0) < 0.0001 AS test_l1;

-- Test 2.7: Distance metrics - Hamming
\echo ''
\echo 'Test 2.7: Distance metrics - Hamming'
SELECT vector_hamming_distance('[1,2,3]'::vector, '[1,2,4]'::vector) = 1 AS test_hamming;

-- Test 2.8: Distance metrics - Chebyshev
\echo ''
\echo 'Test 2.8: Distance metrics - Chebyshev'
SELECT abs(vector_chebyshev_distance('[0,0]'::vector, '[3,4]'::vector) - 4.0) < 0.0001 AS test_chebyshev;

-- Test 2.9: Distance metrics - Minkowski
\echo ''
\echo 'Test 2.9: Distance metrics - Minkowski'
SELECT abs(vector_minkowski_distance('[0,0]'::vector, '[3,4]'::vector, 1.0) - 7.0) < 0.0001 AS test_minkowski_l1;
SELECT abs(vector_minkowski_distance('[0,0]'::vector, '[3,4]'::vector, 2.0) - 5.0) < 0.0001 AS test_minkowski_l2;

-- Test 2.10: Distance metrics - Squared L2
\echo ''
\echo 'Test 2.10: Distance metrics - Squared L2'
SELECT abs(vector_squared_l2_distance('[0,0]'::vector, '[3,4]'::vector) - 25.0) < 0.0001 AS test_squared_l2;

-- Test 2.11: Vector operations - clip
\echo ''
\echo 'Test 2.11: Vector operations - clip'
SELECT vector_clip('[1,5,10]'::vector, 2.0, 8.0) = '[2,5,8]'::vector AS test_clip;

-- Test 2.12: Vector operations - standardize
\echo ''
\echo 'Test 2.12: Vector operations - standardize'
SELECT abs(vector_mean(vector_standardize('[1,2,3,4,5]'::vector))) < 0.0001 AS test_standardize_mean;
SELECT abs(vector_stddev(vector_standardize('[1,2,3,4,5]'::vector)) - 1.0) < 0.1 AS test_standardize_stddev;

-- Test 2.13: Vector operations - minmax normalize
\echo ''
\echo 'Test 2.13: Vector operations - minmax normalize'
SELECT vector_min(vector_minmax_normalize('[1,2,3,4,5]'::vector)) = 0.0 AS test_minmax_min;
SELECT vector_max(vector_minmax_normalize('[1,2,3,4,5]'::vector)) = 1.0 AS test_minmax_max;

-- Test 2.14: Vector hash function
\echo ''
\echo 'Test 2.14: Vector hash function'
SELECT vector_hash('[1,2,3]'::vector) = vector_hash('[1,2,3]'::vector) AS test_hash_consistent;
SELECT vector_hash('[1,2,3]'::vector) <> vector_hash('[1,2,4]'::vector) AS test_hash_different;

-- Test 2.15: Quantization - INT8
\echo ''
\echo 'Test 2.15: Quantization - INT8'
SELECT vector_dims(vector_to_int8('[1.0,2.0,3.0]'::vector)) = 3 AS test_quantize_int8;
SELECT vector_dims(int8_to_vector(vector_to_int8('[1.0,2.0,3.0]'::vector))) = 3 AS test_dequantize_int8;

-- Test 2.16: Quantization - FP16
\echo ''
\echo 'Test 2.16: Quantization - FP16'
SELECT vector_dims(vector_to_float16('[1.0,2.0,3.0]'::vector)) = 3 AS test_quantize_fp16;
SELECT vector_dims(float16_to_vector(vector_to_float16('[1.0,2.0,3.0]'::vector))) = 3 AS test_dequantize_fp16;

-- Test 2.17: Quantization - Binary
\echo ''
\echo 'Test 2.17: Quantization - Binary'
SELECT vector_dims(binary_to_vector(vector_to_binary('[1.0,-1.0,1.0]'::vector))) = 3 AS test_binary_roundtrip;

-- Test 2.18: halfvec type
\echo ''
\echo 'Test 2.18: halfvec type'
SELECT '[1,2,3]'::halfvec IS NOT NULL AS test_halfvec_create;
SELECT halfvec_l2_distance('[1,2,3]'::halfvec, '[1,2,3]'::halfvec) = 0.0 AS test_halfvec_l2;

-- Test 2.19: sparsevec type
\echo ''
\echo 'Test 2.19: sparsevec type'
SELECT '{1:1.0,3:2.0,5:3.0}'::sparsevec IS NOT NULL AS test_sparsevec_create;
SELECT sparsevec_l2_distance('{1:1.0,3:2.0}'::sparsevec, '{1:1.0,3:2.0}'::sparsevec) = 0.0 AS test_sparsevec_l2;

-- ============================================================================
-- SECTION 3: NEGATIVE TESTS - Error Conditions
-- ============================================================================

\echo ''
\echo '============================================================================'
\echo 'SECTION 3: NEGATIVE TESTS - Error Conditions'
\echo '============================================================================'

-- Test 3.1: Dimension mismatch in operations
\echo ''
\echo 'Test 3.1: Dimension mismatch in operations'
\set VERBOSITY verbose
DO $$
BEGIN
    PERFORM '[1,2]'::vector + '[1,2,3]'::vector;
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%dimension%' THEN
            RAISE NOTICE 'PASS: Dimension mismatch caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.2: Invalid vector index (negative)
\echo ''
\echo 'Test 3.2: Invalid vector index (negative)'
DO $$
BEGIN
    PERFORM vector_get('[1,2,3]'::vector, -1);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%out of bounds%' THEN
            RAISE NOTICE 'PASS: Negative index caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.3: Invalid vector index (too large)
\echo ''
\echo 'Test 3.3: Invalid vector index (too large)'
DO $$
BEGIN
    PERFORM vector_get('[1,2,3]'::vector, 10);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%out of bounds%' THEN
            RAISE NOTICE 'PASS: Index out of bounds caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.4: Invalid slice bounds
\echo ''
\echo 'Test 3.4: Invalid slice bounds'
DO $$
BEGIN
    PERFORM vector_slice('[1,2,3]'::vector, 5, 10);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%invalid slice%' THEN
            RAISE NOTICE 'PASS: Invalid slice bounds caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.5: Division by zero in vector operations
\echo ''
\echo 'Test 3.5: Division by zero in vector operations'
DO $$
BEGIN
    PERFORM vector_divide('[1,2,3]'::vector, '[1,0,1]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%division by zero%' THEN
            RAISE NOTICE 'PASS: Division by zero caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.6: Square root of negative values
\echo ''
\echo 'Test 3.6: Square root of negative values'
DO $$
BEGIN
    PERFORM vector_sqrt('[-1,2,3]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%square root of negative%' THEN
            RAISE NOTICE 'PASS: Negative square root caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.7: Invalid Minkowski p parameter
\echo ''
\echo 'Test 3.7: Invalid Minkowski p parameter'
DO $$
BEGIN
    PERFORM vector_minkowski_distance('[1,2]'::vector, '[3,4]'::vector, -1.0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%must be positive%' THEN
            RAISE NOTICE 'PASS: Invalid p parameter caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.8: Invalid clip bounds (min > max)
\echo ''
\echo 'Test 3.8: Invalid clip bounds (min > max)'
DO $$
BEGIN
    PERFORM vector_clip('[1,2,3]'::vector, 10.0, 5.0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%min_val must be <= max_val%' THEN
            RAISE NOTICE 'PASS: Invalid clip bounds caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.9: NULL vector operations
\echo ''
\echo 'Test 3.9: NULL vector operations'
DO $$
BEGIN
    PERFORM vector_norm(NULL::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%NULL%' THEN
            RAISE NOTICE 'PASS: NULL vector caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.10: Empty vector (zero dimensions)
\echo ''
\echo 'Test 3.10: Empty vector (zero dimensions)'
DO $$
BEGIN
    PERFORM vector_mean('[]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%dimension%' OR SQLERRM LIKE '%empty%' THEN
            RAISE NOTICE 'PASS: Empty vector caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.11: Invalid argument count - too few
\echo ''
\echo 'Test 3.11: Invalid argument count - too few'
DO $$
BEGIN
    PERFORM vector_add('[1,2]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' THEN
            RAISE NOTICE 'PASS: Too few arguments caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.12: Invalid argument count - too many
\echo ''
\echo 'Test 3.12: Invalid argument count - too many'
DO $$
BEGIN
    PERFORM vector_add('[1,2]'::vector, '[3,4]'::vector, '[5,6]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' THEN
            RAISE NOTICE 'PASS: Too many arguments caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.13: Invalid vector format
\echo ''
\echo 'Test 3.13: Invalid vector format'
DO $$
BEGIN
    PERFORM 'invalid'::vector;
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%invalid%' THEN
            RAISE NOTICE 'PASS: Invalid format caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.14: Overflow in concatenation
\echo ''
\echo 'Test 3.14: Overflow in concatenation'
DO $$
DECLARE
    v1 vector;
    v2 vector;
BEGIN
    -- Create vectors that would exceed max dimension
    SELECT '[' || string_agg('1.0', ',') || ']' INTO v1 FROM generate_series(1, 8000);
    SELECT '[' || string_agg('1.0', ',') || ']' INTO v2 FROM generate_series(1, 8000);
    v1 := v1::vector;
    v2 := v2::vector;
    PERFORM vector_concat(v1, v2);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%exceed maximum%' THEN
            RAISE NOTICE 'PASS: Overflow caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 3.15: NaN/Infinity handling
\echo ''
\echo 'Test 3.15: NaN/Infinity handling'
DO $$
BEGIN
    PERFORM vector_norm('[NaN,1,2]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%NaN%' OR SQLERRM LIKE '%Infinity%' THEN
            RAISE NOTICE 'PASS: NaN/Infinity caught';
        ELSE
            RAISE;
        END IF;
END $$;

\set VERBOSITY default

-- ============================================================================
-- SECTION 4: PERFORMANCE TESTS
-- ============================================================================

\echo ''
\echo '============================================================================'
\echo 'SECTION 4: PERFORMANCE TESTS'
\echo '============================================================================'

-- Test 4.1: Batch distance calculations
\echo ''
\echo 'Test 4.1: Batch distance calculations'
\timing on
SELECT COUNT(*) FROM (
    SELECT vector_l2_distance('[1,2,3]'::vector, '[4,5,6]'::vector)
    FROM generate_series(1, 1000)
) t;
\timing off

-- Test 4.2: Large vector operations
\echo ''
\echo 'Test 4.2: Large vector operations'
\timing on
SELECT vector_norm(('[' || string_agg(i::text, ',') || ']')::vector)
FROM (SELECT i FROM generate_series(1, 1000) i) t;
\timing off

-- ============================================================================
-- SUMMARY
-- ============================================================================

\echo ''
\echo '============================================================================'
\echo 'TEST SUITE COMPLETE'
\echo '============================================================================'
\echo ''
\echo 'Summary:'
\echo '  ✓ Basic tests: 15 test groups'
\echo '  ✓ Advanced tests: 19 test groups'
\echo '  ✓ Negative tests: 15 test groups'
\echo '  ✓ Performance tests: 2 test groups'
\echo ''
\echo 'Total: 51 comprehensive test groups covering all vector operations'
\echo '============================================================================'

