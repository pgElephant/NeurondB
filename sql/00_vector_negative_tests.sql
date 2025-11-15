/*-------------------------------------------------------------------------
 *
 * 00_vector_negative_tests.sql
 *    Negative tests - error conditions and invalid inputs for all vector functions
 *
 * Tests include:
 *   - Dimension mismatches
 *   - Invalid indices and bounds
 *   - Division by zero
 *   - Invalid parameters
 *   - NULL/empty vector handling
 *   - CHECK_NARGS validation
 *   - Overflow conditions
 *   - NaN/Infinity handling
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 *-------------------------------------------------------------------------
 */

\set ON_ERROR_STOP on

\echo '============================================================================'
\echo 'NEGATIVE TESTS - Error Conditions and Invalid Inputs'
\echo '============================================================================'

-- Test 2.1: Dimension mismatch in addition
\echo ''
\echo 'Test 2.1: Dimension mismatch in addition'
\set VERBOSITY verbose
DO $$
BEGIN
    PERFORM '[1,2]'::vector + '[1,2,3]'::vector;
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%dimension%' THEN
            RAISE NOTICE 'PASS: Dimension mismatch in addition caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.2: Dimension mismatch in subtraction
\echo ''
\echo 'Test 2.2: Dimension mismatch in subtraction'
DO $$
BEGIN
    PERFORM '[1,2]'::vector - '[1,2,3]'::vector;
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%dimension%' THEN
            RAISE NOTICE 'PASS: Dimension mismatch in subtraction caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.3: Dimension mismatch in distance calculations
\echo ''
\echo 'Test 2.3: Dimension mismatch in distance calculations'
DO $$
BEGIN
    PERFORM vector_l2_distance('[1,2]'::vector, '[1,2,3]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%dimension%' THEN
            RAISE NOTICE 'PASS: Dimension mismatch in distance caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.4: Invalid vector index - negative
\echo ''
\echo 'Test 2.4: Invalid vector index - negative'
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

-- Test 2.5: Invalid vector index - too large
\echo ''
\echo 'Test 2.5: Invalid vector index - too large'
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

-- Test 2.6: Invalid slice bounds - start > end
\echo ''
\echo 'Test 2.6: Invalid slice bounds - start > end'
DO $$
BEGIN
    PERFORM vector_slice('[1,2,3]'::vector, 2, 1);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%invalid slice%' THEN
            RAISE NOTICE 'PASS: Invalid slice bounds caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.7: Invalid slice bounds - start out of range
\echo ''
\echo 'Test 2.7: Invalid slice bounds - start out of range'
DO $$
BEGIN
    PERFORM vector_slice('[1,2,3]'::vector, 10, 15);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%invalid slice%' OR SQLERRM LIKE '%out of bounds%' THEN
            RAISE NOTICE 'PASS: Slice start out of range caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.8: Division by zero in vector operations
\echo ''
\echo 'Test 2.8: Division by zero in vector operations'
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

-- Test 2.9: Square root of negative values
\echo ''
\echo 'Test 2.9: Square root of negative values'
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

-- Test 2.10: Invalid Minkowski p parameter - negative
\echo ''
\echo 'Test 2.10: Invalid Minkowski p parameter - negative'
DO $$
BEGIN
    PERFORM vector_minkowski_distance('[1,2]'::vector, '[3,4]'::vector, -1.0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%must be positive%' THEN
            RAISE NOTICE 'PASS: Invalid p parameter (negative) caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.11: Invalid Minkowski p parameter - zero
\echo ''
\echo 'Test 2.11: Invalid Minkowski p parameter - zero'
DO $$
BEGIN
    PERFORM vector_minkowski_distance('[1,2]'::vector, '[3,4]'::vector, 0.0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%must be positive%' THEN
            RAISE NOTICE 'PASS: Invalid p parameter (zero) caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.12: Invalid clip bounds - min > max
\echo ''
\echo 'Test 2.12: Invalid clip bounds - min > max'
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

-- Test 2.13: NULL vector operations - norm
\echo ''
\echo 'Test 2.13: NULL vector operations - norm'
DO $$
BEGIN
    PERFORM vector_norm(NULL::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%NULL%' THEN
            RAISE NOTICE 'PASS: NULL vector in norm caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.14: NULL vector operations - addition
\echo ''
\echo 'Test 2.14: NULL vector operations - addition'
DO $$
BEGIN
    PERFORM vector_add(NULL::vector, '[1,2]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%NULL%' THEN
            RAISE NOTICE 'PASS: NULL vector in addition caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.15: Empty vector (zero dimensions)
\echo ''
\echo 'Test 2.15: Empty vector (zero dimensions)'
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

-- Test 2.16: Invalid argument count - too few arguments
\echo ''
\echo 'Test 2.16: Invalid argument count - too few arguments'
DO $$
BEGIN
    PERFORM vector_add('[1,2]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' OR SQLERRM LIKE '%requires 2%' THEN
            RAISE NOTICE 'PASS: Too few arguments caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.17: Invalid argument count - too many arguments
\echo ''
\echo 'Test 2.17: Invalid argument count - too many arguments'
DO $$
BEGIN
    PERFORM vector_add('[1,2]'::vector, '[3,4]'::vector, '[5,6]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' OR SQLERRM LIKE '%requires 2%' THEN
            RAISE NOTICE 'PASS: Too many arguments caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.18: Invalid vector format - malformed string
\echo ''
\echo 'Test 2.18: Invalid vector format - malformed string'
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

-- Test 2.19: Invalid vector format - missing brackets
\echo ''
\echo 'Test 2.19: Invalid vector format - missing brackets'
DO $$
BEGIN
    PERFORM '1,2,3'::vector;
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%invalid%' THEN
            RAISE NOTICE 'PASS: Missing brackets caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.20: Overflow in concatenation
\echo ''
\echo 'Test 2.20: Overflow in concatenation'
DO $$
DECLARE
    v1 text;
    v2 text;
BEGIN
    -- Create vectors that would exceed max dimension
    SELECT '[' || string_agg('1.0', ',') || ']' INTO v1 FROM generate_series(1, 8000);
    SELECT '[' || string_agg('1.0', ',') || ']' INTO v2 FROM generate_series(1, 8000);
    PERFORM vector_concat(v1::vector, v2::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%exceed maximum%' OR SQLERRM LIKE '%maximum%' THEN
            RAISE NOTICE 'PASS: Overflow caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.21: NaN handling in vector norm
\echo ''
\echo 'Test 2.21: NaN handling in vector norm'
DO $$
BEGIN
    PERFORM vector_norm('[NaN,1,2]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%NaN%' OR SQLERRM LIKE '%Infinity%' THEN
            RAISE NOTICE 'PASS: NaN caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.22: Infinity handling in vector operations
\echo ''
\echo 'Test 2.22: Infinity handling in vector operations'
DO $$
BEGIN
    PERFORM vector_norm('[Infinity,1,2]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%Infinity%' OR SQLERRM LIKE '%NaN%' THEN
            RAISE NOTICE 'PASS: Infinity caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.23: Invalid Mahalanobis covariance matrix dimension
\echo ''
\echo 'Test 2.23: Invalid Mahalanobis covariance matrix dimension'
DO $$
BEGIN
    PERFORM vector_mahalanobis_distance('[1,2,3]'::vector, '[4,5,6]'::vector, '[1,2]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%dimension%' THEN
            RAISE NOTICE 'PASS: Covariance dimension mismatch caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.24: Invalid Mahalanobis covariance - non-positive values
\echo ''
\echo 'Test 2.24: Invalid Mahalanobis covariance - non-positive values'
DO $$
BEGIN
    PERFORM vector_mahalanobis_distance('[1,2]'::vector, '[3,4]'::vector, '[0,0]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%positive%' OR SQLERRM LIKE '%finite%' THEN
            RAISE NOTICE 'PASS: Non-positive covariance caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.25: Invalid vector_set index
\echo ''
\echo 'Test 2.25: Invalid vector_set index'
DO $$
BEGIN
    PERFORM vector_set('[1,2,3]'::vector, 10, 99.0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%out of bounds%' THEN
            RAISE NOTICE 'PASS: Invalid set index caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.26: Invalid vector_append on NULL
\echo ''
\echo 'Test 2.26: Invalid vector_append on NULL'
DO $$
BEGIN
    PERFORM vector_append(NULL::vector, 1.0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%NULL%' THEN
            RAISE NOTICE 'PASS: NULL vector in append caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.27: Invalid vector_prepend on NULL
\echo ''
\echo 'Test 2.27: Invalid vector_prepend on NULL'
DO $$
BEGIN
    PERFORM vector_prepend(1.0, NULL::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%NULL%' THEN
            RAISE NOTICE 'PASS: NULL vector in prepend caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.28: Invalid vector_concat with NULL
\echo ''
\echo 'Test 2.28: Invalid vector_concat with NULL'
DO $$
BEGIN
    PERFORM vector_concat(NULL::vector, '[1,2]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%NULL%' THEN
            RAISE NOTICE 'PASS: NULL vector in concat caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.29: Invalid vector_normalize on zero vector (should handle gracefully)
\echo ''
\echo 'Test 2.29: Invalid vector_normalize on zero vector'
-- This should not fail, but return zero vector
SELECT vector_normalize('[0,0,0]'::vector) = '[0,0,0]'::vector AS test_normalize_zero_handled;

-- Test 2.30: Invalid vector_standardize on constant vector
\echo ''
\echo 'Test 2.30: Invalid vector_standardize on constant vector'
-- This should handle gracefully (stddev = 0)
SELECT vector_standardize('[1,1,1,1,1]'::vector) = '[0,0,0,0,0]'::vector AS test_standardize_constant;

-- Test 2.31: Invalid vector_minmax_normalize on constant vector
\echo ''
\echo 'Test 2.31: Invalid vector_minmax_normalize on constant vector'
-- This should handle gracefully (range = 0)
SELECT vector_minmax_normalize('[5,5,5,5,5]'::vector) = '[0.5,0.5,0.5,0.5,0.5]'::vector AS test_minmax_constant;

-- Test 2.32: Invalid halfvec operations - dimension mismatch
\echo ''
\echo 'Test 2.32: Invalid halfvec operations - dimension mismatch'
DO $$
BEGIN
    PERFORM halfvec_l2_distance('[1,2]'::halfvec, '[1,2,3]'::halfvec);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%dimension%' THEN
            RAISE NOTICE 'PASS: halfvec dimension mismatch caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.33: Invalid sparsevec operations - dimension mismatch
\echo ''
\echo 'Test 2.33: Invalid sparsevec operations - dimension mismatch'
DO $$
BEGIN
    PERFORM sparsevec_l2_distance('{1:1.0,2:2.0}'::sparsevec, '{1:1.0,2:2.0,3:3.0}'::sparsevec);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%dimension%' THEN
            RAISE NOTICE 'PASS: sparsevec dimension mismatch caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.34: Invalid bit operations - length mismatch
\echo ''
\echo 'Test 2.34: Invalid bit operations - length mismatch'
DO $$
BEGIN
    PERFORM bit_hamming_distance('101'::bit, '1010'::bit);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%length%' OR SQLERRM LIKE '%dimension%' THEN
            RAISE NOTICE 'PASS: bit length mismatch caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.35: Invalid quantization - wrong argument count
\echo ''
\echo 'Test 2.35: Invalid quantization - wrong argument count'
DO $$
BEGIN
    PERFORM vector_to_int8('[1,2,3]'::vector, '[4,5,6]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' THEN
            RAISE NOTICE 'PASS: Wrong argument count in quantization caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.36: Invalid dequantization - wrong argument count
\echo ''
\echo 'Test 2.36: Invalid dequantization - wrong argument count'
DO $$
DECLARE
    v8 vector;
BEGIN
    v8 := vector_to_int8('[1,2,3]'::vector);
    PERFORM int8_to_vector(v8, '[4,5,6]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' THEN
            RAISE NOTICE 'PASS: Wrong argument count in dequantization caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.37: Invalid vector comparison with NULL
\echo ''
\echo 'Test 2.37: Invalid vector comparison with NULL'
SELECT (NULL::vector = '[1,2,3]'::vector) IS NULL AS test_eq_null_left;
SELECT ('[1,2,3]'::vector = NULL::vector) IS NULL AS test_eq_null_right;
SELECT (NULL::vector <> '[1,2,3]'::vector) IS NULL AS test_ne_null_left;

-- Test 2.38: Invalid vector_hash on NULL
\echo ''
\echo 'Test 2.38: Invalid vector_hash on NULL'
SELECT vector_hash(NULL::vector) = 0 AS test_hash_null;

-- Test 2.39: Invalid array_to_vector - multi-dimensional array
\echo ''
\echo 'Test 2.39: Invalid array_to_vector - multi-dimensional array'
DO $$
BEGIN
    PERFORM array_to_vector(ARRAY[[1,2],[3,4]]::real[]);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%one-dimensional%' THEN
            RAISE NOTICE 'PASS: Multi-dimensional array caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.40: Invalid array_to_vector - NULL elements
\echo ''
\echo 'Test 2.40: Invalid array_to_vector - NULL elements'
DO $$
BEGIN
    PERFORM array_to_vector(ARRAY[1.0, NULL, 3.0]::real[]);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%null%' THEN
            RAISE NOTICE 'PASS: NULL array elements caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.41: Invalid vector operations - wrong argument count for single-arg functions
\echo ''
\echo 'Test 2.41: Invalid vector operations - wrong argument count for single-arg functions'
DO $$
BEGIN
    PERFORM vector_norm('[1,2,3]'::vector, '[4,5,6]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' THEN
            RAISE NOTICE 'PASS: Too many arguments for single-arg function caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.42: Invalid vector operations - wrong argument count for three-arg functions
\echo ''
\echo 'Test 2.42: Invalid vector operations - wrong argument count for three-arg functions'
DO $$
BEGIN
    PERFORM vector_clip('[1,2,3]'::vector, 0.0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' THEN
            RAISE NOTICE 'PASS: Too few arguments for three-arg function caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.43: Invalid distance functions - wrong argument count
\echo ''
\echo 'Test 2.43: Invalid distance functions - wrong argument count'
DO $$
BEGIN
    PERFORM vector_l2_distance('[1,2]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' THEN
            RAISE NOTICE 'PASS: Too few arguments for distance function caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.44: Invalid Minkowski - wrong argument count
\echo ''
\echo 'Test 2.44: Invalid Minkowski - wrong argument count'
DO $$
BEGIN
    PERFORM vector_minkowski_distance('[1,2]'::vector, '[3,4]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' THEN
            RAISE NOTICE 'PASS: Too few arguments for Minkowski caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.45: Invalid halfvec operations - wrong argument count
\echo ''
\echo 'Test 2.45: Invalid halfvec operations - wrong argument count'
DO $$
BEGIN
    PERFORM halfvec_l2_distance('[1,2]'::halfvec);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' THEN
            RAISE NOTICE 'PASS: Too few arguments for halfvec distance caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.46: Invalid sparsevec operations - wrong argument count
\echo ''
\echo 'Test 2.46: Invalid sparsevec operations - wrong argument count'
DO $$
BEGIN
    PERFORM sparsevec_l2_distance('{1:1.0}'::sparsevec);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' THEN
            RAISE NOTICE 'PASS: Too few arguments for sparsevec distance caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.47: Invalid bit operations - wrong argument count
\echo ''
\echo 'Test 2.47: Invalid bit operations - wrong argument count'
DO $$
BEGIN
    PERFORM bit_hamming_distance('101'::bit);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%requires%argument%' THEN
            RAISE NOTICE 'PASS: Too few arguments for bit distance caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.48: Invalid vector operations - hadamard dimension mismatch
\echo ''
\echo 'Test 2.48: Invalid vector operations - hadamard dimension mismatch'
DO $$
BEGIN
    PERFORM vector_hadamard('[1,2]'::vector, '[1,2,3]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%dimension%' THEN
            RAISE NOTICE 'PASS: Hadamard dimension mismatch caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.49: Invalid vector operations - divide dimension mismatch
\echo ''
\echo 'Test 2.49: Invalid vector operations - divide dimension mismatch'
DO $$
BEGIN
    PERFORM vector_divide('[1,2]'::vector, '[1,2,3]'::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLERRM LIKE '%dimension%' THEN
            RAISE NOTICE 'PASS: Divide dimension mismatch caught';
        ELSE
            RAISE;
        END IF;
END $$;

-- Test 2.50: Invalid vector operations - cosine distance with zero norm
\echo ''
\echo 'Test 2.50: Invalid vector operations - cosine distance with zero norm'
-- This should handle gracefully (return 0.0 for zero vectors)
SELECT vector_cosine_distance('[0,0]'::vector, '[0,0]'::vector) = 0.0 AS test_cosine_zero_norm;

\set VERBOSITY default

-- ============================================================================
-- SUMMARY
-- ============================================================================

\echo ''
\echo '============================================================================'
\echo 'NEGATIVE TEST SUITE COMPLETE'
\echo '============================================================================'
\echo ''
\echo 'Summary:'
\echo '  ✓ Negative tests: 50 test groups'
\echo ''
\echo 'Coverage includes:'
\echo '  • Dimension mismatches in all operations'
\echo '  • Invalid indices (negative, out of bounds)'
\echo '  • Invalid slice bounds'
\echo '  • Division by zero'
\echo '  • Square root of negative values'
\echo '  • Invalid Minkowski p parameter'
\echo '  • Invalid clip bounds'
\echo '  • NULL vector handling'
\echo '  • Empty vector handling'
\echo '  • CHECK_NARGS validation (too few/many arguments)'
\echo '  • Invalid vector formats'
\echo '  • Overflow conditions'
\echo '  • NaN/Infinity handling'
\echo '  • Invalid covariance matrices'
\echo '  • Invalid array conversions'
\echo '  • Wrong argument counts for all function types'
\echo '============================================================================'

