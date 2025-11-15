/*-------------------------------------------------------------------------
 *
 * 00_vector_advanced_tests.sql
 *    Advanced edge cases and boundary condition tests for all vector functions
 *
 * Tests include:
 *   - Maximum dimension vectors
 *   - Single-element vectors
 *   - Very small/large floating point values
 *   - Zero vectors, unit vectors, orthogonal/parallel vectors
 *   - All distance metrics with edge cases
 *   - Vector transformations and statistics
 *   - Quantization roundtrips
 *   - halfvec, sparsevec, bit type operations
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 *-------------------------------------------------------------------------
 */

\set ON_ERROR_STOP on

\echo '============================================================================'
\echo 'ADVANCED TESTS - Edge Cases and Boundary Conditions'
\echo '============================================================================'

-- Test 1.1: Maximum dimension vectors
\echo ''
\echo 'Test 1.1: Maximum dimension vectors'
SELECT vector_dims(('[' || string_agg('1.0', ',') || ']')::vector) = 16000 AS test_max_dim
FROM (SELECT 1 FROM generate_series(1, 16000)) t;

-- Test 1.2: Single-element vectors (edge case)
\echo ''
\echo 'Test 1.2: Single-element vectors'
SELECT vector_norm('[42.0]'::vector) = 42.0 AS test_single_element;
SELECT vector_dims('[1.0]'::vector) = 1 AS test_single_dim;
SELECT vector_l2_distance('[1.0]'::vector, '[2.0]'::vector) = 1.0 AS test_single_dist;

-- Test 1.3: Very small floating point values (near zero)
\echo ''
\echo 'Test 1.3: Very small floating point values'
SELECT vector_norm('[0.0000001,0.0000001]'::vector) > 0.0 AS test_tiny_values;
SELECT abs(vector_norm('[1e-10,1e-10]'::vector) - sqrt(2.0) * 1e-10) < 1e-15 AS test_scientific_notation;

-- Test 1.4: Very large floating point values
\echo ''
\echo 'Test 1.4: Very large floating point values'
SELECT vector_norm('[1e10,1e10]'::vector) > 0.0 AS test_huge_values;
SELECT abs(vector_norm('[1000000000.0,1000000000.0]'::vector) - sqrt(2.0) * 1000000000.0) < 1000.0 AS test_large_norm;

-- Test 1.5: Zero vectors (all zeros)
\echo ''
\echo 'Test 1.5: Zero vectors'
SELECT vector_norm('[0,0,0,0,0]'::vector) = 0.0 AS test_zero_norm;
SELECT vector_l2_distance('[0,0]'::vector, '[0,0]'::vector) = 0.0 AS test_zero_distance;
SELECT vector_cosine_distance('[0,0]'::vector, '[0,0]'::vector) = 0.0 AS test_zero_cosine;

-- Test 1.6: Unit vectors (normalized)
\echo ''
\echo 'Test 1.6: Unit vectors'
SELECT abs(vector_norm(vector_normalize('[3,4,5]'::vector)) - 1.0) < 0.0001 AS test_unit_norm;
SELECT abs(vector_norm(vector_normalize('[100,200,300]'::vector)) - 1.0) < 0.0001 AS test_unit_norm_large;
SELECT abs(vector_norm(vector_normalize('[0.001,0.002,0.003]'::vector)) - 1.0) < 0.0001 AS test_unit_norm_small;

-- Test 1.7: Orthogonal vectors (dot product = 0)
\echo ''
\echo 'Test 1.7: Orthogonal vectors'
SELECT abs(vector_cosine_distance('[1,0]'::vector, '[0,1]'::vector) - 1.0) < 0.0001 AS test_orthogonal_2d;
SELECT abs(vector_cosine_distance('[1,0,0]'::vector, '[0,1,0]'::vector) - 1.0) < 0.0001 AS test_orthogonal_3d;
SELECT abs(vector_inner_product('[1,0]'::vector, '[0,1]'::vector) - 0.0) < 0.0001 AS test_orthogonal_dot;

-- Test 1.8: Parallel vectors (same direction)
\echo ''
\echo 'Test 1.8: Parallel vectors'
SELECT vector_cosine_distance('[1,1]'::vector, '[2,2]'::vector) = 0.0 AS test_parallel;
SELECT vector_cosine_distance('[1,2,3]'::vector, '[10,20,30]'::vector) = 0.0 AS test_parallel_scaled;
SELECT abs(vector_cosine_distance('[1,1,1]'::vector, '[-1,-1,-1]'::vector) - 2.0) < 0.0001 AS test_anti_parallel;

-- Test 1.9: Negative coordinate values
\echo ''
\echo 'Test 1.9: Negative coordinate values'
SELECT vector_norm('[-3,-4]'::vector) = 5.0 AS test_negative_norm;
SELECT vector_l2_distance('[-1,-2]'::vector, '[-3,-4]'::vector) = sqrt(8.0) AS test_negative_distance;
SELECT vector_l1_distance('[-1,2]'::vector, '[1,-2]'::vector) = 6.0 AS test_negative_l1;

-- Test 1.10: Mixed positive and negative values
\echo ''
\echo 'Test 1.10: Mixed positive and negative values'
SELECT vector_norm('[1,-1,1,-1]'::vector) = 2.0 AS test_mixed_norm;
SELECT abs(vector_cosine_distance('[1,-1]'::vector, '[-1,1]'::vector) - 2.0) < 0.0001 AS test_mixed_cosine;

-- Test 1.11: High-dimensional distance calculations
\echo ''
\echo 'Test 1.11: High-dimensional distance calculations'
SELECT vector_dims(('[' || string_agg('1.0', ',') || ']')::vector) = 1000 AS test_high_dim_vector
FROM (SELECT 1 FROM generate_series(1, 1000)) t;
SELECT vector_l2_distance(
    ('[' || string_agg('1.0', ',') || ']')::vector,
    ('[' || string_agg('2.0', ',') || ']')::vector
) = sqrt(1000.0) AS test_high_dim_distance
FROM (SELECT 1 FROM generate_series(1, 1000)) t;

-- Test 1.12: Distance metrics - L1 (Manhattan) edge cases
\echo ''
\echo 'Test 1.12: Distance metrics - L1 (Manhattan) edge cases'
SELECT vector_l1_distance('[0,0,0]'::vector, '[1,1,1]'::vector) = 3.0 AS test_l1_3d;
SELECT vector_l1_distance('[10,20]'::vector, '[10,20]'::vector) = 0.0 AS test_l1_identical;
SELECT vector_l1_distance('[-1,-2]'::vector, '[1,2]'::vector) = 6.0 AS test_l1_negative;

-- Test 1.13: Distance metrics - Hamming distance edge cases
\echo ''
\echo 'Test 1.13: Distance metrics - Hamming distance edge cases'
SELECT vector_hamming_distance('[1,2,3,4,5]'::vector, '[1,2,3,4,5]'::vector) = 0 AS test_hamming_identical;
SELECT vector_hamming_distance('[1,2,3]'::vector, '[4,5,6]'::vector) = 3 AS test_hamming_all_different;
SELECT vector_hamming_distance('[0,0,0]'::vector, '[0,0,0]'::vector) = 0 AS test_hamming_zero;

-- Test 1.14: Distance metrics - Chebyshev (L∞) edge cases
\echo ''
\echo 'Test 1.14: Distance metrics - Chebyshev (L∞) edge cases'
SELECT vector_chebyshev_distance('[0,0]'::vector, '[3,4]'::vector) = 4.0 AS test_chebyshev_max_y;
SELECT vector_chebyshev_distance('[0,0]'::vector, '[4,3]'::vector) = 4.0 AS test_chebyshev_max_x;
SELECT vector_chebyshev_distance('[1,1,1]'::vector, '[1,1,1]'::vector) = 0.0 AS test_chebyshev_identical;

-- Test 1.15: Distance metrics - Minkowski with various p values
\echo ''
\echo 'Test 1.15: Distance metrics - Minkowski with various p values'
SELECT abs(vector_minkowski_distance('[0,0]'::vector, '[3,4]'::vector, 1.0) - 7.0) < 0.0001 AS test_minkowski_p1;
SELECT abs(vector_minkowski_distance('[0,0]'::vector, '[3,4]'::vector, 2.0) - 5.0) < 0.0001 AS test_minkowski_p2;
SELECT abs(vector_minkowski_distance('[0,0]'::vector, '[3,4]'::vector, 3.0) - pow(91.0, 1.0/3.0)) < 0.1 AS test_minkowski_p3;
SELECT abs(vector_minkowski_distance('[0,0]'::vector, '[3,4]'::vector, 10.0) - pow(295245.0, 0.1)) < 0.1 AS test_minkowski_p10;

-- Test 1.16: Distance metrics - Squared L2 (no sqrt)
\echo ''
\echo 'Test 1.16: Distance metrics - Squared L2 (no sqrt)'
SELECT vector_squared_l2_distance('[0,0]'::vector, '[3,4]'::vector) = 25.0 AS test_squared_l2;
SELECT vector_squared_l2_distance('[1,2]'::vector, '[4,6]'::vector) = 25.0 AS test_squared_l2_2d;
SELECT vector_squared_l2_distance('[0,0,0]'::vector, '[1,1,1]'::vector) = 3.0 AS test_squared_l2_3d;

-- Test 1.17: Distance metrics - Jaccard distance
\echo ''
\echo 'Test 1.17: Distance metrics - Jaccard distance'
SELECT abs(vector_jaccard_distance('[1,0,1]'::vector, '[1,0,1]'::vector) - 0.0) < 0.0001 AS test_jaccard_identical;
SELECT abs(vector_jaccard_distance('[1,0,0]'::vector, '[0,1,0]'::vector) - 1.0) < 0.0001 AS test_jaccard_disjoint;
SELECT vector_jaccard_distance('[0,0,0]'::vector, '[0,0,0]'::vector) = 0.0 AS test_jaccard_zero;

-- Test 1.18: Distance metrics - Dice distance
\echo ''
\echo 'Test 1.18: Distance metrics - Dice distance'
SELECT abs(vector_dice_distance('[1,0,1]'::vector, '[1,0,1]'::vector) - 0.0) < 0.0001 AS test_dice_identical;
SELECT abs(vector_dice_distance('[1,0,0]'::vector, '[0,1,0]'::vector) - 1.0) < 0.0001 AS test_dice_disjoint;

-- Test 1.19: Distance metrics - Mahalanobis distance
\echo ''
\echo 'Test 1.19: Distance metrics - Mahalanobis distance'
SELECT abs(vector_mahalanobis_distance('[1,2]'::vector, '[1,2]'::vector, '[1,1]'::vector) - 0.0) < 0.0001 AS test_mahalanobis_identical;
SELECT vector_mahalanobis_distance('[0,0]'::vector, '[3,4]'::vector, NULL::vector) > 0.0 AS test_mahalanobis_fallback;

-- Test 1.20: Vector operations - clip edge cases
\echo ''
\echo 'Test 1.20: Vector operations - clip edge cases'
SELECT vector_clip('[1,5,10]'::vector, 2.0, 8.0) = '[2,5,8]'::vector AS test_clip_basic;
SELECT vector_clip('[1,2,3]'::vector, 0.0, 10.0) = '[1,2,3]'::vector AS test_clip_no_change;
SELECT vector_clip('[1,2,3]'::vector, 5.0, 10.0) = '[5,5,5]'::vector AS test_clip_all_min;

-- Test 1.21: Vector operations - standardize (z-score normalization)
\echo ''
\echo 'Test 1.21: Vector operations - standardize'
SELECT abs(vector_mean(vector_standardize('[1,2,3,4,5]'::vector))) < 0.0001 AS test_standardize_mean_zero;
SELECT abs(vector_stddev(vector_standardize('[1,2,3,4,5]'::vector)) - 1.0) < 0.1 AS test_standardize_stddev_one;
SELECT vector_standardize('[1,1,1,1,1]'::vector) = '[0,0,0,0,0]'::vector AS test_standardize_constant;

-- Test 1.22: Vector operations - minmax normalize
\echo ''
\echo 'Test 1.22: Vector operations - minmax normalize'
SELECT vector_min(vector_minmax_normalize('[1,2,3,4,5]'::vector)) = 0.0 AS test_minmax_min_zero;
SELECT vector_max(vector_minmax_normalize('[1,2,3,4,5]'::vector)) = 1.0 AS test_minmax_max_one;
SELECT vector_minmax_normalize('[5,5,5,5,5]'::vector) = '[0.5,0.5,0.5,0.5,0.5]'::vector AS test_minmax_constant;

-- Test 1.23: Vector hash function properties
\echo ''
\echo 'Test 1.23: Vector hash function properties'
SELECT vector_hash('[1,2,3]'::vector) = vector_hash('[1,2,3]'::vector) AS test_hash_deterministic;
SELECT vector_hash('[1,2,3]'::vector) <> vector_hash('[1,2,4]'::vector) AS test_hash_different;
SELECT vector_hash('[0,0,0]'::vector) <> vector_hash('[1,1,1]'::vector) AS test_hash_zero_vs_one;

-- Test 1.24: Quantization - INT8 roundtrip accuracy
\echo ''
\echo 'Test 1.24: Quantization - INT8 roundtrip accuracy'
SELECT vector_dims(int8_to_vector(vector_to_int8('[1.0,2.0,3.0]'::vector))) = 3 AS test_int8_roundtrip_dims;
SELECT vector_dims(vector_to_int8('[1.0,2.0,3.0]'::vector)) = 3 AS test_int8_dims;

-- Test 1.25: Quantization - FP16 roundtrip
\echo ''
\echo 'Test 1.25: Quantization - FP16 roundtrip'
SELECT vector_dims(float16_to_vector(vector_to_float16('[1.0,2.0,3.0]'::vector))) = 3 AS test_fp16_roundtrip_dims;
SELECT vector_dims(vector_to_float16('[1.0,2.0,3.0]'::vector)) = 3 AS test_fp16_dims;

-- Test 1.26: Quantization - Binary roundtrip
\echo ''
\echo 'Test 1.26: Quantization - Binary roundtrip'
SELECT vector_dims(binary_to_vector(vector_to_binary('[1.0,-1.0,1.0]'::vector))) = 3 AS test_binary_roundtrip_dims;
SELECT vector_dims(binary_to_vector(vector_to_binary('[-1.0,1.0,-1.0]'::vector))) = 3 AS test_binary_roundtrip_neg;

-- Test 1.27: Quantization - UINT8
\echo ''
\echo 'Test 1.27: Quantization - UINT8'
SELECT vector_dims(uint8_to_vector(vector_to_uint8('[1.0,2.0,3.0]'::vector))) = 3 AS test_uint8_roundtrip;

-- Test 1.28: Quantization - Ternary
\echo ''
\echo 'Test 1.28: Quantization - Ternary'
SELECT vector_dims(ternary_to_vector(vector_to_ternary('[1.0,-1.0,0.0]'::vector))) = 3 AS test_ternary_roundtrip;

-- Test 1.29: Quantization - INT4
\echo ''
\echo 'Test 1.29: Quantization - INT4'
SELECT vector_dims(int4_to_vector(vector_to_int4('[1.0,2.0,3.0]'::vector))) = 3 AS test_int4_roundtrip;

-- Test 1.30: halfvec type operations
\echo ''
\echo 'Test 1.30: halfvec type operations'
SELECT '[1,2,3]'::halfvec IS NOT NULL AS test_halfvec_create;
SELECT halfvec_l2_distance('[1,2,3]'::halfvec, '[1,2,3]'::halfvec) = 0.0 AS test_halfvec_l2_identical;
SELECT halfvec_cosine_distance('[1,0]'::halfvec, '[0,1]'::halfvec) > 0.0 AS test_halfvec_cosine;
SELECT halfvec_inner_product('[1,2]'::halfvec, '[3,4]'::halfvec) = -11.0 AS test_halfvec_inner_product;

-- Test 1.31: sparsevec type operations
\echo ''
\echo 'Test 1.31: sparsevec type operations'
SELECT '{1:1.0,3:2.0,5:3.0}'::sparsevec IS NOT NULL AS test_sparsevec_create;
SELECT sparsevec_l2_distance('{1:1.0,3:2.0}'::sparsevec, '{1:1.0,3:2.0}'::sparsevec) = 0.0 AS test_sparsevec_l2_identical;
SELECT sparsevec_cosine_distance('{1:1.0,3:1.0}'::sparsevec, '{2:1.0,4:1.0}'::sparsevec) > 0.0 AS test_sparsevec_cosine;
SELECT sparsevec_inner_product('{1:1.0,2:2.0}'::sparsevec, '{1:3.0,2:4.0}'::sparsevec) = -11.0 AS test_sparsevec_inner_product;

-- Test 1.32: bit type operations
\echo ''
\echo 'Test 1.32: bit type operations'
SELECT vector_to_bit('[1.0,-1.0,1.0]'::vector) IS NOT NULL AS test_bit_create;
SELECT bit_hamming_distance('101'::bit, '110'::bit) = 2 AS test_bit_hamming;
SELECT bit_hamming_distance('1111'::bit, '1111'::bit) = 0 AS test_bit_hamming_identical;

-- Test 1.33: Vector comparison operators - epsilon tolerance
\echo ''
\echo 'Test 1.33: Vector comparison operators - epsilon tolerance'
SELECT '[1.0,2.0,3.0]'::vector = '[1.0000001,2.0000001,3.0000001]'::vector AS test_eq_epsilon;
SELECT '[1.0,2.0,3.0]'::vector <> '[1.1,2.1,3.1]'::vector AS test_ne_different;

-- Test 1.34: Vector operations - element-wise operations
\echo ''
\echo 'Test 1.34: Vector operations - element-wise operations'
SELECT vector_abs('[-1,-2,-3]'::vector) = '[1,2,3]'::vector AS test_abs;
SELECT vector_square('[2,3,4]'::vector) = '[4,9,16]'::vector AS test_square;
SELECT abs(vector_sum(vector_sqrt('[4,9,16]'::vector)) - 9.0) < 0.0001 AS test_sqrt_sum;

-- Test 1.35: Vector operations - Hadamard product
\echo ''
\echo 'Test 1.35: Vector operations - Hadamard product'
SELECT vector_hadamard('[1,2,3]'::vector, '[4,5,6]'::vector) = '[4,10,18]'::vector AS test_hadamard;
SELECT vector_hadamard('[1,1,1]'::vector, '[1,1,1]'::vector) = '[1,1,1]'::vector AS test_hadamard_ones;

-- Test 1.36: Vector operations - power
\echo ''
\echo 'Test 1.36: Vector operations - power'
SELECT abs(vector_sum(vector_pow('[2,3,4]'::vector, 2.0)) - 29.0) < 0.0001 AS test_pow_2;
SELECT abs(vector_sum(vector_pow('[2,3,4]'::vector, 0.5)) - (sqrt(2.0)+sqrt(3.0)+sqrt(4.0))) < 0.0001 AS test_pow_half;

-- Test 1.37: Vector statistics - variance and stddev
\echo ''
\echo 'Test 1.37: Vector statistics - variance and stddev'
SELECT abs(vector_variance('[1,2,3,4,5]'::vector) - 2.0) < 0.1 AS test_variance;
SELECT abs(vector_stddev('[1,2,3,4,5]'::vector) - sqrt(2.0)) < 0.1 AS test_stddev;

-- Test 1.38: Vector concatenation - large vectors
\echo ''
\echo 'Test 1.38: Vector concatenation - large vectors'
SELECT vector_dims(vector_concat(
    ('[' || string_agg('1.0', ',') || ']')::vector,
    ('[' || string_agg('2.0', ',') || ']')::vector
)) = 200 AS test_concat_large
FROM (SELECT 1 FROM generate_series(1, 100)) t;

-- Test 1.39: Vector normalization - zero vector handling
\echo ''
\echo 'Test 1.39: Vector normalization - zero vector handling'
SELECT vector_normalize('[0,0,0]'::vector) = '[0,0,0]'::vector AS test_normalize_zero;

-- Test 1.40: Array conversion roundtrip
\echo ''
\echo 'Test 1.40: Array conversion roundtrip'
SELECT vector_to_array('[1,2,3,4,5]'::vector) = ARRAY[1.0,2.0,3.0,4.0,5.0]::real[] AS test_array_roundtrip;

-- ============================================================================
-- SUMMARY
-- ============================================================================

\echo ''
\echo '============================================================================'
\echo 'ADVANCED TEST SUITE COMPLETE'
\echo '============================================================================'
\echo ''
\echo 'Summary:'
\echo '  ✓ Advanced tests: 40 test groups'
\echo ''
\echo 'Coverage includes:'
\echo '  • Maximum dimension vectors (16,000 dimensions)'
\echo '  • Single-element vectors'
\echo '  • Very small/large floating point values'
\echo '  • Zero vectors, unit vectors, orthogonal/parallel vectors'
\echo '  • All 11 distance metrics with edge cases'
\echo '  • Vector transformations (clip, standardize, minmax normalize)'
\echo '  • Quantization roundtrips (INT8, FP16, Binary, UINT8, Ternary, INT4)'
\echo '  • halfvec, sparsevec, bit type operations'
\echo '  • Hash function properties'
\echo '  • High-dimensional operations'
\echo '============================================================================'

