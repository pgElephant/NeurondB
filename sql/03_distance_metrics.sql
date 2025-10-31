-- Test distance functions
SELECT vector_l2_distance('[0.0, 0.0]'::vector, '[3.0, 4.0]'::vector) > 0 AS l2_positive;
SELECT vector_cosine_distance('[1.0, 0.0]'::vector, '[0.0, 1.0]'::vector) >= 0 AS cosine_valid;
SELECT vector_inner_product('[1.0, 2.0, 3.0]'::vector, '[4.0, 5.0, 6.0]'::vector) > 0 AS inner_product_positive;
SELECT vector_l1_distance('[1.0, 2.0]'::vector, '[4.0, 6.0]'::vector) > 0 AS l1_positive;
SELECT vector_hamming_distance('[1.0, 0.0, 1.0]'::vector, '[1.0, 1.0, 0.0]'::vector) >= 0 AS hamming_valid;
SELECT vector_chebyshev_distance('[1.0, 2.0]'::vector, '[4.0, 6.0]'::vector);
SELECT vector_minkowski_distance('[1.0, 2.0]'::vector, '[4.0, 6.0]'::vector, 3.0) > 0 AS minkowski_positive;
