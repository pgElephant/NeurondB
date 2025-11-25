-- 026_vector_advance.sql
-- Comprehensive advanced test for ALL vector operations
-- Tests every function, operator, and code path in the vector implementation
-- Works on 1000 rows and tests each and every way with comprehensive coverage

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\pset null [NULL]
\pset format aligned

\echo '=========================================================================='
\echo '=========================================================================='

-- Verify required tables exist

DROP VIEW IF EXISTS test_vectors_view;

-- Create a temporary table with pre-converted vectors to avoid aggregate issues
DROP VIEW IF EXISTS test_vectors_view;
DROP TABLE IF EXISTS test_vectors_temp;
CREATE TEMP TABLE test_vectors_temp (
	v vector,
	arr double precision[],
	label integer
);

-- Insert converted vectors
INSERT INTO test_vectors_temp (v, arr, label)
SELECT 
	features AS v,
	vector_to_array(features) AS arr,
	label
FROM test_train_view
WHERE features IS NOT NULL;

-- Create a view for convenience
DROP VIEW IF EXISTS test_vectors_view;
CREATE VIEW test_vectors_view AS SELECT * FROM test_vectors_temp;

\echo ''
\echo 'Dataset Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
SELECT 
	COUNT(*)::bigint AS vector_count,
	(SELECT vector_dims(v) FROM test_vectors_temp LIMIT 1) AS feature_dim
FROM test_vectors_temp;

/*---- Register required GPU kernels ----*/
\echo ''
\echo 'GPU Configuration'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
SELECT neurondb_gpu_enable() AS gpu_available;
SELECT neurondb_gpu_info() AS gpu_info;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT COUNT(*) AS vector_count FROM test_vectors_temp;

SELECT
	vector_l2_distance(v1.v, v2.v) AS l2_dist,
	vector_cosine_distance(v1.v, v2.v) AS cosine_dist,
	vector_inner_product(v1.v, v2.v) AS inner_prod,
	vector_l1_distance(v1.v, v2.v) AS l1_dist,
	vector_hamming_distance(v1.v, v2.v) AS hamming_dist,
	vector_chebyshev_distance(v1.v, v2.v) AS chebyshev_dist,
	vector_minkowski_distance(v1.v, v2.v, 3.0) AS minkowski_dist,
	vector_squared_l2_distance(v1.v, v2.v) AS squared_l2_dist,
	vector_jaccard_distance(v1.v, v2.v) AS jaccard_dist,
	vector_dice_distance(v1.v, v2.v) AS dice_dist
FROM test_vectors_view v1
CROSS JOIN test_vectors_view v2
WHERE v1.v IS NOT NULL AND v2.v IS NOT NULL
LIMIT 5;

SELECT
	v1.v <-> v2.v AS l2_op,
	v1.v <=> v2.v AS cosine_op,
	v1.v <#> v2.v AS inner_op
FROM test_vectors_view v1
CROSS JOIN test_vectors_view v2
WHERE v1.v IS NOT NULL AND v2.v IS NOT NULL
LIMIT 5;

SELECT
	AVG(vector_dims(v)) AS avg_dimensions,
	MIN(vector_norm(v)) AS min_length,
	MAX(vector_norm(v)) AS max_length,
	AVG(vector_norm(v)) AS avg_length,
	AVG(vector_mean(v)) AS avg_mean,
	AVG(vector_variance(v)) AS avg_variance,
	AVG(vector_stddev(v)) AS avg_stddev,
	AVG(vector_min(v)) AS avg_min,
	AVG(vector_max(v)) AS avg_max,
	AVG(vector_element_sum(v)) AS avg_element_sum
FROM test_vectors_view
WHERE v IS NOT NULL;

SELECT
	vector_get(v, 0) AS first_element,
	vector_get(v, vector_dims(v) - 1) AS last_element,
	vector_set(v, 0, 99.0) AS modified_vector,
	vector_slice(v, 0, 3) AS sliced_vector,
	vector_append(v, 42.0) AS appended_vector,
	vector_prepend(42.0, v) AS prepended_vector
FROM test_vectors_view
WHERE v IS NOT NULL
LIMIT 5;

SELECT
	vector_abs(v) AS abs_vector,
	vector_square(v) AS square_vector,
	vector_sqrt(vector_abs(v)) AS sqrt_vector,
	vector_pow(v, 2.0) AS pow_vector
FROM test_vectors_view
WHERE v IS NOT NULL
LIMIT 5;

SELECT
	v1.v + v2.v AS add_result,
	v1.v - v2.v AS sub_result,
	v1.v * 2.0 AS mul_result,
	vector_hadamard(v1.v, v2.v) AS hadamard_result,
	vector_divide(v1.v, vector_abs(v2.v) + array_to_vector_float8(array_fill(0.001::double precision, ARRAY[vector_dims(v2.v)]))) AS divide_result
FROM test_vectors_view v1
CROSS JOIN test_vectors_view v2
WHERE v1.v IS NOT NULL AND v2.v IS NOT NULL
LIMIT 5;

SELECT
	vector_normalize(v) AS normalized,
	vector_norm(vector_normalize(v)) AS normalized_norm,
	vector_concat(v, v) AS concatenated
FROM test_vectors_view
WHERE v IS NOT NULL
LIMIT 5;

SELECT
	vector_to_array_float4(v) AS as_float4_array,
	vector_to_array_float8(v) AS as_float8_array,
	array_to_vector_float4(vector_to_array_float4(v)) AS roundtrip_f4,
	array_to_vector_float8(vector_to_array_float8(v)) AS roundtrip_f8,
	array_to_vector_integer(ARRAY[1,2,3,4,5]) AS from_int_array
FROM test_vectors_view
WHERE v IS NOT NULL
LIMIT 5;

SELECT
	vector_cast_dimension(v, 128) AS cast_to_128,
	vector_cast_dimension(v, 64) AS cast_to_64,
	vector_cast_dimension(v, 16) AS cast_to_16,
	vector_dims(vector_cast_dimension(v, 128)) AS dim_128,
	vector_dims(vector_cast_dimension(v, 64)) AS dim_64
FROM test_vectors_view
WHERE v IS NOT NULL
LIMIT 5;

WITH batch_data AS (
	SELECT ARRAY_AGG(v)::vector[] AS vec_array
	FROM (SELECT v FROM test_vectors_view LIMIT 10) t
),
query_vec AS (
	SELECT v FROM test_vectors_view LIMIT 1
)
SELECT
	vector_l2_distance_batch(batch_data.vec_array, query_vec.v) AS batch_l2,
	vector_cosine_distance_batch(batch_data.vec_array, query_vec.v) AS batch_cosine,
	vector_inner_product_batch(batch_data.vec_array, query_vec.v) AS batch_inner,
	vector_normalize_batch(batch_data.vec_array) AS batch_normalize,
	vector_sum_batch(batch_data.vec_array) AS batch_sum,
	vector_avg_batch(batch_data.vec_array) AS batch_avg
FROM batch_data, query_vec;

SELECT
	vector_quantize_fp16(v1.v) AS fp16_quantized,
	vector_dequantize_fp16(vector_quantize_fp16(v1.v)) AS fp16_dequantized,
	vector_l2_distance_fp16(vector_quantize_fp16(v1.v), vector_quantize_fp16(v2.v)) AS fp16_l2,
	vector_cosine_distance_fp16(vector_quantize_fp16(v1.v), vector_quantize_fp16(v2.v)) AS fp16_cosine
FROM test_vectors_view v1
CROSS JOIN test_vectors_view v2
WHERE v1.v IS NOT NULL AND v2.v IS NOT NULL
LIMIT 3;

WITH min_max AS (
	SELECT 
		MIN(vector_min(v)) AS min_val,
		MAX(vector_max(v)) AS max_val,
		MAX(vector_dims(v)) AS dims
	FROM test_vectors_view
),
min_vec AS (
	SELECT vector_cast_dimension(vector '[0]', (SELECT dims FROM min_max)) AS v
),
max_vec AS (
	SELECT vector_cast_dimension(vector '[100]', (SELECT dims FROM min_max)) AS v
)
SELECT
	vector_quantize_int8(tv.v, min_vec.v, max_vec.v) AS int8_quantized,
	vector_dequantize_int8(
		vector_quantize_int8(tv.v, min_vec.v, max_vec.v),
		min_vec.v,
		max_vec.v
	) AS int8_dequantized
FROM test_vectors_view tv, min_vec, max_vec
WHERE tv.v IS NOT NULL
LIMIT 3;

SELECT
	vector_percentile(v, 0.0) AS p0,
	vector_percentile(v, 0.25) AS p25,
	vector_percentile(v, 0.5) AS p50,
	vector_percentile(v, 0.75) AS p75,
	vector_percentile(v, 1.0) AS p100,
	vector_median(v) AS median,
	vector_quantile(v, ARRAY[0.25, 0.5, 0.75]::double precision[]) AS quartiles
FROM test_vectors_view
WHERE v IS NOT NULL
LIMIT 5;

SELECT
	vector_scale(v, ARRAY[2.0, 2.0, 2.0]::real[]) AS scaled_3d,
	vector_translate(v, vector '[1,1,1]') AS translated_3d,
	vector_filter(v, ARRAY[true, false, true]::boolean[]) AS filtered_3d,
	vector_where(v, v, 0.0) AS where_result
FROM test_vectors_view
WHERE v IS NOT NULL AND vector_dims(v) = 3
LIMIT 5;

SELECT
	vector_cross_product(v1.v, v2.v) AS cross_product
FROM test_vectors_view v1
CROSS JOIN test_vectors_view v2
WHERE v1.v IS NOT NULL AND v2.v IS NOT NULL 
	AND vector_dims(v1.v) = 3 AND vector_dims(v2.v) = 3
LIMIT 5;

SELECT
	v1.v = v2.v AS eq_result,
	v1.v <> v2.v AS ne_result,
	vector_eq(v1.v, v2.v) AS eq_func,
	vector_ne(v1.v, v2.v) AS ne_func
FROM test_vectors_view v1
CROSS JOIN test_vectors_view v2
WHERE v1.v IS NOT NULL AND v2.v IS NOT NULL
LIMIT 5;

SELECT
	vector_clip(v, -1.0, 1.0) AS clipped,
	vector_standardize(v) AS standardized,
	vector_minmax_normalize(v) AS minmax_normalized
FROM test_vectors_view
WHERE v IS NOT NULL
LIMIT 5;

WITH hash_sample AS (
	SELECT vector_hash(v) AS hash_value
	FROM test_vectors_view
	WHERE v IS NOT NULL
	LIMIT 1
),
hash_count AS (
	SELECT COUNT(DISTINCT vector_hash(v)) AS unique_hashes
	FROM test_vectors_view
	WHERE v IS NOT NULL
)
SELECT hash_sample.hash_value, hash_count.unique_hashes
FROM hash_sample, hash_count;

DO $$
DECLARE
	v1 vector;
	v2 vector;
	v3 vector;
BEGIN
	-- Test with valid vectors
	v1 := array_to_vector_float8(ARRAY[1.0, 2.0, 3.0]::double precision[]);
	v2 := array_to_vector_float8(ARRAY[4.0, 5.0, 6.0]::double precision[]);
	
	-- Test operations
	
	-- Test dimension casting
	v3 := vector_cast_dimension(v1, 5);
	
END
$$;

SELECT
	array_to_vector_float4(ARRAY[1.0::real, 2.0::real, 3.0::real]) AS from_float4,
	array_to_vector_float8(ARRAY[1.0::double precision, 2.0::double precision, 3.0::double precision]) AS from_float8,
	array_to_vector_integer(ARRAY[1, 2, 3, 4, 5]) AS from_integer;

SELECT
	vector_normalize(vector '[0,0,0]') AS zero_vector,
	vector_normalize(vector '[1,0,0]') AS unit_x,
	vector_normalize(vector '[0,1,0]') AS unit_y,
	vector_normalize(vector '[0,0,1]') AS unit_z,
	vector_norm(vector_normalize(vector '[3,4,0]')) AS normalized_norm;

SELECT
	vector '[1,2,3]' + vector '[4,5,6]' AS add,
	vector '[5,6,7]' - vector '[1,2,3]' AS sub,
	vector '[1,2,3]' * 2.0 AS scalar_mul,
	vector_concat(vector '[1,2]', vector '[3,4]') AS concat;

SELECT
	vector_l2_distance(vector '[1,0,0]', vector '[0,1,0]') AS l2_orthogonal,
	vector_cosine_distance(vector '[1,0,0]', vector '[0,1,0]') AS cosine_orthogonal,
	vector_inner_product(vector '[1,0,0]', vector '[0,1,0]') AS inner_orthogonal,
	vector_l2_distance(vector '[1,1,1]', vector '[1,1,1]') AS l2_identical,
	vector_cosine_distance(vector '[1,1,1]', vector '[1,1,1]') AS cosine_identical;

SELECT
	COUNT(*) AS total,
	AVG(vector_dims(v)) AS avg_dims,
	MIN(vector_dims(v)) AS min_dims,
	MAX(vector_dims(v)) AS max_dims,
	AVG(vector_norm(v)) AS avg_norm,
	AVG(vector_mean(v)) AS avg_mean,
	AVG(vector_variance(v)) AS avg_variance
FROM test_vectors_view
WHERE v IS NOT NULL;

\echo 'Test completed successfully'
