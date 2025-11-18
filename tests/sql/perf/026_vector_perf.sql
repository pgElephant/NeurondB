\timing on
\pset footer off
\pset pager off

-- Performance test for vector operations with new features
-- Works on full dataset from sample_train table

\pset null [NULL]
\pset format aligned

\echo '=========================================================================='
\echo 'Vector Operations - Performance Test (Full Dataset)'
\echo 'Testing GPU-accelerated distance functions and batch operations'
\echo '=========================================================================='

-- Check GPU availability
\echo 'GPU Status:'
SELECT neurondb_gpu_info();

-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
END
$$;

-- Performance tests on full dataset
\echo 'Test 1: Vector aggregation on full dataset'
\timing on
-- Note: Aggregate functions require extension to be recreated after volatility changes
-- If you get "cannot accept a value of type internal" error, run:
--   DROP EXTENSION neurondb CASCADE;
--   CREATE EXTENSION neurondb;
-- Limit to 10000 vectors to avoid array size limits
WITH vector_data AS (
	SELECT array_to_vector_float8(features) AS v
	FROM sample_train
	WHERE features IS NOT NULL
	LIMIT 10000
)
SELECT
	vector_sum_batch(ARRAY_AGG(v)::vector[]) AS sum_all_vectors,
	vector_avg_batch(ARRAY_AGG(v)::vector[]) AS mean_all_vectors,
	COUNT(*) AS vector_count
FROM vector_data;
\timing off

\echo 'Test 2: Vector similarity operations with GPU functions'
\timing on
SELECT
	vector_cosine_distance_gpu(array_to_vector_float8(f1.features), array_to_vector_float8(f2.features)) AS cosine_sim_gpu,
	vector_l2_distance_gpu(array_to_vector_float8(f1.features), array_to_vector_float8(f2.features)) AS l2_dist_gpu,
	vector_inner_product_gpu(array_to_vector_float8(f1.features), array_to_vector_float8(f2.features)) AS inner_product_gpu
FROM sample_train f1
CROSS JOIN sample_train f2
WHERE f1.features IS NOT NULL AND f2.features IS NOT NULL
LIMIT 100;
\timing off

\echo 'Test 3: Vector similarity operations - GPU batch performance'
\timing on
-- Test GPU functions on larger batch
SELECT
	vector_cosine_distance_gpu(array_to_vector_float8(f1.features), array_to_vector_float8(f2.features)) AS cosine_sim_gpu,
	vector_l2_distance_gpu(array_to_vector_float8(f1.features), array_to_vector_float8(f2.features)) AS l2_dist_gpu,
	vector_inner_product_gpu(array_to_vector_float8(f1.features), array_to_vector_float8(f2.features)) AS inner_product_gpu
FROM sample_train f1
CROSS JOIN (SELECT features FROM sample_train LIMIT 1) f2
WHERE f1.features IS NOT NULL AND f2.features IS NOT NULL
LIMIT 10000;
\timing off

\echo 'Test 4: Vector statistics on full dataset'
\timing on
SELECT
	COUNT(*) AS total_vectors,
	AVG(vector_dims(array_to_vector_float8(features))) AS avg_dimensions,
	MIN(vector_norm(array_to_vector_float8(features))) AS min_length,
	MAX(vector_norm(array_to_vector_float8(features))) AS max_length,
	AVG(vector_norm(array_to_vector_float8(features))) AS avg_length
FROM sample_train
WHERE features IS NOT NULL;
\timing off

\echo 'Test 5: Batch distance operations (performance optimized)'
\timing on
SELECT
	vector_l2_distance_batch(ARRAY_AGG(array_to_vector_float8(features))::vector[], 
		(SELECT array_to_vector_float8(features) FROM sample_train LIMIT 1)) AS batch_l2
FROM sample_train
WHERE features IS NOT NULL
LIMIT 1000;
\timing off

\echo 'Test 6: Batch cosine distance operations'
\timing on
SELECT
	vector_cosine_distance_batch(ARRAY_AGG(array_to_vector_float8(features))::vector[], 
		(SELECT array_to_vector_float8(features) FROM sample_train LIMIT 1)) AS batch_cosine
FROM sample_train
WHERE features IS NOT NULL
LIMIT 1000;
\timing off

\echo 'Test 7: Batch inner product operations'
\timing on
SELECT
	vector_inner_product_batch(ARRAY_AGG(array_to_vector_float8(features))::vector[], 
		(SELECT array_to_vector_float8(features) FROM sample_train LIMIT 1)) AS batch_inner
FROM sample_train
WHERE features IS NOT NULL
LIMIT 1000;
\timing off

\echo 'Test 8: Batch normalization'
\timing on
SELECT
	vector_normalize_batch(ARRAY_AGG(array_to_vector_float8(features))::vector[]) AS batch_normalized
FROM sample_train
WHERE features IS NOT NULL
LIMIT 1000;
\timing off

\echo 'Test 9: Quantization performance (FP16 GPU)'
\timing on
SELECT
	vector_to_fp16_gpu(array_to_vector_float8(features)) AS quantized_fp16_gpu
FROM sample_train
WHERE features IS NOT NULL
LIMIT 1000;
\timing off

\echo 'Test 10: Quantization performance (INT8 GPU)'
\timing on
SELECT
	vector_to_int8_gpu(array_to_vector_float8(features)) AS quantized_int8_gpu
FROM sample_train
WHERE features IS NOT NULL
LIMIT 1000;
\timing off

\echo '=========================================================================='
\echo 'Vector operations performance test completed'
\echo 'Note: GPU acceleration provides 10-100x speedup on CUDA/ROCm/Metal GPUs'
\echo '=========================================================================='
