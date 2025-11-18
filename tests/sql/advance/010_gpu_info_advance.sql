\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'GPU Information - Advanced Features Test'
\echo '=========================================================================='

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,linreg_train,linreg_predict';
SELECT neurondb_gpu_enable();

\echo ''
\echo 'Test 1: Detailed GPU Device Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	device_id,
	device_name,
	total_memory_mb,
	free_memory_mb,
	compute_capability_major,
	compute_capability_minor,
	is_available,
	CASE 
		WHEN is_available THEN 'Available'
		ELSE 'Not Available'
	END AS availability_status
FROM neurondb_gpu_info();

\echo ''
\echo 'Test 2: GPU Statistics Tracking'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Get initial stats
SELECT * FROM neurondb_gpu_stats() AS initial_stats;

-- Perform some GPU operations to generate stats
SELECT vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);
SELECT vector_cosine_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);

-- Get updated stats
SELECT * FROM neurondb_gpu_stats() AS updated_stats;

\echo ''
\echo 'Test 3: GPU Statistics Reset'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb_gpu_reset_stats() AS stats_reset;
SELECT * FROM neurondb_gpu_stats() AS stats_after_reset;

\echo ''
\echo 'Test 4: LLM GPU Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb_llm_gpu_available() AS llm_gpu_available;
SELECT * FROM neurondb_llm_gpu_info();

\echo ''
\echo 'Test 5: GPU Distance Functions Performance'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Compare CPU vs GPU distance calculations
WITH test_vectors AS (
	SELECT '[1,2,3,4,5]'::vector AS v1, '[6,7,8,9,10]'::vector AS v2
)
SELECT 
	'vector_l2_distance' AS function_name,
	vector_l2_distance(v1, v2) AS cpu_result,
	vector_l2_distance_gpu(v1, v2) AS gpu_result,
	ABS(vector_l2_distance(v1, v2) - vector_l2_distance_gpu(v1, v2)) AS difference
FROM test_vectors
UNION ALL
SELECT 
	'vector_cosine_distance' AS function_name,
	vector_cosine_distance(v1, v2) AS cpu_result,
	vector_cosine_distance_gpu(v1, v2) AS gpu_result,
	ABS(vector_cosine_distance(v1, v2) - vector_cosine_distance_gpu(v1, v2)) AS difference
FROM test_vectors;

\echo ''
\echo 'Test 6: GPU Kernel Configuration'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test different kernel configurations
SET neurondb.gpu_kernels = 'l2,cosine';
SELECT neurondb_gpu_enable();
SELECT current_setting('neurondb.gpu_kernels') AS kernel_config;

SET neurondb.gpu_kernels = 'l2,cosine,ip,linreg_train,linreg_predict';
SELECT neurondb_gpu_enable();
SELECT current_setting('neurondb.gpu_kernels') AS kernel_config;

\echo ''
\echo 'Test 7: GPU Memory Usage Monitoring'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	device_id,
	device_name,
	total_memory_mb,
	free_memory_mb,
	total_memory_mb - free_memory_mb AS used_memory_mb,
	ROUND((total_memory_mb - free_memory_mb)::numeric / total_memory_mb * 100, 2) AS usage_percent
FROM neurondb_gpu_info()
WHERE is_available = true;

\echo ''
\echo 'Advanced GPU Information Test Complete!'
\echo '=========================================================================='

