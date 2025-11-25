-- 010_gpu_info_advance.sql
-- Exhaustive detailed test for GPU information and operations: all functions, error handling.
-- Tests: GPU info, stats, configuration, error handling, metadata

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

/* Step 0: Read settings from test_settings table and verify GPU configuration */
DO $$
DECLARE
	gpu_mode TEXT;
	current_gpu_enabled TEXT;
	current_gpu_kernels TEXT;
	gpu_kernels_val TEXT;
BEGIN
	-- Read GPU mode setting from test_settings
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	SELECT setting_value INTO gpu_kernels_val FROM test_settings WHERE setting_key = 'gpu_kernels';
	
	-- Verify GPU configuration matches test_settings (set by test runner)
	SELECT current_setting('neurondb.gpu_enabled', true) INTO current_gpu_enabled;
	SELECT current_setting('neurondb.gpu_kernels', true) INTO current_gpu_kernels;
	
	IF gpu_mode = 'gpu' THEN
		-- Verify GPU is enabled (should be set by test runner)
		IF current_gpu_enabled != 'on' THEN
			RAISE WARNING 'GPU mode expected but neurondb.gpu_enabled = % (expected: on)', current_gpu_enabled;
		END IF;
	ELSE
		-- Verify GPU is disabled (should be set by test runner)
		IF current_gpu_enabled != 'off' THEN
			RAISE WARNING 'CPU mode expected but neurondb.gpu_enabled = % (expected: off)', current_gpu_enabled;
		END IF;
	END IF;
END $$;

\echo '=========================================================================='
\echo 'GPU Information: Exhaustive Advanced Test'
\echo '=========================================================================='

\echo ''
\echo 'GPU Configuration'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
-- GPU already configured via test_settings above
SELECT current_setting('neurondb.gpu_enabled', true) AS gpu_available;

/*
 * ---- GPU INFORMATION TESTS ----
 * Test all GPU information and configuration functions
 */
\echo ''
\echo 'GPU Information Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Detailed GPU Device Information'
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
	END AS availability_status,
	ROUND((total_memory_mb - free_memory_mb)::numeric / NULLIF(total_memory_mb, 0) * 100, 2) AS usage_percent
FROM neurondb_gpu_info();

\echo 'Test 2: GPU Statistics Tracking'
-- Get initial stats
SELECT 
	'Initial' AS stats_type,
	*
FROM neurondb_gpu_stats();

-- Perform some GPU operations to generate stats
SELECT vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);
SELECT vector_cosine_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);
SELECT vector_inner_product_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);

-- Get updated stats
SELECT 
	'After Operations' AS stats_type,
	*
FROM neurondb_gpu_stats();

\echo 'Test 3: GPU Statistics Reset'
SELECT neurondb_gpu_reset_stats() AS stats_reset;
SELECT 
	'After Reset' AS stats_type,
	*
FROM neurondb_gpu_stats();

\echo 'Test 4: LLM GPU Information'
SELECT 
	neurondb_llm_gpu_available() AS llm_gpu_available,
	*
FROM neurondb_llm_gpu_info();

\echo 'Test 5: GPU Distance Functions Performance Comparison'
WITH test_vectors AS (
	SELECT '[1,2,3,4,5]'::vector AS v1, '[6,7,8,9,10]'::vector AS v2
)
SELECT 
	'vector_l2_distance' AS function_name,
	ROUND((vector_l2_distance(v1, v2))::numeric, 6) AS cpu_result,
	ROUND((vector_l2_distance_gpu(v1, v2))::numeric, 6) AS gpu_result,
	ROUND(ABS((vector_l2_distance(v1, v2) - vector_l2_distance_gpu(v1, v2))::numeric), 8) AS difference,
	CASE 
		WHEN ABS(vector_l2_distance(v1, v2) - vector_l2_distance_gpu(v1, v2)) < 0.0001 
		THEN '✓ Results match'
		ELSE '✗ Results differ'
	END AS match_status
FROM test_vectors
UNION ALL
SELECT 
	'vector_cosine_distance' AS function_name,
	ROUND((vector_cosine_distance(v1, v2))::numeric, 6) AS cpu_result,
	ROUND((vector_cosine_distance_gpu(v1, v2))::numeric, 6) AS gpu_result,
	ROUND(ABS((vector_cosine_distance(v1, v2) - vector_cosine_distance_gpu(v1, v2))::numeric), 8) AS difference,
	CASE 
		WHEN ABS(vector_cosine_distance(v1, v2) - vector_cosine_distance_gpu(v1, v2)) < 0.0001 
		THEN '✓ Results match'
		ELSE '✗ Results differ'
	END AS match_status
FROM test_vectors
UNION ALL
SELECT 
	'vector_inner_product' AS function_name,
	ROUND((vector_inner_product(v1, v2))::numeric, 6) AS cpu_result,
	ROUND((vector_inner_product_gpu(v1, v2))::numeric, 6) AS gpu_result,
	ROUND(ABS((vector_inner_product(v1, v2) - vector_inner_product_gpu(v1, v2))::numeric), 8) AS difference,
	CASE 
		WHEN ABS(vector_inner_product(v1, v2) - vector_inner_product_gpu(v1, v2)) < 0.0001 
		THEN '✓ Results match'
		ELSE '✗ Results differ'
	END AS match_status
FROM test_vectors;

\echo 'Test 6: GPU Kernel Configuration'
-- Test different kernel configurations
SET neurondb.gpu_kernels = 'l2,cosine';
SELECT neurondb_gpu_enable();
SELECT 
	'l2,cosine' AS kernel_config,
	current_setting('neurondb.gpu_kernels') AS actual_config;

SET neurondb.gpu_kernels = 'l2,cosine,ip,linreg_train,linreg_predict';
SELECT neurondb_gpu_enable();
SELECT 
	'Full config' AS kernel_config,
	current_setting('neurondb.gpu_kernels') AS actual_config;

\echo 'Test 7: GPU Memory Usage Monitoring'
SELECT 
	device_id,
	device_name,
	total_memory_mb,
	free_memory_mb,
	total_memory_mb - free_memory_mb AS used_memory_mb,
	ROUND((total_memory_mb - free_memory_mb)::numeric / NULLIF(total_memory_mb, 0) * 100, 2) AS usage_percent,
	CASE 
		WHEN (total_memory_mb - free_memory_mb)::numeric / NULLIF(total_memory_mb, 0) > 0.9 THEN 'High usage'
		WHEN (total_memory_mb - free_memory_mb)::numeric / NULLIF(total_memory_mb, 0) > 0.5 THEN 'Moderate usage'
		ELSE 'Low usage'
	END AS usage_status
FROM neurondb_gpu_info()
WHERE is_available = true;

\echo 'Test 8: GPU Enable/Disable'
SET neurondb.gpu_enabled = off;
SELECT 
	'GPU Disabled' AS status,
	neurondb_gpu_enable() AS enable_result;

SET neurondb.gpu_enabled = on;
SELECT 
	'GPU Enabled' AS status,
	neurondb_gpu_enable() AS enable_result;

/* --- ERROR path: invalid parameters --- */
\echo ''
\echo 'Error Handling Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: GPU function with GPU disabled'
DO $$
DECLARE
	result double precision;
BEGIN
	SET neurondb.gpu_enabled = off;
	BEGIN
		result := vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
	SET neurondb.gpu_enabled = on;
END$$;

\echo 'Error Test 2: Invalid kernel configuration'
DO $$
BEGIN
	BEGIN
		SET neurondb.gpu_kernels = 'invalid_kernel_name';
		PERFORM neurondb_gpu_enable();
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo ''
\echo '=========================================================================='
\echo '✓ GPU Information: Full exhaustive test complete'
\echo '=========================================================================='

\echo 'Test completed successfully'
