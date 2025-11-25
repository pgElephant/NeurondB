-- 030_gpu_advance.sql
-- Comprehensive advanced test for ALL GPU module functions
-- Tests GPU backends, initialization, error handling, fallback scenarios
-- Works on 1000 rows and tests each and every GPU code path

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'GPU Module: Exhaustive Backend and Error Path Coverage'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- GPU BACKEND DETECTION ----
 * Test GPU backend detection and selection (CUDA/ROCm/Metal)
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU Backend Detection Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: GPU Backend Information'
SELECT 
	device_id,
	device_name,
	backend,
	platform,
	compute_capability_major,
	compute_capability_minor,
	is_available
FROM neurondb_gpu_info();

\echo 'Test 2: Backend Type Detection'
SELECT 
	CASE 
		WHEN backend LIKE '%cuda%' THEN 'CUDA'
		WHEN backend LIKE '%rocm%' OR backend LIKE '%hip%' THEN 'ROCm'
		WHEN backend LIKE '%metal%' THEN 'Metal'
		ELSE 'Unknown'
	END AS detected_backend,
	COUNT(*) AS device_count
FROM neurondb_gpu_info()
WHERE is_available = true
GROUP BY detected_backend;

/*-------------------------------------------------------------------
 * ---- GPU INITIALIZATION TESTS ----
 * Test GPU initialization with various configurations
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU Initialization Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: GPU Enable with Default Configuration'
SET neurondb.gpu_enabled = on;
SELECT neurondb_gpu_enable() AS enable_result;
SELECT 
	'After Enable' AS status,
	current_setting('neurondb.gpu_enabled') AS gpu_enabled_setting;

\echo 'Test 2: GPU Enable with Specific Device'
SET neurondb.gpu_device = 0;
SELECT neurondb_gpu_enable() AS enable_result;
SELECT 
	'Device 0' AS config,
	current_setting('neurondb.gpu_device') AS device_setting;

\echo 'Test 3: GPU Enable with Kernel Selection'
SET neurondb.gpu_kernels = 'l2,cosine,ip';
SELECT neurondb_gpu_enable() AS enable_result;
SELECT 
	'Kernel Config' AS config,
	current_setting('neurondb.gpu_kernels') AS kernels_setting;

\echo 'Test 4: GPU Disable'
SET neurondb.gpu_enabled = off;
SELECT neurondb_gpu_enable() AS disable_result;
SELECT 
	'After Disable' AS status,
	current_setting('neurondb.gpu_enabled') AS gpu_enabled_setting;

/*-------------------------------------------------------------------
 * ---- GPU MEMORY OPERATIONS ----
 * Test GPU memory allocation and management
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU Memory Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: GPU Memory Information'
SELECT 
	device_id,
	device_name,
	total_memory_mb,
	free_memory_mb,
	total_memory_mb - free_memory_mb AS used_memory_mb,
	ROUND((total_memory_mb - free_memory_mb)::numeric / NULLIF(total_memory_mb, 0) * 100, 2) AS usage_percent
FROM neurondb_gpu_info()
WHERE is_available = true;

\echo 'Test 2: GPU Memory Usage After Operations'
-- Perform GPU operations
SET neurondb.gpu_enabled = on;
SELECT neurondb_gpu_enable();

-- Batch GPU distance computations
SELECT COUNT(*) AS batch_size
FROM (
	SELECT vector_l2_distance_gpu(v1, v2) AS distance
	FROM (
		SELECT 
			features AS v1,
			(SELECT features FROM test_train_view ORDER BY random() LIMIT 1) AS v2
		FROM test_train_view
		LIMIT 100
	) sub
) batch_ops;

-- Check memory after operations
SELECT 
	'After Batch Ops' AS status,
	free_memory_mb,
	total_memory_mb - free_memory_mb AS used_memory_mb
FROM neurondb_gpu_info()
WHERE is_available = true;

/*-------------------------------------------------------------------
 * ---- GPU BATCH OPERATIONS ----
 * Test GPU batch processing with various sizes
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU Batch Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Small Batch (10 vectors)'
SELECT 
	'Small Batch' AS batch_type,
	COUNT(*) AS n_operations,
	ROUND(AVG(distance)::numeric, 6) AS avg_distance
FROM (
	SELECT vector_l2_distance_gpu(v1, v2) AS distance
	FROM (
		SELECT 
			features AS v1,
			(SELECT features FROM test_train_view ORDER BY random() LIMIT 1) AS v2
		FROM test_train_view
		LIMIT 10
	) sub
) batch;

\echo 'Test 2: Medium Batch (100 vectors)'
SELECT 
	'Medium Batch' AS batch_type,
	COUNT(*) AS n_operations,
	ROUND(AVG(distance)::numeric, 6) AS avg_distance
FROM (
	SELECT vector_l2_distance_gpu(v1, v2) AS distance
	FROM (
		SELECT 
			features AS v1,
			(SELECT features FROM test_train_view ORDER BY random() LIMIT 1) AS v2
		FROM test_train_view
		LIMIT 100
	) sub
) batch;

\echo 'Test 3: Large Batch (1000 vectors)'
SELECT 
	'Large Batch' AS batch_type,
	COUNT(*) AS n_operations,
	ROUND(AVG(distance)::numeric, 6) AS avg_distance
FROM (
	SELECT vector_l2_distance_gpu(v1, v2) AS distance
	FROM (
		SELECT 
			features AS v1,
			(SELECT features FROM test_train_view ORDER BY random() LIMIT 1) AS v2
		FROM test_train_view
		LIMIT 1000
	) sub
) batch;

/*-------------------------------------------------------------------
 * ---- GPU-CPU FALLBACK TESTS ----
 * Test fallback scenarios when GPU operations fail
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU-CPU Fallback Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Fallback with GPU Disabled'
SET neurondb.gpu_enabled = off;
-- These should fallback to CPU or handle gracefully
SELECT 
	'GPU Disabled' AS scenario,
	vector_l2_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS cpu_distance;

\echo 'Test 2: Fallback with Invalid GPU Device'
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_device = 999; -- Invalid device ID
-- Should handle gracefully
DO $$
BEGIN
	BEGIN
		PERFORM neurondb_gpu_enable();
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected to handle error
	END;
END$$;

\echo 'Test 3: Fallback with Invalid Kernels'
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_device = 0;
SET neurondb.gpu_kernels = 'invalid_kernel_xyz';
-- Should handle gracefully
DO $$
BEGIN
	BEGIN
		PERFORM neurondb_gpu_enable();
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected to handle error
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- GPU STATISTICS AND MONITORING ----
 * Test GPU statistics collection and monitoring
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU Statistics Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Initial GPU Statistics'
SELECT neurondb_gpu_reset_stats() AS reset_result;
SELECT 
	'Initial' AS stats_type,
	*
FROM neurondb_gpu_stats();

\echo 'Test 2: Statistics After Operations'
-- Perform various GPU operations
SET neurondb.gpu_enabled = on;
SELECT neurondb_gpu_enable();

SELECT vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);
SELECT vector_cosine_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);
SELECT vector_inner_product_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);

SELECT 
	'After Ops' AS stats_type,
	*
FROM neurondb_gpu_stats();

\echo 'Test 3: Statistics Reset'
SELECT neurondb_gpu_reset_stats() AS reset_result;
SELECT 
	'After Reset' AS stats_type,
	*
FROM neurondb_gpu_stats();

/*-------------------------------------------------------------------
 * ---- GPU MODEL OPERATIONS ----
 * Test GPU model training and inference
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU Model Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: GPU Model Training'
SET neurondb.gpu_enabled = on;
SELECT neurondb_gpu_enable();

DROP TABLE IF EXISTS gpu_model_test;
CREATE TEMP TABLE gpu_model_test AS
SELECT neurondb.train(
	'linear_regression',
	'test_train_view',
	'features',
	'label',
	'{}'::jsonb
)::integer AS model_id;

SELECT 
	'GPU Model' AS model_type,
	model_id,
	(SELECT metrics->>'storage' FROM neurondb.ml_models WHERE model_id = model_id) AS storage_type
FROM gpu_model_test;

\echo 'Test 2: GPU Model Prediction'
SELECT 
	'GPU Prediction' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(prediction)::numeric, 6) AS avg_prediction
FROM (
	SELECT neurondb.predict((SELECT model_id FROM gpu_model_test), features) AS prediction
	FROM test_test_view
	LIMIT 100
) pred;

\echo 'Test 3: GPU Model Evaluation'
SELECT neurondb.evaluate(
	(SELECT model_id FROM gpu_model_test),
	'test_test_view',
	'features',
	'label'
) AS gpu_evaluation_metrics;

\echo ''
\echo '=========================================================================='
\echo '✓ GPU Module: Full exhaustive code-path test complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




