-- 030_gpu_negative.sql
-- Negative test cases for GPU module: error handling, invalid inputs, fallback scenarios

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'GPU Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- GPU INITIALIZATION ERRORS ----
 * Test error handling for GPU initialization failures
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU Initialization Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: GPU Enable with Invalid Device ID'
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_device = 999;
SELECT neurondb_gpu_enable();

\echo 'Error Test 2: GPU Enable with Invalid Kernel Configuration'
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_device = 0;
SET neurondb.gpu_kernels = 'invalid_kernel_name_xyz';
SELECT neurondb_gpu_enable();

\echo 'Error Test 3: GPU Enable with NULL Device'
DO $$
BEGIN
	BEGIN
		SET neurondb.gpu_device = NULL;
		PERFORM neurondb_gpu_enable();
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- GPU FUNCTION ERRORS ----
 * Test error handling for GPU function calls with invalid inputs
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU Function Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 4: GPU Distance with NULL vectors'
SELECT vector_l2_distance_gpu(NULL::vector, vector '[1,2,3]'::vector);
SELECT vector_l2_distance_gpu(vector '[1,2,3]'::vector, NULL::vector);
SELECT vector_l2_distance_gpu(NULL::vector, NULL::vector);

\echo 'Error Test 5: GPU Distance with Dimension Mismatch'
SELECT vector_l2_distance_gpu(
	vector '[1,2,3]'::vector,
	vector '[1,2,3,4]'::vector
);

\echo 'Error Test 6: GPU Distance with GPU Disabled'
SET neurondb.gpu_enabled = off;
SELECT vector_l2_distance_gpu(
	vector '[1,2,3]'::vector,
	vector '[4,5,6]'::vector
);

\echo 'Error Test 7: GPU Cosine Distance with Invalid Inputs'
SET neurondb.gpu_enabled = on;
SELECT vector_cosine_distance_gpu(
	vector '[1,2,3]'::vector,
	vector '[1,2]'::vector
);

\echo 'Error Test 8: GPU Inner Product with Invalid Inputs'
SELECT vector_inner_product_gpu(
	vector '[1,2,3,4]'::vector,
	vector '[1,2]'::vector
);

/*-------------------------------------------------------------------
 * ---- GPU MEMORY ERRORS ----
 * Test error handling for GPU memory allocation failures
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU Memory Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 9: GPU Operations with Very Large Vectors'
-- Test with vectors that might exceed GPU memory
DO $$
BEGIN
	BEGIN
		-- Create very large vector (if supported)
		PERFORM vector_l2_distance_gpu(
			(SELECT features FROM test_train_view LIMIT 1),
			(SELECT features FROM test_train_view LIMIT 1)
		);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- May error if vector too large
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- GPU STATISTICS ERRORS ----
 * Test error handling for GPU statistics operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU Statistics Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 10: GPU Stats with GPU Disabled'
SET neurondb.gpu_enabled = off;
SELECT * FROM neurondb_gpu_stats();

\echo 'Error Test 11: GPU Stats Reset with GPU Disabled'
SELECT neurondb_gpu_reset_stats();

/*-------------------------------------------------------------------
 * ---- GPU BACKEND ERRORS ----
 * Test error handling for GPU backend selection
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU Backend Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 12: GPU Info with Invalid Backend Configuration'
-- This should still work but may show no devices
SELECT * FROM neurondb_gpu_info();

\echo 'Error Test 13: GPU Operations with Unsupported Backend'
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_backend = 'unsupported_backend';
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
 * ---- GPU MODEL ERRORS ----
 * Test error handling for GPU model operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'GPU Model Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 14: GPU Model Training with Invalid Data'
SET neurondb.gpu_enabled = on;
SELECT neurondb_gpu_enable();

-- Try training with empty table
CREATE TEMP TABLE empty_gpu_train (
	features vector,
	label float8
);

SELECT neurondb.train(
	'linear_regression',
	'empty_gpu_train',
	'features',
	'label',
	'{}'::jsonb
);

DROP TABLE IF EXISTS empty_gpu_train;

\echo 'Error Test 15: GPU Model Prediction with Invalid Model ID'
SELECT neurondb.predict(-1, vector '[1,2,3]'::vector);

\echo 'Error Test 16: GPU Model Evaluation with Invalid Model ID'
SELECT neurondb.evaluate(-1, 'test_test_view', 'features', 'label');

\echo ''
\echo '=========================================================================='
\echo '✓ GPU Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'
