\timing on
\pset footer off
\pset pager off

-- This test uses test_train_view table created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist

\set ON_ERROR_STOP on

/* Step 0: Read settings from test_settings table and verify GPU configuration */
DO $$
DECLARE
	gpu_mode TEXT;
	current_gpu_enabled TEXT;
BEGIN
	-- Read GPU mode setting from test_settings
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	
	-- Verify GPU configuration matches test_settings (set by test runner)
	SELECT current_setting('neurondb.gpu_enabled', true) INTO current_gpu_enabled;
	
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
\echo '=========================================================================='

-- Test Mini-Batch K-Means clustering with k=3, batch_size=100
-- Don't output full array, just verify we got assignments for all rows
WITH clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 100)) AS cluster_id
)
SELECT 
	(SELECT COUNT(*) FROM test_train_view) AS total_rows,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM clusters;

\echo 'Test completed successfully'
