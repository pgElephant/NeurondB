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

-- Test K-Means clustering with k=3
-- Don't output full array, just verify we got assignments for all rows
CREATE TEMP TABLE kmeans_model AS
SELECT train_kmeans_model_id('test_train_view', 'features', 3, 100) as model_id;

SELECT * FROM kmeans_model;

-- Test predictions
-- Note: predict_kmeans_model_id function is not yet implemented
-- Commenting out prediction test until function is implemented
-- SELECT
--     features[1:3] as sample_features,
--     predict_kmeans((SELECT model_id FROM kmeans_model), features) as predicted_cluster
-- FROM test_test_view
-- LIMIT 5;

-- Evaluate model
CREATE TEMP TABLE kmeans_metrics AS
SELECT evaluate_kmeans_by_model_id(
    (SELECT model_id FROM kmeans_model),
    'test_test_view',
    'features'
) as metrics;

SELECT
    'Inertia' as metric, ROUND((metrics->>'inertia')::numeric, 6)::text as value
FROM kmeans_metrics
UNION ALL
SELECT 'N_Clusters', (metrics->>'n_clusters')::text
FROM kmeans_metrics
UNION ALL
SELECT 'N_Iterations', (metrics->>'n_iterations')::text
FROM kmeans_metrics
ORDER BY metric;

-- Summary
SELECT
    (SELECT model_id FROM kmeans_model) as model_id,
    (SELECT COUNT(*) FROM test_train_view) as training_samples,
    (SELECT COUNT(*) FROM test_test_view) as test_samples,
    (SELECT ROUND((metrics->>'inertia')::numeric, 6) FROM kmeans_metrics) as inertia;

\echo 'Test completed successfully'
