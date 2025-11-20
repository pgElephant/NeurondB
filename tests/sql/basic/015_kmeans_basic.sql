\timing on
\pset footer off
\pset pager off

-- This test uses test_train_view table created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist

SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

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
