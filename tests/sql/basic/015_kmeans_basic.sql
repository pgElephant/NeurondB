\timing on
\pset footer off
\pset pager off

-- This test uses test_train_view table created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
END
$$;

-- Create views with 1000 rows for basic tests
DROP VIEW IF EXISTS test_train_view;
DROP VIEW IF EXISTS test_test_view;

CREATE VIEW test_train_view AS
SELECT features, label FROM sample_train LIMIT 1000;

CREATE VIEW test_test_view AS
SELECT features, label FROM sample_test LIMIT 1000;

SET neurondb.gpu_enabled = on;
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'K-Means Clustering - Basic Test'
\echo '=========================================================================='

-- Test K-Means clustering with k=3
-- Don't output full array, just verify we got assignments for all rows
CREATE TEMP TABLE kmeans_model AS
SELECT train_kmeans_model_id('test_train_view', 'features', 3, 100) as model_id;

SELECT * FROM kmeans_model;

-- Test predictions
SELECT
    features[1:3] as sample_features,
    predict_kmeans((SELECT model_id FROM kmeans_model), features) as predicted_cluster
FROM test_test_view
LIMIT 5;

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

\echo 'K-Means basic test completed successfully'

