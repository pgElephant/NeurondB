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
