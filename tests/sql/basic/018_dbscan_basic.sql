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
		RAISE EXCEPTION 'test_train_view table does not exist. Please run: python ml_dataset.py <dataset_name>';
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
\echo 'DBSCAN Clustering - Basic Test'
\echo '=========================================================================='

-- Test DBSCAN clustering with eps=0.5, min_pts=5
-- Note: DBSCAN may return -1 for noise points
-- Don't output full array, just verify we got assignments for all rows
WITH clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 5)) AS cluster_id
)
SELECT 
	(SELECT COUNT(*) FROM test_train_view) AS total_rows,
	COUNT(DISTINCT cluster_id) AS num_clusters_plus_noise
FROM clusters;

-- Count noise points (-1)
WITH clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 5)) AS cluster_id
)
SELECT 
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points
FROM clusters;

\echo 'DBSCAN basic test completed successfully'

