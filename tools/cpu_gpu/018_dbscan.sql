\timing on
\pset footer off
\pset pager off

-- This test uses sample_train table created by ml_dataset.py
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

SET neurondb.gpu_enabled = on;
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'DBSCAN Clustering - Basic Test'
\echo '=========================================================================='

-- Test DBSCAN clustering with eps=0.5, min_pts=5
-- Note: DBSCAN may return -1 for noise points
SELECT cluster_dbscan('sample_train', 'features', 0.5, 5) AS cluster_assignments;

-- Verify we got assignments for all rows
WITH clusters AS (
	SELECT unnest(cluster_dbscan('sample_train', 'features', 0.5, 5)) AS cluster_id
)
SELECT 
	(SELECT COUNT(*) FROM sample_train) AS total_rows,
	COUNT(DISTINCT cluster_id) AS num_clusters_plus_noise
FROM clusters;

-- Count noise points (-1)
WITH clusters AS (
	SELECT unnest(cluster_dbscan('sample_train', 'features', 0.5, 5)) AS cluster_id
)
SELECT 
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points
FROM clusters;

\echo 'DBSCAN basic test completed successfully'

