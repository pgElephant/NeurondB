\timing on
\pset footer off
\pset pager off

-- This test uses sample_train table created by ml_dataset.py
-- Performance test: Works on the whole 11M row view
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
\echo 'K-Means Clustering - Performance Test (Full Dataset)'
\echo '=========================================================================='

-- Test K-Means clustering with k=3 on full dataset
SELECT cluster_kmeans('sample_train', 'features', 3, 100) AS cluster_assignments;

-- Verify we got assignments for all rows
WITH clusters AS (
	SELECT unnest(cluster_kmeans('sample_train', 'features', 3, 100)) AS cluster_id
)
SELECT 
	(SELECT COUNT(*) FROM sample_train) AS total_rows,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM clusters;

\echo 'K-Means performance test completed successfully'

