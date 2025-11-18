-- 018_dbscan_perf.sql
-- Performance test for DBSCAN clustering with GPU acceleration
-- Works on full dataset from sample_train table

\timing on
\pset footer off
\pset pager off
\pset tuples_only off
\set ON_ERROR_STOP on
SET client_min_messages TO WARNING;

\echo '=========================================================================='
\echo 'DBSCAN Clustering - Performance Test (Full Dataset with GPU)'
\echo '=========================================================================='

-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
END
$$;

-- Configure GPU for performance
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip';
SELECT neurondb_gpu_enable() AS gpu_available;
SELECT neurondb_gpu_info() AS gpu_info;

\echo ''
\echo 'Testing DBSCAN clustering on full dataset (eps=0.5, min_pts=5)...'

-- Test DBSCAN clustering with eps=0.5, min_pts=5 on full dataset
-- Note: DBSCAN may return -1 for noise points
WITH clusters AS (
	SELECT unnest(cluster_dbscan('sample_train', 'features', 0.5, 5)) AS cluster_id
)
SELECT 
	(SELECT COUNT(*) FROM sample_train) AS total_rows,
	COUNT(DISTINCT cluster_id) AS num_clusters_plus_noise,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points
FROM clusters;

\echo 'DBSCAN performance test completed successfully'
