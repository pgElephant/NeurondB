-- 018_dbscan_basic.sql
-- Basic test for DBSCAN clustering with GPU acceleration

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off
SET client_min_messages TO WARNING;

/* Step 1: Verify prerequisites and create views with 1000 rows */

SELECT 
	(SELECT COUNT(*)::bigint FROM test_train_view) AS train_rows,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_rows;

/* Step 2: Configure GPU */
\echo 'Step 2: Configuring GPU acceleration...'

SELECT neurondb_gpu_enable() AS gpu_available;

\echo '=========================================================================='
\echo '=========================================================================='

/* Step 3: Test DBSCAN clustering */

-- Test DBSCAN clustering with eps=0.5, min_pts=5
-- Note: DBSCAN may return -1 for noise points
WITH clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 5)) AS cluster_id
)
SELECT 
	(SELECT COUNT(*) FROM test_train_view) AS total_rows,
	COUNT(DISTINCT cluster_id) AS num_clusters_plus_noise,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points
FROM clusters;

\echo 'Test completed successfully'
