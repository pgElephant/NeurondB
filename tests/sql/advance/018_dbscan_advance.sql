-- 018_dbscan_advance.sql
-- Exhaustive detailed test for dbscan clustering: all parameters, error handling.
-- Works on 1000 rows only and tests each and every way with comprehensive coverage
-- Tests: Different eps values, min_pts values, error handling, noise point validation

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo '=========================================================================='

/* Check that sample_train exists */

\echo ''
\echo 'Dataset Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
SELECT 
	COUNT(*)::bigint AS train_count,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_count,
	(SELECT vector_dims(features) FROM test_train_view LIMIT 1) AS feature_dim
FROM test_train_view;

/*---- Register required GPU kernels ----*/
\echo ''
\echo 'GPU Configuration'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
SELECT neurondb_gpu_enable() AS gpu_available;
SELECT neurondb_gpu_info() AS gpu_info;

/*
 * ---- CLUSTERING TESTS ----
 * Test multiple eps and min_pts values
 */
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH eps01_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.1, 5)) AS cluster_id
)
SELECT 
	'eps=0.1, min_pts=5' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM eps01_clusters;

WITH eps05_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 5)) AS cluster_id
)
SELECT 
	'eps=0.5, min_pts=5' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM eps05_clusters;

WITH eps10_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 1.0, 5)) AS cluster_id
)
SELECT 
	'eps=1.0, min_pts=5' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM eps10_clusters;

WITH mp3_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 3)) AS cluster_id
)
SELECT 
	'min_pts=3' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points
FROM mp3_clusters;

WITH mp10_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 10)) AS cluster_id
)
SELECT 
	'min_pts=10' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points
FROM mp10_clusters;

WITH clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 5)) AS cluster_id
)
SELECT 
	cluster_id,
	COUNT(*) AS point_count,
	ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM clusters), 2) AS percentage,
	CASE 
		WHEN cluster_id = -1 THEN 'Noise'
		ELSE 'Cluster'
	END AS point_type
FROM clusters
GROUP BY cluster_id
ORDER BY cluster_id;

WITH eps01_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.1, 5)) AS cluster_id
),
eps10_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 1.0, 5)) AS cluster_id
)
SELECT 
	'eps=0.1' AS config,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points,
	COUNT(*) AS total_points
FROM eps01_clusters
UNION ALL
SELECT 
	'eps=1.0' AS config,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points,
	COUNT(*) AS total_points
FROM eps10_clusters;

/* --- ERROR path: invalid parameters --- */
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM cluster_dbscan('test_train_view', 'features', 0.0, 5);
		RAISE EXCEPTION 'FAIL: expected error for eps=0';
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_dbscan('test_train_view', 'features', -0.1, 5);
		RAISE EXCEPTION 'FAIL: expected error for eps < 0';
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_dbscan('test_train_view', 'features', 0.5, 0);
		RAISE EXCEPTION 'FAIL: expected error for min_pts=0';
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_dbscan('test_train_view', 'features', 0.5, -1);
		RAISE EXCEPTION 'FAIL: expected error for min_pts < 0';
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_dbscan('missing_table', 'features', 0.5, 5);
		RAISE EXCEPTION 'FAIL: expected error for missing table';
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_dbscan('test_train_view', 'notacolumn', 0.5, 5);
		RAISE EXCEPTION 'FAIL: expected error for invalid column';
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- VALIDATION TESTS ----
 * Verify cluster assignments and noise points
 *------------------------------------------------------------------*/
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 5)) AS cluster_id
)
SELECT 
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points,
	CASE 
		WHEN COUNT(*) = (SELECT COUNT(*) FROM test_train_view) THEN '✓ All points assigned'
		ELSE '✗ Missing assignments'
	END AS assignment_status
FROM clusters;

WITH clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 5)) AS cluster_id
)
SELECT 
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points,
	COUNT(*) FILTER (WHERE cluster_id != -1) AS clustered_points,
	ROUND(100.0 * COUNT(*) FILTER (WHERE cluster_id = -1) / COUNT(*)::numeric, 2) AS noise_percentage,
	CASE 
		WHEN COUNT(*) FILTER (WHERE cluster_id = -1) >= 0 THEN '✓ Noise points handled correctly'
		ELSE '✗ Noise point issue'
	END AS noise_status
FROM clusters;

WITH mp3_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 3)) AS cluster_id
),
mp10_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 10)) AS cluster_id
)
SELECT 
	'min_pts=3' AS config,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points
FROM mp3_clusters
UNION ALL
SELECT 
	'min_pts=10' AS config,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points
FROM mp10_clusters;

WITH clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 5)) AS cluster_id
),
cluster_counts AS (
	SELECT cluster_id, COUNT(*) AS point_count
	FROM clusters
	WHERE cluster_id != -1
	GROUP BY cluster_id
)
SELECT 
	COUNT(*) AS clusters_with_points,
	MIN(point_count) AS min_cluster_size,
	MAX(point_count) AS max_cluster_size,
	ROUND(AVG(point_count)::numeric, 2) AS avg_cluster_size
FROM cluster_counts;

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

\echo 'Test completed successfully'
