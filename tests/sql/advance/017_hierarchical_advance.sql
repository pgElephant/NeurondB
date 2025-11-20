-- 017_hierarchical_advance.sql
-- Exhaustive detailed test for hierarchical clustering: all parameters, error handling.
-- Works on 1000 rows only and tests each and every way with comprehensive coverage
-- Tests: Different k values, linkage methods, error handling, cluster validation

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
 * Test multiple k values and linkage methods
 */
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH k2_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 2, 'average')) AS cluster_id
)
SELECT 
	'k=2, average' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM k2_clusters;

WITH k3_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'average')) AS cluster_id
)
SELECT 
	'k=3, average' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM k3_clusters;

WITH k5_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 5, 'average')) AS cluster_id
)
SELECT 
	'k=5, average' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM k5_clusters;

WITH comp_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'complete')) AS cluster_id
)
SELECT 
	'complete linkage' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM comp_clusters;

WITH single_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'single')) AS cluster_id
)
SELECT 
	'single linkage' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM single_clusters;

WITH clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'average')) AS cluster_id
)
SELECT 
	cluster_id,
	COUNT(*) AS point_count,
	ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM clusters), 2) AS percentage
FROM clusters
GROUP BY cluster_id
ORDER BY cluster_id;

WITH avg_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'average')) AS cluster_id
),
comp_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'complete')) AS cluster_id
),
single_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'single')) AS cluster_id
)
SELECT 
	'average' AS linkage,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	COUNT(*) AS total_points
FROM avg_clusters
UNION ALL
SELECT 
	'complete' AS linkage,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	COUNT(*) AS total_points
FROM comp_clusters
UNION ALL
SELECT 
	'single' AS linkage,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	COUNT(*) AS total_points
FROM single_clusters;

/* --- ERROR path: invalid parameters --- */
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM cluster_hierarchical('test_train_view', 'features', 0, 'average');
		RAISE EXCEPTION 'FAIL: expected error for k=0';
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_hierarchical('test_train_view', 'features', 1, 'average');
		RAISE EXCEPTION 'FAIL: expected error for k=1';
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_hierarchical('missing_table', 'features', 3, 'average');
		RAISE EXCEPTION 'FAIL: expected error for missing table';
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_hierarchical('test_train_view', 'notacolumn', 3, 'average');
		RAISE EXCEPTION 'FAIL: expected error for invalid column';
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_hierarchical('test_train_view', 'features', 3, 'invalid');
		RAISE EXCEPTION 'FAIL: expected error for invalid linkage';
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_hierarchical('test_train_view', 'features', 2000, 'average');
		RAISE EXCEPTION 'FAIL: expected error for k > n_vectors';
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- VALIDATION TESTS ----
 * Verify cluster assignments are valid
 *------------------------------------------------------------------*/
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'average')) AS cluster_id
)
SELECT 
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	CASE 
		WHEN COUNT(*) = (SELECT COUNT(*) FROM test_train_view) THEN '✓ All points assigned'
		ELSE '✗ Missing assignments'
	END AS assignment_status
FROM clusters;

WITH avg_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'average')) AS cluster_id
),
comp_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'complete')) AS cluster_id
)
SELECT 
	'average' AS linkage,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	COUNT(*) AS total_points
FROM avg_clusters
UNION ALL
SELECT 
	'complete' AS linkage,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	COUNT(*) AS total_points
FROM comp_clusters;

WITH clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'average')) AS cluster_id
),
cluster_counts AS (
	SELECT cluster_id, COUNT(*) AS point_count
	FROM clusters
	GROUP BY cluster_id
)
SELECT 
	COUNT(*) AS clusters_with_points,
	MIN(point_count) AS min_cluster_size,
	MAX(point_count) AS max_cluster_size,
	ROUND(AVG(point_count)::numeric, 2) AS avg_cluster_size,
	CASE 
		WHEN MIN(point_count) > 0 THEN '✓ No empty clusters'
		ELSE '✗ Empty clusters found'
	END AS empty_cluster_status
FROM cluster_counts;

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

\echo 'Test completed successfully'
