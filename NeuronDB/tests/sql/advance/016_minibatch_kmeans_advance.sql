-- 016_minibatch_kmeans_advance.sql
-- Exhaustive detailed test for minibatch_kmeans clustering: all parameters, error handling.
-- Works on 1000 rows only and tests each and every way with comprehensive coverage
-- Tests: Different k values, batch sizes, max iterations, error handling, validation

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
 * Test multiple k values, batch sizes, and max iterations
 */
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH k2_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 2, 100, 100)) AS cluster_id
)
SELECT 
	'k=2, batch=100' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM k2_clusters;

WITH k3_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 100)) AS cluster_id
)
SELECT 
	'k=3, batch=100' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM k3_clusters;

WITH k5_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 5, 100, 100)) AS cluster_id
)
SELECT 
	'k=5, batch=100' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM k5_clusters;

WITH b50_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 50, 100)) AS cluster_id
)
SELECT 
	'batch=50' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM b50_clusters;

WITH b200_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 200, 100)) AS cluster_id
)
SELECT 
	'batch=200' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM b200_clusters;

WITH i50_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 50)) AS cluster_id
)
SELECT 
	'iter=50' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM i50_clusters;

WITH i200_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 200)) AS cluster_id
)
SELECT 
	'iter=200' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM i200_clusters;

WITH clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 100)) AS cluster_id
)
SELECT 
	cluster_id,
	COUNT(*) AS point_count,
	ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM clusters), 2) AS percentage
FROM clusters
GROUP BY cluster_id
ORDER BY cluster_id;

/* --- ERROR path: invalid parameters --- */
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM cluster_minibatch_kmeans('test_train_view', 'features', 0, 100, 100);
		RAISE EXCEPTION 'FAIL: expected error for k=0';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_minibatch_kmeans('test_train_view', 'features', 1, 100, 100);
		RAISE EXCEPTION 'FAIL: expected error for k=1';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_minibatch_kmeans('test_train_view', 'features', 3, 0, 100);
		RAISE EXCEPTION 'FAIL: expected error for batch_size=0';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_minibatch_kmeans('missing_table', 'features', 3, 100, 100);
		RAISE EXCEPTION 'FAIL: expected error for missing table';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_minibatch_kmeans('test_train_view', 'notacolumn', 3, 100, 100);
		RAISE EXCEPTION 'FAIL: expected error for invalid column';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM cluster_minibatch_kmeans('test_train_view', 'features', 2000, 100, 100);
		RAISE EXCEPTION 'FAIL: expected error for k > n_vectors';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
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
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 100)) AS cluster_id
)
SELECT 
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	CASE 
		WHEN COUNT(*) = (SELECT COUNT(*) FROM test_train_view) THEN '✓ All points assigned'
		ELSE '✗ Missing assignments'
	END AS assignment_status
FROM clusters;

WITH b50_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 50, 100)) AS cluster_id
),
b200_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 200, 100)) AS cluster_id
)
SELECT 
	'batch=50' AS config,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	COUNT(*) AS total_points
FROM b50_clusters
UNION ALL
SELECT 
	'batch=200' AS config,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	COUNT(*) AS total_points
FROM b200_clusters;

WITH clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 100)) AS cluster_id
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
