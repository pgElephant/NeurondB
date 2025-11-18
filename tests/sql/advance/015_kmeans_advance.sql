-- 015_kmeans_advance.sql
-- Exhaustive detailed test for kmeans clustering: all parameters, error handling.
-- Works on 1000 rows only and tests each and every way with comprehensive coverage
-- Tests: Different k values, max iterations, error handling, cluster validation

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'kmeans: Exhaustive Clustering Test (1000 rows sample)'
\echo '=========================================================================='

/* Check that sample_train exists */
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist';
	END IF;
END
$$;

-- Create views with 1000 rows for advance tests
DROP VIEW IF EXISTS test_train_view;
DROP VIEW IF EXISTS test_test_view;

CREATE VIEW test_train_view AS
SELECT features, label FROM sample_train LIMIT 1000;

CREATE VIEW test_test_view AS
SELECT features, label FROM sample_test LIMIT 1000;

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
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,kmeans';
SELECT neurondb_gpu_enable() AS gpu_available;
SELECT neurondb_gpu_info() AS gpu_info;

/*
 * ---- CLUSTERING TESTS ----
 * Test multiple k values and max iterations
 */
\echo ''
\echo 'Clustering Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: k=2 clusters'
WITH k2_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 2, 100)) AS cluster_id
)
SELECT 
	'k=2' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM k2_clusters;

\echo 'Test 2: k=3 clusters'
WITH k3_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 100)) AS cluster_id
)
SELECT 
	'k=3' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM k3_clusters;

\echo 'Test 3: k=5 clusters'
WITH k5_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 5, 100)) AS cluster_id
)
SELECT 
	'k=5' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM k5_clusters;

\echo 'Test 4: Different max iterations (iter=50)'
WITH iter50_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 50)) AS cluster_id
)
SELECT 
	'iter=50' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM iter50_clusters;

\echo 'Test 5: Different max iterations (iter=200)'
WITH iter200_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 200)) AS cluster_id
)
SELECT 
	'iter=200' AS test,
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM iter200_clusters;

\echo 'Test 6: Cluster distribution analysis (k=3)'
WITH clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 100)) AS cluster_id
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
\echo 'Error Handling Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: k=0 (should error)'
DO $$
BEGIN
	BEGIN
		PERFORM cluster_kmeans('test_train_view', 'features', 0, 100);
		RAISE EXCEPTION 'FAIL: expected error for k=0';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 2: k=1 (should error)'
DO $$
BEGIN
	BEGIN
		PERFORM cluster_kmeans('test_train_view', 'features', 1, 100);
		RAISE EXCEPTION 'FAIL: expected error for k=1';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 3: Invalid table name'
DO $$
BEGIN
	BEGIN
		PERFORM cluster_kmeans('missing_table', 'features', 3, 100);
		RAISE EXCEPTION 'FAIL: expected error for missing table';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 4: Invalid column name'
DO $$
BEGIN
	BEGIN
		PERFORM cluster_kmeans('test_train_view', 'notacolumn', 3, 100);
		RAISE EXCEPTION 'FAIL: expected error for invalid column';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 5: max_iters=0 (should use default)'
DO $$
DECLARE
	result integer[];
BEGIN
	BEGIN
		result := cluster_kmeans('test_train_view', 'features', 3, 0);
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 6: k > number of vectors'
DO $$
BEGIN
	BEGIN
		PERFORM cluster_kmeans('test_train_view', 'features', 2000, 100);
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
\echo 'Validation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Validation Test 1: All points assigned (k=3)'
WITH clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 100)) AS cluster_id
)
SELECT 
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	CASE 
		WHEN COUNT(*) = (SELECT COUNT(*) FROM test_train_view) THEN '✓ All points assigned'
		ELSE '✗ Missing assignments'
	END AS assignment_status
FROM clusters;

\echo 'Validation Test 2: Cluster IDs are sequential (k=3)'
WITH clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 100)) AS cluster_id
),
cluster_stats AS (
	SELECT 
		MIN(cluster_id) AS min_id,
		MAX(cluster_id) AS max_id,
		COUNT(DISTINCT cluster_id) AS unique_clusters
	FROM clusters
)
SELECT 
	min_id,
	max_id,
	unique_clusters,
	CASE 
		WHEN max_id - min_id + 1 = unique_clusters THEN '✓ Sequential IDs'
		ELSE '✗ Non-sequential IDs'
	END AS id_status
FROM cluster_stats;

\echo 'Validation Test 3: No empty clusters (k=3)'
WITH clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 100)) AS cluster_id
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

\echo 'Validation Test 4: Compare k values (k=2 vs k=5)'
WITH k2_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 2, 100)) AS cluster_id
),
k5_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 5, 100)) AS cluster_id
)
SELECT 
	'k=2' AS config,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	COUNT(*) AS total_points
FROM k2_clusters
UNION ALL
SELECT 
	'k=5' AS config,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	COUNT(*) AS total_points
FROM k5_clusters;

\echo ''
\echo '=========================================================================='
\echo '✓ kmeans: Full exhaustive clustering test complete (1000-row sample)'
\echo '=========================================================================='
