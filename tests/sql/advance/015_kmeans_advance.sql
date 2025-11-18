-- 015_kmeans_advance.sql
-- Advanced test for kmeans
-- Works on 1000 rows only and tests each and every way

SET client_min_messages TO WARNING;

\echo '=== kmeans Advanced Test ==='

-- Verify required tables exist
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

SET neurondb.gpu_enabled = on;
SELECT neurondb_gpu_enable();

\echo 'Test 1: Different k values'
WITH k2_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 2, 100)) AS cluster_id
)
SELECT 'k=2' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM k2_clusters;
WITH k3_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 100)) AS cluster_id
)
SELECT 'k=3' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM k3_clusters;
WITH k5_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 5, 100)) AS cluster_id
)
SELECT 'k=5' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM k5_clusters;

\echo 'Test 2: Different max iterations'
WITH iter50_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 50)) AS cluster_id
)
SELECT 'iter=50' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM iter50_clusters;
WITH iter100_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 100)) AS cluster_id
)
SELECT 'iter=100' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM iter100_clusters;
WITH iter200_clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 200)) AS cluster_id
)
SELECT 'iter=200' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM iter200_clusters;

\echo 'Test 3: Verify cluster assignments'
WITH clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 100)) AS cluster_id
)
SELECT 
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM clusters;

\echo 'Test 4: Cluster distribution'
WITH clusters AS (
	SELECT unnest(cluster_kmeans('test_train_view', 'features', 3, 100)) AS cluster_id
)
SELECT 
	cluster_id,
	COUNT(*) AS point_count
FROM clusters
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '✓ kmeans advance test complete'

