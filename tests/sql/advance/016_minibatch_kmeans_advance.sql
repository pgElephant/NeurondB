-- 016_minibatch_kmeans_advance.sql
-- Advanced test for minibatch_kmeans
-- Works on 1000 rows only and tests each and every way

SET client_min_messages TO WARNING;

\echo '=== minibatch_kmeans Advanced Test ==='

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
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 2, 100, 100)) AS cluster_id
)
SELECT 'k=2' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM k2_clusters;
WITH k3_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 100)) AS cluster_id
)
SELECT 'k=3' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM k3_clusters;
WITH k5_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 5, 100, 100)) AS cluster_id
)
SELECT 'k=5' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM k5_clusters;

\echo 'Test 2: Different batch sizes'
WITH b50_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 50, 100)) AS cluster_id
)
SELECT 'batch=50' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM b50_clusters;
WITH b100_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 100)) AS cluster_id
)
SELECT 'batch=100' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM b100_clusters;
WITH b200_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 200, 100)) AS cluster_id
)
SELECT 'batch=200' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM b200_clusters;

\echo 'Test 3: Different max iterations'
WITH i50_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 50)) AS cluster_id
)
SELECT 'iter=50' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM i50_clusters;
WITH i100_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 100)) AS cluster_id
)
SELECT 'iter=100' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM i100_clusters;
WITH i200_clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 200)) AS cluster_id
)
SELECT 'iter=200' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM i200_clusters;

\echo 'Test 4: Verify cluster assignments'
WITH clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 100)) AS cluster_id
)
SELECT 
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM clusters;

\echo '✓ minibatch_kmeans advance test complete'

