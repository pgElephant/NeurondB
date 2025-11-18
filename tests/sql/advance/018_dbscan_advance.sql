-- 018_dbscan_advance.sql
-- Advanced test for dbscan
-- Works on 1000 rows only and tests each and every way

SET client_min_messages TO WARNING;

\echo '=== dbscan Advanced Test ==='

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

\echo 'Test 1: Different eps values'
WITH eps01_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.1, 5)) AS cluster_id
)
SELECT 'eps=0.1' AS test, COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters FROM eps01_clusters;
WITH eps05_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 5)) AS cluster_id
)
SELECT 'eps=0.5' AS test, COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters FROM eps05_clusters;
WITH eps10_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 1.0, 5)) AS cluster_id
)
SELECT 'eps=1.0' AS test, COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters FROM eps10_clusters;

\echo 'Test 2: Different min_pts values'
WITH mp3_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 3)) AS cluster_id
)
SELECT 'min_pts=3' AS test, COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters FROM mp3_clusters;
WITH mp5_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 5)) AS cluster_id
)
SELECT 'min_pts=5' AS test, COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters FROM mp5_clusters;
WITH mp10_clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 10)) AS cluster_id
)
SELECT 'min_pts=10' AS test, COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters FROM mp10_clusters;

\echo 'Test 3: Verify cluster assignments and noise points'
WITH clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 5)) AS cluster_id
)
SELECT 
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id != -1) AS num_clusters,
	COUNT(*) FILTER (WHERE cluster_id = -1) AS noise_points,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM clusters;

\echo 'Test 4: Cluster size distribution'
WITH clusters AS (
	SELECT unnest(cluster_dbscan('test_train_view', 'features', 0.5, 5)) AS cluster_id
)
SELECT 
	cluster_id,
	COUNT(*) AS point_count
FROM clusters
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '✓ dbscan advance test complete'

