-- 017_hierarchical_advance.sql
-- Advanced test for hierarchical clustering
-- Works on 1000 rows only and tests each and every way

SET client_min_messages TO WARNING;

\echo '=== hierarchical Advanced Test ==='

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
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 2, 'average')) AS cluster_id
)
SELECT 'k=2' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM k2_clusters;
WITH k3_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'average')) AS cluster_id
)
SELECT 'k=3' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM k3_clusters;
WITH k5_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 5, 'average')) AS cluster_id
)
SELECT 'k=5' AS test, COUNT(DISTINCT cluster_id) AS num_clusters FROM k5_clusters;

\echo 'Test 2: Different linkage methods'
WITH avg_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'average')) AS cluster_id
)
SELECT 'average' AS linkage, COUNT(DISTINCT cluster_id) AS num_clusters FROM avg_clusters;
WITH comp_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'complete')) AS cluster_id
)
SELECT 'complete' AS linkage, COUNT(DISTINCT cluster_id) AS num_clusters FROM comp_clusters;
WITH single_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'single')) AS cluster_id
)
SELECT 'single' AS linkage, COUNT(DISTINCT cluster_id) AS num_clusters FROM single_clusters;

\echo 'Test 3: Verify cluster assignments'
WITH clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'average')) AS cluster_id
)
SELECT 
	COUNT(*) AS total_points,
	COUNT(DISTINCT cluster_id) AS num_clusters,
	MIN(cluster_id) AS min_cluster,
	MAX(cluster_id) AS max_cluster
FROM clusters;

\echo 'Test 4: Compare linkage methods'
WITH avg_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'average')) AS cluster_id
),
comp_clusters AS (
	SELECT unnest(cluster_hierarchical('test_train_view', 'features', 3, 'complete')) AS cluster_id
)
SELECT 
	'average' AS linkage,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM avg_clusters
UNION ALL
SELECT 
	'complete' AS linkage,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM comp_clusters;

\echo '✓ hierarchical advance test complete'

