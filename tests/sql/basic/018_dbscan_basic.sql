-- 018_dbscan_basic.sql
-- Basic test for DBSCAN clustering with GPU acceleration

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off
SET client_min_messages TO WARNING;

/* Step 1: Verify prerequisites and create views with 1000 rows */
\echo 'Step 1: Verifying prerequisites and creating test views...'

DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables 
		WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist';
	END IF;
END
$$;

-- Create views with 1000 rows for basic tests
DROP VIEW IF EXISTS test_train_view;
DROP VIEW IF EXISTS test_test_view;

CREATE VIEW test_train_view AS
SELECT features, label FROM sample_train LIMIT 1000;

CREATE VIEW test_test_view AS
SELECT features, label FROM sample_test LIMIT 1000;

SELECT 
	(SELECT COUNT(*)::bigint FROM test_train_view) AS train_rows,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_rows;

/* Step 2: Configure GPU */
\echo 'Step 2: Configuring GPU acceleration...'

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip';
SELECT neurondb_gpu_enable() AS gpu_available;

\echo '=========================================================================='
\echo 'DBSCAN Clustering - Basic Test'
\echo '=========================================================================='

/* Step 3: Test DBSCAN clustering */
\echo 'Step 3: Testing DBSCAN clustering (eps=0.5, min_pts=5)...'

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

\echo 'DBSCAN basic test completed successfully'
