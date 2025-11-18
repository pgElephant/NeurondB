-- 018_dbscan_negative.sql
-- Negative test for dbscan
-- All possible negative tests with 1000 rows only

\timing on
\pset footer off
\pset pager off
\pset tuples_only off
\set ON_ERROR_STOP off
SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=========================================================================='
\echo 'DBSCAN - Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/* Setup: Create test views if they don't exist */
DO $$
BEGIN
	IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		DROP VIEW IF EXISTS test_train_view;
		CREATE VIEW test_train_view AS
		SELECT features, label FROM sample_train LIMIT 1000;
	END IF;
	IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_test') THEN
		DROP VIEW IF EXISTS test_test_view;
		CREATE VIEW test_test_view AS
		SELECT features, label FROM sample_test LIMIT 1000;
	END IF;
END
$$;

\echo ''
\echo 'Test 1: Invalid Table Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT cluster_dbscan('nonexistent_table', 'features', 0.5, 5);

\echo ''
\echo 'Test 2: Invalid Column Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT cluster_dbscan('test_train_view', 'invalid_col', 0.5, 5);

\echo ''
\echo 'Test 3: Invalid eps (eps < 0)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT cluster_dbscan('test_train_view', 'features', -0.1, 5);

\echo ''
\echo 'Test 4: Invalid eps (eps = 0)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT cluster_dbscan('test_train_view', 'features', 0.0, 5);

\echo ''
\echo 'Test 5: Invalid min_pts (min_pts < 1)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT cluster_dbscan('test_train_view', 'features', 0.5, 0);

\echo ''
\echo 'Test 6: Invalid min_pts (min_pts < 0)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT cluster_dbscan('test_train_view', 'features', 0.5, -1);

\echo ''
\echo 'Test 7: NULL Table Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT cluster_dbscan(NULL, 'features', 0.5, 5);

\echo ''
\echo 'Test 8: NULL Column Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT cluster_dbscan('test_train_view', NULL, 0.5, 5);

\echo ''
\echo 'Test 9: NULL eps'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT cluster_dbscan('test_train_view', 'features', NULL, 5);

\echo ''
\echo 'Test 10: NULL min_pts'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT cluster_dbscan('test_train_view', 'features', 0.5, NULL);

\echo ''
\echo '✓ dbscan negative test complete'
