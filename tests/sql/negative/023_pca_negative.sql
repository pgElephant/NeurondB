-- 023_pca_negative.sql
-- Negative test for PCA (Principal Component Analysis)
-- All possible negative tests with 1000 rows only

\timing on
\pset footer off
\pset pager off
\pset tuples_only off
\set ON_ERROR_STOP off
SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=========================================================================='
\echo 'PCA - Negative Test Cases (Error Handling)'
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

-- Create test data
DROP TABLE IF EXISTS pca_test_data;
CREATE TEMP TABLE pca_test_data AS
SELECT features[1:5]::vector(5) AS feat_5d
FROM test_train_view
LIMIT 100;

\echo ''
\echo 'Test 1: Invalid Table Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('nonexistent_table', ARRAY['feat_5d'], 2);

\echo ''
\echo 'Test 2: Invalid Column Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', ARRAY['invalid_col'], 2);

\echo ''
\echo 'Test 3: Invalid n_components (n_components < 1)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 0);

\echo ''
\echo 'Test 4: Invalid n_components (n_components < 0)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], -1);

\echo ''
\echo 'Test 5: NULL Table Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca(NULL, ARRAY['feat_5d'], 2);

\echo ''
\echo 'Test 6: NULL Column Array'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', NULL, 2);

\echo ''
\echo 'Test 7: Empty Column Array'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', ARRAY[]::text[], 2);

\echo ''
\echo 'Test 8: NULL n_components'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], NULL);

\echo ''
\echo 'Test 9: n_components > input dimensions (may be handled gracefully)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 10);

DROP TABLE IF EXISTS pca_test_data;

\echo ''
\echo '✓ PCA negative test complete'
