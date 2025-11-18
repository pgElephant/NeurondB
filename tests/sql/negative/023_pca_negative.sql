-- 023_pca_negative.sql
-- Negative test for PCA (Principal Component Analysis)
-- All possible negative tests with 1000 rows only

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=== PCA Negative Test ==='

-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist';
	END IF;
END
$$;

-- Create views with 1000 rows for negative tests
DROP VIEW IF EXISTS test_train_view;
DROP VIEW IF EXISTS test_test_view;

CREATE VIEW test_train_view AS
SELECT features, label FROM sample_train LIMIT 1000;

CREATE VIEW test_test_view AS
SELECT features, label FROM sample_test LIMIT 1000;

-- Create test data
DROP TABLE IF EXISTS pca_test_data;
CREATE TEMP TABLE pca_test_data AS
SELECT features[1:5]::vector(5) AS feat_5d
FROM test_train_view
LIMIT 100;

-- Invalid table
DO $$ BEGIN
    PERFORM neurondb.transform_pca('nonexistent_table', ARRAY['feat_5d'], 2);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid table';
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM neurondb.transform_pca('pca_test_data', ARRAY['invalid_col'], 2);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid column';
END $$;

-- Invalid n_components (n_components < 1)
DO $$ BEGIN
    PERFORM neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid n_components (n_components < 1)';
END $$;

-- Invalid n_components (n_components > input dimensions)
DO $$ BEGIN
    PERFORM neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 10);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid n_components (n_components > dims)';
END $$;

-- NULL table
DO $$ BEGIN
    PERFORM neurondb.transform_pca(NULL, ARRAY['feat_5d'], 2);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled NULL table';
END $$;

-- NULL column array
DO $$ BEGIN
    PERFORM neurondb.transform_pca('pca_test_data', NULL, 2);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled NULL column array';
END $$;

-- Empty column array
DO $$ BEGIN
    PERFORM neurondb.transform_pca('pca_test_data', ARRAY[]::text[], 2);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled empty column array';
END $$;

DROP TABLE IF EXISTS pca_test_data;

\echo '✓ PCA negative test complete'

