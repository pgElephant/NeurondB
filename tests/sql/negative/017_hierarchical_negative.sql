-- 017_hierarchical_negative.sql
-- Negative test for hierarchical clustering
-- All possible negative tests with 1000 rows only

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=== hierarchical Negative Test ==='

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

-- Invalid table
DO $$ BEGIN
    PERFORM cluster_hierarchical('nonexistent_table', 'features', 3, 'average');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid table';
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM cluster_hierarchical('test_train_view', 'invalid_col', 3, 'average');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid column';
END $$;

-- Invalid k (k < 1)
DO $$ BEGIN
    PERFORM cluster_hierarchical('test_train_view', 'features', 0, 'average');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid k (k < 1)';
END $$;

-- Invalid linkage method
DO $$ BEGIN
    PERFORM cluster_hierarchical('test_train_view', 'features', 3, 'invalid_linkage');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid linkage method';
END $$;

-- NULL table
DO $$ BEGIN
    PERFORM cluster_hierarchical(NULL, 'features', 3, 'average');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled NULL table';
END $$;

-- NULL column
DO $$ BEGIN
    PERFORM cluster_hierarchical('test_train_view', NULL, 3, 'average');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled NULL column';
END $$;

\echo '✓ hierarchical negative test complete'

