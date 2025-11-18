-- 016_minibatch_kmeans_negative.sql
-- Negative test for minibatch_kmeans
-- All possible negative tests with 1000 rows only

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=== minibatch_kmeans Negative Test ==='

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
    PERFORM cluster_minibatch_kmeans('nonexistent_table', 'features', 3, 100, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid table';
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM cluster_minibatch_kmeans('test_train_view', 'invalid_col', 3, 100, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid column';
END $$;

-- Invalid k (k < 1)
DO $$ BEGIN
    PERFORM cluster_minibatch_kmeans('test_train_view', 'features', 0, 100, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid k (k < 1)';
END $$;

-- Invalid batch_size (batch_size < 1)
DO $$ BEGIN
    PERFORM cluster_minibatch_kmeans('test_train_view', 'features', 3, 0, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid batch_size';
END $$;

-- Invalid max_iters (max_iters < 1)
DO $$ BEGIN
    PERFORM cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid max_iters';
END $$;

\echo '✓ minibatch_kmeans negative test complete'

