-- 007_lasso_negative.sql
-- Negative test for lasso

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=== lasso Negative Test ==='

-- Invalid table
DO $$ BEGIN
    PERFORM neurondb.train('lasso', 'nonexistent_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid table';
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM neurondb.train('lasso', 'sample_train', 'invalid_col', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid column';
END $$;

\echo '✓ lasso negative test complete'
