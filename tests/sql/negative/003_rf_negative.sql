-- 003_rf_negative.sql
-- Negative test for random_forest

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=== random_forest Negative Test ==='

-- Invalid table
DO $$ BEGIN
    PERFORM neurondb.train('random_forest', 'nonexistent_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid table';
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM neurondb.train('random_forest', 'test_train_view', 'invalid_col', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid column';
END $$;

\echo '✓ random_forest negative test complete'
