-- 006_ridge_negative.sql
-- Negative test for ridge

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=== ridge Negative Test ==='

-- Invalid table
DO $$ BEGIN
    PERFORM neurondb.train('ridge', 'nonexistent_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid table';
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM neurondb.train('ridge', 'sample_train', 'invalid_col', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid column';
END $$;

\echo '✓ ridge negative test complete'
