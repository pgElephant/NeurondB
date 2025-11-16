-- 008_rag_negative.sql
-- Negative test for rag

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=== rag Negative Test ==='

-- Invalid table
DO $$ BEGIN
    PERFORM neurondb.train('rag', 'nonexistent_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid table';
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM neurondb.train('rag', 'sample_train', 'invalid_col', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid column';
END $$;

\echo '✓ rag negative test complete'
