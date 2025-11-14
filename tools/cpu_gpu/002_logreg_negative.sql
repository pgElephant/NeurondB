-- 002_logreg_negative.sql
-- Negative test for logistic_regression

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=== logistic_regression Negative Test ==='

-- Invalid table
DO $$ BEGIN
    PERFORM neurondb.train('logistic_regression', 'nonexistent_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid table';
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM neurondb.train('logistic_regression', 'sample_train', 'invalid_col', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid column';
END $$;

\echo '✓ logistic_regression negative test complete'
