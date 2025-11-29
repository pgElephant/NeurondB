-- 004_svm_negative.sql
-- Negative test for svm

SET client_min_messages TO WARNING;


-- Invalid table
DO $$ BEGIN
    PERFORM neurondb.train('svm', 'nonexistent_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM neurondb.train('svm', 'test_train_view', 'invalid_col', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

\echo 'Test completed successfully'
