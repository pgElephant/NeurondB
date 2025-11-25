-- 005_dt_negative.sql
-- Negative test for decision_tree

SET client_min_messages TO WARNING;


-- Invalid table
DO $$ BEGIN
    PERFORM neurondb.train('decision_tree', 'nonexistent_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM neurondb.train('decision_tree', 'test_train_view', 'invalid_col', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

\echo 'Test completed successfully'
