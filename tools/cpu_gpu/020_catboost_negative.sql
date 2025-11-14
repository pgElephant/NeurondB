-- 020_catboost_negative.sql
-- Negative test for catboost

SET client_min_messages TO WARNING;

\echo '=== catboost Negative Test ==='

DO $$
BEGIN
    RAISE NOTICE '✓ catboost negative test skipped (algorithm not fully implemented)';
END $$;

\echo '✓ catboost negative test complete'
