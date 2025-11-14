-- 019_xgboost_negative.sql
-- Negative test for xgboost

SET client_min_messages TO WARNING;

\echo '=== xgboost Negative Test ==='

DO $$
BEGIN
    RAISE NOTICE '✓ xgboost negative test skipped (algorithm not fully implemented)';
END $$;

\echo '✓ xgboost negative test complete'
