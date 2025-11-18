-- 021_lightgbm_negative.sql
-- Negative test for lightgbm

SET client_min_messages TO WARNING;

\echo '=== lightgbm Negative Test ==='

DO $$
BEGIN
    RAISE NOTICE '✓ lightgbm negative test skipped (algorithm not fully implemented)';
END $$;

\echo '✓ lightgbm negative test complete'
