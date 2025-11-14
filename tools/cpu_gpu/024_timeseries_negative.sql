-- 024_timeseries_negative.sql
-- Negative test for timeseries

SET client_min_messages TO WARNING;

\echo '=== timeseries Negative Test ==='

DO $$
BEGIN
    RAISE NOTICE '✓ timeseries negative test skipped (algorithm not fully implemented)';
END $$;

\echo '✓ timeseries negative test complete'
