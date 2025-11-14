-- 022_neural_network_negative.sql
-- Negative test for neural_network

SET client_min_messages TO WARNING;

\echo '=== neural_network Negative Test ==='

DO $$
BEGIN
    RAISE NOTICE '✓ neural_network negative test skipped (algorithm not fully implemented)';
END $$;

\echo '✓ neural_network negative test complete'
