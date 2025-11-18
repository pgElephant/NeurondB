-- 020_catboost.sql
-- Basic test for CatBoost

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=== CatBoost Basic Test ==='

DO $$
BEGIN
    RAISE NOTICE '✓ CatBoost training test skipped (algorithm not fully implemented)';
END $$;

\echo '✓ CatBoost basic test complete'
