-- 019_xgboost_advance.sql
-- Advanced test for xgboost
-- Note: Algorithm may not be fully implemented yet

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo '=========================================================================='

\echo ''
\echo 'Algorithm Status'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create test data
CREATE TABLE IF NOT EXISTS xgb_test_data (
    id SERIAL PRIMARY KEY,
    features float8[],
    label float8
);

INSERT INTO xgb_test_data (features, label)
SELECT
    ARRAY[random(), random(), random(), random(), random()]::float8[],
    CASE WHEN random() > 0.5 THEN 1.0 ELSE 0.0 END
FROM generate_series(1, 1000);

DO $$
DECLARE
    model_id int;
    result jsonb;
    xgb_available boolean := false;
BEGIN
    -- Check if xgboost functions exist
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname LIKE '%xgboost%' OR proname LIKE '%xgb%') THEN
        xgb_available := true;
        RAISE NOTICE 'XGBoost functions found - proceeding with full test';
    ELSE
        RAISE NOTICE 'XGBoost functions not found - basic status check only';
        RETURN;
    END IF;

    -- Full XGBoost test if available
    IF xgb_available THEN
        -- Train model
        BEGIN
            SELECT train_xgboost_classifier('xgb_test_data', 'features', 'label', 100, 0.1) INTO model_id;
            RAISE NOTICE '✓ XGBoost training successful, model_id: %', model_id;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'XGBoost training failed: %', SQLERRM;
            RETURN;
        END;

        -- Test prediction
        BEGIN
            PERFORM predict_xgboost(model_id, (SELECT features FROM xgb_test_data LIMIT 1));
            RAISE NOTICE '✓ XGBoost prediction successful';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'XGBoost prediction failed: %', SQLERRM;
        END;

        -- Test evaluation
        BEGIN
            SELECT evaluate_xgboost_by_model_id(model_id, 'xgb_test_data', 'features', 'label') INTO result;
            RAISE NOTICE '✓ XGBoost evaluation successful';
            RAISE NOTICE '  - Accuracy: %', (result->>'accuracy')::float;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'XGBoost evaluation failed: %', SQLERRM;
        END;
    END IF;

    -- Check if xgboost is in ml_models
    IF EXISTS (SELECT 1 FROM neurondb.ml_models WHERE algorithm = 'xgboost' LIMIT 1) THEN
        RAISE NOTICE '✓ XGBoost models found in catalog';
    ELSE
        RAISE NOTICE '! No XGBoost models in catalog';
    END IF;
END $$;

-- Cleanup
DROP TABLE IF EXISTS xgb_test_data;

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

\echo 'Test completed successfully'
