-- 007_lasso_advance.sql
-- Advanced test for lasso

SET client_min_messages TO WARNING;

\echo '=== lasso Advanced Test ==='

-- GPU Training
SET neurondb.gpu_enabled = on;
SELECT neurondb.train('lasso', 'test_train_view', 'features', 'label', '{}'::jsonb)::integer AS gpu_model;

-- CPU Training
SET neurondb.gpu_enabled = off;
SELECT neurondb.train('lasso', 'test_train_view', 'features', 'label', '{}'::jsonb)::integer AS cpu_model;

-- Verify models exist
SELECT COUNT(*) AS model_count FROM neurondb.ml_models WHERE algorithm::text = 'lasso';

\echo '✓ lasso advance test complete'
