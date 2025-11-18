-- 006_ridge_advance.sql
-- Advanced test for ridge

SET client_min_messages TO WARNING;

\echo '=== ridge Advanced Test ==='

-- GPU Training
SET neurondb.gpu_enabled = on;
SELECT neurondb.train('ridge', 'test_train_view', 'features', 'label', '{}'::jsonb)::integer AS gpu_model;

-- CPU Training
SET neurondb.gpu_enabled = off;
SELECT neurondb.train('ridge', 'test_train_view', 'features', 'label', '{}'::jsonb)::integer AS cpu_model;

-- Verify models exist
SELECT COUNT(*) AS model_count FROM neurondb.ml_models WHERE algorithm::text = 'ridge';

\echo '✓ ridge advance test complete'
