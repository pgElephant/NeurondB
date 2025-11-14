-- 003_rf_advance.sql
-- Advanced test for random_forest

SET client_min_messages TO WARNING;

\echo '=== random_forest Advanced Test ==='

-- GPU Training
SET neurondb.gpu_enabled = on;
SELECT neurondb.train('random_forest', 'sample_train', 'features', 'label', '{}'::jsonb)::integer AS gpu_model;

-- CPU Training
SET neurondb.gpu_enabled = off;
SELECT neurondb.train('random_forest', 'sample_train', 'features', 'label', '{}'::jsonb)::integer AS cpu_model;

-- Verify models exist
SELECT COUNT(*) AS model_count FROM neurondb.ml_models WHERE algorithm::text = 'random_forest';

\echo '✓ random_forest advance test complete'
