-- 005_dt_advance.sql
-- Advanced test for decision_tree

SET client_min_messages TO WARNING;

\echo '=== decision_tree Advanced Test ==='

-- GPU Training
SET neurondb.gpu_enabled = on;
SELECT neurondb.train('decision_tree', 'sample_train', 'features', 'label', '{}'::jsonb)::integer AS gpu_model;

-- CPU Training
SET neurondb.gpu_enabled = off;
SELECT neurondb.train('decision_tree', 'sample_train', 'features', 'label', '{}'::jsonb)::integer AS cpu_model;

-- Verify models exist
SELECT COUNT(*) AS model_count FROM neurondb.ml_models WHERE algorithm::text = 'decision_tree';

\echo '✓ decision_tree advance test complete'
