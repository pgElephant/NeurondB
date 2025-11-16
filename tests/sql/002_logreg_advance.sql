-- 002_logreg_advance.sql
-- Advanced test for logistic_regression

SET client_min_messages TO WARNING;

\echo '=== logistic_regression Advanced Test ==='

-- GPU Training
SET neurondb.gpu_enabled = on;
SELECT neurondb.train('logistic_regression', 'sample_train', 'features', 'label', '{}'::jsonb)::integer AS gpu_model;

-- CPU Training
SET neurondb.gpu_enabled = off;
SELECT neurondb.train('logistic_regression', 'sample_train', 'features', 'label', '{}'::jsonb)::integer AS cpu_model;

-- Verify models exist
SELECT COUNT(*) AS model_count FROM neurondb.ml_models WHERE algorithm::text = 'logistic_regression';

\echo '✓ logistic_regression advance test complete'
