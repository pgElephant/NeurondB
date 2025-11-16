-- 004_svm_advance.sql
-- Advanced test for svm

SET client_min_messages TO WARNING;

\echo '=== svm Advanced Test ==='

-- GPU Training
SET neurondb.gpu_enabled = on;
SELECT neurondb.train('svm', 'sample_train', 'features', 'label', '{}'::jsonb)::integer AS gpu_model;

-- CPU Training
SET neurondb.gpu_enabled = off;
SELECT neurondb.train('svm', 'sample_train', 'features', 'label', '{}'::jsonb)::integer AS cpu_model;

-- Verify models exist
SELECT COUNT(*) AS model_count FROM neurondb.ml_models WHERE algorithm::text = 'svm';

\echo '✓ svm advance test complete'
