-- 008_rag_advance.sql
-- Advanced test for rag

SET client_min_messages TO WARNING;

\echo '=== rag Advanced Test ==='

-- GPU Training
SET neurondb.gpu_enabled = on;
SELECT neurondb.train('default', 'rag', 'test_train_view', 'features', 'label', '{}'::jsonb)::integer AS gpu_model;

-- CPU Training
SET neurondb.gpu_enabled = off;
SELECT neurondb.train('default', 'rag', 'test_train_view', 'features', 'label', '{}'::jsonb)::integer AS cpu_model;

-- Verify models exist
SELECT COUNT(*) AS model_count FROM neurondb.ml_models WHERE algorithm::text = 'rag';

\echo '✓ rag advance test complete'
