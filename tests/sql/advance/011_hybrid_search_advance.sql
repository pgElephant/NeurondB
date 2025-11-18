-- 011_hybrid_search_advance.sql
-- Advanced test for hybrid_search

SET client_min_messages TO WARNING;

\echo '=== hybrid_search Advanced Test ==='

-- GPU Training
SET neurondb.gpu_enabled = on;
SELECT neurondb.train('default', 'hybrid_search', 'test_train_view', 'features', 'label', '{}'::jsonb)::integer AS gpu_model;

-- CPU Training
SET neurondb.gpu_enabled = off;
SELECT neurondb.train('default', 'hybrid_search', 'test_train_view', 'features', 'label', '{}'::jsonb)::integer AS cpu_model;

-- Verify models exist
SELECT COUNT(*) AS model_count FROM neurondb.ml_models WHERE algorithm::text = 'hybrid_search';

\echo '✓ hybrid_search advance test complete'
