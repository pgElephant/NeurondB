\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Gaussian Mixture Model - Advanced Features Test'
\echo '=========================================================================='

-- This test uses sample_train and sample_test tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_test') THEN
		RAISE EXCEPTION 'sample_test table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
END
$$;

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,gmm_train,gmm_predict';
SELECT neurondb_gpu_enable();

\echo ''
\echo 'Test 1: Training with Different Cluster Counts'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Train with k=3
DROP TABLE IF EXISTS gmm_k3;
CREATE TEMP TABLE gmm_k3 AS
SELECT neurondb.train(
	'gmm',
	'sample_train',
	'features',
	NULL,
	'{"k": 3, "max_iters": 100}'::jsonb
)::integer AS model_id;

-- Train with k=5
DROP TABLE IF EXISTS gmm_k5;
CREATE TEMP TABLE gmm_k5 AS
SELECT neurondb.train(
	'gmm',
	'sample_train',
	'features',
	NULL,
	'{"k": 5, "max_iters": 100}'::jsonb
)::integer AS model_id;

SELECT 
	'k=3' AS cluster_count,
	model_id AS model_id_k3
FROM gmm_k3
UNION ALL
SELECT 
	'k=5' AS cluster_count,
	model_id AS model_id_k5
FROM gmm_k5;

\echo ''
\echo 'Test 2: Model Metadata and Storage Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	m.model_id,
	m.algorithm,
	m.created_at,
	m.metrics->>'storage' AS storage_type,
	m.metrics->>'n_components' AS n_components,
	m.metrics->>'silhouette_score' AS silhouette_score,
	CASE 
		WHEN m.model_data IS NULL THEN 'NULL (GPU model)'
		ELSE format('%s bytes', pg_column_size(m.model_data))
	END AS model_data_status
FROM neurondb.ml_models m, gmm_k3 t
WHERE m.model_id = t.model_id;

\echo ''
\echo 'Test 3: Cluster Distribution Analysis'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Analyze cluster assignments
SELECT 
	neurondb.predict(m.model_id, features)::int AS cluster_id,
	COUNT(*) AS cluster_size,
	ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM sample_test), 2) AS percentage
FROM sample_test, gmm_k3 m
GROUP BY cluster_id
ORDER BY cluster_id;

\echo ''
\echo 'Test 4: Cluster Comparison (k=3 vs k=5)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Compare silhouette scores
SELECT 
	'k=3' AS model_type,
	ROUND((m.metrics->>'silhouette_score')::numeric, 4) AS silhouette_score,
	ROUND((m.metrics->>'inertia')::numeric, 4) AS inertia
FROM neurondb.ml_models m, gmm_k3 g
WHERE m.model_id = g.model_id
UNION ALL
SELECT 
	'k=5' AS model_type,
	ROUND((m.metrics->>'silhouette_score')::numeric, 4) AS silhouette_score,
	ROUND((m.metrics->>'inertia')::numeric, 4) AS inertia
FROM neurondb.ml_models m, gmm_k5 g
WHERE m.model_id = g.model_id;

\echo ''
\echo 'Test 5: Batch Prediction Performance'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Batch prediction
SELECT 
	COUNT(*) AS total_predictions,
	COUNT(DISTINCT neurondb.predict(m.model_id, features)::int) AS unique_clusters,
	AVG(neurondb.predict(m.model_id, features)) AS avg_cluster_id,
	MIN(neurondb.predict(m.model_id, features)) AS min_cluster_id,
	MAX(neurondb.predict(m.model_id, features)) AS max_cluster_id
FROM sample_test, gmm_k3 m;

\echo ''
\echo 'Test 6: Model Persistence and Retrieval'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Check models are persisted
SELECT 
	COUNT(*) AS total_models,
	COUNT(DISTINCT (metrics->>'n_components')::int) AS unique_k_values,
	MIN(created_at) AS oldest_model,
	MAX(created_at) AS newest_model
FROM neurondb.ml_models
WHERE algorithm = 'gmm';

-- Cleanup
DROP TABLE IF EXISTS gmm_k3;
DROP TABLE IF EXISTS gmm_k5;

\echo ''
\echo 'Advanced GMM Test Complete!'
\echo '=========================================================================='

