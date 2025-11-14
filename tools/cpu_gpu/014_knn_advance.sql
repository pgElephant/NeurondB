\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'K-Nearest Neighbors - Advanced Features Test'
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
SET neurondb.gpu_kernels = 'l2,cosine,ip,knn_train,knn_predict';
SELECT neurondb_gpu_enable();

\echo ''
\echo 'Test 1: Training with Different K Values'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Train with k=3
DROP TABLE IF EXISTS knn_k3;
CREATE TEMP TABLE knn_k3 AS
SELECT neurondb.train(
	'knn',
	'sample_train',
	'features',
	'label',
	'{"k": 3}'::jsonb
)::integer AS model_id;

-- Train with k=5
DROP TABLE IF EXISTS knn_k5;
CREATE TEMP TABLE knn_k5 AS
SELECT neurondb.train(
	'knn',
	'sample_train',
	'features',
	'label',
	'{"k": 5}'::jsonb
)::integer AS model_id;

-- Train with k=7
DROP TABLE IF EXISTS knn_k7;
CREATE TEMP TABLE knn_k7 AS
SELECT neurondb.train(
	'knn',
	'sample_train',
	'features',
	'label',
	'{"k": 7}'::jsonb
)::integer AS model_id;

SELECT 
	'k=3' AS k_value,
	model_id AS model_id_k3
FROM knn_k3
UNION ALL
SELECT 
	'k=5' AS k_value,
	model_id AS model_id_k5
FROM knn_k5
UNION ALL
SELECT 
	'k=7' AS k_value,
	model_id AS model_id_k7
FROM knn_k7;

\echo ''
\echo 'Test 2: Model Metadata and Storage Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	m.model_id,
	m.algorithm,
	m.created_at,
	m.metrics->>'storage' AS storage_type,
	m.metrics->>'k' AS k_value,
	m.metrics->>'accuracy' AS accuracy,
	CASE 
		WHEN m.model_data IS NULL THEN 'NULL (GPU model)'
		ELSE format('%s bytes', pg_column_size(m.model_data))
	END AS model_data_status
FROM neurondb.ml_models m, knn_k5 t
WHERE m.model_id = t.model_id;

\echo ''
\echo 'Test 3: K Value Comparison'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Compare accuracy for different k values
SELECT 
	'k=3' AS k_value,
	ROUND((m.metrics->>'accuracy')::numeric, 4) AS accuracy
FROM neurondb.ml_models m, knn_k3 g
WHERE m.model_id = g.model_id
UNION ALL
SELECT 
	'k=5' AS k_value,
	ROUND((m.metrics->>'accuracy')::numeric, 4) AS accuracy
FROM neurondb.ml_models m, knn_k5 g
WHERE m.model_id = g.model_id
UNION ALL
SELECT 
	'k=7' AS k_value,
	ROUND((m.metrics->>'accuracy')::numeric, 4) AS accuracy
FROM neurondb.ml_models m, knn_k7 g
WHERE m.model_id = g.model_id
ORDER BY k_value;

\echo ''
\echo 'Test 4: Batch Prediction Performance'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Batch prediction
SELECT 
	COUNT(*) AS total_predictions,
	AVG(neurondb.predict(m.model_id, features)) AS avg_prediction,
	MIN(neurondb.predict(m.model_id, features)) AS min_prediction,
	MAX(neurondb.predict(m.model_id, features)) AS max_prediction,
	STDDEV(neurondb.predict(m.model_id, features)) AS stddev_prediction
FROM sample_test, knn_k5 m;

\echo ''
\echo 'Test 5: Prediction Distribution Analysis'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH predictions AS (
	SELECT 
		neurondb.predict(m.model_id, features) AS prediction,
		label AS actual
	FROM sample_test, knn_k5 m
)
SELECT 
	prediction::int AS predicted_class,
	COUNT(*) AS count,
	AVG(actual) AS avg_actual,
	SUM(CASE WHEN prediction::int = actual::int THEN 1 ELSE 0 END) AS correct_predictions
FROM predictions
GROUP BY prediction::int
ORDER BY predicted_class;

\echo ''
\echo 'Test 6: Model Comparison (GPU vs CPU)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Train GPU model
DROP TABLE IF EXISTS gpu_model_adv;
CREATE TEMP TABLE gpu_model_adv AS
SELECT neurondb.train(
	'knn',
	'sample_train',
	'features',
	'label',
	'{"k": 5, "storage": "gpu"}'::jsonb
)::integer AS model_id;

-- Train CPU model (if possible)
DROP TABLE IF EXISTS cpu_model_adv;
CREATE TEMP TABLE cpu_model_adv AS
SELECT neurondb.train(
	'knn',
	'sample_train',
	'features',
	'label',
	'{"k": 5, "storage": "cpu"}'::jsonb
)::integer AS model_id;

-- Compare metrics
SELECT 
	'GPU Model' AS model_type,
	ROUND((m.metrics->>'accuracy')::numeric, 4) AS accuracy
FROM neurondb.ml_models m, gpu_model_adv g
WHERE m.model_id = g.model_id
UNION ALL
SELECT 
	'CPU Model' AS model_type,
	ROUND((m.metrics->>'accuracy')::numeric, 4) AS accuracy
FROM neurondb.ml_models m, cpu_model_adv c
WHERE m.model_id = c.model_id;

\echo ''
\echo 'Test 7: Model Persistence and Retrieval'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Check models are persisted
SELECT 
	COUNT(*) AS total_models,
	COUNT(DISTINCT (metrics->>'k')::int) AS unique_k_values,
	MIN(created_at) AS oldest_model,
	MAX(created_at) AS newest_model
FROM neurondb.ml_models
WHERE algorithm = 'knn';

-- Cleanup
DROP TABLE IF EXISTS knn_k3;
DROP TABLE IF EXISTS knn_k5;
DROP TABLE IF EXISTS knn_k7;
DROP TABLE IF EXISTS gpu_model_adv;
DROP TABLE IF EXISTS cpu_model_adv;

\echo ''
\echo 'Advanced KNN Test Complete!'
\echo '=========================================================================='

