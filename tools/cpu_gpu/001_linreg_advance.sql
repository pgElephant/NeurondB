\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Linear Regression - Advanced Features Test'
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
SET neurondb.gpu_kernels = 'l2,cosine,ip,linreg_train,linreg_predict';
SELECT neurondb_gpu_enable();

\echo ''
\echo 'Test 1: Training with Custom Hyperparameters'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Train with custom parameters
DROP TABLE IF EXISTS adv_model_temp;
CREATE TEMP TABLE adv_model_temp AS
SELECT neurondb.train(
	'linear_regression',
	'sample_train',
	'features',
	'label',
	'{"fit_intercept": true, "normalize": false}'::jsonb
)::integer AS model_id;

SELECT model_id FROM adv_model_temp;

\echo ''
\echo 'Test 2: Model Metadata and Storage Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	m.model_id,
	m.algorithm,
	m.created_at,
	m.metrics->>'storage' AS storage_type,
	m.metrics->>'r2' AS r_squared,
	m.metrics->>'mse' AS mean_squared_error,
	CASE 
		WHEN m.model_data IS NULL THEN 'NULL (GPU model)'
		ELSE format('%s bytes', pg_column_size(m.model_data))
	END AS model_data_status
FROM neurondb.ml_models m, adv_model_temp t
WHERE m.model_id = t.model_id;

\echo ''
\echo 'Test 3: Batch Prediction Performance'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Batch prediction
SELECT 
	COUNT(*) AS total_predictions,
	AVG(neurondb.predict(m.model_id, features)) AS avg_prediction,
	MIN(neurondb.predict(m.model_id, features)) AS min_prediction,
	MAX(neurondb.predict(m.model_id, features)) AS max_prediction,
	STDDEV(neurondb.predict(m.model_id, features)) AS stddev_prediction
FROM sample_test, adv_model_temp m;

\echo ''
\echo 'Test 4: Prediction Distribution Analysis'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH predictions AS (
	SELECT 
		neurondb.predict(m.model_id, features) AS prediction,
		label AS actual
	FROM sample_test, adv_model_temp m
)
SELECT 
	ROUND(prediction::numeric, 2) AS prediction_bucket,
	COUNT(*) AS count,
	AVG(actual) AS avg_actual,
	AVG(ABS(prediction - actual)) AS avg_error
FROM predictions
GROUP BY ROUND(prediction::numeric, 2)
ORDER BY prediction_bucket
LIMIT 10;

\echo ''
\echo 'Test 5: Residual Analysis'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH residuals AS (
	SELECT 
		neurondb.predict(m.model_id, features) - label AS residual
	FROM sample_test, adv_model_temp m
)
SELECT 
	AVG(residual) AS mean_residual,
	STDDEV(residual) AS stddev_residual,
	MIN(residual) AS min_residual,
	MAX(residual) AS max_residual,
	PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY residual) AS median_residual
FROM residuals;

\echo ''
\echo 'Test 6: Model Comparison (GPU vs CPU)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Train GPU model
DROP TABLE IF EXISTS gpu_model_adv;
CREATE TEMP TABLE gpu_model_adv AS
SELECT neurondb.train(
	'linear_regression',
	'sample_train',
	'features',
	'label',
	'{"storage": "gpu"}'::jsonb
)::integer AS model_id;

-- Train CPU model (if possible)
DROP TABLE IF EXISTS cpu_model_adv;
CREATE TEMP TABLE cpu_model_adv AS
SELECT neurondb.train(
	'linear_regression',
	'sample_train',
	'features',
	'label',
	'{"storage": "cpu"}'::jsonb
)::integer AS model_id;

-- Compare metrics
SELECT 
	'GPU Model' AS model_type,
	ROUND((m.metrics->>'r2')::numeric, 4) AS r_squared,
	ROUND((m.metrics->>'mse')::numeric, 4) AS mse
FROM neurondb.ml_models m, gpu_model_adv g
WHERE m.model_id = g.model_id
UNION ALL
SELECT 
	'CPU Model' AS model_type,
	ROUND((m.metrics->>'r2')::numeric, 4) AS r_squared,
	ROUND((m.metrics->>'mse')::numeric, 4) AS mse
FROM neurondb.ml_models m, cpu_model_adv c
WHERE m.model_id = c.model_id;

\echo ''
\echo 'Test 7: Model Persistence and Retrieval'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Check model is persisted
SELECT 
	COUNT(*) AS total_models,
	COUNT(DISTINCT algorithm) AS unique_algorithms,
	MIN(created_at) AS oldest_model,
	MAX(created_at) AS newest_model
FROM neurondb.ml_models
WHERE algorithm = 'linear_regression';

-- Cleanup
DROP TABLE IF EXISTS adv_model_temp;
DROP TABLE IF EXISTS gpu_model_adv;
DROP TABLE IF EXISTS cpu_model_adv;

\echo ''
\echo 'Advanced Linear Regression Test Complete!'
\echo '=========================================================================='

