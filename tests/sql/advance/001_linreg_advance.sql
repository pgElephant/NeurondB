\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Linear Regression - Advanced Features Test'
\echo '=========================================================================='

-- This test uses test_train_view and test_test_view tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'test_train_view') THEN
		RAISE EXCEPTION 'test_train_view table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'test_test_view') THEN
		RAISE EXCEPTION 'test_test_view table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
END
$$;
-- Create views with 1000 rows for advance tests
DROP VIEW IF EXISTS test_train_view;
DROP VIEW IF EXISTS test_test_view;

CREATE VIEW test_train_view AS
SELECT features, label FROM sample_train LIMIT 1000;

CREATE VIEW test_test_view AS
SELECT features, label FROM sample_test LIMIT 1000;

-- Configure GPU (if available)
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,linreg_train,linreg_predict';
SELECT neurondb_gpu_enable() AS gpu_available;

\echo ''
\echo 'Test 1: Training with Custom Hyperparameters'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Train with custom parameters
DROP TABLE IF EXISTS adv_model_temp;
CREATE TEMP TABLE adv_model_temp AS
SELECT neurondb.train(
	'linear_regression',
	'test_train_view',
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
	m.metrics->>'r_squared' AS r_squared,
	m.metrics->>'mse' AS mean_squared_error,
	CASE 
		WHEN m.model_data IS NULL THEN 'NULL (GPU model)'
		ELSE format('%s bytes', pg_column_size(m.model_data))
	END AS model_data_status
FROM neurondb.ml_models m, adv_model_temp t
WHERE m.model_id = t.model_id;

\echo ''
\echo 'Test 3: Batch Evaluation Performance (Optimized C Batch Processing)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Use optimized batch evaluation instead of per-row predictions
CREATE TEMP TABLE adv_eval_temp AS
SELECT neurondb.evaluate((SELECT model_id FROM adv_model_temp), 'test_test_view', 'features', 'label') AS metrics;

SELECT
	'Total Samples' AS metric,
	(SELECT COUNT(*)::bigint FROM test_test_view WHERE features IS NOT NULL AND label IS NOT NULL)::text AS value
UNION ALL
SELECT 'MSE', ROUND((metrics->>'mse')::numeric, 6)::text
FROM adv_eval_temp
WHERE metrics->>'mse' IS NOT NULL
UNION ALL
SELECT 'RMSE', ROUND((metrics->>'rmse')::numeric, 6)::text
FROM adv_eval_temp
WHERE metrics->>'rmse' IS NOT NULL
UNION ALL
SELECT 'MAE', ROUND((metrics->>'mae')::numeric, 6)::text
FROM adv_eval_temp
WHERE metrics->>'mae' IS NOT NULL
UNION ALL
SELECT 'R²', ROUND((metrics->>'r_squared')::numeric, 6)::text
FROM adv_eval_temp
WHERE metrics->>'r_squared' IS NOT NULL
ORDER BY metric;

\echo ''
\echo 'Test 4: Evaluation Metrics Summary'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Show detailed evaluation metrics from optimized batch processing
SELECT
	'MSE' AS metric,
	ROUND((metrics->>'mse')::numeric, 6)::numeric AS value
FROM adv_eval_temp
WHERE metrics->>'mse' IS NOT NULL
UNION ALL
SELECT 'RMSE', ROUND((metrics->>'rmse')::numeric, 6)::numeric
FROM adv_eval_temp
WHERE metrics->>'rmse' IS NOT NULL
UNION ALL
SELECT 'MAE', ROUND((metrics->>'mae')::numeric, 6)::numeric
FROM adv_eval_temp
WHERE metrics->>'mae' IS NOT NULL
UNION ALL
SELECT 'R²', ROUND((metrics->>'r_squared')::numeric, 6)::numeric
FROM adv_eval_temp
WHERE metrics->>'r_squared' IS NOT NULL
ORDER BY metric;

\echo ''
\echo 'Test 5: Model Quality Metrics'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Use evaluation metrics for quality assessment (much faster than per-row residuals)
SELECT
	ROUND((metrics->>'mse')::numeric, 6) AS mean_squared_error,
	ROUND((metrics->>'rmse')::numeric, 6) AS root_mean_squared_error,
	ROUND((metrics->>'mae')::numeric, 6) AS mean_absolute_error,
	ROUND((metrics->>'r_squared')::numeric, 6) AS r_squared,
	CASE 
		WHEN (metrics->>'r_squared')::numeric > 0.5 THEN 'Good fit'
		WHEN (metrics->>'r_squared')::numeric > 0.1 THEN 'Moderate fit'
		ELSE 'Poor fit (may need feature engineering)'
	END AS fit_quality
FROM adv_eval_temp;

\echo ''
\echo 'Test 6: Model Storage Type Verification'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Train model (storage is automatically determined by GPU availability)
DROP TABLE IF EXISTS model_storage_test;
CREATE TEMP TABLE model_storage_test AS
SELECT neurondb.train(
	'linear_regression',
	'test_train_view',
	'features',
	'label',
	'{}'::jsonb
)::integer AS model_id;

-- Check actual storage type used (GPU if available, CPU otherwise)
SELECT 
	m.model_id,
	m.metrics->>'storage' AS storage_type,
	ROUND((m.metrics->>'r_squared')::numeric, 4) AS r_squared,
	ROUND((m.metrics->>'mse')::numeric, 4) AS mse,
	CASE 
		WHEN m.metrics->>'storage' = 'gpu' THEN 'GPU training succeeded'
		WHEN m.metrics->>'storage' = 'cpu' THEN 'CPU training (GPU unavailable or failed)'
		ELSE 'Unknown storage type'
	END AS storage_status
FROM neurondb.ml_models m, model_storage_test t
WHERE m.model_id = t.model_id;

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
DROP TABLE IF EXISTS adv_eval_temp;
DROP TABLE IF EXISTS model_storage_test;

\echo ''
\echo 'Advanced Linear Regression Test Complete!'
\echo '=========================================================================='

