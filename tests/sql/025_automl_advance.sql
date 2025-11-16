\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'AutoML - Advanced Features Test'
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
SET neurondb.automl.use_gpu = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,linreg_train,linreg_predict,lr_train,lr_predict,rf_train,rf_predict,dt_train,dt_predict,ridge_train,ridge_predict,lasso_train,lasso_predict';
SELECT neurondb_gpu_enable();

\echo ''
\echo 'Test 1: AutoML with Custom Metric and Timeout'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test with different metrics
DROP TABLE IF EXISTS automl_custom_metric;
CREATE TEMP TABLE automl_custom_metric AS
SELECT auto_train(
	'sample_train',
	'features',
	'label',
	'classification',
	'f1_score'
) AS result;

SELECT result FROM automl_custom_metric;

\echo ''
\echo 'Test 2: AutoML Model Comparison and Leaderboard'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Get all models created by AutoML
SELECT 
	m.model_id,
	m.algorithm,
	m.created_at,
	m.metrics->>'accuracy' AS accuracy,
	m.metrics->>'precision' AS precision,
	m.metrics->>'recall' AS recall,
	m.metrics->>'f1_score' AS f1_score,
	m.metrics->>'storage' AS storage_type
FROM neurondb.ml_models m
WHERE m.project_id IN (
	SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'default'
)
ORDER BY (m.metrics->>'accuracy')::numeric DESC NULLS LAST
LIMIT 10;

\echo ''
\echo 'Test 3: AutoML for Regression Tasks'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create regression dataset (using label as continuous)
DROP TABLE IF EXISTS automl_regression_result;
CREATE TEMP TABLE automl_regression_result AS
SELECT auto_train(
	'sample_train',
	'features',
	'label',
	'regression',
	'r2'
) AS result;

SELECT result FROM automl_regression_result;

\echo ''
\echo 'Test 4: AutoML Best Model Evaluation'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Find best model and evaluate it
DO $$
DECLARE
	best_model_id integer;
	metrics_result jsonb;
BEGIN
	-- Get best model by accuracy
	SELECT model_id INTO best_model_id
	FROM neurondb.ml_models
	WHERE project_id IN (
		SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'default'
	)
	AND algorithm IN ('random_forest', 'logistic_regression', 'decision_tree', 'svm')
	ORDER BY (metrics->>'accuracy')::numeric DESC NULLS LAST
	LIMIT 1;
	
	IF best_model_id IS NOT NULL THEN
		metrics_result := neurondb.evaluate(best_model_id, 'sample_test', 'features', 'label');
		RAISE NOTICE 'Best model ID: %, Metrics: %', best_model_id, metrics_result;
	END IF;
END
$$;

\echo ''
\echo 'Test 5: AutoML Cross-Validation Results'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Analyze model performance across different algorithms
SELECT 
	algorithm,
	COUNT(*) AS model_count,
	AVG((metrics->>'accuracy')::numeric) AS avg_accuracy,
	MAX((metrics->>'accuracy')::numeric) AS max_accuracy,
	MIN((metrics->>'accuracy')::numeric) AS min_accuracy,
	STDDEV((metrics->>'accuracy')::numeric) AS stddev_accuracy
FROM neurondb.ml_models
WHERE project_id IN (
	SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'default'
)
AND metrics->>'accuracy' IS NOT NULL
GROUP BY algorithm
ORDER BY avg_accuracy DESC;

\echo ''
\echo 'Test 6: AutoML GPU vs CPU Performance Comparison'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Compare GPU and CPU models
SELECT 
	CASE 
		WHEN metrics->>'storage' = 'gpu' THEN 'GPU'
		ELSE 'CPU'
	END AS storage_type,
	algorithm,
	AVG((metrics->>'accuracy')::numeric) AS avg_accuracy,
	AVG(EXTRACT(EPOCH FROM (created_at - (SELECT MIN(created_at) FROM neurondb.ml_models WHERE project_id IN (SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'default'))))) AS avg_training_time_seconds
FROM neurondb.ml_models
WHERE project_id IN (
	SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'default'
)
AND metrics->>'accuracy' IS NOT NULL
GROUP BY storage_type, algorithm
ORDER BY storage_type, avg_accuracy DESC;

\echo ''
\echo 'Test 7: AutoML Hyperparameter Analysis'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Analyze hyperparameters used in best models
SELECT 
	algorithm,
	metrics->>'max_depth' AS max_depth,
	metrics->>'n_estimators' AS n_estimators,
	metrics->>'learning_rate' AS learning_rate,
	(metrics->>'accuracy')::numeric AS accuracy
FROM neurondb.ml_models
WHERE project_id IN (
	SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'default'
)
AND metrics->>'accuracy' IS NOT NULL
ORDER BY (metrics->>'accuracy')::numeric DESC
LIMIT 5;

\echo ''
\echo 'Advanced AutoML Test Complete!'
\echo '=========================================================================='

