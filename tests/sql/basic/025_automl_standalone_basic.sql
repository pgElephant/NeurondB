\timing on
\pset footer off
\pset pager off

-- This test uses sample_train and sample_test tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist

SET neurondb.automl.use_gpu = on;
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo '=========================================================================='

-- Test AutoML for classification
DROP TABLE IF EXISTS automl_classification_result;
CREATE TEMP TABLE automl_classification_result AS
SELECT auto_train(
	'sample_train',
	'features',
	'label',
	'classification',
	'accuracy'
) AS result;

-- Show AutoML classification results
SELECT result FROM automl_classification_result;

-- Extract best model_id from result (if available)
-- The result contains the best model_id in the text
-- For demonstration, we'll show the leaderboard

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

-- Test AutoML for regression
DROP TABLE IF EXISTS automl_regression_result;
CREATE TEMP TABLE automl_regression_result AS
SELECT auto_train(
	'sample_train',
	'features',
	'label',
	'regression',
	'r2'
) AS result;

-- Show AutoML regression results
SELECT result FROM automl_regression_result;

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

-- Test classification with F1 score
DROP TABLE IF EXISTS automl_f1_result;
CREATE TEMP TABLE automl_f1_result AS
SELECT auto_train(
	'sample_train',
	'features',
	'label',
	'classification',
	'f1'
) AS result;

SELECT 'F1 Score Results:' AS test_type, result FROM automl_f1_result;

-- Test regression with MSE
DROP TABLE IF EXISTS automl_mse_result;
CREATE TEMP TABLE automl_mse_result AS
SELECT auto_train(
	'sample_train',
	'features',
	'label',
	'regression',
	'mse'
) AS result;

SELECT 'MSE Results:' AS test_type, result FROM automl_mse_result;

\echo ''
\echo '=========================================================================='
\echo 'AutoML: Verify GPU usage'
\echo '=========================================================================='

-- Check if GPU models were created
-- Explicitly exclude model_data (bytea) to avoid pager issues
SELECT 
	m.model_id,
	m.algorithm,
	m.metrics->>'storage' AS storage_backend,
	m.metrics->>'n_features' AS n_features,
	m.metrics->>'n_samples' AS n_samples,
	m.created_at,
	CASE WHEN m.model_data IS NULL THEN 'NULL' ELSE 'present' END AS model_data_status
FROM neurondb.ml_models m
WHERE m.algorithm IN ('linear_regression', 'logistic_regression', 'random_forest', 'decision_tree', 'ridge', 'lasso', 'svm')
	AND m.created_at > NOW() - INTERVAL '10 minutes'
ORDER BY m.created_at DESC
LIMIT 10;

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

-- Cleanup
DROP TABLE IF EXISTS automl_classification_result;
DROP TABLE IF EXISTS automl_regression_result;
DROP TABLE IF EXISTS automl_f1_result;
DROP TABLE IF EXISTS automl_mse_result;

\echo 'Test completed successfully'
