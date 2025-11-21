\timing on
\pset footer off
\pset pager off

-- This test uses sample_train and sample_test tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist

\set ON_ERROR_STOP on

/* Step 0: Read settings from test_settings table and verify GPU configuration */
DO $$
DECLARE
	gpu_mode TEXT;
	current_gpu_enabled TEXT;
	current_automl_gpu TEXT;
BEGIN
	-- Read GPU mode setting from test_settings
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	
	-- Verify GPU configuration matches test_settings (set by test runner)
	SELECT current_setting('neurondb.gpu_enabled', true) INTO current_gpu_enabled;
	SELECT current_setting('neurondb.automl.use_gpu', true) INTO current_automl_gpu;
	
	IF gpu_mode = 'gpu' THEN
		-- Verify GPU is enabled (should be set by test runner)
		IF current_gpu_enabled != 'on' THEN
			RAISE WARNING 'GPU mode expected but neurondb.gpu_enabled = % (expected: on)', current_gpu_enabled;
		END IF;
	ELSE
		-- Verify GPU is disabled (should be set by test runner)
		IF current_gpu_enabled != 'off' THEN
			RAISE WARNING 'CPU mode expected but neurondb.gpu_enabled = % (expected: off)', current_gpu_enabled;
		END IF;
	END IF;
END $$;

\echo '=========================================================================='
\echo '=========================================================================='

-- Test AutoML for classification task
DROP TABLE IF EXISTS automl_model_temp;
CREATE TEMP TABLE automl_model_temp AS
SELECT auto_train(
	'test_train_view',
	'features',
	'label',
	'classification',
	'accuracy'
)::integer AS model_id;

SELECT model_id FROM automl_model_temp;

-- Evaluate the best model
CREATE TEMP TABLE automl_metrics_temp (metrics jsonb);
DO $$
DECLARE
	mid integer;
	metrics_result jsonb;
BEGIN
	SELECT model_id INTO mid FROM automl_model_temp LIMIT 1;
	metrics_result := neurondb.evaluate(mid, 'test_test_view', 'features', 'label');
	INSERT INTO automl_metrics_temp VALUES (metrics_result);
END
$$;

-- Show metrics
SELECT
	format('%-15s', 'Algorithm') AS metric,
	(SELECT m.algorithm::text FROM neurondb.ml_models m, automl_model_temp t WHERE m.model_id = t.model_id) AS value
UNION ALL
SELECT
	format('%-15s', 'Accuracy'),
	CASE WHEN (m.metrics::jsonb ? 'accuracy')
		THEN ROUND((m.metrics::jsonb ->> 'accuracy')::numeric, 4)::text
		ELSE NULL END
FROM automl_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'Precision'),
	CASE WHEN (m.metrics::jsonb ? 'precision')
		THEN ROUND((m.metrics::jsonb ->> 'precision')::numeric, 4)::text
		ELSE NULL END
FROM automl_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'Recall'),
	CASE WHEN (m.metrics::jsonb ? 'recall')
		THEN ROUND((m.metrics::jsonb ->> 'recall')::numeric, 4)::text
		ELSE NULL END
FROM automl_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'F1 Score'),
	CASE WHEN (m.metrics::jsonb ? 'f1_score')
		THEN ROUND((m.metrics::jsonb ->> 'f1_score')::numeric, 4)::text
		ELSE NULL END
FROM automl_metrics_temp m;

-- Cleanup
DROP TABLE IF EXISTS automl_model_temp;
DROP TABLE IF EXISTS automl_metrics_temp;

\echo 'Test completed successfully'
