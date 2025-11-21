\timing on
\pset footer off
\pset pager off

-- This test uses test_train_view and test_test_view tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist
-- GPU/CPU mode is configured via GUC (ALTER SYSTEM) before running tests

\set ON_ERROR_STOP on

/* Step 0: Read settings from test_settings table and apply them */
DO $$
DECLARE
	gpu_mode TEXT;
BEGIN
	-- Read GPU mode setting and enable/disable GPU accordingly
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	IF gpu_mode = 'gpu' THEN
		PERFORM neurondb_gpu_enable();
	ELSE
		-- GPU disabled or CPU mode - ensure GPU is off
		PERFORM set_config('neurondb.gpu_enabled', 'off', false);
	END IF;
END $$;

-- Train model once and store in temp table to reuse
DROP TABLE IF EXISTS gpu_model_temp;
CREATE TEMP TABLE gpu_model_temp AS
SELECT neurondb.train(
	'default',
	'knn',
	'test_train_view',
	'label',
	ARRAY['features'],
	'{"k": 5}'::jsonb
)::integer AS model_id;

-- Debug: Show model_id
SELECT model_id FROM gpu_model_temp;

-- Use optimized batch evaluation instead of per-row predictions
-- (Accuracy is already computed in evaluate() below)

-- Evaluate model and store result
DROP TABLE IF EXISTS gpu_metrics_temp;
CREATE TEMP TABLE gpu_metrics_temp (metrics jsonb);
DO $$
DECLARE
	mid integer;
	metrics_result jsonb;
BEGIN
	SELECT model_id INTO mid FROM gpu_model_temp LIMIT 1;
	metrics_result := neurondb.evaluate(mid, 'test_test_view', 'features', 'label');
	INSERT INTO gpu_metrics_temp VALUES (metrics_result);
END
$$;

-- Show metrics
SELECT
	format('%-15s', 'Accuracy') AS metric,
	CASE WHEN (m.metrics::jsonb ? 'accuracy')
		THEN ROUND((m.metrics::jsonb ->> 'accuracy')::numeric, 4)
		ELSE NULL END AS value
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'Precision'),
	CASE WHEN (m.metrics::jsonb ? 'precision')
		THEN ROUND((m.metrics::jsonb ->> 'precision')::numeric, 4)
		ELSE NULL END
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'Recall'),
	CASE WHEN (m.metrics::jsonb ? 'recall')
		THEN ROUND((m.metrics::jsonb ->> 'recall')::numeric, 4)
		ELSE NULL END
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'F1 Score'),
	CASE WHEN (m.metrics::jsonb ? 'f1_score')
		THEN ROUND((m.metrics::jsonb ->> 'f1_score')::numeric, 4)
		ELSE NULL END
FROM gpu_metrics_temp m;

-- Cleanup
DROP TABLE IF EXISTS gpu_model_temp;
DROP TABLE IF EXISTS gpu_metrics_temp;

\echo 'Test completed successfully'
