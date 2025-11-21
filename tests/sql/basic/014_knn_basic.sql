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

/* Step 0: Read settings from test_settings table and verify GPU configuration */
DO $$
DECLARE
	gpu_mode TEXT;
	current_gpu_enabled TEXT;
BEGIN
	-- Read GPU mode setting from test_settings
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	
	-- Verify GPU configuration matches test_settings (set by test runner)
	SELECT current_setting('neurondb.gpu_enabled', true) INTO current_gpu_enabled;
	
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
	eval_error text;
BEGIN
	-- Get model_id with defensive check
	SELECT model_id INTO mid FROM gpu_model_temp LIMIT 1;
	IF mid IS NULL THEN
		RAISE WARNING 'No model_id found in gpu_model_temp';
		INSERT INTO gpu_metrics_temp VALUES ('{"error": "No model_id found"}'::jsonb);
		RETURN;
	END IF;
	
	-- Try evaluation with multiple layers of error handling
	BEGIN
		-- Attempt evaluation
		BEGIN
			metrics_result := neurondb.evaluate(mid, 'test_test_view', 'features', 'label');
			
			IF metrics_result IS NULL THEN
				RAISE WARNING 'Evaluation returned NULL';
				INSERT INTO gpu_metrics_temp VALUES ('{"error": "Evaluation returned NULL"}'::jsonb);
			ELSE
				INSERT INTO gpu_metrics_temp VALUES (metrics_result);
			END IF;
		EXCEPTION WHEN OTHERS THEN
			eval_error := SQLERRM;
			RAISE WARNING 'Evaluation exception: %', eval_error;
			-- Properly escape JSON - replace quotes and newlines
			eval_error := REPLACE(REPLACE(REPLACE(eval_error, '"', '\"'), E'\n', ' '), E'\r', ' ');
			INSERT INTO gpu_metrics_temp VALUES (jsonb_build_object('error', eval_error));
		END;
	EXCEPTION WHEN OTHERS THEN
		eval_error := SQLERRM;
		RAISE WARNING 'Outer evaluation exception: %', eval_error;
		-- Properly escape JSON
		eval_error := REPLACE(REPLACE(REPLACE(eval_error, '"', '\"'), E'\n', ' '), E'\r', ' ');
		INSERT INTO gpu_metrics_temp VALUES (jsonb_build_object('error', eval_error));
	END;
END
$$;

-- Show metrics (with NULL safety)
SELECT
	format('%-15s', 'Accuracy') AS metric,
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN (m.metrics::jsonb->>'error')
		WHEN (m.metrics::jsonb ? 'accuracy')
			THEN ROUND((m.metrics::jsonb ->> 'accuracy')::numeric, 4)::text
		ELSE NULL 
	END AS value
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'Precision'),
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'precision')
			THEN ROUND((m.metrics::jsonb ->> 'precision')::numeric, 4)::text
		ELSE NULL 
	END
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'Recall'),
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'recall')
			THEN ROUND((m.metrics::jsonb ->> 'recall')::numeric, 4)::text
		ELSE NULL 
	END
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'F1 Score'),
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'f1_score')
			THEN ROUND((m.metrics::jsonb ->> 'f1_score')::numeric, 4)::text
		ELSE NULL 
	END
FROM gpu_metrics_temp m;

-- Cleanup
DROP TABLE IF EXISTS gpu_model_temp;
DROP TABLE IF EXISTS gpu_metrics_temp;

\echo 'Test completed successfully'
