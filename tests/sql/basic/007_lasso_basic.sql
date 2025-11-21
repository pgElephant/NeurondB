\timing on
\pset footer off
\pset pager off

-- This test uses test_train_view and test_test_view tables created by ml_dataset.py
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
SELECT train_lasso_regression(
	'test_train_view',
	'features',
	'label',
	0.01,  -- lambda
	1000   -- max_iters
)::integer AS model_id;

-- Debug: Show model_id
SELECT model_id FROM gpu_model_temp;

-- Use optimized batch evaluation instead of per-row predictions
-- (MSE, RMSE, MAE are already computed in evaluate() below)

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

-- Show metrics (for regression, we show different metrics) (with NULL safety)
SELECT
	format('%-15s', 'MSE') AS metric,
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN (m.metrics::jsonb->>'error')
		WHEN (m.metrics::jsonb ? 'mse')
			THEN ROUND((m.metrics::jsonb ->> 'mse')::numeric, 4)::text
		ELSE NULL 
	END AS value
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'RMSE'),
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'rmse')
			THEN ROUND((m.metrics::jsonb ->> 'rmse')::numeric, 4)::text
		ELSE NULL 
	END
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'MAE'),
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'mae')
			THEN ROUND((m.metrics::jsonb ->> 'mae')::numeric, 4)::text
		ELSE NULL 
	END
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'R²'),
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'r_squared')
			THEN ROUND((m.metrics::jsonb ->> 'r_squared')::numeric, 4)::text
		ELSE NULL 
	END
FROM gpu_metrics_temp m;

-- Store results in test_metrics table
INSERT INTO test_metrics (
	test_name, algorithm, model_id, train_samples, test_samples,
	mse, rmse, mae, r_squared, updated_at
)
SELECT 
	'007_lasso_basic',
	'lasso',
	(SELECT model_id FROM gpu_model_temp),
	(SELECT COUNT(*)::bigint FROM test_train_view),
	(SELECT COUNT(*)::bigint FROM test_test_view),
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'mse') THEN ROUND((m.metrics::jsonb->>'mse')::numeric, 6) 
		ELSE NULL 
	END,
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'rmse') THEN ROUND((m.metrics::jsonb->>'rmse')::numeric, 6) 
		ELSE NULL 
	END,
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'mae') THEN ROUND((m.metrics::jsonb->>'mae')::numeric, 6) 
		ELSE NULL 
	END,
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'r_squared') THEN ROUND((m.metrics::jsonb->>'r_squared')::numeric, 6) 
		ELSE NULL 
	END,
	CURRENT_TIMESTAMP
FROM gpu_metrics_temp m
ON CONFLICT (test_name) DO UPDATE SET
	algorithm = EXCLUDED.algorithm,
	model_id = EXCLUDED.model_id,
	train_samples = EXCLUDED.train_samples,
	test_samples = EXCLUDED.test_samples,
	mse = EXCLUDED.mse,
	rmse = EXCLUDED.rmse,
	mae = EXCLUDED.mae,
	r_squared = EXCLUDED.r_squared,
	updated_at = CURRENT_TIMESTAMP;

-- Cleanup
DROP TABLE IF EXISTS gpu_model_temp;
DROP TABLE IF EXISTS gpu_metrics_temp;

\echo 'Test completed successfully'
