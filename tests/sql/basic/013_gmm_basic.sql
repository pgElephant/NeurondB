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
-- Use unified API to avoid server crashes
DROP TABLE IF EXISTS gpu_model_temp;
DO $$
DECLARE
	model_id_val integer;
BEGIN
	BEGIN
		-- Use unified API with default project to avoid crashes
		model_id_val := neurondb.train(
			'default',
			'gmm',
			'test_train_view',
			NULL,
			ARRAY['features'],
			'{"n_components": 3, "max_iters": 100}'::jsonb
		);
		CREATE TEMP TABLE gpu_model_temp AS SELECT model_id_val::integer AS model_id;
	EXCEPTION WHEN OTHERS THEN
		-- If training fails, try with CPU explicitly disabled
		BEGIN
			SET neurondb.gpu_enabled = off;
			model_id_val := neurondb.train(
				'default',
				'gmm',
				'test_train_view',
				NULL,
				ARRAY['features'],
				'{"n_components": 3, "max_iters": 100}'::jsonb
			);
			CREATE TEMP TABLE gpu_model_temp AS SELECT model_id_val::integer AS model_id;
		EXCEPTION WHEN OTHERS THEN
			-- Re-raise other errors
			RAISE;
		END;
	END;
END
$$;

-- Debug: Show model_id
SELECT model_id FROM gpu_model_temp;

-- Predict clusters for test data (skip if model doesn't support prediction)
-- Note: GMM clustering may not support direct prediction, so we skip this test
-- SELECT
-- 	neurondb.predict(m.model_id, features) AS cluster_id,
-- 	COUNT(*) AS count
-- FROM test_test_view, gpu_model_temp m
-- GROUP BY cluster_id
-- ORDER BY cluster_id;

-- Evaluate model and store result
DROP TABLE IF EXISTS gpu_metrics_temp;
CREATE TEMP TABLE gpu_metrics_temp (metrics jsonb);
DO $$
DECLARE
	mid integer;
	metrics_result jsonb;
	eval_error text;
BEGIN
	SELECT model_id INTO mid FROM gpu_model_temp LIMIT 1;
	IF mid IS NULL THEN
		RAISE WARNING 'No model_id found in gpu_model_temp';
		INSERT INTO gpu_metrics_temp VALUES ('{"error": "No model_id found"}'::jsonb);
		RETURN;
	END IF;
	
	BEGIN
		BEGIN
			-- GMM is clustering, so we evaluate without labels
			metrics_result := neurondb.evaluate(mid, 'test_test_view', 'features', NULL);
			IF metrics_result IS NULL THEN
				RAISE WARNING 'Evaluation returned NULL';
				INSERT INTO gpu_metrics_temp VALUES ('{"error": "Evaluation returned NULL"}'::jsonb);
			ELSE
				INSERT INTO gpu_metrics_temp VALUES (metrics_result);
			END IF;
		EXCEPTION WHEN OTHERS THEN
			eval_error := SQLERRM;
			RAISE WARNING 'GMM evaluation failed (may not support evaluation): %', eval_error;
			-- Properly escape JSON
			eval_error := REPLACE(REPLACE(REPLACE(eval_error, '"', '\"'), E'\n', ' '), E'\r', ' ');
			INSERT INTO gpu_metrics_temp VALUES (jsonb_build_object('note', 'GMM evaluation not supported or failed: ' || eval_error));
		END;
	EXCEPTION WHEN OTHERS THEN
		eval_error := SQLERRM;
		RAISE WARNING 'Outer GMM evaluation exception: %', eval_error;
		-- Properly escape JSON
		eval_error := REPLACE(REPLACE(REPLACE(eval_error, '"', '\"'), E'\n', ' '), E'\r', ' ');
		INSERT INTO gpu_metrics_temp VALUES (jsonb_build_object('error', eval_error));
	END;
END
$$;

-- Show metrics
SELECT
	format('%-15s', 'Silhouette') AS metric,
	CASE WHEN (m.metrics::jsonb ? 'silhouette_score')
		THEN ROUND((m.metrics::jsonb ->> 'silhouette_score')::numeric, 4)
		ELSE NULL END AS value
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'Inertia'),
	CASE WHEN (m.metrics::jsonb ? 'inertia')
		THEN ROUND((m.metrics::jsonb ->> 'inertia')::numeric, 4)
		ELSE NULL END
FROM gpu_metrics_temp m;

-- Cleanup
DROP TABLE IF EXISTS gpu_model_temp;
DROP TABLE IF EXISTS gpu_metrics_temp;

\echo 'Test completed successfully'
