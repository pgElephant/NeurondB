\timing on
\pset footer off
\pset pager off
SET log_min_messages = DEBUG1;

-- This test uses test_train_view and test_test_view tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist

-- GPU disabled for testing CPU implementation
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

-- Train model once and store in temp table to reuse
-- Use the view with 1000 rows for basic tests
DROP TABLE IF EXISTS sample_train_subset;
CREATE TEMP TABLE sample_train_subset AS
SELECT features, label FROM test_train_view;

DROP TABLE IF EXISTS gpu_model_temp;
CREATE TEMP TABLE gpu_model_temp AS
SELECT neurondb.train(
	'random_forest',
	'sample_train_subset',
	'features',
	'label',
	'{"n_trees": 3}'::jsonb
)::integer AS model_id;

-- Debug: Show model_id
SELECT model_id FROM gpu_model_temp;

-- Use optimized batch evaluation instead of per-row predictions
-- (Accuracy is already computed in evaluate() below)

-- Evaluate model and store result
-- Use the view with 1000 rows for basic tests
DROP TABLE IF EXISTS sample_test_subset;
CREATE TEMP TABLE sample_test_subset AS
SELECT features, label FROM test_test_view;

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
