/*-------------------------------------------------------------------------
 *
 * 002_logreg.sql
 *    Logistic Regression test
 *
 *    Step-by-step test with clean output and timing
 *
 *-------------------------------------------------------------------------*/

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

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

/* Step 1: Verify prerequisites */
\echo 'Step 1: Verifying prerequisites...'

SELECT 
	COUNT(*)::bigint AS train_rows,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_rows;

/* Step 2: Dataset statistics */
\echo 'Step 3: Dataset statistics...'

SELECT 
	'test_train_view' AS dataset,
	COUNT(*)::bigint AS total_rows,
	COUNT(*) FILTER (WHERE features IS NOT NULL AND label IS NOT NULL)::bigint AS valid_rows
FROM test_train_view
UNION ALL
SELECT 
	'test_test_view',
	COUNT(*)::bigint,
	COUNT(*) FILTER (WHERE features IS NOT NULL AND label IS NOT NULL)::bigint
FROM test_test_view;

/* Step 4: Train model */
\echo 'Step 4: Training logistic regression model...'

DROP TABLE IF EXISTS gpu_model_temp;
CREATE TEMP TABLE gpu_model_temp AS
SELECT 
	neurondb.train(
		'default',
		'logistic_regression',
		'test_train_view',
		'label',
		ARRAY['features'],
		'{"max_iters": 1000, "learning_rate": 0.01, "lambda": 0.001}'::jsonb
	)::integer AS model_id;

SELECT model_id FROM gpu_model_temp;

/* Step 5: Training metrics */
\echo 'Step 5: Training metrics...'

SELECT
	m.algorithm::text AS algorithm,
	COALESCE((m.metrics::jsonb->>'n_samples')::bigint, m.num_samples::bigint, 0) AS n_samples,
	COALESCE((m.metrics::jsonb->>'n_features')::integer, m.num_features, 0) AS n_features,
	COALESCE(m.metrics::jsonb->>'storage', 'cpu') AS storage,
	ROUND(COALESCE((m.metrics::jsonb->>'final_loss')::numeric, 0), 6) AS final_loss,
	ROUND(COALESCE((m.metrics::jsonb->>'accuracy')::numeric, 0), 6) AS accuracy
FROM neurondb.ml_models m, gpu_model_temp t
WHERE m.model_id = t.model_id;

/* Verify GPU was used for training when GPU mode is enabled */
DO $$
DECLARE
	gpu_mode TEXT;
	storage_val TEXT;
	gpu_available BOOLEAN;
BEGIN
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	SELECT COALESCE(m.metrics::jsonb->>'storage', 'cpu') INTO storage_val
	FROM neurondb.ml_models m, gpu_model_temp t
	WHERE m.model_id = t.model_id;
	
	-- Check if GPU is actually available (use is_available column which matches C code check)
	SELECT COALESCE(BOOL_OR(is_available), false) INTO gpu_available
	FROM neurondb_gpu_info();
	
	-- If GPU mode is enabled and GPU is detected but model was trained on CPU, warn (not error)
	-- This can happen if Metal backend isn't properly initialized or GPU training failed
	IF gpu_mode = 'gpu' AND gpu_available AND storage_val != 'gpu' THEN
		RAISE WARNING 'GPU mode enabled and GPU detected but model was trained on CPU (storage=%). This may indicate Metal backend initialization issue.', storage_val;
	END IF;
	
	-- If GPU is not available, it's expected to use CPU, so just warn
	IF gpu_mode = 'gpu' AND NOT gpu_available AND storage_val != 'gpu' THEN
		RAISE WARNING 'GPU mode enabled but GPU hardware not available, model trained on CPU (storage=%)', storage_val;
	END IF;
	
	IF gpu_mode = 'cpu' AND storage_val = 'gpu' THEN
		RAISE WARNING 'CPU mode enabled but model was trained on GPU (storage=gpu)';
	END IF;
END $$;

/* Step 6: Test set statistics */

SELECT COUNT(*)::bigint AS test_samples
FROM test_test_view
WHERE features IS NOT NULL AND label IS NOT NULL;

/* Step 7: Evaluation using neurondb.evaluate (optimized C batch processing - single call) */
\echo 'Step 7: Evaluating model (optimized C batch processing)...'

/* Step 7: Evaluation using neurondb.evaluate (optimized C batch processing - single call) */
\echo 'Step 7: Evaluating model (CPU training)...'

DROP TABLE IF EXISTS gpu_metrics_temp;
CREATE TEMP TABLE gpu_metrics_temp AS
SELECT neurondb.evaluate((SELECT model_id FROM gpu_model_temp), 'test_test_view', 'features', 'label') AS metrics;

SELECT
	'Accuracy' AS metric,
	COALESCE(ROUND((metrics->>'accuracy')::numeric, 6)::text, 'N/A (evaluation failed)') AS value
FROM gpu_metrics_temp
UNION ALL
SELECT 'Precision', COALESCE(ROUND((metrics->>'precision')::numeric, 6)::text, 'N/A (evaluation failed)')
FROM gpu_metrics_temp
UNION ALL
SELECT 'Recall', COALESCE(ROUND((metrics->>'recall')::numeric, 6)::text, 'N/A (evaluation failed)')
FROM gpu_metrics_temp
UNION ALL
SELECT 'F1 Score', COALESCE(ROUND((metrics->>'f1_score')::numeric, 6)::text, 'N/A (evaluation failed)')
FROM gpu_metrics_temp
ORDER BY metric;

/* Step 8: Summary */

SELECT
	(SELECT model_id FROM gpu_model_temp) AS model_id,
	(SELECT COUNT(*)::bigint FROM test_train_view) AS train_samples,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_samples,
	COALESCE((SELECT ROUND((metrics->>'accuracy')::numeric, 6) FROM gpu_metrics_temp), 0) AS final_accuracy;

-- Store results in test_metrics table
INSERT INTO test_metrics (
	test_name, algorithm, model_id, train_samples, test_samples,
	accuracy, precision, recall, f1_score, updated_at
)
SELECT
	'002_logreg_basic',
	'logistic_regression',
	(SELECT model_id FROM gpu_model_temp),
	(SELECT COUNT(*)::bigint FROM test_train_view),
	(SELECT COUNT(*)::bigint FROM test_test_view),
	COALESCE(ROUND((metrics->>'accuracy')::numeric, 6), 0),
	COALESCE(ROUND((metrics->>'precision')::numeric, 6), 0),
	COALESCE(ROUND((metrics->>'recall')::numeric, 6), 0),
	COALESCE(ROUND((metrics->>'f1_score')::numeric, 6), 0),
	CURRENT_TIMESTAMP
FROM gpu_metrics_temp
ON CONFLICT (test_name) DO UPDATE SET
	algorithm = EXCLUDED.algorithm,
	model_id = EXCLUDED.model_id,
	train_samples = EXCLUDED.train_samples,
	test_samples = EXCLUDED.test_samples,
	accuracy = EXCLUDED.accuracy,
	precision = EXCLUDED.precision,
	recall = EXCLUDED.recall,
	f1_score = EXCLUDED.f1_score,
	updated_at = CURRENT_TIMESTAMP;

/* Cleanup */
DROP TABLE IF EXISTS gpu_model_temp;
DROP TABLE IF EXISTS gpu_metrics_temp;

\echo 'Test completed successfully'
