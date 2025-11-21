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
	COALESCE(m.metrics::jsonb->>'storage', 'default') AS storage,
	ROUND(COALESCE((m.metrics::jsonb->>'final_loss')::numeric, 0), 6) AS final_loss,
	ROUND(COALESCE((m.metrics::jsonb->>'accuracy')::numeric, 0), 6) AS accuracy
FROM neurondb.ml_models m, gpu_model_temp t
WHERE m.model_id = t.model_id;

/* Verify GPU was used for training when GPU mode is enabled */
DO $$
DECLARE
	gpu_mode TEXT;
	storage_val TEXT;
BEGIN
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	SELECT COALESCE(m.metrics::jsonb->>'storage', 'cpu') INTO storage_val
	FROM neurondb.ml_models m, gpu_model_temp t
	WHERE m.model_id = t.model_id;
	
	IF gpu_mode = 'gpu' AND storage_val != 'gpu' THEN
		RAISE EXCEPTION 'GPU mode enabled but model was trained on CPU (storage=%)', storage_val;
	END IF;
END $$;

/* Step 6: Test set statistics */

SELECT COUNT(*)::bigint AS test_samples
FROM test_test_view
WHERE features IS NOT NULL AND label IS NOT NULL;

/* Step 7: Evaluation using neurondb.evaluate (optimized C batch processing - single call) */
\echo 'Step 7: Evaluating model (optimized C batch processing)...'

DROP TABLE IF EXISTS gpu_metrics_temp;
CREATE TEMP TABLE gpu_metrics_temp AS
SELECT neurondb.evaluate((SELECT model_id FROM gpu_model_temp), 'test_test_view', 'features', 'label') AS metrics;

SELECT
	'Accuracy' AS metric,
	ROUND((metrics->>'accuracy')::numeric, 6)::text AS value
FROM gpu_metrics_temp
WHERE metrics->>'accuracy' IS NOT NULL
UNION ALL
SELECT 'Precision', ROUND((metrics->>'precision')::numeric, 6)::text
FROM gpu_metrics_temp
WHERE metrics->>'precision' IS NOT NULL
UNION ALL
SELECT 'Recall', ROUND((metrics->>'recall')::numeric, 6)::text
FROM gpu_metrics_temp
WHERE metrics->>'recall' IS NOT NULL
UNION ALL
SELECT 'F1 Score', ROUND((metrics->>'f1_score')::numeric, 6)::text
FROM gpu_metrics_temp
WHERE metrics->>'f1_score' IS NOT NULL
ORDER BY metric;

/* Step 8: Summary */

SELECT 
	(SELECT model_id FROM gpu_model_temp) AS model_id,
	(SELECT COUNT(*)::bigint FROM test_train_view) AS train_samples,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_samples,
	(SELECT ROUND((metrics->>'accuracy')::numeric, 6) FROM gpu_metrics_temp) AS final_accuracy;

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
	ROUND((metrics->>'accuracy')::numeric, 6),
	ROUND((metrics->>'precision')::numeric, 6),
	ROUND((metrics->>'recall')::numeric, 6),
	ROUND((metrics->>'f1_score')::numeric, 6),
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
