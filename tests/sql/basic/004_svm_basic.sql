\timing on
\pset footer off
\pset pager off

-- This test uses test_train_view and test_test_view tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist

SET log_min_messages = debug1;

\set ON_ERROR_STOP on

-- Train model once and store in temp table to reuse
DROP TABLE IF EXISTS gpu_model_temp;
CREATE TEMP TABLE gpu_model_temp AS
SELECT neurondb.train(
	'default',
	'svm',
	'test_train_view',
	'label',
	ARRAY['features'],
	'{"C": 1.0, "max_iters": 1000}'::jsonb
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

-- Store results in test_metrics table
INSERT INTO test_metrics (
	test_name, algorithm, model_id, train_samples, test_samples,
	accuracy, precision, recall, f1_score, updated_at
)
SELECT 
	'004_svm_basic',
	'svm',
	(SELECT model_id FROM gpu_model_temp),
	(SELECT COUNT(*)::bigint FROM test_train_view),
	(SELECT COUNT(*)::bigint FROM test_test_view),
	CASE WHEN (m.metrics::jsonb ? 'accuracy') THEN ROUND((m.metrics::jsonb->>'accuracy')::numeric, 6) ELSE NULL END,
	CASE WHEN (m.metrics::jsonb ? 'precision') THEN ROUND((m.metrics::jsonb->>'precision')::numeric, 6) ELSE NULL END,
	CASE WHEN (m.metrics::jsonb ? 'recall') THEN ROUND((m.metrics::jsonb->>'recall')::numeric, 6) ELSE NULL END,
	CASE WHEN (m.metrics::jsonb ? 'f1_score') THEN ROUND((m.metrics::jsonb->>'f1_score')::numeric, 6) ELSE NULL END,
	CURRENT_TIMESTAMP
FROM gpu_metrics_temp m
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

-- Cleanup
DROP TABLE IF EXISTS gpu_model_temp;
DROP TABLE IF EXISTS gpu_metrics_temp;

\echo 'Test completed successfully'
