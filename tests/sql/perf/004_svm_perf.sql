\timing on
\pset footer off
\pset pager off


-- This test uses sample_train and sample_test tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_test') THEN
		RAISE EXCEPTION 'sample_test table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
END
$$;

-- Performance test: Works on the whole 11M row view

SET log_min_messages = debug1;
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,svm_train,svm_predict';
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

-- Train model once and store in temp table to reuse
DROP TABLE IF EXISTS gpu_model_temp;
CREATE TEMP TABLE gpu_model_temp AS
SELECT neurondb.train(
	'svm',
	'sample_train',
	'features',
	'label',
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
	metrics_result := neurondb.evaluate(mid, 'sample_test', 'features', 'label');
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

