\timing on
\pset footer off
\pset pager off

-- This test uses sample_train and sample_test tables created by ml_dataset.py
-- Performance test: Works on the whole 11M row view
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

SET neurondb.gpu_enabled = on;
SET neurondb.automl.use_gpu = on;
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'AutoML - Performance Test (Full Dataset)'
\echo '=========================================================================='

-- Test AutoML for classification task on full dataset
DROP TABLE IF EXISTS automl_model_temp;
CREATE TEMP TABLE automl_model_temp AS
SELECT auto_train(
	'sample_train',
	'features',
	'label',
	'classification',
	'accuracy'
)::integer AS model_id;

SELECT model_id FROM automl_model_temp;

-- Evaluate the best model on full test set
CREATE TEMP TABLE automl_metrics_temp (metrics jsonb);
DO $$
DECLARE
	mid integer;
	metrics_result jsonb;
BEGIN
	SELECT model_id INTO mid FROM automl_model_temp LIMIT 1;
	metrics_result := neurondb.evaluate(mid, 'sample_test', 'features', 'label');
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

\echo 'AutoML performance test completed successfully'

