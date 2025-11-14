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

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,lasso_train,lasso_predict';
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

-- Train model once and store in temp table to reuse
DROP TABLE IF EXISTS gpu_model_temp;
CREATE TEMP TABLE gpu_model_temp AS
SELECT train_lasso_regression(
	'sample_train',
	'features',
	'label',
	0.01,  -- lambda
	1000   -- max_iters
)::integer AS model_id;

-- Debug: Show model_id
SELECT model_id FROM gpu_model_temp;

-- Calculate predictions and metrics
SELECT
	AVG(POWER(predict_lasso_regression_model_id(m.model_id, features) - label, 2))::float8 AS mse,
	SQRT(AVG(POWER(predict_lasso_regression_model_id(m.model_id, features) - label, 2)))::float8 AS rmse,
	AVG(ABS(predict_lasso_regression_model_id(m.model_id, features) - label))::float8 AS mae
FROM sample_test, gpu_model_temp m;

-- Evaluate model and store result
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

-- Show metrics (for regression, we show different metrics)
SELECT
	format('%-15s', 'MSE') AS metric,
	CASE WHEN (m.metrics::jsonb ? 'mse')
		THEN ROUND((m.metrics::jsonb ->> 'mse')::numeric, 4)
		ELSE NULL END AS value
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'RMSE'),
	CASE WHEN (m.metrics::jsonb ? 'rmse')
		THEN ROUND((m.metrics::jsonb ->> 'rmse')::numeric, 4)
		ELSE NULL END
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'MAE'),
	CASE WHEN (m.metrics::jsonb ? 'mae')
		THEN ROUND((m.metrics::jsonb ->> 'mae')::numeric, 4)
		ELSE NULL END
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'R²'),
	CASE WHEN (m.metrics::jsonb ? 'r_squared')
		THEN ROUND((m.metrics::jsonb ->> 'r_squared')::numeric, 4)
		ELSE NULL END
FROM gpu_metrics_temp m;

-- Cleanup
DROP TABLE IF EXISTS gpu_model_temp;
DROP TABLE IF EXISTS gpu_metrics_temp;

