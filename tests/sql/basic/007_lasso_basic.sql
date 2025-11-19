\timing on
\pset footer off
\pset pager off

-- This test uses test_train_view and test_test_view tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'test_train_view table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_test') THEN
		RAISE EXCEPTION 'test_test_view table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
END
$$;

-- Create views with 1000 rows for basic tests
DROP VIEW IF EXISTS test_train_view;
DROP VIEW IF EXISTS test_test_view;

CREATE VIEW test_train_view AS
SELECT features, label FROM sample_train LIMIT 1000;

CREATE VIEW test_test_view AS
SELECT features, label FROM sample_test LIMIT 1000;

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,lasso_train,lasso_predict';
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

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
BEGIN
	SELECT model_id INTO mid FROM gpu_model_temp LIMIT 1;
	metrics_result := neurondb.evaluate(mid, 'test_test_view', 'features', 'label');
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

