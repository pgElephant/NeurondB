\timing on
\pset footer off
\pset pager off

-- This test uses test_train_view and test_test_view tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist

\set ON_ERROR_STOP on

-- Train model once and store in temp table to reuse
DROP TABLE IF EXISTS gpu_model_temp;
CREATE TEMP TABLE gpu_model_temp AS
SELECT train_ridge_regression(
	'test_train_view',
	'features',
	'label',
	0.01  -- lambda
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

-- Store results in test_metrics table
INSERT INTO test_metrics (
	test_name, algorithm, model_id, train_samples, test_samples,
	mse, rmse, mae, r_squared, updated_at
)
SELECT 
	'006_ridge_basic',
	'ridge',
	(SELECT model_id FROM gpu_model_temp),
	(SELECT COUNT(*)::bigint FROM test_train_view),
	(SELECT COUNT(*)::bigint FROM test_test_view),
	CASE WHEN (m.metrics::jsonb ? 'mse') THEN ROUND((m.metrics::jsonb->>'mse')::numeric, 6) ELSE NULL END,
	CASE WHEN (m.metrics::jsonb ? 'rmse') THEN ROUND((m.metrics::jsonb->>'rmse')::numeric, 6) ELSE NULL END,
	CASE WHEN (m.metrics::jsonb ? 'mae') THEN ROUND((m.metrics::jsonb->>'mae')::numeric, 6) ELSE NULL END,
	CASE WHEN (m.metrics::jsonb ? 'r_squared') THEN ROUND((m.metrics::jsonb->>'r_squared')::numeric, 6) ELSE NULL END,
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
