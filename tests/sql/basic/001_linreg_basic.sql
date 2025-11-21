/*-------------------------------------------------------------------------
 *
 * 001_linreg.sql
 *    Linear Regression test
 *
 *    Step-by-step test with clean output and timing
 *
 *-------------------------------------------------------------------------*/

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

/* Step 0: Read settings from test_settings table and apply them */
DO $$
DECLARE
	gpu_mode TEXT;
BEGIN
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	IF gpu_mode = 'gpu' THEN
		PERFORM neurondb_gpu_enable();
	ELSE
		PERFORM set_config('neurondb.gpu_enabled', 'off', false);
	END IF;
END $$;

DROP TABLE IF EXISTS gpu_model_temp;

SELECT 
	(SELECT COUNT(*)::bigint FROM test_train_view) AS train_rows,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_rows;

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

CREATE TEMP TABLE gpu_model_temp AS
SELECT 
	neurondb.train(
		'default',
		'linear_regression',
		'test_train_view',
		'label',
		ARRAY['features'],
		'{}'::jsonb
	)::integer AS model_id;

SELECT model_id FROM gpu_model_temp;

SELECT 
	m.algorithm::text AS algorithm,
	COALESCE((m.metrics::jsonb->>'n_samples')::bigint, m.num_samples::bigint, 0) AS n_samples,
	COALESCE((m.metrics::jsonb->>'n_features')::integer, m.num_features, 0) AS n_features,
	COALESCE(m.metrics::jsonb->>'storage', 'default') AS storage,
	ROUND(COALESCE((m.metrics::jsonb->>'mse')::numeric, 0), 6) AS mse,
	ROUND(COALESCE((m.metrics::jsonb->>'mae')::numeric, 0), 6) AS mae,
	ROUND(COALESCE((m.metrics::jsonb->>'r_squared')::numeric, 0), 6) AS r_squared
FROM neurondb.ml_models m, gpu_model_temp t
WHERE m.model_id = t.model_id;

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
	'MSE' AS metric,
	ROUND((metrics->>'mse')::numeric, 6)::text AS value
FROM gpu_metrics_temp
WHERE metrics->>'mse' IS NOT NULL
UNION ALL
SELECT 'RMSE', ROUND((metrics->>'rmse')::numeric, 6)::text
FROM gpu_metrics_temp
WHERE metrics->>'rmse' IS NOT NULL
UNION ALL
SELECT 'MAE', ROUND((metrics->>'mae')::numeric, 6)::text
FROM gpu_metrics_temp
WHERE metrics->>'mae' IS NOT NULL
UNION ALL
SELECT 'R²', ROUND((metrics->>'r_squared')::numeric, 6)::text
FROM gpu_metrics_temp
WHERE metrics->>'r_squared' IS NOT NULL
ORDER BY metric;

SELECT 
	(SELECT model_id FROM gpu_model_temp) AS model_id,
	(SELECT COUNT(*)::bigint FROM test_train_view) AS train_samples,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_samples,
	(SELECT ROUND((metrics->>'mse')::numeric, 6) FROM gpu_metrics_temp) AS final_mse;

-- Store results in test_metrics table
INSERT INTO test_metrics (
	test_name, algorithm, model_id, train_samples, test_samples,
	mse, rmse, mae, r_squared, updated_at
)
SELECT 
	'001_linreg_basic',
	'linear_regression',
	(SELECT model_id FROM gpu_model_temp),
	(SELECT COUNT(*)::bigint FROM test_train_view),
	(SELECT COUNT(*)::bigint FROM test_test_view),
	ROUND((metrics->>'mse')::numeric, 6),
	ROUND((metrics->>'rmse')::numeric, 6),
	ROUND((metrics->>'mae')::numeric, 6),
	ROUND((metrics->>'r_squared')::numeric, 6),
	CURRENT_TIMESTAMP
FROM gpu_metrics_temp
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

DROP TABLE IF EXISTS gpu_model_temp;

\echo 'Test completed successfully'
