/*-------------------------------------------------------------------------
 *
 * 001_linreg.sql
 *    Linear Regression test with GPU acceleration
 *
 *    Step-by-step test with clean output and timing
 *
 *-------------------------------------------------------------------------*/
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,linreg_train,linreg_predict';

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables 
		WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist';
	END IF;
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables 
		WHERE table_schema = 'public' AND table_name = 'sample_test') THEN
		RAISE EXCEPTION 'sample_test table does not exist';
	END IF;
END
$$;

-- Create views with 1000 rows for basic tests
DROP VIEW IF EXISTS test_train_view;
DROP VIEW IF EXISTS test_test_view;
DROP TABLE IF EXISTS gpu_model_temp;

CREATE VIEW test_train_view AS
SELECT features, label FROM sample_train LIMIT 1000;

CREATE VIEW test_test_view AS
SELECT features, label FROM sample_test LIMIT 1000;

SELECT 
	(SELECT COUNT(*)::bigint FROM test_train_view) AS train_rows,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_rows;


SELECT 
	neurondb_gpu_enable() AS gpu_available,
	current_setting('neurondb.gpu_enabled') AS gpu_enabled,
	current_setting('neurondb.gpu_kernels') AS gpu_kernels;

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
		'linear_regression',
		'test_train_view',
		'features',
		'label',
		'{}'::jsonb
	)::integer AS model_id;

SELECT model_id FROM gpu_model_temp;

SELECT 
	m.algorithm::text AS algorithm,
	COALESCE((m.metrics::jsonb->>'n_samples')::bigint, m.num_samples::bigint, 0) AS n_samples,
	COALESCE((m.metrics::jsonb->>'n_features')::integer, m.num_features, 0) AS n_features,
	COALESCE(m.metrics::jsonb->>'storage', 'cpu') AS storage,
	ROUND(COALESCE((m.metrics::jsonb->>'mse')::numeric, 0), 6) AS mse,
	ROUND(COALESCE((m.metrics::jsonb->>'mae')::numeric, 0), 6) AS mae,
	ROUND(COALESCE((m.metrics::jsonb->>'r_squared')::numeric, 0), 6) AS r_squared
FROM neurondb.ml_models m, gpu_model_temp t
WHERE m.model_id = t.model_id;

/* Step 6: Test set statistics */
\echo 'Step 6: Test set statistics...'

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

DROP VIEW IF EXISTS test_train_view;
DROP VIEW IF EXISTS test_test_view;
DROP TABLE IF EXISTS gpu_model_temp;

\echo 'Test completed successfully'
