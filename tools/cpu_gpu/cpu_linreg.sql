\timing on
\pset footer off

SET neurondb.gpu_enabled = off;

-- Train
SELECT neurondb.train(
	'linear_regression',
	'sample_train',
	'features',
	'label',
	'{}'::jsonb
) AS cpu_model_id \gset

-- Calculate predictions and metrics
SELECT
	AVG(POWER(neurondb.predict(:cpu_model_id::integer, features) - label, 2))::float8 AS mse,
	SQRT(AVG(POWER(neurondb.predict(:cpu_model_id::integer, features) - label, 2)))::float8 AS rmse,
	AVG(ABS(neurondb.predict(:cpu_model_id::integer, features) - label))::float8 AS mae
FROM sample_test;

SELECT neurondb.evaluate(
	:cpu_model_id::integer,
	'sample_test',
	'features',
	'label'
) AS cpu_metrics \gset

SELECT
	format('%-15s', 'MSE') AS metric,
	ROUND((:cpu_metrics::jsonb ->> 'mse')::numeric, 4) AS value
UNION ALL
SELECT
	format('%-15s', 'RMSE'),
	ROUND((:cpu_metrics::jsonb ->> 'rmse')::numeric, 4)
UNION ALL
SELECT
	format('%-15s', 'MAE'),
	ROUND((:cpu_metrics::jsonb ->> 'mae')::numeric, 4)
UNION ALL
SELECT
	format('%-15s', 'R²'),
	ROUND((:cpu_metrics::jsonb ->> 'r_squared')::numeric, 4);

