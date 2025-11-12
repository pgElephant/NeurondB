\timing on
\pset footer off

SET neurondb.gpu_enabled = off;

-- Train
SELECT neurondb.train(
	'logistic_regression',
	'sample_train',
	'features',
	'label',
	'{"max_iters": 1000, "learning_rate": 0.01, "lambda": 0.001}'::jsonb
) AS cpu_model_id \gset

SELECT
	AVG((CASE WHEN neurondb.predict(:cpu_model_id::integer, features) >= 0.5 THEN 1 ELSE 0 END = label::int)::int)::float8 AS accuracy
FROM sample_test;

SELECT neurondb.evaluate(
	:cpu_model_id::integer,
	'sample_test',
	'features',
	'label'
) AS cpu_metrics \gset

SELECT
	format('%-15s', 'Accuracy') AS metric,
	ROUND((:cpu_metrics::jsonb ->> 'accuracy')::numeric, 4) AS value
UNION ALL
SELECT
	format('%-15s', 'Precision'),
	ROUND((:cpu_metrics::jsonb ->> 'precision')::numeric, 4)
UNION ALL
SELECT
	format('%-15s', 'Recall'),
	ROUND((:cpu_metrics::jsonb ->> 'recall')::numeric, 4)
UNION ALL
SELECT
	format('%-15s', 'F1 Score'),
	ROUND((:cpu_metrics::jsonb ->> 'f1_score')::numeric, 4);

