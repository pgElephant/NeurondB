\timing on
\pset footer off

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,rf_split,rf_predict,rf_train';
SELECT neurondb_gpu_enable();

SELECT neurondb.train(
	'random_forest',
	'sample_train',
	'features',
	'label',
	'{}'::jsonb
) AS gpu_model_id \gset

SELECT
	AVG((CASE WHEN predict_random_forest(:gpu_model_id::integer, features) >= 0.5 THEN 1 ELSE 0 END = label::int)::int)::float8 AS accuracy
FROM sample_test;

SELECT neurondb.evaluate(
	:gpu_model_id::integer,
	'sample_test',
	'features',
	'label'
) AS gpu_metrics \gset

SELECT
	format('%-15s', 'Accuracy') AS metric,
	ROUND((:'gpu_metrics'::jsonb ->> 'accuracy')::numeric, 4) AS value
UNION ALL
SELECT
	format('%-15s', 'Precision'),
	ROUND((:'gpu_metrics'::jsonb ->> 'precision')::numeric, 4)
UNION ALL
SELECT
	format('%-15s', 'Recall'),
	ROUND((:'gpu_metrics'::jsonb ->> 'recall')::numeric, 4)
UNION ALL
SELECT
	format('%-15s', 'F1 Score'),
	ROUND((:'gpu_metrics'::jsonb ->> 'f1_score')::numeric, 4);
