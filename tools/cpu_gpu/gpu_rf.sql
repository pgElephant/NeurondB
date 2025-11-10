\timing on
\pset footer off

\echo '=========================================================================='
\echo ' NeuronDB CPU vs GPU Benchmark (GPU Random Forest)'
\echo '=========================================================================='
\echo ''

\echo 'Step 3: GPU training (enable NeuronDB GPU acceleration)...'
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_device = 0;
SELECT pg_stat_reset();

\echo '   Preparing ML catalog (project + defaults)...'
DO $$
DECLARE
	bench_project_id integer;
BEGIN
	INSERT INTO neurondb.ml_projects (project_name, model_type, description)
	VALUES ('cpu_gpu_benchmark',
		'classification'::neurondb.ml_model_type,
		'CPU vs GPU benchmark project')
	ON CONFLICT (project_name)
	DO UPDATE
	SET updated_at = CURRENT_TIMESTAMP
	RETURNING project_id INTO bench_project_id;

	IF bench_project_id IS NULL THEN
		SELECT project_id INTO bench_project_id
		FROM neurondb.ml_projects
		WHERE project_name = 'cpu_gpu_benchmark';
	END IF;

	IF bench_project_id IS NULL THEN
		RAISE EXCEPTION 'Failed to provision benchmark project';
	END IF;

	EXECUTE format(
		'ALTER TABLE neurondb.ml_models ALTER COLUMN project_id SET DEFAULT %s',
		bench_project_id);
END;
$$;

\echo '   Initializing GPU runtime...'
DO $$
DECLARE
	ok boolean;
BEGIN
	ok := neurondb_gpu_enable();
	IF NOT ok THEN
		RAISE EXCEPTION 'GPU initialization failed, aborting benchmark.';
	END IF;
	IF current_setting('neurondb.gpu_enabled') <> 'on' THEN
		RAISE EXCEPTION 'neurondb.gpu_enabled GUC did not remain enabled.';
	END IF;
END;
$$;

\echo '   Training random forest on GPU via unified API...'
SELECT neurondb.train(
	'random_forest',
	'neurondb_cpu_gpu_train',
	'features',
	'label',
	jsonb_build_object(
		'n_trees', 120,
		'max_depth', 12,
		'min_samples_split', 2,
		'max_features', 0
	)
) AS gpu_model_id \gset

\echo '   Evaluating GPU model on test set...'
WITH preds AS (
	SELECT
		label::int AS actual,
		CASE
			WHEN predict_random_forest(:gpu_model_id::integer, features) >= 0.5 THEN 1
			ELSE 0
		END AS predicted
	FROM neurondb_cpu_gpu_test
), metrics AS (
	SELECT
		AVG((predicted = actual)::int)::float8 AS accuracy,
		SUM(CASE WHEN predicted = 1 AND actual = 1 THEN 1 ELSE 0 END)::float8 AS tp,
		SUM(CASE WHEN predicted = 1 AND actual = 0 THEN 1 ELSE 0 END)::float8 AS fp,
		SUM(CASE WHEN predicted = 0 AND actual = 1 THEN 1 ELSE 0 END)::float8 AS fn
	FROM preds
), calc AS (
	SELECT
		accuracy,
		CASE WHEN tp + fp > 0 THEN tp / (tp + fp) ELSE 0 END AS precision,
		CASE WHEN tp + fn > 0 THEN tp / (tp + fn) ELSE 0 END AS recall,
		CASE WHEN (2 * tp + fp + fn) > 0 THEN (2 * tp) / (2 * tp + fp + fn) ELSE 0 END AS f1
	FROM metrics
)
SELECT jsonb_build_object(
	'accuracy', accuracy,
	'precision', precision,
	'recall', recall,
	'f1_score', f1
) AS gpu_metrics FROM calc \gset

\echo '   GPU metrics:'
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
\echo ''

\echo 'GPU random forest benchmark completed.'
\echo '=========================================================================='


