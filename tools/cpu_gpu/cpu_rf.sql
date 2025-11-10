\timing on
\pset footer off

\echo '=========================================================================='
\echo ' NeuronDB CPU vs GPU Benchmark (CPU Random Forest)'
\echo '=========================================================================='
\echo ''

\echo 'Step 4: CPU training (GPU disabled)...'
SET neurondb.gpu_enabled = off;
SELECT pg_stat_reset();

\echo '   Training random forest on CPU via unified API...'
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
) AS cpu_model_id \gset

\echo '   Evaluating CPU model on test set...'
WITH preds AS (
	SELECT
		label::int AS actual,
		CASE
			WHEN predict_random_forest(:cpu_model_id::integer, features) >= 0.5 THEN 1
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
SELECT
	accuracy AS cpu_accuracy,
	precision AS cpu_precision,
	recall AS cpu_recall,
	f1 AS cpu_f1
FROM calc
\gset

\echo '   CPU metrics:'
SELECT
	format('%-15s', 'Accuracy') AS metric,
	ROUND(:cpu_accuracy::numeric, 4) AS value
UNION ALL
SELECT
	format('%-15s', 'Precision'),
	ROUND(:cpu_precision::numeric, 4)
UNION ALL
SELECT
	format('%-15s', 'Recall'),
	ROUND(:cpu_recall::numeric, 4)
UNION ALL
SELECT
	format('%-15s', 'F1 Score'),
	ROUND(:cpu_f1::numeric, 4);
\echo ''

\echo 'Step 5: Summary'
DO $$
DECLARE
	gpu_acc numeric := NULL;
	gpu_f1 numeric := NULL;
	cpu_acc numeric := NULL;
	cpu_f1 numeric := NULL;
BEGIN
	BEGIN
		gpu_acc := ROUND((current_setting('gpu_metrics', true)::jsonb ->> 'accuracy')::numeric, 4);
		gpu_f1 := ROUND((current_setting('gpu_metrics', true)::jsonb ->> 'f1_score')::numeric, 4);
	EXCEPTION WHEN OTHERS THEN
		gpu_acc := NULL;
		gpu_f1 := NULL;
	END;

	BEGIN
		cpu_acc := ROUND(:cpu_accuracy::numeric, 4);
		cpu_f1 := ROUND(:cpu_f1::numeric, 4);
	EXCEPTION WHEN OTHERS THEN
		cpu_acc := NULL;
		cpu_f1 := NULL;
	END;

	RAISE NOTICE 'run     | accuracy | f1_score';
	IF gpu_acc IS NOT NULL AND gpu_f1 IS NOT NULL THEN
		RAISE NOTICE 'GPU     | %        | %', gpu_acc, gpu_f1;
	END IF;
	IF cpu_acc IS NOT NULL AND cpu_f1 IS NOT NULL THEN
		RAISE NOTICE 'CPU     | %        | %', cpu_acc, cpu_f1;
	END IF;
END
$$;
\echo ''

\echo 'CPU random forest benchmark completed.'
\echo '=========================================================================='

