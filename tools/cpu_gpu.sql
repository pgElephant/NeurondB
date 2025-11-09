\set ON_ERROR_STOP on
\timing on
\pset footer off

\if :{?sample_count}
\else
\set sample_count 2000
\endif

\echo Using sample_count = :sample_count

\echo '=========================================================================='
\echo ' NeuronDB CPU vs GPU Benchmark (Synthetic ~1 GB Dataset)'
\echo '=========================================================================='
\echo ''

\echo 'Step 0: Ensuring NeuronDB extension is available...'
CREATE EXTENSION IF NOT EXISTS neurondb;

\echo '   Cleaning up previous run (if any)...'
DROP TABLE IF EXISTS neurondb_cpu_gpu_data CASCADE;
DROP TABLE IF EXISTS neurondb_cpu_gpu_train CASCADE;
DROP TABLE IF EXISTS neurondb_cpu_gpu_test CASCADE;
\echo '   Removing prior benchmark models (if any)...'
DELETE FROM neurondb.ml_models 
WHERE training_table IN ('neurondb_cpu_gpu_train', 'neurondb_cpu_gpu_test');
\echo '   Cleanup complete.'
\echo ''

\echo 'Step 1: Generating synthetic dataset (~20 MB)...'
\echo '   Creating :sample_count samples with 256-dim float32 vectors.'
CREATE TABLE neurondb_cpu_gpu_data AS
SELECT
    gs AS sample_id,
    (SELECT array_agg((random() * 2 - 1)::float4 ORDER BY dim) FROM generate_series(1, 256) AS dim)::vector(256) AS features,
    (random() > 0.5)::int AS label
FROM generate_series(1, :sample_count) AS gs;

ANALYZE neurondb_cpu_gpu_data;
\echo '   Dataset ready.'
\echo ''

\echo 'Step 2: Creating train/test splits (80/20)...'
CREATE TABLE neurondb_cpu_gpu_train AS
SELECT * FROM neurondb_cpu_gpu_data WHERE sample_id % 5 <> 0;

CREATE TABLE neurondb_cpu_gpu_test AS
SELECT * FROM neurondb_cpu_gpu_data WHERE sample_id % 5 = 0;

ANALYZE neurondb_cpu_gpu_train;
ANALYZE neurondb_cpu_gpu_test;
\echo '   Train/Test tables created.'
\echo ''

\echo 'Step 3: GPU training (enable NeuronDB GPU acceleration)...'
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_fail_open = on;
SELECT pg_stat_reset();

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
SELECT neurondb.evaluate(
    :gpu_model_id::integer,
    'neurondb_cpu_gpu_test',
    'features',
    'label'
) AS gpu_metrics \gset

\echo '   GPU metrics:'
SELECT 
    format('%-15s', 'Accuracy')  AS metric,
    ROUND(((':' || 'gpu_metrics')::jsonb ->> 'accuracy')::numeric, 4)   AS value
UNION ALL SELECT 
    format('%-15s', 'Precision'),
    ROUND(((':' || 'gpu_metrics')::jsonb ->> 'precision')::numeric, 4)
UNION ALL SELECT 
    format('%-15s', 'Recall'),
    ROUND(((':' || 'gpu_metrics')::jsonb ->> 'recall')::numeric, 4)
UNION ALL SELECT 
    format('%-15s', 'F1 Score'),
    ROUND(((':' || 'gpu_metrics')::jsonb ->> 'f1_score')::numeric, 4);
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
SELECT neurondb.evaluate(
    :cpu_model_id::integer,
    'neurondb_cpu_gpu_test',
    'features',
    'label'
) AS cpu_metrics \gset

\echo '   CPU metrics:'
SELECT 
    format('%-15s', 'Accuracy')  AS metric,
    ROUND(((':' || 'cpu_metrics')::jsonb ->> 'accuracy')::numeric, 4)   AS value
UNION ALL SELECT 
    format('%-15s', 'Precision'),
    ROUND(((':' || 'cpu_metrics')::jsonb ->> 'precision')::numeric, 4)
UNION ALL SELECT 
    format('%-15s', 'Recall'),
    ROUND(((':' || 'cpu_metrics')::jsonb ->> 'recall')::numeric, 4)
UNION ALL SELECT 
    format('%-15s', 'F1 Score'),
    ROUND(((':' || 'cpu_metrics')::jsonb ->> 'f1_score')::numeric, 4);
\echo ''

\echo 'Step 5: Summary'
SELECT
    'GPU',
    ROUND(((':' || 'gpu_metrics')::jsonb ->> 'accuracy')::numeric, 4),
    ROUND(((':' || 'gpu_metrics')::jsonb ->> 'f1_score')::numeric, 4)
UNION ALL
SELECT
    'CPU' AS run,
    ROUND(((':' || 'cpu_metrics')::jsonb ->> 'accuracy')::numeric, 4) AS accuracy,
    ROUND(((':' || 'cpu_metrics')::jsonb ->> 'f1_score')::numeric, 4) AS f1_score;
\echo ''

\echo 'Step 6: Cleanup models created during benchmark...'
SELECT neurondb.drop_model(:cpu_model_id::integer);
SELECT neurondb.drop_model(:gpu_model_id::integer);
\echo ''

\echo 'TIP: Enable \timing in psql to compare training durations.'
\echo 'Completed NeuronDB CPU vs GPU benchmark.'
\echo '=========================================================================='

