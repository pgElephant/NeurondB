\set ON_ERROR_STOP on
\timing on
\pset footer off

\if :{?sample_count}
\else
\set sample_count 100
\endif

\echo Using sample_count = :sample_count

\echo '=========================================================================='
\echo ' NeuronDB CPU vs GPU Benchmark (Dataset Preparation)'
\echo '=========================================================================='
\echo ''

\echo 'Step 0: Ensuring NeuronDB extension is available...'
CREATE EXTENSION IF NOT EXISTS neurondb;

\echo '   Cleaning up previous run (if any)...'
DROP TABLE IF EXISTS neurondb_cpu_gpu_data CASCADE;
DROP TABLE IF EXISTS neurondb_cpu_gpu_train CASCADE;
DROP TABLE IF EXISTS neurondb_cpu_gpu_test CASCADE;
DELETE FROM neurondb.ml_models
WHERE training_table IN ('neurondb_cpu_gpu_train', 'neurondb_cpu_gpu_test');
\echo '   Cleanup complete.'
\echo ''

\echo 'Step 1: Generating synthetic dataset (~1 GB)...'
\echo '   Creating :sample_count samples with 256-dim float32 vectors.'
CREATE TABLE neurondb_cpu_gpu_data AS
SELECT
	gs AS sample_id,
	(
		SELECT array_agg((random() * 2 - 1)::float4 ORDER BY dim)
		FROM generate_series(1, 256) AS dim
	)::vector(256) AS features,
	(random() > 0.5)::int AS label
FROM generate_series(1, :sample_count) AS gs;

ANALYZE neurondb_cpu_gpu_data;
\echo '   Dataset ready.'
\echo ''

\echo 'Step 2: Creating train/test splits (80/20)...'
CREATE TABLE neurondb_cpu_gpu_train AS
SELECT *
FROM neurondb_cpu_gpu_data
WHERE sample_id % 5 <> 0;

CREATE TABLE neurondb_cpu_gpu_test AS
SELECT *
FROM neurondb_cpu_gpu_data
WHERE sample_id % 5 = 0;

ANALYZE neurondb_cpu_gpu_train;
ANALYZE neurondb_cpu_gpu_test;
\echo '   Train/Test tables created.'
\echo ''

\echo 'Dataset preparation completed.'
\echo '=========================================================================='


