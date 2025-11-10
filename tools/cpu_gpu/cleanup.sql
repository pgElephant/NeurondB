\timing on
\pset footer off

\echo '=========================================================================='
\echo ' NeuronDB CPU vs GPU Benchmark (Cleanup)'
\echo '=========================================================================='
\echo ''

\if :{?cpu_model_id}
\echo 'Dropping CPU model...'
SELECT neurondb.drop_model(:cpu_model_id::integer);
\else
\echo 'cpu_model_id not set; skipping CPU model drop.'
\endif

\if :{?gpu_model_id}
\echo 'Dropping GPU model...'
SELECT neurondb.drop_model(:gpu_model_id::integer);
\else
\echo 'gpu_model_id not set; skipping GPU model drop.'
\endif

\echo 'Dropping benchmark tables...'
DROP TABLE IF EXISTS neurondb_cpu_gpu_test CASCADE;
DROP TABLE IF EXISTS neurondb_cpu_gpu_train CASCADE;
DROP TABLE IF EXISTS neurondb_cpu_gpu_data CASCADE;

\echo 'Cleanup complete.'
\echo '=========================================================================='


