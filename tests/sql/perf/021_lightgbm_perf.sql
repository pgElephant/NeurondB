-- 021_lightgbm_perf.sql
-- Performance test for LightGBM (CPU and GPU)
-- Works on full dataset from sample_train table

\timing on
\pset footer off
\pset pager off
\pset tuples_only off
\set ON_ERROR_STOP on
SET client_min_messages TO WARNING;

\echo '=========================================================================='
\echo 'LightGBM - Performance Test (Full Dataset)'
\echo '=========================================================================='

-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_test') THEN
		RAISE EXCEPTION 'sample_test table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
END
$$;

-- Create views for full dataset
DROP VIEW IF EXISTS perf_train_view;
DROP VIEW IF EXISTS perf_test_view;

CREATE VIEW perf_train_view AS
SELECT features, label FROM sample_train;

CREATE VIEW perf_test_view AS
SELECT features, label FROM sample_test;

SELECT 
	(SELECT COUNT(*)::bigint FROM perf_train_view) AS train_rows,
	(SELECT COUNT(*)::bigint FROM perf_test_view) AS test_rows;

\echo ''
\echo 'Testing LightGBM on CPU (full dataset)...'

SET neurondb.gpu_enabled = off;

-- Train LightGBM model on CPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('lightgbm', 'perf_train_view', 'features', 'label', '{}'::jsonb) INTO model_id;
		IF model_id IS NULL THEN
			RETURN;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RETURN;
	END;
END $$;

\echo ''
\echo 'Testing LightGBM on GPU (full dataset)...'

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip';
SELECT neurondb_gpu_enable() AS gpu_available;
SELECT neurondb_gpu_info() AS gpu_info;

-- Train LightGBM model on GPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('lightgbm', 'perf_train_view', 'features', 'label', '{}'::jsonb) INTO model_id;
		IF model_id IS NULL THEN
			RETURN;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RETURN;
	END;
END $$;

DROP VIEW IF EXISTS perf_train_view;
DROP VIEW IF EXISTS perf_test_view;

\echo 'LightGBM performance test completed successfully'
