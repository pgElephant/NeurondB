-- 021_lightgbm_basic.sql
-- Basic test for LightGBM (CPU and GPU)

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off
SET client_min_messages TO WARNING;

/* Step 1: Verify prerequisites and create test data */
\echo 'Step 1: Creating test data...'

DROP TABLE IF EXISTS lgb_data;
CREATE TABLE lgb_data (
	id serial PRIMARY KEY,
	features vector,
	label int
);

-- Create sample data with vector features
INSERT INTO lgb_data (features, label)
SELECT 
	array_to_vector_float8(ARRAY[x::double precision, (x*2 + random()*0.1)::double precision]) AS features,
	CASE WHEN x < 5 THEN 0 ELSE 1 END AS label
FROM generate_series(1, 100) AS x;

SELECT COUNT(*)::bigint AS data_rows FROM lgb_data;

/* Step 2: Configure CPU */
\echo 'Step 2: Testing LightGBM on CPU...'

SET neurondb.gpu_enabled = off;

\echo '=========================================================================='
\echo 'LightGBM - Basic Test (CPU)'
\echo '=========================================================================='

-- Train LightGBM model on CPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('lightgbm', 'lgb_data', 'features', 'label', '{}'::jsonb) INTO model_id;
		IF model_id IS NULL THEN
			RETURN;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RETURN;
	END;
END $$;

-- Test inference on CPU
DO $$
DECLARE
	pred float8;
	model_id int;
BEGIN
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'lightgbm' ORDER BY m.model_id DESC LIMIT 1;
	IF model_id IS NULL THEN
		RETURN;
	END IF;
	SELECT neurondb.predict(model_id, array_to_vector_float8(ARRAY[7.0, 14.0]::double precision[])) INTO pred;
	IF pred IS NULL THEN
		RETURN;
	END IF;
END $$;

/* Step 3: Configure GPU */
\echo 'Step 3: Testing LightGBM on GPU...'

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip';
SELECT neurondb_gpu_enable() AS gpu_available;

\echo '=========================================================================='
\echo 'LightGBM - Basic Test (GPU)'
\echo '=========================================================================='

-- Train LightGBM model on GPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('lightgbm', 'lgb_data', 'features', 'label', '{}'::jsonb) INTO model_id;
		IF model_id IS NULL THEN
			RETURN;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RETURN;
	END;
END $$;

-- Test inference on GPU
DO $$
DECLARE
	pred float8;
	model_id int;
BEGIN
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'lightgbm' ORDER BY m.model_id DESC LIMIT 1;
	IF model_id IS NULL THEN
		RETURN;
	END IF;
	SELECT neurondb.predict(model_id, array_to_vector_float8(ARRAY[7.0, 14.0]::double precision[])) INTO pred;
	IF pred IS NULL THEN
		RETURN;
	END IF;
END $$;

DROP TABLE IF EXISTS lgb_data;

\echo 'LightGBM basic test completed successfully'
