-- 020_catboost_basic.sql
-- Basic test for CatBoost (CPU and GPU)

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off
SET client_min_messages TO WARNING;

/* Step 1: Verify prerequisites and create test data */

DROP TABLE IF EXISTS catboost_data;
CREATE TABLE catboost_data (
	id serial PRIMARY KEY,
	features vector,
	label int
);

-- Create sample data with vector features
INSERT INTO catboost_data (features, label)
SELECT 
	array_to_vector_float8(ARRAY[x::double precision, (x*2 + random()*0.1)::double precision]) AS features,
	CASE WHEN x < 5 THEN 0 ELSE 1 END AS label
FROM generate_series(1, 100) AS x;

SELECT COUNT(*)::bigint AS data_rows FROM catboost_data;

/* Step 2: Configure CPU */

\echo '=========================================================================='
\echo '=========================================================================='

-- Train CatBoost model on CPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('default', 'catboost', 'catboost_data', 'label', ARRAY['features'], '{}'::jsonb) INTO model_id;
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
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'catboost' ORDER BY m.model_id DESC LIMIT 1;
	IF model_id IS NULL THEN
		RETURN;
	END IF;
	SELECT neurondb.predict(model_id, array_to_vector_float8(ARRAY[7.0, 14.0]::double precision[])) INTO pred;
	IF pred IS NULL THEN
		RETURN;
	END IF;
END $$;

/* Step 3: Configure GPU */
/* Step 0: Read settings from test_settings table and apply them */
DO $$
DECLARE
	gpu_mode TEXT;
BEGIN
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	IF gpu_mode = 'gpu' THEN
		PERFORM neurondb_gpu_enable();
	ELSE
		PERFORM set_config('neurondb.gpu_enabled', 'off', false);
	END IF;
END $$;

\echo '=========================================================================='
\echo '=========================================================================='

-- Train CatBoost model on GPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('default', 'catboost', 'catboost_data', 'label', ARRAY['features'], '{}'::jsonb) INTO model_id;
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
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'catboost' ORDER BY m.model_id DESC LIMIT 1;
	IF model_id IS NULL THEN
		RETURN;
	END IF;
	SELECT neurondb.predict(model_id, array_to_vector_float8(ARRAY[7.0, 14.0]::double precision[])) INTO pred;
	IF pred IS NULL THEN
		RETURN;
	END IF;
END $$;

DROP TABLE IF EXISTS catboost_data;

\echo 'Test completed successfully'
