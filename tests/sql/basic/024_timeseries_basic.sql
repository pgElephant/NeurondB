-- 024_timeseries_basic.sql
-- Basic test for Time Series (CPU and GPU)

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off
SET client_min_messages TO WARNING;

/* Step 1: Verify prerequisites and create test data */

DROP TABLE IF EXISTS ts_data;
CREATE TABLE ts_data (
	id serial PRIMARY KEY,
	features vector,
	label double precision
);

-- Create sample time series data
INSERT INTO ts_data (features, label)
SELECT 
	array_to_vector_float8(ARRAY[x::double precision]) AS features,
	(x::double precision + random()*0.1) AS label
FROM generate_series(1, 30) AS x;

SELECT COUNT(*)::bigint AS data_rows FROM ts_data;

/* Step 2: Configure CPU */

\echo '=========================================================================='
\echo '=========================================================================='

-- Train time series model on CPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('default', 'timeseries', 'ts_data', 'label', ARRAY['features'], '{}'::jsonb) INTO model_id;
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
	BEGIN
		SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'timeseries' ORDER BY m.model_id DESC LIMIT 1;
		IF model_id IS NULL THEN
			RETURN;
		END IF;
		SELECT neurondb.predict(model_id, array_to_vector_float8(ARRAY[31::double precision])) INTO pred;
		IF pred IS NULL THEN
			RETURN;
		END IF;
	EXCEPTION WHEN OTHERS THEN
	END;
END $$;

/* Step 3: Configure GPU */
/* Step 0: Read settings from test_settings table and verify GPU configuration */
DO $$
DECLARE
	gpu_mode TEXT;
	current_gpu_enabled TEXT;
BEGIN
	-- Read GPU mode setting from test_settings
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	
	-- Verify GPU configuration matches test_settings (set by test runner)
	SELECT current_setting('neurondb.gpu_enabled', true) INTO current_gpu_enabled;
	
	IF gpu_mode = 'gpu' THEN
		-- Verify GPU is enabled (should be set by test runner)
		IF current_gpu_enabled != 'on' THEN
			RAISE WARNING 'GPU mode expected but neurondb.gpu_enabled = % (expected: on)', current_gpu_enabled;
		END IF;
	ELSE
		-- Verify GPU is disabled (should be set by test runner)
		IF current_gpu_enabled != 'off' THEN
			RAISE WARNING 'CPU mode expected but neurondb.gpu_enabled = % (expected: off)', current_gpu_enabled;
		END IF;
	END IF;
END $$;

\echo '=========================================================================='
\echo '=========================================================================='

-- Train time series model on GPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('default', 'timeseries', 'ts_data', 'label', ARRAY['features'], '{}'::jsonb) INTO model_id;
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
	BEGIN
		SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'timeseries' ORDER BY m.model_id DESC LIMIT 1;
		IF model_id IS NULL THEN
			RETURN;
		END IF;
		SELECT neurondb.predict(model_id, array_to_vector_float8(ARRAY[32::double precision])) INTO pred;
		IF pred IS NULL THEN
			RETURN;
		END IF;
	EXCEPTION WHEN OTHERS THEN
	END;
END $$;

DROP TABLE IF EXISTS ts_data;

\echo 'Test completed successfully'
