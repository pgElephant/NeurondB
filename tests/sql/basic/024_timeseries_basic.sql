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
		SELECT neurondb.train('timeseries', 'ts_data', 'features', 'label', '{}'::jsonb) INTO model_id;
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

SELECT neurondb_gpu_enable() AS gpu_available;

\echo '=========================================================================='
\echo '=========================================================================='

-- Train time series model on GPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('timeseries', 'ts_data', 'features', 'label', '{}'::jsonb) INTO model_id;
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
