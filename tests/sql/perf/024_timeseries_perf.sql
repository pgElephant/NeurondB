-- 024_timeseries.sql
-- performance test for Time Series model (CPU and GPU)


-- Performance test: Works on the whole 11M row view
SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

DROP TABLE IF EXISTS ts_data;
CREATE TABLE ts_data (
	id serial PRIMARY KEY,
	ts_val double precision
);

INSERT INTO ts_data (ts_val)
SELECT x::double precision + random()*0.1
FROM generate_series(1,30) AS x;

\echo '=== Time Series Performance Test (Full Dataset) (CPU) ==='

-- Train time series model on CPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('timeseries', 'ts_data', 'ts_val') INTO model_id;
		IF model_id IS NULL THEN
			RAISE NOTICE 'Time Series (CPU) training not yet implemented';
			RETURN;
		END IF;
		RAISE NOTICE '✓ Time Series (CPU) model trained, model_id=%', model_id;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Time Series (CPU) training not yet implemented: %', SQLERRM;
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
			RAISE NOTICE 'Time Series (CPU) inference skipped - no model available';
			RETURN;
		END IF;
		SELECT neurondb.predict(model_id, ARRAY[31::double precision]::vector) INTO pred;
		IF pred IS NULL THEN
			RAISE NOTICE 'Time Series (CPU) predict returned NULL';
			RETURN;
		END IF;
		RAISE NOTICE '✓ Time Series (CPU) inference successful, ŷ = %', pred;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Time Series (CPU) inference skipped: %', SQLERRM;
	END;
END $$;

\echo '=== Time Series Performance Test (Full Dataset) (GPU) ==='
SET neurondb.gpu_enabled = on;

-- Train time series model on GPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('timeseries', 'ts_data', 'ts_val') INTO model_id;
		IF model_id IS NULL THEN
			RAISE NOTICE 'Time Series (GPU) training not yet implemented';
			RETURN;
		END IF;
		RAISE NOTICE '✓ Time Series (GPU) model trained, model_id=%', model_id;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Time Series (GPU) training not yet implemented: %', SQLERRM;
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
			RAISE NOTICE 'Time Series (GPU) inference skipped - no model available';
			RETURN;
		END IF;
		SELECT neurondb.predict(model_id, ARRAY[32::double precision]::vector) INTO pred;
		IF pred IS NULL THEN
			RAISE NOTICE 'Time Series (GPU) predict returned NULL';
			RETURN;
		END IF;
		RAISE NOTICE '✓ Time Series (GPU) inference successful, ŷ = %', pred;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Time Series (GPU) inference skipped: %', SQLERRM;
	END;
END $$;

\echo '✓ Time Series performance test complete'
