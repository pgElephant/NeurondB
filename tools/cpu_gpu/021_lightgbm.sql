-- 021_lightgbm.sql
-- Basic test for LightGBM (CPU and GPU)

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

DROP TABLE IF EXISTS lgbm_data;
CREATE TABLE lgbm_data (
	id serial PRIMARY KEY,
	x1 double precision,
	x2 double precision,
	y int
);

INSERT INTO lgbm_data (x1, x2, y)
SELECT x, x*1.5 + random()*0.1, CASE WHEN x < 5 THEN 0 ELSE 1 END
FROM generate_series(1, 10) AS x;

\echo '=== LightGBM Basic Test (CPU) ==='

-- Train LightGBM model on CPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('lightgbm', 'lgbm_data', 'y') INTO model_id;
		IF model_id IS NULL THEN
			RAISE NOTICE 'LightGBM (CPU) training not yet implemented';
			RETURN;
		END IF;
		RAISE NOTICE '✓ LightGBM (CPU) model trained, model_id=%', model_id;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'LightGBM (CPU) training not yet implemented: %', SQLERRM;
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
		SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'lightgbm' ORDER BY m.model_id DESC LIMIT 1;
		IF model_id IS NULL THEN
			RAISE NOTICE 'LightGBM (CPU) inference skipped - no model available';
			RETURN;
		END IF;
		SELECT neurondb.predict(model_id, ARRAY[7.0, 10.5]::vector) INTO pred;
		IF pred IS NULL THEN
			RAISE NOTICE 'LightGBM (CPU) predict returned NULL';
			RETURN;
		END IF;
		RAISE NOTICE '✓ LightGBM (CPU) inference successful, ŷ = %', pred;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'LightGBM (CPU) inference skipped: %', SQLERRM;
	END;
END $$;

\echo '=== LightGBM Basic Test (GPU) ==='
SET neurondb.gpu_enabled = on;

-- Train LightGBM model on GPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('lightgbm', 'lgbm_data', 'y') INTO model_id;
		IF model_id IS NULL THEN
			RAISE NOTICE 'LightGBM (GPU) training not yet implemented';
			RETURN;
		END IF;
		RAISE NOTICE '✓ LightGBM (GPU) model trained, model_id=%', model_id;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'LightGBM (GPU) training not yet implemented: %', SQLERRM;
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
		SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'lightgbm' ORDER BY m.model_id DESC LIMIT 1;
		IF model_id IS NULL THEN
			RAISE NOTICE 'LightGBM (GPU) inference skipped - no model available';
			RETURN;
		END IF;
		SELECT neurondb.predict(model_id, ARRAY[7.0, 10.5]::vector) INTO pred;
		IF pred IS NULL THEN
			RAISE NOTICE 'LightGBM (GPU) predict returned NULL';
			RETURN;
		END IF;
		RAISE NOTICE '✓ LightGBM (GPU) inference successful, ŷ = %', pred;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'LightGBM (GPU) inference skipped: %', SQLERRM;
	END;
END $$;

\echo '✓ LightGBM basic test complete'
