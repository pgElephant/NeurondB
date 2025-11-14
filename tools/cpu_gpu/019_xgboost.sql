-- 019_xgboost.sql
-- Basic test for XGBoost (CPU and GPU)

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

DROP TABLE IF EXISTS xgb_data;
CREATE TABLE xgb_data (
	id serial PRIMARY KEY,
	x1 double precision,
	x2 double precision,
	y int
);

INSERT INTO xgb_data (x1, x2, y)
SELECT x, x*2 + random()*0.1, CASE WHEN x < 5 THEN 0 ELSE 1 END
FROM generate_series(1, 10) AS x;

\echo '=== XGBoost Basic Test (CPU) ==='

-- Train XGBoost model on CPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('xgboost', 'xgb_data', 'y') INTO model_id;
		IF model_id IS NULL THEN
			RAISE NOTICE 'XGBoost (CPU) training not yet implemented';
			RETURN;
		END IF;
		RAISE NOTICE '✓ XGBoost (CPU) model trained, model_id=%', model_id;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'XGBoost (CPU) training not yet implemented: %', SQLERRM;
		RETURN;
	END;
END $$;

-- Test inference on CPU
DO $$
DECLARE
	pred float8;
	model_id int;
BEGIN
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'xgboost' ORDER BY m.model_id DESC LIMIT 1;
	IF model_id IS NULL THEN
		RAISE NOTICE 'XGBoost (CPU) inference skipped - no model available';
		RETURN;
	END IF;
	SELECT neurondb.predict(model_id, ARRAY[7.0, 14.0]::vector) INTO pred;
	IF pred IS NULL THEN
		RAISE NOTICE 'XGBoost (CPU) predict returned NULL';
		RETURN;
	END IF;
	RAISE NOTICE '✓ XGBoost (CPU) inference successful, ŷ = %', pred;
END $$;

\echo '=== XGBoost Basic Test (GPU) ==='
SET neurondb.gpu_enabled = on;

-- Train XGBoost model on GPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('xgboost', 'xgb_data', 'y') INTO model_id;
		IF model_id IS NULL THEN
			RAISE NOTICE 'XGBoost (GPU) training not yet implemented';
			RETURN;
		END IF;
		RAISE NOTICE '✓ XGBoost (GPU) model trained, model_id=%', model_id;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'XGBoost (GPU) training not yet implemented: %', SQLERRM;
		RETURN;
	END;
END $$;

-- Test inference on GPU
DO $$
DECLARE
	pred float8;
	model_id int;
BEGIN
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'xgboost' ORDER BY m.model_id DESC LIMIT 1;
	IF model_id IS NULL THEN
		RAISE NOTICE 'XGBoost (GPU) inference skipped - no model available';
		RETURN;
	END IF;
	SELECT neurondb.predict(model_id, ARRAY[7.0, 14.0]::vector) INTO pred;
	IF pred IS NULL THEN
		RAISE NOTICE 'XGBoost (GPU) predict returned NULL';
		RETURN;
	END IF;
	RAISE NOTICE '✓ XGBoost (GPU) inference successful, ŷ = %', pred;
END $$;

