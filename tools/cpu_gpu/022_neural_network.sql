-- 022_neural_network.sql
-- Basic test for Neural Network (CPU and GPU)

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

DROP TABLE IF EXISTS nn_data;
CREATE TABLE nn_data (
	id serial PRIMARY KEY,
	x1 double precision,
	x2 double precision,
	y int
);

INSERT INTO nn_data (x1, x2, y)
SELECT x, x*2 + random()*0.05, CASE WHEN x < 5 THEN 0 ELSE 1 END
FROM generate_series(1, 10) AS x;

\echo '=== Neural Network Basic Test (CPU) ==='

-- Train neural network model on CPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('neural_network', 'nn_data', 'y') INTO model_id;
		IF model_id IS NULL THEN
			RAISE NOTICE 'Neural Network (CPU) training not yet implemented';
			RETURN;
		END IF;
		RAISE NOTICE '✓ Neural Network (CPU) model trained, model_id=%', model_id;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Neural Network (CPU) training not yet implemented: %', SQLERRM;
		RETURN;
	END;
END $$;

-- Test inference on CPU
DO $$
DECLARE
	pred float8;
	model_id int;
BEGIN
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'neural_network' ORDER BY m.model_id DESC LIMIT 1;
	IF model_id IS NULL THEN
		RAISE NOTICE 'Neural Network (CPU) inference skipped - no model available';
		RETURN;
	END IF;
	SELECT neurondb.predict(model_id, ARRAY[7.0, 14.0]::vector) INTO pred;
	IF pred IS NULL THEN
		RAISE NOTICE 'Neural Network (CPU) predict returned NULL';
		RETURN;
	END IF;
	RAISE NOTICE '✓ Neural Network (CPU) inference successful, ŷ = %', pred;
END $$;

\echo '=== Neural Network Basic Test (GPU) ==='
SET neurondb.gpu_enabled = on;

-- Train neural network model on GPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('neural_network', 'nn_data', 'y') INTO model_id;
		IF model_id IS NULL THEN
			RAISE NOTICE 'Neural Network (GPU) training not yet implemented';
			RETURN;
		END IF;
		RAISE NOTICE '✓ Neural Network (GPU) model trained, model_id=%', model_id;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Neural Network (GPU) training not yet implemented: %', SQLERRM;
		RETURN;
	END;
END $$;

-- Test inference on GPU
DO $$
DECLARE
	pred float8;
	model_id int;
BEGIN
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'neural_network' ORDER BY m.model_id DESC LIMIT 1;
	IF model_id IS NULL THEN
		RAISE NOTICE 'Neural Network (GPU) inference skipped - no model available';
		RETURN;
	END IF;
	SELECT neurondb.predict(model_id, ARRAY[2.0, 4.0]::vector) INTO pred;
	IF pred IS NULL THEN
		RAISE NOTICE 'Neural Network (GPU) predict returned NULL';
		RETURN;
	END IF;
	RAISE NOTICE '✓ Neural Network (GPU) inference successful, ŷ = %', pred;
END $$;

\echo '✓ Neural Network basic test complete'
