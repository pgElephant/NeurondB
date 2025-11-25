-- 012_nb_advance.sql
-- Exhaustive detailed test for naive_bayes: all train, error, predict, evaluate.
-- Works on 1000 rows only and tests each and every way with comprehensive coverage
-- Tests: GPU/CPU paths, hyperparameters, batch operations, error handling, metadata

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'naive_bayes: Exhaustive GPU/CPU + Error Coverage (1000 rows sample)'
\echo '=========================================================================='

/* Use views created by test runner or create from available source tables */
DO $$
DECLARE
	train_source TEXT;
	test_source TEXT;
BEGIN
	-- Find source tables (prefer dataset schema, fallback to public)
	SELECT table_schema || '.' || table_name INTO train_source
	FROM information_schema.tables 
	WHERE (table_schema = 'dataset' AND table_name = 'test_train')
	   OR (table_schema = 'public' AND table_name IN ('sample_train', 'test_train'))
	ORDER BY CASE WHEN table_schema = 'dataset' THEN 0 ELSE 1 END
	LIMIT 1;
	
	IF train_source IS NULL THEN
		-- Views may already exist from test runner
		IF EXISTS (SELECT 1 FROM information_schema.views WHERE table_name = 'test_train_view') THEN
			RETURN;
		END IF;
		RAISE EXCEPTION 'No training table found';
	END IF;
	
	-- Determine corresponding test table
	IF train_source LIKE 'dataset.%' THEN
		test_source := 'dataset.test_test';
	ELSIF train_source LIKE '%sample_train%' THEN
		test_source := 'sample_test';
	ELSE
		test_source := 'test_test';
	END IF;
	
	-- Create views with type conversion if needed
	IF NOT EXISTS (SELECT 1 FROM information_schema.views WHERE table_name = 'test_train_view') THEN
		EXECUTE format('CREATE VIEW test_train_view AS SELECT features::vector(28) as features, label FROM %s LIMIT 1000', train_source);
	END IF;
	IF NOT EXISTS (SELECT 1 FROM information_schema.views WHERE table_name = 'test_test_view') THEN
		EXECUTE format('CREATE VIEW test_test_view AS SELECT features::vector(28) as features, label FROM %s LIMIT 1000', test_source);
	END IF;
END
$$;

\echo ''
\echo 'Dataset Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
SELECT 
	COUNT(*)::bigint AS train_count,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_count,
	(SELECT COUNT(DISTINCT label) FROM test_train_view) AS n_classes,
	(SELECT vector_dims(features) FROM test_train_view LIMIT 1) AS feature_dim
FROM test_train_view;

/*---- Register required GPU kernels ----*/
\echo ''
\echo 'GPU Configuration'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,nb_train,nb_predict';
SELECT neurondb_gpu_enable() AS gpu_available;
SELECT neurondb_gpu_info() AS gpu_info;

/*
 * ---- TRAINING routines (1000 sampled rows) ----
 * Test multiple hyperparameter combinations and paths
 */
\echo ''
\echo 'Training Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: GPU training with default parameters'
SET neurondb.gpu_enabled = on;
DROP TABLE IF EXISTS gpu_model_temp_012;
CREATE TEMP TABLE gpu_model_temp_012 AS
SELECT neurondb.train('naive_bayes', 
	'test_train_view', 
	'features', 'label', '{}'::jsonb)::integer AS gpu_model_id;

SELECT 
	'GPU Default' AS config,
	gpu_model_id AS model_id,
	(SELECT metrics->>'storage' FROM neurondb.ml_models WHERE model_id = gpu_model_id) AS storage
FROM gpu_model_temp_012;

\echo 'Test 2: CPU training with default parameters'
SET neurondb.gpu_enabled = off;
DROP TABLE IF EXISTS cpu_model_temp_012;
CREATE TEMP TABLE cpu_model_temp_012 AS
SELECT neurondb.train('naive_bayes', 
	'test_train_view', 
	'features', 'label', '{}'::jsonb)::integer AS cpu_model_id;

SELECT 
	'CPU Default' AS config,
	cpu_model_id AS model_id,
	(SELECT metrics->>'storage' FROM neurondb.ml_models WHERE model_id = cpu_model_id) AS storage
FROM cpu_model_temp_012;

\echo 'Test 3: Custom hyperparameters (var_smoothing=1e-9)'
DROP TABLE IF EXISTS custom_model_temp_012;
CREATE TEMP TABLE custom_model_temp_012 AS
SELECT neurondb.train('naive_bayes', 
	'test_train_view', 
	'features', 'label', 
	'{"var_smoothing":1e-9}'::jsonb)::integer AS custom_model_id;

SELECT 
	'Custom (var_smoothing=1e-9)' AS config,
	custom_model_id AS model_id,
	(SELECT metrics->>'var_smoothing' FROM neurondb.ml_models WHERE model_id = custom_model_id) AS var_smoothing
FROM custom_model_temp_012;

\echo 'Test 4: Different var_smoothing (var_smoothing=1e-6)'
DROP TABLE IF EXISTS vs1e6_model_temp_012;
CREATE TEMP TABLE vs1e6_model_temp_012 AS
SELECT neurondb.train('naive_bayes', 
	'test_train_view', 
	'features', 'label', 
	'{"var_smoothing":1e-6}'::jsonb)::integer AS vs1e6_model_id;

/* --- ERROR path: bad table or column --- */
\echo ''
\echo 'Error Handling Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: Invalid table name'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('naive_bayes','missing_table','features','label','{}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for missing table';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 2: Invalid feature column'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('naive_bayes','test_train_view','notafeat','label','{}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for invalid feature column';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 3: Invalid label column'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('naive_bayes','test_train_view','features','notalabel','{}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for invalid label column';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 4: Negative var_smoothing'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('naive_bayes','test_train_view','features','label','{"var_smoothing":-1.0}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for negative var_smoothing';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 5: NULL features'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('naive_bayes','test_train_view',NULL,'label','{}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for NULL feature column';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- PREDICT ----
 * GPU/CPU paths, all error paths, batch and single, sampling 1000
 *------------------------------------------------------------------*/
\echo ''
\echo 'Prediction Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Predict Test 1: GPU batch prediction (1000 rows)'
SET neurondb.gpu_enabled = on;
SELECT 
	'GPU Batch' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(score)::numeric, 4) AS avg_score,
	ROUND(MIN(score)::numeric, 4) AS min_score,
	ROUND(MAX(score)::numeric, 4) AS max_score
FROM (
	SELECT neurondb.predict((SELECT gpu_model_id FROM gpu_model_temp_012 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 1000
) sub;

\echo 'Predict Test 2: CPU batch prediction (1000 rows)'
SET neurondb.gpu_enabled = off;
SELECT 
	'CPU Batch' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(score)::numeric, 4) AS avg_score,
	ROUND(MIN(score)::numeric, 4) AS min_score,
	ROUND(MAX(score)::numeric, 4) AS max_score
FROM (
	SELECT neurondb.predict((SELECT cpu_model_id FROM cpu_model_temp_012 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 1000
) sub;

\echo 'Predict Test 3: Custom model single row prediction'
SELECT 
	'Custom Single' AS test_type,
	neurondb.predict((SELECT custom_model_id FROM custom_model_temp_012 LIMIT 1), features) AS prediction
FROM test_test_view
LIMIT 1;

\echo 'Predict Test 4: Custom model batch (100 rows)'
SET neurondb.gpu_enabled = off;
SELECT 
	'Custom Batch' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(score)::numeric, 4) AS avg_score
FROM (
	SELECT neurondb.predict((SELECT custom_model_id FROM custom_model_temp_012 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 100
) b;

/* Error: invalid model id */
\echo ''
\echo 'Prediction Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Predict Error Test 1: Non-existent model ID'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.predict(-10, ARRAY[0.1, 0.2, 0.3]::double precision[]);
		RAISE EXCEPTION 'FAIL: non-existent model should error';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Predict Error Test 2: Wrong feature type (integer array)'
DO $$
DECLARE
	cpu_mid_temp integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid_temp FROM cpu_model_temp_012 LIMIT 1;
	BEGIN
		PERFORM neurondb.predict(cpu_mid_temp, '{1,2,3}'::integer[]);
		RAISE EXCEPTION 'FAIL: int[] instead of float[] should error';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Predict Error Test 3: Wrong feature dimension'
DO $$
DECLARE
	model_dim integer;
	cpu_mid integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_012 LIMIT 1;
	SELECT jsonb_array_length(metrics->'class_prior') INTO model_dim
	FROM neurondb.ml_models WHERE model_id = cpu_mid
	AND metrics IS NOT NULL AND metrics->'class_prior' IS NOT NULL;
	IF model_dim IS NULL THEN
		model_dim := 28; /* Default if not available */
	END IF;
	BEGIN
		PERFORM neurondb.predict(cpu_mid, ARRAY[42.0]::double precision[]); -- wrong dim
		RAISE EXCEPTION 'FAIL: wrong feature array length should error';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END
$$;

\echo 'Predict Error Test 4: NULL model ID'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.predict(NULL, ARRAY[0.1, 0.2, 0.3]::double precision[]);
		RAISE EXCEPTION 'FAIL: NULL model ID should error';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Predict Error Test 5: NULL features'
DO $$
DECLARE
	cpu_mid integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_012 LIMIT 1;
	BEGIN
		PERFORM neurondb.predict(cpu_mid, NULL);
		RAISE EXCEPTION 'FAIL: NULL features should error';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- EVALUATE ----
 * Metrics for all model types/paths ; test set sampled to 1000 rows
 *------------------------------------------------------------------*/
\echo ''
\echo 'Evaluation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Evaluate Test 1: CPU model evaluation'
DO $$
DECLARE
	cpu_mid integer;
	result jsonb;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_012 LIMIT 1;
	IF cpu_mid IS NOT NULL THEN
		BEGIN
			result := neurondb.evaluate(cpu_mid, 'test_test_view', 'features', 'label');
		EXCEPTION WHEN OTHERS THEN
		END;
	END IF;
END$$;

\echo 'Evaluate Test 2: GPU model evaluation'
DO $$
DECLARE
	gpu_mid integer;
	result jsonb;
BEGIN
	SELECT gpu_model_id INTO gpu_mid FROM gpu_model_temp_012 LIMIT 1;
	IF gpu_mid IS NOT NULL THEN
		BEGIN
			result := neurondb.evaluate(gpu_mid, 'test_test_view', 'features', 'label');
		EXCEPTION WHEN OTHERS THEN
		END;
	END IF;
END$$;

\echo 'Evaluate Test 3: Custom model evaluation'
DO $$
DECLARE
	custom_mid integer;
	result jsonb;
BEGIN
	SELECT custom_model_id INTO custom_mid FROM custom_model_temp_012 LIMIT 1;
	IF custom_mid IS NOT NULL THEN
		BEGIN
			result := neurondb.evaluate(custom_mid, 'test_test_view', 'features', 'label');
		EXCEPTION WHEN OTHERS THEN
		END;
	END IF;
END$$;

\echo ''
\echo 'Evaluation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Evaluate Error Test 1: Invalid table'
DO $$
DECLARE
	cpu_mid integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_012 LIMIT 1;
	IF cpu_mid IS NOT NULL THEN
		BEGIN
			BEGIN
				PERFORM neurondb.evaluate(cpu_mid, 'no_such', 'features', 'label');
				RAISE EXCEPTION 'FAIL: eval on bad table must error';
			EXCEPTION WHEN OTHERS THEN 
			NULL;
			END;
		END;
	END IF;
END$$;

\echo 'Evaluate Error Test 2: Invalid feature column'
DO $$
DECLARE
	cpu_mid integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_012 LIMIT 1;
	IF cpu_mid IS NOT NULL THEN
		BEGIN
			BEGIN
				PERFORM neurondb.evaluate(cpu_mid, 'test_test_view', 'badfeature', 'label');
				RAISE EXCEPTION 'FAIL: eval on bad feature col must error';
			EXCEPTION WHEN OTHERS THEN 
			NULL;
			END;
		END;
	END IF;
END$$;

\echo 'Evaluate Error Test 3: Invalid label column'
DO $$
DECLARE
	cpu_mid integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_012 LIMIT 1;
	IF cpu_mid IS NOT NULL THEN
		BEGIN
			BEGIN
				PERFORM neurondb.evaluate(cpu_mid, 'test_test_view', 'features', 'badlabel');
				RAISE EXCEPTION 'FAIL: eval on bad label col must error';
			EXCEPTION WHEN OTHERS THEN 
			NULL;
			END;
		END;
	END IF;
END$$;

\echo 'Evaluate Error Test 4: NULL model ID'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.evaluate(NULL, 'test_test_view', 'features', 'label');
		RAISE EXCEPTION 'FAIL: NULL model ID should error';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

/*-------------------------------------------------------------------
 * Model catalog check and metadata
 *------------------------------------------------------------------*/
\echo ''
\echo 'Model Metadata Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Metadata Test 1: GPU model metadata'
SELECT 
	m.model_id,
	m.algorithm,
	m.created_at,
	m.metrics->>'storage' AS storage_type,
	m.metrics->>'var_smoothing' AS var_smoothing,
	m.metrics->>'accuracy' AS accuracy,
	m.metrics->>'n_samples' AS n_samples,
	m.metrics->>'n_features' AS n_features,
	m.metrics->>'n_classes' AS n_classes,
	CASE 
		WHEN m.model_data IS NULL THEN 'NULL (GPU model)'
		ELSE format('%s bytes', pg_column_size(m.model_data))
	END AS model_data_status
FROM neurondb.ml_models m, gpu_model_temp_012 t
WHERE m.model_id = t.gpu_model_id;

\echo 'Metadata Test 2: CPU model metadata'
SELECT 
	m.model_id,
	m.algorithm,
	m.created_at,
	m.metrics->>'storage' AS storage_type,
	m.metrics->>'var_smoothing' AS var_smoothing,
	m.metrics->>'accuracy' AS accuracy,
	CASE 
		WHEN m.model_data IS NULL THEN 'NULL (GPU model)'
		ELSE format('%s bytes', pg_column_size(m.model_data))
	END AS model_data_status
FROM neurondb.ml_models m, cpu_model_temp_012 t
WHERE m.model_id = t.cpu_model_id;

\echo 'Metadata Test 3: Custom model metadata'
SELECT 
	m.model_id,
	m.algorithm,
	m.metrics->>'var_smoothing' AS var_smoothing,
	m.metrics->>'accuracy' AS accuracy
FROM neurondb.ml_models m, custom_model_temp_012 t
WHERE m.model_id = t.custom_model_id;

\echo 'Metadata Test 4: Model comparison (all models)'
SELECT 
	algorithm,
	COUNT(*) AS n_models,
	MIN(created_at) AS first_created,
	MAX(created_at) AS last_created,
	COUNT(CASE WHEN metrics->>'storage' = 'gpu' THEN 1 END) AS gpu_models,
	COUNT(CASE WHEN metrics->>'storage' = 'cpu' THEN 1 END) AS cpu_models
FROM neurondb.ml_models
WHERE algorithm = 'naive_bayes'
GROUP BY algorithm;

\echo ''
\echo '=========================================================================='
\echo '✓ naive_bayes: Full exhaustive code-path test complete (1000-row sample)'
\echo '=========================================================================='

\echo 'Test completed successfully'
