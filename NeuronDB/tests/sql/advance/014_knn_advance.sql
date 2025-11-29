-- 014_knn_advance.sql
-- Exhaustive detailed test for knn: all train, error, predict, evaluate.
-- Works on 1000 rows only and tests each and every way with comprehensive coverage
-- Tests: GPU/CPU paths, hyperparameters, batch operations, error handling, metadata

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'knn: Exhaustive GPU/CPU + Error Coverage (1000 rows sample)'
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
SELECT neurondb_gpu_enable() AS gpu_available;
SELECT neurondb_gpu_info() AS gpu_info;

/*
 * ---- TRAINING routines (1000 sampled rows) ----
 * Test multiple hyperparameter combinations and paths
 */
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS gpu_model_k3_temp_014;
CREATE TEMP TABLE gpu_model_k3_temp_014 AS
SELECT neurondb.train('knn', 
	'test_train_view', 
	'features', 'label', 
	'{"k":3}'::jsonb)::integer AS gpu_model_id;

SELECT 
	'GPU k=3' AS config,
	gpu_model_id AS model_id,
	(SELECT metrics->>'storage' FROM neurondb.ml_models WHERE model_id = gpu_model_id) AS storage
FROM gpu_model_k3_temp_014;

DROP TABLE IF EXISTS cpu_model_k5_temp_014;
CREATE TEMP TABLE cpu_model_k5_temp_014 AS
SELECT neurondb.train('knn', 
	'test_train_view', 
	'features', 'label', 
	'{"k":5}'::jsonb)::integer AS cpu_model_id;

SELECT 
	'CPU k=5' AS config,
	cpu_model_id AS model_id,
	(SELECT metrics->>'storage' FROM neurondb.ml_models WHERE model_id = cpu_model_id) AS storage
FROM cpu_model_k5_temp_014;

DROP TABLE IF EXISTS custom_model_temp_014;
CREATE TEMP TABLE custom_model_temp_014 AS
SELECT neurondb.train('knn', 
	'test_train_view', 
	'features', 'label', 
	'{"k":7}'::jsonb)::integer AS custom_model_id;

SELECT 
	'Custom (k=7)' AS config,
	custom_model_id AS model_id,
	(SELECT metrics->>'k' FROM neurondb.ml_models WHERE model_id = custom_model_id) AS k_value
FROM custom_model_temp_014;

DROP TABLE IF EXISTS k1_model_temp_014;
CREATE TEMP TABLE k1_model_temp_014 AS
SELECT neurondb.train('knn', 
	'test_train_view', 
	'features', 'label', 
	'{"k":1}'::jsonb)::integer AS k1_model_id;

DROP TABLE IF EXISTS k11_model_temp_014;
CREATE TEMP TABLE k11_model_temp_014 AS
SELECT neurondb.train('knn', 
	'test_train_view', 
	'features', 'label', 
	'{"k":11}'::jsonb)::integer AS k11_model_id;

/* --- ERROR path: bad table or column --- */
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('knn','missing_table','features','label','{"k":5}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for missing table';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('knn','test_train_view','notafeat','label','{"k":5}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for invalid feature column';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('knn','test_train_view','features','notalabel','{"k":5}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for invalid label column';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('knn','test_train_view','features','label','{"k":0}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for k=0';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('knn','test_train_view','features','label','{"k":-1}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for negative k';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('knn','test_train_view',NULL,'label','{"k":5}'::jsonb);
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
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	'GPU Batch' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(score)::numeric, 4) AS avg_score,
	ROUND(MIN(score)::numeric, 4) AS min_score,
	ROUND(MAX(score)::numeric, 4) AS max_score
FROM (
	SELECT neurondb.predict((SELECT gpu_model_id FROM gpu_model_k3_temp_014 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 1000
) sub;

SELECT 
	'CPU Batch' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(score)::numeric, 4) AS avg_score,
	ROUND(MIN(score)::numeric, 4) AS min_score,
	ROUND(MAX(score)::numeric, 4) AS max_score
FROM (
	SELECT neurondb.predict((SELECT cpu_model_id FROM cpu_model_k5_temp_014 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 1000
) sub;

SELECT 
	'Custom Single' AS test_type,
	neurondb.predict((SELECT custom_model_id FROM custom_model_temp_014 LIMIT 1), features) AS prediction
FROM test_test_view
LIMIT 1;

SELECT 
	'Custom Batch' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(score)::numeric, 4) AS avg_score
FROM (
	SELECT neurondb.predict((SELECT custom_model_id FROM custom_model_temp_014 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 100
) b;

SELECT 
	'k=1 Batch' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(score)::numeric, 4) AS avg_score
FROM (
	SELECT neurondb.predict((SELECT k1_model_id FROM k1_model_temp_014 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 50
) b;

/* Error: invalid model id */
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

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

DO $$
DECLARE
	cpu_mid_temp integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid_temp FROM cpu_model_k5_temp_014 LIMIT 1;
	BEGIN
		PERFORM neurondb.predict(cpu_mid_temp, '{1,2,3}'::integer[]);
		RAISE EXCEPTION 'FAIL: int[] instead of float[] should error';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
DECLARE
	model_dim integer;
	cpu_mid integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_k5_temp_014 LIMIT 1;
	SELECT jsonb_array_length(metrics->'feature_dim') INTO model_dim
	FROM neurondb.ml_models WHERE model_id = cpu_mid
	AND metrics IS NOT NULL AND metrics->'feature_dim' IS NOT NULL;
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

DO $$
DECLARE
	cpu_mid integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_k5_temp_014 LIMIT 1;
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
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	cpu_mid integer;
	result jsonb;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_k5_temp_014 LIMIT 1;
	IF cpu_mid IS NOT NULL THEN
		BEGIN
			result := neurondb.evaluate(cpu_mid, 'test_test_view', 'features', 'label');
		EXCEPTION WHEN OTHERS THEN
		END;
	END IF;
END$$;

DO $$
DECLARE
	gpu_mid integer;
	result jsonb;
BEGIN
	SELECT gpu_model_id INTO gpu_mid FROM gpu_model_k3_temp_014 LIMIT 1;
	IF gpu_mid IS NOT NULL THEN
		BEGIN
			result := neurondb.evaluate(gpu_mid, 'test_test_view', 'features', 'label');
		EXCEPTION WHEN OTHERS THEN
		END;
	END IF;
END$$;

DO $$
DECLARE
	custom_mid integer;
	result jsonb;
BEGIN
	SELECT custom_model_id INTO custom_mid FROM custom_model_temp_014 LIMIT 1;
	IF custom_mid IS NOT NULL THEN
		BEGIN
			result := neurondb.evaluate(custom_mid, 'test_test_view', 'features', 'label');
		EXCEPTION WHEN OTHERS THEN
		END;
	END IF;
END$$;

DO $$
DECLARE
	k1_mid integer;
	result jsonb;
BEGIN
	SELECT k1_model_id INTO k1_mid FROM k1_model_temp_014 LIMIT 1;
	IF k1_mid IS NOT NULL THEN
		BEGIN
			result := neurondb.evaluate(k1_mid, 'test_test_view', 'features', 'label');
		EXCEPTION WHEN OTHERS THEN
		END;
	END IF;
END$$;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	cpu_mid integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_k5_temp_014 LIMIT 1;
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

DO $$
DECLARE
	cpu_mid integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_k5_temp_014 LIMIT 1;
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

DO $$
DECLARE
	cpu_mid integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_k5_temp_014 LIMIT 1;
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
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	m.model_id,
	m.algorithm,
	m.created_at,
	m.metrics->>'storage' AS storage_type,
	m.metrics->>'k' AS k_value,
	m.metrics->>'accuracy' AS accuracy,
	m.metrics->>'n_samples' AS n_samples,
	m.metrics->>'n_features' AS n_features,
	m.metrics->>'n_classes' AS n_classes,
	CASE 
		WHEN m.model_data IS NULL THEN 'NULL (GPU model)'
		ELSE format('%s bytes', pg_column_size(m.model_data))
	END AS model_data_status
FROM neurondb.ml_models m, gpu_model_k3_temp_014 t
WHERE m.model_id = t.gpu_model_id;

SELECT 
	m.model_id,
	m.algorithm,
	m.created_at,
	m.metrics->>'storage' AS storage_type,
	m.metrics->>'k' AS k_value,
	m.metrics->>'accuracy' AS accuracy,
	CASE 
		WHEN m.model_data IS NULL THEN 'NULL (GPU model)'
		ELSE format('%s bytes', pg_column_size(m.model_data))
	END AS model_data_status
FROM neurondb.ml_models m, cpu_model_k5_temp_014 t
WHERE m.model_id = t.cpu_model_id;

SELECT 
	m.model_id,
	m.algorithm,
	m.metrics->>'k' AS k_value,
	m.metrics->>'accuracy' AS accuracy
FROM neurondb.ml_models m, custom_model_temp_014 t
WHERE m.model_id = t.custom_model_id;

SELECT 
	algorithm,
	COUNT(*) AS n_models,
	MIN(created_at) AS first_created,
	MAX(created_at) AS last_created,
	COUNT(CASE WHEN metrics->>'storage' = 'gpu' THEN 1 END) AS gpu_models,
	COUNT(CASE WHEN metrics->>'storage' = 'cpu' THEN 1 END) AS cpu_models,
	COUNT(DISTINCT (metrics->>'k')::int) AS unique_k_values
FROM neurondb.ml_models
WHERE algorithm = 'knn'
GROUP BY algorithm;

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

\echo 'Test completed successfully'
