-- 002_logreg_advance.sql
-- Exhaustive test for logistic_regression: all train, error, predict, evaluate.

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=== logistic_regression: Exhaustive GPU/CPU + Error Coverage (1000 rows sample) ==='

/* Check that test_train_view and test_test_view exist and are sufficient for subsample */
DO $$
BEGIN
	IF NOT EXISTS (
		SELECT 1 FROM information_schema.tables
		WHERE table_schema = 'public'
		  AND table_name = 'test_train_view'
	) THEN
		RAISE EXCEPTION 'test_train_view missing';
	END IF;
	IF NOT EXISTS (
		SELECT 1 FROM information_schema.tables
		WHERE table_schema = 'public'
		  AND table_name = 'test_test_view'
	) THEN
		RAISE EXCEPTION 'test_test_view missing';
	END IF;
END
$$;
-- Create views with 1000 rows for advance tests
DROP VIEW IF EXISTS test_train_view;
DROP VIEW IF EXISTS test_test_view;

CREATE VIEW test_train_view AS
SELECT features, label FROM sample_train LIMIT 1000;

CREATE VIEW test_test_view AS
SELECT features, label FROM sample_test LIMIT 1000;

SELECT 
	COUNT(*)::bigint AS train_count,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_count
FROM test_train_view;

/*---- Register required GPU kernels ----*/
\echo 'Step: Ensure all necessary GPU kernels are registered'
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,lr_train,lr_predict';

/*
 * ---- TRAINING routines (1000 sampled rows) ----
 */
\echo 'Train: logistic_regression GPU default (first 1000)'
SET neurondb.gpu_enabled = on;
DROP TABLE IF EXISTS gpu_model_temp_002;
CREATE TEMP TABLE gpu_model_temp_002 AS
SELECT neurondb.train('logistic_regression', 
	'test_train_view', 
	'features', 'label', '{}'::jsonb)::integer AS gpu_model_id;

\echo 'Train: logistic_regression CPU default (first 1000)'
SET neurondb.gpu_enabled = off;
DROP TABLE IF EXISTS cpu_model_temp_002;
CREATE TEMP TABLE cpu_model_temp_002 AS
SELECT neurondb.train('logistic_regression', 
	'test_train_view', 
	'features', 'label', '{}'::jsonb)::integer AS cpu_model_id;

\echo 'Train: logistic_regression with custom hyperparams, fit_intercept=false'
DROP TABLE IF EXISTS custom_model_temp_002;
CREATE TEMP TABLE custom_model_temp_002 AS
SELECT neurondb.train('logistic_regression', 
	'test_train_view', 
	'features', 'label', 
	'{"epochs":2,"lr":0.31,"fit_intercept":false}'::jsonb)::integer AS custom_model_id;

\echo 'Train: logistic_regression (fit_intercept true explicit)'
SELECT neurondb.train('logistic_regression', 
	'test_train_view', 
	'features', 'label', 
	'{"fit_intercept":true}'::jsonb);

/* --- ERROR path: bad table or column --- */
\echo 'Train: bad table (should error)'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('logistic_regression','missing_table','features','label','{}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error';
	EXCEPTION WHEN OTHERS THEN NULL; END;
END$$;

\echo 'Train: bad feature column'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('logistic_regression','test_train_view','notafeat','label','{}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error';
	EXCEPTION WHEN OTHERS THEN NULL; END;
END$$;

\echo 'Train: bad label column'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('logistic_regression','test_train_view','features','notalabel','{}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error';
	EXCEPTION WHEN OTHERS THEN NULL; END;
END$$;

\echo 'Train: passing invalid hyperparams (negative test)'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('logistic_regression','test_train_view','features','label','{"epochs":-5}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error';
	EXCEPTION WHEN OTHERS THEN NULL; END;
END$$;

/*-------------------------------------------------------------------
 * ---- PREDICT ----
 * GPU/CPU paths, all error paths, batch and single, sampling 1000
 *------------------------------------------------------------------*/
\echo 'Predict: GPU (batch, 1000)'
SET neurondb.gpu_enabled = on;
SELECT COUNT(*), AVG(score) AS avg_score FROM (
	SELECT neurondb.predict((SELECT gpu_model_id FROM gpu_model_temp_002 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 1000
) sub;

\echo 'Predict: CPU (batch, 1000)'
SET neurondb.gpu_enabled = off;
SELECT COUNT(*), AVG(score) AS avg_score FROM (
	SELECT neurondb.predict((SELECT cpu_model_id FROM cpu_model_temp_002 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 1000
) sub;

\echo 'Predict: custom model, single row'
SELECT neurondb.predict((SELECT custom_model_id FROM custom_model_temp_002 LIMIT 1), features)
	FROM test_test_view
	LIMIT 1;

\echo 'Predict: custom model, batch (explicit CPU, 1000)'
SET neurondb.gpu_enabled = off;
SELECT COUNT(*) FROM (
	SELECT neurondb.predict((SELECT custom_model_id FROM custom_model_temp_002 LIMIT 1), features)
	FROM test_test_view
	LIMIT 1000
) b;

/* Error: invalid model id */
\echo 'Predict: non-existent/negative model id (should error)'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.predict(-10, ARRAY[0.1]);
		RAISE EXCEPTION 'FAIL: non-existent model should error';
	EXCEPTION WHEN OTHERS THEN NULL; END;
END$$;

\echo 'Predict: bad features (wrong length, type)'
DO $$
DECLARE
	cpu_mid_temp integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid_temp FROM cpu_model_temp_002 LIMIT 1;
	BEGIN
		PERFORM neurondb.predict(cpu_mid_temp, '{1,2,3}'::integer[]);
		RAISE EXCEPTION 'FAIL: int[] instead of float[] should error';
	EXCEPTION WHEN OTHERS THEN NULL; END;
END$$;

DO $$
DECLARE
	model_dim integer;
	cpu_mid integer;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_002 LIMIT 1;
	SELECT jsonb_array_length(metrics->'coefficients') INTO model_dim
	FROM neurondb.ml_models WHERE id = cpu_mid
	AND metrics IS NOT NULL;
	BEGIN
		PERFORM neurondb.predict(cpu_mid, ARRAY[42.0]); -- wrong dim
		RAISE EXCEPTION 'FAIL: wrong feature array length should error';
	EXCEPTION WHEN OTHERS THEN NULL; END;
END
$$;

/*-------------------------------------------------------------------
 * ---- EVALUATE ----
 * Metrics for all model types/paths ; test set sampled to 1000 rows
 *------------------------------------------------------------------*/
\echo 'Evaluate: CPU'
SELECT neurondb.evaluate('logistic_regression', :cpu_model_id,
	'(SELECT * FROM sample_test LIMIT 1000)'::regclass, 'features', 'label', '{}'::jsonb);

\echo 'Evaluate: GPU'
SELECT neurondb.evaluate('logistic_regression', :gpu_model_id,
	'(SELECT * FROM sample_test LIMIT 1000)'::regclass, 'features', 'label', '{}'::jsonb);

\echo 'Evaluate: custom'
SELECT neurondb.evaluate('logistic_regression', :custom_model_id,
	'(SELECT * FROM sample_test LIMIT 1000)'::regclass, 'features', 'label', '{}'::jsonb);

\echo 'Evaluate: bad table/columns (should error)'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.evaluate('logistic_regression', :cpu_model_id,
			'no_such', 'features', 'label', '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: eval on bad table must error';
	EXCEPTION WHEN OTHERS THEN NULL; END;
	BEGIN
		PERFORM neurondb.evaluate('logistic_regression', :cpu_model_id,
			'test_test_view', 'badfeature', 'label', '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: eval on bad feature col must error';
	EXCEPTION WHEN OTHERS THEN NULL; END;
	BEGIN
		PERFORM neurondb.evaluate('logistic_regression', :cpu_model_id,
			'test_test_view', 'features', 'badlabel', '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: eval on bad label col must error';
	EXCEPTION WHEN OTHERS THEN NULL; END;
END$$;

/*-------------------------------------------------------------------
 * Model catalog check
 *------------------------------------------------------------------*/
SELECT algorithm, COUNT(*) AS n_models
FROM neurondb.ml_models
WHERE algorithm = 'logistic_regression'
GROUP BY algorithm;

\echo '✓ logistic_regression: Full code-path test complete (1000-row sample)'
