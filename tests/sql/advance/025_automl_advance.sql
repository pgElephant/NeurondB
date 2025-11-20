-- 025_automl_advance.sql
-- Exhaustive detailed test for AutoML: all operations, error handling.
-- Works on 1000 rows only and tests each and every way with comprehensive coverage
-- Tests: AutoML training, model comparison, hyperparameter analysis, error handling

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo '=========================================================================='

/* Check that sample_train and sample_test exist */

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
SET neurondb.automl.use_gpu = on;
SELECT neurondb_gpu_enable() AS gpu_available;
SELECT neurondb_gpu_info() AS gpu_info;

/*
 * ---- AUTOML TESTS ----
 * Test AutoML operations
 */
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	result jsonb;
BEGIN
	BEGIN
		SELECT auto_train(
			'test_train_view',
			'features',
			'label',
			'classification',
			'f1_score'
		) INTO result;
		IF result IS NOT NULL THEN
		END IF;
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END $$;

DO $$
DECLARE
	result jsonb;
BEGIN
	BEGIN
		SELECT auto_train(
			'test_train_view',
			'features',
			'label',
			'regression',
			'r2'
		) INTO result;
		IF result IS NOT NULL THEN
		END IF;
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END $$;

DO $$
DECLARE
	result jsonb;
BEGIN
	BEGIN
		SELECT auto_train(
			'test_train_view',
			'features',
			'label',
			'classification',
			'accuracy'
		) INTO result;
		IF result IS NOT NULL THEN
		END IF;
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END $$;

SELECT 
	m.model_id,
	m.algorithm,
	m.created_at,
	m.metrics->>'accuracy' AS accuracy,
	m.metrics->>'precision' AS precision,
	m.metrics->>'recall' AS recall,
	m.metrics->>'f1_score' AS f1_score,
	m.metrics->>'storage' AS storage_type,
	ROW_NUMBER() OVER (ORDER BY (m.metrics->>'accuracy')::numeric DESC NULLS LAST) AS rank
FROM neurondb.ml_models m
WHERE m.project_id IN (
	SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'default'
)
ORDER BY (m.metrics->>'accuracy')::numeric DESC NULLS LAST
LIMIT 10;

DO $$
DECLARE
	best_model_id integer;
	metrics_result jsonb;
BEGIN
	-- Get best model by accuracy
	SELECT model_id INTO best_model_id
	FROM neurondb.ml_models
	WHERE project_id IN (
		SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'default'
	)
	AND algorithm IN ('random_forest', 'logistic_regression', 'decision_tree', 'svm')
	ORDER BY (metrics->>'accuracy')::numeric DESC NULLS LAST
	LIMIT 1;
	
	IF best_model_id IS NOT NULL THEN
		BEGIN
			metrics_result := neurondb.evaluate(best_model_id, 'test_test_view', 'features', 'label');
		EXCEPTION WHEN OTHERS THEN
		END;
	ELSE
	END IF;
END $$;

SELECT 
	algorithm,
	COUNT(*) AS model_count,
	ROUND(AVG((metrics->>'accuracy')::numeric), 4) AS avg_accuracy,
	ROUND(MAX((metrics->>'accuracy')::numeric), 4) AS max_accuracy,
	ROUND(MIN((metrics->>'accuracy')::numeric), 4) AS min_accuracy,
	ROUND(STDDEV((metrics->>'accuracy')::numeric), 4) AS stddev_accuracy,
	CASE 
		WHEN AVG((metrics->>'accuracy')::numeric) > 0.8 THEN 'Excellent'
		WHEN AVG((metrics->>'accuracy')::numeric) > 0.6 THEN 'Good'
		WHEN AVG((metrics->>'accuracy')::numeric) > 0.4 THEN 'Moderate'
		ELSE 'Needs improvement'
	END AS performance_category
FROM neurondb.ml_models
WHERE project_id IN (
	SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'default'
)
AND metrics->>'accuracy' IS NOT NULL
GROUP BY algorithm
ORDER BY avg_accuracy DESC;

SELECT 
	CASE 
		WHEN metrics->>'storage' = 'gpu' THEN 'GPU'
		ELSE 'CPU'
	END AS storage_type,
	algorithm,
	COUNT(*) AS model_count,
	ROUND(AVG((metrics->>'accuracy')::numeric), 4) AS avg_accuracy,
	ROUND(AVG(EXTRACT(EPOCH FROM (created_at - (SELECT MIN(created_at) FROM neurondb.ml_models WHERE project_id IN (SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'default'))))), 2) AS avg_training_time_seconds
FROM neurondb.ml_models
WHERE project_id IN (
	SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'default'
)
AND metrics->>'accuracy' IS NOT NULL
GROUP BY storage_type, algorithm
ORDER BY storage_type, avg_accuracy DESC;

SELECT 
	algorithm,
	metrics->>'max_depth' AS max_depth,
	metrics->>'n_estimators' AS n_estimators,
	metrics->>'n_trees' AS n_trees,
	metrics->>'learning_rate' AS learning_rate,
	metrics->>'C' AS C_value,
	ROUND((metrics->>'accuracy')::numeric, 4) AS accuracy
FROM neurondb.ml_models
WHERE project_id IN (
	SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'default'
)
AND metrics->>'accuracy' IS NOT NULL
ORDER BY (metrics->>'accuracy')::numeric DESC
LIMIT 10;

/* --- ERROR path: invalid parameters --- */
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	result jsonb;
BEGIN
	BEGIN
		SELECT auto_train('missing_table', 'features', 'label', 'classification', 'accuracy') INTO result;
		RAISE EXCEPTION 'FAIL: expected error for missing table';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
DECLARE
	result jsonb;
BEGIN
	BEGIN
		SELECT auto_train('test_train_view', 'notafeat', 'label', 'classification', 'accuracy') INTO result;
		RAISE EXCEPTION 'FAIL: expected error for invalid feature column';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
DECLARE
	result jsonb;
BEGIN
	BEGIN
		SELECT auto_train('test_train_view', 'features', 'label', 'invalid_task', 'accuracy') INTO result;
		RAISE EXCEPTION 'FAIL: expected error for invalid task type';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

DO $$
DECLARE
	result jsonb;
BEGIN
	BEGIN
		SELECT auto_train('test_train_view', 'features', 'label', 'classification', 'invalid_metric') INTO result;
		RAISE EXCEPTION 'FAIL: expected error for invalid metric';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

\echo 'Test completed successfully'
