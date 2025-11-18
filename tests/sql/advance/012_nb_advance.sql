\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Naive Bayes - Advanced Features Test'
\echo '=========================================================================='

-- This test uses test_train_view and test_test_view tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'test_train_view') THEN
		RAISE EXCEPTION 'test_train_view table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'test_test_view') THEN
		RAISE EXCEPTION 'test_test_view table does not exist. Please run: python ml_dataset.py <dataset_name>';
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

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,nb_train,nb_predict';
SELECT neurondb_gpu_enable();

\echo ''
\echo 'Test 1: Training with Custom Hyperparameters'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Train with custom parameters
DROP TABLE IF EXISTS adv_model_temp;
CREATE TEMP TABLE adv_model_temp AS
SELECT neurondb.train(
	'naive_bayes',
	'test_train_view',
	'features',
	'label',
	'{"var_smoothing": 1e-9}'::jsonb
)::integer AS model_id;

SELECT model_id FROM adv_model_temp;

\echo ''
\echo 'Test 2: Model Metadata and Storage Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	m.model_id,
	m.algorithm,
	m.created_at,
	m.metrics->>'storage' AS storage_type,
	m.metrics->>'accuracy' AS accuracy,
	m.metrics->>'n_classes' AS n_classes,
	CASE 
		WHEN m.model_data IS NULL THEN 'NULL (GPU model)'
		ELSE format('%s bytes', pg_column_size(m.model_data))
	END AS model_data_status
FROM neurondb.ml_models m, adv_model_temp t
WHERE m.model_id = t.model_id;

\echo ''
\echo 'Test 3: Batch Evaluation Performance (Optimized C Batch Processing)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Use optimized batch evaluation instead of per-row predictions
CREATE TEMP TABLE adv_eval_temp AS
SELECT neurondb.evaluate((SELECT model_id FROM adv_model_temp), 'test_test_view', 'features', 'label') AS metrics;

SELECT
	'Total Samples' AS metric,
	(SELECT COUNT(*)::bigint FROM test_test_view WHERE features IS NOT NULL AND label IS NOT NULL)::text AS value
UNION ALL
SELECT 'Accuracy', ROUND((metrics->>'accuracy')::numeric, 6)::text
FROM adv_eval_temp
WHERE metrics->>'accuracy' IS NOT NULL
UNION ALL
SELECT 'Precision', ROUND((metrics->>'precision')::numeric, 6)::text
FROM adv_eval_temp
WHERE metrics->>'precision' IS NOT NULL
UNION ALL
SELECT 'Recall', ROUND((metrics->>'recall')::numeric, 6)::text
FROM adv_eval_temp
WHERE metrics->>'recall' IS NOT NULL
UNION ALL
SELECT 'F1 Score', ROUND((metrics->>'f1_score')::numeric, 6)::text
FROM adv_eval_temp
WHERE metrics->>'f1_score' IS NOT NULL
ORDER BY metric;

\echo ''
\echo 'Test 4: Evaluation Metrics Summary'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Show detailed evaluation metrics from optimized batch processing
SELECT
	'Accuracy' AS metric,
	ROUND((metrics->>'accuracy')::numeric, 6)::numeric AS value
FROM adv_eval_temp
WHERE metrics->>'accuracy' IS NOT NULL
UNION ALL
SELECT 'Precision', ROUND((metrics->>'precision')::numeric, 6)::numeric
FROM adv_eval_temp
WHERE metrics->>'precision' IS NOT NULL
UNION ALL
SELECT 'Recall', ROUND((metrics->>'recall')::numeric, 6)::numeric
FROM adv_eval_temp
WHERE metrics->>'recall' IS NOT NULL
UNION ALL
SELECT 'F1 Score', ROUND((metrics->>'f1_score')::numeric, 6)::numeric
FROM adv_eval_temp
WHERE metrics->>'f1_score' IS NOT NULL
ORDER BY metric;

\echo ''
\echo 'Test 5: Model Quality Metrics'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Use evaluation metrics for quality assessment (much faster than per-row predictions)
SELECT
	ROUND((metrics->>'accuracy')::numeric, 6) AS accuracy,
	ROUND((metrics->>'precision')::numeric, 6) AS precision,
	ROUND((metrics->>'recall')::numeric, 6) AS recall,
	ROUND((metrics->>'f1_score')::numeric, 6) AS f1_score,
	CASE 
		WHEN (metrics->>'accuracy')::numeric > 0.8 THEN 'Excellent'
		WHEN (metrics->>'accuracy')::numeric > 0.6 THEN 'Good'
		WHEN (metrics->>'accuracy')::numeric > 0.4 THEN 'Moderate'
		ELSE 'Poor (may need feature engineering)'
	END AS fit_quality
FROM adv_eval_temp;

\echo ''
\echo 'Test 6: Model Comparison (GPU vs CPU)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Train GPU model
DROP TABLE IF EXISTS gpu_model_adv;
CREATE TEMP TABLE gpu_model_adv AS
SELECT neurondb.train(
	'naive_bayes',
	'test_train_view',
	'features',
	'label',
	'{"storage": "gpu"}'::jsonb
)::integer AS model_id;

-- Train CPU model (if possible)
DROP TABLE IF EXISTS cpu_model_adv;
CREATE TEMP TABLE cpu_model_adv AS
SELECT neurondb.train(
	'naive_bayes',
	'test_train_view',
	'features',
	'label',
	'{"storage": "cpu"}'::jsonb
)::integer AS model_id;

-- Compare metrics
SELECT 
	'GPU Model' AS model_type,
	ROUND((m.metrics->>'accuracy')::numeric, 4) AS accuracy,
	ROUND((m.metrics->>'precision')::numeric, 4) AS precision
FROM neurondb.ml_models m, gpu_model_adv g
WHERE m.model_id = g.model_id
UNION ALL
SELECT 
	'CPU Model' AS model_type,
	ROUND((m.metrics->>'accuracy')::numeric, 4) AS accuracy,
	ROUND((m.metrics->>'precision')::numeric, 4) AS precision
FROM neurondb.ml_models m, cpu_model_adv c
WHERE m.model_id = c.model_id;

\echo ''
\echo 'Test 7: Model Persistence and Retrieval'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Check model is persisted
SELECT 
	COUNT(*) AS total_models,
	COUNT(DISTINCT algorithm) AS unique_algorithms,
	MIN(created_at) AS oldest_model,
	MAX(created_at) AS newest_model
FROM neurondb.ml_models
WHERE algorithm = 'naive_bayes';

-- Cleanup
DROP TABLE IF EXISTS adv_model_temp;
DROP TABLE IF EXISTS adv_eval_temp;
DROP TABLE IF EXISTS gpu_model_adv;
DROP TABLE IF EXISTS cpu_model_adv;

\echo ''
\echo 'Advanced Naive Bayes Test Complete!'
\echo '=========================================================================='

