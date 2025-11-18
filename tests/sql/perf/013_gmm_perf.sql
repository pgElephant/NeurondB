\timing on
\pset footer off
\pset pager off

-- This test uses sample_train and sample_test tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_test') THEN
		RAISE EXCEPTION 'sample_test table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
END
$$;

-- Performance test: Works on the whole 11M row view

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip,gmm_train,gmm_predict';
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

-- Train model once and store in temp table to reuse
DROP TABLE IF EXISTS gpu_model_temp;
CREATE TEMP TABLE gpu_model_temp AS
SELECT neurondb.train(
	'gmm',
	'sample_train',
	'features',
	NULL,
	'{"k": 3, "max_iters": 100}'::jsonb
)::integer AS model_id;

-- Debug: Show model_id
SELECT model_id FROM gpu_model_temp;

-- Predict clusters for test data
SELECT
	neurondb.predict(m.model_id, features) AS cluster_id,
	COUNT(*) AS count
FROM sample_test, gpu_model_temp m
GROUP BY cluster_id
ORDER BY cluster_id;

-- Evaluate model and store result
CREATE TEMP TABLE gpu_metrics_temp (metrics jsonb);
DO $$
DECLARE
	mid integer;
	metrics_result jsonb;
BEGIN
	SELECT model_id INTO mid FROM gpu_model_temp LIMIT 1;
	-- GMM is clustering, so we evaluate without labels
	metrics_result := neurondb.evaluate(mid, 'sample_test', 'features', NULL);
	INSERT INTO gpu_metrics_temp VALUES (metrics_result);
END
$$;

-- Show metrics
SELECT
	format('%-15s', 'Silhouette') AS metric,
	CASE WHEN (m.metrics::jsonb ? 'silhouette_score')
		THEN ROUND((m.metrics::jsonb ->> 'silhouette_score')::numeric, 4)
		ELSE NULL END AS value
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'Inertia'),
	CASE WHEN (m.metrics::jsonb ? 'inertia')
		THEN ROUND((m.metrics::jsonb ->> 'inertia')::numeric, 4)
		ELSE NULL END
FROM gpu_metrics_temp m;

-- Cleanup
DROP TABLE IF EXISTS gpu_model_temp;
DROP TABLE IF EXISTS gpu_metrics_temp;

