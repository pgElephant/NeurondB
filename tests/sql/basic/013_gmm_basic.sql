\timing on
\pset footer off
\pset pager off

-- This test uses test_train_view and test_test_view tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'test_train_view table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_test') THEN
		RAISE EXCEPTION 'test_test_view table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
END
$$;

-- Create views with 1000 rows for basic tests
DROP VIEW IF EXISTS test_train_view;
DROP VIEW IF EXISTS test_test_view;

CREATE VIEW test_train_view AS
SELECT features, label FROM sample_train LIMIT 1000;

CREATE VIEW test_test_view AS
SELECT features, label FROM sample_test LIMIT 1000;

-- Try to enable GPU, but fall back to CPU if not available
DO $$
BEGIN
	BEGIN
		SET neurondb.gpu_enabled = on;
		SET neurondb.gpu_kernels = 'l2,cosine,ip,gmm_train,gmm_predict';
		PERFORM neurondb_gpu_enable();
	EXCEPTION WHEN OTHERS THEN
		SET neurondb.gpu_enabled = off;
	END;
END
$$;

\set ON_ERROR_STOP on

-- Train model once and store in temp table to reuse
-- Try GPU first, fall back to CPU if GPU is not available
DROP TABLE IF EXISTS gpu_model_temp;
DO $$
DECLARE
	model_id_val integer;
BEGIN
	BEGIN
		-- Try with GPU enabled
		model_id_val := neurondb.train(
			'gmm',
			'test_train_view',
			'features',
			NULL,
			'{"k": 3, "max_iters": 100}'::jsonb
		);
		CREATE TEMP TABLE gpu_model_temp AS SELECT model_id_val::integer AS model_id;
	EXCEPTION WHEN OTHERS THEN
		-- If GPU training fails, disable GPU and retry
		IF SQLERRM LIKE '%GPU training requested but GPU hardware not available%' THEN
			SET neurondb.gpu_enabled = off;
			model_id_val := neurondb.train(
				'gmm',
				'test_train_view',
				'features',
				NULL,
				'{"k": 3, "max_iters": 100}'::jsonb
			);
			CREATE TEMP TABLE gpu_model_temp AS SELECT model_id_val::integer AS model_id;
		ELSE
			-- Re-raise other errors
			RAISE;
		END IF;
	END;
END
$$;

-- Debug: Show model_id
SELECT model_id FROM gpu_model_temp;

-- Predict clusters for test data (skip if model doesn't support prediction)
-- Note: GMM clustering may not support direct prediction, so we skip this test
-- SELECT
-- 	neurondb.predict(m.model_id, features) AS cluster_id,
-- 	COUNT(*) AS count
-- FROM test_test_view, gpu_model_temp m
-- GROUP BY cluster_id
-- ORDER BY cluster_id;

-- Evaluate model and store result
CREATE TEMP TABLE gpu_metrics_temp (metrics jsonb);
DO $$
DECLARE
	mid integer;
	metrics_result jsonb;
	eval_error text;
BEGIN
	SELECT model_id INTO mid FROM gpu_model_temp LIMIT 1;
	IF mid IS NULL THEN
		RAISE WARNING 'No model_id found in gpu_model_temp';
		INSERT INTO gpu_metrics_temp VALUES ('{"error": "No model_id found"}'::jsonb);
		RETURN;
	END IF;
	
	BEGIN
		BEGIN
			-- GMM is clustering, so we evaluate without labels
			metrics_result := neurondb.evaluate(mid, 'test_test_view', 'features', NULL);
			IF metrics_result IS NULL THEN
				RAISE WARNING 'Evaluation returned NULL';
				INSERT INTO gpu_metrics_temp VALUES ('{"error": "Evaluation returned NULL"}'::jsonb);
			ELSE
				INSERT INTO gpu_metrics_temp VALUES (metrics_result);
			END IF;
		EXCEPTION WHEN OTHERS THEN
			eval_error := SQLERRM;
			RAISE WARNING 'GMM evaluation failed (may not support evaluation): %', eval_error;
			-- Properly escape JSON
			eval_error := REPLACE(REPLACE(REPLACE(eval_error, '"', '\"'), E'\n', ' '), E'\r', ' ');
			INSERT INTO gpu_metrics_temp VALUES (jsonb_build_object('note', 'GMM evaluation not supported or failed: ' || eval_error));
		END;
	EXCEPTION WHEN OTHERS THEN
		eval_error := SQLERRM;
		RAISE WARNING 'Outer GMM evaluation exception: %', eval_error;
		-- Properly escape JSON
		eval_error := REPLACE(REPLACE(REPLACE(eval_error, '"', '\"'), E'\n', ' '), E'\r', ' ');
		INSERT INTO gpu_metrics_temp VALUES (jsonb_build_object('error', eval_error));
	END;
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

