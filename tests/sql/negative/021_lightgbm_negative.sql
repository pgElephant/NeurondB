-- 021_lightgbm_negative.sql
-- Negative test for lightgbm
-- All possible negative tests with 1000 rows only

\timing on
\pset footer off
\pset pager off
\pset tuples_only off
\set ON_ERROR_STOP off
SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=========================================================================='
\echo 'LightGBM - Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/* Setup: Create test views if they don't exist */
DO $$
BEGIN
	IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		DROP VIEW IF EXISTS test_train_view;
		CREATE VIEW test_train_view AS
		SELECT features, label FROM sample_train LIMIT 1000;
	END IF;
	IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_test') THEN
		DROP VIEW IF EXISTS test_test_view;
		CREATE VIEW test_test_view AS
		SELECT features, label FROM sample_test LIMIT 1000;
	END IF;
END
$$;

\echo ''
\echo 'Test 1: NULL Table Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('lightgbm', NULL, 'features', 'label', '{}'::jsonb);

\echo ''
\echo 'Test 2: NULL Feature Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('lightgbm', 'test_train_view', NULL, 'label', '{}'::jsonb);

\echo ''
\echo 'Test 3: NULL Label Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('lightgbm', 'test_train_view', 'features', NULL, '{}'::jsonb);

\echo ''
\echo 'Test 4: Invalid Table Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('lightgbm', 'nonexistent_table', 'features', 'label', '{}'::jsonb);

\echo ''
\echo 'Test 5: Invalid Feature Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('lightgbm', 'test_train_view', 'invalid_feature', 'label', '{}'::jsonb);

\echo ''
\echo 'Test 6: Invalid Label Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('lightgbm', 'test_train_view', 'features', 'invalid_label', '{}'::jsonb);

\echo ''
\echo 'Test 7: Invalid Hyperparameters'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('lightgbm', 'test_train_view', 'features', 'label', '{"num_iterations":-1}'::jsonb);

\echo ''
\echo 'Test 8: NULL Model ID for Prediction'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.predict(NULL, array_to_vector_float8(ARRAY[1.0, 2.0]::double precision[]));

\echo ''
\echo 'Test 9: Invalid Model ID for Prediction'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.predict(-1, array_to_vector_float8(ARRAY[1.0, 2.0]::double precision[]));

\echo ''
\echo 'Test 10: NULL Features for Prediction'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	model_id int;
BEGIN
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'lightgbm' ORDER BY m.model_id DESC LIMIT 1;
	IF model_id IS NOT NULL THEN
		SELECT neurondb.predict(model_id, NULL);
	END IF;
END $$;

\echo ''
\echo '✓ lightgbm negative test complete'
