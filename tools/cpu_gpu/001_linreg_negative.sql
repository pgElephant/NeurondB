\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Linear Regression - Negative Test Cases (Error Handling)'
\echo '=========================================================================='

\echo ''
\echo 'Test 1: NULL Table Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	NULL,
	'features',
	'label',
	'{}'::jsonb
);

\echo ''
\echo 'Test 2: NULL Feature Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	'sample_train',
	NULL,
	'label',
	'{}'::jsonb
);

\echo ''
\echo 'Test 3: NULL Target Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	'sample_train',
	'features',
	NULL,
	'{}'::jsonb
);

\echo ''
\echo 'Test 4: Non-existent Table'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	'nonexistent_table_xyz',
	'features',
	'label',
	'{}'::jsonb
);

\echo ''
\echo 'Test 5: Non-existent Feature Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	'sample_train',
	'nonexistent_column',
	'label',
	'{}'::jsonb
);

\echo ''
\echo 'Test 6: Non-existent Target Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	'sample_train',
	'features',
	'nonexistent_label',
	'{}'::jsonb
);

\echo ''
\echo 'Test 7: Invalid Model ID for Prediction'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.predict(-1, '[1,2,3,4,5]'::vector);

\echo ''
\echo 'Test 8: NULL Model ID for Prediction'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.predict(NULL, '[1,2,3,4,5]'::vector);

\echo ''
\echo 'Test 9: NULL Features for Prediction'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- First get a valid model_id
DO $$
DECLARE
	mid integer;
BEGIN
	SELECT neurondb.train(
		'linear_regression',
		'sample_train',
		'features',
		'label',
		'{}'::jsonb
	)::integer INTO mid;
	
	-- Try prediction with NULL features
	PERFORM neurondb.predict(mid, NULL::vector);
EXCEPTION
	WHEN OTHERS THEN
		RAISE NOTICE 'Expected error: %', SQLERRM;
END
$$;

\echo ''
\echo 'Test 10: Mismatched Feature Dimensions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	mid integer;
BEGIN
	SELECT neurondb.train(
		'linear_regression',
		'sample_train',
		'features',
		'label',
		'{}'::jsonb
	)::integer INTO mid;
	
	-- Try prediction with wrong dimension vector
	PERFORM neurondb.predict(mid, '[1,2,3]'::vector);
EXCEPTION
	WHEN OTHERS THEN
		RAISE NOTICE 'Expected error: %', SQLERRM;
END
$$;

\echo ''
\echo 'Test 11: Invalid Hyperparameters JSON'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	'sample_train',
	'features',
	'label',
	'{"invalid_param": "invalid_value"}'::jsonb
);

\echo ''
\echo 'Test 12: Invalid Algorithm Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'invalid_algorithm_name',
	'sample_train',
	'features',
	'label',
	'{}'::jsonb
);

\echo ''
\echo 'Test 13: Empty Training Table'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE empty_train (
	features vector,
	label float8
);

SELECT neurondb.train(
	'linear_regression',
	'empty_train',
	'features',
	'label',
	'{}'::jsonb
);

DROP TABLE IF EXISTS empty_train;

\echo ''
\echo 'Test 14: Evaluation with Invalid Model ID'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.evaluate(-1, 'sample_test', 'features', 'label');

\echo ''
\echo 'Test 15: Evaluation with Non-existent Test Table'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	mid integer;
BEGIN
	SELECT neurondb.train(
		'linear_regression',
		'sample_train',
		'features',
		'label',
		'{}'::jsonb
	)::integer INTO mid;
	
	SELECT neurondb.evaluate(mid, 'nonexistent_test_table', 'features', 'label');
EXCEPTION
	WHEN OTHERS THEN
		RAISE NOTICE 'Expected error: %', SQLERRM;
END
$$;

\echo ''
\echo 'Negative Test Cases Complete!'
\echo '=========================================================================='

