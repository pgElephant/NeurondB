\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Gaussian Mixture Model - Negative Test Cases (Error Handling)'
\echo '=========================================================================='

\echo ''
\echo 'Test 1: NULL Table Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	NULL,
	'features',
	NULL,
	'{}'::jsonb
);

\echo ''
\echo 'Test 2: NULL Feature Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	'test_train_view',
	NULL,
	NULL,
	'{}'::jsonb
);

\echo ''
\echo 'Test 3: Non-existent Table'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	'nonexistent_table_xyz',
	'features',
	NULL,
	'{}'::jsonb
);

\echo ''
\echo 'Test 4: Non-existent Feature Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	'test_train_view',
	'nonexistent_column',
	NULL,
	'{}'::jsonb
);

\echo ''
\echo 'Test 5: Invalid Model ID for Prediction'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.predict(-1, '[1,2,3,4,5]'::vector);

\echo ''
\echo 'Test 6: NULL Model ID for Prediction'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.predict(NULL, '[1,2,3,4,5]'::vector);

\echo ''
\echo 'Test 7: NULL Features for Prediction'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	mid integer;
BEGIN
	SELECT neurondb.train(
		'gmm',
		'test_train_view',
		'features',
		NULL,
		'{"k": 3}'::jsonb
	)::integer INTO mid;
	
	PERFORM neurondb.predict(mid, NULL::vector);
EXCEPTION
	WHEN OTHERS THEN
		RAISE NOTICE 'Expected error: %', SQLERRM;
END
$$;

\echo ''
\echo 'Test 8: Mismatched Feature Dimensions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	mid integer;
BEGIN
	SELECT neurondb.train(
		'gmm',
		'test_train_view',
		'features',
		NULL,
		'{"k": 3}'::jsonb
	)::integer INTO mid;
	
	PERFORM neurondb.predict(mid, '[1,2,3]'::vector);
EXCEPTION
	WHEN OTHERS THEN
		RAISE NOTICE 'Expected error: %', SQLERRM;
END
$$;

\echo ''
\echo 'Test 9: Invalid Hyperparameters (k=0)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	'test_train_view',
	'features',
	NULL,
	'{"k": 0}'::jsonb
);

\echo ''
\echo 'Test 10: Invalid Hyperparameters (k < 0)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	'test_train_view',
	'features',
	NULL,
	'{"k": -1}'::jsonb
);

\echo ''
\echo 'Test 11: Invalid Hyperparameters (k too large)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	'test_train_view',
	'features',
	NULL,
	'{"k": 10000}'::jsonb
);

\echo ''
\echo 'Test 12: Empty Training Table'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE empty_train (
	features vector
);

SELECT neurondb.train(
	'gmm',
	'empty_train',
	'features',
	NULL,
	'{"k": 3}'::jsonb
);

DROP TABLE IF EXISTS empty_train;

\echo ''
\echo 'Test 13: Evaluation with Invalid Model ID'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.evaluate(-1, 'test_test_view', 'features', NULL);

\echo ''
\echo 'Negative GMM Test Complete!'
\echo '=========================================================================='

