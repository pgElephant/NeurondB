\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'K-Nearest Neighbors - Negative Test Cases (Error Handling)'
\echo '=========================================================================='

\echo ''
\echo 'Test 1: NULL Table Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	NULL,
	'features',
	'label',
	'{}'::jsonb
);

\echo ''
\echo 'Test 2: NULL Feature Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	NULL,
	'label',
	'{}'::jsonb
);

\echo ''
\echo 'Test 3: NULL Target Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	'features',
	NULL,
	'{}'::jsonb
);

\echo ''
\echo 'Test 4: Non-existent Table'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'nonexistent_table_xyz',
	'features',
	'label',
	'{}'::jsonb
);

\echo ''
\echo 'Test 5: Non-existent Feature Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	'nonexistent_column',
	'label',
	'{}'::jsonb
);

\echo ''
\echo 'Test 6: Non-existent Target Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
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

DO $$
DECLARE
	mid integer;
BEGIN
	SELECT neurondb.train(
		'knn',
		'test_train_view',
		'features',
		'label',
		'{"k": 5}'::jsonb
	)::integer INTO mid;
	
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
		'knn',
		'test_train_view',
		'features',
		'label',
		'{"k": 5}'::jsonb
	)::integer INTO mid;
	
	PERFORM neurondb.predict(mid, '[1,2,3]'::vector);
EXCEPTION
	WHEN OTHERS THEN
		RAISE NOTICE 'Expected error: %', SQLERRM;
END
$$;

\echo ''
\echo 'Test 11: Invalid Hyperparameters (k=0)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	'features',
	'label',
	'{"k": 0}'::jsonb
);

\echo ''
\echo 'Test 12: Invalid Hyperparameters (k < 0)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	'features',
	'label',
	'{"k": -1}'::jsonb
);

\echo ''
\echo 'Test 13: Invalid Hyperparameters (k too large)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	'features',
	'label',
	'{"k": 100000}'::jsonb
);

\echo ''
\echo 'Test 14: Empty Training Table'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE empty_train (
	features vector,
	label float8
);

SELECT neurondb.train(
	'knn',
	'empty_train',
	'features',
	'label',
	'{"k": 5}'::jsonb
);

DROP TABLE IF EXISTS empty_train;

\echo ''
\echo 'Test 15: Evaluation with Invalid Model ID'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.evaluate(-1, 'test_test_view', 'features', 'label');

\echo ''
\echo 'Negative KNN Test Complete!'
\echo '=========================================================================='

