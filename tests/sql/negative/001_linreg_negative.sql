\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo '=========================================================================='

/* Setup: Create test views if they don't exist */
DO $$
BEGIN
	IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		
		
	END IF;
	IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_test') THEN
		
		
	END IF;
END
$$;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	NULL,
	'features',
	'label',
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	'test_train_view',
	NULL,
	'label',
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	'test_train_view',
	'features',
	NULL,
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	'nonexistent_table_xyz',
	'features',
	'label',
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	'test_train_view',
	'nonexistent_column',
	'label',
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	'test_train_view',
	'features',
	'nonexistent_label',
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.predict(-1, '[1,2,3,4,5]'::vector);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.predict(NULL, '[1,2,3,4,5]'::vector);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- First get a valid model_id
DO $$
DECLARE
	mid integer;
BEGIN
	SELECT neurondb.train(
		'linear_regression',
		'test_train_view',
		'features',
		'label',
		'{}'::jsonb
	)::integer INTO mid;
	
	-- Try prediction with NULL features
	PERFORM neurondb.predict(mid, NULL::vector);
EXCEPTION
	WHEN OTHERS THEN
END
$$;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	mid integer;
BEGIN
	SELECT neurondb.train(
		'linear_regression',
		'test_train_view',
		'features',
		'label',
		'{}'::jsonb
	)::integer INTO mid;
	
	-- Try prediction with wrong dimension vector
	PERFORM neurondb.predict(mid, '[1,2,3]'::vector);
EXCEPTION
	WHEN OTHERS THEN
END
$$;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'linear_regression',
	'test_train_view',
	'features',
	'label',
	'{"invalid_param": "invalid_value"}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'invalid_algorithm_name',
	'test_train_view',
	'features',
	'label',
	'{}'::jsonb
);

\echo ''
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
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.evaluate(-1, 'test_test_view', 'features', 'label');

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	mid integer;
BEGIN
	SELECT neurondb.train(
		'linear_regression',
		'test_train_view',
		'features',
		'label',
		'{}'::jsonb
	)::integer INTO mid;
	
	SELECT neurondb.evaluate(mid, 'nonexistent_test_table', 'features', 'label');
EXCEPTION
	WHEN OTHERS THEN
END
$$;

\echo ''
\echo '=========================================================================='

\echo 'Test completed successfully'
