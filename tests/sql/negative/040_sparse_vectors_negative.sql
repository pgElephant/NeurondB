\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: Sparse Vectors Negative Tests'
\echo '=========================================================================='

-- Test 1: Invalid sparse vector format
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Invalid sparse vector format'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM sparse_vector_in('invalid format');
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected invalid format: %', SQLERRM;
	END;
END$$;

-- Test 2: Empty tokens array
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Empty tokens array'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM sparse_vector_in('{vocab_size:30522, model:SPLADE, tokens:[], weights:[]}');
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected empty tokens: %', SQLERRM;
	END;
END$$;

-- Test 3: Mismatched dimensions in dot product
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Mismatched vocab sizes'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Dot product with different vocab sizes' AS test_name,
	sparse_vector_dot_product(
		'{vocab_size:30522, model:SPLADE, tokens:[100], weights:[0.5]}'::sparse_vector,
		'{vocab_size:50000, model:SPLADE, tokens:[100], weights:[0.3]}'::sparse_vector
	) AS result;

\echo ''
\echo '✅ Negative sparse vectors tests completed'

\echo 'Test completed successfully'
