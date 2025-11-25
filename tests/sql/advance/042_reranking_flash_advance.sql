\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: Flash Attention Reranking Advance Tests'
\echo '=========================================================================='

-- Test 1: Large batch reranking
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Large batch reranking'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH large_batch AS (
	SELECT array_agg('Document ' || generate_series::text) AS docs
	FROM generate_series(1, 100)
)
SELECT
	'Large batch rerank' AS test_name,
	COUNT(*) AS result_count
FROM large_batch,
LATERAL rerank_flash(
	'search query',
	docs,
	NULL,
	10
);

-- Test 2: GPU support check
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: GPU support check'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'GPU support' AS test_name,
	current_setting('neurondb.gpu_enabled', true) AS gpu_enabled,
	current_setting('neurondb.gpu_backend', true) AS gpu_backend;

\echo ''
\echo '✅ Advance Flash Attention reranking tests completed'

\echo 'Test completed successfully'
