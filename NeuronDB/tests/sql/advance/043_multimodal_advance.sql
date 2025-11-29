\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: Multi-Modal Embeddings Advance Tests'
\echo '=========================================================================='

-- Test 1: Cross-modal search setup
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Cross-modal search setup'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE test_multimodal (
	id serial PRIMARY KEY,
	content text,
	modality text,
	embedding vector(512)
);

INSERT INTO test_multimodal (content, modality, embedding) VALUES
	('cat image', 'image', '[0.1,0.2,0.3]'::vector || array_fill(0.0::float4, ARRAY[509])),
	('dog image', 'image', '[0.2,0.3,0.4]'::vector || array_fill(0.0::float4, ARRAY[509])),
	('bird image', 'image', '[0.3,0.4,0.5]'::vector || array_fill(0.0::float4, ARRAY[509]));

SELECT
	'Cross-modal setup' AS test_name,
	COUNT(*) AS doc_count
FROM test_multimodal;

-- Test 2: Cross-modal search (text query, image results)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Cross-modal search'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Cross-modal search' AS test_name,
	COUNT(*) AS result_count
FROM cross_modal_search(
	'test_multimodal',
	'embedding',
	'text',
	'find a cat',
	'image',
	5
);

\echo ''
\echo '✅ Advance multi-modal embeddings tests completed'

\echo 'Test completed successfully'
