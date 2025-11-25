\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: Sparse Vectors Advance Tests'
\echo '=========================================================================='

-- Test 1: Sparse index creation
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Sparse index creation'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE test_sparse_docs (
	id serial PRIMARY KEY,
	content text,
	sparse_embedding sparse_vector
);

INSERT INTO test_sparse_docs (content, sparse_embedding) VALUES
	('machine learning', '{vocab_size:30522, model:SPLADE, tokens:[100,200], weights:[0.5,0.8]}'::sparse_vector),
	('deep learning', '{vocab_size:30522, model:SPLADE, tokens:[150,250], weights:[0.6,0.7]}'::sparse_vector),
	('neural networks', '{vocab_size:30522, model:SPLADE, tokens:[120,180], weights:[0.4,0.9]}'::sparse_vector);

SELECT
	'Sparse index creation' AS test_name,
	sparse_index_create('test_sparse_docs', 'sparse_embedding', 'idx_sparse_test', 1) AS created;

-- Test 2: Hybrid dense+sparse search
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Hybrid dense+sparse search'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

ALTER TABLE test_sparse_docs ADD COLUMN dense_embedding vector(384);

UPDATE test_sparse_docs SET dense_embedding = '[0.1,0.2,0.3]'::vector || array_fill(0.0::float4, ARRAY[381]);

SELECT
	'Hybrid search' AS test_name,
	COUNT(*) AS result_count
FROM hybrid_dense_sparse_search(
	'test_sparse_docs',
	'dense_embedding',
	'sparse_embedding',
	'[0.1,0.2,0.3]'::vector || array_fill(0.0::float4, ARRAY[381]),
	'{vocab_size:30522, model:SPLADE, tokens:[100,200], weights:[0.5,0.8]}'::sparse_vector,
	10,
	0.6,
	0.4
);

-- Test 3: RRF fusion
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: RRF fusion'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'RRF fusion' AS test_name,
	rrf_fusion(10, 1.0, 2.0, 60.0) AS rrf_score;

\echo ''
\echo '✅ Advance sparse vectors tests completed'

\echo 'Test completed successfully'
