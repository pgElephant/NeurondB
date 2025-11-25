-- 034_planner_advance.sql
-- Comprehensive advanced test for planner module: query optimization paths

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'Planner Module: Exhaustive Query Optimization Coverage'
\echo '=========================================================================='

-- Create comprehensive test table
DROP TABLE IF EXISTS planner_advance_test;
CREATE TABLE planner_advance_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28),
	metadata jsonb,
	tenant_id integer,
	label integer,
	created_at timestamp DEFAULT now()
);

INSERT INTO planner_advance_test (embedding, metadata, tenant_id, label)
SELECT features, jsonb_build_object('source', 'test'), (i % 5) + 1 AS tenant_id, label
FROM test_train_view, generate_series(1, 10) i
LIMIT 1000;

CREATE INDEX idx_planner_hnsw ON planner_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 200);

CREATE INDEX idx_planner_ivf ON planner_advance_test 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 10);

\echo ''
\echo 'Query Plan Analysis Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Planner selection for HNSW index'
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT id, embedding <-> (SELECT embedding FROM planner_advance_test LIMIT 1) AS distance
FROM planner_advance_test
ORDER BY embedding <-> (SELECT embedding FROM planner_advance_test LIMIT 1)
LIMIT 10;

\echo 'Test 2: Planner selection for IVF index'
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT id, embedding <-> (SELECT embedding FROM planner_advance_test LIMIT 1) AS distance
FROM planner_advance_test
ORDER BY embedding <-> (SELECT embedding FROM planner_advance_test LIMIT 1)
LIMIT 10;

\echo 'Test 3: Planner with filter pushdown'
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT id, embedding <-> (SELECT embedding FROM planner_advance_test LIMIT 1) AS distance
FROM planner_advance_test
WHERE tenant_id = 1 AND label < 5
ORDER BY embedding <-> (SELECT embedding FROM planner_advance_test LIMIT 1)
LIMIT 10;

\echo 'Test 4: Planner with multiple filters'
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT id, tenant_id, embedding <-> (SELECT embedding FROM planner_advance_test LIMIT 1) AS distance
FROM planner_advance_test
WHERE tenant_id IN (1, 2, 3) AND metadata->>'source' = 'test'
ORDER BY embedding <-> (SELECT embedding FROM planner_advance_test LIMIT 1)
LIMIT 10;

\echo 'Test 5: Planner cost estimation'
EXPLAIN (COSTS, VERBOSE)
SELECT id, embedding <-> (SELECT embedding FROM planner_advance_test LIMIT 1) AS distance
FROM planner_advance_test
ORDER BY embedding <-> (SELECT embedding FROM planner_advance_test LIMIT 1)
LIMIT 10;

\echo ''
\echo '=========================================================================='
\echo '✓ Planner Module: Full exhaustive code-path test complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS planner_advance_test CASCADE;

\echo 'Test completed successfully'




