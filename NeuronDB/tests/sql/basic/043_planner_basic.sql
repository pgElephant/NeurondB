-- 034_planner_basic.sql
-- Basic test for planner module: query optimization paths

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Planner Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- QUERY OPTIMIZATION ----
 * Test planner optimization through EXPLAIN
 *------------------------------------------------------------------*/
\echo ''
\echo 'Query Optimization Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create test table
DROP TABLE IF EXISTS planner_test_table;
CREATE TABLE planner_test_table (
	id SERIAL PRIMARY KEY,
	embedding vector(28),
	label integer
);

INSERT INTO planner_test_table (embedding, label)
SELECT features, label
FROM test_train_view
LIMIT 100;

CREATE INDEX idx_planner_hnsw ON planner_test_table 
USING hnsw (embedding vector_l2_ops);

\echo 'Test 1: EXPLAIN query plan for KNN search'
EXPLAIN (ANALYZE, BUFFERS) 
SELECT id, embedding <-> (SELECT embedding FROM planner_test_table LIMIT 1) AS distance
FROM planner_test_table
ORDER BY embedding <-> (SELECT embedding FROM planner_test_table LIMIT 1)
LIMIT 10;

\echo 'Test 2: EXPLAIN query plan with filter'
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, embedding <-> (SELECT embedding FROM planner_test_table LIMIT 1) AS distance
FROM planner_test_table
WHERE label < 5
ORDER BY embedding <-> (SELECT embedding FROM planner_test_table LIMIT 1)
LIMIT 10;

\echo ''
\echo '=========================================================================='
\echo '✓ Planner Module: Basic tests complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS planner_test_table CASCADE;

\echo 'Test completed successfully'
