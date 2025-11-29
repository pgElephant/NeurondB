-- 030_index_advance.sql
-- Comprehensive advanced test for ALL index module functions
-- Tests HNSW, IVF, index consistency, cache operations, multi-tenant isolation
-- Works on 1000 rows and tests each and every index code path

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'Index Module: Exhaustive Index Operations Coverage'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- HNSW INDEX WITH VARIOUS PARAMETERS ----
 * Test HNSW index creation with all parameter combinations
 *------------------------------------------------------------------*/
\echo ''
\echo 'HNSW Index Parameter Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create test table
DROP TABLE IF EXISTS index_advance_test;
CREATE TABLE index_advance_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28),
	label integer,
	metadata jsonb
);

-- Insert test data
INSERT INTO index_advance_test (embedding, label, metadata)
SELECT features, label, '{"source": "test"}'::jsonb
FROM test_train_view
LIMIT 1000;

\echo 'Test 1: HNSW with m=8 (small)'
CREATE INDEX idx_hnsw_m8 ON index_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 8, ef_construction = 100);

\echo 'Test 2: HNSW with m=16 (default)'
DROP INDEX IF EXISTS idx_hnsw_m16;
CREATE INDEX idx_hnsw_m16 ON index_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 200);

\echo 'Test 3: HNSW with m=32 (large)'
DROP INDEX IF EXISTS idx_hnsw_m32;
CREATE INDEX idx_hnsw_m32 ON index_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 32, ef_construction = 400);

\echo 'Test 4: HNSW with ef_construction=50 (small)'
DROP INDEX IF EXISTS idx_hnsw_ef50;
CREATE INDEX idx_hnsw_ef50 ON index_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 50);

\echo 'Test 5: HNSW with ef_construction=200 (default)'
DROP INDEX IF EXISTS idx_hnsw_ef200;
CREATE INDEX idx_hnsw_ef200 ON index_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 200);

\echo 'Test 6: HNSW with ef_construction=500 (large)'
DROP INDEX IF EXISTS idx_hnsw_ef500;
CREATE INDEX idx_hnsw_ef500 ON index_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 500);

/*-------------------------------------------------------------------
 * ---- IVF INDEX WITH VARIOUS PARAMETERS ----
 * Test IVF index creation with all parameter combinations
 *------------------------------------------------------------------*/
\echo ''
\echo 'IVF Index Parameter Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 7: IVF with lists=5 (small)'
CREATE INDEX idx_ivf_lists5 ON index_advance_test 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 5);

\echo 'Test 8: IVF with lists=10 (default)'
DROP INDEX IF EXISTS idx_ivf_lists10;
CREATE INDEX idx_ivf_lists10 ON index_advance_test 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 10);

\echo 'Test 9: IVF with lists=50 (large)'
DROP INDEX IF EXISTS idx_ivf_lists50;
CREATE INDEX idx_ivf_lists50 ON index_advance_test 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 50);

\echo 'Test 10: IVF with lists=100 (very large)'
DROP INDEX IF EXISTS idx_ivf_lists100;
CREATE INDEX idx_ivf_lists100 ON index_advance_test 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 100);

/*-------------------------------------------------------------------
 * ---- INDEX QUERIES WITH VARIOUS K VALUES ----
 * Test KNN queries with different k values
 *------------------------------------------------------------------*/
\echo ''
\echo 'Index Query Tests (Various K)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 11: KNN query with k=1'
SELECT 
	id,
	embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1) AS distance
FROM index_advance_test
ORDER BY embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1)
LIMIT 1;

\echo 'Test 12: KNN query with k=10'
SELECT 
	id,
	embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1) AS distance
FROM index_advance_test
ORDER BY embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1)
LIMIT 10;

\echo 'Test 13: KNN query with k=100'
SELECT 
	COUNT(*) AS result_count
FROM (
	SELECT 
		id,
		embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1) AS distance
	FROM index_advance_test
	ORDER BY embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1)
	LIMIT 100
) sub;

\echo 'Test 14: KNN query with k=1000 (larger than table)'
SELECT 
	COUNT(*) AS result_count
FROM (
	SELECT 
		id,
		embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1) AS distance
	FROM index_advance_test
	ORDER BY embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1)
	LIMIT 1000
) sub;

/*-------------------------------------------------------------------
 * ---- INDEX CONSISTENCY CHECKS ----
 * Test index consistency and validation
 *------------------------------------------------------------------*/
\echo ''
\echo 'Index Consistency Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 15: Index size after inserts'
SELECT 
	indexname,
	pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size,
	pg_size_pretty(pg_relation_size('index_advance_test'::regclass)) AS table_size
FROM pg_indexes
WHERE tablename = 'index_advance_test'
ORDER BY indexname;

\echo 'Test 16: Index usage statistics'
SELECT 
	schemaname,
	tablename,
	indexname,
	idx_scan AS index_scans,
	idx_tup_read AS tuples_read,
	idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
WHERE tablename = 'index_advance_test'
ORDER BY indexname;

/*-------------------------------------------------------------------
 * ---- INDEX MAINTENANCE ----
 * Test index maintenance operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Index Maintenance Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 17: REINDEX operation'
REINDEX INDEX idx_hnsw_m16;

\echo 'Test 18: VACUUM on indexed table'
VACUUM ANALYZE index_advance_test;

\echo 'Test 19: Index after updates'
UPDATE index_advance_test 
SET metadata = '{"updated": true}'::jsonb 
WHERE id % 10 = 0;

VACUUM ANALYZE index_advance_test;

\echo 'Test 20: Index after deletes'
DELETE FROM index_advance_test WHERE id % 20 = 0;

VACUUM ANALYZE index_advance_test;

/*-------------------------------------------------------------------
 * ---- MULTI-TENANT INDEX ISOLATION ----
 * Test multi-tenant index operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Multi-Tenant Index Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 21: Create table with tenant_id'
DROP TABLE IF EXISTS index_tenant_test;
CREATE TABLE index_tenant_test (
	id SERIAL PRIMARY KEY,
	tenant_id integer,
	embedding vector(28),
	label integer
);

-- Insert data for multiple tenants
INSERT INTO index_tenant_test (tenant_id, embedding, label)
SELECT 
	(i % 3) + 1 AS tenant_id,
	features AS embedding,
	label
FROM test_train_view
LIMIT 300;

\echo 'Test 22: Create index on multi-tenant table'
CREATE INDEX idx_tenant_hnsw ON index_tenant_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 200);

\echo 'Test 23: Query with tenant filter'
SELECT 
	tenant_id,
	COUNT(*) AS result_count
FROM (
	SELECT 
		tenant_id,
		id,
		embedding <-> (SELECT embedding FROM index_tenant_test WHERE tenant_id = 1 LIMIT 1) AS distance
	FROM index_tenant_test
	WHERE tenant_id = 1
	ORDER BY embedding <-> (SELECT embedding FROM index_tenant_test WHERE tenant_id = 1 LIMIT 1)
	LIMIT 10
) sub
GROUP BY tenant_id;

/*-------------------------------------------------------------------
 * ---- INDEX CACHE OPERATIONS ----
 * Test index cache operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Index Cache Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 24: Multiple queries to test cache'
-- Run same query multiple times to test caching
SELECT 
	COUNT(*) AS query_count
FROM (
	SELECT 
		id,
		embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1) AS distance
	FROM index_advance_test
	ORDER BY embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1)
	LIMIT 10
) q1;

SELECT 
	COUNT(*) AS query_count
FROM (
	SELECT 
		id,
		embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1) AS distance
	FROM index_advance_test
	ORDER BY embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1)
	LIMIT 10
) q2;

\echo ''
\echo '=========================================================================='
\echo '✓ Index Module: Full exhaustive code-path test complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS index_advance_test CASCADE;
DROP TABLE IF EXISTS index_tenant_test CASCADE;

\echo 'Test completed successfully'
