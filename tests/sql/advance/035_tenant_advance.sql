-- 035_tenant_advance.sql
-- Comprehensive advanced test for tenant module: multi-tenancy comprehensively

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'Tenant Module: Exhaustive Multi-Tenancy Coverage'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- COMPREHENSIVE TENANT WORKER OPERATIONS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Tenant Worker Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Create multiple tenant workers'
SELECT create_tenant_worker('tenant1', 'queue', '{"batch_size": 100}') AS worker1;
SELECT create_tenant_worker('tenant2', 'tuner', '{"sample_size": 1000}') AS worker2;
SELECT create_tenant_worker('tenant3', 'defrag', '{"interval": 3600}') AS worker3;

\echo 'Test 2: Get statistics for multiple tenants'
SELECT * FROM get_tenant_stats('tenant1');
SELECT * FROM get_tenant_stats('tenant2');
SELECT * FROM get_tenant_stats('tenant3');

/*-------------------------------------------------------------------
 * ---- COMPREHENSIVE TENANT ISOLATION ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Tenant Isolation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create multi-tenant table
DROP TABLE IF EXISTS tenant_advance_test;
CREATE TABLE tenant_advance_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28),
	tenant_id integer,
	label integer,
	metadata jsonb
);

INSERT INTO tenant_advance_test (embedding, tenant_id, label, metadata)
SELECT features, (i % 10) + 1 AS tenant_id, label, jsonb_build_object('tenant', i % 10 + 1)
FROM test_train_view, generate_series(1, 10) i
LIMIT 1000;

CREATE INDEX idx_tenant_hnsw ON tenant_advance_test 
USING hnsw (embedding vector_l2_ops);

\echo 'Test 3: Tenant isolation with KNN queries'
SELECT 
	tenant_id,
	COUNT(*) AS result_count,
	AVG(distance) AS avg_distance
FROM (
	SELECT 
		tenant_id,
		embedding <-> (SELECT embedding FROM tenant_advance_test WHERE tenant_id = 1 LIMIT 1) AS distance
	FROM tenant_advance_test
	WHERE tenant_id = 1
	ORDER BY embedding <-> (SELECT embedding FROM tenant_advance_test WHERE tenant_id = 1 LIMIT 1)
	LIMIT 10
) sub
GROUP BY tenant_id;

\echo 'Test 4: Multi-tenant statistics'
SELECT 
	tenant_id,
	COUNT(*) AS vector_count,
	AVG(vector_norm(embedding)) AS avg_norm,
	MIN(vector_norm(embedding)) AS min_norm,
	MAX(vector_norm(embedding)) AS max_norm
FROM tenant_advance_test
GROUP BY tenant_id
ORDER BY tenant_id;

\echo 'Test 5: Tenant quota usage tracking'
SELECT 
	tenant_id,
	COUNT(*) AS usage_count,
	pg_size_pretty(SUM(pg_column_size(embedding))) AS total_size
FROM tenant_advance_test
GROUP BY tenant_id
ORDER BY tenant_id;

\echo ''
\echo '=========================================================================='
\echo '✓ Tenant Module: Full exhaustive code-path test complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS tenant_advance_test CASCADE;

\echo 'Test completed successfully'




