-- 035_tenant_basic.sql
-- Basic test for tenant module: multi-tenancy operations

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Tenant Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- TENANT WORKER OPERATIONS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Tenant Worker Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Create tenant worker'
SELECT create_tenant_worker('tenant1', 'queue', '{"batch_size": 100}') AS worker_created;

\echo 'Test 2: Get tenant statistics'
SELECT * FROM get_tenant_stats('tenant1');

/*-------------------------------------------------------------------
 * ---- TENANT ISOLATION ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Tenant Isolation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 3: Create table with tenant_id'
DROP TABLE IF EXISTS tenant_test_table;
CREATE TABLE tenant_test_table (
	id SERIAL PRIMARY KEY,
	embedding vector(28),
	tenant_id integer,
	label integer
);

INSERT INTO tenant_test_table (embedding, tenant_id, label)
SELECT features, (i % 3) + 1 AS tenant_id, label
FROM test_train_view, generate_series(1, 3) i
LIMIT 300;

\echo 'Test 4: Query with tenant filter'
SELECT 
	tenant_id,
	COUNT(*) AS vector_count
FROM tenant_test_table
GROUP BY tenant_id
ORDER BY tenant_id;

\echo ''
\echo '=========================================================================='
\echo '✓ Tenant Module: Basic tests complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS tenant_test_table CASCADE;

\echo 'Test completed successfully'
