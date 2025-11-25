-- 033_util_basic.sql
-- Basic test for utility module: configuration, security, hooks, distributed, safe memory

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Utility Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- CONFIGURATION MANAGEMENT ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Configuration Management Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Show all vector configuration'
SELECT * FROM show_vector_config() LIMIT 20;

\echo 'Test 2: Get specific configuration'
SELECT * FROM get_vector_config('ef_construction');

\echo 'Test 3: Set configuration'
SELECT set_vector_config('ef_construction', '200') AS config_set;

\echo 'Test 4: Reset configuration'
SELECT reset_vector_config('ef_construction') AS config_reset;

/*-------------------------------------------------------------------
 * ---- SECURITY OPERATIONS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Security Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 5: Post-quantum encryption'
SELECT encrypt_postquantum(vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector) AS encrypted;

\echo 'Test 6: Confidential compute mode'
SELECT enable_confidential_compute(true) AS confidential_enabled;
SELECT enable_confidential_compute(false) AS confidential_disabled;

\echo 'Test 7: Access mask setting'
SELECT set_access_mask('test_role', 'l2,cosine', 'hnsw,ivf') AS access_mask_set;

/*-------------------------------------------------------------------
 * ---- HOOK OPERATIONS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Hook Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 8: Register custom operator'
SELECT register_custom_operator('custom_distance', 'vector_custom_distance') AS operator_registered;

\echo 'Test 9: Enable vector replication'
SELECT enable_vector_replication('vector_pub') AS replication_enabled;

\echo 'Test 10: Create vector FDW'
SELECT create_vector_fdw('vector_fdw', 'remote_host', 5432) AS fdw_created;

/*-------------------------------------------------------------------
 * ---- DISTRIBUTED OPERATIONS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Distributed Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 11: Federated vector query'
SELECT federated_vector_query('localhost', 'SELECT * FROM test_table') AS federated_result;

\echo ''
\echo '=========================================================================='
\echo '✓ Utility Module: Basic tests complete'
\echo '=========================================================================='

\echo 'Test completed successfully'
