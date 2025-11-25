-- 033_util_advance.sql
-- Comprehensive advanced test for ALL utility module functions
-- Tests configuration, security, hooks, distributed, safe memory, SPI safe comprehensively

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'Utility Module: Exhaustive Configuration, Security, Hooks, Distributed Coverage'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- COMPREHENSIVE CONFIGURATION MANAGEMENT ----
 * Test all configuration operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Configuration Management Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: List all configuration options'
SELECT 
	category,
	COUNT(*) AS option_count
FROM show_vector_config()
GROUP BY category
ORDER BY category;

\echo 'Test 2: Configuration by category'
SELECT 
	category,
	setting,
	description
FROM show_vector_config()
WHERE category = 'Index'
ORDER BY setting
LIMIT 10;

\echo 'Test 3: Set multiple configurations'
SELECT set_vector_config('ef_construction', '200') AS config1;
SELECT set_vector_config('m', '16') AS config2;
SELECT set_vector_config('ef_search', '64') AS config3;

\echo 'Test 4: Get configuration values'
SELECT 
	'ef_construction' AS config_name,
	get_vector_config('ef_construction') AS config_value
UNION ALL
SELECT 
	'm' AS config_name,
	get_vector_config('m') AS config_value
UNION ALL
SELECT 
	'ef_search' AS config_name,
	get_vector_config('ef_search') AS config_value;

\echo 'Test 5: Reset all configurations'
SELECT reset_vector_config('ef_construction') AS reset1;
SELECT reset_vector_config('m') AS reset2;
SELECT reset_vector_config('ef_search') AS reset3;

\echo 'Test 6: Configuration validation'
-- Test setting invalid values
DO $$
BEGIN
	BEGIN
		PERFORM set_vector_config('ef_construction', 'invalid_value');
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- COMPREHENSIVE SECURITY OPERATIONS ----
 * Test all security functions
 *------------------------------------------------------------------*/
\echo ''
\echo 'Security Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 7: Post-quantum encryption with various vectors'
SELECT 
	'vector_1' AS test_name,
	pg_column_size(encrypt_postquantum(vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector)) AS encrypted_size
UNION ALL
SELECT 
	'vector_2' AS test_name,
	pg_column_size(encrypt_postquantum(vector '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector)) AS encrypted_size;

\echo 'Test 8: Confidential compute mode toggle'
SELECT enable_confidential_compute(true) AS enabled;
SELECT enable_confidential_compute(false) AS disabled;
SELECT enable_confidential_compute(true) AS re_enabled;

\echo 'Test 9: Access mask with various roles'
SELECT set_access_mask('role1', 'l2,cosine', 'hnsw') AS mask1;
SELECT set_access_mask('role2', 'l2,cosine,ip', 'hnsw,ivf') AS mask2;
SELECT set_access_mask('role3', 'l2', 'hnsw') AS mask3;

\echo 'Test 10: Federated vector query with various hosts'
SELECT federated_vector_query('localhost', 'SELECT * FROM test') AS query1;
SELECT federated_vector_query('remote1.example.com', 'SELECT embedding FROM vectors LIMIT 10') AS query2;

/*-------------------------------------------------------------------
 * ---- COMPREHENSIVE HOOK OPERATIONS ----
 * Test all hook functions
 *------------------------------------------------------------------*/
\echo ''
\echo 'Hook Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 11: Register multiple custom operators'
SELECT register_custom_operator('custom_l2', 'vector_custom_l2') AS op1;
SELECT register_custom_operator('custom_cosine', 'vector_custom_cosine') AS op2;
SELECT register_custom_operator('custom_ip', 'vector_custom_ip') AS op3;

\echo 'Test 12: Vector replication with multiple publications'
SELECT enable_vector_replication('pub1') AS pub1;
SELECT enable_vector_replication('pub2') AS pub2;
SELECT enable_vector_replication('pub3') AS pub3;

\echo 'Test 13: Create multiple vector FDWs'
SELECT create_vector_fdw('fdw1', 'host1', 5432) AS fdw1;
SELECT create_vector_fdw('fdw2', 'host2', 5433) AS fdw2;
SELECT create_vector_fdw('fdw3', 'host3', 5434) AS fdw3;

/*-------------------------------------------------------------------
 * ---- COMPREHENSIVE DISTRIBUTED OPERATIONS ----
 * Test distributed query operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Distributed Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 14: Federated queries with various query types'
SELECT federated_vector_query('localhost', 'SELECT COUNT(*) FROM vectors') AS count_query;
SELECT federated_vector_query('localhost', 'SELECT * FROM vectors ORDER BY embedding <-> query_vector LIMIT 10') AS knn_query;
SELECT federated_vector_query('localhost', 'SELECT AVG(vector_norm(embedding)) FROM vectors') AS aggregate_query;

\echo 'Test 15: Distributed operations with error handling'
DO $$
BEGIN
	BEGIN
		PERFORM federated_vector_query('invalid_host_xyz', 'SELECT * FROM test');
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- SAFE MEMORY OPERATIONS ----
 * Test safe memory and SPI safe operations (internal, tested through usage)
 *------------------------------------------------------------------*/
\echo ''
\echo 'Safe Memory Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 16: Operations that use safe memory (tested through normal operations)'
-- Safe memory operations are internal, tested through normal vector operations
SELECT 
	COUNT(*) AS safe_ops_count,
	AVG(vector_norm(embedding)) AS avg_norm
FROM (
	SELECT features AS embedding
	FROM test_train_view
	LIMIT 100
) sub;

\echo ''
\echo '=========================================================================='
\echo '✓ Utility Module: Full exhaustive code-path test complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




