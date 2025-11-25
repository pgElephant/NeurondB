\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo '=========================================================================='

-- Test 1: GPU Enable and Availability
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb_gpu_enable() AS gpu_enabled;
-- GPU availability is shown in neurondb_gpu_info() below

-- Test 2: GPU Information
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT * FROM neurondb_gpu_info();

-- Test 3: GPU Statistics
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT * FROM neurondb_gpu_stats();

-- Test 4: LLM GPU Information
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb_llm_gpu_available() AS llm_gpu_available;
SELECT * FROM neurondb_llm_gpu_info();

-- Test 5: GPU Distance Functions (if available)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- GPU already enabled via test_settings above

SELECT 
	vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector) AS gpu_l2_distance,
	vector_cosine_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector) AS gpu_cosine_distance;

\echo ''
\echo '=========================================================================='

\echo 'Test completed successfully'
