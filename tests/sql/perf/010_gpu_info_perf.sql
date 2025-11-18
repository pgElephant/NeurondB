\timing on
\pset footer off
\pset pager off


-- Performance test: Works on the whole 11M row view
\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'GPU Information and Status Test'
\echo '=========================================================================='

-- Test 1: GPU Enable and Availability
\echo ''
\echo 'Test 1: GPU Enable and Availability'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET neurondb.gpu_enabled = on;
SELECT neurondb_gpu_enable() AS gpu_enabled;
-- GPU availability is shown in neurondb_gpu_info() below

-- Test 2: GPU Information
\echo ''
\echo 'Test 2: GPU Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT * FROM neurondb_gpu_info();

-- Test 3: GPU Statistics
\echo ''
\echo 'Test 3: GPU Statistics'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT * FROM neurondb_gpu_stats();

-- Test 4: LLM GPU Information
\echo ''
\echo 'Test 4: LLM GPU Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb_llm_gpu_available() AS llm_gpu_available;
SELECT * FROM neurondb_llm_gpu_info();

-- Test 5: GPU Distance Functions (if available)
\echo ''
\echo 'Test 5: GPU Distance Functions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET neurondb.gpu_kernels = 'l2,cosine,ip';
SELECT neurondb_gpu_enable();

SELECT 
	vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector) AS gpu_l2_distance,
	vector_cosine_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector) AS gpu_cosine_distance;

\echo ''
\echo 'GPU Information Test Complete!'
\echo '=========================================================================='

