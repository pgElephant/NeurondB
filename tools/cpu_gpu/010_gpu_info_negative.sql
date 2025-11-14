\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'GPU Information - Negative Test Cases (Error Handling)'
\echo '=========================================================================='

\echo ''
\echo 'Test 1: Invalid GPU Kernel Configuration'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET neurondb.gpu_kernels = 'invalid_kernel_name';
SELECT neurondb_gpu_enable();

\echo ''
\echo 'Test 2: Empty GPU Kernel Configuration'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET neurondb.gpu_kernels = '';
SELECT neurondb_gpu_enable();

\echo ''
\echo 'Test 3: GPU Distance Functions with NULL Vectors'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_l2_distance_gpu(NULL::vector, '[1,2,3]'::vector);
SELECT vector_l2_distance_gpu('[1,2,3]'::vector, NULL::vector);
SELECT vector_cosine_distance_gpu(NULL::vector, '[1,2,3]'::vector);
SELECT vector_cosine_distance_gpu('[1,2,3]'::vector, NULL::vector);

\echo ''
\echo 'Test 4: GPU Distance Functions with Dimension Mismatch'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5]'::vector);
SELECT vector_cosine_distance_gpu('[1,2]'::vector, '[4,5,6]'::vector);

\echo ''
\echo 'Test 5: GPU Operations When GPU is Disabled'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET neurondb.gpu_enabled = off;
SELECT neurondb_gpu_enable();
SELECT vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);

\echo ''
\echo 'Test 6: GPU Stats Reset with No Stats'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb_gpu_reset_stats();
SELECT * FROM neurondb_gpu_stats();

\echo ''
\echo 'Test 7: GPU Info Query When No GPU Available'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- This should still work, just return empty or indicate no GPU
SELECT * FROM neurondb_gpu_info();

\echo ''
\echo 'Test 8: Invalid GPU Enable State'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Try enabling with invalid configuration
SET neurondb.gpu_enabled = NULL;
SELECT neurondb_gpu_enable();

\echo ''
\echo 'Negative GPU Information Test Complete!'
\echo '=========================================================================='

