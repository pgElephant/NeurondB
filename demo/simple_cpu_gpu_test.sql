\set ON_ERROR_STOP on
\set QUIET on

-- ============================================================================
-- Simple CPU vs GPU Test - Apple M4 Metal
-- ============================================================================

DROP EXTENSION IF EXISTS neurondb CASCADE;
CREATE EXTENSION neurondb;

\echo '══════════════════════════════════════════════════════════════════'
\echo '         NeurondB - Simple CPU vs GPU Test'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

-- Create test data
\echo 'Step 1: Creating test dataset (1000 vectors, 128 dimensions)...'
CREATE TEMP TABLE test_vectors AS
SELECT 
    i as id,
    array_agg(random()::real ORDER BY j)::real[]::vector(128) as vec
FROM generate_series(1, 1000) i,
     generate_series(1, 128) j
GROUP BY i;

\echo '  ✓ Created 1000 vectors x 128 dimensions'
\echo ''

-- ============================================================================
-- TEST 1: CPU EXECUTION
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 1: CPU Execution (GPU disabled)'
\echo '══════════════════════════════════════════════════════════════════'

SET neurondb.gpu_enabled = false;

\echo ''
\echo 'CPU Test: L2 Distance (100 x 100 = 10,000 comparisons)'
\timing on
SELECT 
    COUNT(*) as total_comparisons,
    MIN(v1.vec <-> v2.vec)::numeric(8,4) as min_distance,
    MAX(v1.vec <-> v2.vec)::numeric(8,4) as max_distance,
    AVG(v1.vec <-> v2.vec)::numeric(8,4) as avg_distance
FROM test_vectors v1
CROSS JOIN test_vectors v2
WHERE v1.id <= 100 AND v2.id <= 100 AND v1.id < v2.id;
\timing off

\echo ''
\echo 'CPU Test: Sample distance calculations'
SELECT 
    v1.id as vec1_id,
    v2.id as vec2_id,
    (v1.vec <-> v2.vec)::numeric(8,4) as cpu_distance
FROM test_vectors v1
CROSS JOIN test_vectors v2
WHERE v1.id <= 5 AND v2.id <= 5 AND v1.id < v2.id
ORDER BY v1.id, v2.id;
\echo ''

-- ============================================================================
-- TEST 2: GPU EXECUTION
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 2: GPU Execution (Metal enabled on Apple M4)'
\echo '══════════════════════════════════════════════════════════════════'

SET neurondb.gpu_enabled = true;
SET neurondb.gpu_backend = 'metal';

\echo ''
\echo 'GPU Test: L2 Distance (100 x 100 = 10,000 comparisons)'
\timing on
SELECT 
    COUNT(*) as total_comparisons,
    MIN(vector_l2_distance_gpu(v1.vec, v2.vec))::numeric(8,4) as min_distance,
    MAX(vector_l2_distance_gpu(v1.vec, v2.vec))::numeric(8,4) as max_distance,
    AVG(vector_l2_distance_gpu(v1.vec, v2.vec))::numeric(8,4) as avg_distance
FROM test_vectors v1
CROSS JOIN test_vectors v2
WHERE v1.id <= 100 AND v2.id <= 100 AND v1.id < v2.id;
\timing off

\echo ''
\echo 'GPU Test: Sample distance calculations'
SELECT 
    v1.id as vec1_id,
    v2.id as vec2_id,
    vector_l2_distance_gpu(v1.vec, v2.vec)::numeric(8,4) as gpu_distance
FROM test_vectors v1
CROSS JOIN test_vectors v2
WHERE v1.id <= 5 AND v2.id <= 5 AND v1.id < v2.id
ORDER BY v1.id, v2.id;
\echo ''

-- ============================================================================
-- TEST 3: SIDE-BY-SIDE COMPARISON
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 3: CPU vs GPU Side-by-Side Comparison'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

SET neurondb.gpu_enabled = true;

\echo 'Comparing CPU and GPU results (should match):'
SELECT 
    v1.id as vec1,
    v2.id as vec2,
    (v1.vec <-> v2.vec)::numeric(8,4) as cpu_result,
    vector_l2_distance_gpu(v1.vec, v2.vec)::numeric(8,4) as gpu_result,
    CASE 
        WHEN ABS((v1.vec <-> v2.vec) - vector_l2_distance_gpu(v1.vec, v2.vec)) < 0.0001 
        THEN '✓ MATCH' 
        ELSE '✗ DIFF'
    END as verification
FROM test_vectors v1
CROSS JOIN test_vectors v2
WHERE v1.id <= 3 AND v2.id <= 3 AND v1.id < v2.id
ORDER BY v1.id, v2.id;

\echo ''

-- ============================================================================
-- CONFIGURATION SUMMARY
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' GPU Configuration Summary'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

SELECT 
    name,
    setting,
    CASE 
        WHEN name = 'neurondb.gpu_enabled' AND setting = 'on' THEN '✓'
        WHEN name = 'neurondb.gpu_backend' AND setting = 'metal' THEN '✓'
        ELSE ' '
    END as status
FROM pg_settings
WHERE name LIKE 'neurondb.gpu%'
ORDER BY name;

\echo ''
\echo '══════════════════════════════════════════════════════════════════'
\echo '                    Test Complete!'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''
\echo 'Summary:'
\echo '  • CPU execution: Using standard <-> operator'
\echo '  • GPU execution: Using vector_l2_distance_gpu() with Metal'
\echo '  • Results: Should match within rounding tolerance'
\echo '  • Platform: Apple M4 with Metal GPU support'
\echo ''
\echo 'Note: GPU may use CPU fallback for vectors < 64 dimensions'
\echo '      This is expected and ensures optimal performance.'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

