\set ON_ERROR_STOP on
\set QUIET on

-- ============================================================================
-- Final CPU vs GPU Test - 5 Minute CPU Baseline
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS neurondb;

\echo '══════════════════════════════════════════════════════════════════'
\echo '  Final CPU vs GPU Performance Test - Apple M4 Metal'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

-- Create MASSIVE dataset for 5-minute CPU test
\echo 'Step 1: Creating MASSIVE dataset...'
\echo '  • 10,000 vectors'
\echo '  • 1024 dimensions each'
\echo '  • ~40 MB total data'
\echo ''

CREATE TEMP TABLE massive_vectors AS
SELECT 
    i as id,
    array_agg(random()::real ORDER BY j)::real[]::vector(1024) as vec
FROM generate_series(1, 10000) i,
     generate_series(1, 1024) j
GROUP BY i;

\echo '  ✓ Dataset ready: 10,000 vectors x 1024 dimensions'
\echo ''

-- ============================================================================
-- CPU BASELINE - Should take ~5 minutes
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 1: CPU Baseline (Target: ~5 minutes)'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

SET neurondb.gpu_enabled = false;

\echo 'CPU: Computing 5 MILLION distance calculations'
\echo '  (2,000 x 2,500 vectors, 1024 dimensions each)'
\echo '  Expected time: ~5 minutes on CPU'
\echo ''
\echo 'Starting CPU test...'

\timing on
SELECT 
    COUNT(*) as total_operations,
    AVG(v1.vec <-> v2.vec)::numeric(8,4) as avg_distance,
    MIN(v1.vec <-> v2.vec)::numeric(8,4) as min_distance,
    MAX(v1.vec <-> v2.vec)::numeric(8,4) as max_distance
FROM massive_vectors v1
CROSS JOIN massive_vectors v2
WHERE v1.id <= 2000 AND v2.id <= 2500 AND v1.id < v2.id;
\timing off

\echo ''
\echo 'CPU test complete! Time recorded above.'
\echo ''

-- ============================================================================
-- GPU ACCELERATION - Compare against CPU
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 2: GPU Execution (Metal M4 - Same workload)'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

SET neurondb.gpu_enabled = true;
SET neurondb.gpu_backend = 'metal';

\echo 'GPU: Computing SAME 5 MILLION distance calculations'
\echo '  (2,000 x 2,500 vectors, 1024 dimensions each)'
\echo '  Using vector_l2_distance_gpu() with Metal backend'
\echo ''
\echo 'Starting GPU test...'

\timing on
SELECT 
    COUNT(*) as total_operations,
    AVG(vector_l2_distance_gpu(v1.vec, v2.vec))::numeric(8,4) as avg_distance,
    MIN(vector_l2_distance_gpu(v1.vec, v2.vec))::numeric(8,4) as min_distance,
    MAX(vector_l2_distance_gpu(v1.vec, v2.vec))::numeric(8,4) as max_distance
FROM massive_vectors v1
CROSS JOIN massive_vectors v2
WHERE v1.id <= 2000 AND v2.id <= 2500 AND v1.id < v2.id;
\timing off

\echo ''
\echo 'GPU test complete! Time recorded above.'
\echo ''

-- ============================================================================
-- VERIFICATION
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' Verification: Results Match'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

SELECT 
    v1.id,
    v2.id,
    (v1.vec <-> v2.vec)::numeric(10,6) as cpu_result,
    vector_l2_distance_gpu(v1.vec, v2.vec)::numeric(10,6) as gpu_result,
    CASE 
        WHEN ABS((v1.vec <-> v2.vec) - vector_l2_distance_gpu(v1.vec, v2.vec)) < 0.001 
        THEN '✓ MATCH' 
        ELSE '✗ DIFF'
    END as verification
FROM massive_vectors v1
CROSS JOIN massive_vectors v2
WHERE v1.id <= 3 AND v2.id <= 3 AND v1.id < v2.id
ORDER BY v1.id, v2.id;

\echo ''
\echo '══════════════════════════════════════════════════════════════════'
\echo '                 Performance Summary'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''
\echo 'Dataset: 10,000 vectors x 1024 dimensions'
\echo 'Test: 5 Million L2 distance calculations'
\echo ''
\echo 'Compare the two times above:'
\echo '  • CPU Time: [see TEST 1 above]'
\echo '  • GPU Time: [see TEST 2 above]'
\echo ''
\echo 'GPU Configuration:'
SELECT name, setting FROM pg_settings 
WHERE name LIKE 'neurondb.gpu%' AND name IN ('neurondb.gpu_enabled', 'neurondb.gpu_backend');
\echo ''
\echo 'Shared Preload:'
SELECT setting FROM pg_settings WHERE name = 'shared_preload_libraries';
\echo ''
\echo '══════════════════════════════════════════════════════════════════'

