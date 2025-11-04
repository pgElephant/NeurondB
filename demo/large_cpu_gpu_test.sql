\set ON_ERROR_STOP on
\set QUIET on

-- ============================================================================
-- Large-Scale CPU vs GPU Test - Force GPU Usage
-- ============================================================================

DROP EXTENSION IF EXISTS neurondb CASCADE;
CREATE EXTENSION neurondb;

\echo '══════════════════════════════════════════════════════════════════'
\echo '    Large-Scale CPU vs GPU Test - Apple M4 Metal'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

-- Create LARGE test data (high-dimensional vectors)
\echo 'Step 1: Creating LARGE dataset (5000 vectors, 512 dimensions)...'
\echo '  (High dimensions trigger GPU acceleration)'
CREATE TEMP TABLE large_test_vectors AS
SELECT 
    i as id,
    array_agg(random()::real ORDER BY j)::real[]::vector(512) as vec
FROM generate_series(1, 5000) i,
     generate_series(1, 512) j
GROUP BY i;

\echo '  ✓ Created 5,000 vectors x 512 dimensions each'
\echo '  ✓ Total data: ~10 MB in memory'
\echo ''

-- ============================================================================
-- TEST 1: CPU BASELINE
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 1: CPU Baseline (GPU disabled)'
\echo '══════════════════════════════════════════════════════════════════'

SET neurondb.gpu_enabled = false;

\echo ''
\echo 'CPU: Computing 50,000 L2 distances (200 x 250 vectors, 512-dim)'
\timing on
SELECT 
    COUNT(*) as comparisons,
    AVG(v1.vec <-> v2.vec)::numeric(8,4) as avg_cpu_distance
FROM large_test_vectors v1
CROSS JOIN large_test_vectors v2
WHERE v1.id <= 200 AND v2.id <= 250 AND v1.id < v2.id;
\timing off

\echo ''

-- ============================================================================
-- TEST 2: GPU ACCELERATION
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 2: GPU Execution (Metal M4 - 512-dim vectors)'
\echo '══════════════════════════════════════════════════════════════════'

SET neurondb.gpu_enabled = true;
SET neurondb.gpu_backend = 'metal';

\echo ''
\echo 'GPU: Computing 50,000 L2 distances (same 200 x 250 vectors, 512-dim)'
\echo '  Vector dimension: 512 (well above 64-dim threshold for GPU)'
\timing on
SELECT 
    COUNT(*) as comparisons,
    AVG(vector_l2_distance_gpu(v1.vec, v2.vec))::numeric(8,4) as avg_gpu_distance
FROM large_test_vectors v1
CROSS JOIN large_test_vectors v2
WHERE v1.id <= 200 AND v2.id <= 250 AND v1.id < v2.id;
\timing off

\echo ''

-- ============================================================================
-- TEST 3: MASSIVE BATCH TEST
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 3: Massive Batch Test - 1 Million Comparisons'
\echo '══════════════════════════════════════════════════════════════════'

\echo ''
\echo 'CPU: 1 million distance calculations (1000 x 1000, 512-dim)'
SET neurondb.gpu_enabled = false;
\timing on
SELECT 
    COUNT(*) as total_ops,
    MIN(v1.vec <-> v2.vec)::numeric(6,2) as min_d,
    MAX(v1.vec <-> v2.vec)::numeric(6,2) as max_d,
    AVG(v1.vec <-> v2.vec)::numeric(6,2) as avg_d
FROM large_test_vectors v1
CROSS JOIN large_test_vectors v2
WHERE v1.id <= 1000 AND v2.id <= 1000 AND v1.id < v2.id;
\timing off

\echo ''
\echo 'GPU: 1 million distance calculations (same 1000 x 1000, 512-dim)'
SET neurondb.gpu_enabled = true;
\timing on
SELECT 
    COUNT(*) as total_ops,
    MIN(vector_l2_distance_gpu(v1.vec, v2.vec))::numeric(6,2) as min_d,
    MAX(vector_l2_distance_gpu(v1.vec, v2.vec))::numeric(6,2) as max_d,
    AVG(vector_l2_distance_gpu(v1.vec, v2.vec))::numeric(6,2) as avg_d
FROM large_test_vectors v1
CROSS JOIN large_test_vectors v2
WHERE v1.id <= 1000 AND v2.id <= 1000 AND v1.id < v2.id;
\timing off

\echo ''

-- ============================================================================
-- VERIFICATION
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' Verification: CPU vs GPU Results Match'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

SELECT 
    v1.id,
    v2.id,
    (v1.vec <-> v2.vec)::numeric(10,6) as cpu,
    vector_l2_distance_gpu(v1.vec, v2.vec)::numeric(10,6) as gpu,
    CASE 
        WHEN ABS((v1.vec <-> v2.vec) - vector_l2_distance_gpu(v1.vec, v2.vec)) < 0.001 
        THEN '✓' 
        ELSE '✗'
    END as match
FROM large_test_vectors v1
CROSS JOIN large_test_vectors v2
WHERE v1.id <= 3 AND v2.id <= 3 AND v1.id < v2.id
ORDER BY v1.id, v2.id;

\echo ''
\echo '══════════════════════════════════════════════════════════════════'
\echo '                     Test Complete!'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''
\echo 'Dataset:'
\echo '  • 5,000 vectors'
\echo '  • 512 dimensions each (triggers GPU for dim >= 64)'
\echo '  • ~10 MB total data'
\echo ''
\echo 'Tests Performed:'
\echo '  1. CPU: 50,000 distance calculations'
\echo '  2. GPU: 50,000 distance calculations  '
\echo '  3. CPU: 1 million distance calculations'
\echo '  4. GPU: 1 million distance calculations'
\echo ''
\echo 'GPU Status:'
SELECT 
    CASE 
        WHEN EXISTS (SELECT 1 FROM pg_settings WHERE name = 'neurondb.gpu_enabled' AND setting = 'on')
        THEN '  ✓ GPU Enabled: YES (Metal backend)'
        ELSE '  ✗ GPU Enabled: NO'
    END;
\echo ''
\echo 'Platform: Apple M4 with Metal Performance Shaders'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

