-- Verify Metal GPU is actually being used
DROP EXTENSION IF EXISTS neurondb CASCADE;
CREATE EXTENSION neurondb;

SET neurondb.gpu_enabled = true;
SET neurondb.gpu_backend = 'metal';
SET client_min_messages = 'DEBUG1';

\echo '══════════════════════════════════════════════════════════════════'
\echo ' Metal GPU Usage Verification Test'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

-- Create high-dimensional vectors (forces GPU)
CREATE TEMP TABLE gpu_vectors AS
SELECT 
    i as id,
    array_agg(random()::real ORDER BY j)::real[]::vector(512) as vec
FROM generate_series(1, 100) i,
     generate_series(1, 512) j
GROUP BY i;

\echo ''
\echo 'Test 1: Single GPU distance call (512 dimensions)'
\echo '  This should trigger Metal initialization if working'
\timing on
SELECT vector_l2_distance_gpu(
    (SELECT vec FROM gpu_vectors WHERE id = 1),
    (SELECT vec FROM gpu_vectors WHERE id = 2)
)::numeric(10,6) as distance;
\timing off

\echo ''
\echo 'Test 2: Multiple GPU distance calculations'
\timing on
SELECT 
    COUNT(*) as total,
    AVG(vector_l2_distance_gpu(v1.vec, v2.vec))::numeric(8,4) as avg_dist
FROM gpu_vectors v1
CROSS JOIN gpu_vectors v2
WHERE v1.id <= 10 AND v2.id <= 10 AND v1.id < v2.id;
\timing off

\echo ''
\echo 'Test 3: Check for Metal-specific messages in output above'
\echo '  Look for: "Attempting Metal GPU initialization"'
\echo '  Look for: "Metal GPU backend active"'
\echo '══════════════════════════════════════════════════════════════════'
