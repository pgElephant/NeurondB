-- Most detailed Metal GPU test possible
SET client_min_messages = 'DEBUG1';

DROP EXTENSION IF EXISTS neurondb CASCADE;
CREATE EXTENSION neurondb;

\echo 'Enabling GPU with detailed logging...'
SET neurondb.gpu_enabled = true;
SET neurondb.gpu_backend = 'metal';

\echo ''
\echo 'Creating 512-dimensional vectors...'
CREATE TEMP TABLE gpu_test AS
SELECT 
    array_agg(random()::real ORDER BY i)::real[]::vector(512) as vec
FROM generate_series(1, 512) i
LIMIT 10;

\echo ''
\echo '══════════════════════════════════════════════════════════════════'
\echo ' TRIGGERING METAL GPU INITIALIZATION'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''
\echo 'Calling GPU function (should show debug messages above)...'

SELECT 
    vector_l2_distance_gpu(
        (SELECT vec FROM gpu_test LIMIT 1 OFFSET 0),
        (SELECT vec FROM gpu_test LIMIT 1 OFFSET 1)
    ) as result;

\echo ''
\echo 'Check DEBUG messages above for:'
\echo '  - "Attempting Metal GPU initialization"'
\echo '  - "Metal GPU backend active"'
\echo '  - "Metal backend initialized"'
\echo ''
