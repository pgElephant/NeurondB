-- NeurondB Comprehensive Benchmark Suite
-- Tests performance across different dimensions, distances, and operations

\timing on

-- Setup
CREATE EXTENSION IF NOT EXISTS neurondb;

-- Benchmark configuration
\set dim_small 64
\set dim_medium 384
\set dim_large 1536
\set rows_small 1000
\set rows_medium 10000
\set rows_large 100000

\echo '==================================================='
\echo 'NeurondB Benchmark Suite'
\echo '==================================================='
\echo ''

-- ===================================================================
-- Part 1: Vector Creation and I/O Performance
-- ===================================================================

\echo '--- Part 1: Vector Creation and I/O ---'
\echo ''

CREATE TEMP TABLE bench_vectors_64 (
    id serial PRIMARY KEY,
    vec vector(64)
);

\echo 'Inserting 10K vectors (dim=64)...'
INSERT INTO bench_vectors_64 (vec)
SELECT ('[' || string_agg((random() * 2 - 1)::text, ',') || ']')::vector
FROM generate_series(1, 64) dim
CROSS JOIN generate_series(1, 10000) row
GROUP BY row;

\echo 'Vector serialization (output)...'
EXPLAIN ANALYZE
SELECT vec::text FROM bench_vectors_64 LIMIT 1000;

\echo 'Vector deserialization (input)...'
EXPLAIN ANALYZE
SELECT vec::text::vector FROM bench_vectors_64 LIMIT 1000;

-- ===================================================================
-- Part 2: Distance Metric Performance
-- ===================================================================

\echo ''
\echo '--- Part 2: Distance Metrics (Sequential Scan) ---'
\echo ''

WITH query AS (
    SELECT vec as query_vec FROM bench_vectors_64 LIMIT 1
)
\echo 'L2 Distance (10K comparisons)...'
EXPLAIN ANALYZE
SELECT COUNT(*) 
FROM bench_vectors_64 b, query q
WHERE b.vec <-> q.query_vec < 10;

WITH query AS (
    SELECT vec as query_vec FROM bench_vectors_64 LIMIT 1
)
\echo 'Cosine Distance (10K comparisons)...'
EXPLAIN ANALYZE
SELECT COUNT(*) 
FROM bench_vectors_64 b, query q
WHERE b.vec <=> q.query_vec < 1;

WITH query AS (
    SELECT vec as query_vec FROM bench_vectors_64 LIMIT 1
)
\echo 'Inner Product (10K comparisons)...'
EXPLAIN ANALYZE
SELECT COUNT(*) 
FROM bench_vectors_64 b, query q
WHERE b.vec <#> q.query_vec > -10;

-- ===================================================================
-- Part 3: Vector Operations Performance
-- ===================================================================

\echo ''
\echo '--- Part 3: Vector Operations ---'
\echo ''

\echo 'Vector Addition (10K ops)...'
EXPLAIN ANALYZE
SELECT vec + '[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]'::vector
FROM bench_vectors_64;

\echo 'Vector Normalization (10K ops)...'
EXPLAIN ANALYZE
SELECT vector_normalize(vec) FROM bench_vectors_64;

\echo 'Vector Norm (10K ops)...'
EXPLAIN ANALYZE
SELECT vector_norm(vec) FROM bench_vectors_64;

-- ===================================================================
-- Part 4: Quantization Performance
-- ===================================================================

\echo ''
\echo '--- Part 4: Quantization ---'
\echo ''

\echo 'INT8 Quantization (10K vectors)...'
EXPLAIN ANALYZE
SELECT vector_to_int8(vec) FROM bench_vectors_64;

\echo 'Binary Quantization (10K vectors)...'
EXPLAIN ANALYZE
SELECT vector_to_binary(vec) FROM bench_vectors_64;

-- ===================================================================
-- Part 5: Aggregate Functions
-- ===================================================================

\echo ''
\echo '--- Part 5: Aggregate Functions ---'
\echo ''

\echo 'Vector AVG (10K vectors)...'
EXPLAIN ANALYZE
SELECT vector_avg(vec) FROM bench_vectors_64;

\echo 'Vector SUM (10K vectors)...'
EXPLAIN ANALYZE
SELECT vector_sum(vec) FROM bench_vectors_64;

-- ===================================================================
-- Part 6: Dimension Scaling Test
-- ===================================================================

\echo ''
\echo '--- Part 6: Dimension Scaling ---'
\echo ''

CREATE TEMP TABLE bench_vectors_384 (
    id serial PRIMARY KEY,
    vec vector(384)
);

CREATE TEMP TABLE bench_vectors_1536 (
    id serial PRIMARY KEY,
    vec vector(1536)
);

\echo 'Inserting 1K vectors (dim=384)...'
INSERT INTO bench_vectors_384 (vec)
SELECT ('[' || string_agg((random() * 2 - 1)::text, ',') || ']')::vector
FROM generate_series(1, 384) dim
CROSS JOIN generate_series(1, 1000) row
GROUP BY row;

\echo 'Inserting 1K vectors (dim=1536)...'
INSERT INTO bench_vectors_1536 (vec)
SELECT ('[' || string_agg((random() * 2 - 1)::text, ',') || ']')::vector
FROM generate_series(1, 1536) dim
CROSS JOIN generate_series(1, 1000) row
GROUP BY row;

WITH query AS (SELECT vec as qv FROM bench_vectors_384 LIMIT 1)
\echo 'L2 Distance (dim=384, 1K comparisons)...'
EXPLAIN ANALYZE
SELECT COUNT(*) FROM bench_vectors_384 b, query q WHERE b.vec <-> q.qv < 10;

WITH query AS (SELECT vec as qv FROM bench_vectors_1536 LIMIT 1)
\echo 'L2 Distance (dim=1536, 1K comparisons)...'
EXPLAIN ANALYZE
SELECT COUNT(*) FROM bench_vectors_1536 b, query q WHERE b.vec <-> q.qv < 20;

-- ===================================================================
-- Part 7: Top-K Query Performance
-- ===================================================================

\echo ''
\echo '--- Part 7: Top-K Queries ---'
\echo ''

WITH query AS (SELECT vec as qv FROM bench_vectors_64 LIMIT 1)
\echo 'Top-10 Nearest Neighbors (L2, 10K vectors)...'
EXPLAIN ANALYZE
SELECT id, vec <-> qv as distance
FROM bench_vectors_64, query
ORDER BY distance
LIMIT 10;

WITH query AS (SELECT vec as qv FROM bench_vectors_64 LIMIT 1)
\echo 'Top-100 Nearest Neighbors (L2, 10K vectors)...'
EXPLAIN ANALYZE
SELECT id, vec <-> qv as distance
FROM bench_vectors_64, query
ORDER BY distance
LIMIT 100;

-- ===================================================================
-- Summary
-- ===================================================================

\echo ''
\echo '==================================================='
\echo 'Benchmark Summary'
\echo '==================================================='
\echo ''

SELECT 
    'bench_vectors_64' as table_name,
    COUNT(*) as row_count,
    vector_dims((SELECT vec FROM bench_vectors_64 LIMIT 1)) as dimensions,
    pg_size_pretty(pg_total_relation_size('bench_vectors_64')) as total_size
FROM bench_vectors_64

UNION ALL

SELECT 
    'bench_vectors_384',
    COUNT(*),
    vector_dims((SELECT vec FROM bench_vectors_384 LIMIT 1)),
    pg_size_pretty(pg_total_relation_size('bench_vectors_384'))
FROM bench_vectors_384

UNION ALL

SELECT 
    'bench_vectors_1536',
    COUNT(*),
    vector_dims((SELECT vec FROM bench_vectors_1536 LIMIT 1)),
    pg_size_pretty(pg_total_relation_size('bench_vectors_1536'))
FROM bench_vectors_1536;

\echo ''
\echo 'Benchmark Complete!'
\echo ''

-- Cleanup
DROP TABLE bench_vectors_64;
DROP TABLE bench_vectors_384;
DROP TABLE bench_vectors_1536;

\timing off

