-- Test GPU Features
-- This test suite validates GPU acceleration functionality
-- Tests will gracefully skip if GPU is not available

-- Create extension
CREATE EXTENSION IF NOT EXISTS neurondb;

-- Test 1: GPU info function
SELECT neurondb_gpu_info();

-- Test 2: Try to enable GPU (will fallback to CPU if no GPU)
SELECT neurondb_gpu_enable(true);

-- Test 3: GPU statistics
SELECT * FROM neurondb_gpu_stats();

-- Test 4: Create test data for GPU distance operations
CREATE TABLE gpu_test_vectors (
    id SERIAL PRIMARY KEY,
    vec vector(128)
);

-- Insert test vectors
INSERT INTO gpu_test_vectors (vec)
SELECT ('[' || array_to_string(ARRAY(
    SELECT (random() * 2 - 1)::float4
    FROM generate_series(1, 128)
), ',') || ']')::vector
FROM generate_series(1, 100);

-- Test 5: GPU L2 distance (will use CPU if GPU unavailable)
SELECT id, vector_l2_distance_gpu(vec, (SELECT vec FROM gpu_test_vectors WHERE id = 1))
FROM gpu_test_vectors
ORDER BY 2
LIMIT 5;

-- Test 6: GPU cosine distance
SELECT id, vector_cosine_distance_gpu(vec, (SELECT vec FROM gpu_test_vectors WHERE id = 1))
FROM gpu_test_vectors
ORDER BY 2
LIMIT 5;

-- Test 7: GPU inner product
SELECT id, vector_inner_product_gpu(vec, (SELECT vec FROM gpu_test_vectors WHERE id = 1))
FROM gpu_test_vectors
ORDER BY 2
LIMIT 5;

-- Test 8: Compare GPU vs CPU distance (should be nearly identical)
SELECT 
    id,
    ABS(
        vector_l2_distance_gpu(vec, (SELECT vec FROM gpu_test_vectors WHERE id = 1)) -
        vector_l2_distance(vec, (SELECT vec FROM gpu_test_vectors WHERE id = 1))
    ) < 0.001 AS distances_match
FROM gpu_test_vectors
WHERE id <= 10;

-- Test 9: GPU quantization INT8
SELECT vector_to_int8_gpu(vec) FROM gpu_test_vectors LIMIT 1;

-- Test 10: GPU quantization FP16
SELECT vector_to_fp16_gpu(vec) FROM gpu_test_vectors LIMIT 1;

-- Test 11: GPU quantization binary
SELECT vector_to_binary_gpu(vec) FROM gpu_test_vectors LIMIT 1;

-- Test 12: GPU HNSW search (will use CPU path if GPU unavailable)
CREATE INDEX gpu_test_hnsw_idx ON gpu_test_vectors USING hnsw (vec vector_l2_ops);

SELECT id FROM hnsw_knn_search_gpu(
    (SELECT vec FROM gpu_test_vectors WHERE id = 1),
    5,
    100
);

-- Test 13: GPU IVF search
SELECT id FROM ivf_knn_search_gpu(
    (SELECT vec FROM gpu_test_vectors WHERE id = 1),
    5,
    10
);

-- Test 14: GPU KMeans clustering (if GPU available)
SELECT cluster_kmeans_gpu('gpu_test_vectors', 'vec', 3, 10);

-- Test 15: GPU statistics after operations
SELECT * FROM neurondb_gpu_stats();

-- Test 16: Reset GPU statistics
SELECT neurondb_gpu_stats_reset();

-- Test 17: Verify stats reset
SELECT * FROM neurondb_gpu_stats();

-- Test 18: Disable GPU
SELECT neurondb_gpu_enable(false);

-- Test 19: GPU info after disable
SELECT neurondb_gpu_info();

-- Cleanup
DROP TABLE gpu_test_vectors CASCADE;

