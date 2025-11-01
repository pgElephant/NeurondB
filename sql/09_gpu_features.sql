-- Test GPU Features
-- This test suite validates GPU acceleration functionality
-- Tests will gracefully skip if GPU is not available

-- Extension is created in 01_types_basic
SET neurondb.gpu_enabled = off;  -- enforce CPU fallback for deterministic results

-- Test 1: GPU info function
-- Skip info/stats in regression (environment-specific)
SELECT 'gpu_info_skipped' AS note;

-- Test 2: Try to enable GPU (will fallback to CPU if no GPU)
SELECT 'gpu_enable_skipped' AS note;

-- Test 3: GPU statistics
SELECT 'gpu_stats_skipped' AS note;

-- Test 4: Create test data for GPU distance operations
CREATE TABLE gpu_test_vectors (
    id SERIAL PRIMARY KEY,
    vec vector(4)
);

-- Insert test vectors
INSERT INTO gpu_test_vectors (vec)
SELECT ('[' || array_to_string(ARRAY(
    SELECT (i)::float4 FROM generate_series(1, 4) AS g(i)
), ',') || ']')::vector
FROM generate_series(1, 10);

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
-- ANN GPU entry points intentionally skipped in regression (planner-dependent)

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

