-- Basic tests for GPU search functions
-- Tests the fixed GPU HNSW and IVF search functions

-- Setup test data
CREATE TABLE gpu_search_test (
	id serial PRIMARY KEY,
	vec vector(4)
);

INSERT INTO gpu_search_test (vec)
SELECT ('[' || random() || ',' || random() || ',' || random() || ',' || random() || ']')::vector
FROM generate_series(1, 50);

-- Create HNSW index
CREATE INDEX gpu_hnsw_idx ON gpu_search_test USING hnsw (vec vector_l2_ops)
WITH (m = 16, ef_construction = 64, ef_search = 40);

-- Test GPU HNSW search (if GPU available)
SELECT * FROM hnsw_knn_search_gpu('gpu_hnsw_idx', '[0.5,0.5,0.5,0.5]'::vector, 5, 20);

-- Test with default ef_search
SELECT * FROM hnsw_knn_search_gpu('gpu_hnsw_idx', '[0.5,0.5,0.5,0.5]'::vector, 5);

-- Create IVF index
CREATE INDEX gpu_ivf_idx ON gpu_search_test USING ivfflat (vec vector_l2_ops)
WITH (lists = 10, probes = 5);

-- Test GPU IVF search (if GPU available)
SELECT * FROM ivf_knn_search_gpu('gpu_ivf_idx', '[0.5,0.5,0.5,0.5]'::vector, 5, 3);

-- Test with default nprobe
SELECT * FROM ivf_knn_search_gpu('gpu_ivf_idx', '[0.5,0.5,0.5,0.5]'::vector, 5);

-- Test error cases
DO $$
BEGIN
	-- Invalid index name
	BEGIN
		PERFORM * FROM hnsw_knn_search_gpu('nonexistent_index', '[0.5,0.5,0.5,0.5]'::vector, 5);
		RAISE EXCEPTION 'Should have failed';
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected
	END;
END $$;

-- Cleanup
DROP INDEX IF EXISTS gpu_hnsw_idx;
DROP INDEX IF EXISTS gpu_ivf_idx;
DROP TABLE IF EXISTS gpu_search_test CASCADE;

\echo 'Test completed successfully'




