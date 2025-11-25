-- Basic tests for OPQ rotation and PQ quantization fixes

-- Setup test data
CREATE TABLE opq_pq_test (
	id serial PRIMARY KEY,
	vec vector(8)
);

INSERT INTO opq_pq_test (vec)
SELECT ('[' || 
	random() || ',' || random() || ',' || random() || ',' || random() || ',' ||
	random() || ',' || random() || ',' || random() || ',' || random() || ']')::vector
FROM generate_series(1, 100);

-- Test OPQ rotation training
-- Returns float8[] rotation matrix
SELECT array_length(train_opq_rotation('opq_pq_test', 'vec', 4), 1) AS rotation_matrix_size;

-- Test PQ codebook training
-- Returns bytea codebook
SELECT octet_length(train_pq_codebook('opq_pq_test', 'vec', 4, 256)) AS codebook_size;

-- Test with different parameters
SELECT octet_length(train_pq_codebook('opq_pq_test', 'vec', 2, 128)) AS codebook_size_2sub;

-- Verify functions handle edge cases
DO $$
DECLARE
	rotation_matrix float8[];
	codebook bytea;
BEGIN
	-- Test OPQ rotation
	SELECT train_opq_rotation('opq_pq_test', 'vec', 4) INTO rotation_matrix;
	IF rotation_matrix IS NULL THEN
		RAISE EXCEPTION 'train_opq_rotation returned NULL';
	END IF;
	
	-- Test PQ codebook
	SELECT train_pq_codebook('opq_pq_test', 'vec', 4, 256) INTO codebook;
	IF codebook IS NULL THEN
		RAISE EXCEPTION 'train_pq_codebook returned NULL';
	END IF;
	
	RAISE NOTICE 'OPQ and PQ training completed successfully';
END $$;

-- Cleanup
DROP TABLE IF EXISTS opq_pq_test CASCADE;

\echo 'Test completed successfully'
