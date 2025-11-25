-- Basic tests for IVF index access method
-- Tests the fixed IVF scan implementation

-- Setup test data
CREATE TABLE ivf_test_vectors (
	id serial PRIMARY KEY,
	vec vector(4)
);

INSERT INTO ivf_test_vectors (vec)
SELECT ('[' || random() || ',' || random() || ',' || random() || ',' || random() || ']')::vector
FROM generate_series(1, 100);

-- Create IVF index
CREATE INDEX ivf_test_idx ON ivf_test_vectors USING ivfflat (vec vector_l2_ops)
WITH (lists = 10, probes = 5);

-- Test basic scan
SELECT id, vec <-> '[0.5,0.5,0.5,0.5]'::vector AS dist
FROM ivf_test_vectors
ORDER BY vec <-> '[0.5,0.5,0.5,0.5]'::vector
LIMIT 10;

-- Test with different k values
SELECT id, vec <-> '[0.5,0.5,0.5,0.5]'::vector AS dist
FROM ivf_test_vectors
ORDER BY vec <-> '[0.5,0.5,0.5,0.5]'::vector
LIMIT 5;

-- Cleanup
DROP INDEX IF EXISTS ivf_test_idx;
DROP TABLE IF EXISTS ivf_test_vectors CASCADE;

\echo 'Test completed successfully'




