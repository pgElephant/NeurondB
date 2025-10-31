-- Test aggregate functions
CREATE TEMP TABLE test_vectors (id int, v vector);
INSERT INTO test_vectors VALUES 
    (1, '[1.0, 0.0]'),
    (2, '[0.0, 1.0]'),
    (3, '[1.0, 1.0]'),
    (4, '[2.0, 2.0]');

-- Test vector average
SELECT vector_avg(v) FROM test_vectors;

-- Test vector sum
SELECT vector_sum(v) FROM test_vectors;

-- Test with WHERE clause
SELECT vector_avg(v) FROM test_vectors WHERE id <= 2;

