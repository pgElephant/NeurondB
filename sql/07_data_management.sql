-- Test data management functions
CREATE TABLE test_vectors_dm (
    id serial PRIMARY KEY,
    embedding vector,
    created_at timestamptz DEFAULT now(),
    last_accessed timestamptz,
    is_compressed boolean DEFAULT false
);

INSERT INTO test_vectors_dm (embedding) VALUES 
    ('[1.0, 0.0, 0.0]'),
    ('[0.0, 1.0, 0.0]'),
    ('[0.0, 0.0, 1.0]');

-- Test vacuum (should return count)
SELECT vacuum_vectors('test_vectors_dm', false) >= 0 AS vacuum_executed;

-- Test compress (should return count)
SELECT compress_cold_tier('test_vectors_dm', 30) >= 0 AS compress_executed;

-- Test rebalance
SELECT rebalance_index('test_index', 0.8) AS rebalance_executed;

DROP TABLE test_vectors_dm;

