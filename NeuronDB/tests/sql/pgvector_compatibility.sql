-- ============================================================================
-- pgvector Compatibility Test Suite
-- Tests NeuronDB API compatibility with pgvector
-- ============================================================================

-- Test 1: Distance Operators
-- pgvector supports: <-> (L2), <=> (cosine), <#> (inner product), 
--                    <+> (L1), <~> (Hamming), <*~*> (Jaccard)

SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector AS l2_distance;
SELECT '[1,2,3]'::vector <=> '[4,5,6]'::vector AS cosine_distance;
SELECT '[1,2,3]'::vector <#> '[4,5,6]'::vector AS inner_product;
SELECT '[1,2,3]'::vector <+> '[4,5,6]'::vector AS l1_distance;
SELECT '[1,0,1]'::vector <~> '[0,1,0]'::vector AS hamming_distance;
SELECT '[1,1,0]'::vector <*~*> '[1,0,1]'::vector AS jaccard_distance;

-- Test 2: halfvec Type
-- pgvector supports halfvec for 2x compression

SELECT '[1.0, 2.0, 3.0]'::halfvec AS halfvec_test;
SELECT vector_to_halfvec('[1.0, 2.0, 3.0]'::vector) AS vector_to_halfvec;
SELECT halfvec_to_vector('[1.0, 2.0, 3.0]'::halfvec) AS halfvec_to_vector;
SELECT '[1.0, 2.0]'::halfvec <-> '[3.0, 4.0]'::halfvec AS halfvec_l2;

-- Test 3: sparsevec Type
-- pgvector supports sparsevec for sparse vectors

SELECT '{dim:10,1:0.5,5:0.3,10:0.8}'::sparsevec AS sparsevec_test;
SELECT sparsevec_l2_norm('{dim:10,1:0.5,5:0.3}'::sparsevec) AS sparsevec_norm;
SELECT sparsevec_l2_normalize('{dim:10,1:0.5,5:0.3}'::sparsevec) AS sparsevec_normalize;
SELECT '{dim:10,1:0.5}'::sparsevec <-> '{dim:10,5:0.3}'::sparsevec AS sparsevec_l2;
SELECT '{dim:10,1:0.5}'::sparsevec <=> '{dim:10,5:0.3}'::sparsevec AS sparsevec_cosine;
SELECT '{dim:10,1:0.5}'::sparsevec <#> '{dim:10,5:0.3}'::sparsevec AS sparsevec_inner;

-- Test 4: Type Conversions
-- pgvector supports conversions between types

SELECT vector_to_halfvec('[1.0, 2.0, 3.0]'::vector) AS v_to_halfvec;
SELECT halfvec_to_vector('[1.0, 2.0, 3.0]'::halfvec) AS halfvec_to_v;
SELECT vector_to_sparsevec('[0,1,0,0,2,0,0,0,0,3]'::vector) AS v_to_sparsevec;
SELECT sparsevec_to_vector('{dim:10,1:1.0,4:2.0,9:3.0}'::sparsevec) AS sparsevec_to_v;

-- Test 5: Index Parameters
-- pgvector HNSW: m, ef_construction, ef_search (runtime)
-- pgvector IVF: lists, probes (runtime)

-- HNSW index with parameters
CREATE TABLE IF NOT EXISTS test_hnsw (
    id SERIAL PRIMARY KEY,
    vec vector(128)
);

-- Test HNSW index creation with parameters (should work)
DO $$
BEGIN
    BEGIN
        CREATE INDEX test_hnsw_idx ON test_hnsw USING hnsw (vec vector_l2_ops)
        WITH (m = 16, ef_construction = 64);
        RAISE NOTICE 'HNSW index created successfully with m and ef_construction';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'HNSW index creation: %', SQLERRM;
    END;
END $$;

-- Test IVF index creation with parameters (should work)
CREATE TABLE IF NOT EXISTS test_ivf (
    id SERIAL PRIMARY KEY,
    vec vector(128)
);

DO $$
BEGIN
    BEGIN
        CREATE INDEX test_ivf_idx ON test_ivf USING ivf (vec vector_l2_ops)
        WITH (lists = 100);
        RAISE NOTICE 'IVF index created successfully with lists parameter';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'IVF index creation: %', SQLERRM;
    END;
END $$;

-- Test 6: Runtime Parameters
-- pgvector supports SET ef_search and SET probes at runtime

SET neurondb.hnsw_ef_search = 100;
SET neurondb.ivf_probes = 20;

-- Test 7: Binary/Bit Type
-- pgvector supports bit type for binary vectors

SELECT vector_to_bit('[1,0,1,0]'::vector) AS vector_to_bit_test;
SELECT bit_to_vector('1010'::bit) AS bit_to_vector_test;

-- Test 8: Sparse Vector Operators
-- pgvector sparsevec supports: <->, <#>, <=>, <+>

SELECT '{dim:10,1:0.5,5:0.3}'::sparsevec <-> '{dim:10,2:0.4,6:0.2}'::sparsevec AS sparse_l2;
SELECT '{dim:10,1:0.5,5:0.3}'::sparsevec <=> '{dim:10,2:0.4,6:0.2}'::sparsevec AS sparse_cosine;
SELECT '{dim:10,1:0.5,5:0.3}'::sparsevec <#> '{dim:10,2:0.4,6:0.2}'::sparsevec AS sparse_inner;

-- Cleanup
DROP TABLE IF EXISTS test_hnsw CASCADE;
DROP TABLE IF EXISTS test_ivf CASCADE;

-- Summary
SELECT 'pgvector compatibility tests completed' AS status;

