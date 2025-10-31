-- Test sync (should not crash)
SELECT sync_index_async('test_index', 'replica_host') AS sync_initiated;

-- Test array conversion roundtrip
SELECT vector_to_array(array_to_vector(ARRAY[1.0, 2.0, 3.0]::real[]));

-- Verify catalog integrity
SELECT COUNT(*) >= 12 AS all_catalogs_present 
FROM pg_tables 
WHERE tablename LIKE 'neurondb_%';

-- Verify extension info
SELECT extname, extversion 
FROM pg_extension 
WHERE extname = 'neurondb';

