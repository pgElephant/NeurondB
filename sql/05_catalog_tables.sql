-- Test catalog table creation
SELECT tablename FROM pg_tables 
WHERE tablename LIKE 'neurondb_%' 
ORDER BY tablename;

-- Test job queue table
INSERT INTO neurondb_job_queue (job_type, payload, tenant_id) 
VALUES ('test_job', '{"key": "value"}'::jsonb, 1);
SELECT job_type, tenant_id, status FROM neurondb_job_queue;

-- Test query metrics table
INSERT INTO neurondb_query_metrics (query_type, latency_ms, recall_at_k, ef_search)
VALUES ('knn_search', 25.5, 0.95, 64);
SELECT query_type, ef_search FROM neurondb_query_metrics;

-- Test embedding cache table
INSERT INTO neurondb_embedding_cache (cache_key, embedding, model_name)
VALUES ('test_key', '[1.0, 2.0, 3.0]'::vector, 'test_model');
SELECT cache_key, model_name FROM neurondb_embedding_cache;

-- Count all catalog tables
SELECT COUNT(*) AS catalog_table_count FROM pg_tables WHERE tablename LIKE 'neurondb_%';

