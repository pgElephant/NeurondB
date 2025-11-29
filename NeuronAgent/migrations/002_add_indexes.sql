-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON neurondb_agent.sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON neurondb_agent.sessions(last_activity_at);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON neurondb_agent.messages(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session_role ON neurondb_agent.messages(session_id, role);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_agent_id ON neurondb_agent.memory_chunks(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_session_id ON neurondb_agent.memory_chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON neurondb_agent.jobs(status, created_at) WHERE status IN ('queued', 'running');
CREATE INDEX IF NOT EXISTS idx_jobs_agent_session ON neurondb_agent.jobs(agent_id, session_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON neurondb_agent.api_keys(key_prefix);

-- HNSW index on memory chunks embedding (NeuronDB)
CREATE INDEX IF NOT EXISTS idx_memory_chunks_embedding_hnsw ON neurondb_agent.memory_chunks 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

