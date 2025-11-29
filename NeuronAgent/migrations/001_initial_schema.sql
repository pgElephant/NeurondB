-- Schema: neurondb_agent
CREATE SCHEMA IF NOT EXISTS neurondb_agent;

-- Agents table: Agent profiles and configurations
CREATE TABLE neurondb_agent.agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    system_prompt TEXT NOT NULL,
    model_name TEXT NOT NULL,  -- NeuronDB model identifier
    memory_table TEXT,          -- Optional per-agent memory table name
    enabled_tools TEXT[] DEFAULT '{}',
    config JSONB DEFAULT '{}',  -- temperature, max_tokens, top_p, etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_model_name CHECK (model_name ~ '^[a-zA-Z0-9_-]+$'),
    CONSTRAINT valid_memory_table CHECK (memory_table IS NULL OR memory_table ~ '^[a-z][a-z0-9_]*$')
);

-- Sessions table: User conversation sessions
CREATE TABLE neurondb_agent.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    external_user_id TEXT,  -- Optional external user identifier
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_external_user_id CHECK (external_user_id IS NULL OR length(external_user_id) > 0)
);

-- Messages table: Conversation history
CREATE TABLE neurondb_agent.messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tool_name TEXT,  -- NULL unless role = 'tool'
    tool_call_id TEXT,  -- For associating tool calls with results
    token_count INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_tool_message CHECK (
        (role = 'tool' AND tool_name IS NOT NULL) OR
        (role != 'tool' AND tool_name IS NULL)
    )
);

-- Memory chunks table: Vector-embedded long-term memory
CREATE TABLE neurondb_agent.memory_chunks (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    message_id BIGINT REFERENCES neurondb_agent.messages(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    embedding neurondb_vector(768),  -- NeuronDB vector type, configurable dimension
    importance_score REAL DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_embedding CHECK (embedding IS NOT NULL)
);

-- Tools table: Tool registry
CREATE TABLE neurondb_agent.tools (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    arg_schema JSONB NOT NULL,  -- JSON Schema for arguments
    handler_type TEXT NOT NULL CHECK (handler_type IN ('sql', 'http', 'code', 'shell', 'queue')),
    handler_config JSONB DEFAULT '{}',
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_arg_schema CHECK (jsonb_typeof(arg_schema) = 'object')
);

-- Jobs table: Background job queue
CREATE TABLE neurondb_agent.jobs (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    type TEXT NOT NULL CHECK (type IN ('http_call', 'sql_task', 'shell_task', 'custom')),
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    priority INT DEFAULT 0,
    payload JSONB NOT NULL,
    result JSONB,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- API keys table: Authentication
CREATE TABLE neurondb_agent.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash TEXT NOT NULL UNIQUE,  -- Bcrypt hash of API key
    key_prefix TEXT NOT NULL,  -- First 8 chars for identification
    organization_id TEXT,
    user_id TEXT,
    rate_limit_per_minute INT DEFAULT 60,
    roles TEXT[] DEFAULT '{user}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    CONSTRAINT valid_roles CHECK (array_length(roles, 1) > 0)
);

