package db

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/google/uuid"
	"github.com/jmoiron/sqlx"
)

// Agent queries
const (
	createAgentQuery = `
		INSERT INTO neurondb_agent.agents 
		(name, description, system_prompt, model_name, memory_table, enabled_tools, config)
		VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
		RETURNING id, created_at, updated_at`

	getAgentByIDQuery = `SELECT * FROM neurondb_agent.agents WHERE id = $1`

	listAgentsQuery = `SELECT * FROM neurondb_agent.agents ORDER BY created_at DESC`

	updateAgentQuery = `
		UPDATE neurondb_agent.agents 
		SET name = $2, description = $3, system_prompt = $4, model_name = $5,
			memory_table = $6, enabled_tools = $7, config = $8::jsonb
		WHERE id = $1
		RETURNING updated_at`

	deleteAgentQuery = `DELETE FROM neurondb_agent.agents WHERE id = $1`
)

// Session queries
const (
	createSessionQuery = `
		INSERT INTO neurondb_agent.sessions (agent_id, external_user_id, metadata)
		VALUES ($1, $2, $3::jsonb)
		RETURNING id, created_at, last_activity_at`

	getSessionQuery = `SELECT * FROM neurondb_agent.sessions WHERE id = $1`

	listSessionsQuery = `
		SELECT * FROM neurondb_agent.sessions 
		WHERE agent_id = $1 
		ORDER BY last_activity_at DESC 
		LIMIT $2 OFFSET $3`

	deleteSessionQuery = `DELETE FROM neurondb_agent.sessions WHERE id = $1`
)

// Message queries
const (
	createMessageQuery = `
		INSERT INTO neurondb_agent.messages 
		(session_id, role, content, tool_name, tool_call_id, token_count, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
		RETURNING id, created_at`

	getMessagesQuery = `
		SELECT * FROM neurondb_agent.messages 
		WHERE session_id = $1 
		ORDER BY created_at ASC 
		LIMIT $2 OFFSET $3`

	getRecentMessagesQuery = `
		SELECT * FROM neurondb_agent.messages 
		WHERE session_id = $1 
		ORDER BY created_at DESC 
		LIMIT $2`
)

// Memory chunk queries
const (
	createMemoryChunkQuery = `
		INSERT INTO neurondb_agent.memory_chunks 
		(agent_id, session_id, message_id, content, embedding, importance_score, metadata)
		VALUES ($1, $2, $3, $4, $5::neurondb_vector, $6, $7::jsonb)
		RETURNING id, created_at`

	searchMemoryQuery = `
		SELECT id, agent_id, session_id, message_id, content, importance_score, metadata, created_at,
			   1 - (embedding <=> $1::neurondb_vector) AS similarity
		FROM neurondb_agent.memory_chunks
		WHERE agent_id = $2
		ORDER BY embedding <=> $1::neurondb_vector
		LIMIT $3`
)

// Tool queries
const (
	createToolQuery = `
		INSERT INTO neurondb_agent.tools 
		(name, description, arg_schema, handler_type, handler_config, enabled)
		VALUES ($1, $2, $3::jsonb, $4, $5::jsonb, $6)
		RETURNING created_at, updated_at`

	getToolQuery = `SELECT * FROM neurondb_agent.tools WHERE name = $1`

	listToolsQuery = `SELECT * FROM neurondb_agent.tools WHERE enabled = true ORDER BY name`

	updateToolQuery = `
		UPDATE neurondb_agent.tools 
		SET description = $2, arg_schema = $3::jsonb, handler_type = $4, 
			handler_config = $5::jsonb, enabled = $6
		WHERE name = $1
		RETURNING updated_at`

	deleteToolQuery = `DELETE FROM neurondb_agent.tools WHERE name = $1`
)

// Job queries
const (
	createJobQuery = `
		INSERT INTO neurondb_agent.jobs 
		(agent_id, session_id, type, status, priority, payload, max_retries)
		VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
		RETURNING id, created_at, updated_at`

	getJobQuery = `SELECT * FROM neurondb_agent.jobs WHERE id = $1`

	claimJobQuery = `
		UPDATE neurondb_agent.jobs 
		SET status = 'running', started_at = NOW(), updated_at = NOW()
		WHERE id = (
			SELECT id FROM neurondb_agent.jobs
			WHERE status = 'queued'
			ORDER BY priority DESC, created_at ASC
			LIMIT 1
			FOR UPDATE SKIP LOCKED
		)
		RETURNING id, agent_id, session_id, type, status, priority, payload, 
		          result, error_message, retry_count, max_retries, 
		          created_at, updated_at, started_at, completed_at`

	updateJobQuery = `
		UPDATE neurondb_agent.jobs 
		SET status = $2, result = $3::jsonb, error_message = $4, 
			retry_count = $5, completed_at = $6, updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	listJobsQuery = `
		SELECT * FROM neurondb_agent.jobs 
		WHERE ($1::uuid IS NULL OR agent_id = $1)
		AND ($2::uuid IS NULL OR session_id = $2)
		ORDER BY created_at DESC 
		LIMIT $3 OFFSET $4`
)

// API Key queries
const (
	createAPIKeyQuery = `
		INSERT INTO neurondb_agent.api_keys 
		(key_hash, key_prefix, organization_id, user_id, rate_limit_per_minute, roles, metadata, expires_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
		RETURNING id, created_at`

	getAPIKeyByPrefixQuery = `SELECT * FROM neurondb_agent.api_keys WHERE key_prefix = $1`

	getAPIKeyByIDQuery = `SELECT * FROM neurondb_agent.api_keys WHERE id = $1`

	listAPIKeysQuery = `
		SELECT * FROM neurondb_agent.api_keys 
		WHERE ($1::text IS NULL OR organization_id = $1)
		ORDER BY created_at DESC`

	updateAPIKeyLastUsedQuery = `
		UPDATE neurondb_agent.api_keys 
		SET last_used_at = NOW()
		WHERE id = $1`

	deleteAPIKeyQuery = `DELETE FROM neurondb_agent.api_keys WHERE id = $1`
)

// NeuronDB function wrappers
const (
	embedTextQuery   = `SELECT neurondb_embed($1, $2) AS embedding`
	llmGenerateQuery = `SELECT neurondb_llm_generate($1, $2, $3) AS output`
)

type Queries struct {
	db *sqlx.DB
}

func NewQueries(db *sqlx.DB) *Queries {
	return &Queries{db: db}
}

// Agent methods
func (q *Queries) CreateAgent(ctx context.Context, agent *Agent) error {
	return q.db.GetContext(ctx, agent, createAgentQuery,
		agent.Name, agent.Description, agent.SystemPrompt, agent.ModelName,
		agent.MemoryTable, agent.EnabledTools, agent.Config)
}

func (q *Queries) GetAgentByID(ctx context.Context, id uuid.UUID) (*Agent, error) {
	var agent Agent
	err := q.db.GetContext(ctx, &agent, getAgentByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("agent not found: %w", err)
	}
	return &agent, err
}

func (q *Queries) ListAgents(ctx context.Context) ([]Agent, error) {
	var agents []Agent
	err := q.db.SelectContext(ctx, &agents, listAgentsQuery)
	return agents, err
}

func (q *Queries) UpdateAgent(ctx context.Context, agent *Agent) error {
	return q.db.GetContext(ctx, agent, updateAgentQuery,
		agent.ID, agent.Name, agent.Description, agent.SystemPrompt, agent.ModelName,
		agent.MemoryTable, agent.EnabledTools, agent.Config)
}

func (q *Queries) DeleteAgent(ctx context.Context, id uuid.UUID) error {
	result, err := q.db.ExecContext(ctx, deleteAgentQuery, id)
	if err != nil {
		return err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return fmt.Errorf("agent not found")
	}
	return nil
}

// Session methods
func (q *Queries) CreateSession(ctx context.Context, session *Session) error {
	return q.db.GetContext(ctx, session, createSessionQuery,
		session.AgentID, session.ExternalUserID, session.Metadata)
}

func (q *Queries) GetSession(ctx context.Context, id uuid.UUID) (*Session, error) {
	var session Session
	err := q.db.GetContext(ctx, &session, getSessionQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("session not found: %w", err)
	}
	return &session, err
}

func (q *Queries) ListSessions(ctx context.Context, agentID uuid.UUID, limit, offset int) ([]Session, error) {
	var sessions []Session
	err := q.db.SelectContext(ctx, &sessions, listSessionsQuery, agentID, limit, offset)
	return sessions, err
}

func (q *Queries) DeleteSession(ctx context.Context, id uuid.UUID) error {
	result, err := q.db.ExecContext(ctx, deleteSessionQuery, id)
	if err != nil {
		return err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return fmt.Errorf("session not found")
	}
	return nil
}

// Message methods
func (q *Queries) CreateMessage(ctx context.Context, message *Message) (*Message, error) {
	err := q.db.GetContext(ctx, message, createMessageQuery,
		message.SessionID, message.Role, message.Content, message.ToolName,
		message.ToolCallID, message.TokenCount, message.Metadata)
	return message, err
}

func (q *Queries) GetMessages(ctx context.Context, sessionID uuid.UUID, limit, offset int) ([]Message, error) {
	var messages []Message
	err := q.db.SelectContext(ctx, &messages, getMessagesQuery, sessionID, limit, offset)
	return messages, err
}

func (q *Queries) GetRecentMessages(ctx context.Context, sessionID uuid.UUID, limit int) ([]Message, error) {
	var messages []Message
	err := q.db.SelectContext(ctx, &messages, getRecentMessagesQuery, sessionID, limit)
	return messages, err
}

// Memory chunk methods
func (q *Queries) CreateMemoryChunk(ctx context.Context, chunk *MemoryChunk) (*MemoryChunk, error) {
	// Convert embedding to string format for neurondb_vector
	embeddingStr := formatVector(chunk.Embedding)
	err := q.db.GetContext(ctx, chunk, createMemoryChunkQuery,
		chunk.AgentID, chunk.SessionID, chunk.MessageID, chunk.Content,
		embeddingStr, chunk.ImportanceScore, chunk.Metadata)
	return chunk, err
}

func (q *Queries) SearchMemory(ctx context.Context, agentID uuid.UUID, queryEmbedding []float32, topK int) ([]MemoryChunkWithSimilarity, error) {
	embeddingStr := formatVector(queryEmbedding)
	var chunks []MemoryChunkWithSimilarity
	err := q.db.SelectContext(ctx, &chunks, searchMemoryQuery, embeddingStr, agentID, topK)
	return chunks, err
}

// Tool methods
func (q *Queries) CreateTool(ctx context.Context, tool *Tool) error {
	return q.db.GetContext(ctx, tool, createToolQuery,
		tool.Name, tool.Description, tool.ArgSchema, tool.HandlerType,
		tool.HandlerConfig, tool.Enabled)
}

func (q *Queries) GetTool(ctx context.Context, name string) (*Tool, error) {
	var tool Tool
	err := q.db.GetContext(ctx, &tool, getToolQuery, name)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("tool not found: %w", err)
	}
	return &tool, err
}

func (q *Queries) ListTools(ctx context.Context) ([]Tool, error) {
	var tools []Tool
	err := q.db.SelectContext(ctx, &tools, listToolsQuery)
	return tools, err
}

func (q *Queries) UpdateTool(ctx context.Context, tool *Tool) error {
	return q.db.GetContext(ctx, tool, updateToolQuery,
		tool.Name, tool.Description, tool.ArgSchema, tool.HandlerType,
		tool.HandlerConfig, tool.Enabled)
}

func (q *Queries) DeleteTool(ctx context.Context, name string) error {
	result, err := q.db.ExecContext(ctx, deleteToolQuery, name)
	if err != nil {
		return err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return fmt.Errorf("tool not found")
	}
	return nil
}

// Job methods
func (q *Queries) CreateJob(ctx context.Context, job *Job) (*Job, error) {
	err := q.db.GetContext(ctx, job, createJobQuery,
		job.AgentID, job.SessionID, job.Type, job.Status, job.Priority,
		job.Payload, job.MaxRetries)
	return job, err
}

func (q *Queries) GetJob(ctx context.Context, id int64) (*Job, error) {
	var job Job
	err := q.db.GetContext(ctx, &job, getJobQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("job not found: %w", err)
	}
	return &job, err
}

func (q *Queries) ClaimJob(ctx context.Context) (*Job, error) {
	var job Job
	err := q.db.GetContext(ctx, &job, claimJobQuery)
	if err == sql.ErrNoRows {
		return nil, nil // No jobs available
	}
	return &job, err
}

func (q *Queries) UpdateJob(ctx context.Context, id int64, status string, result map[string]interface{}, errorMsg *string, retryCount int, completedAt *sql.NullTime) error {
	var completedAtVal interface{}
	if completedAt != nil && completedAt.Valid {
		completedAtVal = completedAt.Time
	} else {
		completedAtVal = nil
	}
	_, err := q.db.ExecContext(ctx, updateJobQuery, id, status, result, errorMsg, retryCount, completedAtVal)
	return err
}

func (q *Queries) ListJobs(ctx context.Context, agentID *uuid.UUID, sessionID *uuid.UUID, limit, offset int) ([]Job, error) {
	var jobs []Job
	err := q.db.SelectContext(ctx, &jobs, listJobsQuery, agentID, sessionID, limit, offset)
	return jobs, err
}

// API Key methods
func (q *Queries) CreateAPIKey(ctx context.Context, apiKey *APIKey) error {
	return q.db.GetContext(ctx, apiKey, createAPIKeyQuery,
		apiKey.KeyHash, apiKey.KeyPrefix, apiKey.OrganizationID, apiKey.UserID,
		apiKey.RateLimitPerMin, apiKey.Roles, apiKey.Metadata, apiKey.ExpiresAt)
}

func (q *Queries) GetAPIKeyByPrefix(ctx context.Context, prefix string) (*APIKey, error) {
	var apiKey APIKey
	err := q.db.GetContext(ctx, &apiKey, getAPIKeyByPrefixQuery, prefix)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("API key not found: %w", err)
	}
	return &apiKey, err
}

func (q *Queries) GetAPIKeyByID(ctx context.Context, id uuid.UUID) (*APIKey, error) {
	var apiKey APIKey
	err := q.db.GetContext(ctx, &apiKey, getAPIKeyByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("API key not found: %w", err)
	}
	return &apiKey, err
}

func (q *Queries) ListAPIKeys(ctx context.Context, organizationID *string) ([]APIKey, error) {
	var keys []APIKey
	err := q.db.SelectContext(ctx, &keys, listAPIKeysQuery, organizationID)
	return keys, err
}

func (q *Queries) UpdateAPIKeyLastUsed(ctx context.Context, id uuid.UUID) error {
	_, err := q.db.ExecContext(ctx, updateAPIKeyLastUsedQuery, id)
	return err
}

func (q *Queries) DeleteAPIKey(ctx context.Context, id uuid.UUID) error {
	result, err := q.db.ExecContext(ctx, deleteAPIKeyQuery, id)
	if err != nil {
		return err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return fmt.Errorf("API key not found")
	}
	return nil
}

// Helper function to format vector for PostgreSQL
func formatVector(vec []float32) string {
	if len(vec) == 0 {
		return "[]"
	}
	result := "["
	for i, v := range vec {
		if i > 0 {
			result += ","
		}
		result += fmt.Sprintf("%.6f", v)
	}
	result += "]"
	return result
}

