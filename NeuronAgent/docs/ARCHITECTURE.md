# NeuronAgent Architecture

## Overview

NeuronAgent is a production-ready AI agent runtime system that integrates with NeuronDB for LLM and vector operations.

## Components

### Database Layer (`internal/db/`)
- **Models**: Go structs representing database entities
- **Connection**: Connection pool management with health checks
- **Queries**: All SQL queries with prepared statements

### Agent Runtime (`internal/agent/`)
- **Runtime**: Main execution engine with state machine
- **Memory**: HNSW-based vector search for long-term memory
- **LLM**: Integration with NeuronDB LLM functions
- **Context**: Context loading and management
- **Prompt**: Prompt construction with templating

### Tools System (`internal/tools/`)
- **Registry**: Tool registration and discovery
- **Executor**: Tool execution with timeout
- **Validators**: JSON Schema validation
- **Handlers**: SQL, HTTP, Code, Shell tools

### API Layer (`internal/api/`)
- **Handlers**: REST API endpoints
- **WebSocket**: Streaming support
- **Middleware**: Auth, rate limiting, CORS, logging

### Authentication (`internal/auth/`)
- **API Keys**: Bcrypt hashing and validation
- **Rate Limiting**: Per-key rate limits
- **Roles**: RBAC support

### Background Jobs (`internal/jobs/`)
- **Queue**: PostgreSQL-based job queue (SKIP LOCKED)
- **Worker**: Worker pool with graceful shutdown
- **Processor**: Job type processors

## Data Flow

1. User sends message via API
2. Runtime loads agent and session
3. Context is loaded (recent messages + memory chunks)
4. Prompt is built
5. LLM generates response (via NeuronDB)
6. Tool calls are parsed and executed if needed
7. Final response is generated
8. Messages and memory chunks are stored

## Security

- API key authentication with bcrypt hashing
- Rate limiting per API key
- Tool execution sandboxing
- SQL tool restricted to read-only queries
- HTTP tool with URL allowlist
- Code tool with directory restrictions
- Shell tool with command whitelist

