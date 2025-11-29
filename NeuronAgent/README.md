# NeuronAgent

A production-ready AI agent runtime system that integrates with NeuronDB for LLM and vector operations.

## Features

- **Agent Runtime**: Complete state machine for agent execution
- **Long-term Memory**: HNSW-based vector search for context retrieval
- **Tool System**: Extensible tool registry with SQL, HTTP, Code, and Shell tools
- **REST API**: Full CRUD API for agents, sessions, and messages
- **WebSocket Support**: Streaming agent responses
- **Authentication**: API key-based auth with rate limiting
- **Background Jobs**: PostgreSQL-based job queue with worker pool
- **NeuronDB Integration**: Direct integration with NeuronDB embedding and LLM functions

## Quick Start

1. **Setup Database**:
```bash
createdb neurondb
psql -d neurondb -c "CREATE EXTENSION neurondb;"
psql -d neurondb -f migrations/001_initial_schema.sql
psql -d neurondb -f migrations/002_add_indexes.sql
psql -d neurondb -f migrations/003_add_triggers.sql
```

2. **Configure**:
Set environment variables or create `config.yaml` (see `docs/DEPLOYMENT.md`)

3. **Run**:
```bash
go run cmd/agent-server/main.go
```

4. **Test**:
```bash
curl -H "Authorization: Bearer <api_key>" http://localhost:8080/api/v1/agents
```

## Documentation

- [API Documentation](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## License

See LICENSE file.
