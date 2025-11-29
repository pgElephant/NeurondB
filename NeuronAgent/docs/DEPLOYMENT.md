# NeuronAgent Deployment Guide

## Prerequisites

- PostgreSQL 15+ with NeuronDB extension
- Go 1.21+
- Docker (optional)

## Configuration

Create a `config.yaml` file:

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  read_timeout: 30s
  write_timeout: 30s

database:
  host: "localhost"
  port: 5432
  database: "neurondb"
  user: "postgres"
  password: "postgres"
  max_open_conns: 25
  max_idle_conns: 5
  conn_max_lifetime: 5m

auth:
  api_key_header: "Authorization"

logging:
  level: "info"
  format: "json"
```

Or use environment variables:
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `SERVER_HOST`, `SERVER_PORT`
- `CONFIG_PATH`

## Database Setup

1. Create database:
```sql
CREATE DATABASE neurondb;
```

2. Install NeuronDB extension:
```sql
CREATE EXTENSION neurondb;
```

3. Run migrations:
```bash
psql -d neurondb -f migrations/001_initial_schema.sql
psql -d neurondb -f migrations/002_add_indexes.sql
psql -d neurondb -f migrations/003_add_triggers.sql
```

## Running

### Local Development

```bash
go mod download
go run cmd/agent-server/main.go
```

### Docker

```bash
docker-compose up -d
```

### Production

Build binary:
```bash
go build -o agent-server ./cmd/agent-server
```

Run:
```bash
./agent-server
```

## API Key Generation

Use the API key manager to generate keys programmatically or create them directly in the database.

## Health Check

```
GET /health
```

Returns 200 if healthy, 503 if database connection fails.

