# NeuronAgent Docker Setup

NeuronAgent connects to an external NeuronDB PostgreSQL instance. This Docker setup provides a container image that runs the service.

## Prerequisites

Docker and Docker Compose installed. Access to a running NeuronDB instance. NeuronDB extension installed in the target database.

## Quick Start

Copy the example environment file:

```bash
cp .env.example .env
```

Edit .env with your NeuronDB connection:

```bash
DB_HOST=localhost
DB_PORT=5433
DB_NAME=neurondb
DB_USER=neurondb
DB_PASSWORD=neurondb
```

Build and start:

```bash
docker compose build
docker compose up -d
```

View logs:

```bash
docker compose logs -f agent-server
```

Check health:

```bash
curl http://localhost:8080/health
```

Verify connection:

```bash
docker compose ps
curl -H "Authorization: Bearer <your-api-key>" http://localhost:8080/api/v1/agents
```

## Configuration

Configure via environment variables or config file. Environment variables take precedence.

Database configuration variables: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_MAX_OPEN_CONNS, DB_MAX_IDLE_CONNS, DB_CONN_MAX_LIFETIME.

Server configuration variables: SERVER_HOST, SERVER_PORT, SERVER_READ_TIMEOUT, SERVER_WRITE_TIMEOUT.

Logging variables: LOG_LEVEL, LOG_FORMAT.

Default values: DB_HOST=localhost, DB_PORT=5433, DB_NAME=neurondb, DB_USER=neurondb, DB_PASSWORD=neurondb, SERVER_HOST=0.0.0.0, SERVER_PORT=8080, LOG_LEVEL=info, LOG_FORMAT=json.

## Connecting to NeuronDB

From host machine, set DB_HOST=localhost, DB_PORT=5433.

From Docker network, set DB_HOST=neurondb-cpu, DB_PORT=5432.

To use Docker network: Create shared network. Connect both services. Use container name as hostname.

## Database Setup

NeuronAgent runs migrations automatically on startup. Ensure NeuronDB extension is installed in the target database:

```sql
CREATE EXTENSION IF NOT EXISTS neurondb;
```

Grant necessary permissions:

```sql
GRANT ALL PRIVILEGES ON DATABASE neurondb TO neurondb;
GRANT ALL ON SCHEMA neurondb_agent TO neurondb;
```

Migrations run in order: 001_initial_schema.sql creates schema and tables, 002_add_indexes.sql adds indexes, 003_add_triggers.sql adds triggers.

## Building the Image

Standard build:

```bash
docker compose build
```

Custom build:

```bash
docker build -f docker/Dockerfile -t neuronagent:latest ..
```

## Health Checks

Container includes health check. Check status:

```bash
docker inspect neuronagent | jq '.[0].State.Health'
```

Manual check:

```bash
curl http://localhost:8080/health
```

Health endpoint returns 200 OK if healthy, 503 Service Unavailable if database connection fails.

## Troubleshooting

Container will not start. Check logs:

```bash
docker compose logs agent-server
```

Database connection failed. Verify NeuronDB is running and accessible. Check DB_HOST, DB_PORT, DB_USER, DB_PASSWORD. Ensure network connectivity.

Extension not found. Ensure NeuronDB extension is installed. Verify database name is correct.

Port already in use. Change SERVER_PORT in .env file. Or modify port mapping in docker-compose.yml.

API not responding. Check service is running. Check logs. Test health endpoint. Verify API key.

## API Endpoints

Once running, NeuronAgent exposes: GET /health, GET /metrics, POST /api/v1/agents, GET /api/v1/agents, GET /api/v1/agents/{id}, PUT /api/v1/agents/{id}, DELETE /api/v1/agents/{id}, POST /api/v1/sessions, POST /api/v1/sessions/{id}/messages, GET /ws.

See API.md for details.

## Security

Container runs as user neuronagent. Uses Debian slim base image. Use Docker secrets or environment variables for passwords. Use Docker networks to isolate services. Store API keys securely. Rotate keys regularly.

## Integration with NeuronDB

NeuronAgent requires PostgreSQL 15 or later with NeuronDB extension. Database must have NeuronDB extension enabled. User must have appropriate permissions.

See NeuronDB/docker/README.md for setup instructions.

## Support

Documentation: NeuronAgent/README.md
GitHub Issues
Email: admin@pgelephant.com
