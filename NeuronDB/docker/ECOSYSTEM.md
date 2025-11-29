# NeuronDB Ecosystem Docker Setup

Run NeuronDB, NeuronAgent, and NeuronMCP as separate Docker services. Each service connects to NeuronDB PostgreSQL independently.

## Quick Start

Start NeuronDB:

```bash
cd NeuronDB/docker
docker compose build neurondb
docker compose up -d neurondb
```

Wait for healthy status:

```bash
docker compose ps neurondb
```

Verify connection:

```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT version();"
```

Start NeuronAgent:

```bash
cd ../../NeuronAgent/docker
cp .env.example .env
```

Edit .env:

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
docker compose up -d agent-server
```

Verify:

```bash
curl http://localhost:8080/health
```

Start NeuronMCP:

```bash
cd ../../NeuronMCP/docker
cp .env.example .env
```

Edit .env:

```bash
NEURONDB_HOST=localhost
NEURONDB_PORT=5433
NEURONDB_DATABASE=neurondb
NEURONDB_USER=neurondb
NEURONDB_PASSWORD=neurondb
```

Build and start:

```bash
docker compose build
docker compose up -d neurondb-mcp
```

## Network Configuration

Option 1: Host networking. All services use localhost. Set DB_HOST=localhost for NeuronAgent. Set NEURONDB_HOST=localhost for NeuronMCP.

Option 2: Docker network. Create shared network:

```bash
docker network create neurondb-ecosystem
```

Connect NeuronDB:

```bash
docker network connect neurondb-ecosystem neurondb-cpu
```

Connect NeuronAgent:

```bash
docker network connect neurondb-ecosystem neuronagent
```

Connect NeuronMCP:

```bash
docker network connect neurondb-ecosystem neurondb-mcp
```

Use container names as hostnames. Set DB_HOST=neurondb-cpu for NeuronAgent. Set NEURONDB_HOST=neurondb-cpu for NeuronMCP.

## Service Connection Matrix

NeuronDB: Port 5433 on host, 5432 in container.

NeuronAgent: Port 8080. Connects to NeuronDB using DB_HOST and DB_PORT.

NeuronMCP: Uses stdio. Connects to NeuronDB using NEURONDB_HOST and NEURONDB_PORT.

## Complete Setup Example

Create network:

```bash
docker network create neurondb-ecosystem
```

Start NeuronDB:

```bash
cd NeuronDB/docker
docker compose up -d neurondb
docker network connect neurondb-ecosystem neurondb-cpu
```

Start NeuronAgent:

```bash
cd ../../NeuronAgent/docker

cat > .env << EOF
DB_HOST=neurondb-cpu
DB_PORT=5432
DB_NAME=neurondb
DB_USER=neurondb
DB_PASSWORD=neurondb
SERVER_PORT=8080
EOF

docker compose build
docker compose up -d agent-server
docker network connect neurondb-ecosystem neuronagent
```

Start NeuronMCP:

```bash
cd ../../NeuronMCP/docker

cat > .env << EOF
NEURONDB_HOST=neurondb-cpu
NEURONDB_PORT=5432
NEURONDB_DATABASE=neurondb
NEURONDB_USER=neurondb
NEURONDB_PASSWORD=neurondb
EOF

docker compose build
docker compose up -d neurondb-mcp
docker network connect neurondb-ecosystem neurondb-mcp
```

Verify all services:

```bash
docker compose -f NeuronDB/docker/docker-compose.yml ps neurondb
curl http://localhost:8080/health
docker compose -f NeuronMCP/docker/docker-compose.yml logs neurondb-mcp
```

## Configuration Reference

NeuronDB default port mapping: 5433:5432. Database: neurondb. User: neurondb. Password: neurondb. Change password in production.

NeuronAgent key variables: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, SERVER_PORT.

NeuronMCP key variables: NEURONDB_HOST, NEURONDB_PORT, NEURONDB_DATABASE, NEURONDB_USER, NEURONDB_PASSWORD.

See individual README files for full configuration options.

## Service Management

Start all services:

```bash
cd NeuronDB/docker && docker compose up -d neurondb
cd ../../NeuronAgent/docker && docker compose up -d agent-server
cd ../../NeuronMCP/docker && docker compose up -d neurondb-mcp
```

Stop all services:

```bash
cd NeuronMCP/docker && docker compose down
cd ../NeuronAgent/docker && docker compose down
cd ../../NeuronDB/docker && docker compose down neurondb
```

View logs:

```bash
docker compose -f NeuronDB/docker/docker-compose.yml logs -f neurondb
docker compose -f NeuronAgent/docker/docker-compose.yml logs -f agent-server
docker compose -f NeuronMCP/docker/docker-compose.yml logs -f neurondb-mcp
```

Check health:

```bash
docker inspect neurondb-cpu | jq '.[0].State.Health'
curl http://localhost:8080/health
docker compose -f NeuronMCP/docker/docker-compose.yml logs neurondb-mcp | tail -20
```

## Troubleshooting

Services cannot connect to NeuronDB. Verify NeuronDB is running. Check network connectivity. Test connection manually. Verify firewall settings. Check environment variables match.

Port already in use. Change port mappings in docker-compose.yml. Stop conflicting services. Use different ports.

NeuronDB extension not found. Verify extension installed with \dx neurondb. Create extension if missing. Check database name matches configuration.

MCP client cannot connect. Verify container running with stdio enabled. Check MCP client configuration path. Test stdio manually. Verify Docker command syntax.

## Support

GitHub Issues: https://github.com/pgElephant/NeurondB/issues
Documentation: https://pgelephant.com/neurondb
Email: admin@pgelephant.com
