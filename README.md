# NeuronDB Ecosystem

Three components that work together. Each runs independently. All connect to NeuronDB PostgreSQL.

## Components

### NeuronDB

PostgreSQL extension. Adds vector search, machine learning algorithms, and embedding generation to your database.

Location: NeuronDB/

Documentation:
- NeuronDB/README.md
- NeuronDB/docker/README.md

### NeuronAgent

Agent runtime system. Provides REST API and WebSocket support. Uses long-term memory with vector search.

Location: NeuronAgent/

Documentation:
- NeuronAgent/README.md
- NeuronAgent/docker/README.md

### NeuronMCP

MCP protocol server. Connects MCP clients to NeuronDB. Uses stdio communication.

Location: NeuronMCP/

Documentation:
- NeuronMCP/README.md
- NeuronMCP/docker/README.md

## How They Connect

All three components connect to the same NeuronDB PostgreSQL instance. You configure each service separately. They do not require each other to run.

NeuronAgent connects via database connection string. NeuronMCP connects via database connection string. Both read from environment variables or config files.

## Quick Start

Start NeuronDB first:

```bash
cd NeuronDB/docker
docker compose up -d neurondb
```

Wait until the container shows healthy status.

Start NeuronAgent:

```bash
cd ../../NeuronAgent/docker
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

Then start:

```bash
docker compose build
docker compose up -d
```

Start NeuronMCP:

```bash
cd ../../NeuronMCP/docker
cp .env.example .env
```

Edit .env with your NeuronDB connection:

```bash
NEURONDB_HOST=localhost
NEURONDB_PORT=5433
NEURONDB_DATABASE=neurondb
NEURONDB_USER=neurondb
NEURONDB_PASSWORD=neurondb
```

Then start:

```bash
docker compose build
docker compose up -d
```

## Configuration

All services connect to the same database. Default values:

Host: localhost
Port: 5432 for direct connection, 5433 for Docker
Database: neurondb
User: neurondb
Password: neurondb

Set environment variables to override defaults.

For NeuronAgent, set DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD.

For NeuronMCP, set NEURONDB_HOST, NEURONDB_PORT, NEURONDB_DATABASE, NEURONDB_USER, NEURONDB_PASSWORD.

## Network Options

Option 1: All services on same host. Use localhost as hostname.

Option 2: Docker network. Create shared network. Connect all containers. Use container names as hostnames.

See NeuronDB/docker/ECOSYSTEM.md for complete setup instructions.

## Directory Structure

```
NeuronDB/
├── NeuronDB/
│   ├── docker/
│   ├── docs/
│   └── README.md
├── NeuronAgent/
│   ├── docker/
│   ├── docs/
│   └── README.md
├── NeuronMCP/
│   ├── docker/
│   └── README.md
└── README.md
```

## Requirements

NeuronDB requires PostgreSQL 16, 17, or 18. See NeuronDB/INSTALL.md.

NeuronAgent requires Go 1.23 or later and PostgreSQL with NeuronDB extension.

NeuronMCP requires Go 1.23 or later and PostgreSQL with NeuronDB extension.

## Docker Setup

Each component includes Docker configuration:

- NeuronDB/docker/ supports CPU, CUDA, ROCm, and Metal variants
- NeuronAgent/docker/ provides standalone service container
- NeuronMCP/docker/ provides standalone service container

See NeuronDB/docker/ECOSYSTEM.md for running all services together.

## Documentation

Component documentation:
- NeuronDB: NeuronDB/README.md, NeuronDB/docker/README.md
- NeuronAgent: NeuronAgent/README.md, NeuronAgent/docker/README.md
- NeuronMCP: NeuronMCP/README.md, NeuronMCP/docker/README.md

Ecosystem documentation:
- NeuronDB/docker/ECOSYSTEM.md explains running all services together

## Support

GitHub Issues: https://github.com/pgElephant/NeurondB/issues
Documentation: https://pgelephant.com/neurondb
Email: admin@pgelephant.com
