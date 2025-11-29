# NeuronMCP Docker Setup

NeuronMCP connects to an external NeuronDB PostgreSQL instance. It implements the MCP protocol using stdio for communication. Compatible with MCP clients like Claude Desktop.

## Prerequisites

Docker and Docker Compose installed. Access to a running NeuronDB instance. NeuronDB extension installed in the target database. MCP client for connecting.

## Quick Start

Copy the example environment file:

```bash
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

Build and start:

```bash
docker compose build
docker compose up -d
```

View logs:

```bash
docker compose logs -f neurondb-mcp
```

Verify connection:

```bash
docker compose exec neurondb-mcp ./neurondb-mcp
```

## Configuration

Configure via environment variables or config file. Environment variables take precedence.

Database variables: NEURONDB_HOST, NEURONDB_PORT, NEURONDB_DATABASE, NEURONDB_USER, NEURONDB_PASSWORD, NEURONDB_CONNECTION_STRING.

Logging variables: NEURONDB_LOG_LEVEL, NEURONDB_LOG_FORMAT, NEURONDB_LOG_OUTPUT.

Feature flags: NEURONDB_ENABLE_GPU.

Default values: NEURONDB_HOST=localhost, NEURONDB_PORT=5433, NEURONDB_DATABASE=neurondb, NEURONDB_USER=neurondb, NEURONDB_PASSWORD=neurondb, NEURONDB_LOG_LEVEL=info, NEURONDB_LOG_FORMAT=text.

## Configuration File

Use JSON config file as alternative. Copy mcp-config.json.example to mcp-config.json. Edit settings. Mount in docker-compose.yml. Set NEURONDB_MCP_CONFIG path.

## Connecting to NeuronDB

From host machine, set NEURONDB_HOST=localhost, NEURONDB_PORT=5433.

From Docker network, set NEURONDB_HOST=neurondb-cpu, NEURONDB_PORT=5432.

To use Docker network: Create shared network. Connect both services. Use container name as hostname.

## MCP Protocol

NeuronMCP uses Model Context Protocol over stdio. No HTTP endpoints. Communication via stdin and stdout. Messages follow JSON-RPC 2.0 format. Clients initiate all requests.

## Using with Claude Desktop

Create Claude Desktop config file claude_desktop_config.json:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--network", "neurondb-network",
        "-e", "NEURONDB_HOST=neurondb-cpu",
        "-e", "NEURONDB_PORT=5432",
        "-e", "NEURONDB_DATABASE=neurondb",
        "-e", "NEURONDB_USER=neurondb",
        "-e", "NEURONDB_PASSWORD=neurondb",
        "neurondb-mcp:latest"
      ]
    }
  }
}
```

Place config file in Claude Desktop directory:
- macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
- Windows: %APPDATA%\Claude\claude_desktop_config.json
- Linux: ~/.config/Claude/claude_desktop_config.json

Restart Claude Desktop.

## Using with Other MCP Clients

Run container interactively:

```bash
docker run -i --rm \
  -e NEURONDB_HOST=neurondb-cpu \
  -e NEURONDB_PORT=5432 \
  -e NEURONDB_DATABASE=neurondb \
  -e NEURONDB_USER=neurondb \
  -e NEURONDB_PASSWORD=neurondb \
  --network neurondb-network \
  neurondb-mcp:latest
```

## Building the Image

Standard build:

```bash
docker compose build
```

Custom build:

```bash
docker build -f docker/Dockerfile -t neurondb-mcp:latest ..
```

## Troubleshooting

Container will not start. Check logs:

```bash
docker compose logs neurondb-mcp
```

Database connection failed. Verify NeuronDB is running and accessible. Check NEURONDB_HOST, NEURONDB_PORT, NEURONDB_USER, NEURONDB_PASSWORD. Ensure network connectivity.

Extension not found. Ensure NeuronDB extension is installed. Verify database name is correct.

Stdio not working. Ensure stdin_open: true and tty: true in docker-compose.yml. For interactive use, run with docker run -i -t.

MCP client connection issues. Check container is running. Test stdio manually. Verify network connectivity. Check MCP client configuration path.

Configuration issues. Verify config file path is correct. Check file permissions. Ensure file is mounted if using volume. Verify environment variable names start with NEURONDB_. Check for typos. Ensure variables set before container starts.

## Security

Container runs as user neuronmcp. Uses Debian slim base image. Use Docker secrets or environment variables for passwords. Use Docker networks to isolate services. Store sensitive config securely.

## Integration with NeuronDB

NeuronMCP requires PostgreSQL 15 or later with NeuronDB extension. Database must have NeuronDB extension enabled. User must have appropriate permissions.

See NeuronDB/docker/README.md for setup instructions.

## MCP Tools and Resources

NeuronMCP exposes tools: vector operations, ML operations, analytics, RAG operations.

Exposes resources: schema information, model configurations, index configurations, worker status, statistics.

See NeuronMCP/README.md for full documentation.

## Support

Documentation: NeuronMCP/README.md
GitHub Issues
Email: admin@pgelephant.com
