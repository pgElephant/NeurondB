# NeuronMCP Go Server

Model Context Protocol (MCP) server for NeuronDB PostgreSQL extension, implemented in Go.

## Features

- **MCP Protocol**: Full JSON-RPC 2.0 implementation with stdio transport
- **Vector Operations**: Search, embedding generation, indexing
- **ML Tools**: Training and prediction for various algorithms
- **Resources**: Schema, models, indexes, config, workers, stats
- **Middleware**: Validation, logging, timeout, error handling
- **Configuration**: JSON config files with environment variable overrides
- **Modular Architecture**: Clean separation of concerns

## Building

```bash
go build ./cmd/neurondb-mcp
```

## Configuration

Create `mcp-config.json`:

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
    "password": "password"
  },
  "server": {
    "name": "neurondb-mcp-server",
    "version": "1.0.0"
  },
  "logging": {
    "level": "info",
    "format": "text"
  },
  "features": {
    "vector": { "enabled": true },
    "ml": { "enabled": true },
    "analytics": { "enabled": true }
  }
}
```

## Usage

```bash
./neurondb-mcp
```

The server communicates via stdio using the MCP protocol.
