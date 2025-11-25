# Neurondb MCP Server

Model Context Protocol (MCP) server for Neurondb PostgreSQL extension. This server exposes Neurondb's comprehensive vector search, ML inference, analytics, and RAG capabilities through the MCP protocol.

## Architecture

The server is built with a clean, modular architecture:

```
src/
├── index.ts          # Main server entry point
├── config.ts          # Configuration management
├── logger.ts          # Structured logging
├── middleware.ts      # Middleware system
├── plugin.ts         # Plugin system
├── db.ts             # Database connection management
├── types.ts           # TypeScript type definitions
├── resources.ts       # Resource handlers
└── tools/
    ├── index.ts      # Tool exports
    ├── vector.ts     # Vector operations
    ├── ml.ts         # ML model training & prediction
    ├── analytics.ts  # Clustering & analytics
    ├── rag.ts        # RAG pipeline operations
    └── projects.ts   # ML project management
```

## Features

### Configuration System
- **File-based configuration** - JSON config file support
- **Environment variable override** - All settings can be overridden via env vars
- **Feature flags** - Enable/disable features (vector, ML, analytics, RAG, projects)
- **Database pooling** - Configurable connection pooling
- **SSL support** - Full SSL/TLS configuration
- **Logging configuration** - Structured logging with multiple levels and formats

### Modular Architecture
- **Plugin system** - Load custom plugins for extensibility
- **Middleware system** - Request/response processing pipeline
- **Feature-based tool registration** - Tools only registered if features enabled
- **Dependency injection** - Clean separation of concerns

### Vector Operations
- **vector_search** - Similarity search with L2, cosine, or inner product
- **generate_embedding** - Single text embedding generation
- **batch_embedding** - Batch embedding generation
- **create_hnsw_index** - Create HNSW indexes
- **hybrid_search** - Combine vector and full-text search

### Machine Learning
- **train_ml_model** - Train models (linear_regression, ridge, lasso, logistic, random_forest, svm, knn, decision_tree, naive_bayes)
- **predict_ml_model** - Predict using trained models
- **get_model_info** - Query model catalog
- **GPU support** - Configurable GPU acceleration

📖 **See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for complete training examples and workflows**

### Analytics
- **cluster_data** - K-means, Mini-batch K-means, GMM clustering
- **detect_outliers** - Z-score outlier detection

### RAG Pipeline
- **rag_chunk_text** - Text chunking for RAG
- **rerank_results** - Cross-encoder or LLM reranking

### Project Management
- **create_ml_project** - Create ML projects
- **list_ml_projects** - List all projects
- **train_kmeans_project** - Train K-means within project

### Resources
- **neurondb://schema** - Database schema information
- **neurondb://models** - ML models catalog
- **neurondb://indexes** - Vector indexes status
- **neurondb://config** - Configuration settings
- **neurondb://workers** - Background workers status
- **neurondb://vector_stats** - Vector statistics
- **neurondb://index_health** - Index health dashboard

## Installation

```bash
cd bin/mcp
npm install
npm run build
```

## Quick Start

**How do people use MCP servers?** There are 3 main ways:

1. **Claude Desktop** (Easiest - No code) - AI assistant integration
2. **Programmatic** (Node.js/Python) - For developers and automation
3. **Direct SQL** (Alternative) - No MCP server needed!

📖 **See [USAGE_GUIDE.md](USAGE_GUIDE.md) for complete usage instructions**

## Configuration

### Configuration File

Create `mcp-config.json` (see `mcp-config.example.json`):

```json
{
  "database": {
    "connectionString": "postgresql://user:password@localhost:5432/database",
    "pool": {
      "min": 2,
      "max": 10,
      "idleTimeoutMillis": 30000
    }
  },
  "server": {
    "name": "neurondb-mcp-server",
    "timeout": 30000,
    "enableMetrics": true
  },
  "logging": {
    "level": "info",
    "format": "text",
    "output": "stderr"
  },
  "features": {
    "vector": { "enabled": true },
    "ml": { "enabled": true, "gpuEnabled": false },
    "analytics": { "enabled": true },
    "rag": { "enabled": true },
    "projects": { "enabled": true }
  },
  "plugins": [
    {
      "name": "custom-plugin",
      "enabled": true,
      "path": "./plugins/custom.js"
    }
  ]
}
```

### Environment Variables

All settings can be overridden via environment variables:

```bash
# Database connection
export NEURONDB_CONNECTION_STRING="postgresql://user:password@localhost:5432/database"
# OR
export NEURONDB_HOST="localhost"
export NEURONDB_PORT="5432"
export NEURONDB_DATABASE="postgres"
export NEURONDB_USER="postgres"
export NEURONDB_PASSWORD="password"

# Logging
export NEURONDB_LOG_LEVEL="debug"
export NEURONDB_LOG_FORMAT="json"

# Features
export NEURONDB_ENABLE_GPU="true"

# Config file location
export NEURONDB_MCP_CONFIG="/path/to/mcp-config.json"
```

### Configuration File Locations

The server searches for config files in this order:
1. Path specified in `NEURONDB_MCP_CONFIG` env var
2. `./mcp-config.json` (current directory)
3. `./bin/mcp/mcp-config.json` (relative to server)
4. `~/.neurondb/mcp-config.json` (user home)

## Usage

### Development

```bash
npm run dev
```

### Production

```bash
npm start
```

### As MCP Server

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "node",
      "args": ["/path/to/neurondb/bin/mcp/dist/index.js"],
      "env": {
        "NEURONDB_CONNECTION_STRING": "postgresql://user:password@localhost:5432/database"
      }
    }
  }
}
```

## Plugin System

Create custom plugins to extend functionality:

```typescript
// plugins/my-plugin.ts
import { Plugin } from "../src/plugin.js";

export default {
  name: "my-plugin",
  version: "1.0.0",
  initialize: async (config, db, logger, middleware) => {
    logger.info("My plugin initialized");
  },
  tools: [
    {
      name: "my_tool",
      description: "My custom tool",
      inputSchema: {
        type: "object",
        properties: {
          param: { type: "string" }
        }
      },
      handler: async (params) => {
        return { result: "success" };
      }
    }
  ],
  resources: [
    {
      uri: "neurondb://my-resource",
      name: "My Resource",
      description: "Custom resource",
      mimeType: "application/json",
      handler: async () => {
        return { data: "value" };
      }
    }
  ]
} as Plugin;
```

Register in config:

```json
{
  "plugins": [
    {
      "name": "my-plugin",
      "enabled": true,
      "path": "./plugins/my-plugin.js"
    }
  ]
}
```

## Middleware System

Built-in middleware:
- **Validation** - Request validation
- **Logging** - Request/response logging
- **Timeout** - Request timeout handling
- **Error Handling** - Global error handling

Custom middleware can be added via plugins.

## Examples

### Vector Search

```json
{
  "tool": "vector_search",
  "arguments": {
    "table": "documents",
    "vector_column": "embedding",
    "query_vector": [0.1, 0.2, 0.3],
    "limit": 10,
    "distance_metric": "cosine"
  }
}
```

### Train ML Model

```json
{
  "tool": "train_ml_model",
  "arguments": {
    "table": "training_data",
    "feature_col": "features",
    "label_col": "label",
    "algorithm": "random_forest",
    "params": {
      "n_estimators": 100,
      "max_depth": 10
    }
  }
}
```

### Cluster Data

```json
{
  "tool": "cluster_data",
  "arguments": {
    "table": "vectors",
    "vector_column": "embedding",
    "k": 5,
    "algorithm": "kmeans",
    "max_iter": 100
  }
}
```

## Requirements

- Node.js 18+
- PostgreSQL 16, 17, or 18
- Neurondb extension installed and configured

## Resources

- **[Usage Guide](USAGE_GUIDE.md)** - **How to use the MCP server** (Claude Desktop, programmatic, SQL)
- [Training Guide](TRAINING_GUIDE.md) - Complete guide on training ML models
- [Features Overview](FEATURES.md) - All available features
- [Architecture](ARCHITECTURE.md) - Server architecture details
- [NeuronDB Documentation](https://pgelephant.com/neurondb)

## License

PostgreSQL License (same as Neurondb)
