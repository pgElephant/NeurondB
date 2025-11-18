# Claude Desktop Configuration for NeuronDB MCP Server

## Configuration File Location

The Claude Desktop configuration file is located at:
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

## Complete Configuration

Create or update `~/Library/Application Support/Claude/claude_desktop_config.json` with the following content:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "node",
      "args": [
        "/Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/dist/index.js"
      ],
      "env": {
        "NODE_ENV": "production"
      }
    }
  }
}
```

## Alternative: Using Absolute Path with Environment Variables

If you want to use environment variables for database connection or other settings:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "node",
      "args": [
        "/Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/dist/index.js"
      ],
      "env": {
        "NODE_ENV": "production",
        "PGHOST": "localhost",
        "PGPORT": "5432",
        "PGDATABASE": "postgres",
        "PGUSER": "postgres",
        "PGPASSWORD": "your_password_here"
      }
    }
  }
}
```

## Using Configuration File (Recommended)

If you have a `mcp-config.json` file in the MCP server directory, you can configure database settings there instead of environment variables:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "node",
      "args": [
        "/Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/dist/index.js"
      ]
    }
  }
}
```

Then create `/Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/mcp-config.json`:

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
    "password": "your_password_here",
    "pool": {
      "min": 2,
      "max": 10,
      "idleTimeoutMillis": 30000,
      "connectionTimeoutMillis": 10000
    },
    "ssl": false
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
    "vector": {
      "enabled": true
    },
    "ml": {
      "enabled": true,
      "gpuEnabled": true,
      "algorithms": [
        "linear_regression",
        "ridge",
        "lasso",
        "logistic",
        "random_forest",
        "svm",
        "decision_tree",
        "naive_bayes",
        "gmm",
        "knn"
      ]
    },
    "rag": {
      "enabled": true
    },
    "reranking": {
      "enabled": true
    },
    "gpu": {
      "enabled": true
    },
    "quantization": {
      "enabled": true
    },
    "dimensionality": {
      "enabled": true
    },
    "drift": {
      "enabled": true
    },
    "metrics": {
      "enabled": true
    },
    "hybrid": {
      "enabled": true
    },
    "indexing": {
      "enabled": true
    },
    "dataManagement": {
      "enabled": true
    },
    "workers": {
      "enabled": true
    },
    "projects": {
      "enabled": true
    }
  }
}
```

## Verification

After updating the configuration:

1. **Restart Claude Desktop** - The MCP server will be loaded on startup
2. **Check MCP Connection** - In Claude Desktop, you should see the NeuronDB MCP server connected
3. **Test Tools** - Try asking Claude to use NeuronDB tools, for example:
   - "List all available NeuronDB tools"
   - "Train a linear regression model on my data"
   - "Rerank these documents using LLM"

## Troubleshooting

### MCP Server Not Connecting

1. **Check Node.js Path**: Ensure `node` is in your PATH
   ```bash
   which node
   # Should output: /opt/homebrew/bin/node (or similar)
   ```

2. **Check File Permissions**: Ensure the MCP server file is executable
   ```bash
   ls -la /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/dist/index.js
   ```

3. **Check Logs**: Claude Desktop logs are in:
   ```
   ~/Library/Logs/Claude/
   ```

### Database Connection Issues

1. **Verify PostgreSQL is Running**:
   ```bash
   pg_isready -h localhost -p 5432
   ```

2. **Test Connection**:
   ```bash
   psql -h localhost -p 5432 -U postgres -d postgres
   ```

3. **Check Extension**: Ensure NeuronDB extension is installed
   ```sql
   CREATE EXTENSION IF NOT EXISTS neurondb;
   ```

### Tool Not Available

1. **Check Feature Flags**: Ensure the feature is enabled in `mcp-config.json`
2. **Check Database**: Ensure the required tables/functions exist
3. **Check Logs**: Review Claude Desktop logs for errors

## Available Tools

Once configured, Claude Desktop will have access to all NeuronDB MCP tools:

### Vector Tools
- Vector similarity search
- Vector operations
- Index management

### ML Tools
- Model training (9 algorithms)
- Model prediction
- Model management

### Reranking Tools (All 7)
- `mmr_rerank` - Maximal Marginal Relevance
- `rerank_cross_encoder` - Neural reranking
- `rerank_llm` - LLM-powered reranking (with promptTemplate & temperature)
- `rerank_colbert` - ColBERT late interaction
- `rerank_rrf` - Reciprocal Rank Fusion
- `rerank_ensemble_weighted` - Weighted ensemble
- `rerank_ensemble_borda` - Borda count ensemble

### Other Tools
- RAG tools
- GPU tools
- Analytics tools
- And more...

## Next Steps

1. Copy the configuration to Claude Desktop config file
2. Restart Claude Desktop
3. Start using NeuronDB tools through Claude!

