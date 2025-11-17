# NeuronDB MCP Server - Usage Guide

Complete guide on how to use the NeuronDB MCP server in different scenarios.

## Table of Contents

1. [What is MCP?](#what-is-mcp)
2. [Usage Methods](#usage-methods)
3. [Claude Desktop Integration](#claude-desktop-integration)
4. [Programmatic Usage](#programmatic-usage)
5. [Direct SQL (Alternative)](#direct-sql-alternative)
6. [Examples](#examples)

## What is MCP?

**Model Context Protocol (MCP)** is a protocol that allows AI assistants (like Claude) to interact with external tools and data sources. The NeuronDB MCP server exposes all NeuronDB features through this protocol.

## Usage Methods

### Method 1: Claude Desktop (Easiest - No Code Required)

**Best for:** Non-developers, quick access, interactive use

#### Setup

1. **Install Claude Desktop** (if not already installed)
   - Download from: https://claude.ai/download

2. **Configure Claude Desktop**

   Edit the Claude Desktop config file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

3. **Add NeuronDB MCP Server**

   ```json
   {
     "mcpServers": {
       "neurondb": {
         "command": "node",
         "args": [
           "/path/to/Neurondb/bin/mcp/dist/index.js"
         ],
         "env": {
           "PGHOST": "localhost",
           "PGPORT": "5432",
           "PGDATABASE": "postgres",
           "PGUSER": "postgres",
           "PGPASSWORD": "your_password"
         }
       }
     }
   }
   ```

4. **Restart Claude Desktop**

5. **Use in Claude Chat**

   Simply ask Claude:
   - "Train a linear regression model on the sample_train table"
   - "Show me all available ML models"
   - "Make a prediction using model ID 1"
   - "Enable GPU acceleration"

   Claude will automatically use the MCP server tools!

#### Example Conversation

```
You: Train a linear regression model on sample_train table

Claude: I'll train a linear regression model for you.

[Claude uses train_ml_model tool automatically]

Claude: ✅ Model trained successfully! Model ID: 1
       Algorithm: linear_regression
       Training table: sample_train
```

### Method 2: Programmatic Usage (Node.js/Python)

**Best for:** Developers, automation, integration with applications

#### Node.js Example

```javascript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

// Connect to MCP server
const transport = new StdioClientTransport({
  command: "node",
  args: ["/path/to/Neurondb/bin/mcp/dist/index.js"],
  env: {
    PGHOST: "localhost",
    PGPORT: "5432",
    PGDATABASE: "postgres",
    PGUSER: "postgres",
    PGPASSWORD: "your_password"
  }
});

const client = new Client({
  name: "my-app",
  version: "1.0.0"
}, {
  capabilities: {}
});

await client.connect(transport);

// Train a model
const result = await client.callTool({
  name: "train_ml_model",
  arguments: {
    table: "sample_train",
    feature_col: "features",
    label_col: "label",
    algorithm: "linear_regression"
  }
});

console.log(JSON.parse(result.content[0].text));
// { model_id: 1, algorithm: "linear_regression", ... }
```

#### Python Example

```python
import subprocess
import json

# Start MCP server process
process = subprocess.Popen(
    ["node", "/path/to/Neurondb/bin/mcp/dist/index.js"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={
        "PGHOST": "localhost",
        "PGPORT": "5432",
        "PGDATABASE": "postgres",
        "PGUSER": "postgres",
        "PGPASSWORD": "your_password"
    }
)

# Send MCP request
request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "train_ml_model",
        "arguments": {
            "table": "sample_train",
            "feature_col": "features",
            "label_col": "label",
            "algorithm": "linear_regression"
        }
    }
}

process.stdin.write(json.dumps(request).encode() + b"\n")
process.stdin.flush()

# Read response
response = json.loads(process.stdout.readline())
print(response)
```

### Method 3: Direct SQL (No MCP Server)

**Best for:** SQL users, direct database access, scripts

You don't need the MCP server at all! You can use SQL directly:

```sql
-- Train a model
SELECT neurondb.train(
    'linear_regression',
    'sample_train',
    'features',
    'label',
    '{}'::jsonb
) AS model_id;

-- Or use specific function
SELECT train_linear_regression('sample_train', 'features', 'label') AS model_id;

-- Make predictions
SELECT neurondb_predict(1, features::real[]) AS prediction
FROM sample_test;

-- Get model info
SELECT * FROM neurondb.ml_models WHERE model_id = 1;
```

**Advantages:**
- ✅ No additional setup
- ✅ Works with any PostgreSQL client (psql, pgAdmin, DBeaver, etc.)
- ✅ Can be used in stored procedures, triggers, views
- ✅ Direct database access

**Disadvantages:**
- ❌ No AI assistant integration
- ❌ Manual SQL writing required
- ❌ No automatic tool discovery

## Comparison

| Method | Best For | Setup Complexity | AI Integration |
|--------|----------|------------------|----------------|
| **Claude Desktop** | Non-developers, quick access | Easy | ✅ Full |
| **Programmatic** | Developers, automation | Medium | ❌ Manual |
| **Direct SQL** | SQL users, scripts | None | ❌ None |

## Detailed Examples

### Example 1: Using with Claude Desktop

1. **Setup** (one-time):
   ```json
   {
     "mcpServers": {
       "neurondb": {
         "command": "node",
         "args": ["/Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/dist/index.js"]
       }
     }
   }
   ```

2. **Use in chat**:
   ```
   You: Train a random forest model on my training data
   
   Claude: [Automatically uses train_ml_model tool]
          ✅ Model trained! Model ID: 5
   ```

### Example 2: Using in Your Application

```javascript
// In your Node.js application
import { MCPClient } from './mcp-client.js';

const mcp = new MCPClient({
  serverPath: './bin/mcp/dist/index.js',
  dbConfig: {
    host: 'localhost',
    port: 5432,
    database: 'postgres',
    user: 'postgres',
    password: 'password'
  }
});

// Train model
const model = await mcp.trainModel({
  table: 'sample_train',
  feature_col: 'features',
  label_col: 'label',
  algorithm: 'linear_regression'
});

console.log(`Model ${model.model_id} trained!`);

// Make prediction
const prediction = await mcp.predict({
  model_id: model.model_id,
  features: [1.0, 2.0, 3.0, 4.0, 5.0]
});

console.log(`Prediction: ${prediction.prediction}`);
```

### Example 3: Using Direct SQL

```sql
-- psql, pgAdmin, or any SQL client

-- 1. Train model
SELECT neurondb.train(
    'linear_regression',
    'sample_train',
    'features',
    'label',
    '{}'::jsonb
) AS model_id;

-- 2. Check model info
SELECT 
    model_id,
    algorithm,
    metrics::jsonb->>'r_squared' as r_squared,
    metrics::jsonb->>'mse' as mse
FROM neurondb.ml_models
WHERE model_id = 1;

-- 3. Make predictions
SELECT 
    id,
    features,
    label,
    neurondb_predict(1, features::real[]) AS prediction
FROM sample_test
LIMIT 10;

-- 4. Evaluate model
SELECT 
    AVG(ABS(label - neurondb_predict(1, features::real[]))) AS mae,
    SQRT(AVG(POWER(label - neurondb_predict(1, features::real[]), 2))) AS rmse
FROM sample_test;
```

## Configuration

### Environment Variables

The MCP server can be configured via environment variables:

```bash
# Database connection
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=postgres
export PGUSER=postgres
export PGPASSWORD=your_password

# Or use connection string
export DATABASE_URL=postgresql://user:password@localhost:5432/database
```

### Configuration File

Create `mcp-config.json` in the MCP server directory:

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
    "password": "your_password"
  },
  "features": {
    "ml": {
      "enabled": true,
      "gpuEnabled": true
    }
  }
}
```

## Common Use Cases

### 1. Interactive AI Assistant
**Use:** Claude Desktop
- Ask questions in natural language
- Claude uses MCP tools automatically
- No code required

### 2. Application Integration
**Use:** Programmatic (Node.js/Python)
- Integrate ML training into your app
- Automate model training pipelines
- Build custom interfaces

### 3. Data Science Workflow
**Use:** Direct SQL + Jupyter Notebooks
- Use SQL directly in notebooks
- Combine with pandas, numpy
- Full control over queries

### 4. Production Systems
**Use:** Direct SQL + Application Code
- Train models via SQL scripts
- Deploy models in production
- Use in stored procedures

## Troubleshooting

### MCP Server Not Starting

```bash
# Check if built
cd bin/mcp
npm run build

# Test server
node dist/index.js
```

### Connection Issues

```bash
# Test database connection
psql -h localhost -U postgres -d postgres -c "SELECT 1"

# Check environment variables
echo $PGHOST
echo $PGDATABASE
```

### Claude Desktop Not Finding Server

- Check config file path
- Verify server path is absolute
- Check Node.js is in PATH
- Restart Claude Desktop

## Next Steps

- **For Claude Desktop users:** See [Claude Desktop Setup](#claude-desktop-integration)
- **For developers:** See [Programmatic Usage](#programmatic-usage)
- **For SQL users:** See [Direct SQL](#direct-sql-alternative)
- **For training:** See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

