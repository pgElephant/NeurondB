# Testing NeuronDB MCP Server Connection

## Quick Test Checklist

### 1. Verify MCP Server File Exists
```bash
ls -la /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/dist/index.js
```

### 2. Test MCP Server Can Start
```bash
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
node dist/index.js
```
Press Ctrl+C after a few seconds. If it starts without errors, it's working.

### 3. Verify PostgreSQL 18 is Running
```bash
/usr/local/pgsql.18/bin/pg_isready -h localhost -p 5438
```

Expected: `localhost:5438 - accepting connections`

### 4. Test PostgreSQL Connection
```bash
/usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "SELECT version();"
```

### 5. Check Claude Desktop Config Location
The config file should be at:
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

Copy your config there:
```bash
mkdir -p ~/Library/Application\ Support/Claude
cp /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/claude_desktop_config.json \
   ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

### 6. Check Claude Desktop Logs
```bash
tail -f ~/Library/Logs/Claude/*.log
```

Look for:
- MCP server startup messages
- Connection errors
- Database connection errors

### 7. Verify Node.js is Available
```bash
which node
node --version
```

Should show Node.js path and version.

## Common Issues

### Issue: "Cannot find module"
**Solution:** Ensure all dependencies are installed:
```bash
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
npm install
```

### Issue: "Database connection failed"
**Solution:** 
1. Verify PostgreSQL 18 is running on port 5438
2. Check user permissions
3. Verify database exists

### Issue: "MCP server not showing tools"
**Solution:**
1. Restart Claude Desktop completely
2. Check logs for errors
3. Verify the config file is in the correct location
4. Ensure the MCP server starts without errors

## Manual MCP Server Test

Test the MCP server manually with a simple request:

```bash
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | node dist/index.js
```

This should return a list of available tools.

## Expected Tools

When connected, Claude should have access to these NeuronDB tools:

### Vector Tools
- Vector search
- Vector operations
- Index management

### ML Tools
- Model training
- Model prediction
- Model management

### Reranking Tools (7 total)
- mmr_rerank
- rerank_cross_encoder
- rerank_llm
- rerank_colbert
- rerank_rrf
- rerank_ensemble_weighted
- rerank_ensemble_borda

### Other Tools
- RAG tools
- GPU tools
- Analytics tools
- And more...

## Debugging Steps

1. **Check if server starts:**
   ```bash
   cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
   node dist/index.js 2>&1 | head -20
   ```

2. **Check PostgreSQL connection:**
   ```bash
   /usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "SELECT 1;"
   ```

3. **Verify config file:**
   ```bash
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | jq .
   ```

4. **Check Claude Desktop logs:**
   ```bash
   ls -la ~/Library/Logs/Claude/
   tail -50 ~/Library/Logs/Claude/*.log | grep -i "mcp\|neurondb\|error"
   ```

## Next Steps After Fixing

Once the connection works:
1. Ask Claude: "What NeuronDB tools are available?"
2. Try: "List all available tools"
3. Test a tool: "Train a linear regression model..."

