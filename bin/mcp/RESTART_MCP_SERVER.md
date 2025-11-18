# How to Restart the MCP Server

## Automatic Restart (Recommended)

The MCP server is managed by Claude Desktop and will automatically restart when:
1. **Claude Desktop restarts** - The MCP server will be launched automatically
2. **New conversation starts** - Claude Desktop reconnects to the MCP server

## Manual Restart Methods

### Method 1: Restart Claude Desktop (Easiest)
1. Quit Claude Desktop completely (⌘Q on Mac)
2. Reopen Claude Desktop
3. The MCP server will start automatically

### Method 2: Kill and Let Claude Desktop Restart
```bash
# Kill any running MCP server processes
pkill -f "node.*dist/index.js"

# Claude Desktop will automatically restart it when needed
```

### Method 3: Test MCP Server Manually
```bash
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
node test_mcp_server.js
```

## Verify MCP Server is Running

### Check Process
```bash
ps aux | grep "node.*dist/index.js" | grep -v grep
```

### Test Connection
```bash
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
node test_mcp_server.js
```

Expected output:
```
✅ MCP server module imported successfully
✅ Database connection successful
✅ All tests passed! MCP server should work.
```

## Current Configuration

- **Port**: 5438
- **Host**: localhost
- **Database**: postgres
- **User**: ibrarahmed
- **Config File**: `/Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/mcp-config.json`
- **Claude Config**: `~/Library/Application Support/Claude/claude_desktop_config.json`

## Troubleshooting

If the MCP server doesn't connect:

1. **Check PostgreSQL is running on port 5438:**
   ```bash
   /usr/local/pgsql.18/bin/pg_isready -h localhost -p 5438
   ```

2. **Check Claude Desktop logs:**
   ```bash
   tail -f ~/Library/Logs/Claude/*.log
   ```

3. **Verify configuration:**
   ```bash
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

4. **Test MCP server directly:**
   ```bash
   cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
   node test_mcp_server.js
   ```

## Note

The MCP server runs via stdio (standard input/output) and is managed by Claude Desktop. It's not a standalone server that runs in the background - it's spawned by Claude Desktop when needed.

