# NeuronDB MCP Server Troubleshooting Guide

## Current Status

✅ **MCP Server File**: Exists and valid  
✅ **Node.js**: Working (v24.6.0)  
✅ **Claude Desktop Config**: Copied to correct location  
❌ **PostgreSQL 18**: Not running on port 5438  

## Quick Fix: Start PostgreSQL 18

### Option 1: Use the Start Script
```bash
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
./start_postgresql18.sh
```

### Option 2: Manual Start
```bash
/usr/local/pgsql.18/bin/pg_ctl -D /usr/local/pgsql.18/data -l /usr/local/pgsql.18/data/logfile start
```

### Option 3: Check if it's a Service
```bash
# Check for launchd service
launchctl list | grep postgresql18

# Or check Homebrew services
brew services list | grep postgresql
```

## Verify Connection

After starting PostgreSQL 18:

```bash
# Check if it's running
/usr/local/pgsql.18/bin/pg_isready -h localhost -p 5438

# Test connection
/usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "SELECT version();"
```

Expected output:
```
PostgreSQL 18.0
```

## Install NeuronDB Extension

```bash
/usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "CREATE EXTENSION IF NOT EXISTS neurondb;"
```

## Test MCP Server

### 1. Test Server Starts
```bash
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
node dist/index.js
```
Press Ctrl+C after a few seconds. Should start without errors.

### 2. Test Tools List
```bash
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | node dist/index.js 2>&1 | head -50
```

This should return a JSON list of available tools.

## Claude Desktop Connection

### 1. Verify Config Location
```bash
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

Should show your NeuronDB MCP server configuration.

### 2. Check Claude Desktop Logs
```bash
tail -f ~/Library/Logs/Claude/*.log
```

Look for:
- `neurondb` MCP server messages
- Connection errors
- Database errors

### 3. Restart Claude Desktop
After making changes:
1. Quit Claude Desktop completely
2. Restart it
3. Check if MCP server connects

## Common Issues

### Issue: "Connection refused" on port 5438
**Solution:** Start PostgreSQL 18 (see above)

### Issue: "Database connection failed" in MCP logs
**Solutions:**
1. Verify PostgreSQL 18 is running: `/usr/local/pgsql.18/bin/pg_isready -h localhost -p 5438`
2. Check user permissions
3. Verify database exists: `/usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "SELECT 1;"`

### Issue: "Extension neurondb does not exist"
**Solution:** Install the extension (see above)

### Issue: "Cannot find module" when starting MCP server
**Solution:** Install dependencies:
```bash
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
npm install
```

### Issue: MCP server not showing tools in Claude
**Solutions:**
1. Restart Claude Desktop completely
2. Check logs for errors
3. Verify the config file is correct
4. Test MCP server manually (see above)

## Expected Behavior

When everything is working:

1. **PostgreSQL 18** is running on port 5438
2. **NeuronDB extension** is installed
3. **MCP server** starts without errors
4. **Claude Desktop** shows NeuronDB tools available

## Test in Claude Desktop

Once connected, try asking Claude:

1. "What NeuronDB tools are available?"
2. "List all available tools"
3. "Train a linear regression model..."
4. "Rerank these documents using LLM..."

## Configuration Summary

| Setting | Value | Status |
|---------|-------|--------|
| MCP Server | `/Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/dist/index.js` | ✅ |
| PostgreSQL Host | `localhost` | ✅ |
| PostgreSQL Port | `5438` | ❌ (not running) |
| PostgreSQL User | `ibrarahmed` | ✅ |
| Database | `postgres` | ✅ |
| Config Location | `~/Library/Application Support/Claude/claude_desktop_config.json` | ✅ |

## Next Steps

1. ✅ Start PostgreSQL 18
2. ✅ Install NeuronDB extension
3. ✅ Test MCP server manually
4. ✅ Restart Claude Desktop
5. ✅ Verify tools are available

