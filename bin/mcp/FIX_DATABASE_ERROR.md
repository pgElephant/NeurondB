# Fix: Database "ibrarahmed" does not exist

## Issue
Error: `FATAL: database "ibrarahmed" does not exist`

## Root Cause
The MCP server is trying to connect to database "ibrarahmed" instead of "postgres". This can happen when:
1. Claude Desktop has cached the old configuration
2. Environment variables are not being read correctly
3. PostgreSQL defaults to username when database is not specified

## Solution

### Step 1: Verify Configuration
Both config files are correctly set to use "postgres":
- ✅ `mcp-config.json`: `"database": "postgres"`
- ✅ `claude_desktop_config.json`: `"NEURONDB_DATABASE": "postgres"`

### Step 2: Restart Claude Desktop
1. **Quit Claude Desktop completely** (⌘Q on Mac)
2. **Wait a few seconds**
3. **Reopen Claude Desktop**
4. The MCP server will restart with the correct configuration

### Step 3: Verify Connection
After restarting, test by asking Claude:
- "What NeuronDB tools are available?"
- "List all available tools"

## Current Configuration

```json
{
  "database": {
    "host": "localhost",
    "port": 5438,
    "database": "postgres",  ✅ Correct
    "user": "ibrarahmed",
    "password": ""
  }
}
```

## Why This Happens

PostgreSQL has a default behavior: if no database is specified in the connection, it tries to connect to a database with the same name as the username. Since:
- Username: `ibrarahmed`
- Database should be: `postgres`

If the database name isn't being passed correctly, PostgreSQL defaults to `ibrarahmed`.

## Verification Commands

```bash
# Check config
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | grep NEURONDB_DATABASE

# Test connection manually
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
node test_mcp_server.js

# Verify postgres database exists
/usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "SELECT current_database();"
```

## If Problem Persists

1. **Check Claude Desktop logs:**
   ```bash
   tail -f ~/Library/Logs/Claude/*.log
   ```

2. **Verify environment variables are set:**
   The MCP server should see `NEURONDB_DATABASE=postgres` in its environment

3. **Try using connection string instead:**
   Add to `claude_desktop_config.json`:
   ```json
   "NEURONDB_CONNECTION_STRING": "postgresql://ibrarahmed@localhost:5438/postgres"
   ```

