# PostgreSQL 18 Configuration Summary

## ✅ Configuration Complete

The MCP server is now configured to connect to **PostgreSQL 18** on port **5438** using user **ibrarahmed**.

## Current Configuration

### MCP Server Config (`bin/mcp/mcp-config.json`)
```json
{
  "database": {
    "host": "localhost",
    "port": 5438,
    "database": "postgres",
    "user": "ibrarahmed",
    "password": "",
    "pool": {
      "min": 2,
      "max": 10,
      "idleTimeoutMillis": 30000,
      "connectionTimeoutMillis": 5000
    },
    "ssl": false
  }
}
```

### Claude Desktop Config (`claude_desktop_config.json`)
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
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5438",
        "NEURONDB_DATABASE": "postgres",
        "NEURONDB_USER": "ibrarahmed",
        "PATH": "/usr/local/pgsql.18/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
      }
    }
  }
}
```

## ⚠️ Next Step: Start PostgreSQL 18

PostgreSQL 18 is configured but **not currently running**. You need to start it:

### Option 1: Start PostgreSQL 18 Server

```bash
# Start PostgreSQL 18 server
/usr/local/pgsql.18/bin/pg_ctl -D /usr/local/pgsql.18/data -l /usr/local/pgsql.18/data/logfile start
```

### Option 2: Check if it's managed by a service

```bash
# Check for launchd service
launchctl list | grep postgresql18

# Or check for Homebrew service
brew services list | grep postgresql
```

### Option 3: Verify PostgreSQL 18 is ready

After starting, verify it's running:

```bash
/usr/local/pgsql.18/bin/pg_isready -h localhost -p 5438
```

Expected output:
```
localhost:5438 - accepting connections
```

## Verify Connection

Once PostgreSQL 18 is running, test the connection:

```bash
/usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "SELECT version();"
```

Expected output:
```
PostgreSQL 18.0
```

## Install NeuronDB Extension

After PostgreSQL 18 is running, install the NeuronDB extension:

```bash
/usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "CREATE EXTENSION IF NOT EXISTS neurondb;"
```

Verify installation:

```bash
/usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'neurondb';"
```

## Configuration Summary

| Setting | Value |
|---------|-------|
| PostgreSQL Version | 18.0 |
| Installation Path | `/usr/local/pgsql.18/` |
| Host | `localhost` |
| Port | `5438` |
| Database | `postgres` |
| User | `ibrarahmed` |
| Password | (empty - uses peer authentication) |
| Config File | `/Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/mcp-config.json` |
| Claude Config | `~/Library/Application Support/Claude/claude_desktop_config.json` |

## Troubleshooting

### Issue: "Connection refused" on port 5438

**Solution:** Start PostgreSQL 18 server (see above)

### Issue: "Database does not exist"

**Solution:** Create the database:
```bash
/usr/local/pgsql.18/bin/createdb -h localhost -p 5438 -U ibrarahmed postgres
```

### Issue: "Extension neurondb does not exist"

**Solution:** Install NeuronDB extension (see above)

### Issue: "Password authentication failed"

**Solution:** The configuration uses peer authentication (no password). If you need password authentication, update `mcp-config.json`:
```json
{
  "database": {
    "password": "your_password_here"
  }
}
```

## After Starting PostgreSQL 18

1. ✅ Verify PostgreSQL 18 is running on port 5438
2. ✅ Install NeuronDB extension
3. ✅ Restart Claude Desktop
4. ✅ Test MCP server connection

## Quick Test Commands

```bash
# Check if PostgreSQL 18 is running
/usr/local/pgsql.18/bin/pg_isready -h localhost -p 5438

# Connect to PostgreSQL 18
/usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres

# Check version
/usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "SELECT version();"

# Check NeuronDB extension
/usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'neurondb';"
```

