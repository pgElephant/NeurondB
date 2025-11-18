# PostgreSQL 18 Configuration for NeuronDB MCP Server

## Overview

This guide ensures the NeuronDB MCP server correctly connects to PostgreSQL 18 installed at `/usr/local/pgsql.18/`.

## Current Setup

- **PostgreSQL Version**: 18.0
- **Installation Path**: `/usr/local/pgsql.18/`
- **Connection**: `localhost:5432`
- **Default Database**: `postgres`
- **Default User**: `postgres`

## Configuration Files

### 1. MCP Server Configuration (`bin/mcp/mcp-config.json`)

The MCP server configuration file is located at:
```
/Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/mcp-config.json
```

**Current Configuration:**
```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
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

**Notes:**
- `password: ""` means it will use PostgreSQL peer authentication (your OS user)
- If you need password authentication, set the password here or use environment variables
- Connection timeout is set to 5 seconds for better reliability

### 2. Claude Desktop Configuration

The Claude Desktop configuration is at:
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Current Configuration:**
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
        "NEURONDB_PORT": "5432",
        "NEURONDB_DATABASE": "postgres",
        "NEURONDB_USER": "postgres",
        "PATH": "/usr/local/pgsql.18/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
      }
    }
  }
}
```

**Important:** The `PATH` environment variable includes `/usr/local/pgsql.18/bin` to ensure PostgreSQL 18 tools are available if needed.

## Connection Methods

The MCP server supports multiple ways to configure the database connection:

### Method 1: Configuration File (Current Setup)

The `mcp-config.json` file is automatically loaded. This is the recommended method.

### Method 2: Environment Variables

Environment variables in `claude_desktop_config.json` will override config file settings:

- `NEURONDB_HOST` - Database host (default: localhost)
- `NEURONDB_PORT` - Database port (default: 5432)
- `NEURONDB_DATABASE` - Database name (default: postgres)
- `NEURONDB_USER` - Database user (default: postgres)
- `NEURONDB_PASSWORD` - Database password (optional)
- `NEURONDB_CONNECTION_STRING` - Full connection string (overrides all above)

### Method 3: Connection String

You can use a PostgreSQL connection string:

```json
{
  "database": {
    "connectionString": "postgresql://postgres@localhost:5432/postgres"
  }
}
```

## Verification

### 1. Test PostgreSQL Connection

```bash
/usr/local/pgsql.18/bin/psql -h localhost -p 5432 -U postgres -d postgres -c "SELECT version();"
```

Expected output:
```
PostgreSQL 18.0
```

### 2. Verify NeuronDB Extension

```bash
/usr/local/pgsql.18/bin/psql -h localhost -p 5432 -U postgres -d postgres -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'neurondb';"
```

If the extension is not installed:
```sql
CREATE EXTENSION IF NOT EXISTS neurondb;
```

### 3. Test MCP Server Connection

1. Restart Claude Desktop
2. Check Claude Desktop logs: `~/Library/Logs/Claude/`
3. Try asking Claude: "List all NeuronDB tools"

## Authentication Methods

### Peer Authentication (Current Setup)

With `password: ""` or no password set, PostgreSQL uses peer authentication:
- The MCP server connects as the OS user running Claude Desktop
- No password required
- Works if your OS username matches a PostgreSQL role

### Password Authentication

If you need password authentication:

1. **Set password in config file:**
   ```json
   {
     "database": {
       "password": "your_password_here"
     }
   }
   ```

2. **Or use environment variable:**
   ```json
   {
     "env": {
       "NEURONDB_PASSWORD": "your_password_here"
     }
   }
   ```

3. **Or use connection string:**
   ```json
   {
     "database": {
       "connectionString": "postgresql://postgres:password@localhost:5432/postgres"
     }
   }
   ```

## Troubleshooting

### Issue: "Connection refused"

**Check PostgreSQL is running:**
```bash
/usr/local/pgsql.18/bin/pg_isready -h localhost -p 5432
```

**Start PostgreSQL if needed:**
```bash
# Check how PostgreSQL is managed (Homebrew, launchd, etc.)
brew services list | grep postgresql
# or
launchctl list | grep postgres
```

### Issue: "Password authentication failed"

**Solutions:**
1. Set the correct password in `mcp-config.json`
2. Or use peer authentication (ensure OS user matches PostgreSQL role)
3. Check `pg_hba.conf` for authentication method

### Issue: "Database does not exist"

**Create the database:**
```bash
/usr/local/pgsql.18/bin/createdb -h localhost -p 5432 -U postgres your_database_name
```

### Issue: "Extension neurondb does not exist"

**Install NeuronDB extension:**
```sql
CREATE EXTENSION IF NOT EXISTS neurondb;
```

**Verify installation:**
```sql
SELECT neurondb_version();
```

### Issue: "Connection timeout"

**Increase timeout:**
```json
{
  "database": {
    "pool": {
      "connectionTimeoutMillis": 10000
    }
  }
}
```

## Best Practices

1. **Use configuration file** - Easier to manage than environment variables
2. **Set appropriate pool size** - Match your PostgreSQL `max_connections`
3. **Use connection timeout** - Prevents hanging connections
4. **Monitor connections** - Check `pg_stat_activity` view
5. **Use SSL in production** - Always use SSL for remote connections

## Connection Pool Settings

Current settings:
- **min**: 2 connections (always available)
- **max**: 10 connections (maximum pool size)
- **idleTimeoutMillis**: 30000 (30 seconds)
- **connectionTimeoutMillis**: 5000 (5 seconds)

**Adjust based on:**
- PostgreSQL `max_connections` setting
- Expected concurrent MCP requests
- Available system resources

## Next Steps

1. ✅ Configuration files created
2. ✅ PostgreSQL 18 connection verified
3. ⏭️ Restart Claude Desktop
4. ⏭️ Test MCP server connection
5. ⏭️ Verify all NeuronDB tools work

## Quick Reference

**PostgreSQL 18 Path**: `/usr/local/pgsql.18/`
**Connection**: `localhost:5432`
**Database**: `postgres`
**User**: `postgres`
**Config File**: `/Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/mcp-config.json`
**Claude Config**: `~/Library/Application Support/Claude/claude_desktop_config.json`

