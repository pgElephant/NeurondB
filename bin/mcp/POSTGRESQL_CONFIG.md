# PostgreSQL Configuration for NeuronDB MCP Server

## Overview

The NeuronDB MCP server uses the `pg` (node-postgres) library to connect to PostgreSQL. The configuration supports multiple methods for setting database connection parameters.

## Configuration Methods

### Method 1: Configuration File (Recommended)

Create `mcp-config.json` in the MCP server directory:

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
      "connectionTimeoutMillis": 2000
    },
    "ssl": false
  }
}
```

**Config File Search Order:**
1. Path specified in `load(configPath)`
2. `process.env.NEURONDB_MCP_CONFIG`
3. `./mcp-config.json` (current working directory)
4. `bin/mcp/mcp-config.json` (MCP server directory)
5. `~/.neurondb/mcp-config.json` (user home directory)

### Method 2: Environment Variables

Set PostgreSQL connection via environment variables:

```bash
export NEURONDB_HOST=localhost
export NEURONDB_PORT=5432
export NEURONDB_DATABASE=postgres
export NEURONDB_USER=postgres
export NEURONDB_PASSWORD=your_password_here
```

Or use a connection string:

```bash
export NEURONDB_CONNECTION_STRING="postgresql://user:password@localhost:5432/database"
```

### Method 3: Claude Desktop Environment Variables

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "node",
      "args": [
        "/Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp/dist/index.js"
      ],
      "env": {
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5432",
        "NEURONDB_DATABASE": "postgres",
        "NEURONDB_USER": "postgres",
        "NEURONDB_PASSWORD": "your_password_here"
      }
    }
  }
}
```

## Default Values

If no configuration is provided, the MCP server uses these defaults:

```javascript
{
  host: "localhost",
  port: 5432,
  database: "postgres",
  user: "postgres",
  password: undefined  // Uses PostgreSQL peer authentication if not set
}
```

## PostgreSQL-Specific Settings

### Connection Pooling

The MCP server uses connection pooling for better performance:

```json
{
  "pool": {
    "min": 2,              // Minimum connections in pool
    "max": 10,             // Maximum connections in pool
    "idleTimeoutMillis": 30000,        // Close idle connections after 30s
    "connectionTimeoutMillis": 2000    // Connection timeout: 2s
  }
}
```

**Recommended Settings:**
- **Development**: min: 2, max: 5
- **Production**: min: 5, max: 20
- Adjust based on your PostgreSQL `max_connections` setting

### SSL Configuration

For secure connections, configure SSL:

```json
{
  "database": {
    "ssl": {
      "rejectUnauthorized": true,
      "ca": "/path/to/ca-certificate.crt",
      "key": "/path/to/client-key.key",
      "cert": "/path/to/client-certificate.crt"
    }
  }
}
```

For local development (no SSL):

```json
{
  "database": {
    "ssl": false
  }
}
```

### Connection String Format

You can use a PostgreSQL connection string instead of individual parameters:

```json
{
  "database": {
    "connectionString": "postgresql://user:password@localhost:5432/database?sslmode=require"
  }
}
```

Connection string format:
```
postgresql://[user[:password]@][host][:port][/database][?param1=value1&...]
```

## Verification

### Test Database Connection

1. **Check PostgreSQL is running:**
   ```bash
   pg_isready -h localhost -p 5432
   ```

2. **Test connection manually:**
   ```bash
   psql -h localhost -p 5432 -U postgres -d postgres
   ```

3. **Verify NeuronDB extension:**
   ```sql
   CREATE EXTENSION IF NOT EXISTS neurondb;
   SELECT neurondb_version();
   ```

### Test MCP Server Connection

1. **Check MCP server logs** (if logging is enabled)
2. **Test a simple tool** through Claude Desktop:
   - "List all NeuronDB tools"
   - "What is the NeuronDB version?"

## Common Issues

### Issue: "Database not connected"

**Solution:**
- Verify PostgreSQL is running
- Check connection parameters (host, port, user, password)
- Ensure `neurondb` extension is installed
- Check firewall settings

### Issue: "password authentication failed"

**Solutions:**
1. **Use password in config:**
   ```json
   {
     "database": {
       "password": "correct_password"
     }
   }
   ```

2. **Use peer authentication** (leave password empty/undefined):
   ```json
   {
     "database": {
       "user": "your_os_username"
     }
   }
   ```

3. **Check `pg_hba.conf`** for authentication method

### Issue: "Connection timeout"

**Solutions:**
1. **Increase timeout:**
   ```json
   {
     "database": {
       "pool": {
         "connectionTimeoutMillis": 10000
       }
     }
   }
   ```

2. **Check PostgreSQL is listening:**
   ```bash
   netstat -an | grep 5432
   # or
   lsof -i :5432
   ```

### Issue: "Too many connections"

**Solutions:**
1. **Reduce pool size:**
   ```json
   {
     "database": {
       "pool": {
         "max": 5
       }
     }
   }
   ```

2. **Check PostgreSQL max_connections:**
   ```sql
   SHOW max_connections;
   ```

## Best Practices

1. **Use connection pooling** - Always configure pool settings
2. **Use environment variables for passwords** - Don't commit passwords to config files
3. **Set appropriate timeouts** - Balance between responsiveness and reliability
4. **Monitor connection usage** - Check PostgreSQL `pg_stat_activity` view
5. **Use SSL in production** - Always use SSL for remote connections

## Example: Complete Production Configuration

```json
{
  "database": {
    "host": "db.example.com",
    "port": 5432,
    "database": "neurondb_prod",
    "user": "neurondb_user",
    "password": "${NEURONDB_PASSWORD}",
    "pool": {
      "min": 5,
      "max": 20,
      "idleTimeoutMillis": 30000,
      "connectionTimeoutMillis": 5000
    },
    "ssl": {
      "rejectUnauthorized": true
    }
  },
  "logging": {
    "level": "warn",
    "format": "json"
  }
}
```

## Next Steps

1. Create `mcp-config.json` with your PostgreSQL settings
2. Test the connection
3. Verify all NeuronDB tools work correctly
4. Monitor connection pool usage

