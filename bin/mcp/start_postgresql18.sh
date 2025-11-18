#!/bin/bash
# Start PostgreSQL 18 for NeuronDB MCP Server

echo "Starting PostgreSQL 18..."

# Check if PostgreSQL 18 is already running
if /usr/local/pgsql.18/bin/pg_isready -h localhost -p 5438 > /dev/null 2>&1; then
    echo "✅ PostgreSQL 18 is already running on port 5438"
    exit 0
fi

# Try to start PostgreSQL 18
PGDATA="/usr/local/pgsql.18/data"
LOGFILE="/usr/local/pgsql.18/data/logfile"

if [ ! -d "$PGDATA" ]; then
    echo "❌ PostgreSQL 18 data directory not found at $PGDATA"
    echo "Please initialize PostgreSQL 18 first:"
    echo "  /usr/local/pgsql.18/bin/initdb -D $PGDATA"
    exit 1
fi

# Start PostgreSQL 18
echo "Starting PostgreSQL 18 server..."
/usr/local/pgsql.18/bin/pg_ctl -D "$PGDATA" -l "$LOGFILE" start

# Wait a moment for server to start
sleep 2

# Check if it's running
if /usr/local/pgsql.18/bin/pg_isready -h localhost -p 5438 > /dev/null 2>&1; then
    echo "✅ PostgreSQL 18 started successfully on port 5438"
    
    # Check if NeuronDB extension exists
    echo "Checking NeuronDB extension..."
    /usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'neurondb';" 2>&1 | grep -q neurondb
    
    if [ $? -ne 0 ]; then
        echo "⚠️  NeuronDB extension not found. Installing..."
        /usr/local/pgsql.18/bin/psql -h localhost -p 5438 -U ibrarahmed -d postgres -c "CREATE EXTENSION IF NOT EXISTS neurondb;" 2>&1
        echo "✅ NeuronDB extension installed"
    else
        echo "✅ NeuronDB extension is installed"
    fi
    
    echo ""
    echo "PostgreSQL 18 is ready for MCP server!"
    echo "Next: Restart Claude Desktop to connect the MCP server"
else
    echo "❌ Failed to start PostgreSQL 18"
    echo "Check the log file: $LOGFILE"
    exit 1
fi

