#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PG_BIN="/usr/local/pgsql.17/bin"
PG_DATA="/Users/pgedge/pge/neurondb_data17"
LOG_FILE="/Users/pgedge/pge/neurondb_test.log"

echo "=========================================="
echo "NeurondB PostgreSQL 17 Worker Test"
echo "=========================================="
echo ""

# Function to check if postgres is running
check_postgres_running() {
    if $PG_BIN/pg_ctl status -D $PG_DATA > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to check workers
check_workers() {
    echo -e "${YELLOW}Checking NeurondB workers...${NC}"
    $PG_BIN/psql -d postgres -c "SELECT pid, backend_type, state FROM pg_stat_activity WHERE backend_type LIKE '%worker%' OR backend_type LIKE '%NeurondB%' ORDER BY backend_type;" 2>/dev/null
    
    # Check for background workers in logs
    echo ""
    echo -e "${YELLOW}Checking for worker processes...${NC}"
    ps aux | grep -i "postgres.*worker" | grep -v grep
    
    echo ""
    echo -e "${YELLOW}Recent log entries about workers:${NC}"
    if [ -f "$LOG_FILE" ]; then
        tail -50 "$LOG_FILE" | grep -i "worker" || echo "No worker-related log entries found"
    else
        echo "Log file not found at $LOG_FILE"
    fi
}

# Step 1: Check current status
echo -e "${YELLOW}Step 1: Checking current PostgreSQL status...${NC}"
if check_postgres_running; then
    echo -e "${GREEN}PostgreSQL is currently running${NC}"
    NEED_TO_START=false
else
    echo -e "${YELLOW}PostgreSQL is not running${NC}"
    NEED_TO_START=true
fi
echo ""

# Step 2: Stop if running
if ! $NEED_TO_START; then
    echo -e "${YELLOW}Step 2: Stopping PostgreSQL...${NC}"
    $PG_BIN/pg_ctl stop -D $PG_DATA -m fast
    sleep 2
    
    if check_postgres_running; then
        echo -e "${RED}FAILED: PostgreSQL is still running after stop${NC}"
        exit 1
    else
        echo -e "${GREEN}SUCCESS: PostgreSQL stopped successfully${NC}"
    fi
    echo ""
fi

# Step 3: Start PostgreSQL
echo -e "${YELLOW}Step 3: Starting PostgreSQL...${NC}"
$PG_BIN/pg_ctl start -D $PG_DATA -l $LOG_FILE

sleep 3

if check_postgres_running; then
    echo -e "${GREEN}SUCCESS: PostgreSQL started successfully${NC}"
else
    echo -e "${RED}FAILED: PostgreSQL did not start${NC}"
    echo "Check log file: $LOG_FILE"
    exit 1
fi
echo ""

# Step 4: Check workers
echo -e "${YELLOW}Step 4: Checking all workers...${NC}"
check_workers
echo ""

# Step 5: Verify NeurondB extension is loaded
echo -e "${YELLOW}Step 5: Verifying NeurondB extension...${NC}"
$PG_BIN/psql -d postgres -c "SHOW shared_preload_libraries;" 2>/dev/null
$PG_BIN/psql -d postgres -c "SELECT * FROM pg_available_extensions WHERE name = 'neurondb';" 2>/dev/null
echo ""

# Step 6: Check for any errors in logs
echo -e "${YELLOW}Step 6: Checking for errors in logs...${NC}"
if [ -f "$LOG_FILE" ]; then
    ERRORS=$(tail -100 "$LOG_FILE" | grep -i "error\|fatal\|panic" | grep -v "ERROR:  relation" | tail -10)
    if [ -n "$ERRORS" ]; then
        echo -e "${RED}Recent errors found:${NC}"
        echo "$ERRORS"
    else
        echo -e "${GREEN}No critical errors found${NC}"
    fi
else
    echo -e "${YELLOW}Log file not found${NC}"
fi
echo ""

# Step 7: Stop PostgreSQL again
echo -e "${YELLOW}Step 7: Testing graceful shutdown...${NC}"
$PG_BIN/pg_ctl stop -D $PG_DATA -m smart
sleep 2

if check_postgres_running; then
    echo -e "${RED}FAILED: PostgreSQL is still running after stop${NC}"
    exit 1
else
    echo -e "${GREEN}SUCCESS: PostgreSQL stopped successfully${NC}"
fi
echo ""

# Step 8: Final restart
echo -e "${YELLOW}Step 8: Final restart test...${NC}"
$PG_BIN/pg_ctl start -D $PG_DATA -l $LOG_FILE
sleep 3

if check_postgres_running; then
    echo -e "${GREEN}SUCCESS: PostgreSQL restarted successfully${NC}"
else
    echo -e "${RED}FAILED: PostgreSQL did not restart${NC}"
    exit 1
fi
echo ""

# Final worker check
echo -e "${YELLOW}Final worker status check...${NC}"
check_workers
echo ""

echo "=========================================="
echo -e "${GREEN}All tests completed successfully!${NC}"
echo "=========================================="
echo ""
echo "PostgreSQL is currently running."
echo "To stop: $PG_BIN/pg_ctl stop -D $PG_DATA"
