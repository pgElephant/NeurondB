#!/bin/bash

# Build, install, restart PostgreSQL, and run a test SQL file
# Usage: ./build_and_test.sh <sql_filename>
# Example: ./build_and_test.sh 013_gmm_basic.sql

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Exit if an uninitialized variable is used

if [ $# -eq 0 ]; then
    echo "Usage: $0 <sql_filename>"
    echo "Example: $0 013_gmm_basic.sql"
    exit 1
fi

SQL_FILE="$1"

# Check if SQL file exists
if [ ! -f "tests/sql/basic/$SQL_FILE" ]; then
    echo "Error: SQL file 'tests/sql/basic/$SQL_FILE' not found"
    exit 1
fi

echo "Cleaning build..."
if ! make clean; then
    echo "Error: make clean failed"
    exit 1
fi

echo "Building with 12 jobs..."
if ! make -j12; then
    echo "Error: make -j12 failed"
    exit 1
fi

echo "Installing..."
if ! make install; then
    echo "Error: make install failed"
    exit 1
fi

echo "Restarting PostgreSQL..."
if ! pg_ctl restart -l pg.log; then
    echo "Error: pg_ctl restart failed"
    exit 1
fi

echo "Waiting for PostgreSQL to start..."
sleep 2

echo "Running SQL test: tests/sql/basic/$SQL_FILE"
if ! psql neurondb -f "tests/sql/basic/$SQL_FILE"; then
    echo "Error: SQL test failed"
    exit 1
fi

echo "Done!"
exit 0

