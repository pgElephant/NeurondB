#!/bin/bash
# API key generator utility for NeuronAgent

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/../agent-server"

DB_NAME="${DB_NAME:-neurondb}"
DB_USER="${DB_USER:-postgres}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

ORGANIZATION_ID="${1:-default}"
USER_ID="${2:-default}"
RATE_LIMIT="${3:-60}"
ROLES="${4:-user}"

echo "Generating API key..."
echo "Organization: $ORGANIZATION_ID"
echo "User: $USER_ID"
echo "Rate Limit: $RATE_LIMIT/min"
echo "Roles: $ROLES"

# Generate key using Go program
go run -tags tools "$SCRIPT_DIR/../cmd/generate-key/main.go" \
    -org "$ORGANIZATION_ID" \
    -user "$USER_ID" \
    -rate "$RATE_LIMIT" \
    -roles "$ROLES" \
    -db-host "$DB_HOST" \
    -db-port "$DB_PORT" \
    -db-name "$DB_NAME" \
    -db-user "$DB_USER"

