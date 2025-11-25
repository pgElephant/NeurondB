#!/bin/bash
# Coverage Setup Script for NeuronDB
# Compiles NeuronDB with coverage flags for gcov analysis

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "NeuronDB Coverage Setup"
echo "======================"
echo ""

# Check for pg_config
if [ -z "$PG_CONFIG" ]; then
	if command -v pg_config >/dev/null 2>&1; then
		PG_CONFIG="pg_config"
	else
		echo "ERROR: pg_config not found. Please set PG_CONFIG environment variable."
		exit 1
	fi
fi

echo "Using pg_config: $PG_CONFIG"
echo "Project root: $PROJECT_ROOT"
echo ""

# Clean previous build
echo "Cleaning previous build..."
cd "$PROJECT_ROOT"
make clean PG_CONFIG="$PG_CONFIG" || true

# Compile with coverage flags
echo ""
echo "Compiling with coverage flags..."
echo "  CFLAGS: -fprofile-arcs -ftest-coverage"
echo "  LDFLAGS: -lgcov"
echo ""

make PG_CONFIG="$PG_CONFIG" \
	CFLAGS="-fprofile-arcs -ftest-coverage" \
	LDFLAGS="-lgcov" \
	all

echo ""
echo "✓ Build completed with coverage flags"
echo ""
echo "Next steps:"
echo "  1. Install the extension: sudo make install PG_CONFIG=\"$PG_CONFIG\""
echo "  2. Run tests: python3 tests/run_test.py --category all"
echo "  3. View coverage report: open tests/coverage/coverage.html"





