#!/bin/bash
# Coverage Verification Script for NeuronDB
# Verifies that all source files have test coverage and identifies gaps

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COVERAGE_DIR="$SCRIPT_DIR/coverage"

echo "NeuronDB Coverage Verification"
echo "==============================="
echo ""

# Check if coverage directory exists
if [ ! -d "$COVERAGE_DIR" ]; then
	echo "ERROR: Coverage directory not found: $COVERAGE_DIR"
	echo "Please run tests with coverage enabled first:"
	echo "  python3 tests/run_test.py --category all"
	exit 1
fi

# Check for gcovr
if ! command -v gcovr &> /dev/null; then
	echo "ERROR: gcovr not found. Install with: pip install gcovr"
	exit 1
fi

echo "Generating coverage report..."
cd "$PROJECT_ROOT"

# Generate coverage report
gcovr -r "$PROJECT_ROOT" \
	--filter "src/" \
	--exclude ".*/tests/.*" \
	--exclude ".*/test/.*" \
	--exclude ".*/tools/.*" \
	--exclude ".*/dataset/.*" \
	--exclude ".*/demo/.*" \
	--exclude ".*/build/.*" \
	--exclude ".*/node_modules/.*" \
	--exclude ".*/venv/.*" \
	--exclude ".*/\.git/.*" \
	--exclude-unreachable-branches \
	--exclude-throw-branches \
	--print-summary \
	--sort-percentage \
	--sort-uncovered \
	--sort-reverse \
	-o "$COVERAGE_DIR/coverage_summary.txt"

# Extract files with < 100% coverage
echo ""
echo "Files with < 100% line coverage:"
echo "---------------------------------"
grep -E "^\s+src/" "$COVERAGE_DIR/coverage_summary.txt" | \
	awk '{if ($2 < 100.0) print $0}' | \
	head -20

# Count total files
TOTAL_FILES=$(grep -E "^\s+src/" "$COVERAGE_DIR/coverage_summary.txt" | wc -l)
LOW_COVERAGE=$(grep -E "^\s+src/" "$COVERAGE_DIR/coverage_summary.txt" | \
	awk '{if ($2 < 100.0) print $0}' | wc -l)

echo ""
echo "Coverage Summary:"
echo "  Total files: $TOTAL_FILES"
echo "  Files with < 100% coverage: $LOW_COVERAGE"
echo "  Files with 100% coverage: $((TOTAL_FILES - LOW_COVERAGE))"

# Generate HTML report
echo ""
echo "Generating HTML coverage report..."
gcovr -r "$PROJECT_ROOT" \
	--filter "src/" \
	--exclude ".*/tests/.*" \
	--exclude ".*/test/.*" \
	--exclude ".*/tools/.*" \
	--exclude ".*/dataset/.*" \
	--exclude ".*/demo/.*" \
	--exclude ".*/build/.*" \
	--exclude ".*/node_modules/.*" \
	--exclude ".*/venv/.*" \
	--exclude ".*/\.git/.*" \
	--exclude-unreachable-branches \
	--exclude-throw-branches \
	--html --html-details \
	-o "$COVERAGE_DIR/coverage.html"

echo ""
echo "✓ Coverage verification complete"
echo ""
echo "Reports:"
echo "  Summary: $COVERAGE_DIR/coverage_summary.txt"
echo "  HTML: $COVERAGE_DIR/coverage.html"
echo ""
echo "To view HTML report:"
echo "  open $COVERAGE_DIR/coverage.html"





