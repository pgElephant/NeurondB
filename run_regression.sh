#!/bin/bash
# ====================================================================
# NeurondB Comprehensive Regression Testing with Real Datasets
# ====================================================================
# This script:
#   1. Builds NeurondB extension
#   2. Loads comprehensive datasets using dataset/gen_dataset.py
#   3. Installs NeurondB in the dataset database
#   4. Runs full regression suite
#   5. Reports results
# ====================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect PostgreSQL installation
if command -v pg_config >/dev/null 2>&1; then
    PG_BINDIR=$(pg_config --bindir)
    PG_VERSION=$(pg_config --version | awk '{print $2}' | cut -d. -f1)
    echo -e "${GREEN}✓ Found PostgreSQL $PG_VERSION at: $PG_BINDIR${NC}"
else
    echo -e "${RED}✗ pg_config not found. Please install PostgreSQL or add it to PATH${NC}"
    exit 1
fi

# Configuration - use system user and PostgreSQL in PATH
export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-5432}"
export PGUSER="${PGUSER:-$(whoami)}"
TEST_DB="nurondb_dataset"  # This is what gen_dataset.py creates
export PGDATABASE="$TEST_DB"

# PostgreSQL commands from detected installation
PSQL="$PG_BINDIR/psql"
CREATEDB="$PG_BINDIR/createdb"
DROPDB="$PG_BINDIR/dropdb"

echo "========================================"
echo "NeurondB Regression Testing with Datasets"
echo "========================================"
echo "PostgreSQL: $PGHOST:$PGPORT"
echo "User: $PGUSER"
echo "Test Database: $TEST_DB"
echo "Project Root: $SCRIPT_DIR"
echo "========================================"
echo ""

# Step 1: Build NeurondB Extension
echo -e "${BLUE}Step 1: Building NeurondB Extension${NC}"
echo "----------------------------------------"
if [ ! -f "neurondb.so" ] && [ ! -f "neurondb.dylib" ]; then
    echo "Extension not built. Running 'make clean && make'..."
    make clean
    make
    echo -e "${GREEN}✓ Build complete${NC}"
else
    echo -e "${GREEN}✓ Extension already built${NC}"
fi
echo ""

# Step 2: Set up Python environment for dataset loading
echo -e "${BLUE}Step 2: Setting up Python Environment${NC}"
echo "----------------------------------------"
cd dataset
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Python environment ready${NC}"
echo ""

# Step 3: Load datasets using gen_dataset.py
echo -e "${BLUE}Step 3: Loading Datasets into PostgreSQL${NC}"
echo "----------------------------------------"
echo "Running dataset/gen_dataset.py (this will take 10-30 minutes)..."
echo ""
echo "This will:"
echo "  - Drop and recreate database '$TEST_DB'"
echo "  - Load MS MARCO passages (document retrieval)"
echo "  - Load Wikipedia embeddings (clustering, PCA)"
echo "  - Load HotpotQA (question answering)"
echo "  - Load SIFT1M vectors (high-dimensional, quantization)"
echo "  - Load Deep1B vectors (large-scale embeddings)"
echo "  - Create schemas: ms_marco, wikipedia_embeddings, hotpotqa, sift1m, deep1b"
echo ""
echo "You can skip HDF5 datasets (SIFT, Deep1B) by setting: export SKIP_HDF5=1"
echo ""

# Run gen_dataset.py which handles database creation and all data loading
if python3 gen_dataset.py; then
    echo ""
    echo -e "${GREEN}✓ All datasets loaded successfully${NC}"
else
    echo ""
    echo -e "${RED}✗ Dataset loading failed${NC}"
    deactivate
    exit 1
fi
echo ""

deactivate
cd "$SCRIPT_DIR"

# Step 4: Install NeurondB extension in the dataset database
echo -e "${BLUE}Step 4: Installing NeurondB Extension in Database${NC}"
echo "----------------------------------------"
echo "Installing NeurondB and dependencies..."

# Create SQL script to install extension
cat > /tmp/install_neurondb.sql << 'EOF'
-- Install required extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Install NeurondB (this will create the neurondb schema and all functions)
CREATE EXTENSION IF NOT EXISTS neurondb CASCADE;

-- Verify installation
SELECT extname, extversion FROM pg_extension WHERE extname = 'neurondb';

-- Show loaded schemas
SELECT nspname FROM pg_namespace WHERE nspname IN ('neurondb', 'ms_marco', 'wikipedia_embeddings', 'hotpotqa', 'sift1m', 'deep1b') ORDER BY nspname;
EOF

if $PSQL -d "$TEST_DB" -v ON_ERROR_STOP=1 -f /tmp/install_neurondb.sql; then
    echo -e "${GREEN}✓ NeurondB extension installed${NC}"
else
    echo -e "${RED}✗ Failed to install NeurondB extension${NC}"
    exit 1
fi
rm -f /tmp/install_neurondb.sql
echo ""

# Step 5: Show dataset statistics
echo -e "${BLUE}Step 5: Dataset Statistics${NC}"
echo "----------------------------------------"
$PSQL -d "$TEST_DB" -c "
SELECT 
    schemaname,
    tablename,
    (xpath('/row/count/text()', xml_count))[1]::text::int as row_count
FROM (
    SELECT 
        schemaname,
        tablename,
        query_to_xml(format('SELECT count(*) FROM %I.%I', schemaname, tablename), false, true, '') as xml_count
    FROM pg_tables
    WHERE schemaname IN ('ms_marco', 'wikipedia_embeddings', 'hotpotqa', 'sift1m', 'deep1b')
) sub
ORDER BY schemaname, tablename;
" 2>/dev/null || echo "Statistics not available"
echo ""

# Step 6: Run regression tests
echo -e "${BLUE}Step 6: Running Regression Test Suite${NC}"
echo "----------------------------------------"
echo "Running all tests (22 test suites)..."
echo ""

# Set PostgreSQL environment for regression tests
export PGDATABASE="$TEST_DB"

# Run regression tests
if make installcheck; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo -e "${GREEN}========================================${NC}"
    RESULT=0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check regression.diffs for details:"
    if [ -f "regression.diffs" ]; then
        echo ""
        head -100 regression.diffs
        echo ""
        echo "(showing first 100 lines, see regression.diffs for full output)"
    fi
    RESULT=1
fi

# Step 7: Generate test report
echo ""
echo -e "${BLUE}Step 7: Test Summary${NC}"
echo "----------------------------------------"
if [ -f "regression.out" ]; then
    PASSED=$(grep -c "^ok" regression.out || true)
    FAILED=$(grep -c "^not ok" regression.out || true)
    TOTAL=$((PASSED + FAILED))
    echo "Tests Passed: $PASSED / $TOTAL"
    echo "Tests Failed: $FAILED / $TOTAL"
    
    if [ $FAILED -gt 0 ]; then
        echo ""
        echo "Failed tests:"
        grep "^not ok" regression.out || true
    fi
else
    echo "No test results found"
fi

echo ""
echo "Test suites executed:"
echo "  ✓ Core types and operations (01-04)"
echo "  ✓ Catalog and workers (05-06)"
echo "  ✓ Data management (07-08)"
echo "  ✓ GPU features (09-11)"
echo "  ✓ Catalog validation (12)"
echo "  ✓ ML: Clustering (K-Means, DBSCAN, GMM, Hierarchical) (13)"
echo "  ✓ ML: Dimensionality (PCA, Whitening) (14)"
echo "  ✓ ML: Quantization (PQ, OPQ) (15)"
echo "  ✓ ML: Reranking (MMR, RRF, Ensemble) (16)"
echo "  ✓ ML: Outliers (Z-score, IQR, Isolation Forest) (17)"
echo "  ✓ ML: Metrics (Recall@K, Precision@K, F1, MRR) (18)"
echo "  ✓ ML: Drift Detection (Centroid, Distribution, Temporal) (19)"
echo "  ✓ ML: Hybrid Search & LTR (20)"
echo "  ✓ ML: Analytics (KNN Graph, Quality, Topics, Histograms) (21)"
echo ""

# Dataset information
echo -e "${BLUE}Datasets Available for Testing:${NC}"
echo "  • ms_marco.data - MS MARCO passages (docid, content)"
echo "  • wikipedia_embeddings.items - Wikipedia (id, title, text, embedding)"
echo "  • hotpotqa.qa - HotpotQA (id, title, question, context, answer)"
echo "  • sift1m.vectors - SIFT vectors (id, embedding)"
echo "  • deep1b.vectors - Deep1B vectors (id, embedding)"
echo ""

# Cleanup option
echo -e "${YELLOW}Test database '$TEST_DB' is still running with all datasets.${NC}"
echo "To remove it and free up disk space, run:"
echo "  $DROPDB $TEST_DB"
echo ""
echo "To keep datasets and re-run tests:"
echo "  export PGDATABASE=$TEST_DB"
echo "  make installcheck"
echo ""

exit $RESULT
