#!/bin/bash
# NeuronDB ML Demo Runner
# Usage: ./run_demo.sh --db-size=100MB --ml

set -e

# Default values
DB_SIZE="100MB"
RUN_ML=false
PGPORT=5433
PGHOST=localhost
CLEANUP=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --db-size=*)
            DB_SIZE="${arg#*=}"
            shift
            ;;
        --ml)
            RUN_ML=true
            shift
            ;;
        --port=*)
            PGPORT="${arg#*=}"
            shift
            ;;
        --host=*)
            PGHOST="${arg#*=}"
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --help)
            echo "NeuronDB ML Demo Runner"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --db-size=SIZE    Dataset size (default: 100MB)"
            echo "  --ml              Run ML algorithm tests"
            echo "  --port=PORT        PostgreSQL port (default: 5433)"
            echo "  --host=HOST        PostgreSQL host (default: localhost)"
            echo "  --cleanup          Clean up before running"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --db-size=100MB --ml"
            echo "  $0 --db-size=50MB --ml --port=5432"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$SCRIPT_DIR/ML"
SQL_DIR="$ML_DIR/sql"
OUT_DIR="$ML_DIR/out"

# Create output directory
mkdir -p "$OUT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   NeuronDB ML Demo Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Database Size: $DB_SIZE"
echo "  ML Tests: $RUN_ML"
echo "  PostgreSQL: $PGHOST:$PGPORT"
echo "  Output Dir: $OUT_DIR"
echo ""

# Export PostgreSQL connection
export PGPORT=$PGPORT
export PGHOST=$PGHOST

# Function to run SQL file
run_sql() {
    local sql_file=$1
    local out_file=$2
    local description=$3
    
    echo -e "${YELLOW}[$description]${NC}"
    echo "  Running: $sql_file"
    
    # Suppress NOTICE messages but keep errors
    if psql -p "$PGPORT" -h "$PGHOST" postgres -q -v ON_ERROR_STOP=1 -f "$sql_file" > "$out_file" 2>&1; then
        # Check for actual errors in output (not just connection issues)
        if grep -qi "error\|fatal\|panic" "$out_file"; then
            echo -e "  ${RED}✗ Failed (check output)${NC}"
            return 1
        else
            echo -e "  ${GREEN}✓ Success${NC}"
            return 0
        fi
    else
        # Check if it's just a connection issue
        if grep -q "Connection refused\|could not connect" "$out_file"; then
            echo -e "  ${YELLOW}⚠ PostgreSQL not running${NC}"
            echo "  Start PostgreSQL: pg_ctl -D /path/to/data start"
            return 2
        else
            echo -e "  ${RED}✗ Failed${NC}"
            echo "  Check output: $out_file"
            return 1
        fi
    fi
}

# Cleanup if requested
if [ "$CLEANUP" = true ]; then
    echo -e "${YELLOW}Cleaning up...${NC}"
    run_sql "$SQL_DIR/999_drop_dataset.sql" "$OUT_DIR/00_cleanup.out" "Cleanup"
    echo ""
fi

# Step 1: Generate Dataset
echo -e "${BLUE}Step 1: Dataset Generation${NC}"
run_sql "$SQL_DIR/001_generate_dataset.sql" "$OUT_DIR/01_generate_dataset.out" "Generate $DB_SIZE dataset"
echo ""

if [ "$RUN_ML" = true ]; then
    echo -e "${BLUE}Step 2: ML Algorithm Tests${NC}"
    
    # K-means (comprehensive 9-step test)
    run_sql "$SQL_DIR/002_kmeans_clustering.sql" "$OUT_DIR/02_kmeans_clustering.out" "K-means Clustering"
    echo ""
    
    # GMM (4-step test)
    run_sql "$SQL_DIR/003_gmm_clustering.sql" "$OUT_DIR/03_gmm_clustering.out" "GMM Clustering"
    echo ""
    
    # Mini-batch K-means (9-step test)
    run_sql "$SQL_DIR/004_minibatch_kmeans.sql" "$OUT_DIR/04_minibatch_kmeans.out" "Mini-batch K-means"
    echo ""
    
    # Outlier Detection (9-step test)
    run_sql "$SQL_DIR/005_outlier_detection.sql" "$OUT_DIR/05_outlier_detection.out" "Outlier Detection"
    echo ""
    
    # Hierarchical (9-step test, small sample)
    run_sql "$SQL_DIR/006_hierarchical_clustering.sql" "$OUT_DIR/06_hierarchical_clustering.out" "Hierarchical Clustering"
    echo ""
    
    # Complete Comparison
    run_sql "$SQL_DIR/007_complete_comparison.sql" "$OUT_DIR/07_complete_comparison.out" "Complete Comparison"
    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Demo Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Output files:"
ls -lh "$OUT_DIR"/*.out 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "View results:"
echo "  tail -f $OUT_DIR/*.out"
echo ""

