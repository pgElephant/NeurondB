#!/bin/bash
# ====================================================================
# NeurondB Dataset Loading Script
# ====================================================================
# Loads comprehensive datasets for testing all NeurondB features
# ====================================================================

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Database configuration
export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-5432}"
export PGUSER="${PGUSER:-postgres}"
export PGDATABASE="${PGDATABASE:-neurondb_test}"

echo "========================================"
echo "NeurondB Dataset Loading"
echo "========================================"
echo "Host: $PGHOST"
echo "Port: $PGPORT"
echo "Database: $PGDATABASE"
echo "========================================"

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "Step 1: Creating/Recreating Database"
echo "--------------------------------------"
python3 gen_dataset.py --recreate-db

echo ""
echo "Step 2: Loading MS MARCO Dataset (Document Retrieval)"
echo "------------------------------------------------------"
echo "MS MARCO: Large-scale information retrieval dataset"
echo "Usage: Testing semantic search, reranking, hybrid search"
python3 gen_dataset.py --load-msmarco --limit 10000

echo ""
echo "Step 3: Loading Wikipedia Embeddings (General Knowledge)"
echo "---------------------------------------------------------"
echo "Wikipedia: General knowledge articles with embeddings"
echo "Usage: Testing clustering, PCA, outlier detection"
python3 gen_dataset.py --load-wikipedia --limit 5000

echo ""
echo "Step 4: Loading HotpotQA Dataset (Question Answering)"
echo "------------------------------------------------------"
echo "HotpotQA: Multi-hop question answering dataset"
echo "Usage: Testing MMR, recall metrics, topic discovery"
python3 gen_dataset.py --load-hotpotqa --limit 3000

echo ""
echo "Step 5: Loading SIFT1M Dataset (Computer Vision)"
echo "-------------------------------------------------"
echo "SIFT: SIFT descriptors for image matching"
echo "Usage: Testing PQ, OPQ, high-dimensional vectors"
python3 gen_dataset.py --load-sift --limit 50000

echo ""
echo "Step 6: Loading Deep1B Sample (Large-Scale Vectors)"
echo "----------------------------------------------------"
echo "Deep1B: Deep learning embeddings at scale"
echo "Usage: Testing scalability, performance benchmarks"
python3 gen_dataset.py --load-deep1b --limit 20000

echo ""
echo "Step 7: Creating Synthetic Test Datasets"
echo "-----------------------------------------"
python3 gen_dataset.py --create-synthetic

echo ""
echo "Step 8: Creating Full-Text Search Indexes"
echo "------------------------------------------"
python3 gen_dataset.py --create-fts-indexes

echo ""
echo "Step 9: Generating Dataset Statistics"
echo "--------------------------------------"
python3 gen_dataset.py --show-stats

echo ""
echo "========================================"
echo "Dataset Loading Complete!"
echo "========================================"
echo ""
echo "Datasets loaded:"
echo "  - neurondb_datasets.msmarco_passages (~10K passages)"
echo "  - neurondb_datasets.wikipedia_articles (~5K articles)"
echo "  - neurondb_datasets.hotpotqa_questions (~3K questions)"
echo "  - neurondb_datasets.sift_vectors (~50K SIFT descriptors)"
echo "  - neurondb_datasets.deep1b_vectors (~20K embeddings)"
echo "  - neurondb_datasets.synthetic_* (various synthetic datasets)"
echo ""
echo "Ready for regression testing!"
echo "========================================"

deactivate

