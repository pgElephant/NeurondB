# NeurondB Dataset Loading & Regression Testing

This directory contains scripts and tools for loading comprehensive datasets and running regression tests for NeurondB.

## Overview

The dataset loading system supports:

- **MS MARCO**: Document retrieval and passage ranking (~10K passages)
- **Wikipedia**: General knowledge articles with embeddings (~5K articles)
- **HotpotQA**: Multi-hop question answering dataset (~3K questions)
- **SIFT1M**: High-dimensional SIFT descriptors for computer vision (~50K vectors)
- **Deep1B**: Deep learning embeddings at scale (~20K vectors)
- **Synthetic**: Generated test datasets for specific ML algorithms

## Quick Start

### Complete Regression Testing (Recommended)

Run the full test suite with all datasets:

```bash
cd /Users/pgedge/pge/NeurondB/dataset
./run_regression_with_datasets.sh
```

This script will:
1. Build NeurondB extension
2. Set up Python environment
3. Create test database
4. Load all datasets
5. Run complete regression suite
6. Report results

**Time**: 20-40 minutes (depending on network speed)

### Manual Dataset Loading

Load specific datasets individually:

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create database
python3 gen_dataset_enhanced.py --recreate-db --dbname neurondb_test

# Load specific datasets
python3 gen_dataset_enhanced.py --load-msmarco --limit 10000 --dbname neurondb_test
python3 gen_dataset_enhanced.py --load-wikipedia --limit 5000 --dbname neurondb_test
python3 gen_dataset_enhanced.py --load-hotpotqa --limit 3000 --dbname neurondb_test
python3 gen_dataset_enhanced.py --load-sift --limit 50000 --dbname neurondb_test
python3 gen_dataset_enhanced.py --load-deep1b --limit 20000 --dbname neurondb_test

# Create synthetic test data
python3 gen_dataset_enhanced.py --create-synthetic --dbname neurondb_test

# Create FTS indexes
python3 gen_dataset_enhanced.py --create-fts-indexes --dbname neurondb_test

# Show statistics
python3 gen_dataset_enhanced.py --show-stats --dbname neurondb_test
```

### Load All Datasets at Once

```bash
python3 gen_dataset_enhanced.py --load-all --limit 10000 --dbname neurondb_test
```

## Files

- **`gen_dataset.py`**: Original dataset loader (basic functionality)
- **`gen_dataset_enhanced.py`**: Enhanced loader with CLI arguments
- **`load_all_datasets.sh`**: Bash script to load all datasets
- **`run_regression_with_datasets.sh`**: Complete test runner (build + load + test)
- **`setup_test_db.sql`**: SQL script to set up test database with NeurondB
- **`requirements.txt`**: Python dependencies

## Dataset Schema

All datasets are loaded into the `neurondb_datasets` schema:

```
neurondb_datasets.data           -- MS MARCO passages (docid, content)
neurondb_datasets.items          -- Wikipedia (id, title, text, embedding)
neurondb_datasets.qa             -- HotpotQA (id, title, question, context, answer)
neurondb_datasets_sift.vectors   -- SIFT vectors (id, embedding)
neurondb_datasets_deep1b.vectors -- Deep1B vectors (id, embedding)
neurondb_datasets.synthetic_*    -- Synthetic test datasets
```

## Environment Variables

Configure database connection:

```bash
export PGHOST=localhost
export PGPORT=5432
export PGUSER=postgres
export PGDATABASE=neurondb_test
export PGPASSWORD=yourpassword  # if needed
```

## Dataset Sizes & Network Requirements

| Dataset | Size | Rows | Download Time (100 Mbps) |
|---------|------|------|--------------------------|
| MS MARCO | ~500 MB | 10K | ~40s |
| Wikipedia | ~2 GB | 5K | ~3min |
| HotpotQA | ~50 MB | 3K | ~5s |
| SIFT1M | ~500 MB | 50K | ~40s |
| Deep1B | ~400 MB | 20K | ~30s |

**Total**: ~3.5 GB, ~5-10 minutes download

## Regression Test Coverage

The regression suite tests all NeurondB features:

### Core Features (01-12)
- Vector types and operations
- Distance metrics
- Aggregates
- Catalog tables
- Worker functions
- Data management
- GPU features
- Quantization

### ML Algorithms (13-21)
- **Clustering**: K-Means, Mini-batch K-Means, DBSCAN, GMM, Hierarchical
- **Dimensionality**: PCA, PCA Whitening
- **Quantization**: Product Quantization (PQ), Optimized PQ (OPQ)
- **Reranking**: MMR, Reciprocal Rank Fusion (RRF), Ensemble
- **Outliers**: Z-score, Modified Z-score, IQR, Isolation Forest
- **Metrics**: Recall@K, Precision@K, F1@K, MRR, Davies-Bouldin
- **Drift**: Centroid Drift, Distribution Divergence, Temporal Monitoring
- **Search**: Hybrid Search (lexical + semantic), Learning to Rank (LTR)
- **Analytics**: KNN Graph, Embedding Quality, Topic Discovery, Histograms

## Usage in Tests

Regression tests can use the loaded datasets:

```sql
-- Use MS MARCO for document retrieval tests
SELECT * FROM neurondb_datasets.data LIMIT 10;

-- Use Wikipedia for clustering tests
SELECT * FROM neurondb_datasets.items LIMIT 10;

-- Use SIFT for high-dimensional vector tests
SELECT * FROM neurondb_datasets_sift.vectors LIMIT 10;
```

## Troubleshooting

### Database Connection Errors

```bash
# Check PostgreSQL is running
pg_ctl status -D /usr/local/pgsql.17/data

# Check connection
psql -h localhost -p 5432 -U postgres -c "SELECT version();"
```

### Python Environment Issues

```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Download Failures

- Check internet connection
- Some datasets (HotpotQA, SIFT) may have temporary availability issues
- Use `--skip-hdf5` to skip SIFT/Deep1B if h5py installation fails

### Disk Space

Ensure at least **10 GB** free space for:
- Downloaded datasets (~3.5 GB)
- PostgreSQL database (~2-3 GB)
- Temporary files (~1 GB)

## Performance Notes

### Loading Speed

- HuggingFace datasets (MS MARCO, Wikipedia) use streaming for efficiency
- SIFT/Deep1B use memory-mapped HDF5 for fast access
- Batch inserts (1000 rows) for optimal performance
- FTS indexes created after data loading

### Test Execution

- Full regression suite: ~10-15 minutes
- GPU tests: ~2-3 minutes (if GPU available)
- Individual ML algorithm tests: ~30-60 seconds each

## Cleanup

Remove test database after testing:

```bash
dropdb neurondb_test
```

Remove downloaded datasets:

```bash
rm -rf datasets/
```

## Contributing

When adding new datasets:

1. Add loader function in `gen_dataset.py`
2. Add CLI argument in `gen_dataset_enhanced.py`
3. Update `load_all_datasets.sh`
4. Add regression test in `sql/`
5. Update this README

## Support

For issues or questions:
- Check regression.diffs for test failures
- Review logs in regression.out
- File issues on GitHub with dataset name and error message

