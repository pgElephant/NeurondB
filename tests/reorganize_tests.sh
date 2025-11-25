#!/bin/bash
# Reorganize test files with correct numbering for maximum code coverage

cd "$(dirname "$0")/sql"

echo "Reorganizing test files..."

# Rename 030_* files to proper sequential numbers
# 028: Core (core module)
# 029: Index (index module) 
# 030: Worker (worker module)
# 031: Storage (storage module)
# 032: Scan (scan module)
# 033: Utility (util module)
# 034: Planner (planner module)
# 035: Tenant (tenant module)
# 036: Types (types module)
# 037: Metrics (metrics module)
# 038: ONNX (onnx module)

# Core module: 028
mv basic/030_core_basic.sql basic/028_core_basic.sql 2>/dev/null || true
mv advance/030_core_advance.sql advance/028_core_advance.sql 2>/dev/null || true
mv negative/030_core_negative.sql negative/028_core_negative.sql 2>/dev/null || true

# Index module: 029
mv basic/030_index_basic.sql basic/029_index_basic.sql 2>/dev/null || true
mv advance/030_index_advance.sql advance/029_index_advance.sql 2>/dev/null || true
mv negative/030_index_negative.sql negative/029_index_negative.sql 2>/dev/null || true

# Worker module: 030
mv basic/030_worker_basic.sql basic/030_worker_basic.sql 2>/dev/null || true
mv advance/030_worker_advance.sql advance/030_worker_advance.sql 2>/dev/null || true
mv negative/030_worker_negative.sql negative/030_worker_negative.sql 2>/dev/null || true

# GPU module: Merge into 010_gpu_info (enhance existing)
# LLM module: Merge into 027_embeddings (enhance existing)
# Vector module: Merge into 026_vector (enhance existing)

# Remove duplicate 030_gpu, 030_llm, 030_vector files (will merge into existing)
rm -f advance/030_gpu_advance.sql
rm -f negative/030_gpu_negative.sql
rm -f advance/030_llm_advance.sql
rm -f negative/030_llm_negative.sql
rm -f advance/030_vector_advance.sql
rm -f negative/030_vector_negative.sql

echo "Reorganization complete!"





