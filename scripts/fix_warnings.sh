#!/bin/bash
# Fix all compilation warnings in NeuronDB
# Target: 0 warnings, 0 errors for PostgreSQL 16, 17, 18

set -e

cd "$(dirname "$0")/.."

echo "Fixing format specifier warnings (%ld → %lld for int64)..."

# Fix format warnings in worker files
files=(
    "src/worker/worker_queue.c"
    "src/worker/worker_tuner.c"
    "src/worker/worker_defrag.c"
    "src/util/data_management.c"
    "src/storage/ann_buffer.c"
    "src/storage/buffer.c"
    "src/storage/vector_wal.c"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        # Fix %ld for int64 → %lld
        sed -i '' 's/%ld\([^l]\)/%lld\1/g' "$file"
        sed -i '' 's/, %ld/, %lld/g' "$file"
        # Add casts where needed
        sed -i '' 's/elog(.*%ld.*job_id)/elog(\1, (long long)job_id)/g' "$file" 2>/dev/null || true
        echo "  Fixed: $file"
    fi
done

echo "✅ Format specifier warnings fixed"

