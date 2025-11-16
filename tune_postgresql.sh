#!/bin/bash
# PostgreSQL Tuning Script for NeurondB ML Workloads
# This script optimizes PostgreSQL settings for large-scale ML operations

set -e

PGDATA="${PGDATA:-/home/pge/data/18}"
PGCONF="${PGDATA}/postgresql.conf"

if [ ! -f "$PGCONF" ]; then
	echo "Error: postgresql.conf not found at $PGCONF"
	exit 1
fi

echo "=== NeurondB PostgreSQL Tuning ==="
echo "Config file: $PGCONF"
echo ""

# Get system resources
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
CPU_CORES=$(nproc)
AVAILABLE_RAM_GB=$((TOTAL_RAM_GB - 2))  # Reserve 2GB for OS

# Calculate optimal settings
SHARED_BUFFERS=$((AVAILABLE_RAM_GB * 25 / 100))  # 25% of available RAM
EFFECTIVE_CACHE_SIZE=$((AVAILABLE_RAM_GB * 75 / 100))  # 75% of available RAM
WORK_MEM=$((AVAILABLE_RAM_GB * 1024 / (CPU_CORES * 2)))  # Per operation, account for parallelism
MAINTENANCE_WORK_MEM=$((AVAILABLE_RAM_GB * 1024 / 4))  # 25% of available RAM
MAX_PARALLEL_WORKERS=$((CPU_CORES - 1))
MAX_PARALLEL_WORKERS_PER_GATHER=$((CPU_CORES / 2))

# Clamp values to reasonable ranges
if [ $SHARED_BUFFERS -lt 1 ]; then SHARED_BUFFERS=1; fi
if [ $SHARED_BUFFERS -gt 8 ]; then SHARED_BUFFERS=8; fi
if [ $WORK_MEM -lt 16 ]; then WORK_MEM=16; fi
if [ $WORK_MEM -gt 512 ]; then WORK_MEM=512; fi
if [ $MAINTENANCE_WORK_MEM -lt 128 ]; then MAINTENANCE_WORK_MEM=128; fi
if [ $MAINTENANCE_WORK_MEM -gt 2048 ]; then MAINTENANCE_WORK_MEM=2048; fi
if [ $MAX_PARALLEL_WORKERS -lt 2 ]; then MAX_PARALLEL_WORKERS=2; fi
if [ $MAX_PARALLEL_WORKERS -gt 16 ]; then MAX_PARALLEL_WORKERS=16; fi

echo "System Resources:"
echo "  Total RAM: ${TOTAL_RAM_GB}GB"
echo "  CPU Cores: ${CPU_CORES}"
echo "  Available RAM: ${AVAILABLE_RAM_GB}GB"
echo ""
echo "Recommended Settings:"
echo "  shared_buffers = ${SHARED_BUFFERS}GB"
echo "  effective_cache_size = ${EFFECTIVE_CACHE_SIZE}GB"
echo "  work_mem = ${WORK_MEM}MB"
echo "  maintenance_work_mem = ${MAINTENANCE_WORK_MEM}MB"
echo "  max_parallel_workers = ${MAX_PARALLEL_WORKERS}"
echo "  max_parallel_workers_per_gather = ${MAX_PARALLEL_WORKERS_PER_GATHER}"
echo ""

# Backup original config
cp "$PGCONF" "${PGCONF}.backup.$(date +%Y%m%d_%H%M%S)"
echo "Backed up config to ${PGCONF}.backup.$(date +%Y%m%d_%H%M%S)"
echo ""

# Function to set or update a config parameter
set_config() {
	local param="$1"
	local value="$2"
	local comment="$3"
	
	if grep -q "^[[:space:]]*${param}[[:space:]]*=" "$PGCONF"; then
		# Update existing setting
		sed -i "s|^[[:space:]]*${param}[[:space:]]*=.*|${param} = ${value}  # ${comment}|" "$PGCONF"
		echo "  Updated: ${param} = ${value}"
	else
		# Add new setting at end of file
		echo "" >> "$PGCONF"
		echo "# NeurondB ML workload optimization: ${comment}" >> "$PGCONF"
		echo "${param} = ${value}" >> "$PGCONF"
		echo "  Added: ${param} = ${value}"
	fi
}

echo "Applying optimizations..."

# Memory settings
set_config "shared_buffers" "${SHARED_BUFFERS}GB" "25% of available RAM for shared buffer cache"
set_config "effective_cache_size" "${EFFECTIVE_CACHE_SIZE}GB" "75% of RAM for query planner estimates"
set_config "work_mem" "${WORK_MEM}MB" "Memory for sorting/hashing per operation (increased for ML)"
set_config "maintenance_work_mem" "${MAINTENANCE_WORK_MEM}MB" "Memory for VACUUM, CREATE INDEX, etc."

# Parallelism settings
set_config "max_parallel_workers" "${MAX_PARALLEL_WORKERS}" "Max parallel worker processes"
set_config "max_parallel_workers_per_gather" "${MAX_PARALLEL_WORKERS_PER_GATHER}" "Max workers per parallel query"
set_config "max_parallel_maintenance_workers" "$((MAX_PARALLEL_WORKERS / 2))" "Max workers for parallel maintenance"

# Query planner settings (optimized for SSD)
set_config "random_page_cost" "1.1" "Lower cost for SSD (default 4.0 for HDD)"
set_config "effective_io_concurrency" "200" "Concurrent I/O operations (higher for SSD)"

# Checkpoint and WAL settings (for better write performance)
set_config "checkpoint_timeout" "15min" "Time between checkpoints"
set_config "max_wal_size" "4GB" "Maximum WAL size before checkpoint"
set_config "min_wal_size" "1GB" "Minimum WAL size to retain"
set_config "wal_buffers" "16MB" "WAL buffer size"

# Connection and query settings
set_config "max_connections" "100" "Max concurrent connections"
set_config "statement_timeout" "0" "Disable statement timeout for long ML operations"
set_config "lock_timeout" "0" "Disable lock timeout for long operations"

# JIT settings (can help with complex queries)
set_config "jit" "on" "Enable JIT compilation"
set_config "jit_above_cost" "100000" "Use JIT for queries above this cost"
set_config "jit_optimize_above_cost" "500000" "Optimize JIT for queries above this cost"

# Logging (useful for debugging)
set_config "log_min_duration_statement" "1000" "Log queries taking longer than 1 second"
set_config "log_line_prefix" "'%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '" "Detailed log prefix"

# NeurondB-specific optimizations
set_config "shared_preload_libraries" "'neurondb'" "Load NeurondB extension at startup"

echo ""
echo "=== Optimization Complete ==="
echo ""
echo "To apply changes, restart PostgreSQL:"
echo "  pg_ctl -D ${PGDATA} restart"
echo ""
echo "Or reload configuration (for settings that support it):"
echo "  pg_ctl -D ${PGDATA} reload"
echo ""
echo "To verify settings:"
echo "  psql -d neurondb -c \"SHOW shared_buffers; SHOW work_mem; SHOW max_parallel_workers;\""
echo ""

