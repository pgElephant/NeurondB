# NeurondB: Advanced AI Database Extension
# Copyright (c) 2024-2025, pgElephant, Inc.

MODULE_big = neurondb
EXTENSION = neurondb
DATA = neurondb--1.0.sql
DOCS = README.md

# Source files in src/ directory
OBJS = \
	src/neurondb.o \
	src/distance.o \
	src/quantization.o \
	src/ml_inference.o \
	src/hybrid_search.o \
	src/vector_ops.o \
	src/aggregates.o \
	src/types_core.o \
	src/index_hnsw_tenant.o \
	src/index_hybrid.o \
	src/index_temporal.o \
	src/index_consistent.o \
	src/index_rerank.o \
	src/operators.o \
	src/pg_stat_neurondb.o \
	src/usability.o \
	src/data_management.o \
	src/multi_tenant.o \
	src/planner.o \
	src/security.o \
	src/developer_hooks.o \
	src/distributed.o \
	src/analytics.o \
	src/buffer.o \
	src/model_runtime.o \
	src/ann_buffer.o \
	src/vector_wal.o \
	src/vector_config.o

REGRESS = \
	01_types_basic \
	02_types_operations \
	03_distance_metrics

PGFILEDESC = "neurondb - Advanced AI Database with ML Integration"

# Optimization flags for production
PG_CPPFLAGS = -Iinclude -I$(libpq_srcdir) -march=native -O3 -Wall -Wextra
SHLIB_LINK = -lm -lz -lcrypto -lcurl

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
