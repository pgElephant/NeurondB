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
	src/vector_config.o \
	src/bgworker_queue.o \
	src/bgworker_tuner.o \
	src/bgworker_defrag.o \
	src/bgworker_init.o \
	src/gpu_core.o \
	src/gpu_distance.o \
	src/gpu_batch.o \
	src/gpu_quantization.o \
	src/gpu_clustering.o \
	src/gpu_inference.o \
	src/gpu_sql.o

REGRESS = \
	01_types_basic \
	02_types_operations \
	03_distance_metrics \
	04_aggregates \
	05_catalog_tables \
	06_worker_functions \
	07_data_management \
	08_advanced_features \
	09_gpu_features \
	10_gpu_distance_wrappers \
	11_quantization_detail \
	12_catalog_presence

PGFILEDESC = "neurondb - Advanced AI Database with ML Integration"

# GPU support detection (optional, controlled via make arguments)
# Usage: make CUDA_PATH=/custom/cuda ROCM_PATH=/custom/rocm
# If not specified, auto-detect in common paths

# Auto-detect CUDA if not specified
ifndef CUDA_PATH
	CUDA_DETECT := $(shell for p in /usr/local/cuda /opt/cuda /usr/cuda; do \
		test -f $$p/include/cuda_runtime.h && echo $$p && break; \
	done)
	ifneq ($(CUDA_DETECT),)
		CUDA_PATH := $(CUDA_DETECT)
	endif
endif

# Auto-detect ROCm if not specified
ifndef ROCM_PATH
	ROCM_DETECT := $(shell for p in /opt/rocm /usr/rocm; do \
		test -f $$p/include/hip/hip_runtime.h && echo $$p && break; \
	done)
	ifneq ($(ROCM_DETECT),)
		ROCM_PATH := $(ROCM_DETECT)
	endif
endif

# Auto-detect ONNX Runtime if not specified
ifndef ONNX_PATH
	ONNX_DETECT := $(shell for p in /usr/local /usr /opt/onnxruntime; do \
		test -f $$p/include/onnxruntime/core/session/onnxruntime_cxx_api.h && echo $$p && break; \
	done)
	ifneq ($(ONNX_DETECT),)
		ONNX_PATH := $(ONNX_DETECT)
	endif
endif

# Check for CUDA
ifdef CUDA_PATH
	HAVE_CUDA := $(shell test -f $(CUDA_PATH)/include/cuda_runtime.h && echo "yes" || echo "no")
	ifeq ($(HAVE_CUDA),yes)
		NVCC := $(CUDA_PATH)/bin/nvcc
		PG_CPPFLAGS += -DNDB_GPU_CUDA -I$(CUDA_PATH)/include
		CUDA_LIBDIR := $(shell test -d $(CUDA_PATH)/lib64 && echo lib64 || echo lib)
		SHLIB_LINK += -L$(CUDA_PATH)/$(CUDA_LIBDIR) -lcudart -lcublas -lcublasLt
		NVCC_FLAGS = -O3 -arch=sm_80 -Xcompiler -fPIC
		GPU_OBJS = src/gpu_kernels.o
		
		# ONNX Runtime with CUDA provider
		ifdef ONNX_PATH
			ifneq ($(wildcard $(ONNX_PATH)/include/onnxruntime/core/session/onnxruntime_cxx_api.h),)
				PG_CPPFLAGS += -DHAVE_ONNXRUNTIME_GPU -I$(ONNX_PATH)/include
				ONNX_LIBDIR := $(shell test -d $(ONNX_PATH)/lib64 && echo lib64 || echo lib)
				SHLIB_LINK += -L$(ONNX_PATH)/$(ONNX_LIBDIR) -lonnxruntime -lonnxruntime_providers_cuda
			endif
		endif
	endif
endif

# Check for ROCm
ifdef ROCM_PATH
	HAVE_ROCM := $(shell test -f $(ROCM_PATH)/include/hip/hip_runtime.h && echo "yes" || echo "no")
	ifeq ($(HAVE_ROCM),yes)
		HIPCC := $(ROCM_PATH)/bin/hipcc
		PG_CPPFLAGS += -DNDB_GPU_HIP -I$(ROCM_PATH)/include
		SHLIB_LINK += -L$(ROCM_PATH)/lib -lamdhip64 -lrocblas
		GPU_OBJS = src/gpu_kernels_hip.o
	endif
endif

# Add GPU objects if available
ifdef GPU_OBJS
	OBJS += $(GPU_OBJS)
endif

# Optimization flags for production
PG_CPPFLAGS += -Iinclude -I$(libpq_srcdir) -march=native -O3 -Wall -Wextra
SHLIB_LINK += -lm -lz -lcrypto -lcurl

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Custom rule for CUDA kernel compilation
ifdef NVCC
src/gpu_kernels.o: src/gpu_kernels.cu
	$(NVCC) $(NVCC_FLAGS) -I$(shell $(PG_CONFIG) --includedir-server) -Iinclude -c -o $@ $<
endif

# Custom rule for HIP kernel compilation
ifdef HIPCC
src/gpu_kernels_hip.o: src/gpu_kernels.cu
	$(HIPCC) -O3 -fPIC -I$(shell $(PG_CONFIG) --includedir-server) -Iinclude -c -o $@ $<
endif
