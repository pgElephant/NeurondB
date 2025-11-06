# NeurondB: Advanced AI Database Extension
# Copyright (c) 2024-2025, pgElephant, Inc.

MODULE_big = neurondb
EXTENSION = neurondb
DATA = neurondb--1.0.sql
DOCS = README.md

# Source files organized by subdirectory
# Core vector types and operations
OBJS = \
	src/core/neurondb.o \
	src/core/operators.o

# Vector module (consolidated)
OBJS += \
	src/vector/vector_distance.o \
	src/vector/vector_types.o \
	src/vector/vector_ops.o \
	src/vector/vector_wal.o

# Index access methods
OBJS += \
	src/index/index_hnsw_tenant.o \
	src/index/index_hybrid.o \
	src/index/index_temporal.o \
	src/index/index_consistent.o \
	src/index/index_rerank.o \
	src/index/hnsw_am.o \
	src/index/ivf_am.o \
	src/index/opclass.o \
	src/index/index_validator.o \
	src/index/index_cache.o

# Scan nodes
OBJS += \
	src/scan/hnsw_scan.o \
	src/scan/custom_hybrid_scan.o \
	src/scan/scan_rls.o \
	src/scan/scan_quota.o

# Background workers
OBJS += \
	src/worker/worker_queue.o \
	src/worker/worker_tuner.o \
	src/worker/worker_defrag.o \
	src/worker/worker_init.o \
	src/worker/worker_llm.o

# LLM/Hugging Face integration
OBJS += \
	src/llm/llm_runtime.o \
	src/llm/llm_cache.o \
	src/llm/llm_jobs.o \
	src/llm/hf_http.o

# ONNX Runtime integration for HuggingFace models
OBJS += \
	src/onnx/neurondb_onnx.o \
	src/onnx/neurondb_tokenizer.o \
	src/onnx/neurondb_hf.o

# GPU acceleration
OBJS += \
	src/gpu/gpu_core.o \
	src/gpu/gpu_distance.o \
	src/gpu/gpu_batch.o \
	src/gpu/gpu_quantization.o \
	src/gpu/gpu_clustering.o \
	src/gpu/gpu_inference.o \
	src/gpu/gpu_sql.o

# Types and operations
OBJS += \
	src/types/quantization.o \
	src/types/aggregates.o

# Machine learning (NEW supervised learning modules FIRST for better loading)
OBJS += \
	src/ml/ml_unified_api.o \
	src/ml/ml_ridge_lasso.o \
	src/ml/ml_random_forest.o \
	src/ml/ml_svm.o \
	src/ml/ml_naive_bayes.o \
	src/ml/ml_decision_tree.o \
	src/ml/ml_linear_regression.o \
	src/ml/ml_logistic_regression.o \
	src/ml/ml_knn.o \
	src/ml/ml_projects.o \
	src/ml/ml_utils.o \
	src/ml/analytics.o \
	src/ml/ml_kmeans.o \
	src/ml/ml_dbscan.o \
	src/ml/ml_minibatch_kmeans.o \
	src/ml/ml_product_quantization.o \
	src/ml/ml_mmr.o \
	src/ml/ml_davies_bouldin.o \
	src/ml/ml_outlier_detection.o \
	src/ml/ml_pca_whitening.o \
	src/ml/ml_recall_metrics.o \
	src/ml/ml_drift_detection.o \
	src/ml/ml_gmm.o \
	src/ml/ml_hierarchical.o \
	src/ml/ml_histogram.o \
	src/ml/ml_rerank_ensemble.o \
	src/ml/ml_ltr.o \
	src/ml/ml_hybrid_search.o \
	src/ml/ml_topic_discovery.o \
	src/ml/ml_drift_time.o \
	src/ml/ml_opq.o \
	src/ml/ml_automl.o \
	src/ml/ml_catboost.o \
	src/ml/ml_deeplearning.o \
	src/ml/ml_feature_store.o \
	src/ml/ml_hyperparameter_tuning.o \
	src/ml/ml_lightgbm.o \
	src/ml/ml_mlops_advanced.o \
	src/ml/ml_neural_network.o \
	src/ml/ml_nlp_production.o \
	src/ml/ml_rag.o \
	src/ml/ml_recommender.o \
	src/ml/ml_text.o \
	src/ml/ml_timeseries.o \
	src/ml/ml_xgboost.o \
	src/ml/model_runtime.o \
	src/ml/embeddings.o \
	src/ml/reranking.o

# Metrics and observability
OBJS += \
	src/metrics/pg_stat_neurondb.o \
	src/metrics/prometheus.o

# Search algorithms
OBJS += \
	src/search/hybrid_search.o \
	src/search/temporal_integration.o

# Storage and memory
OBJS += \
	src/storage/ann_buffer.o \
	src/storage/buffer.o

# Multi-tenancy
OBJS += \
	src/tenant/multi_tenant.o

# Query planning
OBJS += \
	src/planner/planner.o

# Utilities
OBJS += \
	src/util/config.o \
	src/util/security.o \
	src/util/hooks.o \
	src/util/distributed.o \
	src/util/usability.o \
	src/util/data_management.o

REGRESS = \
	00_create \
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
	12_catalog_presence \
	13_ml_clustering \
	14_ml_dimensionality \
	15_ml_quantization \
	16_ml_reranking \
	17_ml_outliers \
	18_ml_metrics \
	19_ml_drift \
	20_ml_hybrid_search \
	21_ml_analytics \
	99_cleanup

# GPU-only tests
REGRESS_GPU = \
	00_create \
	09_gpu_features \
	10_gpu_distance_wrappers \
	11_quantization_detail \
	99_cleanup

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

# Check for Metal (Apple Silicon)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
# Check for Metal (Apple Silicon)
ifeq ($(UNAME_S),Darwin)
	ifeq ($(UNAME_M),arm64)
		HAVE_METAL := yes
		PG_CPPFLAGS += -DNDB_GPU_METAL
		SHLIB_LINK += -framework Metal -framework MetalPerformanceShaders -framework Accelerate -framework Foundation
		METAL_OBJS = src/gpu/gpu_metal.o src/gpu/gpu_metal_impl.o
	endif
endif

# Add GPU objects if available
ifdef GPU_OBJS
	OBJS += $(GPU_OBJS)
endif

# Add Metal objects if available
ifdef METAL_OBJS
	OBJS += $(METAL_OBJS)
endif

# Special rule for Metal implementation to avoid Protocol conflicts
src/gpu/gpu_metal_impl.o: src/gpu/gpu_metal_impl.m
	$(CC) $(filter-out -I/usr/local/include, $(PG_CPPFLAGS)) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

# Optimization flags for production with SIMD
PG_CPPFLAGS += -Iinclude -I$(libpq_srcdir) -I/usr/include \
               -march=native -O3 -Wall -Wextra \
               -fno-math-errno -fstrict-aliasing -funroll-loops \
               -fomit-frame-pointer -ffp-contract=fast -fopenmp-simd \
               -mtune=native -fno-trapping-math \
               -Wno-unused-parameter -Wno-missing-field-initializers \
               -Wno-declaration-after-statement

# Check if compiler supports -fno-signaling-nans (GCC supports it, clang doesn't)
# Test by compiling empty file - if it fails with "unsupported", don't use the flag
# Check if compiler is clang (which warns about -fno-signaling-nans)
# Use $(CC) from PostgreSQL, check if it's actually clang
IS_CLANG := $(shell $(CC) --version 2>&1 | head -1 | grep -qi "clang" && echo yes || echo no)
ifeq ($(IS_CLANG),yes)
	# clang - skip -fno-signaling-nans to avoid warnings
else
	# GCC - safe to use
	PG_CPPFLAGS += -fno-signaling-nans
endif

# Detect architecture and add appropriate SIMD flags
ARCH := $(shell uname -m)
ifeq ($(ARCH),x86_64)
	# x86_64: Enable AVX2
	PG_CPPFLAGS += -mavx2 -mfma -DUSE_AVX2
	
	# Detect AVX512
	HAS_AVX512 := $(shell $(CC) -march=native -dM -E - < /dev/null 2>/dev/null | grep -q AVX512F && echo yes || echo no)
	ifeq ($(HAS_AVX512),yes)
		PG_CPPFLAGS += -mavx512f -mavx512vl -mavx512bw -DUSE_AVX512
	endif
else ifeq ($(ARCH),arm64)
	# ARM64: Enable NEON (always available on ARM64)
	PG_CPPFLAGS += -DUSE_NEON
else ifeq ($(ARCH),aarch64)
	# ARM64: Enable NEON (Linux naming)
	PG_CPPFLAGS += -DUSE_NEON
endif
SHLIB_LINK += -lm -lz -lcrypto -lcurl -lssl

# macOS: Note - symbol export issue being investigated
# Functions are exported but PostgreSQL 17 loader cannot find them
# This appears to be a PostgreSQL 17/macOS dylib loader bug
ifeq ($(shell uname -s),Darwin)
	# Keep default linking for now
endif

PG_CONFIG ?= /usr/local/pgsql.18/bin/pg_config
# Generate SQL file from template before build
neurondb--1.0.sql: neurondb--1.0.sql.in Makefile Makefile.sql-functions Makefile.header
	$(eval include Makefile.header)
	@echo "=========================================================================="
	@echo "Generating version-specific SQL file..."
	@echo "  PostgreSQL: $(PGSQL_VERSION) (version_num: $(PGSQL_VERSION_NUM))"
	@echo "  Platform: $(PLATFORM) ($(UNAME_S))"
	@echo "  Build Type: $(BUILD_TYPE)"
	@echo "  Build Date: $(BUILD_DATE)"
	@echo "=========================================================================="
	@sed -e 's|@PGSQL_VERSION@|$(PGSQL_VERSION)|g' \
	     -e 's|@PGSQL_VERSION_NUM@|$(PGSQL_VERSION_NUM)|g' \
	     -e 's|@PLATFORM@|$(PLATFORM)|g' \
	     -e 's|@UNAME_S@|$(UNAME_S)|g' \
	     -e 's|@BUILD_DATE@|$(BUILD_DATE)|g' \
	     -e 's|@BUILD_TYPE@|$(BUILD_TYPE)|g' \
	     neurondb--1.0.sql.in > neurondb--1.0.sql.tmp
	@$(MAKE) -s -f Makefile.sql-functions ml-regression-functions > neurondb--1.0.sql.reg.funcs
	@$(MAKE) -s -f Makefile.sql-functions ml-classification-functions > neurondb--1.0.sql.clf.funcs
	@$(MAKE) -s -f Makefile.sql-functions ml-knn-functions > neurondb--1.0.sql.knn.funcs
	@$(MAKE) -s -f Makefile.sql-functions ml-ensemble-functions > neurondb--1.0.sql.ens.funcs
	@sed '/@ML_REGRESSION_FUNCTIONS@/r neurondb--1.0.sql.reg.funcs' neurondb--1.0.sql.tmp | \
	     sed '/@ML_REGRESSION_FUNCTIONS@/d' | \
	     sed '/@ML_CLASSIFICATION_FUNCTIONS@/r neurondb--1.0.sql.clf.funcs' | \
	     sed '/@ML_CLASSIFICATION_FUNCTIONS@/d' | \
	     sed '/@ML_KNN_FUNCTIONS@/r neurondb--1.0.sql.knn.funcs' | \
	     sed '/@ML_KNN_FUNCTIONS@/d' | \
	     sed '/@ML_ENSEMBLE_FUNCTIONS@/r neurondb--1.0.sql.ens.funcs' | \
	     sed '/@ML_ENSEMBLE_FUNCTIONS@/d' > neurondb--1.0.sql
	@rm -f neurondb--1.0.sql.tmp neurondb--1.0.sql.reg.funcs neurondb--1.0.sql.clf.funcs neurondb--1.0.sql.knn.funcs neurondb--1.0.sql.ens.funcs
	@echo "✓ Generated neurondb--1.0.sql for $(BUILD_TYPE)"
	@echo "=========================================================================="

# Ensure SQL exists before building
all: metal-shaders

# ONNX Runtime configuration
ONNX_RUNTIME_PATH = /usr/local/onnxruntime
ifneq ($(wildcard $(ONNX_RUNTIME_PATH)),)
    PG_CPPFLAGS += -I$(ONNX_RUNTIME_PATH)/include -DHAVE_ONNX_RUNTIME
    SHLIB_LINK += -L$(ONNX_RUNTIME_PATH)/lib -lonnxruntime -Wl,-rpath $(ONNX_RUNTIME_PATH)/lib
    $(info ✓ ONNX Runtime found at $(ONNX_RUNTIME_PATH))
else
    $(warning ⚠ ONNX Runtime not found at $(ONNX_RUNTIME_PATH) - building without HuggingFace support)
endif

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Custom target for GPU-only tests
installcheck-gpu:
	$(pg_regress_installcheck) $(REGRESS_OPTS) $(REGRESS_GPU)

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
# include Makefile.metal  # Removed - conflicting with main Makefile rules
include Makefile.metal.precompile
