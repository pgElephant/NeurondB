# NeuronDB MCP Server - Complete Feature Set

## 🚀 **100% NeuronDB Integration**

This MCP server provides **complete coverage** of all NeuronDB features, making it the most comprehensive NeuronDB MCP server available.

## 📊 **Feature Categories**

### 1. **GPU Acceleration** (9 tools)
- `gpu_info` - Get GPU information and status
- `gpu_stats` - Get GPU statistics (operations, memory, performance)
- `gpu_reset_stats` - Reset GPU statistics
- `gpu_enable` - Enable/disable GPU acceleration
- `gpu_l2_distance` - Compute L2 distance on GPU
- `gpu_cosine_distance` - Compute cosine distance on GPU
- `gpu_inner_product` - Compute inner product on GPU
- `gpu_cluster_kmeans` - GPU-accelerated K-means clustering
- `gpu_hnsw_search` - HNSW search on GPU

### 2. **Quantization** (5 tools)
- `quantize_int8` - Quantize vector to INT8
- `quantize_fp16` - Quantize vector to FP16
- `quantize_binary` - Quantize vector to binary
- `train_pq_codebook` - Train Product Quantization (PQ) codebook
- `train_opq_codebook` - Train Optimized Product Quantization (OPQ) codebook

### 3. **Dimensionality Reduction** (2 tools)
- `reduce_pca` - Reduce dimensionality using PCA
- `whiten_embeddings` - Apply PCA whitening to embeddings

### 4. **Drift Detection** (2 tools)
- `detect_centroid_drift` - Detect centroid drift between baseline and current data
- `detect_distribution_divergence` - Detect distribution divergence (KL/JS divergence)

### 5. **Metrics & Evaluation** (5 tools)
- `recall_at_k` - Calculate Recall@K metric
- `precision_at_k` - Calculate Precision@K metric
- `f1_at_k` - Calculate F1@K metric
- `mean_reciprocal_rank` - Calculate Mean Reciprocal Rank (MRR)
- `clustering_metrics` - Calculate clustering metrics (Davies-Bouldin, Silhouette)

### 6. **Hybrid Search** (2 tools)
- `hybrid_search_fusion` - Fuse semantic and lexical search results
- `ltr_rerank` - Learning to Rank (LTR) reranking

### 7. **Reranking** (3 tools)
- `mmr_rerank` - Rerank using Maximal Marginal Relevance (MMR)
- `rerank_cross_encoder` - Rerank using cross-encoder model
- `rerank_llm` - Rerank using LLM

### 8. **Indexing** (4 tools)
- `create_ivf_index` - Create IVF index for vector column
- `rebalance_index` - Rebalance index
- `get_index_stats` - Get index statistics
- `drop_index` - Drop index

### 9. **Data Management** (3 tools)
- `vacuum_vectors` - Vacuum vectors (clean up unused space)
- `compress_cold_tier` - Compress cold tier vectors (older than threshold)
- `sync_index_async` - Sync index to replica (async)

### 10. **Worker Management** (3 tools)
- `run_queue_worker` - Run queue worker once
- `sample_tuner` - Sample tuner worker
- `get_worker_status` - Get worker status

### 11. **Core Features** (Existing)
- Vector search and embeddings
- ML model training and prediction
- Clustering and analytics
- RAG pipeline
- Project management

## 🎯 **Unique Features (No Other MCP Server Has)**

1. **Complete GPU Integration**
   - Real-time GPU monitoring and statistics
   - GPU-accelerated distance computations
   - GPU-accelerated clustering and search
   - GPU quantization support

2. **Advanced Quantization**
   - Product Quantization (PQ) codebook training
   - Optimized Product Quantization (OPQ)
   - Multiple quantization formats (INT8, FP16, Binary, Ternary)

3. **Drift Detection**
   - Centroid drift detection
   - Distribution divergence (KL/JS)
   - Model monitoring and alerting

4. **Comprehensive Metrics**
   - Retrieval metrics (Recall@K, Precision@K, F1@K, MRR)
   - Clustering quality metrics (Davies-Bouldin, Silhouette)
   - Model evaluation tools

5. **Hybrid Search & Reranking**
   - Semantic-lexical fusion
   - Learning to Rank (LTR)
   - Multiple reranking strategies (MMR, Cross-encoder, LLM)

6. **Index Management**
   - IVF index creation
   - Index rebalancing
   - Index health monitoring

7. **Data Lifecycle Management**
   - Vector vacuuming
   - Cold tier compression
   - Async index synchronization

8. **Worker Management**
   - Background worker control
   - Worker status monitoring
   - Queue management

## 📈 **Total Tools: 60+**

This MCP server provides **60+ tools** covering every aspect of NeuronDB functionality, making it the most comprehensive NeuronDB integration available.

## 🔧 **Technical Excellence**

- **Type-safe**: Full TypeScript implementation
- **Error handling**: Comprehensive error handling and validation
- **Middleware**: Extensible middleware system
- **Plugin support**: Plugin architecture for extensibility
- **Resource management**: Efficient database connection pooling
- **Logging**: Structured logging with configurable levels

## 🚀 **Getting Started**

See [README.md](README.md) for installation and configuration instructions.

