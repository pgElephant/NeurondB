#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema, ListResourcesRequestSchema, ReadResourceRequestSchema, } from "@modelcontextprotocol/sdk/types.js";
import { Database } from "./db.js";
import { Resources } from "./resources.js";
import { VectorTools, MLTools, AnalyticsTools, RAGTools, ProjectTools, GPUTools, QuantizationTools, DimensionalityTools, DriftTools, MetricsTools, HybridTools, RerankingTools, IndexingTools, DataManagementTools, WorkerTools, } from "./tools/index.js";
import { ConfigManager } from "./config.js";
import { Logger } from "./logger.js";
import { MiddlewareManager } from "./middleware.js";
import { PluginManager } from "./plugin.js";
import { createLoggingMiddleware, createErrorHandlingMiddleware, createValidationMiddleware, createTimeoutMiddleware, } from "./middleware.js";
class NeurondbMCPServer {
    server;
    db;
    resources;
    vectorTools;
    mlTools;
    analyticsTools;
    ragTools;
    projectTools;
    gpuTools;
    quantizationTools;
    dimensionalityTools;
    driftTools;
    metricsTools;
    hybridTools;
    rerankingTools;
    indexingTools;
    dataManagementTools;
    workerTools;
    config;
    logger;
    middleware;
    pluginManager;
    constructor() {
        // Load configuration
        this.config = new ConfigManager();
        const serverConfig = this.config.load();
        // Initialize logger
        this.logger = new Logger(this.config.getLoggingConfig());
        // Initialize database
        this.db = new Database();
        // Initialize middleware
        this.middleware = new MiddlewareManager(this.logger);
        this.setupBuiltInMiddleware();
        // Initialize plugin manager
        this.pluginManager = new PluginManager(this.logger, this.db, this.middleware);
        // Initialize server
        const serverSettings = this.config.getServerSettings();
        this.server = new Server({
            name: serverSettings.name || "neurondb-mcp-server",
            version: serverSettings.version || "1.0.0",
        }, {
            capabilities: {
                tools: {},
                resources: {},
            },
        });
        // Initialize tools
        this.resources = new Resources(this.db);
        this.vectorTools = new VectorTools(this.db);
        this.mlTools = new MLTools(this.db);
        this.analyticsTools = new AnalyticsTools(this.db);
        this.ragTools = new RAGTools(this.db);
        this.projectTools = new ProjectTools(this.db);
        this.gpuTools = new GPUTools(this.db);
        this.quantizationTools = new QuantizationTools(this.db);
        this.dimensionalityTools = new DimensionalityTools(this.db);
        this.driftTools = new DriftTools(this.db);
        this.metricsTools = new MetricsTools(this.db);
        this.hybridTools = new HybridTools(this.db);
        this.rerankingTools = new RerankingTools(this.db);
        this.indexingTools = new IndexingTools(this.db);
        this.dataManagementTools = new DataManagementTools(this.db);
        this.workerTools = new WorkerTools(this.db);
        this.setupHandlers();
    }
    setupBuiltInMiddleware() {
        const serverSettings = this.config.getServerSettings();
        // Add built-in middleware - import at top level
        this.middleware.register(createValidationMiddleware());
        this.middleware.register(createLoggingMiddleware(this.logger));
        if (serverSettings.timeout) {
            this.middleware.register(createTimeoutMiddleware(serverSettings.timeout, this.logger));
        }
        this.middleware.register(createErrorHandlingMiddleware(this.logger));
    }
    setupHandlers() {
        this.setupTools();
        this.setupResources();
        this.setupToolHandlers();
    }
    setupTools() {
        this.server.setRequestHandler(ListToolsRequestSchema, async () => {
            const tools = [];
            // Add built-in tools based on feature flags
            const features = this.config.getFeaturesConfig();
            if (features.vector?.enabled) {
                tools.push({
                    name: "vector_search",
                    description: "Perform vector similarity search using L2, cosine, or inner product distance",
                    inputSchema: {
                        type: "object",
                        properties: {
                            table: { type: "string" },
                            vector_column: { type: "string" },
                            query_vector: { type: "array", items: { type: "number" } },
                            limit: { type: "number", default: features.vector?.defaultLimit || 10 },
                            distance_metric: {
                                type: "string",
                                enum: ["l2", "cosine", "inner_product"],
                                default: features.vector?.defaultDistanceMetric || "l2",
                            },
                        },
                        required: ["table", "vector_column", "query_vector"],
                    },
                }, {
                    name: "generate_embedding",
                    description: "Generate text embedding using configured model",
                    inputSchema: {
                        type: "object",
                        properties: {
                            text: { type: "string" },
                            model: { type: "string" },
                        },
                        required: ["text"],
                    },
                }, {
                    name: "batch_embedding",
                    description: "Generate embeddings for multiple texts efficiently",
                    inputSchema: {
                        type: "object",
                        properties: {
                            texts: { type: "array", items: { type: "string" } },
                            model: { type: "string" },
                        },
                        required: ["texts"],
                    },
                }, {
                    name: "create_hnsw_index",
                    description: "Create HNSW index for vector column",
                    inputSchema: {
                        type: "object",
                        properties: {
                            table: { type: "string" },
                            vector_column: { type: "string" },
                            index_name: { type: "string" },
                            m: { type: "number", default: 16 },
                            ef_construction: { type: "number", default: 200 },
                        },
                        required: ["table", "vector_column", "index_name"],
                    },
                }, {
                    name: "hybrid_search",
                    description: "Combine vector similarity and full-text search",
                    inputSchema: {
                        type: "object",
                        properties: {
                            table: { type: "string" },
                            query_vector: { type: "array", items: { type: "number" } },
                            query_text: { type: "string" },
                            text_column: { type: "string" },
                            vector_column: { type: "string" },
                            vector_weight: { type: "number", default: 0.7 },
                            limit: { type: "number", default: features.vector?.defaultLimit || 10 },
                        },
                        required: ["table", "query_vector", "query_text", "text_column", "vector_column"],
                    },
                });
            }
            if (features.ml?.enabled) {
                tools.push({
                    name: "train_ml_model",
                    description: `Train ML model. Supported algorithms: ${features.ml.algorithms?.join(", ") || "all"}`,
                    inputSchema: {
                        type: "object",
                        properties: {
                            table: { type: "string" },
                            feature_col: { type: "string" },
                            label_col: { type: "string" },
                            algorithm: {
                                type: "string",
                                enum: features.ml.algorithms || [
                                    "linear_regression",
                                    "ridge",
                                    "lasso",
                                    "logistic",
                                    "random_forest",
                                    "svm",
                                    "knn",
                                    "decision_tree",
                                    "naive_bayes",
                                ],
                            },
                            params: { type: "object" },
                        },
                        required: ["table", "feature_col", "label_col", "algorithm"],
                    },
                }, {
                    name: "predict_ml_model",
                    description: "Predict using trained ML model",
                    inputSchema: {
                        type: "object",
                        properties: {
                            model_id: { type: "number" },
                            features: { type: "array", items: { type: "number" } },
                        },
                        required: ["model_id", "features"],
                    },
                }, {
                    name: "get_model_info",
                    description: "Get information about registered ML models",
                    inputSchema: {
                        type: "object",
                        properties: {
                            model_id: { type: "number" },
                        },
                    },
                });
            }
            if (features.analytics?.enabled) {
                tools.push({
                    name: "cluster_data",
                    description: "Cluster vectors using kmeans, minibatch_kmeans, or gmm",
                    inputSchema: {
                        type: "object",
                        properties: {
                            table: { type: "string" },
                            vector_column: { type: "string" },
                            k: { type: "number", maximum: features.analytics?.maxClusters },
                            max_iter: {
                                type: "number",
                                default: 100,
                                maximum: features.analytics?.maxIterations,
                            },
                            algorithm: {
                                type: "string",
                                enum: ["kmeans", "minibatch_kmeans", "gmm"],
                                default: "kmeans",
                            },
                        },
                        required: ["table", "vector_column", "k"],
                    },
                }, {
                    name: "detect_outliers",
                    description: "Detect outliers using Z-score method",
                    inputSchema: {
                        type: "object",
                        properties: {
                            table: { type: "string" },
                            vector_column: { type: "string" },
                            threshold: { type: "number", default: 3.0 },
                        },
                        required: ["table", "vector_column"],
                    },
                });
            }
            if (features.rag?.enabled) {
                tools.push({
                    name: "rag_chunk_text",
                    description: "Chunk text for RAG pipeline",
                    inputSchema: {
                        type: "object",
                        properties: {
                            text: { type: "string" },
                            chunk_size: {
                                type: "number",
                                default: features.rag?.defaultChunkSize || 500,
                            },
                            overlap: {
                                type: "number",
                                default: features.rag?.defaultOverlap || 50,
                            },
                        },
                        required: ["text"],
                    },
                });
            }
            if (features.projects?.enabled) {
                tools.push({
                    name: "create_ml_project",
                    description: "Create ML project for model management",
                    inputSchema: {
                        type: "object",
                        properties: {
                            project_name: { type: "string" },
                            model_type: { type: "string" },
                            description: { type: "string" },
                        },
                        required: ["project_name", "model_type"],
                    },
                }, {
                    name: "list_ml_projects",
                    description: "List all ML projects",
                    inputSchema: {
                        type: "object",
                        properties: {},
                    },
                }, {
                    name: "train_kmeans_project",
                    description: "Train K-means model within a project",
                    inputSchema: {
                        type: "object",
                        properties: {
                            project_name: { type: "string" },
                            table_name: { type: "string" },
                            vector_col: { type: "string" },
                            num_clusters: { type: "number" },
                            max_iters: { type: "number", default: 100 },
                        },
                        required: ["project_name", "table_name", "vector_col", "num_clusters"],
                    },
                });
            }
            // GPU Tools - Always available
            tools.push({
                name: "gpu_info",
                description: "Get GPU information and status",
                inputSchema: { type: "object", properties: {} },
            }, {
                name: "gpu_stats",
                description: "Get GPU statistics (operations, memory, performance)",
                inputSchema: { type: "object", properties: {} },
            }, {
                name: "gpu_reset_stats",
                description: "Reset GPU statistics",
                inputSchema: { type: "object", properties: {} },
            }, {
                name: "gpu_enable",
                description: "Enable or disable GPU acceleration",
                inputSchema: {
                    type: "object",
                    properties: {
                        enabled: { type: "boolean" },
                    },
                    required: ["enabled"],
                },
            }, {
                name: "gpu_l2_distance",
                description: "Compute L2 distance on GPU",
                inputSchema: {
                    type: "object",
                    properties: {
                        vector_a: { type: "array", items: { type: "number" } },
                        vector_b: { type: "array", items: { type: "number" } },
                    },
                    required: ["vector_a", "vector_b"],
                },
            }, {
                name: "gpu_cosine_distance",
                description: "Compute cosine distance on GPU",
                inputSchema: {
                    type: "object",
                    properties: {
                        vector_a: { type: "array", items: { type: "number" } },
                        vector_b: { type: "array", items: { type: "number" } },
                    },
                    required: ["vector_a", "vector_b"],
                },
            }, {
                name: "gpu_inner_product",
                description: "Compute inner product on GPU",
                inputSchema: {
                    type: "object",
                    properties: {
                        vector_a: { type: "array", items: { type: "number" } },
                        vector_b: { type: "array", items: { type: "number" } },
                    },
                    required: ["vector_a", "vector_b"],
                },
            }, {
                name: "gpu_cluster_kmeans",
                description: "GPU-accelerated K-means clustering",
                inputSchema: {
                    type: "object",
                    properties: {
                        table: { type: "string" },
                        vector_column: { type: "string" },
                        k: { type: "number" },
                        max_iter: { type: "number", default: 100 },
                    },
                    required: ["table", "vector_column", "k"],
                },
            }, {
                name: "gpu_hnsw_search",
                description: "HNSW search on GPU",
                inputSchema: {
                    type: "object",
                    properties: {
                        table: { type: "string" },
                        vector_column: { type: "string" },
                        query_vector: { type: "array", items: { type: "number" } },
                        limit: { type: "number", default: 10 },
                    },
                    required: ["table", "vector_column", "query_vector"],
                },
            });
            // Quantization Tools
            tools.push({
                name: "quantize_int8",
                description: "Quantize vector to INT8",
                inputSchema: {
                    type: "object",
                    properties: {
                        vector: { type: "array", items: { type: "number" } },
                    },
                    required: ["vector"],
                },
            }, {
                name: "quantize_fp16",
                description: "Quantize vector to FP16",
                inputSchema: {
                    type: "object",
                    properties: {
                        vector: { type: "array", items: { type: "number" } },
                    },
                    required: ["vector"],
                },
            }, {
                name: "quantize_binary",
                description: "Quantize vector to binary",
                inputSchema: {
                    type: "object",
                    properties: {
                        vector: { type: "array", items: { type: "number" } },
                    },
                    required: ["vector"],
                },
            }, {
                name: "train_pq_codebook",
                description: "Train Product Quantization (PQ) codebook",
                inputSchema: {
                    type: "object",
                    properties: {
                        table: { type: "string" },
                        vector_column: { type: "string" },
                        num_subvectors: { type: "number" },
                        num_centroids: { type: "number" },
                        max_iter: { type: "number", default: 50 },
                    },
                    required: ["table", "vector_column", "num_subvectors", "num_centroids"],
                },
            }, {
                name: "train_opq_codebook",
                description: "Train Optimized Product Quantization (OPQ) codebook",
                inputSchema: {
                    type: "object",
                    properties: {
                        table: { type: "string" },
                        vector_column: { type: "string" },
                        num_subvectors: { type: "number" },
                        num_centroids: { type: "number" },
                        max_iter: { type: "number", default: 50 },
                    },
                    required: ["table", "vector_column", "num_subvectors", "num_centroids"],
                },
            });
            // Dimensionality Reduction Tools
            tools.push({
                name: "reduce_pca",
                description: "Reduce dimensionality using PCA",
                inputSchema: {
                    type: "object",
                    properties: {
                        table: { type: "string" },
                        vector_column: { type: "string" },
                        target_dimensions: { type: "number" },
                    },
                    required: ["table", "vector_column", "target_dimensions"],
                },
            }, {
                name: "whiten_embeddings",
                description: "Apply PCA whitening to embeddings",
                inputSchema: {
                    type: "object",
                    properties: {
                        table: { type: "string" },
                        vector_column: { type: "string" },
                    },
                    required: ["table", "vector_column"],
                },
            });
            // Drift Detection Tools
            tools.push({
                name: "detect_centroid_drift",
                description: "Detect centroid drift between baseline and current data",
                inputSchema: {
                    type: "object",
                    properties: {
                        baseline_table: { type: "string" },
                        baseline_column: { type: "string" },
                        current_table: { type: "string" },
                        current_column: { type: "string" },
                        filter_column: { type: "string" },
                        filter_value: { type: "string" },
                        threshold: { type: "number", default: 0.3 },
                    },
                    required: ["baseline_table", "baseline_column", "current_table", "current_column"],
                },
            }, {
                name: "detect_distribution_divergence",
                description: "Detect distribution divergence (KL/JS divergence)",
                inputSchema: {
                    type: "object",
                    properties: {
                        baseline_table: { type: "string" },
                        baseline_column: { type: "string" },
                        current_table: { type: "string" },
                        current_column: { type: "string" },
                        method: { type: "string", enum: ["kl", "js"], default: "kl" },
                    },
                    required: ["baseline_table", "baseline_column", "current_table", "current_column"],
                },
            });
            // Metrics Tools
            tools.push({
                name: "recall_at_k",
                description: "Calculate Recall@K metric",
                inputSchema: {
                    type: "object",
                    properties: {
                        ground_truth_table: { type: "string" },
                        ground_truth_column: { type: "string" },
                        predictions_table: { type: "string" },
                        predictions_column: { type: "string" },
                        query_column: { type: "string" },
                        query_id: { type: "number" },
                        k: { type: "number" },
                    },
                    required: ["ground_truth_table", "ground_truth_column", "predictions_table", "predictions_column", "query_column", "query_id", "k"],
                },
            }, {
                name: "precision_at_k",
                description: "Calculate Precision@K metric",
                inputSchema: {
                    type: "object",
                    properties: {
                        ground_truth_table: { type: "string" },
                        ground_truth_column: { type: "string" },
                        predictions_table: { type: "string" },
                        predictions_column: { type: "string" },
                        query_column: { type: "string" },
                        query_id: { type: "number" },
                        k: { type: "number" },
                    },
                    required: ["ground_truth_table", "ground_truth_column", "predictions_table", "predictions_column", "query_column", "query_id", "k"],
                },
            }, {
                name: "f1_at_k",
                description: "Calculate F1@K metric",
                inputSchema: {
                    type: "object",
                    properties: {
                        ground_truth_table: { type: "string" },
                        ground_truth_column: { type: "string" },
                        predictions_table: { type: "string" },
                        predictions_column: { type: "string" },
                        query_column: { type: "string" },
                        query_id: { type: "number" },
                        k: { type: "number" },
                    },
                    required: ["ground_truth_table", "ground_truth_column", "predictions_table", "predictions_column", "query_column", "query_id", "k"],
                },
            }, {
                name: "mean_reciprocal_rank",
                description: "Calculate Mean Reciprocal Rank (MRR)",
                inputSchema: {
                    type: "object",
                    properties: {
                        ground_truth_table: { type: "string" },
                        ground_truth_column: { type: "string" },
                        predictions_table: { type: "string" },
                        predictions_column: { type: "string" },
                        rank_column: { type: "string" },
                        query_column: { type: "string" },
                    },
                    required: ["ground_truth_table", "ground_truth_column", "predictions_table", "predictions_column", "rank_column", "query_column"],
                },
            }, {
                name: "clustering_metrics",
                description: "Calculate clustering metrics (Davies-Bouldin, Silhouette)",
                inputSchema: {
                    type: "object",
                    properties: {
                        table: { type: "string" },
                        vector_column: { type: "string" },
                        cluster_column: { type: "string" },
                        metric: { type: "string", enum: ["davies_bouldin", "silhouette"], default: "davies_bouldin" },
                    },
                    required: ["table", "vector_column", "cluster_column"],
                },
            });
            // Hybrid Search Tools
            tools.push({
                name: "hybrid_search_fusion",
                description: "Fuse semantic and lexical search results",
                inputSchema: {
                    type: "object",
                    properties: {
                        semantic_table: { type: "string" },
                        lexical_table: { type: "string" },
                        id_column: { type: "string" },
                        semantic_score_column: { type: "string" },
                        lexical_score_column: { type: "string" },
                        alpha: { type: "number", default: 0.5 },
                    },
                    required: ["semantic_table", "lexical_table", "id_column", "semantic_score_column", "lexical_score_column"],
                },
            }, {
                name: "ltr_rerank",
                description: "Learning to Rank (LTR) reranking",
                inputSchema: {
                    type: "object",
                    properties: {
                        query: { type: "string" },
                        documents: { type: "array", items: { type: "string" } },
                        features: { type: "object" },
                    },
                    required: ["query", "documents"],
                },
            });
            // Reranking Tools
            tools.push({
                name: "mmr_rerank",
                description: "Rerank using Maximal Marginal Relevance (MMR)",
                inputSchema: {
                    type: "object",
                    properties: {
                        table: { type: "string" },
                        query_vector: { type: "array", items: { type: "number" } },
                        vector_column: { type: "string" },
                        lambda: { type: "number", default: 0.5 },
                        top_k: { type: "number", default: 10 },
                    },
                    required: ["table", "query_vector", "vector_column"],
                },
            }, {
                name: "rerank_cross_encoder",
                description: "Rerank using cross-encoder model",
                inputSchema: {
                    type: "object",
                    properties: {
                        query: { type: "string" },
                        documents: { type: "array", items: { type: "string" } },
                        model: { type: "string" },
                        top_k: { type: "number" },
                    },
                    required: ["query", "documents"],
                },
            }, {
                name: "rerank_llm",
                description: "Rerank using LLM",
                inputSchema: {
                    type: "object",
                    properties: {
                        query: { type: "string" },
                        documents: { type: "array", items: { type: "string" } },
                        model: { type: "string" },
                        top_k: { type: "number" },
                    },
                    required: ["query", "documents"],
                },
            });
            // Indexing Tools
            tools.push({
                name: "create_ivf_index",
                description: "Create IVF index for vector column",
                inputSchema: {
                    type: "object",
                    properties: {
                        table: { type: "string" },
                        vector_column: { type: "string" },
                        index_name: { type: "string" },
                        num_lists: { type: "number", default: 100 },
                    },
                    required: ["table", "vector_column", "index_name"],
                },
            }, {
                name: "rebalance_index",
                description: "Rebalance index",
                inputSchema: {
                    type: "object",
                    properties: {
                        index_name: { type: "string" },
                        threshold: { type: "number", default: 0.8 },
                    },
                    required: ["index_name"],
                },
            }, {
                name: "get_index_stats",
                description: "Get index statistics",
                inputSchema: {
                    type: "object",
                    properties: {
                        index_name: { type: "string" },
                    },
                    required: ["index_name"],
                },
            }, {
                name: "drop_index",
                description: "Drop index",
                inputSchema: {
                    type: "object",
                    properties: {
                        index_name: { type: "string" },
                    },
                    required: ["index_name"],
                },
            });
            // Data Management Tools
            tools.push({
                name: "vacuum_vectors",
                description: "Vacuum vectors (clean up unused space)",
                inputSchema: {
                    type: "object",
                    properties: {
                        table: { type: "string" },
                        dry_run: { type: "boolean", default: false },
                    },
                    required: ["table"],
                },
            }, {
                name: "compress_cold_tier",
                description: "Compress cold tier vectors (older than threshold)",
                inputSchema: {
                    type: "object",
                    properties: {
                        table: { type: "string" },
                        days_threshold: { type: "number", default: 30 },
                    },
                    required: ["table"],
                },
            }, {
                name: "sync_index_async",
                description: "Sync index to replica (async)",
                inputSchema: {
                    type: "object",
                    properties: {
                        index_name: { type: "string" },
                        replica_host: { type: "string" },
                    },
                    required: ["index_name", "replica_host"],
                },
            });
            // Worker Tools
            tools.push({
                name: "run_queue_worker",
                description: "Run queue worker once",
                inputSchema: { type: "object", properties: {} },
            }, {
                name: "sample_tuner",
                description: "Sample tuner worker",
                inputSchema: { type: "object", properties: {} },
            }, {
                name: "get_worker_status",
                description: "Get worker status",
                inputSchema: { type: "object", properties: {} },
            });
            // Always include execute_sql for flexibility
            tools.push({
                name: "execute_sql",
                description: "Execute arbitrary SQL query (use with caution)",
                inputSchema: {
                    type: "object",
                    properties: {
                        query: { type: "string" },
                    },
                    required: ["query"],
                },
            });
            // Add plugin tools
            const pluginTools = this.pluginManager.getAllTools();
            tools.push(...pluginTools);
            return { tools };
        });
    }
    setupResources() {
        this.server.setRequestHandler(ListResourcesRequestSchema, async () => {
            const resources = [
                {
                    uri: "neurondb://schema",
                    name: "Neurondb Schema",
                    description: "Database schema information",
                    mimeType: "application/json",
                },
                {
                    uri: "neurondb://models",
                    name: "ML Models Catalog",
                    description: "List of registered ML models",
                    mimeType: "application/json",
                },
                {
                    uri: "neurondb://indexes",
                    name: "Vector Indexes",
                    description: "Status and information about vector indexes",
                    mimeType: "application/json",
                },
                {
                    uri: "neurondb://config",
                    name: "Neurondb Configuration",
                    description: "Current Neurondb configuration settings",
                    mimeType: "application/json",
                },
                {
                    uri: "neurondb://workers",
                    name: "Background Workers Status",
                    description: "Status of background workers",
                    mimeType: "application/json",
                },
                {
                    uri: "neurondb://vector_stats",
                    name: "Vector Statistics",
                    description: "Aggregate vector statistics",
                    mimeType: "application/json",
                },
                {
                    uri: "neurondb://index_health",
                    name: "Index Health",
                    description: "Index health dashboard",
                    mimeType: "application/json",
                },
            ];
            // Add plugin resources
            const pluginResources = this.pluginManager.getAllResources();
            resources.push(...pluginResources);
            return { resources };
        });
        this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
            const { uri } = request.params;
            try {
                let data;
                // Check plugin resources first
                const pluginResources = this.pluginManager.getAllResources();
                const pluginResource = pluginResources.find((r) => r.uri === uri);
                if (pluginResource) {
                    data = await pluginResource.handler();
                }
                else {
                    // Built-in resources
                    switch (uri) {
                        case "neurondb://schema":
                            data = await this.resources.getSchema();
                            break;
                        case "neurondb://models":
                            data = await this.resources.getModels();
                            break;
                        case "neurondb://indexes":
                            data = await this.resources.getIndexes();
                            break;
                        case "neurondb://config":
                            data = await this.resources.getConfig();
                            break;
                        case "neurondb://workers":
                            data = await this.resources.getWorkerStatus();
                            break;
                        case "neurondb://vector_stats":
                            data = await this.resources.getVectorStats();
                            break;
                        case "neurondb://index_health":
                            data = await this.resources.getIndexHealth();
                            break;
                        default:
                            throw new Error(`Unknown resource URI: ${uri}`);
                    }
                }
                return {
                    contents: [
                        {
                            uri,
                            mimeType: "application/json",
                            text: JSON.stringify(data, null, 2),
                        },
                    ],
                };
            }
            catch (error) {
                throw new Error(`Failed to read resource ${uri}: ${error instanceof Error ? error.message : String(error)}`);
            }
        });
    }
    setupToolHandlers() {
        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            const { name, arguments: args } = request.params;
            const mcpRequest = {
                method: name,
                params: args,
            };
            const response = await this.middleware.execute(mcpRequest, async () => {
                try {
                    let result;
                    // Check plugin tools first
                    const pluginTools = this.pluginManager.getAllTools();
                    const pluginTool = pluginTools.find((t) => t.name === name);
                    if (pluginTool) {
                        result = await pluginTool.handler(args);
                    }
                    else {
                        // Built-in tools
                        result = await this.handleBuiltInTool(name, args);
                    }
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify(result, null, 2),
                            },
                        ],
                    };
                }
                catch (error) {
                    this.logger.error(`Tool execution error: ${name}`, error, { args });
                    return {
                        content: [
                            {
                                type: "text",
                                text: `Error: ${error instanceof Error ? error.message : String(error)}`,
                            },
                        ],
                        isError: true,
                    };
                }
            });
            return response;
        });
    }
    async handleBuiltInTool(name, args) {
        const features = this.config.getFeaturesConfig();
        switch (name) {
            case "vector_search":
                if (!features.vector?.enabled)
                    throw new Error("Vector features are disabled");
                return this.vectorTools.vectorSearch(args);
            case "generate_embedding":
                if (!features.vector?.enabled)
                    throw new Error("Vector features are disabled");
                return this.vectorTools.generateEmbedding(args);
            case "batch_embedding":
                if (!features.vector?.enabled)
                    throw new Error("Vector features are disabled");
                return this.vectorTools.batchEmbedding(args);
            case "create_hnsw_index":
                if (!features.vector?.enabled)
                    throw new Error("Vector features are disabled");
                return this.vectorTools.createHNSWIndex(args);
            case "hybrid_search":
                if (!features.vector?.enabled)
                    throw new Error("Vector features are disabled");
                return this.vectorTools.hybridSearch(args);
            case "train_ml_model":
                if (!features.ml?.enabled)
                    throw new Error("ML features are disabled");
                return this.trainMLModel(args);
            case "predict_ml_model":
                if (!features.ml?.enabled)
                    throw new Error("ML features are disabled");
                return this.mlTools.predict(args);
            case "cluster_data":
                if (!features.analytics?.enabled)
                    throw new Error("Analytics features are disabled");
                return this.clusterData(args);
            case "detect_outliers":
                if (!features.analytics?.enabled)
                    throw new Error("Analytics features are disabled");
                return this.analyticsTools.detectOutliersZScore(args);
            case "rag_chunk_text":
                if (!features.rag?.enabled)
                    throw new Error("RAG features are disabled");
                return this.ragTools.chunkText(args);
            
            
            // Reranking tools - All 7 tools routed through RerankingTools.handleToolCall()
            // This ensures consistent parameter handling and error management
            case "mmr_rerank":
            case "rerank_cross_encoder":
            case "rerank_llm":
            case "rerank_colbert":
            case "rerank_rrf":
            case "rerank_ensemble_weighted":
            case "rerank_ensemble_borda":
                if (features.reranking?.enabled === false) {
                    throw new Error("Reranking features are disabled");
                }
                // handleToolCall() maps tool names to methods and handles parameter conversion
                return await this.rerankingTools.handleToolCall(name, args);
            case "create_ml_project":
                if (!features.projects?.enabled)
                    throw new Error("Project features are disabled");
                return this.projectTools.createProject(args);
            case "list_ml_projects":
                if (!features.projects?.enabled)
                    throw new Error("Project features are disabled");
                return this.projectTools.listProjects();
            case "train_kmeans_project":
                if (!features.projects?.enabled)
                    throw new Error("Project features are disabled");
                return this.projectTools.trainKMeansProject(args);
            case "get_model_info":
                if (!features.ml?.enabled)
                    throw new Error("ML features are disabled");
                return this.mlTools.getModelInfo(args.model_id);
            // GPU Tools
            case "gpu_info":
                return this.gpuTools.getGPUInfo();
            case "gpu_stats":
                return this.gpuTools.getGPUStats();
            case "gpu_reset_stats":
                return this.gpuTools.resetGPUStats();
            case "gpu_enable":
                return this.gpuTools.enableGPU(args.enabled);
            case "gpu_l2_distance":
                return this.gpuTools.l2DistanceGPU(args.vector_a, args.vector_b);
            case "gpu_cosine_distance":
                return this.gpuTools.cosineDistanceGPU(args.vector_a, args.vector_b);
            case "gpu_inner_product":
                return this.gpuTools.innerProductGPU(args.vector_a, args.vector_b);
            case "gpu_cluster_kmeans":
                return this.gpuTools.clusterKMeansGPU(args.table, args.vector_column, args.k, args.max_iter);
            case "gpu_hnsw_search":
                return this.gpuTools.hnswSearchGPU(args.table, args.vector_column, args.query_vector, args.limit);
            // Quantization Tools
            case "quantize_int8":
                return this.quantizationTools.quantizeINT8(args.vector);
            case "quantize_fp16":
                return this.quantizationTools.quantizeFP16(args.vector);
            case "quantize_binary":
                return this.quantizationTools.quantizeBinary(args.vector);
            case "train_pq_codebook":
                return this.quantizationTools.trainPQCodebook(args.table, args.vector_column, args.num_subvectors, args.num_centroids, args.max_iter);
            case "train_opq_codebook":
                return this.quantizationTools.trainOPQCodebook(args.table, args.vector_column, args.num_subvectors, args.num_centroids, args.max_iter);
            // Dimensionality Reduction Tools
            case "reduce_pca":
                return this.dimensionalityTools.reducePCA(args.table, args.vector_column, args.target_dimensions);
            case "whiten_embeddings":
                return this.dimensionalityTools.whitenEmbeddings(args.table, args.vector_column);
            // Drift Detection Tools
            case "detect_centroid_drift":
                return this.driftTools.detectCentroidDrift(args.baseline_table, args.baseline_column, args.current_table, args.current_column, args.filter_column, args.filter_value, args.threshold);
            case "detect_distribution_divergence":
                return this.driftTools.detectDistributionDivergence(args.baseline_table, args.baseline_column, args.current_table, args.current_column, args.method);
            // Metrics Tools
            case "recall_at_k":
                return this.metricsTools.recallAtK(args.ground_truth_table, args.ground_truth_column, args.predictions_table, args.predictions_column, args.query_column, args.query_id, args.k);
            case "precision_at_k":
                return this.metricsTools.precisionAtK(args.ground_truth_table, args.ground_truth_column, args.predictions_table, args.predictions_column, args.query_column, args.query_id, args.k);
            case "f1_at_k":
                return this.metricsTools.f1AtK(args.ground_truth_table, args.ground_truth_column, args.predictions_table, args.predictions_column, args.query_column, args.query_id, args.k);
            case "mean_reciprocal_rank":
                return this.metricsTools.meanReciprocalRank(args.ground_truth_table, args.ground_truth_column, args.predictions_table, args.predictions_column, args.rank_column, args.query_column);
            case "clustering_metrics":
                return this.metricsTools.clusteringMetrics(args.table, args.vector_column, args.cluster_column, args.metric);
            // Hybrid Search Tools
            case "hybrid_search_fusion":
                return this.hybridTools.hybridSearchFusion(args.semantic_table, args.lexical_table, args.id_column, args.semantic_score_column, args.lexical_score_column, args.alpha);
            case "ltr_rerank":
                return this.hybridTools.ltrRerankPointwise(args.query, args.documents, args.features);
                        // Reranking Tools - All 7 tools from RerankingTools.getToolDefinitions()
            // Includes: mmr_rerank, rerank_cross_encoder, rerank_llm (full params),
            //           rerank_colbert, rerank_rrf, rerank_ensemble_weighted, rerank_ensemble_borda
            if (features.reranking?.enabled !== false) {
                const rerankingToolDefs = this.rerankingTools.getToolDefinitions();
                tools.push(...rerankingToolDefs);
            }
            // Indexing Tools
            case "create_ivf_index":
                return this.indexingTools.createIVFIndex(args.table, args.vector_column, args.index_name, args.num_lists);
            case "rebalance_index":
                return this.indexingTools.rebalanceIndex(args.index_name, args.threshold);
            case "get_index_stats":
                return this.indexingTools.getIndexStats(args.index_name);
            case "drop_index":
                return this.indexingTools.dropIndex(args.index_name);
            // Data Management Tools
            case "vacuum_vectors":
                return this.dataManagementTools.vacuumVectors(args.table, args.dry_run);
            case "compress_cold_tier":
                return this.dataManagementTools.compressColdTier(args.table, args.days_threshold);
            case "sync_index_async":
                return this.dataManagementTools.syncIndexAsync(args.index_name, args.replica_host);
            // Worker Tools
            case "run_queue_worker":
                return this.workerTools.runQueueWorker();
            case "sample_tuner":
                return this.workerTools.sampleTuner();
            case "get_worker_status":
                return this.workerTools.getWorkerStatus();
            case "execute_sql":
                const sqlResult = await this.db.query(args.query);
                return sqlResult.rows;
            default:
                throw new Error(`Unknown tool: ${name}`);
        }
    }
    async trainMLModel(args) {
        const { algorithm, ...rest } = args;
        const trainingParams = { ...rest, params: args.params };
        switch (algorithm) {
            case "linear_regression":
                return this.mlTools.trainLinearRegression(trainingParams);
            case "ridge":
                return this.mlTools.trainRidgeRegression(trainingParams);
            case "lasso":
                return this.mlTools.trainLassoRegression(trainingParams);
            case "logistic":
                return this.mlTools.trainLogisticRegression(trainingParams);
            case "random_forest":
                return this.mlTools.trainRandomForest(trainingParams);
            case "svm":
                return this.mlTools.trainSVM(trainingParams);
            case "knn":
                return this.mlTools.trainKNN(trainingParams);
            case "decision_tree":
                return this.mlTools.trainDecisionTree(trainingParams);
            case "naive_bayes":
                return this.mlTools.trainNaiveBayes(trainingParams);
            default:
                throw new Error(`Unknown algorithm: ${algorithm}`);
        }
    }
    async clusterData(args) {
        const { algorithm = "kmeans", ...rest } = args;
        switch (algorithm) {
            case "kmeans":
                return this.analyticsTools.clusterKMeans(rest);
            case "minibatch_kmeans":
                return this.analyticsTools.clusterMiniBatchKMeans(rest);
            case "gmm":
                return this.analyticsTools.clusterGMM(rest);
            default:
                throw new Error(`Unknown clustering algorithm: ${algorithm}`);
        }
    }
    
    async connect() {
        try {
            const dbConfig = this.config.getDatabaseConfig();
            this.db.connect(dbConfig);
            await this.db.testConnection();
            this.logger.info("Connected to database", {
                host: dbConfig.host,
                database: dbConfig.database,
            });
        }
        catch (error) {
            this.logger.error("Failed to connect to database", error);
            throw error;
        }
    }
    async loadPlugins() {
        const plugins = this.config.getPlugins();
        for (const pluginConfig of plugins) {
            try {
                await this.pluginManager.loadPlugin(pluginConfig);
            }
            catch (error) {
                this.logger.error(`Failed to load plugin: ${pluginConfig.name}`, error);
            }
        }
    }
    async start() {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        // Load plugins
        await this.loadPlugins();
        // Connect to database
        try {
            await this.connect();
        }
        catch (error) {
            this.logger.warn("Database connection failed, server will start but tools may fail");
        }
        this.logger.info("Neurondb MCP server running on stdio");
    }
    async stop() {
        this.logger.info("Shutting down server");
        await this.pluginManager.shutdown();
        await this.db.close();
    }
}
const server = new NeurondbMCPServer();
server.start().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
});
process.on("SIGINT", async () => {
    await server.stop();
    process.exit(0);
});
process.on("SIGTERM", async () => {
    await server.stop();
    process.exit(0);
});
//# sourceMappingURL=index.js.map