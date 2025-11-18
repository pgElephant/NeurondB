#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema, ListResourcesRequestSchema, ReadResourceRequestSchema, } from "@modelcontextprotocol/sdk/types.js";
import { Database } from "./db.js";
import { Resources } from "./resources.js";
import { VectorTools, MLTools, AnalyticsTools, RAGTools, ProjectTools, } from "./tools/index.js";
class NeurondbMCPServer {
    server;
    db;
    resources;
    vectorTools;
    mlTools;
    analyticsTools;
    ragTools;
    projectTools;
    constructor() {
        this.server = new Server({
            name: "neurondb-mcp-server",
            version: "1.0.0",
        }, {
            capabilities: {
                tools: {},
                resources: {},
            },
        });
        this.db = new Database();
        this.resources = new Resources(this.db);
        this.vectorTools = new VectorTools(this.db);
        this.mlTools = new MLTools(this.db);
        this.analyticsTools = new AnalyticsTools(this.db);
        this.ragTools = new RAGTools(this.db);
        this.projectTools = new ProjectTools(this.db);
        this.setupHandlers();
    }
    setupHandlers() {
        this.setupTools();
        this.setupResources();
        this.setupToolHandlers();
    }
    setupTools() {
        this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
            tools: [
                {
                    name: "vector_search",
                    description: "Perform vector similarity search using L2, cosine, or inner product distance",
                    inputSchema: {
                        type: "object",
                        properties: {
                            table: { type: "string" },
                            vector_column: { type: "string" },
                            query_vector: {
                                type: "array",
                                items: { type: "number" },
                            },
                            limit: { type: "number", default: 10 },
                            distance_metric: {
                                type: "string",
                                enum: ["l2", "cosine", "inner_product"],
                                default: "l2",
                            },
                        },
                        required: ["table", "vector_column", "query_vector"],
                    },
                },
                {
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
                },
                {
                    name: "batch_embedding",
                    description: "Generate embeddings for multiple texts efficiently",
                    inputSchema: {
                        type: "object",
                        properties: {
                            texts: {
                                type: "array",
                                items: { type: "string" },
                            },
                            model: { type: "string" },
                        },
                        required: ["texts"],
                    },
                },
                {
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
                },
                {
                    name: "hybrid_search",
                    description: "Combine vector similarity and full-text search",
                    inputSchema: {
                        type: "object",
                        properties: {
                            table: { type: "string" },
                            query_vector: {
                                type: "array",
                                items: { type: "number" },
                            },
                            query_text: { type: "string" },
                            text_column: { type: "string" },
                            vector_column: { type: "string" },
                            vector_weight: { type: "number", default: 0.7 },
                            limit: { type: "number", default: 10 },
                        },
                        required: [
                            "table",
                            "query_vector",
                            "query_text",
                            "text_column",
                            "vector_column",
                        ],
                    },
                },
                {
                    name: "train_ml_model",
                    description: "Train ML model (linear_regression, ridge, lasso, logistic, random_forest, svm, knn, decision_tree, naive_bayes)",
                    inputSchema: {
                        type: "object",
                        properties: {
                            table: { type: "string" },
                            feature_col: { type: "string" },
                            label_col: { type: "string" },
                            algorithm: {
                                type: "string",
                                enum: [
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
                },
                {
                    name: "predict_ml_model",
                    description: "Predict using trained ML model",
                    inputSchema: {
                        type: "object",
                        properties: {
                            model_id: { type: "number" },
                            features: {
                                type: "array",
                                items: { type: "number" },
                            },
                        },
                        required: ["model_id", "features"],
                    },
                },
                {
                    name: "cluster_data",
                    description: "Cluster vectors using kmeans, minibatch_kmeans, or gmm",
                    inputSchema: {
                        type: "object",
                        properties: {
                            table: { type: "string" },
                            vector_column: { type: "string" },
                            k: { type: "number" },
                            max_iter: { type: "number", default: 100 },
                            algorithm: {
                                type: "string",
                                enum: ["kmeans", "minibatch_kmeans", "gmm"],
                                default: "kmeans",
                            },
                        },
                        required: ["table", "vector_column", "k"],
                    },
                },
                {
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
                },
                {
                    name: "rag_chunk_text",
                    description: "Chunk text for RAG pipeline",
                    inputSchema: {
                        type: "object",
                        properties: {
                            text: { type: "string" },
                            chunk_size: { type: "number", default: 500 },
                            overlap: { type: "number", default: 50 },
                        },
                        required: ["text"],
                    },
                },
                {
                    name: "rerank_results",
                    description: "Rerank search results using cross-encoder or LLM",
                    inputSchema: {
                        type: "object",
                        properties: {
                            query: { type: "string" },
                            documents: {
                                type: "array",
                                items: { type: "string" },
                            },
                            method: {
                                type: "string",
                                enum: ["cross_encoder", "llm"],
                                default: "cross_encoder",
                            },
                            model: { type: "string" },
                            top_k: { type: "number", default: 10 },
                        },
                        required: ["query", "documents"],
                    },
                },
                {
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
                },
                {
                    name: "list_ml_projects",
                    description: "List all ML projects",
                    inputSchema: {
                        type: "object",
                        properties: {},
                    },
                },
                {
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
                        required: [
                            "project_name",
                            "table_name",
                            "vector_col",
                            "num_clusters",
                        ],
                    },
                },
                {
                    name: "get_model_info",
                    description: "Get information about registered ML models",
                    inputSchema: {
                        type: "object",
                        properties: {
                            model_id: { type: "number" },
                        },
                    },
                },
                {
                    name: "execute_sql",
                    description: "Execute arbitrary SQL query",
                    inputSchema: {
                        type: "object",
                        properties: {
                            query: { type: "string" },
                        },
                        required: ["query"],
                    },
                },
            ],
        }));
    }
    setupResources() {
        this.server.setRequestHandler(ListResourcesRequestSchema, async () => ({
            resources: [
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
            ],
        }));
        this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
            const { uri } = request.params;
            try {
                let data;
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
            try {
                let result;
                switch (name) {
                    case "vector_search":
                        result = await this.vectorTools.vectorSearch(args);
                        break;
                    case "generate_embedding":
                        result = await this.vectorTools.generateEmbedding(args);
                        break;
                    case "batch_embedding":
                        result = await this.vectorTools.batchEmbedding(args);
                        break;
                    case "create_hnsw_index":
                        result = await this.vectorTools.createHNSWIndex(args);
                        break;
                    case "hybrid_search":
                        result = await this.vectorTools.hybridSearch(args);
                        break;
                    case "train_ml_model":
                        result = await this.trainMLModel(args);
                        break;
                    case "predict_ml_model":
                        result = await this.mlTools.predict(args);
                        break;
                    case "cluster_data":
                        result = await this.clusterData(args);
                        break;
                    case "detect_outliers":
                        result = await this.analyticsTools.detectOutliersZScore(args);
                        break;
                    case "rag_chunk_text":
                        result = await this.ragTools.chunkText(args);
                        break;
                    case "rerank_results":
                        result = await this.rerankResults(args);
                        break;
                    case "create_ml_project":
                        result = await this.projectTools.createProject(args);
                        break;
                    case "list_ml_projects":
                        result = await this.projectTools.listProjects();
                        break;
                    case "train_kmeans_project":
                        result = await this.projectTools.trainKMeansProject(args);
                        break;
                    case "get_model_info":
                        result = await this.mlTools.getModelInfo(args.model_id);
                        break;
                    case "execute_sql":
                        const sqlResult = await this.db.query(args.query);
                        result = sqlResult.rows;
                        break;
                    default:
                        throw new Error(`Unknown tool: ${name}`);
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
    async rerankResults(args) {
        const { method = "cross_encoder", ...rest } = args;
        if (method === "cross_encoder") {
            return this.ragTools.rerankCrossEncoder(rest);
        }
        else {
            return this.ragTools.rerankLLM(rest);
        }
    }
    async connect(config) {
        try {
            this.db.connect(config);
            await this.db.testConnection();
            console.error("Neurondb MCP server connected to database");
        }
        catch (error) {
            console.error(`Warning: Failed to connect to database: ${error instanceof Error ? error.message : String(error)}`);
            console.error("Server will start but tools requiring database will fail. Set NEURONDB_CONNECTION_STRING environment variable.");
        }
    }
    async start() {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        const connectionString = process.env.NEURONDB_CONNECTION_STRING ||
            process.env.DATABASE_URL;
        if (connectionString) {
            await this.connect({ connectionString });
        }
        else {
            const config = {
                host: process.env.NEURONDB_HOST || "localhost",
                port: parseInt(process.env.NEURONDB_PORT || "5432"),
                database: process.env.NEURONDB_DATABASE || "postgres",
                user: process.env.NEURONDB_USER || "postgres",
                password: process.env.NEURONDB_PASSWORD,
            };
            if (config.host && config.database) {
                await this.connect(config);
            }
            else {
                console.error("Warning: No database connection configured. Set NEURONDB_CONNECTION_STRING or individual connection parameters.");
            }
        }
        console.error("Neurondb MCP server running on stdio");
    }
    async stop() {
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
//# sourceMappingURL=index-old.js.map