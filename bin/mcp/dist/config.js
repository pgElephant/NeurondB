import { readFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
export class ConfigManager {
    config = null;
    configPath = null;
    load(configPath) {
        if (this.config) {
            return this.config;
        }
        // Try multiple config locations
        const possiblePaths = [
            configPath,
            process.env.NEURONDB_MCP_CONFIG,
            join(process.cwd(), "mcp-config.json"),
            join(__dirname, "..", "mcp-config.json"),
            join(process.env.HOME || "", ".neurondb", "mcp-config.json"),
        ].filter(Boolean);
        for (const path of possiblePaths) {
            if (existsSync(path)) {
                try {
                    const content = readFileSync(path, "utf-8");
                    this.config = JSON.parse(content);
                    this.configPath = path;
                    console.error(`Loaded config from: ${path}`);
                    break;
                }
                catch (error) {
                    console.error(`Failed to load config from ${path}:`, error);
                }
            }
        }
        // Merge with environment variables
        this.config = this.mergeWithEnv(this.config || this.getDefaultConfig());
        return this.config;
    }
    mergeWithEnv(config) {
        // Database config from env
        if (process.env.NEURONDB_CONNECTION_STRING) {
            config.database.connectionString = process.env.NEURONDB_CONNECTION_STRING;
        }
        if (process.env.NEURONDB_HOST) {
            config.database.host = process.env.NEURONDB_HOST;
        }
        if (process.env.NEURONDB_PORT) {
            config.database.port = parseInt(process.env.NEURONDB_PORT);
        }
        if (process.env.NEURONDB_DATABASE) {
            config.database.database = process.env.NEURONDB_DATABASE;
        }
        if (process.env.NEURONDB_USER) {
            config.database.user = process.env.NEURONDB_USER;
        }
        if (process.env.NEURONDB_PASSWORD) {
            config.database.password = process.env.NEURONDB_PASSWORD;
        }
        // Logging config from env
        if (process.env.NEURONDB_LOG_LEVEL) {
            config.logging.level = process.env.NEURONDB_LOG_LEVEL;
        }
        if (process.env.NEURONDB_LOG_FORMAT) {
            config.logging.format = process.env.NEURONDB_LOG_FORMAT;
        }
        // Features from env
        if (process.env.NEURONDB_ENABLE_GPU === "true") {
            if (!config.features.ml) {
                config.features.ml = { enabled: true };
            }
            config.features.ml.gpuEnabled = true;
        }
        return config;
    }
    getDefaultConfig() {
        return {
            database: {
                host: "localhost",
                port: 5432,
                database: "postgres",
                user: "postgres",
                pool: {
                    min: 2,
                    max: 10,
                    idleTimeoutMillis: 30000,
                    connectionTimeoutMillis: 2000,
                },
            },
            server: {
                name: "neurondb-mcp-server",
                version: "1.0.0",
                timeout: 30000,
                maxRequestSize: 10 * 1024 * 1024, // 10MB
                enableMetrics: true,
                enableHealthCheck: true,
            },
            logging: {
                level: "info",
                format: "text",
                output: "stderr",
                enableRequestLogging: true,
                enableResponseLogging: false,
                enableErrorStack: false,
            },
            features: {
                vector: {
                    enabled: true,
                    defaultDistanceMetric: "l2",
                    maxVectorDimension: 16384,
                    defaultLimit: 10,
                },
                ml: {
                    enabled: true,
                    algorithms: [
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
                    maxTrainingTime: 3600000, // 1 hour
                    gpuEnabled: false,
                },
                analytics: {
                    enabled: true,
                    maxClusters: 1000,
                    maxIterations: 10000,
                },
                rag: {
                    enabled: true,
                    defaultChunkSize: 500,
                    defaultOverlap: 50,
                },
                projects: {
                    enabled: true,
                    maxProjects: 1000,
                },
            },
        };
    }
    getConfig() {
        if (!this.config) {
            return this.load();
        }
        return this.config;
    }
    getDatabaseConfig() {
        return this.getConfig().database;
    }
    getServerSettings() {
        return this.getConfig().server;
    }
    getLoggingConfig() {
        return this.getConfig().logging;
    }
    getFeaturesConfig() {
        return this.getConfig().features;
    }
    getPlugins() {
        return this.getConfig().plugins || [];
    }
    getMiddleware() {
        return this.getConfig().middleware || [];
    }
    isFeatureEnabled(feature) {
        const features = this.getFeaturesConfig();
        const featureConfig = features[feature];
        return featureConfig ? featureConfig.enabled !== false : false;
    }
}
//# sourceMappingURL=config.js.map