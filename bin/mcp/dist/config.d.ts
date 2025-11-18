export interface ServerConfig {
    database: DatabaseConfig;
    server: ServerSettings;
    logging: LoggingConfig;
    features: FeaturesConfig;
    plugins?: PluginConfig[];
    middleware?: MiddlewareConfig[];
}
export interface DatabaseConfig {
    connectionString?: string;
    host?: string;
    port?: number;
    database?: string;
    user?: string;
    password?: string;
    pool?: {
        min?: number;
        max?: number;
        idleTimeoutMillis?: number;
        connectionTimeoutMillis?: number;
    };
    ssl?: boolean | {
        rejectUnauthorized?: boolean;
        ca?: string;
        cert?: string;
        key?: string;
    };
}
export interface ServerSettings {
    name?: string;
    version?: string;
    timeout?: number;
    maxRequestSize?: number;
    enableMetrics?: boolean;
    enableHealthCheck?: boolean;
}
export interface LoggingConfig {
    level: "debug" | "info" | "warn" | "error";
    format: "json" | "text";
    output?: "stdout" | "stderr" | string;
    enableRequestLogging?: boolean;
    enableResponseLogging?: boolean;
    enableErrorStack?: boolean;
}
export interface FeaturesConfig {
    vector?: {
        enabled: boolean;
        defaultDistanceMetric?: "l2" | "cosine" | "inner_product";
        maxVectorDimension?: number;
        defaultLimit?: number;
    };
    ml?: {
        enabled: boolean;
        algorithms?: string[];
        maxTrainingTime?: number;
        gpuEnabled?: boolean;
    };
    analytics?: {
        enabled: boolean;
        maxClusters?: number;
        maxIterations?: number;
    };
    rag?: {
        enabled: boolean;
        defaultChunkSize?: number;
        defaultOverlap?: number;
    };
    projects?: {
        enabled: boolean;
        maxProjects?: number;
    };
}
export interface PluginConfig {
    name: string;
    enabled: boolean;
    path?: string;
    config?: Record<string, any>;
}
export interface MiddlewareConfig {
    name: string;
    enabled: boolean;
    order?: number;
    config?: Record<string, any>;
}
export declare class ConfigManager {
    private config;
    private configPath;
    load(configPath?: string): ServerConfig;
    private mergeWithEnv;
    private getDefaultConfig;
    getConfig(): ServerConfig;
    getDatabaseConfig(): DatabaseConfig;
    getServerSettings(): ServerSettings;
    getLoggingConfig(): LoggingConfig;
    getFeaturesConfig(): FeaturesConfig;
    getPlugins(): PluginConfig[];
    getMiddleware(): MiddlewareConfig[];
    isFeatureEnabled(feature: keyof FeaturesConfig): boolean;
}
//# sourceMappingURL=config.d.ts.map