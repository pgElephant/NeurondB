import { Logger } from "./logger.js";
import { PluginConfig } from "./config.js";
import { Database } from "./db.js";
import { MiddlewareManager } from "./middleware.js";
export interface Plugin {
    name: string;
    version?: string;
    initialize?: (config: any, db: Database, logger: Logger, middleware: MiddlewareManager) => Promise<void>;
    tools?: Array<{
        name: string;
        description: string;
        inputSchema: any;
        handler: (params: any) => Promise<any>;
    }>;
    resources?: Array<{
        uri: string;
        name: string;
        description: string;
        mimeType: string;
        handler: () => Promise<any>;
    }>;
    middleware?: Array<{
        name: string;
        order?: number;
        handler: any;
    }>;
    shutdown?: () => Promise<void>;
}
export declare class PluginManager {
    private plugins;
    private logger;
    private db;
    private middleware;
    constructor(logger: Logger, db: Database, middleware: MiddlewareManager);
    loadPlugin(config: PluginConfig): Promise<void>;
    private loadBuiltInPlugin;
    getPlugin(name: string): Plugin | undefined;
    getAllPlugins(): Plugin[];
    getAllTools(): Array<{
        name: string;
        description: string;
        inputSchema: any;
        handler: (params: any) => Promise<any>;
    }>;
    getAllResources(): Array<{
        uri: string;
        name: string;
        description: string;
        mimeType: string;
        handler: () => Promise<any>;
    }>;
    shutdown(): Promise<void>;
}
//# sourceMappingURL=plugin.d.ts.map