export class PluginManager {
    plugins = new Map();
    logger;
    db;
    middleware;
    constructor(logger, db, middleware) {
        this.logger = logger;
        this.db = db;
        this.middleware = middleware;
    }
    async loadPlugin(config) {
        if (!config.enabled) {
            this.logger.debug(`Plugin ${config.name} is disabled`);
            return;
        }
        try {
            let plugin = null;
            if (config.path) {
                // Load from file
                const module = await import(config.path);
                plugin = module.default || module;
            }
            else {
                // Try to load from built-in plugins
                plugin = await this.loadBuiltInPlugin(config.name);
            }
            if (!plugin) {
                throw new Error(`Plugin ${config.name} not found`);
            }
            // Initialize plugin
            if (plugin.initialize) {
                await plugin.initialize(config.config || {}, this.db, this.logger, this.middleware);
            }
            // Register middleware
            if (plugin.middleware) {
                for (const mw of plugin.middleware) {
                    this.middleware.register(mw);
                }
            }
            this.plugins.set(config.name, plugin);
            this.logger.info(`Loaded plugin: ${config.name}`, { version: plugin.version });
        }
        catch (error) {
            this.logger.error(`Failed to load plugin ${config.name}`, error);
            throw error;
        }
    }
    async loadBuiltInPlugin(name) {
        // Built-in plugins would be loaded here
        // For now, return null to indicate no built-in plugin
        return null;
    }
    getPlugin(name) {
        return this.plugins.get(name);
    }
    getAllPlugins() {
        return Array.from(this.plugins.values());
    }
    getAllTools() {
        const tools = [];
        for (const plugin of this.plugins.values()) {
            if (plugin.tools) {
                tools.push(...plugin.tools);
            }
        }
        return tools;
    }
    getAllResources() {
        const resources = [];
        for (const plugin of this.plugins.values()) {
            if (plugin.resources) {
                resources.push(...plugin.resources);
            }
        }
        return resources;
    }
    async shutdown() {
        for (const plugin of this.plugins.values()) {
            if (plugin.shutdown) {
                try {
                    await plugin.shutdown();
                }
                catch (error) {
                    this.logger.error(`Error shutting down plugin ${plugin.name}`, error);
                }
            }
        }
        this.plugins.clear();
    }
}
//# sourceMappingURL=plugin.js.map