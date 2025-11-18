import { Pool } from "pg";
export class Database {
    pool = null;
    connect(config) {
        const poolConfig = {};
        if (config.connectionString) {
            poolConfig.connectionString = config.connectionString;
        }
        else {
            poolConfig.host = config.host || "localhost";
            poolConfig.port = config.port || 5432;
            poolConfig.database = config.database || "postgres";
            poolConfig.user = config.user || "postgres";
            poolConfig.password = config.password;
        }
        // Apply pool settings
        if (config.pool) {
            poolConfig.min = config.pool.min;
            poolConfig.max = config.pool.max;
            poolConfig.idleTimeoutMillis = config.pool.idleTimeoutMillis;
            poolConfig.connectionTimeoutMillis = config.pool.connectionTimeoutMillis;
        }
        // Apply SSL settings
        if (config.ssl) {
            poolConfig.ssl = config.ssl;
        }
        this.pool = new Pool(poolConfig);
        // Handle pool errors
        this.pool.on("error", (err) => {
            console.error("Unexpected database pool error:", err);
        });
    }
    async query(text, params) {
        if (!this.pool) {
            throw new Error("Database not connected");
        }
        return this.pool.query(text, params);
    }
    escapeIdentifier(identifier) {
        return `"${identifier.replace(/"/g, '""')}"`;
    }
    async testConnection() {
        await this.query("SELECT 1");
    }
    async close() {
        if (this.pool) {
            await this.pool.end();
            this.pool = null;
        }
    }
}
//# sourceMappingURL=db.js.map