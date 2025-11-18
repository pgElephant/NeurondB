export class IndexingTools {
    db;
    constructor(db) {
        this.db = db;
    }
    /**
     * Create HNSW index
     */
    async createHNSWIndex(table, vectorColumn, indexName, m = 16, efConstruction = 200) {
        const result = await this.db.query("SELECT hnsw_create_index($1, $2, $3, $4, $5) AS success", [table, vectorColumn, indexName, m, efConstruction]);
        return { success: result.rows[0].success, indexName };
    }
    /**
     * Create IVF index
     */
    async createIVFIndex(table, vectorColumn, indexName, numLists = 100) {
        const result = await this.db.query("SELECT ivf_create_index($1, $2, $3, $4) AS success", [table, vectorColumn, indexName, numLists]);
        return { success: result.rows[0].success, indexName };
    }
    /**
     * Rebalance index
     */
    async rebalanceIndex(indexName, threshold = 0.8) {
        const result = await this.db.query("SELECT rebalance_index($1, $2) AS rebalanced", [indexName, threshold]);
        return { rebalanced: result.rows[0].rebalanced };
    }
    /**
     * Get index statistics
     */
    async getIndexStats(indexName) {
        const result = await this.db.query(`
			SELECT 
				indexname,
				indexdef,
				pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
			FROM pg_indexes
			WHERE indexname = $1
		`, [indexName]);
        return result.rows[0];
    }
    /**
     * Drop index
     */
    async dropIndex(indexName) {
        const escaped = this.db.escapeIdentifier(indexName);
        await this.db.query(`DROP INDEX IF EXISTS ${escaped}`);
        return { success: true, message: `Index ${indexName} dropped` };
    }
}
//# sourceMappingURL=indexing.js.map