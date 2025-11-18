export class DataManagementTools {
    db;
    constructor(db) {
        this.db = db;
    }
    /**
     * Vacuum vectors (clean up unused space)
     */
    async vacuumVectors(table, dryRun = false) {
        const result = await this.db.query("SELECT vacuum_vectors($1, $2) AS affected", [
            table,
            dryRun,
        ]);
        return { affected: result.rows[0].affected, dryRun };
    }
    /**
     * Compress cold tier vectors (older than threshold)
     */
    async compressColdTier(table, daysThreshold = 30) {
        const result = await this.db.query("SELECT compress_cold_tier($1, $2) AS compressed", [table, daysThreshold]);
        return { compressed: result.rows[0].compressed, threshold: daysThreshold };
    }
    /**
     * Sync index to replica (async)
     */
    async syncIndexAsync(indexName, replicaHost) {
        const result = await this.db.query("SELECT sync_index_async($1, $2) AS sync_id", [indexName, replicaHost]);
        return { syncId: result.rows[0].sync_id };
    }
}
//# sourceMappingURL=data_management.js.map