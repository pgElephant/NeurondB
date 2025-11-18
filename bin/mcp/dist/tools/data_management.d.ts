import { Database } from "../db.js";
export declare class DataManagementTools {
    private db;
    constructor(db: Database);
    /**
     * Vacuum vectors (clean up unused space)
     */
    vacuumVectors(table: string, dryRun?: boolean): Promise<{
        affected: any;
        dryRun: boolean;
    }>;
    /**
     * Compress cold tier vectors (older than threshold)
     */
    compressColdTier(table: string, daysThreshold?: number): Promise<{
        compressed: any;
        threshold: number;
    }>;
    /**
     * Sync index to replica (async)
     */
    syncIndexAsync(indexName: string, replicaHost: string): Promise<{
        syncId: any;
    }>;
}
//# sourceMappingURL=data_management.d.ts.map