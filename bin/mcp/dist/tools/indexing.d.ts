import { Database } from "../db.js";
export declare class IndexingTools {
    private db;
    constructor(db: Database);
    /**
     * Create HNSW index
     */
    createHNSWIndex(table: string, vectorColumn: string, indexName: string, m?: number, efConstruction?: number): Promise<{
        success: any;
        indexName: string;
    }>;
    /**
     * Create IVF index
     */
    createIVFIndex(table: string, vectorColumn: string, indexName: string, numLists?: number): Promise<{
        success: any;
        indexName: string;
    }>;
    /**
     * Rebalance index
     */
    rebalanceIndex(indexName: string, threshold?: number): Promise<{
        rebalanced: any;
    }>;
    /**
     * Get index statistics
     */
    getIndexStats(indexName: string): Promise<any>;
    /**
     * Drop index
     */
    dropIndex(indexName: string): Promise<{
        success: boolean;
        message: string;
    }>;
}
//# sourceMappingURL=indexing.d.ts.map