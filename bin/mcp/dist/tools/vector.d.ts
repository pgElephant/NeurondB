import { Database } from "../db.js";
import { VectorSearchParams, EmbeddingParams, BatchEmbeddingParams, HNSWIndexParams, HybridSearchParams } from "../types.js";
export declare class VectorTools {
    private db;
    constructor(db: Database);
    vectorSearch(params: VectorSearchParams): Promise<any[]>;
    generateEmbedding(params: EmbeddingParams): Promise<any>;
    batchEmbedding(params: BatchEmbeddingParams): Promise<any>;
    createHNSWIndex(params: HNSWIndexParams): Promise<any>;
    hybridSearch(params: HybridSearchParams): Promise<any[]>;
}
//# sourceMappingURL=vector.d.ts.map