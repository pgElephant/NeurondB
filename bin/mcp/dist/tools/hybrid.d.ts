import { Database } from "../db.js";
export declare class HybridTools {
    private db;
    constructor(db: Database);
    /**
     * Hybrid search fusion (combine semantic and lexical results)
     */
    hybridSearchFusion(semanticTable: string, lexicalTable: string, idColumn: string, semanticScoreColumn: string, lexicalScoreColumn: string, alpha?: number): Promise<any[]>;
    /**
     * Learning to Rank (LTR) reranking
     */
    ltrRerankPointwise(query: string, documents: string[], features?: Record<string, any>): Promise<any[]>;
}
//# sourceMappingURL=hybrid.d.ts.map