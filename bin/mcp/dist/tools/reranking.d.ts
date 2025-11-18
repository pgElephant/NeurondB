import { Database } from "../db.js";
export declare class RerankingTools {
    private db;
    constructor(db: Database);
    /**
     * Rerank using MMR (Maximal Marginal Relevance)
     */
    mmrRerank(table: string, queryVector: number[], vectorColumn: string, lambda?: number, topK?: number): Promise<any[]>;
    /**
     * Rerank using cross-encoder
     */
    rerankCrossEncoder(query: string, documents: string[], model?: string, topK?: number): Promise<any[]>;
    /**
     * Rerank using LLM
     */
    rerankLLM(query: string, documents: string[], model?: string, topK?: number): Promise<any[]>;
}
//# sourceMappingURL=reranking.d.ts.map