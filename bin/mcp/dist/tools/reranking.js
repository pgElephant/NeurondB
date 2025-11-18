export class RerankingTools {
    db;
    constructor(db) {
        this.db = db;
    }
    /**
     * Rerank using MMR (Maximal Marginal Relevance)
     */
    async mmrRerank(table, queryVector, vectorColumn, lambda = 0.5, topK = 10) {
        const vecStr = `[${queryVector.join(",")}]`;
        const result = await this.db.query("SELECT * FROM neurondb.mmr_rerank_with_scores($1, $2::vector, $3, $4, $5)", [table, vecStr, vectorColumn, lambda, topK]);
        return result.rows;
    }
    /**
     * Rerank using cross-encoder
     */
    async rerankCrossEncoder(query, documents, model, topK) {
        // Implementation depends on actual function signature
        const modelName = model || "ms-marco-MiniLM-L-6-v2";
        const k = topK || documents.length;
        const result = await this.db.query("SELECT * FROM rerank_cross_encoder($1, $2::text[], $3, $4)", [query, documents, modelName, k]);
        return result.rows;
    }
    /**
     * Rerank using LLM
     */
    async rerankLLM(query, documents, model, topK) {
        const k = topK || documents.length;
        const result = await this.db.query("SELECT * FROM rerank_llm($1, $2::text[], $3, $4)", [query, documents, model, k]);
        return result.rows;
    }
}
//# sourceMappingURL=reranking.js.map