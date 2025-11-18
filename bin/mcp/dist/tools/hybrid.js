export class HybridTools {
    db;
    constructor(db) {
        this.db = db;
    }
    /**
     * Hybrid search fusion (combine semantic and lexical results)
     */
    async hybridSearchFusion(semanticTable, lexicalTable, idColumn, semanticScoreColumn, lexicalScoreColumn, alpha = 0.5) {
        const result = await this.db.query("SELECT * FROM neurondb.hybrid_search_fusion($1, $2, $3, $4, $5, $6)", [
            semanticTable,
            lexicalTable,
            idColumn,
            semanticScoreColumn,
            lexicalScoreColumn,
            alpha,
        ]);
        return result.rows;
    }
    /**
     * Learning to Rank (LTR) reranking
     */
    async ltrRerankPointwise(query, documents, features) {
        // This would use the neurondb.ltr_rerank_pointwise function
        // Implementation depends on actual function signature
        const result = await this.db.query("SELECT neurondb.ltr_rerank_pointwise($1, $2::text[], $3::jsonb) AS results", [query, documents, JSON.stringify(features || {})]);
        return result.rows;
    }
}
//# sourceMappingURL=hybrid.js.map