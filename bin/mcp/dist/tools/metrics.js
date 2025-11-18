export class MetricsTools {
    db;
    constructor(db) {
        this.db = db;
    }
    /**
     * Calculate Recall@K metric
     */
    async recallAtK(groundTruthTable, groundTruthColumn, predictionsTable, predictionsColumn, queryColumn, queryId, k) {
        const result = await this.db.query("SELECT neurondb.recall_at_k($1, $2, $3, $4, $5, $6, $7) AS recall", [
            groundTruthTable,
            groundTruthColumn,
            predictionsTable,
            predictionsColumn,
            queryColumn,
            queryId,
            k,
        ]);
        return { recall: result.rows[0].recall };
    }
    /**
     * Calculate Precision@K metric
     */
    async precisionAtK(groundTruthTable, groundTruthColumn, predictionsTable, predictionsColumn, queryColumn, queryId, k) {
        const result = await this.db.query("SELECT neurondb.precision_at_k($1, $2, $3, $4, $5, $6, $7) AS precision", [
            groundTruthTable,
            groundTruthColumn,
            predictionsTable,
            predictionsColumn,
            queryColumn,
            queryId,
            k,
        ]);
        return { precision: result.rows[0].precision };
    }
    /**
     * Calculate F1@K metric
     */
    async f1AtK(groundTruthTable, groundTruthColumn, predictionsTable, predictionsColumn, queryColumn, queryId, k) {
        const result = await this.db.query("SELECT neurondb.f1_at_k($1, $2, $3, $4, $5, $6, $7) AS f1", [
            groundTruthTable,
            groundTruthColumn,
            predictionsTable,
            predictionsColumn,
            queryColumn,
            queryId,
            k,
        ]);
        return { f1: result.rows[0].f1 };
    }
    /**
     * Calculate Mean Reciprocal Rank (MRR)
     */
    async meanReciprocalRank(groundTruthTable, groundTruthColumn, predictionsTable, predictionsColumn, rankColumn, queryColumn) {
        const result = await this.db.query("SELECT neurondb.mean_reciprocal_rank($1, $2, $3, $4, $5, $6) AS mrr", [
            groundTruthTable,
            groundTruthColumn,
            predictionsTable,
            predictionsColumn,
            rankColumn,
            queryColumn,
        ]);
        return { mrr: result.rows[0].mrr };
    }
    /**
     * Calculate clustering metrics (Davies-Bouldin, Silhouette)
     */
    async clusteringMetrics(table, vectorColumn, clusterColumn, metric = "davies_bouldin") {
        let query;
        if (metric === "davies_bouldin") {
            query = `
				SELECT neurondb.davies_bouldin_index($1, $2, $3) AS score
			`;
        }
        else {
            query = `
				SELECT neurondb.silhouette_score($1, $2, $3) AS score
			`;
        }
        const result = await this.db.query(query, [
            table,
            vectorColumn,
            clusterColumn,
        ]);
        return { metric, score: result.rows[0].score };
    }
}
//# sourceMappingURL=metrics.js.map