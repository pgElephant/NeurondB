export class AnalyticsTools {
    db;
    constructor(db) {
        this.db = db;
    }
    async clusterKMeans(params) {
        const { table, vector_column, k, max_iter = 100 } = params;
        const query = `
			SELECT cluster_kmeans($1, $2, $3, $4) AS clusters
		`;
        const result = await this.db.query(query, [
            table,
            vector_column,
            k,
            max_iter,
        ]);
        return result.rows[0];
    }
    async clusterMiniBatchKMeans(params) {
        const { table, vector_column, k, max_iter = 100 } = params;
        const batch_size = params.params?.batch_size || 100;
        const query = `
			SELECT cluster_minibatch_kmeans($1, $2, $3, $4, $5) AS clusters
		`;
        const result = await this.db.query(query, [
            table,
            vector_column,
            k,
            max_iter,
            batch_size,
        ]);
        return result.rows[0];
    }
    async clusterGMM(params) {
        const { table, vector_column, k, max_iter = 100 } = params;
        const query = `
			SELECT cluster_gmm($1, $2, $3, $4) AS probabilities
		`;
        const result = await this.db.query(query, [
            table,
            vector_column,
            k,
            max_iter,
        ]);
        return result.rows[0];
    }
    async detectOutliersZScore(params) {
        const { table, vector_column, threshold = 3.0 } = params;
        const query = `
			SELECT detect_outliers_zscore($1, $2, $3) AS outliers
		`;
        const result = await this.db.query(query, [
            table,
            vector_column,
            threshold,
        ]);
        return result.rows[0];
    }
    async computePCA(params) {
        const { table, vector_column, n_components } = params;
        const query = `
			SELECT compute_pca($1, $2, $3) AS components
		`;
        const result = await this.db.query(query, [
            table,
            vector_column,
            n_components,
        ]);
        return result.rows[0];
    }
}
//# sourceMappingURL=analytics.js.map