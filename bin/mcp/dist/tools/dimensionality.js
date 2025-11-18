export class DimensionalityTools {
    db;
    constructor(db) {
        this.db = db;
    }
    /**
     * Reduce dimensionality using PCA
     */
    async reducePCA(table, vectorColumn, targetDimensions) {
        const result = await this.db.query("SELECT * FROM neurondb.reduce_pca($1, $2, $3)", [table, vectorColumn, targetDimensions]);
        return result.rows;
    }
    /**
     * Apply PCA whitening to embeddings
     */
    async whitenEmbeddings(table, vectorColumn) {
        const result = await this.db.query("SELECT * FROM neurondb.whiten_embeddings($1, $2)", [table, vectorColumn]);
        return result.rows;
    }
}
//# sourceMappingURL=dimensionality.js.map