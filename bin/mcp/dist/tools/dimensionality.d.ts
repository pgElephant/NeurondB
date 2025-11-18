import { Database } from "../db.js";
export declare class DimensionalityTools {
    private db;
    constructor(db: Database);
    /**
     * Reduce dimensionality using PCA
     */
    reducePCA(table: string, vectorColumn: string, targetDimensions: number): Promise<any[]>;
    /**
     * Apply PCA whitening to embeddings
     */
    whitenEmbeddings(table: string, vectorColumn: string): Promise<any[]>;
}
//# sourceMappingURL=dimensionality.d.ts.map