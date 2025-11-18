import { Database } from "../db.js";
import { ClusteringParams } from "../types.js";
export declare class AnalyticsTools {
    private db;
    constructor(db: Database);
    clusterKMeans(params: ClusteringParams): Promise<any>;
    clusterMiniBatchKMeans(params: ClusteringParams): Promise<any>;
    clusterGMM(params: ClusteringParams): Promise<any>;
    detectOutliersZScore(params: {
        table: string;
        vector_column: string;
        threshold?: number;
    }): Promise<any>;
    computePCA(params: {
        table: string;
        vector_column: string;
        n_components: number;
    }): Promise<any>;
}
//# sourceMappingURL=analytics.d.ts.map