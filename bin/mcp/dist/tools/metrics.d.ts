import { Database } from "../db.js";
export declare class MetricsTools {
    private db;
    constructor(db: Database);
    /**
     * Calculate Recall@K metric
     */
    recallAtK(groundTruthTable: string, groundTruthColumn: string, predictionsTable: string, predictionsColumn: string, queryColumn: string, queryId: number, k: number): Promise<{
        recall: any;
    }>;
    /**
     * Calculate Precision@K metric
     */
    precisionAtK(groundTruthTable: string, groundTruthColumn: string, predictionsTable: string, predictionsColumn: string, queryColumn: string, queryId: number, k: number): Promise<{
        precision: any;
    }>;
    /**
     * Calculate F1@K metric
     */
    f1AtK(groundTruthTable: string, groundTruthColumn: string, predictionsTable: string, predictionsColumn: string, queryColumn: string, queryId: number, k: number): Promise<{
        f1: any;
    }>;
    /**
     * Calculate Mean Reciprocal Rank (MRR)
     */
    meanReciprocalRank(groundTruthTable: string, groundTruthColumn: string, predictionsTable: string, predictionsColumn: string, rankColumn: string, queryColumn: string): Promise<{
        mrr: any;
    }>;
    /**
     * Calculate clustering metrics (Davies-Bouldin, Silhouette)
     */
    clusteringMetrics(table: string, vectorColumn: string, clusterColumn: string, metric?: "davies_bouldin" | "silhouette"): Promise<{
        metric: "davies_bouldin" | "silhouette";
        score: any;
    }>;
}
//# sourceMappingURL=metrics.d.ts.map