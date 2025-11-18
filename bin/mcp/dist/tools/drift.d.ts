import { Database } from "../db.js";
export declare class DriftTools {
    private db;
    constructor(db: Database);
    /**
     * Detect centroid drift between baseline and current data
     */
    detectCentroidDrift(baselineTable: string, baselineColumn: string, currentTable: string, currentColumn: string, filterColumn?: string, filterValue?: string, threshold?: number): Promise<any>;
    /**
     * Detect distribution divergence (KL divergence, JS divergence)
     */
    detectDistributionDivergence(baselineTable: string, baselineColumn: string, currentTable: string, currentColumn: string, method?: "kl" | "js"): Promise<any>;
}
//# sourceMappingURL=drift.d.ts.map