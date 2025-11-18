export class DriftTools {
    db;
    constructor(db) {
        this.db = db;
    }
    /**
     * Detect centroid drift between baseline and current data
     */
    async detectCentroidDrift(baselineTable, baselineColumn, currentTable, currentColumn, filterColumn, filterValue, threshold = 0.3) {
        let query;
        let params;
        if (filterColumn && filterValue) {
            query = `
				SELECT * FROM neurondb.detect_centroid_drift(
					$1, $2, $3, $4, $5, $6, $7
				)
			`;
            params = [
                baselineTable,
                baselineColumn,
                currentTable,
                currentColumn,
                filterColumn,
                filterValue,
                threshold,
            ];
        }
        else {
            query = `
				SELECT * FROM neurondb.detect_centroid_drift(
					$1, $2, $3, $4, NULL, NULL, $5
				)
			`;
            params = [
                baselineTable,
                baselineColumn,
                currentTable,
                currentColumn,
                threshold,
            ];
        }
        const result = await this.db.query(query, params);
        return result.rows[0];
    }
    /**
     * Detect distribution divergence (KL divergence, JS divergence)
     */
    async detectDistributionDivergence(baselineTable, baselineColumn, currentTable, currentColumn, method = "kl") {
        const result = await this.db.query("SELECT * FROM neurondb.detect_distribution_divergence($1, $2, $3, $4, $5)", [baselineTable, baselineColumn, currentTable, currentColumn, method]);
        return result.rows[0];
    }
}
//# sourceMappingURL=drift.js.map