export class GPUTools {
    db;
    constructor(db) {
        this.db = db;
    }
    /**
     * Get GPU information and status
     */
    async getGPUInfo() {
        const result = await this.db.query("SELECT * FROM neurondb_gpu_info()");
        return result.rows[0];
    }
    /**
     * Get GPU statistics
     */
    async getGPUStats() {
        const result = await this.db.query("SELECT * FROM neurondb_gpu_stats()");
        return result.rows[0];
    }
    /**
     * Reset GPU statistics
     */
    async resetGPUStats() {
        await this.db.query("SELECT neurondb_gpu_stats_reset()");
        return { success: true, message: "GPU statistics reset" };
    }
    /**
     * Enable or disable GPU
     */
    async enableGPU(enabled) {
        await this.db.query("SELECT neurondb_gpu_enable($1)", [enabled]);
        return { enabled, message: `GPU ${enabled ? "enabled" : "disabled"}` };
    }
    /**
     * Compute L2 distance on GPU
     */
    async l2DistanceGPU(a, b) {
        const aStr = `[${a.join(",")}]`;
        const bStr = `[${b.join(",")}]`;
        const result = await this.db.query("SELECT vector_l2_distance_gpu($1::vector, $2::vector) AS distance", [aStr, bStr]);
        return { distance: result.rows[0].distance };
    }
    /**
     * Compute cosine distance on GPU
     */
    async cosineDistanceGPU(a, b) {
        const aStr = `[${a.join(",")}]`;
        const bStr = `[${b.join(",")}]`;
        const result = await this.db.query("SELECT vector_cosine_distance_gpu($1::vector, $2::vector) AS distance", [aStr, bStr]);
        return { distance: result.rows[0].distance };
    }
    /**
     * Compute inner product on GPU
     */
    async innerProductGPU(a, b) {
        const aStr = `[${a.join(",")}]`;
        const bStr = `[${b.join(",")}]`;
        const result = await this.db.query("SELECT vector_inner_product_gpu($1::vector, $2::vector) AS product", [aStr, bStr]);
        return { product: result.rows[0].product };
    }
    /**
     * Quantize vector to INT8 on GPU
     */
    async quantizeINT8GPU(vector) {
        const vecStr = `[${vector.join(",")}]`;
        const result = await this.db.query("SELECT vector_to_int8_gpu($1::vector) AS quantized", [vecStr]);
        return { quantized: result.rows[0].quantized };
    }
    /**
     * Quantize vector to FP16 on GPU
     */
    async quantizeFP16GPU(vector) {
        const vecStr = `[${vector.join(",")}]`;
        const result = await this.db.query("SELECT vector_to_fp16_gpu($1::vector) AS quantized", [vecStr]);
        return { quantized: result.rows[0].quantized };
    }
    /**
     * Quantize vector to binary on GPU
     */
    async quantizeBinaryGPU(vector) {
        const vecStr = `[${vector.join(",")}]`;
        const result = await this.db.query("SELECT vector_to_binary_gpu($1::vector) AS quantized", [vecStr]);
        return { quantized: result.rows[0].quantized };
    }
    /**
     * Cluster data using GPU-accelerated K-means
     */
    async clusterKMeansGPU(table, vectorColumn, k, maxIter = 100) {
        const result = await this.db.query("SELECT * FROM cluster_kmeans_gpu($1, $2, $3, $4)", [table, vectorColumn, k, maxIter]);
        return result.rows;
    }
    /**
     * HNSW search on GPU
     */
    async hnswSearchGPU(table, vectorColumn, queryVector, limit = 10) {
        const vecStr = `[${queryVector.join(",")}]`;
        const result = await this.db.query("SELECT * FROM neurondb_hnsw_search_gpu($1, $2, $3::vector, $4)", [table, vectorColumn, vecStr, limit]);
        return result.rows;
    }
}
//# sourceMappingURL=gpu.js.map