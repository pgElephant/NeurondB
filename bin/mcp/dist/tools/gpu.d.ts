import { Database } from "../db.js";
export declare class GPUTools {
    private db;
    constructor(db: Database);
    /**
     * Get GPU information and status
     */
    getGPUInfo(): Promise<any>;
    /**
     * Get GPU statistics
     */
    getGPUStats(): Promise<any>;
    /**
     * Reset GPU statistics
     */
    resetGPUStats(): Promise<{
        success: boolean;
        message: string;
    }>;
    /**
     * Enable or disable GPU
     */
    enableGPU(enabled: boolean): Promise<{
        enabled: boolean;
        message: string;
    }>;
    /**
     * Compute L2 distance on GPU
     */
    l2DistanceGPU(a: number[], b: number[]): Promise<{
        distance: any;
    }>;
    /**
     * Compute cosine distance on GPU
     */
    cosineDistanceGPU(a: number[], b: number[]): Promise<{
        distance: any;
    }>;
    /**
     * Compute inner product on GPU
     */
    innerProductGPU(a: number[], b: number[]): Promise<{
        product: any;
    }>;
    /**
     * Quantize vector to INT8 on GPU
     */
    quantizeINT8GPU(vector: number[]): Promise<{
        quantized: any;
    }>;
    /**
     * Quantize vector to FP16 on GPU
     */
    quantizeFP16GPU(vector: number[]): Promise<{
        quantized: any;
    }>;
    /**
     * Quantize vector to binary on GPU
     */
    quantizeBinaryGPU(vector: number[]): Promise<{
        quantized: any;
    }>;
    /**
     * Cluster data using GPU-accelerated K-means
     */
    clusterKMeansGPU(table: string, vectorColumn: string, k: number, maxIter?: number): Promise<any[]>;
    /**
     * HNSW search on GPU
     */
    hnswSearchGPU(table: string, vectorColumn: string, queryVector: number[], limit?: number): Promise<any[]>;
}
//# sourceMappingURL=gpu.d.ts.map