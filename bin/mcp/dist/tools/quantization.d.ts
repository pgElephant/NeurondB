import { Database } from "../db.js";
export declare class QuantizationTools {
    private db;
    constructor(db: Database);
    /**
     * Quantize vector to INT8 (CPU)
     */
    quantizeINT8(vector: number[]): Promise<{
        quantized: any;
    }>;
    /**
     * Quantize vector to FP16 (CPU)
     */
    quantizeFP16(vector: number[]): Promise<{
        quantized: any;
    }>;
    /**
     * Quantize vector to UINT8
     */
    quantizeUINT8(vector: number[]): Promise<{
        quantized: any;
    }>;
    /**
     * Quantize vector to binary
     */
    quantizeBinary(vector: number[]): Promise<{
        quantized: any;
    }>;
    /**
     * Quantize vector to ternary
     */
    quantizeTernary(vector: number[]): Promise<{
        quantized: any;
    }>;
    /**
     * Train Product Quantization (PQ) codebook
     */
    trainPQCodebook(table: string, vectorColumn: string, numSubvectors: number, numCentroids: number, maxIter?: number): Promise<any[]>;
    /**
     * Train Optimized Product Quantization (OPQ) codebook
     */
    trainOPQCodebook(table: string, vectorColumn: string, numSubvectors: number, numCentroids: number, maxIter?: number): Promise<any[]>;
    /**
     * Encode vector using PQ codebook
     */
    encodePQ(vector: number[], numSubvectors: number, numCentroids: number, codebook: any[]): Promise<{
        codes: any;
    }>;
}
//# sourceMappingURL=quantization.d.ts.map