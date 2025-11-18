import { Database } from "../db.js";
export declare class RAGTools {
    private db;
    constructor(db: Database);
    chunkText(params: {
        text: string;
        chunk_size?: number;
        overlap?: number;
    }): Promise<any>;
    rerankCrossEncoder(params: {
        query: string;
        documents: string[];
        model?: string;
        top_k?: number;
    }): Promise<any>;
    rerankLLM(params: {
        query: string;
        documents: string[];
        model?: string;
        top_k?: number;
    }): Promise<any>;
}
//# sourceMappingURL=rag.d.ts.map