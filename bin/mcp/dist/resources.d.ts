import { Database } from "./db.js";
export declare class Resources {
    private db;
    constructor(db: Database);
    getSchema(): Promise<any[]>;
    getModels(): Promise<any[]>;
    getIndexes(): Promise<any[]>;
    getConfig(): Promise<any[]>;
    getWorkerStatus(): Promise<any[]>;
    getVectorStats(): Promise<any[]>;
    getIndexHealth(): Promise<any[]>;
}
//# sourceMappingURL=resources.d.ts.map