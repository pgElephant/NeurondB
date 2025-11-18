import { Database } from "../db.js";
export declare class WorkerTools {
    private db;
    constructor(db: Database);
    /**
     * Run queue worker once
     */
    runQueueWorker(): Promise<{
        processed: any;
    }>;
    /**
     * Sample tuner worker
     */
    sampleTuner(): Promise<{
        sampled: any;
    }>;
    /**
     * Get worker status
     */
    getWorkerStatus(): Promise<any[]>;
}
//# sourceMappingURL=workers.d.ts.map