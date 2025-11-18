import { QueryResult } from "pg";
import { DatabaseConfig } from "./config.js";
export declare class Database {
    private pool;
    connect(config: DatabaseConfig): void;
    query(text: string, params?: any[]): Promise<QueryResult>;
    escapeIdentifier(identifier: string): string;
    testConnection(): Promise<void>;
    close(): Promise<void>;
}
//# sourceMappingURL=db.d.ts.map