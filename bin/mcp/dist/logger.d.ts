import { LoggingConfig } from "./config.js";
export declare enum LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
}
export interface LogEntry {
    timestamp: string;
    level: string;
    message: string;
    context?: Record<string, any>;
    error?: {
        message: string;
        stack?: string;
    };
}
export declare class Logger {
    private config;
    private level;
    private output;
    constructor(config: LoggingConfig);
    private parseLevel;
    private shouldLog;
    private formatMessage;
    private log;
    debug(message: string, context?: Record<string, any>): void;
    info(message: string, context?: Record<string, any>): void;
    warn(message: string, context?: Record<string, any>): void;
    error(message: string, error?: Error, context?: Record<string, any>): void;
    request(method: string, path: string, params?: any): void;
    response(method: string, path: string, duration: number, success: boolean): void;
    private sanitizeParams;
}
//# sourceMappingURL=logger.d.ts.map