import { Logger } from "./logger.js";
export interface MCPRequest {
    method: string;
    params?: any;
}
export interface MCPResponse {
    content?: any[];
    isError?: boolean;
}
export type MiddlewareFunction = (request: MCPRequest, next: () => Promise<MCPResponse>) => Promise<MCPResponse>;
export interface Middleware {
    name: string;
    order?: number;
    handler: MiddlewareFunction;
}
export declare class MiddlewareManager {
    private middlewares;
    private logger;
    constructor(logger: Logger);
    register(middleware: Middleware): void;
    execute(request: MCPRequest, handler: () => Promise<MCPResponse>): Promise<MCPResponse>;
    clear(): void;
}
export declare const createLoggingMiddleware: (logger: Logger) => Middleware;
export declare const createErrorHandlingMiddleware: (logger: Logger) => Middleware;
export declare const createValidationMiddleware: () => Middleware;
export declare const createTimeoutMiddleware: (timeout: number, logger: Logger) => Middleware;
//# sourceMappingURL=middleware.d.ts.map