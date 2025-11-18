export class MiddlewareManager {
    middlewares = [];
    logger;
    constructor(logger) {
        this.logger = logger;
    }
    register(middleware) {
        this.middlewares.push(middleware);
        this.middlewares.sort((a, b) => (a.order || 0) - (b.order || 0));
        this.logger.debug(`Registered middleware: ${middleware.name}`);
    }
    async execute(request, handler) {
        let index = 0;
        const next = async () => {
            if (index >= this.middlewares.length) {
                return handler();
            }
            const middleware = this.middlewares[index++];
            return middleware.handler(request, next);
        };
        return next();
    }
    clear() {
        this.middlewares = [];
    }
}
// Built-in middleware
export const createLoggingMiddleware = (logger) => ({
    name: "logging",
    order: 1,
    handler: async (request, next) => {
        const start = Date.now();
        logger.request(request.method, "mcp", request.params);
        try {
            const response = await next();
            const duration = Date.now() - start;
            logger.response(request.method, "mcp", duration, !response.isError);
            return response;
        }
        catch (error) {
            const duration = Date.now() - start;
            logger.error(`Request failed: ${request.method}`, error, { duration });
            throw error;
        }
    },
});
export const createErrorHandlingMiddleware = (logger) => ({
    name: "error-handling",
    order: 100,
    handler: async (request, next) => {
        try {
            return await next();
        }
        catch (error) {
            logger.error(`Unhandled error in ${request.method}`, error);
            return {
                content: [{
                        type: "text",
                        text: `Error: ${error instanceof Error ? error.message : String(error)}`,
                    }],
                isError: true,
            };
        }
    },
});
export const createValidationMiddleware = () => ({
    name: "validation",
    order: 2,
    handler: async (request, next) => {
        // Basic validation
        if (!request.method) {
            throw new Error("Missing method in request");
        }
        return next();
    },
});
export const createTimeoutMiddleware = (timeout, logger) => ({
    name: "timeout",
    order: 3,
    handler: async (request, next) => {
        return Promise.race([
            next(),
            new Promise((_, reject) => {
                setTimeout(() => {
                    reject(new Error(`Request timeout after ${timeout}ms`));
                }, timeout);
            }),
        ]);
    },
});
//# sourceMappingURL=middleware.js.map