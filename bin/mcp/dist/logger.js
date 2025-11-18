export var LogLevel;
(function (LogLevel) {
    LogLevel[LogLevel["DEBUG"] = 0] = "DEBUG";
    LogLevel[LogLevel["INFO"] = 1] = "INFO";
    LogLevel[LogLevel["WARN"] = 2] = "WARN";
    LogLevel[LogLevel["ERROR"] = 3] = "ERROR";
})(LogLevel || (LogLevel = {}));
export class Logger {
    config;
    level;
    output;
    constructor(config) {
        this.config = config;
        this.level = this.parseLevel(config.level);
        if (config.output === "stdout") {
            this.output = process.stdout;
        }
        else if (config.output === "stderr") {
            this.output = process.stderr;
        }
        else {
            // For file output, would need fs.createWriteStream
            this.output = process.stderr;
        }
    }
    parseLevel(level) {
        switch (level.toLowerCase()) {
            case "debug":
                return LogLevel.DEBUG;
            case "info":
                return LogLevel.INFO;
            case "warn":
                return LogLevel.WARN;
            case "error":
                return LogLevel.ERROR;
            default:
                return LogLevel.INFO;
        }
    }
    shouldLog(level) {
        return level >= this.level;
    }
    formatMessage(entry) {
        if (this.config.format === "json") {
            return JSON.stringify(entry);
        }
        // Text format
        const parts = [
            entry.timestamp,
            `[${entry.level}]`,
            entry.message,
        ];
        if (entry.context && Object.keys(entry.context).length > 0) {
            parts.push(JSON.stringify(entry.context));
        }
        if (entry.error) {
            parts.push(`Error: ${entry.error.message}`);
            if (this.config.enableErrorStack && entry.error.stack) {
                parts.push(`\n${entry.error.stack}`);
            }
        }
        return parts.join(" ");
    }
    log(level, levelName, message, context, error) {
        if (!this.shouldLog(level)) {
            return;
        }
        const entry = {
            timestamp: new Date().toISOString(),
            level: levelName,
            message,
            context,
            error: error ? {
                message: error.message,
                stack: this.config.enableErrorStack ? error.stack : undefined,
            } : undefined,
        };
        const formatted = this.formatMessage(entry);
        this.output.write(formatted + "\n");
    }
    debug(message, context) {
        this.log(LogLevel.DEBUG, "DEBUG", message, context);
    }
    info(message, context) {
        this.log(LogLevel.INFO, "INFO", message, context);
    }
    warn(message, context) {
        this.log(LogLevel.WARN, "WARN", message, context);
    }
    error(message, error, context) {
        this.log(LogLevel.ERROR, "ERROR", message, context, error);
    }
    request(method, path, params) {
        if (this.config.enableRequestLogging) {
            this.debug("Request", {
                method,
                path,
                params: this.sanitizeParams(params),
            });
        }
    }
    response(method, path, duration, success) {
        if (this.config.enableResponseLogging) {
            this.debug("Response", {
                method,
                path,
                duration: `${duration}ms`,
                success,
            });
        }
    }
    sanitizeParams(params) {
        if (!params)
            return params;
        const sanitized = { ...params };
        // Remove sensitive fields
        if (sanitized.password)
            sanitized.password = "***";
        if (sanitized.connectionString)
            sanitized.connectionString = "***";
        return sanitized;
    }
}
//# sourceMappingURL=logger.js.map