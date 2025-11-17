# Plugin Development Guide

## Overview

The Neurondb MCP Server supports a modular plugin architecture that allows you to extend functionality with custom tools, resources, and middleware.

## Plugin Structure

A plugin is a TypeScript/JavaScript module that exports a `Plugin` object:

```typescript
import { Plugin } from "../src/plugin.js";

export default {
  name: "my-plugin",
  version: "1.0.0",
  initialize: async (config, db, logger, middleware) => {
    // Initialization logic
  },
  tools: [ /* tool definitions */ ],
  resources: [ /* resource definitions */ ],
  middleware: [ /* middleware definitions */ ],
  shutdown: async () => {
    // Cleanup logic
  }
} as Plugin;
```

## Plugin Interface

```typescript
interface Plugin {
  name: string;
  version?: string;
  initialize?: (
    config: Record<string, any>,
    db: Database,
    logger: Logger,
    middleware: MiddlewareManager
  ) => Promise<void>;
  tools?: Array<{
    name: string;
    description: string;
    inputSchema: any; // JSON Schema
    handler: (params: any) => Promise<any>;
  }>;
  resources?: Array<{
    uri: string;
    name: string;
    description: string;
    mimeType: string;
    handler: () => Promise<any>;
  }>;
  middleware?: Array<{
    name: string;
    order?: number;
    handler: MiddlewareFunction;
  }>;
  shutdown?: () => Promise<void>;
}
```

## Creating a Plugin

### 1. Create Plugin File

Create a new file in `plugins/` directory:

```typescript
// plugins/my-plugin.ts
import { Plugin } from "../src/plugin.js";

export default {
  name: "my-plugin",
  version: "1.0.0",
  // ... plugin implementation
} as Plugin;
```

### 2. Register in Configuration

Add to `mcp-config.json`:

```json
{
  "plugins": [
    {
      "name": "my-plugin",
      "enabled": true,
      "path": "./plugins/my-plugin.js",
      "config": {
        "customSetting": "value"
      }
    }
  ]
}
```

### 3. Build Plugin

If using TypeScript, compile to JavaScript:

```bash
tsc plugins/my-plugin.ts --outDir plugins --module es2022 --target es2022
```

## Examples

### Simple Tool Plugin

```typescript
export default {
  name: "calculator",
  tools: [
    {
      name: "add",
      description: "Add two numbers",
      inputSchema: {
        type: "object",
        properties: {
          a: { type: "number" },
          b: { type: "number" }
        },
        required: ["a", "b"]
      },
      handler: async (params) => {
        return { result: params.a + params.b };
      }
    }
  ]
} as Plugin;
```

### Database Plugin

```typescript
export default {
  name: "custom-queries",
  initialize: async (config, db, logger) => {
    // Create custom tables, indexes, etc.
    await db.query(`
      CREATE TABLE IF NOT EXISTS custom_data (
        id SERIAL PRIMARY KEY,
        data JSONB
      )
    `);
  },
  tools: [
    {
      name: "store_custom_data",
      description: "Store custom data",
      inputSchema: {
        type: "object",
        properties: {
          data: { type: "object" }
        },
        required: ["data"]
      },
      handler: async (params) => {
        // Access db via closure or dependency injection
        const result = await db.query(
          "INSERT INTO custom_data (data) VALUES ($1) RETURNING id",
          [JSON.stringify(params.data)]
        );
        return { id: result.rows[0].id };
      }
    }
  ]
} as Plugin;
```

### Middleware Plugin

```typescript
export default {
  name: "auth-middleware",
  middleware: [
    {
      name: "authentication",
      order: 0, // Execute first
      handler: async (request, next) => {
        // Check authentication
        const token = request.params?._authToken;
        if (!token || !isValidToken(token)) {
          return {
            content: [{
              type: "text",
              text: "Error: Authentication required"
            }],
            isError: true
          };
        }
        return next();
      }
    }
  ]
} as Plugin;
```

## Best Practices

1. **Error Handling**: Always handle errors gracefully
2. **Logging**: Use the provided logger for all logging
3. **Type Safety**: Use TypeScript for better type safety
4. **Documentation**: Document all tools and resources
5. **Testing**: Test plugins independently
6. **Configuration**: Make plugins configurable via config object
7. **Cleanup**: Implement shutdown hook for resource cleanup

## Plugin Lifecycle

1. **Load**: Plugin module is loaded
2. **Initialize**: `initialize()` is called with config
3. **Register**: Tools, resources, and middleware are registered
4. **Active**: Plugin is active and handling requests
5. **Shutdown**: `shutdown()` is called when server stops

## Accessing Built-in Services

Plugins receive access to:
- **Database**: `db` - Execute queries
- **Logger**: `logger` - Structured logging
- **Middleware**: `middleware` - Register additional middleware

## Limitations

- Plugins run in the same process as the server
- No sandboxing (plugins have full access)
- Must be compatible with Node.js ES modules
- Should not block the event loop

## Distribution

Plugins can be:
- Local files (relative paths)
- NPM packages (if published)
- Git repositories (with proper module resolution)

