/**
 * Example Plugin for Neurondb MCP Server
 * 
 * This demonstrates how to create a custom plugin that extends
 * the MCP server with additional tools, resources, and middleware.
 */

import { Plugin } from "../src/plugin.js";
import { Database } from "../src/db.js";
import { Logger } from "../src/logger.js";
import { MiddlewareManager, Middleware } from "../src/middleware.js";

export default {
	name: "example-plugin",
	version: "1.0.0",

	/**
	 * Initialize the plugin
	 */
	initialize: async (
		config: Record<string, any>,
		db: Database,
		logger: Logger,
		middleware: MiddlewareManager
	) => {
		logger.info("Example plugin initialized", { config });

		// Plugin can perform initialization tasks here
		// e.g., create tables, set up resources, etc.
	},

	/**
	 * Custom tools provided by this plugin
	 */
	tools: [
		{
			name: "example_custom_tool",
			description: "An example custom tool that demonstrates plugin functionality",
			inputSchema: {
				type: "object",
				properties: {
					message: {
						type: "string",
						description: "A message to process",
					},
					repeat: {
						type: "number",
						description: "Number of times to repeat",
						default: 1,
					},
				},
				required: ["message"],
			},
			handler: async (params: { message: string; repeat?: number }) => {
				const repeat = params.repeat || 1;
				return {
					result: params.message.repeat(repeat),
					processed: true,
					timestamp: new Date().toISOString(),
				};
			},
		},
		{
			name: "example_db_query",
			description: "Example tool that queries the database",
			inputSchema: {
				type: "object",
				properties: {
					query: {
						type: "string",
						description: "SQL query to execute",
					},
				},
				required: ["query"],
			},
			handler: async (params: { query: string }, db: Database) => {
				// Note: In real implementation, db would be passed via closure
				// This is just an example structure
				return {
					note: "This is an example - db access would be via closure",
					query: params.query,
				};
			},
		},
	],

	/**
	 * Custom resources provided by this plugin
	 */
	resources: [
		{
			uri: "neurondb://example/status",
			name: "Example Status",
			description: "Example resource showing plugin status",
			mimeType: "application/json",
			handler: async () => {
				return {
					status: "active",
					plugin: "example-plugin",
					version: "1.0.0",
					timestamp: new Date().toISOString(),
				};
			},
		},
	],

	/**
	 * Custom middleware provided by this plugin
	 */
	middleware: [
		{
			name: "example-middleware",
			order: 5, // Executes after validation but before logging
			handler: async (request, next) => {
				// Example: Add custom header or modify request
				const response = await next();
				// Example: Modify response
				return response;
			},
		} as Middleware,
	],

	/**
	 * Cleanup when plugin is shut down
	 */
	shutdown: async () => {
		// Cleanup resources, close connections, etc.
		console.error("Example plugin shutting down");
	},
} as Plugin;

