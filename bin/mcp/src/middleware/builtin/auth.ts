/**
 * Authentication middleware
 */

import { Middleware, MCPRequest, MCPResponse } from "../types.js";
import { Logger } from "../../logging/logger.js";

export function createAuthMiddleware(apiKey?: string): Middleware {
	return {
		name: "auth",
		order: 0,
		handler: async (request: MCPRequest, next: () => Promise<MCPResponse>): Promise<MCPResponse> => {
			// If API key is configured, check for it in request metadata
			if (apiKey) {
				const requestKey = request.metadata?.apiKey || request.params?.apiKey;
				if (requestKey !== apiKey) {
					return {
						content: [
							{
								type: "text",
								text: "Unauthorized: Invalid API key",
							},
						],
						isError: true,
					};
				}
			}
			return next();
		},
	};
}





