/**
 * Middleware types and interfaces
 */

export interface MCPRequest {
	method: string;
	params?: any;
	metadata?: Record<string, any>;
}

export interface MCPResponse {
	content?: any[];
	isError?: boolean;
	metadata?: Record<string, any>;
}

export type MiddlewareFunction = (
	request: MCPRequest,
	next: () => Promise<MCPResponse>
) => Promise<MCPResponse>;

export interface Middleware {
	name: string;
	order?: number;
	enabled?: boolean;
	handler: MiddlewareFunction;
}



