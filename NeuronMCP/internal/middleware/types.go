package middleware

import "context"

// MCPRequest represents an MCP request
type MCPRequest struct {
	Method   string                 `json:"method"`
	Params   map[string]interface{} `json:"params,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// MCPResponse represents an MCP response
type MCPResponse struct {
	Content  []ContentBlock         `json:"content,omitempty"`
	IsError  bool                   `json:"isError,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// ContentBlock represents a content block in a response
type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// Handler is a function that handles a request
type Handler func(ctx context.Context) (*MCPResponse, error)

// Middleware is the interface that all middleware must implement
type Middleware interface {
	Name() string
	Order() int
	Enabled() bool
	Execute(ctx context.Context, req *MCPRequest, next Handler) (*MCPResponse, error)
}

