package tools

import "context"

// Tool is the interface that all tools must implement
type Tool interface {
	Name() string
	Description() string
	InputSchema() map[string]interface{}
	Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error)
}

// ToolExecutor provides database query execution for tools
type ToolExecutor interface {
	ExecuteQuery(ctx context.Context, query string, params []interface{}) ([]map[string]interface{}, error)
	ExecuteQueryOne(ctx context.Context, query string, params []interface{}) (map[string]interface{}, error)
	ExecuteVectorSearch(ctx context.Context, table, vectorColumn string, queryVector []interface{}, distanceMetric string, limit int, additionalColumns []interface{}) ([]map[string]interface{}, error)
}

