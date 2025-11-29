package tools

import (
	"fmt"
	"reflect"
)

// ToolResult represents the result of tool execution
type ToolResult struct {
	Success  bool                   `json:"success"`
	Data     interface{}            `json:"data,omitempty"`
	Error    *ToolError            `json:"error,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// ToolError represents a tool execution error
type ToolError struct {
	Message string      `json:"message"`
	Code    string      `json:"code,omitempty"`
	Details interface{} `json:"details,omitempty"`
}

// BaseTool provides common functionality for tools
type BaseTool struct {
	name        string
	description string
	inputSchema map[string]interface{}
}

// NewBaseTool creates a new base tool
func NewBaseTool(name, description string, inputSchema map[string]interface{}) *BaseTool {
	return &BaseTool{
		name:        name,
		description: description,
		inputSchema: inputSchema,
	}
}

// Name returns the tool name
func (b *BaseTool) Name() string {
	return b.name
}

// Description returns the tool description
func (b *BaseTool) Description() string {
	return b.description
}

// InputSchema returns the input schema
func (b *BaseTool) InputSchema() map[string]interface{} {
	return b.inputSchema
}

// ValidateParams validates parameters against the schema
func (b *BaseTool) ValidateParams(params map[string]interface{}, schema map[string]interface{}) (bool, []string) {
	var errors []string

	// Check required fields
	if required, ok := schema["required"].([]interface{}); ok {
		for _, req := range required {
			if reqStr, ok := req.(string); ok {
				if _, exists := params[reqStr]; !exists {
					errors = append(errors, fmt.Sprintf("Missing required parameter: %s", reqStr))
				}
			}
		}
	}

	// Validate parameter types
	if properties, ok := schema["properties"].(map[string]interface{}); ok {
		for key, value := range params {
			if propSchema, exists := properties[key]; exists {
				if propMap, ok := propSchema.(map[string]interface{}); ok {
					if typeError := validateType(value, propMap); typeError != "" {
						errors = append(errors, fmt.Sprintf("Invalid type for %s: %s", key, typeError))
					}
				}
			}
		}
	}

	return len(errors) == 0, errors
}

func validateType(value interface{}, schema map[string]interface{}) string {
	schemaType, ok := schema["type"].(string)
	if !ok {
		return ""
	}

	valueType := reflect.TypeOf(value).Kind()

	switch schemaType {
	case "string":
		if valueType != reflect.String {
			return "expected string"
		}
	case "number":
		if valueType != reflect.Float64 && valueType != reflect.Int && valueType != reflect.Int64 {
			return "expected number"
		}
	case "integer":
		if valueType != reflect.Int && valueType != reflect.Int64 {
			return "expected integer"
		}
	case "boolean":
		if valueType != reflect.Bool {
			return "expected boolean"
		}
	case "array":
		if valueType != reflect.Slice && valueType != reflect.Array {
			return "expected array"
		}
	case "object":
		if valueType != reflect.Map {
			return "expected object"
		}
	}

	// Validate enum
	if enum, ok := schema["enum"].([]interface{}); ok {
		found := false
		for _, e := range enum {
			if reflect.DeepEqual(value, e) {
				found = true
				break
			}
		}
		if !found {
			return fmt.Sprintf("must be one of: %v", enum)
		}
	}

	// Validate number constraints
	if schemaType == "number" || schemaType == "integer" {
		if min, ok := schema["minimum"].(float64); ok {
			if val, ok := value.(float64); ok && val < min {
				return fmt.Sprintf("must be >= %g", min)
			}
		}
		if max, ok := schema["maximum"].(float64); ok {
			if val, ok := value.(float64); ok && val > max {
				return fmt.Sprintf("must be <= %g", max)
			}
		}
	}

	return ""
}

// Success creates a success result
func Success(data interface{}, metadata map[string]interface{}) *ToolResult {
	return &ToolResult{
		Success:  true,
		Data:     data,
		Metadata: metadata,
	}
}

// Error creates an error result
func Error(message, code string, details interface{}) *ToolResult {
	return &ToolResult{
		Success: false,
		Error: &ToolError{
			Message: message,
			Code:    code,
			Details: details,
		},
	}
}

