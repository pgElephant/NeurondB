package tools

import (
	"fmt"
	"reflect"
)

// ValidateArgs validates arguments against a JSON Schema
func ValidateArgs(args map[string]interface{}, schema map[string]interface{}) error {
	properties, ok := schema["properties"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid schema: missing properties")
	}

	required, _ := schema["required"].([]interface{})

	// Check required fields
	for _, req := range required {
		reqStr, ok := req.(string)
		if !ok {
			continue
		}
		if _, exists := args[reqStr]; !exists {
			return fmt.Errorf("missing required field: %s", reqStr)
		}
	}

	// Validate each argument
	for key, value := range args {
		propSchema, exists := properties[key]
		if !exists {
			// Allow extra fields (could be strict mode later)
			continue
		}

		propMap, ok := propSchema.(map[string]interface{})
		if !ok {
			continue
		}

		if err := validateType(value, propMap); err != nil {
			return fmt.Errorf("invalid type for %s: %w", key, err)
		}
	}

	return nil
}

func validateType(value interface{}, schema map[string]interface{}) error {
	expectedType, ok := schema["type"].(string)
	if !ok {
		return nil // No type constraint
	}

	actualType := reflect.TypeOf(value).Kind()

	switch expectedType {
	case "string":
		if actualType != reflect.String {
			return fmt.Errorf("expected string, got %v", actualType)
		}
	case "number", "integer":
		if actualType != reflect.Float64 && actualType != reflect.Int && actualType != reflect.Int64 {
			return fmt.Errorf("expected number, got %v", actualType)
		}
	case "boolean":
		if actualType != reflect.Bool {
			return fmt.Errorf("expected boolean, got %v", actualType)
		}
	case "array":
		if actualType != reflect.Slice && actualType != reflect.Array {
			return fmt.Errorf("expected array, got %v", actualType)
		}
	case "object":
		if actualType != reflect.Map {
			return fmt.Errorf("expected object, got %v", actualType)
		}
	default:
		// Unknown type, skip validation
	}

	return nil
}

