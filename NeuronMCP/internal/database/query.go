package database

import (
	"fmt"
	"strings"
)

// QueryBuilder provides utilities for building SQL queries
type QueryBuilder struct{}

// Select builds a SELECT query
func (qb *QueryBuilder) Select(table string, columns []string, where map[string]interface{}, orderBy *OrderBy, limit, offset *int) (string, []interface{}) {
	if len(columns) == 0 {
		columns = []string{"*"}
	}

	var params []interface{}
	paramIndex := 1

	// SELECT clause
	selectClause := strings.Join(columns, ", ")

	// FROM clause
	fromClause := EscapeIdentifier(table)

	// WHERE clause
	var whereClause string
	if len(where) > 0 {
		var conditions []string
		for key, value := range where {
			escapedKey := EscapeIdentifier(key)
			conditions = append(conditions, fmt.Sprintf("%s = $%d", escapedKey, paramIndex))
			params = append(params, value)
			paramIndex++
		}
		whereClause = "WHERE " + strings.Join(conditions, " AND ")
	}

	// ORDER BY clause
	var orderByClause string
	if orderBy != nil {
		orderByClause = fmt.Sprintf("ORDER BY %s %s", EscapeIdentifier(orderBy.Column), orderBy.Direction)
	}

	// LIMIT clause
	var limitClause string
	if limit != nil {
		limitClause = fmt.Sprintf("LIMIT $%d", paramIndex)
		params = append(params, *limit)
		paramIndex++
	}

	// OFFSET clause
	var offsetClause string
	if offset != nil {
		offsetClause = fmt.Sprintf("OFFSET $%d", paramIndex)
		params = append(params, *offset)
	}

	parts := []string{
		"SELECT " + selectClause,
		"FROM " + fromClause,
		whereClause,
		orderByClause,
		limitClause,
		offsetClause,
	}

	var nonEmptyParts []string
	for _, part := range parts {
		if part != "" {
			nonEmptyParts = append(nonEmptyParts, part)
		}
	}

	query := strings.Join(nonEmptyParts, " ")
	return query, params
}

// OrderBy represents an ORDER BY clause
type OrderBy struct {
	Column    string
	Direction string // ASC or DESC
}

// VectorSearch builds a vector search query
func (qb *QueryBuilder) VectorSearch(table, vectorColumn string, queryVector []float32, distanceMetric string, limit int, additionalColumns []string, minkowskiP *float64) (string, []interface{}) {
	var params []interface{}
	paramIndex := 1

	// Convert vector to string format
	vectorStr := formatVector(queryVector)
	params = append(params, vectorStr)

	var operator string
	var distanceExpr string

	switch distanceMetric {
	case "cosine":
		operator = "<=>"
		distanceExpr = fmt.Sprintf("%s %s $%d::vector AS distance", EscapeIdentifier(vectorColumn), operator, paramIndex)
	case "inner_product":
		operator = "<#>"
		distanceExpr = fmt.Sprintf("%s %s $%d::vector AS distance", EscapeIdentifier(vectorColumn), operator, paramIndex)
	case "l1":
		distanceExpr = fmt.Sprintf("vector_l1_distance(%s, $%d::vector) AS distance", EscapeIdentifier(vectorColumn), paramIndex)
	case "hamming":
		distanceExpr = fmt.Sprintf("vector_hamming_distance(%s, $%d::vector) AS distance", EscapeIdentifier(vectorColumn), paramIndex)
	case "chebyshev":
		distanceExpr = fmt.Sprintf("vector_chebyshev_distance(%s, $%d::vector) AS distance", EscapeIdentifier(vectorColumn), paramIndex)
	case "minkowski":
		p := 2.0
		if minkowskiP != nil {
			p = *minkowskiP
		}
		paramIndex++
		params = append(params, p)
		distanceExpr = fmt.Sprintf("vector_minkowski_distance(%s, $%d::vector, $%d::double precision) AS distance", EscapeIdentifier(vectorColumn), paramIndex-1, paramIndex)
	default: // l2
		operator = "<->"
		distanceExpr = fmt.Sprintf("%s %s $%d::vector AS distance", EscapeIdentifier(vectorColumn), operator, paramIndex)
	}

	paramIndex++
	selectColumns := []string{"*"}
	for _, col := range additionalColumns {
		selectColumns = append(selectColumns, EscapeIdentifier(col))
	}

	params = append(params, limit)
	limitClause := fmt.Sprintf("LIMIT $%d", paramIndex)

	query := fmt.Sprintf(`
		SELECT %s, %s
		FROM %s
		ORDER BY distance
		%s
	`, strings.Join(selectColumns, ", "), distanceExpr, EscapeIdentifier(table), limitClause)

	return query, params
}

// formatVector formats a float32 slice as a PostgreSQL vector string
func formatVector(vec []float32) string {
	var parts []string
	for _, v := range vec {
		parts = append(parts, fmt.Sprintf("%g", v))
	}
	return "[" + strings.Join(parts, ",") + "]"
}

