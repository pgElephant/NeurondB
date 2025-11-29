package neurondb

import (
	"math"
)

// Distance calculates the distance between two vectors
func Distance(a, b Vector, metric string) float64 {
	if len(a) != len(b) {
		return math.Inf(1) // Return infinity for mismatched dimensions
	}

	switch metric {
	case "cosine":
		return CosineDistance(a, b)
	case "l2", "euclidean":
		return L2Distance(a, b)
	case "inner_product", "dot":
		return InnerProduct(a, b)
	default:
		return L2Distance(a, b) // Default to L2
	}
}

// CosineDistance calculates cosine distance (1 - cosine similarity)
func CosineDistance(a, b Vector) float64 {
	dot := 0.0
	normA := 0.0
	normB := 0.0

	for i := range a {
		dot += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	return 1.0 - similarity
}

// L2Distance calculates L2 (Euclidean) distance
func L2Distance(a, b Vector) float64 {
	sum := 0.0
	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// InnerProduct calculates inner product (negative dot product for distance)
func InnerProduct(a, b Vector) float64 {
	dot := 0.0
	for i := range a {
		dot += float64(a[i] * b[i])
	}
	return -dot
}

// Normalize normalizes a vector to unit length
func Normalize(v Vector) Vector {
	norm := 0.0
	for _, val := range v {
		norm += float64(val * val)
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return v // Return original if zero vector
	}

	normalized := make(Vector, len(v))
	for i, val := range v {
		normalized[i] = float32(float64(val) / norm)
	}
	return normalized
}

