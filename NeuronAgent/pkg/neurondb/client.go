package neurondb

import (
	"github.com/jmoiron/sqlx"
)

// Client provides a unified interface to NeuronDB functions
type Client struct {
	Embedding *EmbeddingClient
	LLM       *LLMClient
}

// NewClient creates a new NeuronDB client
func NewClient(db *sqlx.DB) *Client {
	return &Client{
		Embedding: NewEmbeddingClient(db),
		LLM:       NewLLMClient(db),
	}
}

