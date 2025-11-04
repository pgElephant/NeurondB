# Hybrid Search and Reranking

NeuronDB blends ANN vector search with full‑text semantics and supports reranking pipelines.

## Auto‑routing between ANN and FTS

The planner auto‑routes queries based on embedding length and query semantics. From `src/planner.c`:

- If embedding_length > 128 or query contains similarity/vector semantics → prefer ANN
- Otherwise, prefer FTS path
- It logs routing decisions at DEBUG1 level.

It also learns from query performance in a table `neurondb_query_history` including fields like `ef_search` and `beam_size`, allowing adaptive tuning over time.

## Rerank‑ready index

NeuronDB provides helper functions for rerank caching:

```sql
-- Prepare rerank cache for a table/column
SELECT rerank_index_create('docs', 'embedding', 100000, 256);

-- Retrieve precomputed candidates for a query
SELECT * FROM rerank_get_candidates('docs', 'embedding', '[0.1,0.2,0.3]'::vector, 256) LIMIT 50;

-- Warm the cache with a set of queries
SELECT rerank_index_warm('__rerank_cache_docs_embedding', ARRAY['q1','q2','q3']::text[], 256);
```

These functions manage a cache table storing candidate lists per query signature, improving rerank throughput.

## Practical fusion

- Stage 1: Combine BM25/lexical filter with ANN candidate generation.
- Stage 2: Rerank with cross‑encoder or task‑specific model.
- Stage 3: Temporal or business signals can apply secondary reranking (see `src/search/temporal_integration.c`).

## Tips

- Keep Stage 1 candidate set small but adequately covering recall.
- Use query history to refine ef_search/beam_size for frequent queries.
- Consider pre‑computing candidates for hot queries via `rerank_index_warm`.
