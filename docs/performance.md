# Performance and Tuning

This page summarizes tuning levers for vector search, reranking, and ML utilities.

## Query planner auto‑tuning

NeuronDB logs auto‑routing between ANN and FTS and learns from queries in `neurondb_query_history`, capturing:

- fingerprint, seen_count
- ef_search, beam_size
- last_recall, last_latency

Use this feedback to pick defaults for hot query templates.

## Rerank‑ready index

Reranking throughput benefits from candidate caches:

```sql
SELECT rerank_index_create('docs','embedding', 100000, 256);
SELECT * FROM rerank_get_candidates('docs','embedding', '[0.1,0.2,0.3]'::vector, 256) LIMIT 50;
SELECT rerank_index_warm('__rerank_cache_docs_embedding', ARRAY['q1','q2']::text[], 256);
```

- Cache size: balance memory vs. hit rate
- k candidates: balance recall vs. rerank compute cost

## Embeddings

- Batch updates when (re)embedding to amortize HTTP latency
- Cache common prompts or use the LLM cache utilities
- Increase `neurondb_llm_timeout_ms` for larger models or slower endpoints

## Dimensionality reduction and whitening

From `sql/14_ml_dimensionality.sql`:

```sql
-- PCA reduction (k dims)
SELECT * FROM neurondb.reduce_pca('table_name', 'vec', 2);

-- Whitening
SELECT * FROM neurondb.whiten_embeddings('table_name', 'vec');
```

- Reduce dimensionality to improve ANN speed and index size
- Whitening can stabilize cosine similarity in some corpora

## Clustering (workload shaping)

From `sql/13_ml_clustering.sql`:

```sql
SELECT * FROM neurondb.cluster_kmeans('tbl','vec', 3, 100);
SELECT * FROM neurondb.cluster_minibatch_kmeans('tbl','vec', 3, 3, 50);
SELECT * FROM neurondb.cluster_dbscan('tbl','vec', 1.0, 2);
SELECT * FROM neurondb.cluster_gmm('tbl','vec', 3, 50);
SELECT * FROM neurondb.cluster_hierarchical('tbl','vec', 3, 'single');
```

- Use clustering to pre‑partition data or warm rerank caches by cluster centroids

## System considerations

- Memory: ensure sufficient shared buffers and work_mem for candidate lists
- Storage: prefer fast NVMe; consider TOAST settings for vector columns
- GPU: if available and built in, test gains on your workload; results vary with dims and batch sizes
