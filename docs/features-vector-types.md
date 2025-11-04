# Vector Types

NeuronDB includes multiple vector-like types for dense, packed, sparse, graph-based, and retrieval text data.

## Dense vector: vector

- Dense float vector suitable for ANN search and arithmetic
- Example usage:

```sql
CREATE TABLE items (
  id  bigserial PRIMARY KEY,
  vec vector
);

INSERT INTO items (vec) VALUES ('[0.1, 0.2, 0.3]'::vector);
SELECT vector_dims(vec), vector_norm(vec) FROM items LIMIT 1;
```

## Packed vector: vectorp

- Compact representation with parser and dimension helpers
- Functions from tests:

```sql
SELECT vectorp_in('[1.0, 2.0, 3.0]')::text AS vp_parse;
SELECT vectorp_dims(vectorp_in('[1.0, 2.0, 3.0, 4.0]')) AS vp_dims; -- 4
```

## Sparse vector map: vecmap

- Sparse structure using indices and values

```sql
SELECT vecmap_in('{dim:10, nnz:3, indices:[0,3,7], values:[1.5,2.3,0.8]}')::text AS vm_parse;
```

## Vector graph: vgraph

- Graph-shaped structure useful for neighbor relations / clustering

```sql
SELECT vgraph_in('{nodes:5, edges:[[0,1],[1,2],[2,3],[3,4]]}')::text;
```

## Retrieval text: rtext

- Text type used by retrieval pipelines; validated for UTF-8, rejects binary

```sql
SELECT rtext_in('sample text for retrieval')::text;
SELECT rtext_in('こんにちは')::text;
```

## Notes

- Use CHECK constraints and generated columns to keep de/serialization consistent.
- Prefer vector for search columns; use vectorp/vecmap as auxiliary or storage-optimized forms.
