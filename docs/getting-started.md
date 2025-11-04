# Getting Started

This quick start shows how to enable NeuronDB, create a table with vectors, insert data, and run your first queries.

## 1. Load the extension

```sql
-- As a superuser or a user with sufficient privileges
CREATE EXTENSION IF NOT EXISTS neurondb;

-- Verify install
SELECT neurondb_version(); -- e.g., 'NeurondB 1.0.0 (PG 16)'
```

## 2. Create a table with vectors

```sql
-- A simple items table with a vector column
CREATE TABLE items (
  id   bigserial PRIMARY KEY,
  text text NOT NULL,
  vec  vector NOT NULL -- dense float32 vector
);

INSERT INTO items (text, vec) VALUES
  ('hello world',       '[0.1, 0.2, 0.3]'::vector),
  ('neural database',   '[0.05, 0.7, 0.1]'::vector),
  ('ai search system',  '[0.08, 0.72, 0.12]'::vector);
```

## 3. Basic vector ops

```sql
SELECT id, vector_dims(vec) AS dims, vector_norm(vec) AS l2_norm
FROM items
LIMIT 3;

SELECT '[1,2,3]'::vector + '[4,5,6]'::vector AS added;
SELECT '[2,4]'::vector * 2.0 AS scaled;
```

## 4. KNN query

```sql
WITH q AS (
  SELECT '[0.09,0.7,0.11]'::vector AS qv
)
SELECT i.id, i.text
FROM items i, q
ORDER BY i.vec <=> q.qv  -- distance operator (example)
LIMIT 5;
```

## 5. Packed and sparse vectors

```sql
SELECT vectorp_in('[1.0, 2.0, 3.0]')::text AS vp,
       vectorp_dims(vectorp_in('[1,2,3,4]'));

SELECT vecmap_in('{dim:10, nnz:3, indices:[0,3,7], values:[1.5,2.3,0.8]}');
```

---

Next: Installation, Vector Types, and Embeddings.