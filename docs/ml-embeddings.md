# Embeddings

NeuronDB integrates text embeddings via an LLM runtime and HTTP providers.

## SQL API: embed_text

From `src/ml/embeddings.c`:

```sql
-- Generate an embedding vector for a piece of text
SELECT embed_text('hello world', 'all-MiniLM-L6-v2') AS embedding;
```

- Signature: embed_text(text, model text DEFAULT neurondb_llm_model) RETURNS vector
- Model selection and HTTP details are driven by GUCs below.

## Configuration (GUCs)

Defined in LLM worker/runtime code:

- neurondb_llm_provider: provider id (e.g., 'huggingface')
- neurondb_llm_endpoint: base URL for inference API
- neurondb_llm_model: default model name
- neurondb_llm_api_key: API key/secret
- neurondb_llm_timeout_ms: request timeout in milliseconds

Set these in postgresql.conf and restart if required for preload.

```conf
neurondb_llm_provider   = 'huggingface'
neurondb_llm_endpoint   = 'https://api-inference.huggingface.co'
neurondb_llm_model      = 'all-MiniLM-L6-v2'
neurondb_llm_api_key    = 'YOUR_KEY'
neurondb_llm_timeout_ms = 15000
```

## Usage patterns

- Ingest: compute on write

```sql
ALTER TABLE docs ADD COLUMN embedding vector;
UPDATE docs SET embedding = embed_text(content, 'all-MiniLM-L6-v2');
```

- Search: on-the-fly query embedding

```sql
WITH q AS (
  SELECT embed_text('machine learning', 'all-MiniLM-L6-v2') AS v
)
SELECT id, title
FROM docs, q
ORDER BY docs.embedding <-> q.v
LIMIT 10;
```

## Caching and async

- The LLM subsystem includes a response cache utility; repeated prompts can be cached.
- For high throughput, enqueue embedding jobs to be processed by background workers (see Background Workers). The queue schema and APIs are part of the neurondb extension.
