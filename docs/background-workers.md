# Background Workers

NeuronDB uses background workers for async job processing, tuning, and maintenance.

## Workers

From the worker initialization code:

- neuranq: Queue executor for async jobs
- neuranmon: Tuner/monitor worker
- neurandefrag: Index maintenance/defragmentation
- neuranllm: LLM job processor

All require `shared_preload_libraries = 'neurondb'` and a server restart to activate.

## Configuration

Queue configuration example (from `worker_queue.c`):

```conf
# Maximum job queue size
neurondb.neuranq_queue_depth = 10000
```

LLM configuration (see Embeddings):

```conf
neurondb_llm_provider   = 'huggingface'
neurondb_llm_endpoint   = 'https://api-inference.huggingface.co'
neurondb_llm_model      = 'all-MiniLM-L6-v2'
neurondb_llm_api_key    = 'YOUR_KEY'
neurondb_llm_timeout_ms = 15000
```

## Operations

- Workers log start/stop messages and handle crashes gracefully, automatically restarting their loops.
- The queue worker uses SKIP LOCKED to pull jobs from `neurondb.neurondb_job_queue` and mark them completed/failed with timestamps.
- The LLM worker processes job types like completion, embedding, reranking and maintains a small cache.

## Troubleshooting

- If workers are not running, confirm `shared_preload_libraries` and check logs on startup.
- For excessive queue backlog, increase `neurondb.neuranq_queue_depth` and provision more resources.
