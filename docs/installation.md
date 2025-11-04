# Installation

This guide covers prerequisites, building, installing the extension, and enabling background workers.

## Prerequisites

- PostgreSQL 15 or 16 recommended
- Build tooling: gcc/clang, make, libpq headers
- Network access if using LLM/embeddings via HTTP providers
- Optional: CUDA-compatible GPU and CUDA toolkit if building with GPU acceleration

## Build and Install

If packaging is not provided, build from source:

```bash
# From repository root
make
sudo make install
```

On macOS with Homebrew PostgreSQL, ensure pg_config in PATH:

```bash
export PATH="$(pg_config --bindir):$PATH"
```

## Create the extension

In your database:

```sql
CREATE EXTENSION IF NOT EXISTS neurondb;
SELECT neurondb_version();
```

## Enable background workers (recommended)

Background workers power async queues, tuning, defragmentation, and LLM jobs. Add to postgresql.conf and restart:

```conf
# postgresql.conf
shared_preload_libraries = 'neurondb'

# Queue depth (example)
neurondb.neuranq_queue_depth = 10000

# LLM provider configuration
neurondb_llm_provider    = 'huggingface'
neurondb_llm_endpoint    = 'https://api-inference.huggingface.co'
neurondb_llm_model       = 'gpt2'          # replace with your embedding model
neurondb_llm_api_key     = 'YOUR_KEY'      # use a secure secret manager
neurondb_llm_timeout_ms  = 15000
```

Restart PostgreSQL for shared_preload_libraries to take effect.

## Validate

- Check logs for neurondb initialization messages.
- Run basic type smoke tests:

```sql
SELECT vectorp_dims(vectorp_in('[1,2,3,4]')) AS dims;  -- expects 4
SELECT vecmap_in('{dim:10, nnz:1, indices:[0], values:[1.0]}')::text;
SELECT rtext_in('hello world');
```

If errors occur, re-check the build environment and PostgreSQL version.
