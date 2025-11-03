# NeurondB SQL File Reorganization Plan

## Current Status
- File: neurondb--1.0.sql
- Lines: 2,865
- C Functions: 202
- PL/pgSQL Functions: 23
- Tables: 18
- Views: 7

## Target Structure (Well-Organized)

### Section 1: Header & Metadata (Lines 1-30)
- Copyright and license
- Extension description
- Feature list (concise)
- Usage note

### Section 2: Schema Setup (Lines 31-35)
- CREATE SCHEMA neurondb

### Section 3: Core Types (Lines 36-200)
- vector type (float32)
- svector type (sparse)
- bvector type (binary)
- vectorp type (packed)
- vecmap type (sparse map)

### Section 4: Distance Functions (Lines 201-350)
- L2, L1, cosine, inner product
- Hamming, Jaccard, etc.
- GPU variants

### Section 5: Operators (Lines 351-450)
- +, -, *, /
- <->, <#>, <=>
- Comparison operators

### Section 6: Core Tables (Lines 451-900)
All neurondb.* tables grouped by purpose:
- Configuration: llm_config
- Multi-tenancy: tenant_usage, tenant_quotas
- Security: rls_policies
- Indexing: index_metadata, neurondb_index_maintenance
- Caching: neurondb_embedding_cache, neurondb_llm_cache
- Jobs: neurondb_job_queue, neurondb_llm_jobs
- Metrics: neurondb_query_metrics, neurondb_prometheus_metrics, neurondb_llm_stats, neurondb_histograms
- ML Projects: ml_projects, ml_experiments, ml_trained_models, ml_predictions

###  Section 7: Views (Lines 901-1100)
- vector_stats
- index_health
- tenant_quota_usage
- llm_job_status
- query_performance
- index_maintenance_status
- metrics_summary
- recent_predictions

### Section 8: C Functions by Category (Lines 1101-2400)
Group by purpose:
- Vector operations
- Distance calculations
- Index management
- ML algorithms (clustering, PCA, quantization, etc.)
- Embeddings
- RAG pipeline
- Reranking
- Analytics
- LLM integration
- Monitoring

### Section 9: PL/pgSQL Helper Functions (Lines 2401-2600)
- Project management
- Model management
- Training helpers
- Inference helpers

### Section 10: Permissions & Grants (Lines 2601-2800)
- Schema grants
- Table grants
- View grants
- Function grants (organized)
- Sequence grants

### Section 11: Footer (Lines 2801-2865)
- Extension metadata
- End marker

## Cleanup Rules
1. Remove redundant comments
2. Keep only essential documentation
3. Remove verbose feature lists
4. Consolidate similar functions
5. Use consistent formatting
6. Group related objects together

