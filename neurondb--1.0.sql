-- NeurondB Extension SQL Definitions
-- Advanced AI Database Extension for PostgreSQL
-- 
-- Features:
-- - Multiple vector types (float32, float16, int8, binary)
-- - 10+ distance metrics
-- - HNSW and IVF indexing
-- - ML model inference
-- - Embedding generation
-- - Hybrid search (vector + FTS + metadata)
-- - Reranking (cross-encoder, LLM, ColBERT)
-- - RAG support
-- - Analytics (clustering, PCA, UMAP, outliers)
-- - pg_stat_neurondb statistics view
-- - Vector WAL compression
-- - In-memory ANN buffer
-- - SHOW VECTOR CONFIG command

-- Include all feature modules
\i sql/neurondb_types.sql
\i sql/pg_stat_neurondb.sql
\i sql/vector_config.sql

\echo Use "CREATE EXTENSION neurondb" to load this file. \quit

-- ============================================================================
-- VECTOR TYPE DEFINITIONS
-- ============================================================================

-- Main vector type (float32)
CREATE TYPE vector;

CREATE FUNCTION vector_in(cstring) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_in'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION vector_out(vector) RETURNS cstring
    AS 'MODULE_PATHNAME', 'vector_out'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION vector_recv(internal) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_recv'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION vector_send(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_send'
    LANGUAGE C IMMUTABLE STRICT;

CREATE TYPE vector (
    INPUT = vector_in,
    OUTPUT = vector_out,
    RECEIVE = vector_recv,
    SEND = vector_send,
    STORAGE = extended,
    CATEGORY = 'U'
);

COMMENT ON TYPE vector IS 'NeurondB vector type (float32 array)';

-- ============================================================================
-- BASIC VECTOR FUNCTIONS
-- ============================================================================

CREATE FUNCTION vector_dims(vector) RETURNS integer
    AS 'MODULE_PATHNAME', 'vector_dims'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_dims IS 'Get vector dimensions';

CREATE FUNCTION vector_norm(vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_norm'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_norm IS 'Compute L2 norm of vector';

CREATE FUNCTION vector_normalize(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_normalize'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_normalize IS 'Normalize vector to unit length';

CREATE FUNCTION vector_concat(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_concat'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_concat IS 'Concatenate two vectors';

-- ============================================================================
-- VECTOR ARITHMETIC OPERATORS
-- ============================================================================

CREATE FUNCTION vector_add(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_add'
    LANGUAGE C IMMUTABLE STRICT;

CREATE OPERATOR + (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_add,
    COMMUTATOR = +
);

CREATE FUNCTION vector_sub(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_sub'
    LANGUAGE C IMMUTABLE STRICT;

CREATE OPERATOR - (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_sub
);

CREATE FUNCTION vector_mul(vector, double precision) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_mul'
    LANGUAGE C IMMUTABLE STRICT;

CREATE OPERATOR * (
    LEFTARG = vector,
    RIGHTARG = double precision,
    PROCEDURE = vector_mul,
    COMMUTATOR = *
);

-- ============================================================================
-- DISTANCE FUNCTIONS & OPERATORS
-- ============================================================================

-- L2 (Euclidean) Distance
CREATE FUNCTION vector_l2_distance(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_l2_distance IS 'L2 (Euclidean) distance between vectors';

CREATE OPERATOR <-> (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_l2_distance,
    COMMUTATOR = '<->'
);

-- Inner Product (Negative for ordering)
CREATE FUNCTION vector_inner_product(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_inner_product IS 'Negative inner product (for ordering)';

CREATE OPERATOR <#> (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_inner_product,
    COMMUTATOR = '<#>'
);

-- Cosine Distance
CREATE FUNCTION vector_cosine_distance(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_cosine_distance IS 'Cosine distance (1 - cosine similarity)';

CREATE OPERATOR <=> (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_cosine_distance,
    COMMUTATOR = '<=>'
);

-- L1 (Manhattan) Distance
CREATE FUNCTION vector_l1_distance(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_l1_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_l1_distance IS 'L1 (Manhattan) distance';

-- Hamming Distance
CREATE FUNCTION vector_hamming_distance(vector, vector) RETURNS integer
    AS 'MODULE_PATHNAME', 'vector_hamming_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_hamming_distance IS 'Hamming distance';

-- Chebyshev Distance
CREATE FUNCTION vector_chebyshev_distance(vector, vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_chebyshev_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_chebyshev_distance IS 'Chebyshev (L-infinity) distance';

-- Minkowski Distance
CREATE FUNCTION vector_minkowski_distance(vector, vector, double precision) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_minkowski_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_minkowski_distance IS 'Minkowski distance with parameter p';

-- ============================================================================
-- QUANTIZATION FUNCTIONS
-- ============================================================================

CREATE FUNCTION vector_to_int8(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_int8'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_int8 IS 'Quantize vector to int8 (8x compression)';

CREATE FUNCTION int8_to_vector(bytea) RETURNS vector
    AS 'MODULE_PATHNAME', 'int8_to_vector'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION int8_to_vector IS 'Dequantize int8 vector';

CREATE FUNCTION vector_to_binary(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_binary'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_binary IS 'Convert vector to binary (32x compression)';

CREATE FUNCTION binary_hamming_distance(bytea, bytea) RETURNS integer
    AS 'MODULE_PATHNAME', 'binary_hamming_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION binary_hamming_distance IS 'Hamming distance for binary vectors';

-- ============================================================================
-- INDEXING FUNCTIONS
-- ============================================================================

-- HNSW Index
CREATE FUNCTION hnsw_create_index(regclass, text, text, integer DEFAULT 16, integer DEFAULT 200) RETURNS boolean
    AS 'MODULE_PATHNAME', 'hnsw_create_index'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION hnsw_create_index IS 'Create HNSW index: (table, column, index_name, m, ef_construction)';

CREATE FUNCTION hnsw_knn_search(vector, integer, integer DEFAULT 100) RETURNS TABLE(id bigint, distance real)
    AS 'MODULE_PATHNAME', 'hnsw_knn_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION hnsw_knn_search IS 'HNSW K-NN search: (query, k, ef_search)';

CREATE FUNCTION hnsw_index_stats(text) RETURNS text
    AS 'MODULE_PATHNAME', 'hnsw_index_stats'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION hnsw_index_stats IS 'Get HNSW index statistics';

-- IVF Index
CREATE FUNCTION ivf_knn_search(vector, integer, integer DEFAULT 10) RETURNS TABLE(id bigint, distance real)
    AS 'MODULE_PATHNAME', 'ivf_knn_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION ivf_knn_search IS 'IVF K-NN search: (query, k, nprobe)';

CREATE FUNCTION kmeans_cluster(vector[], integer, integer DEFAULT 100) RETURNS vector[]
    AS 'MODULE_PATHNAME', 'kmeans_cluster'
    LANGUAGE C IMMUTABLE;
COMMENT ON FUNCTION kmeans_cluster IS 'K-means clustering: (vectors, k, max_iters)';

-- ============================================================================
-- ML MODEL INFERENCE
-- ============================================================================

CREATE FUNCTION load_model(text, text, text DEFAULT 'onnx') RETURNS boolean
    AS 'MODULE_PATHNAME', 'load_model'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION load_model IS 'Load ML model: (name, path, type)';

CREATE FUNCTION predict(text, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'predict'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION predict IS 'Run model inference: (model_name, input)';

CREATE FUNCTION predict_batch(text, vector[]) RETURNS vector[]
    AS 'MODULE_PATHNAME', 'predict_batch'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION predict_batch IS 'Batch inference: (model_name, inputs)';

CREATE FUNCTION list_models() RETURNS text
    AS 'MODULE_PATHNAME', 'list_models'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION list_models IS 'List all loaded models';

CREATE FUNCTION finetune_model(text, text, text DEFAULT '{}') RETURNS boolean
    AS 'MODULE_PATHNAME', 'finetune_model'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION finetune_model IS 'Fine-tune model: (model_name, train_table, config_json)';

CREATE FUNCTION export_model(text, text, text DEFAULT 'onnx') RETURNS boolean
    AS 'MODULE_PATHNAME', 'export_model'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION export_model IS 'Export model: (model_name, output_path, format)';

-- ============================================================================
-- EMBEDDING GENERATION
-- ============================================================================

CREATE FUNCTION embed_text(text, text DEFAULT 'all-MiniLM-L6-v2') RETURNS vector
    AS 'MODULE_PATHNAME', 'embed_text'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION embed_text IS 'Generate text embedding: (text, model_name)';

CREATE FUNCTION embed_text_batch(text[], text DEFAULT 'all-MiniLM-L6-v2') RETURNS vector[]
    AS 'MODULE_PATHNAME', 'embed_text_batch'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION embed_text_batch IS 'Batch text embedding: (texts, model_name)';

CREATE FUNCTION embed_image(bytea, text DEFAULT 'clip') RETURNS vector
    AS 'MODULE_PATHNAME', 'embed_image'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION embed_image IS 'Generate image embedding: (image_data, model_name)';

CREATE FUNCTION embed_multimodal(text, bytea, text DEFAULT 'clip') RETURNS vector
    AS 'MODULE_PATHNAME', 'embed_multimodal'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION embed_multimodal IS 'Multimodal embedding: (text, image, model_name)';

CREATE FUNCTION embed_cached(text, text DEFAULT 'all-MiniLM-L6-v2') RETURNS vector
    AS 'MODULE_PATHNAME', 'embed_cached'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION embed_cached IS 'Cached text embedding: (text, model_name)';

CREATE FUNCTION configure_embedding_model(text, text) RETURNS boolean
    AS 'MODULE_PATHNAME', 'configure_embedding_model'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION configure_embedding_model IS 'Configure embedding model: (model_name, config_json)';

-- ============================================================================
-- HYBRID SEARCH
-- ============================================================================

CREATE FUNCTION hybrid_search(text, vector, text, text DEFAULT '{}', double precision DEFAULT 0.7, integer DEFAULT 10)
    RETURNS TABLE(id bigint, score real)
    AS 'MODULE_PATHNAME', 'hybrid_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION hybrid_search IS 'Hybrid search: (table, query_vec, query_text, filters, vector_weight, limit)';

CREATE FUNCTION reciprocal_rank_fusion(anyarray, double precision DEFAULT 60.0) RETURNS anyarray
    AS 'MODULE_PATHNAME', 'reciprocal_rank_fusion'
    LANGUAGE C IMMUTABLE;
COMMENT ON FUNCTION reciprocal_rank_fusion IS 'Reciprocal Rank Fusion: (rankings, k)';

CREATE FUNCTION semantic_keyword_search(text, vector, text, integer DEFAULT 10)
    RETURNS TABLE(id bigint, score real)
    AS 'MODULE_PATHNAME', 'semantic_keyword_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION semantic_keyword_search IS 'Semantic + Keyword search: (table, semantic_query, keyword_query, top_k)';

CREATE FUNCTION multi_vector_search(text, vector[], text DEFAULT 'max', integer DEFAULT 10)
    RETURNS TABLE(id bigint, score real)
    AS 'MODULE_PATHNAME', 'multi_vector_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION multi_vector_search IS 'Multi-vector search: (table, query_vectors, agg_method, top_k)';

CREATE FUNCTION faceted_vector_search(text, vector, text, integer DEFAULT 3)
    RETURNS TABLE(facet text, id bigint, score real)
    AS 'MODULE_PATHNAME', 'faceted_vector_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION faceted_vector_search IS 'Faceted search: (table, query_vec, facet_column, per_facet_limit)';

CREATE FUNCTION temporal_vector_search(text, vector, text, double precision DEFAULT 0.01, integer DEFAULT 10)
    RETURNS TABLE(id bigint, score real)
    AS 'MODULE_PATHNAME', 'temporal_vector_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION temporal_vector_search IS 'Temporal search: (table, query_vec, timestamp_col, decay_rate, top_k)';

CREATE FUNCTION diverse_vector_search(text, vector, double precision DEFAULT 0.5, integer DEFAULT 10)
    RETURNS TABLE(id bigint, score real)
    AS 'MODULE_PATHNAME', 'diverse_vector_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION diverse_vector_search IS 'Diverse search (MMR): (table, query_vec, lambda, top_k)';

-- ============================================================================
-- RERANKING
-- ============================================================================

CREATE FUNCTION rerank_cross_encoder(text, text[], text DEFAULT 'ms-marco-MiniLM-L-6-v2', integer DEFAULT 10)
    RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'rerank_cross_encoder'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_cross_encoder IS 'Cross-encoder reranking: (query, candidates, model, top_k)';

CREATE FUNCTION rerank_llm(text, text[], text DEFAULT 'gpt-3.5-turbo', integer DEFAULT 10)
    RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'rerank_llm'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_llm IS 'LLM-based reranking: (query, candidates, model, top_k)';

CREATE FUNCTION rerank_cohere(text, text[], integer DEFAULT 10)
    RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'rerank_cohere'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_cohere IS 'Cohere-style reranking: (query, documents, top_n)';

CREATE FUNCTION rerank_colbert(text, text[], text DEFAULT 'colbert-v2')
    RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'rerank_colbert'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_colbert IS 'ColBERT late interaction reranking: (query, docs, model)';

CREATE FUNCTION rerank_ltr(text, text[], text, text)
    RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'rerank_ltr'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_ltr IS 'Learning-to-Rank reranking: (query, docs, features_json, model)';

CREATE FUNCTION rerank_ensemble(text, text[], text[], double precision[])
    RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'rerank_ensemble'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_ensemble IS 'Ensemble reranking: (query, docs, models, weights)';

-- ============================================================================
-- ANALYTICS
-- ============================================================================

CREATE FUNCTION cluster_kmeans(text, text, integer, integer DEFAULT 100)
    RETURNS TABLE(id bigint, cluster integer)
    AS 'MODULE_PATHNAME', 'cluster_kmeans'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION cluster_kmeans IS 'K-means clustering: (table, vector_col, num_clusters, max_iters)';

CREATE FUNCTION cluster_dbscan(text, text, double precision, integer DEFAULT 5)
    RETURNS TABLE(id bigint, cluster integer)
    AS 'MODULE_PATHNAME', 'cluster_dbscan'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION cluster_dbscan IS 'DBSCAN clustering: (table, vector_col, eps, min_pts)';

CREATE FUNCTION reduce_pca(vector, integer, text DEFAULT 'default_pca') RETURNS vector
    AS 'MODULE_PATHNAME', 'reduce_pca'
    LANGUAGE C IMMUTABLE;
COMMENT ON FUNCTION reduce_pca IS 'PCA dimensionality reduction: (input, target_dims, model_name)';

CREATE FUNCTION reduce_umap(vector, integer DEFAULT 2) RETURNS vector
    AS 'MODULE_PATHNAME', 'reduce_umap'
    LANGUAGE C IMMUTABLE;
COMMENT ON FUNCTION reduce_umap IS 'UMAP dimensionality reduction: (input, target_dims)';

CREATE FUNCTION detect_outliers(text, text, text DEFAULT 'isolation_forest', double precision DEFAULT 0.1)
    RETURNS TABLE(id bigint, outlier_score real)
    AS 'MODULE_PATHNAME', 'detect_outliers'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION detect_outliers IS 'Outlier detection: (table, vector_col, method, contamination)';

CREATE FUNCTION compute_embedding_quality(text, text, text DEFAULT NULL) RETURNS text
    AS 'MODULE_PATHNAME', 'compute_embedding_quality'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION compute_embedding_quality IS 'Compute embedding quality metrics: (table, vector_col, label_col)';

CREATE FUNCTION build_knn_graph(text, text, integer DEFAULT 10)
    RETURNS TABLE(node_id bigint, neighbor_id bigint, distance real)
    AS 'MODULE_PATHNAME', 'build_knn_graph'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION build_knn_graph IS 'Build k-NN graph: (table, vector_col, k)';

CREATE FUNCTION similarity_distribution(text, text, integer DEFAULT 1000) RETURNS text
    AS 'MODULE_PATHNAME', 'similarity_distribution'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION similarity_distribution IS 'Compute similarity distribution: (table, vector_col, num_samples)';

CREATE FUNCTION discover_topics(text, text, integer DEFAULT 10)
    RETURNS TABLE(topic_id integer, keywords text, size integer)
    AS 'MODULE_PATHNAME', 'discover_topics'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION discover_topics IS 'Topic modeling: (table, vector_col, num_topics)';

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

CREATE FUNCTION array_to_vector(real[]) RETURNS vector
    AS 'MODULE_PATHNAME', 'array_to_vector'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION array_to_vector IS 'Convert float array to vector';

CREATE FUNCTION vector_to_array(vector) RETURNS real[]
    AS 'MODULE_PATHNAME', 'vector_to_array'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_array IS 'Convert vector to float array';

-- ============================================================================
-- SUMMARY
-- ============================================================================

COMMENT ON EXTENSION neurondb IS 'NeurondB: Advanced AI Database - 100+ functions for vector search, ML inference, hybrid search, RAG, and analytics';
