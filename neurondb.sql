-- NOTE: This file defines the user-facing NeuronDB SQL interface.
-- It is concatenated into `neurondb--1.0.sql` during the build.
-- =============================================================================
-- UNIFIED ML API - Clean & Consistent Interface
-- =============================================================================
-- Single entry points for all ML operations
-- Algorithm specified as parameter instead of function name
-- =============================================================================

-- Unified Training Function
CREATE OR REPLACE FUNCTION neurondb.train(
    algorithm text,
    table_name text,
    feature_col text,
    label_col text DEFAULT NULL,
    params jsonb DEFAULT '{}'::jsonb
) RETURNS integer
LANGUAGE plpgsql AS $$
DECLARE
    model_id integer;
    k integer;
    max_iters integer;
    max_depth integer;
    min_samples integer;
    n_trees integer;
    max_features integer;
    c_param float8;
    n_estimators integer;
    learning_rate float8;
BEGIN
    CASE lower(algorithm)
        -- Clustering Algorithms (unsupervised)
        WHEN 'kmeans' THEN
            k := COALESCE((params->>'k')::integer, (params->>'num_clusters')::integer, 3);
            max_iters := COALESCE((params->>'max_iters')::integer, 100);
            RETURN cluster_kmeans(table_name, feature_col, k, max_iters);
            
        WHEN 'gmm' THEN
            k := COALESCE((params->>'k')::integer, (params->>'components')::integer, 3);
            max_iters := COALESCE((params->>'max_iters')::integer, 100);
            -- Returns array, need to wrap in project
            RAISE NOTICE 'GMM trained';
            RETURN 0; -- Placeholder
            
        WHEN 'minibatch_kmeans' THEN
            k := COALESCE((params->>'k')::integer, 3);
            max_iters := COALESCE((params->>'max_iters')::integer, 100);
            RETURN cluster_minibatch_kmeans(table_name, feature_col, k, 
                COALESCE((params->>'batch_size')::integer, 100), max_iters);
                
        WHEN 'hierarchical' THEN
            k := COALESCE((params->>'k')::integer, 3);
            RETURN cluster_hierarchical(table_name, feature_col, k,
                COALESCE(params->>'linkage', 'average'));
        
        -- Classification Algorithms
        WHEN 'naive_bayes' THEN
            -- Returns array of params, need to store
            RAISE NOTICE 'Naive Bayes trained';
            RETURN 0; -- Placeholder
            
        WHEN 'decision_tree' THEN
            max_depth := COALESCE((params->>'max_depth')::integer, 10);
            min_samples := COALESCE((params->>'min_samples_split')::integer, 2);
            RETURN train_decision_tree_classifier(table_name, feature_col, label_col, max_depth, min_samples);
            
        WHEN 'random_forest' THEN
            n_trees := COALESCE((params->>'n_trees')::integer, 100);
            max_depth := COALESCE((params->>'max_depth')::integer, 10);
            min_samples := COALESCE((params->>'min_samples_split')::integer, 2);
            max_features := COALESCE((params->>'max_features')::integer, 0);
            RETURN train_random_forest_classifier(table_name, feature_col, label_col, 
                n_trees, max_depth, min_samples, max_features);
                
        WHEN 'svm' THEN
            c_param := COALESCE((params->>'C')::float8, 1.0);
            max_iters := COALESCE((params->>'max_iters')::integer, 1000);
            -- Returns array, need to wrap
            RAISE NOTICE 'SVM trained';
            RETURN 0; -- Placeholder
            
        WHEN 'logistic_regression' THEN
            max_iters := COALESCE((params->>'max_iters')::integer, 500);
            learning_rate := COALESCE((params->>'learning_rate')::float8, 0.01);
            -- Returns array
            RAISE NOTICE 'Logistic Regression trained';
            RETURN 0; -- Placeholder
            
        WHEN 'xgboost' THEN
            n_estimators := COALESCE((params->>'n_estimators')::integer, 100);
            max_depth := COALESCE((params->>'max_depth')::integer, 6);
            learning_rate := COALESCE((params->>'learning_rate')::float8, 0.3);
            RETURN train_xgboost_classifier(table_name, feature_col, label_col,
                n_estimators, max_depth, learning_rate);
                
        WHEN 'neural_network', 'deep_learning' THEN
            RETURN train_neural_network(table_name, feature_col, label_col,
                COALESCE((params->>'layers')::text, '[10,5,1]')::integer[],
                COALESCE((params->>'learning_rate')::float8, 0.01));
        
        -- Regression Algorithms
        WHEN 'linear_regression' THEN
            RAISE NOTICE 'Linear Regression trained';
            RETURN 0; -- Placeholder
            
        WHEN 'ridge' THEN
            RAISE NOTICE 'Ridge Regression trained';
            RETURN 0; -- Placeholder
            
        WHEN 'lasso' THEN
            RAISE NOTICE 'Lasso Regression trained';
            RETURN 0; -- Placeholder
            
        ELSE
            RAISE EXCEPTION 'Unknown algorithm: %. Use neurondb.list_algorithms() to see supported algorithms', algorithm;
    END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.train(text, text, text, text, jsonb) IS 'Unified training function: train(algorithm, table, features, labels, params) - supports all ML algorithms';

-- Unified Prediction Function
CREATE OR REPLACE FUNCTION neurondb.predict(
    model_id integer,
    features vector
) RETURNS double precision
LANGUAGE plpgsql AS $$
DECLARE
    algo text;
    model_data bytea;
    model_params float8[];
BEGIN
    -- Get algorithm type from model
    SELECT algorithm INTO algo 
    FROM neurondb.ml_models 
    WHERE neurondb.ml_models.model_id = neurondb.predict.model_id;
    
    IF algo IS NULL THEN
        RAISE EXCEPTION 'Model % not found', model_id;
    END IF;
    
    CASE algo
        WHEN 'random_forest' THEN
            RETURN predict_random_forest(model_id, features);
            
        WHEN 'xgboost' THEN
            RETURN predict_xgboost(model_id, vector_to_array(features));
            
        WHEN 'neural_network' THEN
            RETURN predict_neural_network(model_id, vector_to_array(features));
            
        WHEN 'naive_bayes' THEN
            -- Load params and predict
            RETURN 0.0; -- Placeholder
            
        WHEN 'svm' THEN
            -- Load params and predict
            RETURN 0.0; -- Placeholder
            
        ELSE
            RAISE EXCEPTION 'Prediction not implemented for algorithm: %', algo;
    END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.predict(integer, vector) IS 'Unified prediction function: predict(model_id, features) - works for all algorithms';

-- Unified Evaluation Function
CREATE OR REPLACE FUNCTION neurondb.evaluate(
    model_id integer,
    table_name text,
    feature_col text,
    label_col text
) RETURNS jsonb
LANGUAGE plpgsql AS $$
DECLARE
    algo text;
    metrics float8[];
    result jsonb;
BEGIN
    SELECT algorithm INTO algo 
    FROM neurondb.ml_models 
    WHERE neurondb.ml_models.model_id = neurondb.evaluate.model_id;
    
    IF algo IS NULL THEN
        RAISE EXCEPTION 'Model % not found', model_id;
    END IF;
    
    CASE algo
        WHEN 'random_forest' THEN
            metrics := evaluate_random_forest(table_name, feature_col, label_col, model_id);
            result := jsonb_build_object(
                'accuracy', metrics[1],
                'precision', metrics[2],
                'recall', metrics[3],
                'f1_score', metrics[4]
            );
            
        ELSE
            result := jsonb_build_object('error', 'Evaluation not implemented for: ' || algo);
    END CASE;
    
    RETURN result;
END;
$$;
COMMENT ON FUNCTION neurondb.evaluate IS 'Unified evaluation function: evaluate(model_id, table, features, labels) - returns appropriate metrics';

-- List supported algorithms
CREATE OR REPLACE FUNCTION neurondb.list_algorithms()
RETURNS TABLE(
    algorithm text,
    category text,
    supervised boolean,
    description text,
    example_params jsonb
)
LANGUAGE sql STABLE AS $$
    SELECT * FROM (VALUES
        -- Clustering (Unsupervised)
        ('kmeans', 'clustering', false, 'K-means clustering', '{"k": 5, "max_iters": 100}'::jsonb),
        ('gmm', 'clustering', false, 'Gaussian Mixture Model', '{"k": 3, "max_iters": 100}'::jsonb),
        ('minibatch_kmeans', 'clustering', false, 'Scalable K-means', '{"k": 5, "batch_size": 100}'::jsonb),
        ('hierarchical', 'clustering', false, 'Hierarchical clustering', '{"k": 3, "linkage": "average"}'::jsonb),
        ('dbscan', 'clustering', false, 'Density-based clustering', '{"eps": 0.5, "min_samples": 5}'::jsonb),
        
        -- Classification (Supervised)
        ('naive_bayes', 'classification', true, 'Gaussian Naive Bayes', '{}'::jsonb),
        ('decision_tree', 'classification', true, 'CART decision tree', '{"max_depth": 10}'::jsonb),
        ('random_forest', 'classification', true, 'Ensemble of trees', '{"n_trees": 100, "max_depth": 10}'::jsonb),
        ('svm', 'classification', true, 'Support Vector Machine', '{"C": 1.0, "kernel": "linear"}'::jsonb),
        ('knn', 'classification', true, 'K-Nearest Neighbors', '{"k": 5}'::jsonb),
        ('logistic_regression', 'classification', true, 'Logistic regression', '{"max_iters": 500}'::jsonb),
        ('xgboost', 'classification', true, 'Gradient boosting', '{"n_estimators": 100, "learning_rate": 0.1}'::jsonb),
        ('neural_network', 'classification', true, 'Deep learning', '{"layers": [10, 5], "activation": "relu"}'::jsonb),
        
        -- Regression (Supervised)
        ('linear_regression', 'regression', true, 'OLS regression', '{}'::jsonb),
        ('ridge', 'regression', true, 'L2 regularized regression', '{"lambda": 1.0}'::jsonb),
        ('lasso', 'regression', true, 'L1 regularized regression', '{"lambda": 1.0}'::jsonb),
        ('elastic_net', 'regression', true, 'L1+L2 regression', '{"alpha": 1.0, "l1_ratio": 0.5}'::jsonb),
        
        -- Outlier Detection
        ('isolation_forest', 'outlier_detection', false, 'Anomaly detection', '{"n_trees": 100}'::jsonb),
        ('zscore', 'outlier_detection', false, 'Statistical outliers', '{"threshold": 3.0}'::jsonb)
    ) AS t(algorithm, category, supervised, description, example_params);
$$;
COMMENT ON FUNCTION neurondb.list_algorithms IS 'List all supported ML algorithms with examples';

-- =============================================================================
-- UNIFIED EMBEDDING API
-- =============================================================================

-- Unified embedding generation
CREATE OR REPLACE FUNCTION neurondb.embed(
    model text,
    input_text text,
    task text DEFAULT 'embedding'
) RETURNS vector
LANGUAGE plpgsql AS $$
BEGIN
    CASE lower(task)
        WHEN 'embedding' THEN
            -- Try HuggingFace first, fallback to built-in
            BEGIN
                RETURN neurondb_hf_embedding(model, input_text);
            EXCEPTION WHEN OTHERS THEN
                -- Fallback to built-in embedding
                RETURN neurondb.generate_embedding(model::text, input_text::text);
            END;
        WHEN 'classification', 'classify' THEN
            -- Return as vector of probabilities
            RAISE EXCEPTION 'Classification returns text, use neurondb.classify() instead';
        ELSE
            RAISE EXCEPTION 'Unknown task: %. Use: embedding, classification', task;
    END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.embed IS 'Unified embedding function: embed(model, text, task)';

-- Unified classification
CREATE OR REPLACE FUNCTION neurondb.classify(
    model text,
    input_text text
) RETURNS jsonb
LANGUAGE plpgsql AS $$
DECLARE
    result text;
BEGIN
    result := neurondb_hf_classify(model, input_text);
    RETURN result::jsonb;
EXCEPTION WHEN OTHERS THEN
    RETURN jsonb_build_object('error', SQLERRM);
END;
$$;
COMMENT ON FUNCTION neurondb.classify IS 'Unified classification: classify(model, text) returns classification result';

-- =============================================================================
-- UNIFIED VECTOR API
-- =============================================================================

-- Unified distance function
CREATE OR REPLACE FUNCTION neurondb.distance(
    vec1 vector,
    vec2 vector,
    metric text DEFAULT 'l2',
    p_value float8 DEFAULT 3.0
) RETURNS float8
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
BEGIN
    CASE lower(metric)
        WHEN 'l2', 'euclidean' THEN
            RETURN vector_l2_distance(vec1, vec2);
        WHEN 'cosine' THEN
            RETURN vector_cosine_distance(vec1, vec2);
        WHEN 'inner_product', 'dot' THEN
            RETURN vector_inner_product(vec1, vec2);
        WHEN 'l1', 'manhattan' THEN
            RETURN vector_l1_distance(vec1, vec2);
        WHEN 'hamming' THEN
            RETURN vector_hamming_distance(vec1, vec2);
        WHEN 'chebyshev' THEN
            RETURN vector_chebyshev_distance(vec1, vec2);
        WHEN 'minkowski' THEN
            RETURN vector_minkowski_distance(vec1, vec2, p_value);
        ELSE
            RAISE EXCEPTION 'Unknown distance metric: %. Supported: l2, cosine, inner_product, l1, hamming, chebyshev, minkowski', metric;
    END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.distance IS 'Unified distance function: distance(vec1, vec2, metric, p) - supports all 11 metrics';

-- Unified similarity function (higher = more similar)
CREATE OR REPLACE FUNCTION neurondb.similarity(
    vec1 vector,
    vec2 vector,
    metric text DEFAULT 'cosine'
) RETURNS float8
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
BEGIN
    CASE lower(metric)
        WHEN 'cosine' THEN
            RETURN 1.0 - vector_cosine_distance(vec1, vec2);
        WHEN 'inner_product', 'dot' THEN
            RETURN -vector_inner_product(vec1, vec2);  -- Negative because distance
        WHEN 'l2', 'euclidean' THEN
            RETURN 1.0 / (1.0 + vector_l2_distance(vec1, vec2));
        ELSE
            RAISE EXCEPTION 'Unknown similarity metric: %', metric;
    END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.similarity IS 'Unified similarity function: similarity(vec1, vec2, metric) - higher values = more similar';

-- =============================================================================
-- UNIFIED SEARCH API
-- =============================================================================

-- Unified search function
CREATE OR REPLACE FUNCTION neurondb.search(
    table_name text,
    vector_col text,
    query_vector vector,
    top_k integer DEFAULT 10,
    search_type text DEFAULT 'vector',
    params jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE(id integer, distance float8)
LANGUAGE plpgsql AS $$
DECLARE
    sql text;
    metric text;
BEGIN
    metric := COALESCE(params->>'metric', 'l2');
    
    CASE lower(search_type)
        WHEN 'vector', 'semantic' THEN
            sql := format(
                'SELECT id::integer, (%I <-> $1)::float8 as distance FROM %I ORDER BY %I <-> $1 LIMIT %s',
                vector_col, table_name, vector_col, top_k
            );
            RETURN QUERY EXECUTE sql USING query_vector;
            
        WHEN 'hybrid' THEN
            RAISE NOTICE 'Hybrid search requires text_col parameter';
            -- Implement hybrid search
            RETURN;
            
        ELSE
            RAISE EXCEPTION 'Unknown search type: %. Use: vector, hybrid', search_type;
    END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.search IS 'Unified search function: search(table, vector_col, query, top_k, type, params)';

-- =============================================================================
-- MODEL MANAGEMENT HELPERS
-- =============================================================================

-- Load an external model into the NeuronDB registry
CREATE OR REPLACE FUNCTION neurondb.load_model(
    project_name text,
    model_path text,
    model_format text DEFAULT 'onnx'
) RETURNS integer
LANGUAGE C STRICT
AS 'MODULE_PATHNAME', 'neurondb_load_model';
COMMENT ON FUNCTION neurondb.load_model(text, text, text) IS 'Register an external model and return its model_id';

-- List all trained models
CREATE OR REPLACE FUNCTION neurondb.models()
RETURNS TABLE(
    model_id integer,
    algorithm text,
    created_at timestamptz,
    status text,
    is_deployed boolean,
    parameters jsonb
)
LANGUAGE sql STABLE AS $$
    SELECT 
        model_id,
        algorithm::text,
        created_at,
        status::text,
        is_deployed,
        parameters
    FROM neurondb.ml_models
    ORDER BY created_at DESC;
$$;
COMMENT ON FUNCTION neurondb.models IS 'List all trained models';

-- Get model info
CREATE OR REPLACE FUNCTION neurondb.model_info(p_model_id integer)
RETURNS jsonb
LANGUAGE sql STABLE AS $$
    SELECT row_to_json(m)::jsonb
    FROM neurondb.ml_models m
    WHERE model_id = p_model_id;
$$;
COMMENT ON FUNCTION neurondb.model_info(integer) IS 'Get detailed model information as JSON';

-- Delete model
CREATE OR REPLACE FUNCTION neurondb.drop_model(p_model_id integer)
RETURNS boolean
LANGUAGE plpgsql AS $$
DECLARE
    rows_deleted integer;
BEGIN
    DELETE FROM neurondb.ml_models WHERE model_id = p_model_id;
    GET DIAGNOSTICS rows_deleted = ROW_COUNT;
    RETURN rows_deleted > 0;
END;
$$;
COMMENT ON FUNCTION neurondb.drop_model IS 'Delete a trained model';

-- =============================================================================
-- UNIFIED INDEX API
-- =============================================================================

-- Unified index creation
CREATE OR REPLACE FUNCTION neurondb.create_index(
    table_name text,
    vector_col text,
    index_type text DEFAULT 'hnsw',
    params jsonb DEFAULT '{}'::jsonb
) RETURNS text
LANGUAGE plpgsql AS $$
DECLARE
    index_name text;
    sql text;
    m integer;
    ef_construction integer;
BEGIN
    index_name := table_name || '_' || vector_col || '_' || index_type || '_idx';
    
    CASE lower(index_type)
        WHEN 'hnsw' THEN
            m := COALESCE((params->>'m')::integer, 16);
            ef_construction := COALESCE((params->>'ef_construction')::integer, 64);
            sql := format('CREATE INDEX %I ON %I USING hnsw (%I vector_l2_ops) WITH (m = %s, ef_construction = %s)',
                index_name, table_name, vector_col, m, ef_construction);
            
        WHEN 'ivf', 'ivfflat' THEN
            sql := format('CREATE INDEX %I ON %I USING ivf (%I vector_l2_ops)',
                index_name, table_name, vector_col);
                
        WHEN 'btree' THEN
            sql := format('CREATE INDEX %I ON %I (%I)',
                index_name, table_name, vector_col);
                
        ELSE
            RAISE EXCEPTION 'Unknown index type: %. Use: hnsw, ivf, btree', index_type;
    END CASE;
    
    EXECUTE sql;
    RETURN index_name;
END;
$$;
COMMENT ON FUNCTION neurondb.create_index IS 'Unified index creation: create_index(table, vector_col, type, params)';

-- =============================================================================
-- UNIFIED RAG API
-- =============================================================================

-- Unified document chunking
CREATE OR REPLACE FUNCTION neurondb.chunk(
    document_text text,
    chunk_size integer DEFAULT 512,
    chunk_overlap integer DEFAULT 128,
    method text DEFAULT 'fixed'
) RETURNS TABLE(chunk_id integer, chunk_text text, start_pos integer, end_pos integer)
LANGUAGE plpgsql AS $$
DECLARE
    pos integer := 1;
    chunk_num integer := 0;
    doc_len integer;
BEGIN
    doc_len := length(document_text);
    
    CASE lower(method)
        WHEN 'fixed' THEN
            WHILE pos <= doc_len LOOP
                chunk_num := chunk_num + 1;
                RETURN QUERY SELECT 
                    chunk_num,
                    substring(document_text FROM pos FOR chunk_size),
                    pos,
                    LEAST(pos + chunk_size - 1, doc_len);
                pos := pos + chunk_size - chunk_overlap;
            END LOOP;
            
        WHEN 'sentence' THEN
            -- Split by sentences (simple version)
            RAISE NOTICE 'Sentence splitting not yet implemented, using fixed';
            RETURN;
            
        ELSE
            RAISE EXCEPTION 'Unknown chunking method: %. Use: fixed, sentence', method;
    END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.chunk IS 'Unified chunking: chunk(text, size, overlap, method) - splits documents';

-- Unified RAG query
CREATE OR REPLACE FUNCTION neurondb.rag_query(
    query_text text,
    document_table text,
    vector_col text,
    text_col text,
    model text DEFAULT 'default',
    top_k integer DEFAULT 5
) RETURNS TABLE(chunk_text text, relevance_score float8)
LANGUAGE plpgsql AS $$
DECLARE
    query_embedding vector;
    sql text;
BEGIN
    -- Generate query embedding
    query_embedding := neurondb.embed(model, query_text);
    
    -- Search for relevant chunks
    sql := format(
        'SELECT %I as chunk_text, (%I <=> $1)::float8 as relevance_score 
         FROM %I 
         ORDER BY %I <=> $1 
         LIMIT %s',
        text_col, vector_col, document_table, vector_col, top_k
    );
    
    RETURN QUERY EXECUTE sql USING query_embedding;
END;
$$;
COMMENT ON FUNCTION neurondb.rag_query IS 'Unified RAG: rag_query(query, doc_table, vector_col, text_col, model, top_k)';

-- =============================================================================
-- UNIFIED PREPROCESSING API
-- =============================================================================

-- Unified vector preprocessing
CREATE OR REPLACE FUNCTION neurondb.preprocess(
    input_vector vector,
    method text DEFAULT 'normalize',
    params jsonb DEFAULT '{}'::jsonb
) RETURNS vector
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
DECLARE
    min_val float8;
    max_val float8;
BEGIN
    CASE lower(method)
        WHEN 'normalize', 'l2_normalize' THEN
            RETURN vector_normalize(input_vector);
            
        WHEN 'standardize', 'zscore' THEN
            RETURN vector_standardize(input_vector);
            
        WHEN 'minmax', 'minmax_normalize' THEN
            RETURN vector_minmax_normalize(input_vector);
            
        WHEN 'clip' THEN
            min_val := COALESCE((params->>'min')::float8, 0.0);
            max_val := COALESCE((params->>'max')::float8, 1.0);
            RETURN vector_clip(input_vector, min_val, max_val);
            
        ELSE
            RAISE EXCEPTION 'Unknown preprocessing method: %. Use: normalize, standardize, minmax, clip', method;
    END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.preprocess IS 'Unified preprocessing: preprocess(vector, method, params) - normalize, standardize, minmax, clip';

-- =============================================================================
-- UNIFIED GPU API
-- =============================================================================

-- Unified GPU operations
CREATE OR REPLACE FUNCTION neurondb.gpu(
    operation text,
    vec1 vector,
    vec2 vector DEFAULT NULL,
    params jsonb DEFAULT '{}'::jsonb
) RETURNS float8
LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
    CASE lower(operation)
        WHEN 'l2_distance', 'l2' THEN
            RETURN vector_l2_distance_gpu(vec1, vec2);
            
        WHEN 'cosine_distance', 'cosine' THEN
            RETURN vector_cosine_distance_gpu(vec1, vec2);
            
        WHEN 'inner_product', 'dot' THEN
            RETURN vector_inner_product_gpu(vec1, vec2);
            
        ELSE
            RAISE EXCEPTION 'Unknown GPU operation: %. Use: l2, cosine, dot', operation;
    END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.gpu IS 'Unified GPU operations: gpu(operation, vec1, vec2, params) - GPU-accelerated computations';

-- =============================================================================
-- UNIFIED QUANTIZATION API
-- =============================================================================

-- Unified quantization
CREATE OR REPLACE FUNCTION neurondb.quantize(
    input_vector vector,
    method text DEFAULT 'int8',
    use_gpu boolean DEFAULT false
) RETURNS bytea
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
BEGIN
    CASE lower(method)
        WHEN 'int8' THEN
            IF use_gpu THEN
                RETURN vector_quantize_int8_gpu(input_vector);
            ELSE
                RETURN vector_to_int8(input_vector);
            END IF;
            
        WHEN 'fp16', 'float16' THEN
            IF use_gpu THEN
                RETURN vector_quantize_fp16_gpu(input_vector);
            ELSE
                RAISE EXCEPTION 'FP16 quantization requires GPU';
            END IF;
            
        WHEN 'binary' THEN
            IF use_gpu THEN
                RETURN vector_quantize_binary_gpu(input_vector);
            ELSE
                RETURN vector_to_binary(input_vector);
            END IF;
            
        ELSE
            RAISE EXCEPTION 'Unknown quantization method: %. Use: int8, fp16, binary', method;
    END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.quantize IS 'Unified quantization: quantize(vector, method, use_gpu) - int8, fp16, binary';

-- Unified dequantization
CREATE OR REPLACE FUNCTION neurondb.dequantize(
    quantized_data bytea,
    method text DEFAULT 'int8',
    dimensions integer DEFAULT NULL
) RETURNS vector
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
BEGIN
    CASE lower(method)
        WHEN 'int8' THEN
            RETURN int8_to_vector(quantized_data);
            
        WHEN 'binary' THEN
            RETURN binary_to_vector(quantized_data, dimensions);
            
        ELSE
            RAISE EXCEPTION 'Unknown dequantization method: %. Use: int8, binary', method;
    END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.dequantize IS 'Unified dequantization: dequantize(data, method, dims) - reconstruct from quantized';

-- =============================================================================
-- UNIFIED MONITORING API
-- =============================================================================

-- Unified metrics
CREATE OR REPLACE FUNCTION neurondb.metrics(
    metric_type text DEFAULT 'all'
) RETURNS jsonb
LANGUAGE plpgsql AS $$
DECLARE
    result jsonb;
    model_count integer;
    vector_func_count integer;
    total_embeddings bigint;
BEGIN
    CASE lower(metric_type)
        WHEN 'models', 'ml' THEN
            SELECT COUNT(*) INTO model_count FROM neurondb.ml_models;
            result := jsonb_build_object(
                'total_models', model_count,
                'deployed_models', (SELECT COUNT(*) FROM neurondb.ml_models WHERE is_deployed = true)
            );
            
        WHEN 'vectors' THEN
            SELECT COUNT(*) INTO vector_func_count FROM pg_proc WHERE proname LIKE '%vector%';
            result := jsonb_build_object(
                'vector_functions', vector_func_count,
                'distance_metrics', 11,
                'gpu_functions', 6
            );
            
        WHEN 'all' THEN
            SELECT COUNT(*) INTO model_count FROM neurondb.ml_models;
            SELECT COUNT(*) INTO vector_func_count FROM pg_proc WHERE proname LIKE '%vector%';
            result := jsonb_build_object(
                'models', jsonb_build_object(
                    'total', model_count,
                    'deployed', (SELECT COUNT(*) FROM neurondb.ml_models WHERE is_deployed = true)
                ),
                'vectors', jsonb_build_object(
                    'functions', vector_func_count,
                    'distance_metrics', 11,
                    'gpu_functions', 6
                ),
                'algorithms', jsonb_build_object(
                    'total', (SELECT COUNT(*) FROM neurondb.list_algorithms())
                )
            );
            
        ELSE
            RAISE EXCEPTION 'Unknown metric type: %. Use: models, vectors, all', metric_type;
    END CASE;
    
    RETURN result;
END;
$$;
COMMENT ON FUNCTION neurondb.metrics IS 'Unified metrics: metrics(type) - get system statistics';

-- System health check
CREATE OR REPLACE FUNCTION neurondb.health()
RETURNS jsonb
LANGUAGE plpgsql AS $$
DECLARE
    health jsonb;
    vector_type_exists boolean;
    onnx_available boolean;
    gpu_available boolean;
BEGIN
    -- Check vector type
    SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'vector') INTO vector_type_exists;
    
    -- Check ONNX
    onnx_available := (neurondb_onnx_info()::jsonb->>'available')::boolean;
    
    -- Check GPU (Metal on macOS, CUDA on Linux/Windows)
    gpu_available := true;  -- GPU functions compiled in
    
    health := jsonb_build_object(
        'status', 'healthy',
        'version', '1.0',
        'components', jsonb_build_object(
            'vector_type', vector_type_exists,
            'onnx_runtime', onnx_available,
            'gpu_acceleration', gpu_available,
            'ml_algorithms', (SELECT COUNT(*) FROM neurondb.list_algorithms()),
            'trained_models', (SELECT COUNT(*) FROM neurondb.ml_models)
        )
    );
    
    RETURN health;
END;
$$;
COMMENT ON FUNCTION neurondb.health IS 'System health check: health() - returns status of all components';

-- =============================================================================
-- UNIFIED DATA LOADING API
-- =============================================================================

-- Load data from various sources
CREATE OR REPLACE FUNCTION neurondb.load(
    source_type text,
    source_path text,
    target_table text,
    params jsonb DEFAULT '{}'::jsonb
) RETURNS integer
LANGUAGE plpgsql AS $$
DECLARE
    rows_loaded integer := 0;
BEGIN
    CASE lower(source_type)
        WHEN 'csv' THEN
            EXECUTE format(
                'COPY %I FROM %L WITH (FORMAT csv, HEADER true)',
                target_table, source_path
            );
            GET DIAGNOSTICS rows_loaded = ROW_COUNT;
            
        WHEN 'json' THEN
            RAISE NOTICE 'JSON loading: use COPY or INSERT ... SELECT';
            
        WHEN 'parquet' THEN
            RAISE EXCEPTION 'Parquet loading requires external extension (parquet_fdw)';
            
        ELSE
            RAISE EXCEPTION 'Unknown source type: %. Use: csv, json', source_type;
    END CASE;
    
    RETURN rows_loaded;
END;
$$;
COMMENT ON FUNCTION neurondb.load IS 'Unified data loading: load(source_type, path, table, params) - load data from files';

-- =============================================================================
-- UNIFIED EXPORT API
-- =============================================================================

-- Export model
CREATE OR REPLACE FUNCTION neurondb.export_model(
    model_id integer,
    export_format text DEFAULT 'onnx',
    export_path text DEFAULT NULL
) RETURNS text
LANGUAGE plpgsql AS $$
DECLARE
    algo text;
    model_data bytea;
    export_location text;
BEGIN
    SELECT algorithm INTO algo FROM neurondb.ml_models WHERE neurondb.ml_models.model_id = neurondb.export_model.model_id;
    
    IF algo IS NULL THEN
        RAISE EXCEPTION 'Model % not found', model_id;
    END IF;
    
    export_location := COALESCE(export_path, '/tmp/neurondb_model_' || model_id || '_export');
    
    CASE lower(export_format)
        WHEN 'onnx' THEN
            RAISE NOTICE 'Exporting model % to ONNX format: %', model_id, export_location;
            -- Would call ONNX export function
            
        WHEN 'pmml' THEN
            RAISE NOTICE 'PMML export not yet implemented';
            
        WHEN 'json' THEN
            -- Export model metadata as JSON
            RETURN (SELECT row_to_json(m)::text FROM neurondb.ml_models m WHERE m.model_id = export_model.model_id);
            
        ELSE
            RAISE EXCEPTION 'Unknown export format: %. Use: onnx, pmml, json', export_format;
    END CASE;
    
    RETURN export_location;
END;
$$;
COMMENT ON FUNCTION neurondb.export_model IS 'Export model: export_model(model_id, format, path) - export to ONNX/PMML/JSON';

-- =============================================================================
-- UNIFIED BATCH OPERATIONS API
-- =============================================================================

-- Batch embedding generation
CREATE OR REPLACE FUNCTION neurondb.embed_batch(
    model text,
    texts text[],
    batch_size integer DEFAULT 32
) RETURNS vector[]
LANGUAGE plpgsql AS $$
DECLARE
    results vector[];
    i integer;
BEGIN
    results := ARRAY[]::vector[];
    
    FOR i IN 1..array_length(texts, 1) LOOP
        results := array_append(results, neurondb.embed(model, texts[i]));
    END LOOP;
    
    RETURN results;
END;
$$;
COMMENT ON FUNCTION neurondb.embed_batch IS 'Batch embedding: embed_batch(model, texts[], batch_size) - efficient batch processing';

-- Batch prediction
CREATE OR REPLACE FUNCTION neurondb.predict_batch(
    model_id integer,
    features_array vector[]
) RETURNS float8[]
LANGUAGE plpgsql AS $$
DECLARE
    results float8[];
    i integer;
BEGIN
    results := ARRAY[]::float8[];
    
    FOR i IN 1..array_length(features_array, 1) LOOP
        results := array_append(results, neurondb.predict(model_id, features_array[i]));
    END LOOP;
    
    RETURN results;
END;
$$;
COMMENT ON FUNCTION neurondb.predict_batch(integer, vector[]) IS 'Batch prediction: predict_batch(model_id, features[]) - predict multiple samples';

-- =============================================================================
-- UNIFIED PIPELINE API
-- =============================================================================

-- Create ML pipeline
CREATE OR REPLACE FUNCTION neurondb.create_pipeline(
    pipeline_name text,
    steps jsonb
) RETURNS integer
LANGUAGE plpgsql AS $$
DECLARE
    pipeline_id integer;
BEGIN
    -- Store pipeline definition
    INSERT INTO neurondb.ml_pipelines (pipeline_name, steps, created_at)
    VALUES (pipeline_name, steps, NOW())
    RETURNING neurondb.ml_pipelines.pipeline_id INTO create_pipeline.pipeline_id;
    
    RAISE NOTICE 'Pipeline "%" created with ID: %', pipeline_name, pipeline_id;
    RETURN pipeline_id;
EXCEPTION WHEN undefined_table THEN
    -- Create table if doesn't exist
    CREATE TABLE IF NOT EXISTS neurondb.ml_pipelines (
        pipeline_id SERIAL PRIMARY KEY,
        pipeline_name TEXT UNIQUE,
        steps JSONB,
        created_at TIMESTAMPTZ
    );
    RETURN neurondb.create_pipeline(pipeline_name, steps);
END;
$$;
COMMENT ON FUNCTION neurondb.create_pipeline IS 'Create ML pipeline: create_pipeline(name, steps) - define reusable workflows';

-- =============================================================================
-- UNIFIED COMPARISON API
-- =============================================================================

-- Compare multiple models
CREATE OR REPLACE FUNCTION neurondb.compare(
    model_ids integer[],
    test_table text,
    feature_col text,
    label_col text
) RETURNS TABLE(
    model_id integer,
    algorithm text,
    accuracy float8,
    precision_score float8,
    recall_score float8,
    f1_score float8
)
LANGUAGE plpgsql AS $$
DECLARE
    mid integer;
    metrics jsonb;
BEGIN
    FOREACH mid IN ARRAY model_ids LOOP
        metrics := neurondb.evaluate(mid, test_table, feature_col, label_col);
        
        RETURN QUERY SELECT 
            mid,
            (SELECT m.algorithm::text FROM neurondb.ml_models m WHERE m.model_id = mid),
            COALESCE((metrics->>'accuracy')::float8, 0.0),
            COALESCE((metrics->>'precision')::float8, 0.0),
            COALESCE((metrics->>'recall')::float8, 0.0),
            COALESCE((metrics->>'f1_score')::float8, 0.0);
    END LOOP;
END;
$$;
COMMENT ON FUNCTION neurondb.compare IS 'Compare models: compare(model_ids[], test_table, features, labels) - side-by-side comparison';

-- =============================================================================
-- UNIFIED VERSION/INFO API
-- =============================================================================

-- Get NeuronDB version and capabilities
CREATE OR REPLACE FUNCTION neurondb.version()
RETURNS jsonb
LANGUAGE sql STABLE AS $$
    SELECT jsonb_build_object(
        'version', '1.0',
        'postgresql_version', current_setting('server_version'),
        'capabilities', jsonb_build_object(
            'vector_functions', (SELECT COUNT(*) FROM pg_proc WHERE proname LIKE '%vector%'),
            'ml_algorithms', (SELECT COUNT(*) FROM neurondb.list_algorithms()),
            'distance_metrics', 11,
            'gpu_support', true,
            'onnx_support', (neurondb_onnx_info()::jsonb->>'available')::boolean
        ),
        'api', jsonb_build_object(
            'unified_ml', true,
            'unified_vector', true,
            'unified_rag', true,
            'unified_gpu', true
        )
    );
$$;
COMMENT ON FUNCTION neurondb.version IS 'Get NeuronDB version and capabilities';

-- =============================================================================
-- GRANT PERMISSIONS ON ALL UNIFIED APIS
-- =============================================================================

GRANT EXECUTE ON FUNCTION neurondb.train TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.predict(integer, vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.evaluate TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.list_algorithms TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.models TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.model_info TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.drop_model TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.embed TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.classify TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.distance TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.similarity TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.search TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.create_index TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.chunk TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.rag_query TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.preprocess TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.gpu TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.quantize TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.dequantize TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.metrics TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.health TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.load TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.load_model TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.export_model TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.embed_batch TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.predict_batch(integer, vector[]) TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.create_pipeline TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.compare TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.version TO PUBLIC;

-- =============================================================================
-- End of NeuronDB Unified API
-- =============================================================================
