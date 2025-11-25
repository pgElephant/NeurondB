-- ============================================================================
-- Test 001: HuggingFace Model Management via ONNX Runtime
-- ============================================================================
-- Demonstrates: Loading ONNX models, Model registry, Version management
-- Architecture: HuggingFace -> ONNX Export -> ONNX Runtime C API
-- ============================================================================

\echo 'NeuronDB HuggingFace Integration via ONNX Runtime'
\echo '=================================================='
\echo ''
\echo 'Architecture:'
\echo '  1. Export HuggingFace models to ONNX format'
\echo '  2. Load ONNX models using ONNX Runtime C API'
\echo '  3. Run inference via PostgreSQL functions'
\echo ''

-- Create model registry table
CREATE TABLE IF NOT EXISTS hf_models (
    model_id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    hf_model_id TEXT NOT NULL,  -- e.g., 'sentence-transformers/all-MiniLM-L6-v2'
    onnx_path TEXT,              -- Path to ONNX model file
    model_type TEXT NOT NULL,    -- 'embedding', 'classification', 'ner', 'qa', 'generation'
    input_dim INTEGER,
    output_dim INTEGER,
    tokenizer_config JSONB,
    model_config JSONB,
    loaded BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model execution sessions table
CREATE TABLE IF NOT EXISTS model_sessions (
    session_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES hf_models(model_id),
    session_handle BIGINT,       -- ONNX Runtime session pointer
    backend TEXT DEFAULT 'CPU',  -- 'CPU', 'CUDA', 'TensorRT', 'CoreML', etc.
    status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

\echo 'Registering HuggingFace models (ONNX format)...'

-- Register popular HuggingFace models for ONNX export
INSERT INTO hf_models (model_name, hf_model_id, model_type, output_dim, model_config) VALUES
    ('all-MiniLM-L6-v2', 'sentence-transformers/all-MiniLM-L6-v2', 'embedding', 384, 
     '{"task": "sentence-similarity", "max_length": 512}'::jsonb),
    
    ('all-mpnet-base-v2', 'sentence-transformers/all-mpnet-base-v2', 'embedding', 768,
     '{"task": "sentence-similarity", "max_length": 512}'::jsonb),
    
    ('distilbert-sentiment', 'distilbert-base-uncased-finetuned-sst-2-english', 'classification', 2,
     '{"task": "text-classification", "labels": ["NEGATIVE", "POSITIVE"]}'::jsonb),
    
    ('bert-ner', 'dslim/bert-base-NER', 'ner', NULL,
     '{"task": "token-classification", "labels": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]}'::jsonb),
    
    ('distilbert-qa', 'distilbert-base-cased-distilled-squad', 'qa', NULL,
     '{"task": "question-answering"}'::jsonb);

\echo ''
\echo 'ONNX Model Export Instructions:'
\echo '================================'
\echo ''
\echo 'To export HuggingFace models to ONNX format, run:'
\echo ''
\echo 'Python example:'
\echo '```python'
\echo 'from transformers import AutoModel, AutoTokenizer'
\echo 'from optimum.onnxruntime import ORTModelForFeatureExtraction'
\echo ''
\echo '# Export embedding model'
\echo 'model = ORTModelForFeatureExtraction.from_pretrained('
\echo '    "sentence-transformers/all-MiniLM-L6-v2",'
\echo '    export=True'
\echo ')'
\echo 'model.save_pretrained("./models/all-MiniLM-L6-v2-onnx")'
\echo '```'
\echo ''

-- Update model registry with ONNX paths (to be set after export)
\echo 'Model registry:'
SELECT 
    model_id,
    model_name,
    model_type,
    output_dim,
    loaded,
    CASE 
        WHEN onnx_path IS NULL THEN 'Not exported'
        ELSE 'Ready'
    END AS status
FROM hf_models
ORDER BY model_id;

\echo ''
\echo 'ONNX Runtime C API Functions (to be implemented):'
\echo '=================================================='
\echo ''
\echo 'C functions to implement in NeuronDB:'
\echo '  - neurondb_onnx_load_model(model_path, backend) -> session_id'
\echo '  - neurondb_onnx_run_inference(session_id, input_data) -> output'
\echo '  - neurondb_onnx_unload_model(session_id)'
\echo '  - neurondb_hf_tokenize(model_name, text) -> token_ids'
\echo '  - neurondb_hf_embedding(model_name, text) -> vector'
\echo '  - neurondb_hf_classify(model_name, text) -> label, score'
\echo '  - neurondb_hf_ner(model_name, text) -> entities'
\echo '  - neurondb_hf_qa(model_name, question, context) -> answer'
\echo ''


