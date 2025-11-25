-- ============================================================================
-- NeuronDB HuggingFace Integration Demo - Run All Tests
-- ============================================================================
-- This script runs all HuggingFace model integration tests
-- Demonstrates: Model loading, Text generation, Classification, NER, QA,
--               Embeddings, Tokenization, Multiple model types
-- ============================================================================

\echo '======================================================================'
\echo 'NeuronDB HuggingFace Integration Demo Suite'
\echo 'Testing HuggingFace Models in PostgreSQL'
\echo '======================================================================'
\echo ''

-- Set display settings
\timing on
\x auto

-- Test 001: Model management
\echo 'Test 001: HuggingFace Model Management'
\i sql/001_model_management.sql
\echo ''

-- Test 002: Text embeddings
\echo 'Test 002: Text Embeddings with HuggingFace Models'
\i sql/002_text_embeddings.sql
\echo ''

-- Test 003: Text classification
\echo 'Test 003: Text classification'
\i sql/003_text_classification.sql
\echo ''

-- Test 004: Named Entity Recognition
\echo 'Test 004: Named Entity Recognition (NER)'
\i sql/004_ner.sql
\echo ''

-- Test 005: Question answering
\echo 'Test 005: Question Answering'
\i sql/005_question_answering.sql
\echo ''


-- Cleanup
\echo 'Test 999: Cleanup'
\i sql/999_cleanup.sql
\echo ''

\echo '======================================================================'
\echo 'NeuronDB HuggingFace Demo Complete!'
\echo '======================================================================'

