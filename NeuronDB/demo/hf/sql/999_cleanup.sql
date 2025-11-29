-- ============================================================================
-- Test 999: Cleanup HuggingFace Demo
-- ============================================================================

\echo 'Cleaning up HuggingFace demo tables...'

DROP TABLE IF EXISTS hf_embeddings_test CASCADE;
DROP TABLE IF EXISTS hf_classification_test CASCADE;

\echo ''
\echo 'HuggingFace demo cleanup complete!'


