-- ============================================================================
-- Test 004: Named Entity Recognition (NER)
-- ============================================================================
-- Demonstrates: NER with HuggingFace models (future release)
-- ============================================================================

\echo 'NER functionality will be available in next release'
\echo 'Requires token-level classification implementation'
\echo ''
\echo 'Example usage (when available):'
\echo '  SELECT neurondb_hf_ner(''bert-ner'', ''Apple Inc. is in California'');'
\echo '  Result: [{"entity": "ORG", "text": "Apple Inc."}, {"entity": "LOC", "text": "California"}]'


