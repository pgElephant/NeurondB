\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo '=========================================================================='
\echo ''
\echo 'NOTE: embed_text() warnings are expected if LLM is not configured.'
\echo '      To generate real embeddings, configure:'
\echo '      - neurondb.llm_api_key (Hugging Face API key)'
\echo '      - Or enable GPU embedding via GUC (ALTER SYSTEM SET neurondb.gpu_enabled = on)'
\echo '      Without configuration, embed_text() returns zero vectors (graceful fallback).'
\echo ''

-- Test 1: Basic text embedding
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Basic text embedding (embed_text)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Basic text embedding' AS test_name,
	vector_dims(embed_text('Hello, world!')) AS dims,
	embed_text('Hello, world!') IS NOT NULL AS not_null;

-- Test 2: Text embedding with custom model
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Text embedding with custom model'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Custom model embedding' AS test_name,
	vector_dims(embed_text('Test text', 'sentence-transformers/all-MiniLM-L6-v2')) AS dims,
	embed_text('Test text', 'sentence-transformers/all-MiniLM-L6-v2') IS NOT NULL AS not_null;

-- Test 3: Batch text embedding
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Batch text embedding (embed_text_batch)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Batch embedding' AS test_name,
	array_length(embed_text_batch(ARRAY['First text', 'Second text', 'Third text']), 1) AS batch_size,
	vector_dims(embed_text_batch(ARRAY['Test'])[1]) AS first_dim;

-- Test 4: Batch embedding with NULL elements
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Batch embedding with NULL elements'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Batch with NULLs' AS test_name,
	array_length(embed_text_batch(ARRAY['Valid text', NULL, 'Another text']), 1) AS batch_size;

-- Test 5: Empty batch
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: Empty batch embedding'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Empty batch' AS test_name,
	array_length(embed_text_batch(ARRAY[]::text[]), 1) AS batch_size;

-- Test 6: Cached embedding
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 6: Cached embedding (embed_cached)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Cached embedding' AS test_name,
	vector_dims(embed_cached('Cache test text')) AS dims,
	embed_cached('Cache test text') IS NOT NULL AS not_null;

-- Test 7: Image embedding (requires BYTEA)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 7: Image embedding (embed_image)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

/* Create minimal test image data (1x1 PNG) */
DO $$
DECLARE
	test_image bytea;
BEGIN
	/* Minimal valid PNG header */
	test_image := '\x89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082'::bytea;
	PERFORM embed_image(test_image);
	RAISE NOTICE 'Image embedding test passed';
END $$;

-- Test 8: Multimodal embedding
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 8: Multimodal embedding (embed_multimodal)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	test_image bytea;
	result vector;
BEGIN
	/* Minimal valid PNG header */
	test_image := '\x89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082'::bytea;
	result := embed_multimodal('Test text', test_image);
	RAISE NOTICE 'Multimodal embedding test passed, dims: %', vector_dims(result);
END $$;

-- Test 9: Vector consistency
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 9: Vector consistency (same text, same embedding)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH embeddings AS (
	SELECT
		embed_text('Consistency test') AS vec1,
		embed_text('Consistency test') AS vec2
)
SELECT
	'vector consistency' AS test_name,
	vector_dims(vec1) = vector_dims(vec2) AS dims_match,
	vec1 <-> vec2 AS distance;

-- Test 10: Unicode and special characters
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 10: Unicode and special characters'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Unicode test' AS test_name,
	vector_dims(embed_text('Hello 世界 🌍')) AS dims,
	vector_dims(embed_text('Text with "quotes" and ''apostrophes''')) AS dims2;

-- Test 11: Long text
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 11: Long text embedding'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Long text' AS test_name,
	vector_dims(embed_text(repeat('This is a long text. ', 100))) AS dims;

-- Test 12: Model configuration
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 12: Model configuration (configure_embedding_model)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Model config' AS test_name,
	configure_embedding_model(
		'test_model',
		'{"batch_size": 32, "normalize": true, "device": "cpu", "timeout_ms": 5000}'::text
	) AS config_success;

-- Verify configuration was stored
SELECT
	'Config stored' AS test_name,
	config_json->>'batch_size' AS batch_size,
	config_json->>'normalize' AS normalize
FROM neurondb.neurondb_embedding_model_config
WHERE model_name = 'test_model';

-- Test 13: Function name aliases
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 13: Function name aliases (neurondb_embed, neurondb_embed_batch)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Alias neurondb_embed' AS test_name,
	vector_dims(neurondb_embed('Alias test')) AS dims,
	neurondb_embed('Alias test') IS NOT NULL AS not_null;

SELECT
	'Alias neurondb_embed_batch' AS test_name,
	array_length(neurondb_embed_batch(ARRAY['Text 1', 'Text 2']), 1) AS batch_size;

-- Test 14: Get embedding model configuration
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 14: Get embedding model configuration (get_embedding_model_config)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Get config' AS test_name,
	get_embedding_model_config('test_model') IS NOT NULL AS config_exists,
	(get_embedding_model_config('test_model'))->>'batch_size' AS batch_size;

-- Test 15: List embedding model configurations
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 15: List embedding model configurations (list_embedding_model_configs)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'List configs' AS test_name,
	COUNT(*) AS config_count
FROM list_embedding_model_configs();

-- Test 16: Delete embedding model configuration
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 16: Delete embedding model configuration (delete_embedding_model_config)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Delete config' AS test_name,
	delete_embedding_model_config('test_model') AS deleted;

-- Verify deletion
SELECT
	'Config deleted' AS test_name,
	COUNT(*) AS remaining_configs
FROM neurondb.neurondb_embedding_model_config
WHERE model_name = 'test_model';

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'All basic embedding tests completed!'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

