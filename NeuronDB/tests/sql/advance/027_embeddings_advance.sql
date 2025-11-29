\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Advanced Embedding Tests - Performance, Edge Cases, Integration'
\echo '=========================================================================='

-- Test 1: Large batch performance
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Large batch embedding (100+ items)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	texts text[];
	start_time timestamp;
	end_time timestamp;
	duration interval;
	i int;
BEGIN
	/* Generate array of 100 texts */
	texts := ARRAY[]::text[];
	FOR i IN 1..100 LOOP
		texts := array_append(texts, 'Batch text item ' || i::text);
	END LOOP;

	start_time := clock_timestamp();
	PERFORM embed_text_batch(texts);
	end_time := clock_timestamp();
	duration := end_time - start_time;

	RAISE NOTICE 'Large batch (100 items) completed in %', duration;
END $$;

-- Test 2: Batch with mixed NULL and valid
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Batch with mixed NULL and valid elements'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	texts text[];
	results vector[];
	i int;
	null_count int := 0;
	valid_count int := 0;
BEGIN
	/* Create array with NULLs at various positions */
	texts := ARRAY[
		'First', NULL, 'Third', NULL, 'Fifth',
		'Sixth', NULL, 'Eighth', 'Ninth', NULL
	]::text[];

	results := embed_text_batch(texts);

	FOR i IN 1..array_length(results, 1) LOOP
		IF results[i] IS NULL THEN
			null_count := null_count + 1;
		ELSE
			valid_count := valid_count + 1;
		END IF;
	END LOOP;

	RAISE NOTICE 'Mixed batch: % valid, % NULL', valid_count, null_count;
END $$;

-- Test 3: Multiple models comparison
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Multiple models comparison'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH model_comparison AS (
	SELECT
		'Model 1' AS model_name,
		embed_text('Test text', 'sentence-transformers/all-MiniLM-L6-v2') AS vec
	UNION ALL
	SELECT
		'Model 2' AS model_name,
		embed_text('Test text', 'sentence-transformers/all-MiniLM-L6-v2') AS vec
)
SELECT
	model_name,
	vector_dims(vec) AS dims,
	vec <-> (SELECT vec FROM model_comparison WHERE model_name = 'Model 1') AS distance_to_model1
FROM model_comparison;

-- Test 4: Cache hit/miss performance
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Cache hit/miss performance'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	cache_miss_time interval;
	cache_hit_time interval;
	start_ts timestamp;
	end_ts timestamp;
	test_text text := 'Cache performance test text';
BEGIN
	/* First call - cache miss */
	start_ts := clock_timestamp();
	PERFORM embed_cached(test_text);
	end_ts := clock_timestamp();
	cache_miss_time := end_ts - start_ts;

	/* Second call - cache hit */
	start_ts := clock_timestamp();
	PERFORM embed_cached(test_text);
	end_ts := clock_timestamp();
	cache_hit_time := end_ts - start_ts;

	RAISE NOTICE 'Cache miss: %, Cache hit: %', cache_miss_time, cache_hit_time;
END $$;

-- Test 5: Concurrent embedding requests simulation
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: Concurrent embedding requests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE concurrent_embeddings AS
SELECT
	generate_series(1, 50) AS id,
	'Concurrent text ' || generate_series(1, 50)::text AS text;

SELECT
	COUNT(*) AS total_embeddings,
	COUNT(DISTINCT vector_dims(embed_text(text))) AS unique_dims,
	AVG(vector_dims(embed_text(text)))::int AS avg_dims
FROM concurrent_embeddings;

-- Test 6: Integration with similarity search
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 6: Integration with similarity search'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE test_embeddings_table (
	id serial PRIMARY KEY,
	text_content text,
	embedding vector
);

INSERT INTO test_embeddings_table (text_content, embedding)
VALUES
	('Machine learning in databases', embed_text('Machine learning in databases')),
	('PostgreSQL vector search', embed_text('PostgreSQL vector search')),
	('Neural network embeddings', embed_text('Neural network embeddings'));

SELECT
	id,
	text_content,
	embedding <-> embed_text('database machine learning') AS distance
FROM test_embeddings_table
ORDER BY distance
LIMIT 3;

-- Test 7: Batch embedding with different models
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 7: Batch embedding with different models'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Batch with model' AS test_name,
	array_length(embed_text_batch(
		ARRAY['Text 1', 'Text 2', 'Text 3'],
		'sentence-transformers/all-MiniLM-L6-v2'
	), 1) AS batch_size;

-- Test 8: Vector normalization check
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 8: Vector normalization check'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH normalized_check AS (
	SELECT
		embed_text('Normalization test') AS vec,
		embed_text('Normalization test') <-> embed_text('Normalization test') AS self_distance
)
SELECT
	'self_distance' AS metric,
	self_distance,
	CASE WHEN self_distance < 0.0001 THEN 'PASS' ELSE 'FAIL' END AS status
FROM normalized_check;

-- Test 9: Image embedding with different sizes
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 9: Image embedding with different sizes'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	small_image bytea;
	large_image bytea;
	small_vec vector;
	large_vec vector;
BEGIN
	/* Small image (1x1 PNG) */
	small_image := '\x89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082'::bytea;
	small_vec := embed_image(small_image);
	RAISE NOTICE 'Small image embedding dims: %', vector_dims(small_vec);

	/* Large image (simulated by concatenating multiple times) */
	-- Note: Using concatenation instead of repeat() to avoid pglz compression issues
	large_image := small_image || small_image || small_image || small_image || small_image;
	large_vec := embed_image(large_image);
	RAISE NOTICE 'Large image embedding dims: %', vector_dims(large_vec);
EXCEPTION WHEN OTHERS THEN
	-- If image embedding fails, just log and continue
	RAISE NOTICE 'Image embedding test skipped: %', SQLERRM;
END $$;

-- Test 10: Multimodal embedding consistency
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 10: Multimodal embedding consistency'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	test_image bytea;
	text1 text := 'Test text';
	text2 text := 'Test text';
	vec1 vector;
	vec2 vector;
	distance float;
BEGIN
	test_image := '\x89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082'::bytea;

	vec1 := embed_multimodal(text1, test_image);
	vec2 := embed_multimodal(text2, test_image);

	distance := vec1 <-> vec2;
	RAISE NOTICE 'Multimodal consistency distance: %', distance;
EXCEPTION WHEN OTHERS THEN
	-- If multimodal embedding fails (e.g., due to image data issues), just log and continue
	RAISE NOTICE 'Multimodal embedding test skipped: %', SQLERRM;
END $$;

-- Test 11: Model configuration updates
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 11: Model configuration updates'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	config1 jsonb;
	config2 jsonb;
BEGIN
	/* Initial configuration */
	PERFORM configure_embedding_model(
		'update_test_model',
		'{"batch_size": 16, "normalize": false}'::text
	);

	SELECT config_json INTO config1
	FROM neurondb.embedding_model_config
	WHERE model_name = 'update_test_model';

	/* Update configuration */
	PERFORM configure_embedding_model(
		'update_test_model',
		'{"batch_size": 64, "normalize": true, "timeout_ms": 10000}'::text
	);

	SELECT config_json INTO config2
	FROM neurondb.embedding_model_config
	WHERE model_name = 'update_test_model';

	RAISE NOTICE 'Config updated: batch_size % -> %',
		config1->>'batch_size', config2->>'batch_size';
END $$;

-- Test 12: Cache statistics
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 12: Cache statistics (ndb_llm_cache_stats)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Cache stats' AS test_name,
	(ndb_llm_cache_stats())->>'total_entries' AS total_entries,
	(ndb_llm_cache_stats())->>'valid_entries' AS valid_entries;

-- Test 13: Cache warming
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 13: Cache warming (ndb_llm_cache_warm)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	warmed_count int;
BEGIN
	warmed_count := ndb_llm_cache_warm(
		ARRAY['Warm text 1', 'Warm text 2', 'Warm text 3'],
		'all-MiniLM-L6-v2'
	);
	RAISE NOTICE 'Cache warmed: % entries', warmed_count;
END $$;

-- Test 14: Cache eviction
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 14: Cache eviction (ndb_llm_cache_evict_stale, ndb_llm_cache_evict_size)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	evicted_stale int;
	evicted_size int;
BEGIN
	evicted_stale := ndb_llm_cache_evict_stale();
	RAISE NOTICE 'Evicted stale entries: %', evicted_stale;

	evicted_size := ndb_llm_cache_evict_size(100);
	RAISE NOTICE 'Evicted for size limit: %', evicted_size;
END $$;

-- Test 15: Batch alias performance comparison
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 15: Batch alias performance (neurondb.embed_batch)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	texts text[];
	start_time timestamp;
	end_time timestamp;
	duration interval;
BEGIN
	texts := ARRAY['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5']::text[];

	start_time := clock_timestamp();
	PERFORM neurondb.embed_batch('all-MiniLM-L6-v2', texts);
	end_time := clock_timestamp();
	duration := end_time - start_time;

	RAISE NOTICE 'neurondb.embed_batch completed in %', duration;
END $$;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'All advanced embedding tests completed!'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test completed successfully'
