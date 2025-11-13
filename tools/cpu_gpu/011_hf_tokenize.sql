\timing on
\pset footer off

SET neurondb.gpu_enabled = on;
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

-- Test Hugging Face tokenization features
\echo 'Testing Hugging Face tokenization with ONNX Runtime...'

-- Test 1: Basic tokenization
\echo ''
\echo 'Test 1: Tokenize text to token IDs'
SELECT 
	'Tokenization test' AS test_name,
	neurondb_hf_tokenize('all-MiniLM-L6-v2', 'Hello world from PostgreSQL', 128) AS token_ids;

-- Test 2: Detokenization
\echo ''
\echo 'Test 2: Detokenize token IDs back to text'
WITH tokens AS (
	SELECT neurondb_hf_tokenize('all-MiniLM-L6-v2', 'Machine learning is powerful', 128) AS token_ids
)
SELECT 
	'Detokenization test' AS test_name,
	neurondb_hf_detokenize('all-MiniLM-L6-v2', token_ids) AS reconstructed_text
FROM tokens;

-- Test 3: Round-trip tokenization/detokenization
\echo ''
\echo 'Test 3: Round-trip tokenization and detokenization'
WITH original_text AS (
	SELECT 'NeurondB provides GPU-accelerated ML operations' AS text
),
tokenized AS (
	SELECT 
		text,
		neurondb_hf_tokenize('all-MiniLM-L6-v2', text, 128) AS token_ids
	FROM original_text
)
SELECT 
	text AS original,
	neurondb_hf_detokenize('all-MiniLM-L6-v2', token_ids) AS reconstructed,
	CASE 
		WHEN text = neurondb_hf_detokenize('all-MiniLM-L6-v2', token_ids) 
		THEN 'Match' 
		ELSE 'Different' 
	END AS match_status
FROM tokenized;

-- Test 4: Token count
\echo ''
\echo 'Test 4: Count tokens in text'
SELECT 
	'Token count test' AS test_name,
	array_length(neurondb_hf_tokenize('all-MiniLM-L6-v2', 'This is a test sentence for token counting', 128), 1) AS token_count;

-- Test 5: Multiple texts tokenization
\echo ''
\echo 'Test 5: Tokenize multiple texts'
SELECT 
	id,
	LEFT(text_content, 40) AS text_preview,
	array_length(token_ids, 1) AS token_count,
	token_ids[1:5] AS first_5_tokens
FROM (
	VALUES 
		(1, 'PostgreSQL database management'),
		(2, 'Machine learning algorithms'),
		(3, 'Vector similarity search')
) AS texts(id, text_content),
	LATERAL (
		SELECT neurondb_hf_tokenize('all-MiniLM-L6-v2', text_content, 128) AS token_ids
	) AS tokens
ORDER BY id;

