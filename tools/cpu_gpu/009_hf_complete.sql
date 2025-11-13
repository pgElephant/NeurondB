\timing on
\pset footer off

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'hf_complete';
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

-- Test Hugging Face text completion with GPU acceleration
\echo 'Testing Hugging Face text completion with GPU acceleration...'

-- Test simple completion
\echo ''
\echo 'Test 1: Simple text completion'
SELECT 
	'Simple completion' AS test_name,
	ndb_llm_complete(
		'What is PostgreSQL?',
		'{"max_tokens": 50, "temperature": 0.7}'::text
	) AS completion;

-- Test batch completion
\echo ''
\echo 'Test 2: Batch text completion'
SELECT 
	prompt,
	completion
FROM ndb_llm_complete_batch(
	ARRAY[
		'Explain machine learning in one sentence.',
		'What is a vector database?',
		'How does GPU acceleration work?'
	],
	'{"max_tokens": 30, "temperature": 0.7}'::text
) AS t(prompt text, completion text);

-- Test with different parameters
\echo ''
\echo 'Test 3: Completion with custom parameters'
SELECT 
	'Custom params' AS test_name,
	ndb_llm_complete(
		'Write a short summary about neural networks.',
		'{"max_tokens": 100, "temperature": 0.5, "top_p": 0.9}'::text
	) AS completion;

-- Verify GPU usage (if available)
\echo ''
\echo 'GPU availability check:'
SELECT 
	neurondb_llm_gpu_available() AS gpu_available,
	backend_type,
	device_name
FROM neurondb_llm_gpu_info();

