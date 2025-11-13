\timing on
\pset footer off

SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'hf_embed';
SELECT neurondb_gpu_enable();

\set ON_ERROR_STOP on

-- Test Hugging Face embedding with GPU acceleration
\echo 'Testing Hugging Face embedding with GPU acceleration...'

-- Create test table
DROP TABLE IF EXISTS hf_embed_test;
CREATE TEMP TABLE hf_embed_test (
	id SERIAL PRIMARY KEY,
	text_content TEXT NOT NULL,
	embedding VECTOR(384)
);

-- Insert sample texts
INSERT INTO hf_embed_test (text_content) VALUES
	('PostgreSQL is a powerful open source database'),
	('Machine learning enables computers to learn from data'),
	('Vector databases store high-dimensional embeddings'),
	('GPU acceleration speeds up neural network inference'),
	('Hugging Face provides thousands of pretrained models');

\echo 'Sample texts inserted'

-- Generate embeddings using GPU-accelerated embedding
\timing on
UPDATE hf_embed_test
SET embedding = embed_text(text_content, 'all-MiniLM-L6-v2');
\timing off

\echo 'Embeddings generated! Verifying...'

-- Verify embeddings were created
SELECT 
	id,
	LEFT(text_content, 40) AS text_preview,
	vector_dims(embedding) AS embedding_dim,
	ROUND((vector_to_array(embedding))[1]::numeric, 4) AS first_dim
FROM hf_embed_test
WHERE embedding IS NOT NULL
ORDER BY id
LIMIT 5;

-- Test semantic similarity search
\echo ''
\echo 'Testing semantic similarity search...'

WITH query_embedding AS (
	SELECT embed_text('database systems', 'all-MiniLM-L6-v2') AS emb
)
SELECT 
	id,
	LEFT(text_content, 50) AS text_preview,
	ROUND((1 - (embedding <=> qe.emb))::numeric, 4) AS similarity
FROM hf_embed_test, query_embedding qe
WHERE embedding IS NOT NULL
ORDER BY embedding <=> qe.emb
LIMIT 3;

-- Cleanup
DROP TABLE IF EXISTS hf_embed_test;

