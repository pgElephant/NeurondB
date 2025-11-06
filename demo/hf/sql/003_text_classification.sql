-- ============================================================================
-- Test 003: Text Classification with HuggingFace Models
-- ============================================================================
-- Demonstrates: Sentiment analysis, topic classification via ONNX
-- ============================================================================

\echo 'Testing HuggingFace text classification...'

CREATE TABLE IF NOT EXISTS hf_classification_test (
    text_id SERIAL PRIMARY KEY,
    text_content TEXT NOT NULL,
    classification_result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample texts for sentiment analysis
INSERT INTO hf_classification_test (text_content) VALUES
    ('This product is amazing! I absolutely love it!'),
    ('Terrible experience, would not recommend'),
    ('It is okay, nothing special'),
    ('Best purchase I have ever made!'),
    ('Waste of money and time');

\echo ''
\echo 'Classifying texts using distilbert-sentiment model...'

UPDATE hf_classification_test
SET classification_result = neurondb_hf_classify('distilbert-sentiment', text_content)::jsonb;

SELECT 
    text_id,
    text_content,
    classification_result->>'label' AS predicted_class,
    (classification_result->>'score')::float AS confidence
FROM hf_classification_test
ORDER BY text_id;

\echo ''
\echo 'Text classification test complete!'


