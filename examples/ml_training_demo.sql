-- ============================================================================
-- NeurondB ML Training & Inference Demo
-- ============================================================================
-- Complete example of in-database machine learning with NeurondB
-- Demonstrates project management, model training, prediction, and evaluation
-- ============================================================================

\echo '=== NeurondB ML Training Demo ==='
\echo 'This demo shows how to train and deploy ML models directly in PostgreSQL'
\echo ''

-- Enable NeurondB extension
CREATE EXTENSION IF NOT EXISTS neurondb CASCADE;

-- Verify installation
SELECT neurondb.version() AS neurondb_version;

\echo ''
\echo '=== Step 1: Create ML Project ==='

-- Create a new ML project for fraud detection
SELECT neurondb.create_project(
    project_name := 'fraud_detection',
    task_type := 'classification',
    description := 'Real-time fraud detection for transactions'
) AS project_id;

-- Create an experiment to track this approach
SELECT neurondb.create_experiment(
    project_name := 'fraud_detection',
    experiment_name := 'baseline_xgboost',
    description := 'Baseline XGBoost model with default hyperparameters'
) AS experiment_id;

\echo ''
\echo '=== Step 2: Prepare Training Data ==='

-- Create transactions table
DROP TABLE IF EXISTS public.transactions CASCADE;
CREATE TABLE public.transactions (
    id           BIGSERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ DEFAULT NOW(),
    amount       NUMERIC(10,2),
    mcc          INT,  -- Merchant Category Code
    device_risk  FLOAT,
    location_distance FLOAT,  -- Distance from usual location (km)
    hour_of_day  INT,
    is_fraud     INT  -- 0 = legitimate, 1 = fraud
);

COMMENT ON TABLE public.transactions IS 'Transaction history for fraud detection training';

\echo 'Generating 20,000 synthetic transactions...'

-- Helper function for sigmoid
CREATE OR REPLACE FUNCTION sigmoid(x DOUBLE PRECISION)
RETURNS DOUBLE PRECISION
LANGUAGE SQL IMMUTABLE AS $$
    SELECT 1.0 / (1.0 + EXP(-x));
$$;

-- Generate realistic synthetic training data
INSERT INTO public.transactions(amount, mcc, device_risk, location_distance, hour_of_day, is_fraud)
SELECT
    -- Amount: mostly $10-$1000, some outliers
    ROUND(10 + 990 * RANDOM(), 2)::NUMERIC(10,2) AS amount,
    
    -- MCC: clustered around common categories
    4000 + (RANDOM() * 800)::INT AS mcc,
    
    -- Device risk score
    RANDOM()::FLOAT AS device_risk,
    
    -- Location distance (most < 50 km, some > 100 km for fraud)
    CASE 
        WHEN RANDOM() < 0.9 THEN (RANDOM() * 50)::FLOAT
        ELSE (100 + RANDOM() * 500)::FLOAT
    END AS location_distance,
    
    -- Hour of day (0-23)
    (RANDOM() * 24)::INT AS hour_of_day,
    
    -- Fraud label (based on multiple factors)
    CASE
        WHEN 0.4 * sigmoid(amount / 500.0)  -- High amounts more risky
           + 0.3 * device_risk              -- High device risk
           + 0.2 * (CASE WHEN mcc BETWEEN 4810 AND 4899 THEN 1 ELSE 0 END)  -- Telecom MCCs
           + 0.1 * (location_distance / 100.0)  -- Distance from usual location
           + 0.05 * RANDOM()  -- Some noise
           > 0.55
        THEN 1
        ELSE 0
    END AS is_fraud
FROM generate_series(1, 20000);

-- Show data distribution
\echo ''
\echo 'Dataset statistics:'
SELECT 
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate_pct,
    ROUND(AVG(amount)::NUMERIC, 2) AS avg_amount,
    ROUND(AVG(device_risk)::NUMERIC, 3) AS avg_device_risk
FROM public.transactions;

\echo ''
\echo '=== Step 3: Split Data (Train/Test) ==='

-- Add split column
ALTER TABLE public.transactions ADD COLUMN IF NOT EXISTS split TEXT;

-- 80/20 train/test split
UPDATE public.transactions
SET split = CASE WHEN id % 5 = 0 THEN 'test' ELSE 'train' END;

-- Create views for clean access
CREATE OR REPLACE VIEW train_data AS
SELECT 
    id,
    amount::FLOAT8 AS amount,
    mcc::FLOAT8 AS mcc,
    device_risk::FLOAT8 AS device_risk,
    location_distance::FLOAT8 AS location_distance,
    hour_of_day::FLOAT8 AS hour_of_day,
    is_fraud
FROM public.transactions
WHERE split = 'train';

CREATE OR REPLACE VIEW test_data AS
SELECT 
    id,
    amount::FLOAT8 AS amount,
    mcc::FLOAT8 AS mcc,
    device_risk::FLOAT8 AS device_risk,
    location_distance::FLOAT8 AS location_distance,
    hour_of_day::FLOAT8 AS hour_of_day,
    is_fraud
FROM public.transactions
WHERE split = 'test';

-- Show split statistics
SELECT 
    split,
    COUNT(*) AS count,
    SUM(is_fraud) AS fraud_count
FROM public.transactions
GROUP BY split
ORDER BY split;

\echo ''
\echo '=== Step 4: Train ML Model ==='
\echo 'Training logistic regression model (this may take 30-60 seconds)...'

-- Train the model using NeurondB's training API
SELECT neurondb.train_model(
    model_name := 'fraud_detector_v1',
    algorithm := 'logistic_regression',
    training_table := 'train_data',
    feature_columns := ARRAY['amount', 'mcc', 'device_risk', 'location_distance', 'hour_of_day'],
    target_column := 'is_fraud',
    validation_split := 0.2,
    hyperparameters := '{
        "epochs": 100,
        "learning_rate": 0.1
    }'::JSONB,
    random_state := 42
) AS training_result;

\echo ''
\echo 'Model trained successfully!'

\echo ''
\echo '=== Step 5: List Trained Models ==='

-- View all models in the project
SELECT * FROM neurondb.list_models();

\echo ''
\echo '=== Step 6: Single Prediction Example ==='

-- Make a single prediction
SELECT neurondb.predict(
    model_name := 'fraud_detector_v1',
    features := ARRAY[799.00, 4820.0, 0.9, 150.0, 14.0]  -- High-risk transaction
) AS fraud_score;

-- Compare with a normal transaction
SELECT neurondb.predict(
    model_name := 'fraud_detector_v1',
    features := ARRAY[45.00, 5411.0, 0.1, 2.5, 10.0]  -- Low-risk transaction
) AS fraud_score_normal;

\echo ''
\echo '=== Step 7: Probability Predictions ==='

-- Get probability for each class
SELECT neurondb.predict_proba(
    model_name := 'fraud_detector_v1',
    features := ARRAY[799.00, 4820.0, 0.9, 150.0, 14.0]
) AS probabilities;

\echo 'Returns: [P(legitimate), P(fraud)]'

\echo ''
\echo '=== Step 8: Batch Scoring on Test Set ==='

-- Create table to store predictions
DROP TABLE IF EXISTS public.test_scores CASCADE;
CREATE TABLE public.test_scores (
    id BIGINT PRIMARY KEY,
    fraud_score FLOAT,
    actual_label INT,
    predicted_label INT,
    scored_at TIMESTAMPTZ DEFAULT NOW()
);

\echo 'Scoring test set (this may take 30-60 seconds)...'

-- Score all test transactions
INSERT INTO public.test_scores(id, fraud_score, actual_label)
SELECT 
    t.id,
    neurondb.predict(
        model_name := 'fraud_detector_v1',
        features := ARRAY[t.amount, t.mcc, t.device_risk, t.location_distance, t.hour_of_day]
    ) AS fraud_score,
    t.is_fraud AS actual_label
FROM test_data t;

-- Add predicted labels (threshold = 0.5)
UPDATE public.test_scores
SET predicted_label = CASE WHEN fraud_score >= 0.5 THEN 1 ELSE 0 END;

\echo 'Test set scored!'

\echo ''
\echo '=== Step 9: Evaluate Model Performance ==='

-- Calculate accuracy
WITH metrics AS (
    SELECT 
        actual_label,
        predicted_label,
        fraud_score
    FROM public.test_scores
)
SELECT 
    'Accuracy' AS metric,
    ROUND(AVG((actual_label = predicted_label)::INT)::NUMERIC, 4) AS value
FROM metrics
UNION ALL
-- Precision (TP / (TP + FP))
SELECT 
    'Precision' AS metric,
    ROUND(
        SUM(CASE WHEN actual_label = 1 AND predicted_label = 1 THEN 1 ELSE 0 END)::NUMERIC /
        NULLIF(SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END), 0),
        4
    ) AS value
FROM metrics
UNION ALL
-- Recall (TP / (TP + FN))
SELECT 
    'Recall' AS metric,
    ROUND(
        SUM(CASE WHEN actual_label = 1 AND predicted_label = 1 THEN 1 ELSE 0 END)::NUMERIC /
        NULLIF(SUM(CASE WHEN actual_label = 1 THEN 1 ELSE 0 END), 0),
        4
    ) AS value
FROM metrics
UNION ALL
-- F1 Score
SELECT 
    'F1 Score' AS metric,
    ROUND(
        2.0 * 
        (SUM(CASE WHEN actual_label = 1 AND predicted_label = 1 THEN 1 ELSE 0 END)::NUMERIC /
         NULLIF(SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END), 0)) *
        (SUM(CASE WHEN actual_label = 1 AND predicted_label = 1 THEN 1 ELSE 0 END)::NUMERIC /
         NULLIF(SUM(CASE WHEN actual_label = 1 THEN 1 ELSE 0 END), 0)) /
        NULLIF(
            (SUM(CASE WHEN actual_label = 1 AND predicted_label = 1 THEN 1 ELSE 0 END)::NUMERIC /
             NULLIF(SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END), 0)) +
            (SUM(CASE WHEN actual_label = 1 AND predicted_label = 1 THEN 1 ELSE 0 END)::NUMERIC /
             NULLIF(SUM(CASE WHEN actual_label = 1 THEN 1 ELSE 0 END), 0)),
            0
        ),
        4
    ) AS value
FROM metrics;

\echo ''
\echo '=== Confusion Matrix ==='

-- Detailed confusion matrix
WITH preds AS (
    SELECT actual_label, predicted_label FROM public.test_scores
)
SELECT 
    SUM(CASE WHEN actual_label = 1 AND predicted_label = 1 THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN actual_label = 0 AND predicted_label = 0 THEN 1 ELSE 0 END) AS true_negatives,
    SUM(CASE WHEN actual_label = 0 AND predicted_label = 1 THEN 1 ELSE 0 END) AS false_positives,
    SUM(CASE WHEN actual_label = 1 AND predicted_label = 0 THEN 1 ELSE 0 END) AS false_negatives
FROM preds;

\echo ''
\echo '=== Step 10: Score Distribution Analysis ==='

-- Score distribution by actual label
SELECT 
    actual_label,
    ROUND(MIN(fraud_score)::NUMERIC, 3) AS min_score,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY fraud_score)::NUMERIC, 3) AS q25,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY fraud_score)::NUMERIC, 3) AS median,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY fraud_score)::NUMERIC, 3) AS q75,
    ROUND(MAX(fraud_score)::NUMERIC, 3) AS max_score,
    COUNT(*) AS count
FROM public.test_scores
GROUP BY actual_label
ORDER BY actual_label;

\echo ''
\echo '=== Step 11: High-Risk Transactions (Top 10) ==='

-- Find highest risk transactions
SELECT 
    id,
    ROUND(fraud_score::NUMERIC, 4) AS risk_score,
    actual_label AS is_actually_fraud,
    CASE WHEN actual_label = 1 THEN 'Correctly Flagged' ELSE 'False Alarm' END AS result
FROM public.test_scores
WHERE predicted_label = 1
ORDER BY fraud_score DESC
LIMIT 10;

\echo ''
\echo '=== Step 12: Model Information ==='

-- Get detailed model info
SELECT neurondb.model_info('fraud_detector_v1') AS model_metadata;

\echo ''
\echo '=== Step 13: List All Projects ==='

-- View all ML projects
SELECT * FROM neurondb.list_projects();

\echo ''
\echo '=== Step 14: A/B Testing Setup ==='

-- Train a second model with different hyperparameters
SELECT neurondb.train_model(
    model_name := 'fraud_detector_v2',
    algorithm := 'logistic_regression',
    training_table := 'train_data',
    feature_columns := ARRAY['amount', 'mcc', 'device_risk', 'location_distance', 'hour_of_day'],
    target_column := 'is_fraud',
    validation_split := 0.2,
    hyperparameters := '{
        "epochs": 200,
        "learning_rate": 0.05
    }'::JSONB,
    random_state := 42
) AS training_result_v2;

-- Set v1 as default
SELECT neurondb.set_default_model('fraud_detection', 1) AS default_set;

\echo ''
\echo '=== Step 15: Real-time Prediction Examples ==='

-- Example 1: High-risk transaction (large amount, risky device, unusual location)
SELECT 
    'High Risk Example' AS scenario,
    neurondb.predict('fraud_detector_v1', ARRAY[1250.00, 4816.0, 0.95, 250.0, 3.0]) AS fraud_score,
    neurondb.predict_proba('fraud_detector_v1', ARRAY[1250.00, 4816.0, 0.95, 250.0, 3.0]) AS probabilities;

-- Example 2: Normal transaction (small amount, low risk, local)
SELECT 
    'Normal Transaction' AS scenario,
    neurondb.predict('fraud_detector_v1', ARRAY[45.50, 5411.0, 0.05, 2.0, 14.0]) AS fraud_score,
    neurondb.predict_proba('fraud_detector_v1', ARRAY[45.50, 5411.0, 0.05, 2.0, 14.0]) AS probabilities;

-- Example 3: Borderline case
SELECT 
    'Borderline Case' AS scenario,
    neurondb.predict('fraud_detector_v1', ARRAY[299.99, 5999.0, 0.50, 25.0, 12.0]) AS fraud_score,
    neurondb.predict_proba('fraud_detector_v1', ARRAY[299.99, 5999.0, 0.50, 25.0, 12.0]) AS probabilities;

\echo ''
\echo '=== Step 16: Threshold Optimization ==='

-- Find optimal threshold by testing different values
WITH thresholds AS (
    SELECT generate_series(0.3, 0.7, 0.05) AS threshold
),
metrics_at_threshold AS (
    SELECT 
        t.threshold,
        -- Precision
        SUM(CASE WHEN s.actual_label = 1 AND s.fraud_score >= t.threshold THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN s.fraud_score >= t.threshold THEN 1 ELSE 0 END), 0) AS precision,
        -- Recall
        SUM(CASE WHEN s.actual_label = 1 AND s.fraud_score >= t.threshold THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN s.actual_label = 1 THEN 1 ELSE 0 END), 0) AS recall
    FROM public.test_scores s, thresholds t
    GROUP BY t.threshold
)
SELECT 
    ROUND(threshold::NUMERIC, 2) AS threshold,
    ROUND(precision::NUMERIC, 4) AS precision,
    ROUND(recall::NUMERIC, 4) AS recall,
    ROUND((2 * precision * recall / NULLIF(precision + recall, 0))::NUMERIC, 4) AS f1_score
FROM metrics_at_threshold
ORDER BY f1_score DESC NULLS LAST;

\echo ''
\echo '=== Step 17: Feature Importance (Mock) ==='

-- Note: This returns placeholder data - will be implemented with SHAP
SELECT neurondb.explain_prediction(
    project_name := 'fraud_detection',
    input_features := '{
        "amount": 799.00,
        "mcc": 4820,
        "device_risk": 0.9,
        "location_distance": 150.0,
        "hour_of_day": 3
    }'::JSONB
) AS feature_importance;

\echo ''
\echo '=== Step 18: Model Versioning ==='

-- View all models
SELECT 
    model_id,
    algorithm,
    status,
    training_samples,
    created_at
FROM neurondb.ml_trained_models
WHERE project_id = (SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'fraud_detection')
ORDER BY created_at DESC;

\echo ''
\echo '=== Step 19: Cleanup (Optional) ==='
\echo 'To remove demo data, uncomment and run:'
\echo '-- DROP TABLE public.transactions CASCADE;'
\echo '-- DROP TABLE public.test_scores CASCADE;'
\echo '-- SELECT neurondb.delete_model(''fraud_detector_v1'');'
\echo '-- SELECT neurondb.delete_model(''fraud_detector_v2'');'

\echo ''
\echo '========================================'
\echo '✅ Demo Complete!'
\echo '========================================'
\echo ''
\echo 'You have successfully:'
\echo '  1. Created an ML project'
\echo '  2. Generated synthetic training data'
\echo '  3. Trained a fraud detection model'
\echo '  4. Made predictions (single and batch)'
\echo '  5. Evaluated model performance'
\echo '  6. Analyzed results'
\echo ''
\echo 'Next steps:'
\echo '  - Try different algorithms (when implemented)'
\echo '  - Tune hyperparameters with neurondb.train_with_search()'
\echo '  - Deploy model in production triggers'
\echo '  - Monitor with neurondb.model_performance()'
\echo ''

