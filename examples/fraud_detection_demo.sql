-- ====================================================================
-- NeurondB Fraud Detection Demo
-- ====================================================================
-- This demo shows how to use NeurondB's ML training API for fraud detection
-- Similar to pgml but with NeurondB's own implementation
-- ====================================================================

-- Enable extension
CREATE EXTENSION IF NOT EXISTS neurondb CASCADE;

-- Create project for fraud detection
SELECT neurondb.create_project(
    'fraud_detection',
    'Binary classification for transaction fraud detection',
    'classification'
);

-- Data table
DROP TABLE IF EXISTS public.transactions CASCADE;
CREATE TABLE public.transactions (
  id           bigserial PRIMARY KEY,
  ts           timestamptz DEFAULT now(),
  amount       numeric(10,2),
  mcc          int,
  device_risk  float4,
  is_fraud     int
);

-- Helper function for sigmoid
CREATE OR REPLACE FUNCTION sigmoid(x double precision)
RETURNS double precision LANGUAGE sql IMMUTABLE AS $$
  SELECT 1.0 / (1.0 + exp(-x));
$$;

-- Synthetic fraud data
INSERT INTO public.transactions(amount, mcc, device_risk, is_fraud)
SELECT
  round(10 + 990*random(), 2) AS amount,
  4000 + (random()*800)::int  AS mcc,
  random()::float4            AS device_risk,
  CASE
    WHEN 0.6*sigmoid( amount/1000.0 )
       + 0.3*(device_risk)
       + 0.1*(CASE WHEN mcc BETWEEN 4810 AND 4899 THEN 1 ELSE 0 END)
       + 0.05*random() > 0.55
    THEN 1 ELSE 0
  END                         AS is_fraud
FROM generate_series(1, 20000);

-- Train/test split
ALTER TABLE public.transactions ADD COLUMN IF NOT EXISTS split text;
UPDATE public.transactions
SET split = CASE WHEN id % 5 = 0 THEN 'test' ELSE 'train' END;

-- Views for clean train/test relations
DROP VIEW IF EXISTS train_data CASCADE;
DROP VIEW IF EXISTS test_data CASCADE;

CREATE VIEW train_data AS
  SELECT id, amount::float8 AS amount, mcc::float8 AS mcc,
         device_risk::float8 AS device_risk, is_fraud
  FROM public.transactions WHERE split = 'train';

CREATE VIEW test_data AS
  SELECT id, amount::float8 AS amount, mcc::float8 AS mcc,
         device_risk::float8 AS device_risk, is_fraud
  FROM public.transactions WHERE split = 'test';

-- Create experiment for baseline model
SELECT neurondb.create_experiment(
    'fraud_detection',
    'baseline_lr',
    'Baseline logistic regression model',
    '{"algorithm":"logistic_regression","max_iterations":100,"learning_rate":0.01}'::jsonb
);

-- Train baseline model
SELECT neurondb.train_model(
    project_name    => 'fraud_detection',
    experiment_name => 'baseline_lr',
    relation_name   => 'train_data',
    target_column   => 'is_fraud',
    feature_columns => ARRAY['amount','mcc','device_risk'],
    algorithm       => 'logistic_regression',
    hyperparameters => '{"max_iterations":100,"learning_rate":0.01}'::jsonb
);

-- List all models
SELECT * FROM neurondb.list_models('fraud_detection');

-- Get model info
WITH latest_model AS (
    SELECT model_id 
    FROM neurondb.ml_trained_models 
    WHERE project_id = (SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'fraud_detection')
    ORDER BY created_at DESC 
    LIMIT 1
)
SELECT neurondb.model_info(model_id::text) 
FROM latest_model;

-- Set default model for online predictions
WITH latest_model AS (
    SELECT model_id 
    FROM neurondb.ml_trained_models 
    WHERE project_id = (SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'fraud_detection')
    ORDER BY created_at DESC 
    LIMIT 1
)
SELECT neurondb.set_default_model('fraud_detection', model_id::text)
FROM latest_model;

-- Online prediction example (single transaction)
SELECT neurondb.predict(
    'fraud_detection',
    '{"amount":799.00,"mcc":4820,"device_risk":0.9}'::jsonb
) AS predicted_fraud_score;

-- Get probability scores for a transaction
SELECT neurondb.predict_proba(
    'fraud_detection',
    '{"amount":799.00,"mcc":4820,"device_risk":0.9}'::jsonb
) AS fraud_probabilities;

-- Batch scoring on test set
DROP TABLE IF EXISTS public.test_scores CASCADE;
CREATE TABLE public.test_scores(
  id bigint PRIMARY KEY,
  score double precision,
  label int
);

-- Batch predict (more efficient than row-by-row)
WITH test_features AS (
    SELECT 
        id,
        jsonb_build_object(
            'amount', amount,
            'mcc', mcc,
            'device_risk', device_risk
        ) as features,
        is_fraud as label
    FROM test_data
)
INSERT INTO public.test_scores(id, score, label)
SELECT 
    id,
    (neurondb.predict('fraud_detection', features))::double precision,
    label
FROM test_features;

-- Evaluate model performance
SELECT neurondb.model_performance(
    'test_scores',
    'label',
    'score'
);

-- Simple accuracy metrics at 0.5 threshold
WITH preds AS (
  SELECT label, CASE WHEN score >= 0.5 THEN 1 ELSE 0 END AS yhat 
  FROM public.test_scores
)
SELECT
  round(avg((label = yhat)::int), 4) AS accuracy,
  round(avg(CASE WHEN label = 1 THEN score ELSE NULL END), 4) AS mean_pos_score,
  round(avg(CASE WHEN label = 0 THEN score ELSE NULL END), 4) AS mean_neg_score,
  count(*) AS n
FROM preds;

-- Confusion matrix at 0.5 threshold
WITH preds AS (
  SELECT label, CASE WHEN score >= 0.5 THEN 1 ELSE 0 END AS yhat 
  FROM public.test_scores
)
SELECT
  sum(CASE WHEN label = 1 AND yhat = 1 THEN 1 ELSE 0 END) AS true_positives,
  sum(CASE WHEN label = 0 AND yhat = 0 THEN 1 ELSE 0 END) AS true_negatives,
  sum(CASE WHEN label = 0 AND yhat = 1 THEN 1 ELSE 0 END) AS false_positives,
  sum(CASE WHEN label = 1 AND yhat = 0 THEN 1 ELSE 0 END) AS false_negatives
FROM preds;

-- Precision, Recall, F1
WITH preds AS (
  SELECT label, CASE WHEN score >= 0.5 THEN 1 ELSE 0 END AS yhat 
  FROM public.test_scores
),
cm AS (
  SELECT
    sum(CASE WHEN label = 1 AND yhat = 1 THEN 1 ELSE 0 END)::float AS tp,
    sum(CASE WHEN label = 0 AND yhat = 0 THEN 1 ELSE 0 END)::float AS tn,
    sum(CASE WHEN label = 0 AND yhat = 1 THEN 1 ELSE 0 END)::float AS fp,
    sum(CASE WHEN label = 1 AND yhat = 0 THEN 1 ELSE 0 END)::float AS fn
  FROM preds
)
SELECT
  round(tp / NULLIF(tp + fp, 0), 4) AS precision,
  round(tp / NULLIF(tp + fn, 0), 4) AS recall,
  round(2 * (tp / NULLIF(tp + fp, 0)) * (tp / NULLIF(tp + fn, 0)) / 
        NULLIF((tp / NULLIF(tp + fp, 0)) + (tp / NULLIF(tp + fn, 0)), 0), 4) AS f1_score
FROM cm;

-- Explain a specific prediction (feature importance)
SELECT neurondb.explain_prediction(
    'fraud_detection',
    '{"amount":499.50,"mcc":4822,"device_risk":0.3}'::jsonb
) AS feature_importance;

-- View model version history
SELECT 
    model_id,
    experiment_id,
    algorithm,
    hyperparameters,
    metrics,
    created_at,
    CASE WHEN is_default THEN '✓ DEFAULT' ELSE '' END as status
FROM neurondb.ml_trained_models m
JOIN neurondb.ml_projects p ON m.project_id = p.project_id
WHERE p.project_name = 'fraud_detection'
ORDER BY created_at DESC;

-- View recent predictions (monitoring)
SELECT * FROM neurondb.recent_predictions 
WHERE model_id IN (
    SELECT model_id FROM neurondb.ml_trained_models m
    JOIN neurondb.ml_projects p ON m.project_id = p.project_id
    WHERE p.project_name = 'fraud_detection'
)
LIMIT 10;

-- ====================================================================
-- Summary Stats
-- ====================================================================
SELECT '
╔═════════════════════════════════════════════════════════════╗
║            NeurondB Fraud Detection Demo                   ║
╠═════════════════════════════════════════════════════════════╣
║  ✓ Created fraud_detection project                         ║
║  ✓ Generated 20,000 synthetic transactions                 ║
║  ✓ Trained logistic regression model                       ║
║  ✓ Performed batch predictions on test set                 ║
║  ✓ Calculated accuracy, precision, recall, F1               ║
║  ✓ Explained individual predictions                        ║
╚═════════════════════════════════════════════════════════════╝
' as "Demo Complete";

