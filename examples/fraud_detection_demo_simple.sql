-- ====================================================================
-- NeurondB Simple Fraud Detection Demo
-- ====================================================================
-- Uses NeurondB's implemented features: vectors, clustering, distances
-- ====================================================================

-- Enable extension
CREATE EXTENSION IF NOT EXISTS neurondb CASCADE;

-- Data table with vector representation
DROP TABLE IF EXISTS public.transactions CASCADE;
CREATE TABLE public.transactions (
  id           bigserial PRIMARY KEY,
  ts           timestamptz DEFAULT now(),
  amount       numeric(10,2),
  mcc          int,
  device_risk  float4,
  is_fraud     int,
  features     vector(3)  -- [amount_normalized, mcc_normalized, device_risk]
);

-- Helper function for sigmoid
CREATE OR REPLACE FUNCTION sigmoid(x double precision)
RETURNS double precision LANGUAGE sql IMMUTABLE AS $$
  SELECT 1.0 / (1.0 + exp(-x));
$$;

-- Synthetic fraud data
INSERT INTO public.transactions(amount, mcc, device_risk, is_fraud)
SELECT
  (10 + 990*random())::numeric(10,2) AS amount,
  4000 + (random()*800)::int  AS mcc,
  random()::float4            AS device_risk,
  CASE
    WHEN 0.6*sigmoid( (10 + 990*random())/1000.0 )
       + 0.3*(random())
       + 0.1*(CASE WHEN 4000 + (random()*800)::int BETWEEN 4810 AND 4899 THEN 1 ELSE 0 END)
       + 0.05*random() > 0.55
    THEN 1 ELSE 0
  END AS is_fraud
FROM generate_series(1, 10000);

-- Normalize and create feature vectors
UPDATE public.transactions
SET features = ('[' || 
    (amount::float / 1000.0) || ',' ||
    ((mcc - 4000)::float / 800.0) || ',' ||
    device_risk || 
']')::vector(3);

-- Show sample data
SELECT '=== Sample Transactions ===' as step;
SELECT id, amount, mcc, device_risk, is_fraud, features 
FROM public.transactions 
LIMIT 5;

-- Use K-means clustering to find transaction patterns
SELECT '=== Clustering Transactions (K=5) ===' as step;
SELECT cluster, count(*) as count
FROM cluster_kmeans('public.transactions', 'features', 5, 100)
GROUP BY cluster
ORDER BY cluster;

-- Calculate distances between transactions
SELECT '=== Distance Analysis ===' as step;
WITH sample_fraud AS (
    SELECT features FROM public.transactions WHERE is_fraud = 1 LIMIT 1
),
sample_normal AS (
    SELECT features FROM public.transactions WHERE is_fraud = 0 LIMIT 1
)
SELECT 
    'L2 Distance' as metric,
    (sf.features <-> sn.features) as distance
FROM sample_fraud sf, sample_normal sn
UNION ALL
SELECT 
    'Cosine Distance' as metric,
    (sf.features <=> sn.features) as distance
FROM sample_fraud sf, sample_normal sn
UNION ALL
SELECT 
    'Inner Product' as metric,
    (sf.features <#> sn.features) as distance
FROM sample_fraud sf, sample_normal sn;

-- Find similar fraud cases
SELECT '=== Finding Similar Fraud Cases ===' as step;
WITH target_fraud AS (
    SELECT features FROM public.transactions WHERE is_fraud = 1 LIMIT 1
)
SELECT 
    t.id,
    t.amount,
    t.mcc,
    t.device_risk,
    t.is_fraud,
    (t.features <-> tf.features) as distance
FROM public.transactions t, target_fraud tf
WHERE t.is_fraud = 1
ORDER BY distance
LIMIT 10;

-- Aggregate statistics
SELECT '=== Transaction Statistics ===' as step;
SELECT 
    'Fraud' as type,
    count(*) as count,
    avg(amount)::numeric(10,2) as avg_amount,
    avg(device_risk)::numeric(4,3) as avg_risk
FROM public.transactions WHERE is_fraud = 1
UNION ALL
SELECT 
    'Normal' as type,
    count(*) as count,
    avg(amount)::numeric(10,2) as avg_amount,
    avg(device_risk)::numeric(4,3) as avg_risk
FROM public.transactions WHERE is_fraud = 0;

-- Summary
SELECT '
╔═════════════════════════════════════════════════════════════╗
║         NeurondB Fraud Detection Demo (Simple)             ║
╠═════════════════════════════════════════════════════════════╣
║  ✓ Created 10,000 synthetic transactions                   ║
║  ✓ Generated 3D feature vectors                            ║
║  ✓ Clustered transactions using K-means                    ║
║  ✓ Calculated vector distances                             ║
║  ✓ Found similar fraud patterns                            ║
║  ✓ All using NeurondB vector operations                    ║
╚═════════════════════════════════════════════════════════════╝
' as "Demo Complete - PostgreSQL 17 + NeurondB Working!";

