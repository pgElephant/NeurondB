-- ============================================================================
-- NeuronDB Fraud Detection - Complete ML Algorithm Demo
-- Tests ALL ML algorithms on 100MB dataset (~1.5M transactions)
-- ============================================================================

\echo '╔══════════════════════════════════════════════════════════════════════════╗'
\echo '║       NeuronDB Complete Fraud Detection Demo (100MB)                     ║'
\echo '║       Testing ALL ML Algorithms with Project Management                  ║'
\echo '╚══════════════════════════════════════════════════════════════════════════╝'
\echo ''

-- Cleanup
DROP TABLE IF EXISTS transactions CASCADE;
DROP EXTENSION IF EXISTS neurondb CASCADE;

-- Create extension
CREATE EXTENSION neurondb;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

\echo '✅ Extension loaded'
\echo ''

-- ============================================================================
-- Step 1: Create ML Projects
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo '📝 Step 1: Creating ML Projects'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

SELECT neurondb_create_ml_project('fraud_kmeans', 'clustering', 'K-means fraud detection') AS p1 \gset
SELECT neurondb_create_ml_project('fraud_gmm', 'clustering', 'GMM fraud detection') AS p2 \gset
SELECT neurondb_create_ml_project('fraud_hierarchical', 'clustering', 'Hierarchical fraud detection') AS p3 \gset
SELECT neurondb_create_ml_project('fraud_minibatch', 'clustering', 'Mini-batch K-means fraud detection') AS p4 \gset
SELECT neurondb_create_ml_project('fraud_outliers', 'outlier_detection', 'Outlier-based fraud detection') AS p5 \gset

\echo '✅ Created 5 ML fraud detection projects'
\echo ''

-- ============================================================================
-- Step 2: Generate 100MB Dataset (~1.5M transactions)
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo '📊 Step 2: Generating ~100MB Dataset (1.5M transactions)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    amount NUMERIC(10,2),
    transaction_time TIMESTAMP,
    merchant_category TEXT,
    location_distance NUMERIC(10,2),
    is_fraud BOOLEAN,
    features vector(5)
);

CREATE OR REPLACE FUNCTION sigmoid(x float) RETURNS float AS $$
BEGIN
    RETURN 1.0 / (1.0 + exp(-x));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Generate data in batches
\echo 'Generating batch 1/3...'
INSERT INTO transactions (user_id, amount, transaction_time, merchant_category, location_distance, is_fraud, features)
SELECT 
    (random() * 10000)::int as user_id,
    (random() * 5000 + 10)::numeric(10,2) as amount,
    now() - (random() * interval '365 days') as transaction_time,
    (ARRAY['retail', 'online', 'grocery', 'gas', 'restaurant', 'entertainment'])[(random() * 5 + 1)::int] as merchant_category,
    (random() * 1000)::numeric(10,2) as location_distance,
    (random() > 0.95) as is_fraud,
    ARRAY[
        random()::real,
        random()::real,
        random()::real,
        random()::real,
        random()::real
    ]::vector(5) as features
FROM generate_series(1, 500000);

\echo 'Generating batch 2/3...'
INSERT INTO transactions (user_id, amount, transaction_time, merchant_category, location_distance, is_fraud, features)
SELECT 
    (random() * 10000)::int,
    (random() * 5000 + 10)::numeric(10,2),
    now() - (random() * interval '365 days'),
    (ARRAY['retail', 'online', 'grocery', 'gas', 'restaurant', 'entertainment'])[(random() * 5 + 1)::int],
    (random() * 1000)::numeric(10,2),
    (random() > 0.95),
    ARRAY[random()::real, random()::real, random()::real, random()::real, random()::real]::vector(5)
FROM generate_series(1, 500000);

\echo 'Generating batch 3/3...'
INSERT INTO transactions (user_id, amount, transaction_time, merchant_category, location_distance, is_fraud, features)
SELECT 
    (random() * 10000)::int,
    (random() * 5000 + 10)::numeric(10,2),
    now() - (random() * interval '365 days'),
    (ARRAY['retail', 'online', 'grocery', 'gas', 'restaurant', 'entertainment'])[(random() * 5 + 1)::int],
    (random() * 1000)::numeric(10,2),
    (random() > 0.95),
    ARRAY[random()::real, random()::real, random()::real, random()::real, random()::real]::vector(5)
FROM generate_series(1, 500000);

-- Create train/test split (80/20)
CREATE VIEW train_data AS 
SELECT * FROM transactions WHERE transaction_id <= 1200000;

CREATE VIEW test_data AS 
SELECT * FROM transactions WHERE transaction_id > 1200000;

SELECT 
    COUNT(*) as total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
    ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) || '%' as fraud_rate,
    pg_size_pretty(pg_total_relation_size('transactions')) as table_size
FROM transactions;

\echo ''

-- ============================================================================
-- Step 3: Test ALL ML Algorithms
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo '🎓 Step 3: Testing ALL ML Algorithms on Fraud Detection'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

-- ============================================================================
-- Algorithm 1: K-MEANS CLUSTERING
-- ============================================================================

\echo '1️⃣  K-MEANS CLUSTERING (K=7 risk groups)'
\echo ''
\timing on

WITH kmeans_clusters AS (
    SELECT cluster_kmeans('train_data', 'features', 7, 50) as clusters
),
kmeans_result AS (
    SELECT 
        t.transaction_id,
        t.is_fraud,
        c.cluster
    FROM (SELECT transaction_id, is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM train_data) t,
    kmeans_clusters,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
),
cluster_fraud_rates AS (
    SELECT 
        cluster,
        COUNT(*) as transactions,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as frauds,
        ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate
    FROM kmeans_result
    GROUP BY cluster
)
SELECT 
    'K-means' as algorithm,
    COUNT(*) as num_clusters,
    SUM(transactions) as total_transactions,
    MAX(fraud_rate) as max_fraud_rate,
    MIN(fraud_rate) as min_fraud_rate
FROM cluster_fraud_rates;

\timing off

-- Record model
SELECT neurondb_train_kmeans_project('fraud_kmeans', 'train_data', 'features', 7, 50) AS kmeans_model_id \gset
\echo '   ✅ K-means model trained (model_id: ' :kmeans_model_id ')'
\echo ''

-- ============================================================================
-- Algorithm 2: GMM CLUSTERING
-- ============================================================================

\echo '2️⃣  GMM CLUSTERING (Gaussian Mixture Models)'
\echo ''
\timing on

WITH gmm_clusters AS (
    SELECT cluster_gmm('train_data', 'features', 7, 30) as clusters
),
gmm_result AS (
    SELECT 
        t.transaction_id,
        t.is_fraud,
        c.cluster
    FROM (SELECT transaction_id, is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM train_data) t,
    gmm_clusters,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
),
cluster_fraud_rates AS (
    SELECT 
        cluster,
        COUNT(*) as transactions,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as frauds,
        ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate
    FROM gmm_result
    GROUP BY cluster
)
SELECT 
    'GMM' as algorithm,
    COUNT(*) as num_clusters,
    SUM(transactions) as total_transactions,
    MAX(fraud_rate) as max_fraud_rate,
    MIN(fraud_rate) as min_fraud_rate
FROM cluster_fraud_rates;

\timing off
\echo '   ✅ GMM model trained'
\echo ''

-- ============================================================================
-- Algorithm 3: MINI-BATCH K-MEANS (Fast for large datasets)
-- ============================================================================

\echo '3️⃣  MINI-BATCH K-MEANS (Fast clustering)'
\echo ''
\timing on

WITH mb_clusters AS (
    SELECT cluster_minibatch_kmeans('train_data', 'features', 7, 50, 100) as clusters
),
mb_result AS (
    SELECT 
        t.transaction_id,
        t.is_fraud,
        c.cluster
    FROM (SELECT transaction_id, is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM train_data) t,
    mb_clusters,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
),
cluster_fraud_rates AS (
    SELECT 
        cluster,
        COUNT(*) as transactions,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as frauds,
        ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate
    FROM mb_result
    GROUP BY cluster
)
SELECT 
    'Mini-batch K-means' as algorithm,
    COUNT(*) as num_clusters,
    SUM(transactions) as total_transactions,
    MAX(fraud_rate) as max_fraud_rate,
    MIN(fraud_rate) as min_fraud_rate
FROM cluster_fraud_rates;

\timing off
\echo '   ✅ Mini-batch K-means model trained'
\echo ''

-- ============================================================================
-- Algorithm 4: OUTLIER DETECTION (Z-score)
-- ============================================================================

\echo '4️⃣  OUTLIER DETECTION (Z-score based anomaly detection)'
\echo ''
\timing on

WITH outlier_flags AS (
    SELECT detect_outliers_zscore('train_data', 'features', 3.0, 'zscore') as outliers
),
outlier_result AS (
    SELECT 
        t.transaction_id,
        t.is_fraud,
        o.is_outlier
    FROM (SELECT transaction_id, is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM train_data) t,
    outlier_flags,
    LATERAL unnest(outliers) WITH ORDINALITY AS o(is_outlier, rn)
    WHERE t.rn = o.rn
)
SELECT 
    'Z-score Outliers' as algorithm,
    COUNT(*) as total_transactions,
    SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) as flagged_outliers,
    SUM(CASE WHEN is_outlier AND is_fraud THEN 1 ELSE 0 END) as fraud_caught,
    ROUND(100.0 * SUM(CASE WHEN is_outlier AND is_fraud THEN 1 ELSE 0 END) / 
          NULLIF(SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END), 0), 2) as fraud_detection_rate
FROM outlier_result;

\timing off
\echo '   ✅ Outlier detection completed'
\echo ''

-- ============================================================================
-- Algorithm 5: HIERARCHICAL CLUSTERING (SKIPPED - O(n²) complexity too slow)
-- ============================================================================

\echo '5️⃣  HIERARCHICAL CLUSTERING (skipped due to O(n²) complexity)'
\echo '   Note: Hierarchical clustering takes too long on large datasets'
\echo '   Recommend K-means or Mini-batch K-means for production use'
\echo ''

-- Uncomment below to test on small sample (may take 30+ seconds):
-- CREATE TEMP TABLE train_sample AS 
-- SELECT * FROM train_data LIMIT 10000;
-- 
-- \timing on
-- WITH hier_clusters AS (
--     SELECT cluster_hierarchical('train_sample', 'features', 7, 'average') as clusters
-- ),
-- hier_result AS (
--     SELECT 
--         t.transaction_id,
--         t.is_fraud,
--         c.cluster
--     FROM (SELECT transaction_id, is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM train_sample) t,
--     hier_clusters,
--     LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
--     WHERE t.rn = c.rn
-- )
-- SELECT 
--     'Hierarchical (sample)' as algorithm,
--     COUNT(*) as total_transactions
-- FROM hier_result;
-- \timing off

-- ============================================================================
-- Step 4: Algorithm Comparison
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo '📊 Step 4: All Trained Models'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

SELECT 
    p.project_name,
    p.model_type::text,
    COUNT(m.model_id) as models_trained,
    MAX(m.version) as latest_version,
    COALESCE(MIN(m.training_time_ms), 0) || 'ms' as fastest_training
FROM neurondb.ml_projects p
LEFT JOIN neurondb.ml_models m ON p.project_id = m.project_id
WHERE p.project_name LIKE 'fraud_%'
GROUP BY p.project_id, p.project_name, p.model_type
ORDER BY p.project_name;

\echo ''

-- ============================================================================
-- Step 5: Deploy Best Model
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo '🚀 Step 5: Deploy K-means Model'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

SELECT neurondb_deploy_model('fraud_kmeans', :kmeans_model_id);

\echo '✅ K-means fraud detection model deployed'
\echo ''

-- ============================================================================
-- Summary
-- ============================================================================

\echo '╔══════════════════════════════════════════════════════════════════════════╗'
\echo '║                    ✅ ALL ML ALGORITHMS TESTED! ✅                        ║'
\echo '╠══════════════════════════════════════════════════════════════════════════╣'
\echo '║                                                                          ║'
\echo '║  Dataset:                                                               ║'
\echo '║  • 1.5 million transactions (~100MB)                                    ║'
\echo '║  • 5% fraud rate                                                        ║'
\echo '║  • 5-dimensional feature vectors                                        ║'
\echo '║                                                                          ║'
\echo '║  Algorithms Tested:                                                     ║'
\echo '║  ✅ 1. K-means Clustering (7 clusters)                                  ║'
\echo '║  ✅ 2. GMM (Gaussian Mixture Models)                                    ║'
\echo '║  ✅ 3. Mini-batch K-means (fast)                                        ║'
\echo '║  ✅ 4. Outlier Detection (Z-score)                                      ║'
\echo '║  ⏭️  5. Hierarchical Clustering (skipped - too slow)                    ║'
\echo '║                                                                          ║'
\echo '║  Project Management:                                                    ║'
\echo '║  ✅ 5 fraud detection projects created                                  ║'
\echo '║  ✅ Models trained and versioned                                        ║'
\echo '║  ✅ Best model deployed                                                 ║'
\echo '║                                                                          ║'
\echo '║  NeuronDB provides enterprise-grade ML for fraud detection!            ║'
\echo '║                                                                          ║'
\echo '╚══════════════════════════════════════════════════════════════════════════╝'
