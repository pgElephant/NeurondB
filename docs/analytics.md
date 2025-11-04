# ML Analytics

This page mirrors the NeuronDB ML fraud detection demo in `NeurondB/demo/ML` and provides ready-to-run SQL examples for clustering and anomaly detection.

## Dataset assumptions

- A table with a vector column named `features`.
- Optional demo views: `train_data` (80%) and `test_data` (20%) as in `001_generate_dataset.sql`.

See `NeurondB/demo/ML/001_generate_dataset.sql` to generate a ~1.5M row dataset for full-scale testing.

## K-Means clustering

```sql
-- Cluster into K groups
SELECT cluster_kmeans(
  'train_data',   -- table
  'features',     -- vector column
  7,              -- K
  50              -- max iterations
) AS clusters;

-- Project-based training and versioning
SELECT neurondb_train_kmeans_project(
  'fraud_kmeans',
  'train_data', 'features',
  7, 50
) AS model_id;

-- List project models
SELECT version, algorithm, parameters, is_deployed
FROM neurondb_list_project_models('fraud_kmeans')
ORDER BY version;
```

## Mini-batch K-Means (fast)

```sql
SELECT cluster_minibatch_kmeans(
  'train_data', 'features',
  7,      -- K
  50,     -- max iterations
  100     -- batch size
) AS clusters;
```

## Gaussian Mixture Models (GMM)

GMM returns a probability matrix (N×K). Convert to cluster IDs via a small helper (from `003_gmm_clustering.sql`):

```sql
CREATE OR REPLACE FUNCTION gmm_to_clusters(probs float8[][])
RETURNS integer[] LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE r integer[] := ARRAY[]::integer[]; i int; j int; k int; m float8; b int; BEGIN
  k := array_length(probs,2);
  FOR i IN 1..array_length(probs,1) LOOP
    m := -1; b := 1;
    FOR j IN 1..k LOOP
      IF probs[i][j] > m THEN m := probs[i][j]; b := j; END IF;
    END LOOP;
    r := array_append(r, b);
  END LOOP; RETURN r; END; $$;

WITH p AS (
  SELECT cluster_gmm('train_data','features',7,30) AS probs
)
SELECT gmm_to_clusters(probs) FROM p;
```

## Outlier detection (Z-score)

```sql
-- Flag outliers using Z-score threshold
SELECT detect_outliers_zscore(
  'train_data', 'features',
  3.0, 'zscore'
) AS outliers;

-- Example: fraud detection rate among outliers (demo schema)
WITH flags AS (
  SELECT detect_outliers_zscore('train_data','features',3.0,'zscore') AS f
), labeled AS (
  SELECT t.is_fraud, o.is_outlier
  FROM (SELECT is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) rn FROM train_data) t,
       flags,
       LATERAL unnest(f) WITH ORDINALITY AS o(is_outlier, rn)
  WHERE t.rn = o.rn
)
SELECT ROUND(100.0*SUM(CASE WHEN is_outlier AND is_fraud THEN 1 ELSE 0 END)/NULLIF(SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END),0),2) AS fraud_detection_rate
FROM labeled;
```

## Notes

- These examples are CPU-based today. GPU acceleration covers vector distance and quantization; GPU clustering is planned.
- For full workflows (training, testing, deployment), run the SQL scripts in `NeurondB/demo/ML/` in order as detailed in `README.md`.
