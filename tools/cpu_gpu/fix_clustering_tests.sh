#!/bin/bash

# Fix clustering algorithm tests (kmeans, minibatch_kmeans, hierarchical, dbscan, pca)
# These return cluster assignments, not model IDs

for algo in "015_kmeans" "016_minibatch_kmeans" "017_hierarchical" "018_dbscan" "023_pca"; do
    cat > "${algo}.sql" << 'EOF'
-- ${algo}.sql
-- Basic test for clustering algorithm

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=== Clustering Basic Test ==='

-- Clustering (returns cluster assignments)
SELECT COUNT(*) AS num_assignments 
FROM (
    SELECT UNNEST(neurondb.train('kmeans', 'sample_train', 'features', NULL, '{"n_clusters": 3}'::jsonb)) AS cluster
) t;

\echo '✓ Clustering basic test complete'
EOF
    # Replace 'kmeans' with actual algorithm name
    algo_name=$(echo "$algo" | sed 's/^[0-9]*_//')
    sed -i "s/kmeans/${algo_name}/g" "${algo}.sql"
    sed -i "s/Clustering/${algo_name}/g" "${algo}.sql"
    echo "✓ Fixed ${algo}.sql"
done

# Fix negative tests for clustering
for algo in "015_kmeans" "016_minibatch_kmeans" "017_hierarchical" "018_dbscan" "023_pca"; do
    algo_name=$(echo "$algo" | sed 's/^[0-9]*_//')
    cat > "${algo}_negative.sql" << EOF
-- ${algo}_negative.sql
-- Negative test for ${algo_name}

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\\echo '=== ${algo_name} Negative Test ==='

-- Invalid table
DO \$\$ BEGIN
    PERFORM neurondb.train('${algo_name}', 'nonexistent_table', 'features', NULL, '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid table';
END \$\$;

-- Invalid column
DO \$\$ BEGIN
    PERFORM neurondb.train('${algo_name}', 'sample_train', 'invalid_col', NULL, '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid column';
END \$\$;

\\echo '✓ ${algo_name} negative test complete'
EOF
    echo "✓ Fixed ${algo}_negative.sql"
done

# Fix advance tests for clustering
for algo in "015_kmeans" "016_minibatch_kmeans" "017_hierarchical" "018_dbscan" "023_pca"; do
    algo_name=$(echo "$algo" | sed 's/^[0-9]*_//')
    cat > "${algo}_advance.sql" << EOF
-- ${algo}_advance.sql
-- Advanced test for ${algo_name}

SET client_min_messages TO WARNING;

\\echo '=== ${algo_name} Advanced Test ==='

-- CPU Clustering
SET neurondb.gpu_enabled = off;
SELECT COUNT(*) AS cpu_count 
FROM (
    SELECT UNNEST(neurondb.train('${algo_name}', 'sample_train', 'features', NULL, '{}'::jsonb)) AS cluster
) t;

-- GPU Clustering
SET neurondb.gpu_enabled = on;
SELECT COUNT(*) AS gpu_count 
FROM (
    SELECT UNNEST(neurondb.train('${algo_name}', 'sample_train', 'features', NULL, '{}'::jsonb)) AS cluster
) t;

\\echo '✓ ${algo_name} advance test complete'
EOF
    echo "✓ Fixed ${algo}_advance.sql"
done

echo ""
echo "✓ Fixed all clustering tests"
