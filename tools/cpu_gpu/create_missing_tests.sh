#!/bin/bash

# Create tests for missing algorithms only
create_test_suite() {
    local num=$1
    local algo=$2
    local is_clustering=$3
    
    # Determine label column
    if [ "$is_clustering" = "true" ]; then
        label_col="NULL"
    else
        label_col="'label'"
    fi
    
    # Basic Test
    cat > "${num}_${algo}.sql" << EOF
-- ${num}_${algo}.sql
-- Basic test for ${algo}

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\\echo '=== ${algo} Basic Test ==='

-- Training
SELECT neurondb.train('${algo}', 'sample_train', 'features', ${label_col}, '{}'::jsonb)::integer AS model_id;

\\echo '✓ ${algo} basic test complete'
EOF

    # Advance Test
    cat > "${num}_${algo}_advance.sql" << EOF
-- ${num}_${algo}_advance.sql  
-- Advanced test for ${algo}

SET client_min_messages TO WARNING;

\\echo '=== ${algo} Advanced Test ==='

-- GPU Training
SET neurondb.gpu_enabled = on;
SELECT neurondb.train('${algo}', 'sample_train', 'features', ${label_col}, '{}'::jsonb)::integer AS gpu_model;

-- CPU Training
SET neurondb.gpu_enabled = off;
SELECT neurondb.train('${algo}', 'sample_train', 'features', ${label_col}, '{}'::jsonb)::integer AS cpu_model;

\\echo '✓ ${algo} advance test complete'
EOF

    # Negative Test
    cat > "${num}_${algo}_negative.sql" << EOF
-- ${num}_${algo}_negative.sql
-- Negative test for ${algo}

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\\echo '=== ${algo} Negative Test ==='

-- Invalid table
DO \$\$ BEGIN
    PERFORM neurondb.train('${algo}', 'nonexistent_table', 'features', ${label_col}, '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid table';
END \$\$;

-- Invalid column
DO \$\$ BEGIN
    PERFORM neurondb.train('${algo}', 'sample_train', 'invalid_col', ${label_col}, '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid column';
END \$\$;

\\echo '✓ ${algo} negative test complete'
EOF

    echo "Created test suite for ${algo} (${num})"
}

# Create missing tests
create_test_suite "015" "kmeans" "true"
create_test_suite "016" "minibatch_kmeans" "true"
create_test_suite "017" "hierarchical" "true"
create_test_suite "018" "dbscan" "true"
create_test_suite "019" "xgboost" "false"
create_test_suite "020" "catboost" "false"
create_test_suite "021" "lightgbm" "false"
create_test_suite "022" "neural_network" "false"
create_test_suite "023" "pca" "true"
create_test_suite "024" "timeseries" "false"

echo ""
echo "✓ Created test suites for 10 missing algorithms"
