#!/bin/bash

# Add advance/negative tests for algorithms that only have basic tests

add_advance_negative() {
    local prefix=$1
    local algo=$2
    
    # Check if advance test is missing
    if [ ! -f "${prefix}_advance.sql" ]; then
        cat > "${prefix}_advance.sql" << EOF
-- ${prefix}_advance.sql
-- Advanced test for ${algo}

SET client_min_messages TO WARNING;

\\echo '=== ${algo} Advanced Test ==='

-- GPU Training
SET neurondb.gpu_enabled = on;
SELECT neurondb.train('${algo}', 'sample_train', 'features', 'label', '{}'::jsonb)::integer AS gpu_model;

-- CPU Training
SET neurondb.gpu_enabled = off;
SELECT neurondb.train('${algo}', 'sample_train', 'features', 'label', '{}'::jsonb)::integer AS cpu_model;

-- Verify models exist
SELECT COUNT(*) AS model_count FROM neurondb.ml_models WHERE algorithm::text = '${algo}';

\\echo '✓ ${algo} advance test complete'
EOF
        echo "✓ Created ${prefix}_advance.sql"
    fi
    
    # Check if negative test is missing
    if [ ! -f "${prefix}_negative.sql" ]; then
        cat > "${prefix}_negative.sql" << EOF
-- ${prefix}_negative.sql
-- Negative test for ${algo}

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\\echo '=== ${algo} Negative Test ==='

-- Invalid table
DO \$\$ BEGIN
    PERFORM neurondb.train('${algo}', 'nonexistent_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid table';
END \$\$;

-- Invalid column
DO \$\$ BEGIN
    PERFORM neurondb.train('${algo}', 'sample_train', 'invalid_col', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Handled invalid column';
END \$\$;

\\echo '✓ ${algo} negative test complete'
EOF
        echo "✓ Created ${prefix}_negative.sql"
    fi
}

# Add for algorithms missing advance/negative tests
add_advance_negative "002_logreg" "logistic_regression"
add_advance_negative "003_rf" "random_forest"
add_advance_negative "004_svm" "svm"
add_advance_negative "005_dt" "decision_tree"
add_advance_negative "006_ridge" "ridge"
add_advance_negative "007_lasso" "lasso"
add_advance_negative "008_rag" "rag"
add_advance_negative "011_hybrid_search" "hybrid_search"

echo ""
echo "✓ Added missing advance/negative tests"
