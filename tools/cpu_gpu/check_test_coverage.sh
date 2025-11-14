#!/bin/bash

echo "=== ML Algorithm Test Coverage ==="
echo ""

# List of all trainable algorithms from our scan
ALGOS=(
    "linear_regression:001_linreg"
    "logistic_regression:002_logreg"
    "random_forest:003_rf"
    "svm:004_svm"
    "decision_tree:005_dt"
    "ridge:006_ridge"
    "lasso:007_lasso"
    "naive_bayes:012_nb"
    "gmm:013_gmm"
    "knn:014_knn"
    "automl:000_automl"
    "vector_ops:009_vector_ops"
    "gpu_info:010_gpu_info"
    "rag:008_rag"
    "hybrid_search:011_hybrid_search"
)

echo "ALGORITHMS WITH TESTS:"
for algo_info in "${ALGOS[@]}"; do
    IFS=':' read -r algo prefix <<< "$algo_info"
    
    basic=$(ls ${prefix}.sql 2>/dev/null | wc -l)
    advance=$(ls ${prefix}_advance.sql 2>/dev/null | wc -l)
    negative=$(ls ${prefix}_negative.sql 2>/dev/null | wc -l)
    
    total=$((basic + advance + negative))
    
    if [ $total -gt 0 ]; then
        status="✓"
        [ $basic -gt 0 ] && b="B" || b="-"
        [ $advance -gt 0 ] && a="A" || a="-"
        [ $negative -gt 0 ] && n="N" || n="-"
        echo "  $status $algo [$b$a$n]"
    fi
done

echo ""
echo "ALGORITHMS MISSING TESTS:"
# Check for algorithms without tests
MISSING_ALGOS=(
    "kmeans"
    "minibatch_kmeans"
    "hierarchical"
    "dbscan"
    "catboost"
    "lightgbm"
    "neural_network"
    "xgboost"
    "pca"
    "timeseries"
    "text"
    "nlp"
    "recommender"
    "topic_discovery"
)

for algo in "${MISSING_ALGOS[@]}"; do
    # Check if any test file exists for this algorithm
    if ! ls *${algo}*.sql 2>/dev/null | grep -q .; then
        echo "  ✗ $algo [---]"
    fi
done
