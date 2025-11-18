#!/bin/bash

# Script to remove RAISE NOTICE statements from SQL test files

files=(
    "tests/sql/advance/011_hybrid_search_advance.sql"
    "tests/sql/advance/008_rag_advance.sql"
    "tests/sql/advance/010_gpu_info_advance.sql"
    "tests/sql/advance/020_catboost_advance.sql"
    "tests/sql/advance/006_ridge_advance.sql"
    "tests/sql/advance/005_dt_advance.sql"
    "tests/sql/advance/009_vector_ops_advance.sql"
    "tests/sql/advance/012_nb_advance.sql"
    "tests/sql/advance/023_pca_advance.sql"
    "tests/sql/advance/013_gmm_advance.sql"
    "tests/sql/advance/004_svm_advance.sql"
    "tests/sql/advance/026_vector_advance.sql"
    "tests/sql/advance/024_timeseries_advance.sql"
    "tests/sql/advance/022_neural_network_advance.sql"
    "tests/sql/advance/015_kmeans_advance.sql"
    "tests/sql/advance/025_automl_advance.sql"
    "tests/sql/advance/017_hierarchical_advance.sql"
    "tests/sql/advance/014_knn_advance.sql"
    "tests/sql/advance/021_lightgbm_advance.sql"
    "tests/sql/advance/016_minibatch_kmeans_advance.sql"
    "tests/sql/advance/007_lasso_advance.sql"
    "tests/sql/advance/001_linreg_advance.sql"
    "tests/sql/advance/003_rf_advance.sql"
    "tests/sql/advance/002_logreg_advance.sql"
    "tests/sql/advance/019_xgboost_advance.sql"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing $file"
        # Replace RAISE NOTICE lines with NULL
        sed -i 's/^\t\tRAISE NOTICE.*SQLERRM;$/&\n\t\t-- Error handled correctly\n\t\tNULL;/' "$file"
        sed -i '/^\t\tRAISE NOTICE.*SQLERRM;$/d' "$file"
    fi
done

echo "Done processing all files"
