#!/bin/bash
echo "=== Running All Basic Tests ===" > log.txt
echo "Date: $(date)" >> log.txt
echo "" >> log.txt

tests=(
  "tests/sql/basic/001_linreg_basic.sql"
  "tests/sql/basic/002_logreg_basic.sql"
  "tests/sql/basic/003_rf_basic.sql"
  "tests/sql/basic/004_svm_basic.sql"
  "tests/sql/basic/005_dt_basic.sql"
  "tests/sql/basic/006_ridge_basic.sql"
  "tests/sql/basic/007_lasso_basic.sql"
  "tests/sql/basic/008_rag_basic.sql"
  "tests/sql/basic/009_vector_ops_basic.sql"
  "tests/sql/basic/010_gpu_info_basic.sql"
  "tests/sql/basic/011_hybrid_search_basic.sql"
  "tests/sql/basic/012_nb_basic.sql"
  "tests/sql/basic/013_gmm_basic.sql"
  "tests/sql/basic/014_knn_basic.sql"
  "tests/sql/basic/015_kmeans_basic.sql"
  "tests/sql/basic/016_minibatch_kmeans_basic.sql"
  "tests/sql/basic/017_hierarchical_basic.sql"
  "tests/sql/basic/018_dbscan_basic.sql"
  "tests/sql/basic/019_xgboost_basic.sql"
  "tests/sql/basic/020_catboost_basic.sql"
  "tests/sql/basic/021_lightgbm_basic.sql"
  "tests/sql/basic/022_neural_network_basic.sql"
  "tests/sql/basic/023_pca_basic.sql"
  "tests/sql/basic/024_timeseries_basic.sql"
  "tests/sql/basic/025_automl_basic.sql"
  "tests/sql/basic/025_automl_standalone_basic.sql"
  "tests/sql/basic/026_vector_basic.sql"
  "tests/sql/basic/027_recommender_basic.sql"
  "tests/sql/basic/028_arima_basic.sql"
)

for test in "${tests[@]}"; do
  if [ -f "$test" ]; then
    echo "========================================" >> log.txt
    echo "TEST: $test" >> log.txt
    echo "========================================" >> log.txt
    timeout 120 /usr/local/pgsql.18/bin/psql neurondb -f "$test" >> log.txt 2>&1
    exitcode=$?
    if [ $exitcode -eq 0 ]; then
      echo "RESULT: PASS" >> log.txt
    elif [ $exitcode -eq 124 ]; then
      echo "RESULT: TIMEOUT" >> log.txt
    else
      echo "RESULT: FAIL (exit code: $exitcode)" >> log.txt
    fi
    echo "" >> log.txt
  else
    echo "TEST: $test - FILE NOT FOUND" >> log.txt
    echo "" >> log.txt
  fi
done

echo "=== Test Summary ===" >> log.txt
grep "RESULT:" log.txt | sort | uniq -c >> log.txt
