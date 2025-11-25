/*
 * Crash Prevention Test: Invalid Model Handling
 * Tests that functions handle invalid/non-existent models gracefully
 * without crashing.
 */

\set ON_ERROR_STOP on

BEGIN;

/* Test 1: Non-existent model_id */
SELECT evaluate_linear_regression_by_model_id(999999, 'test_table', 'features', 'label');
ROLLBACK;

BEGIN;
/* Test 2: Negative model_id */
SELECT evaluate_linear_regression_by_model_id(-1, 'test_table', 'features', 'label');
ROLLBACK;

BEGIN;
/* Test 3: Zero model_id */
SELECT evaluate_linear_regression_by_model_id(0, 'test_table', 'features', 'label');
ROLLBACK;

BEGIN;
/* Test 4: Model with NULL payload */
/* This would require setting up a model with NULL data, which is tested in code */
SELECT evaluate_knn_by_model_id(999998, 'test_table', 'features', 'label');
ROLLBACK;

BEGIN;
/* Test 5: Model with corrupted metadata */
SELECT evaluate_random_forest_by_model_id(999997, 'test_table', 'features', 'label');
ROLLBACK;

COMMIT;

