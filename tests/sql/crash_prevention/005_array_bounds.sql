/*
 * Crash Prevention Test: Array Bounds and Validation
 * Tests that functions properly validate array bounds and dimensions
 * without crashing.
 */

\set ON_ERROR_STOP on

BEGIN;

/* Test 1: Empty array */
SELECT predict_linear_regression_by_model_id(1, ARRAY[]::float4[]);
ROLLBACK;

BEGIN;
/* Test 2: Wrong dimension array */
/* Model expects 3 features, provide 2 */
SELECT predict_linear_regression_by_model_id(1, ARRAY[1.0, 2.0]::float4[]);
ROLLBACK;

BEGIN;
/* Test 3: Too many dimensions */
SELECT predict_linear_regression_by_model_id(1, ARRAY[1.0, 2.0, 3.0, 4.0, 5.0]::float4[]);
ROLLBACK;

BEGIN;
/* Test 4: NULL elements in array */
SELECT predict_linear_regression_by_model_id(1, ARRAY[1.0, NULL, 3.0]::float4[]);
ROLLBACK;

BEGIN;
/* Test 5: Very large array (should handle gracefully or error, not crash) */
SELECT predict_linear_regression_by_model_id(1, 
	(SELECT array_agg(random()::float4) FROM generate_series(1, 100000))::float4[]
);
ROLLBACK;

COMMIT;

