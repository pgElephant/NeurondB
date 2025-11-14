\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'AutoML - Negative Test Cases (Error Handling)'
\echo '=========================================================================='

\echo ''
\echo 'Test 1: NULL Table Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	NULL,
	'features',
	'label',
	'classification',
	'accuracy'
);

\echo ''
\echo 'Test 2: NULL Feature Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'sample_train',
	NULL,
	'label',
	'classification',
	'accuracy'
);

\echo ''
\echo 'Test 3: NULL Target Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'sample_train',
	'features',
	NULL,
	'classification',
	'accuracy'
);

\echo ''
\echo 'Test 4: NULL Task Type'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'sample_train',
	'features',
	'label',
	NULL,
	'accuracy'
);

\echo ''
\echo 'Test 5: NULL Metric Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'sample_train',
	'features',
	'label',
	'classification',
	NULL
);

\echo ''
\echo 'Test 6: Non-existent Table'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'nonexistent_table_xyz',
	'features',
	'label',
	'classification',
	'accuracy'
);

\echo ''
\echo 'Test 7: Non-existent Feature Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'sample_train',
	'nonexistent_column',
	'label',
	'classification',
	'accuracy'
);

\echo ''
\echo 'Test 8: Non-existent Target Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'sample_train',
	'features',
	'nonexistent_label',
	'classification',
	'accuracy'
);

\echo ''
\echo 'Test 9: Invalid Task Type'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'sample_train',
	'features',
	'label',
	'invalid_task_type',
	'accuracy'
);

\echo ''
\echo 'Test 10: Invalid Metric Name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'sample_train',
	'features',
	'label',
	'classification',
	'invalid_metric_name'
);

\echo ''
\echo 'Test 11: Mismatched Task Type and Metric'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Using regression metric for classification task
SELECT auto_train(
	'sample_train',
	'features',
	'label',
	'classification',
	'r2'
);

\echo ''
\echo 'Test 12: Empty Training Table'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE empty_train (
	features vector,
	label float8
);

SELECT auto_train(
	'empty_train',
	'features',
	'label',
	'classification',
	'accuracy'
);

DROP TABLE IF EXISTS empty_train;

\echo ''
\echo 'Test 13: Insufficient Data for Training'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE tiny_train (
	features vector(5),
	label float8
);

-- Insert only 1 row (insufficient for training)
INSERT INTO tiny_train VALUES ('[1,2,3,4,5]'::vector, 1.0);

SELECT auto_train(
	'tiny_train',
	'features',
	'label',
	'classification',
	'accuracy'
);

DROP TABLE IF EXISTS tiny_train;

\echo ''
\echo 'Test 14: Invalid Feature Dimensions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE invalid_dim_train (
	features vector(3),
	label float8
);

-- Insert rows with different dimensions (if possible)
INSERT INTO invalid_dim_train VALUES 
	('[1,2,3]'::vector, 1.0),
	('[4,5,6]'::vector, 0.0);

SELECT auto_train(
	'invalid_dim_train',
	'features',
	'label',
	'classification',
	'accuracy'
);

DROP TABLE IF EXISTS invalid_dim_train;

\echo ''
\echo 'Test 15: All NULL Values in Target Column'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE null_target_train (
	features vector(5),
	label float8
);

INSERT INTO null_target_train VALUES 
	('[1,2,3,4,5]'::vector, NULL),
	('[6,7,8,9,10]'::vector, NULL);

SELECT auto_train(
	'null_target_train',
	'features',
	'label',
	'classification',
	'accuracy'
);

DROP TABLE IF EXISTS null_target_train;

\echo ''
\echo 'Negative AutoML Test Complete!'
\echo '=========================================================================='

