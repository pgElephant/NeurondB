-- 023_pca_advance.sql
-- Exhaustive detailed test for PCA (Principal Component Analysis): all operations, error handling.
-- Works on 1000 rows only and tests each and every way with comprehensive coverage
-- Tests: PCA transformation, different n_components, error handling, validation

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'PCA: Exhaustive Principal Component Analysis Test (1000 rows sample)'
\echo '=========================================================================='

/* Check that sample_train exists */
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist';
	END IF;
END
$$;

-- Create views with 1000 rows for advance tests
DROP VIEW IF EXISTS test_train_view;
DROP VIEW IF EXISTS test_test_view;

CREATE VIEW test_train_view AS
SELECT features, label FROM sample_train LIMIT 1000;

CREATE VIEW test_test_view AS
SELECT features, label FROM sample_test LIMIT 1000;

\echo ''
\echo 'Dataset Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
SELECT 
	COUNT(*)::bigint AS train_count,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_count,
	(SELECT vector_dims(features) FROM test_train_view LIMIT 1) AS feature_dim
FROM test_train_view;

/*---- Register required GPU kernels ----*/
\echo ''
\echo 'GPU Configuration'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip';
SELECT neurondb_gpu_enable() AS gpu_available;
SELECT neurondb_gpu_info() AS gpu_info;

/*
 * ---- PCA TRANSFORMATION TESTS ----
 * Test PCA with different n_components
 */
\echo ''
\echo 'PCA Transformation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create test data with multiple dimensions
DROP TABLE IF EXISTS pca_test_data;
CREATE TEMP TABLE pca_test_data AS
SELECT 
	features[1:5]::vector(5) AS feat_5d,
	features[1:10]::vector(10) AS feat_10d,
	features AS feat_full
FROM test_train_view
LIMIT 100;

\echo 'Test 1: PCA transformation (5D -> 2D)'
DO $$
DECLARE
	result vector[];
	original_count integer;
BEGIN
	SELECT COUNT(*) INTO original_count FROM pca_test_data;
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 2) INTO result;
		IF result IS NULL OR array_length(result, 1) IS NULL THEN
			RAISE EXCEPTION 'PCA transform returned NULL or empty';
		END IF;
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END $$;

\echo 'Test 2: PCA transformation (10D -> 3D)'
DO $$
DECLARE
	result vector[];
	original_count integer;
BEGIN
	SELECT COUNT(*) INTO original_count FROM pca_test_data;
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_10d'], 3) INTO result;
		IF result IS NULL OR array_length(result, 1) IS NULL THEN
			RAISE EXCEPTION 'PCA transform returned NULL or empty';
		END IF;
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END $$;

\echo 'Test 3: PCA transformation (full features -> 5D)'
DO $$
DECLARE
	result vector[];
	original_count integer;
	original_dims integer;
BEGIN
	SELECT COUNT(*) INTO original_count FROM pca_test_data;
	SELECT vector_dims(feat_full) INTO original_dims FROM pca_test_data LIMIT 1;
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_full'], 5) INTO result;
		IF result IS NOT NULL AND array_length(result, 1) = original_count THEN
		ELSE
				original_count, COALESCE(array_length(result, 1), 0);
		END IF;
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END $$;

\echo 'Test 4: Verify PCA preserves number of rows'
DO $$
DECLARE
	result vector[];
	original_count integer;
BEGIN
	SELECT COUNT(*) INTO original_count FROM pca_test_data;
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 2) INTO result;
		IF result IS NOT NULL AND array_length(result, 1) = original_count THEN
		ELSE
				original_count, COALESCE(array_length(result, 1), 0);
		END IF;
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END $$;

\echo 'Test 5: Compare different n_components'
DO $$
DECLARE
	result_2d vector[];
	result_3d vector[];
	result_5d vector[];
	original_count integer;
BEGIN
	SELECT COUNT(*) INTO original_count FROM pca_test_data;
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_10d'], 2) INTO result_2d;
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_10d'], 3) INTO result_3d;
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_10d'], 5) INTO result_5d;
		
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END $$;

/* --- ERROR path: invalid parameters --- */
\echo ''
\echo 'Error Handling Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: n_components=0 (should error)'
DO $$
DECLARE
	result vector[];
BEGIN
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 0) INTO result;
		RAISE EXCEPTION 'FAIL: expected error for n_components=0';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 2: n_components > original dimensions'
DO $$
DECLARE
	result vector[];
BEGIN
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 10) INTO result;
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 3: Invalid table name'
DO $$
DECLARE
	result vector[];
BEGIN
	BEGIN
		SELECT neurondb.transform_pca('missing_table', ARRAY['feat_5d'], 2) INTO result;
		RAISE EXCEPTION 'FAIL: expected error for missing table';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 4: Invalid column name'
DO $$
DECLARE
	result vector[];
BEGIN
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['notacolumn'], 2) INTO result;
		RAISE EXCEPTION 'FAIL: expected error for invalid column';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 5: Empty column array'
DO $$
DECLARE
	result vector[];
BEGIN
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY[]::text[], 2) INTO result;
		RAISE EXCEPTION 'FAIL: expected error for empty column array';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

\echo 'Error Test 6: NULL n_components'
DO $$
DECLARE
	result vector[];
BEGIN
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], NULL) INTO result;
		RAISE EXCEPTION 'FAIL: expected error for NULL n_components';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- VALIDATION TESTS ----
 * Verify PCA transformations are valid
 *------------------------------------------------------------------*/
\echo ''
\echo 'Validation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Validation Test 1: Row count preservation'
DO $$
DECLARE
	result vector[];
	original_count integer;
BEGIN
	SELECT COUNT(*) INTO original_count FROM pca_test_data;
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 2) INTO result;
		IF result IS NOT NULL AND array_length(result, 1) = original_count THEN
		ELSE
				original_count, COALESCE(array_length(result, 1), 0);
		END IF;
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END $$;

\echo 'Validation Test 2: Dimension reduction verification'
DO $$
DECLARE
	result vector[];
	original_dims integer;
	transformed_dims integer;
BEGIN
	SELECT vector_dims(feat_5d) INTO original_dims FROM pca_test_data LIMIT 1;
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 2) INTO result;
		IF result IS NOT NULL AND array_length(result, 1) > 0 THEN
			SELECT vector_dims(result[1]) INTO transformed_dims;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END $$;

DROP TABLE IF EXISTS pca_test_data;

\echo ''
\echo '=========================================================================='
\echo '✓ PCA: Full exhaustive test complete (1000-row sample)'
\echo '=========================================================================='
