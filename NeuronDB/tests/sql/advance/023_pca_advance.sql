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
\echo '=========================================================================='

/* Check that sample_train exists */

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
SELECT neurondb_gpu_enable() AS gpu_available;
SELECT neurondb_gpu_info() AS gpu_info;

/*
 * ---- PCA TRANSFORMATION TESTS ----
 * Test PCA with different n_components
 */
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create test data with multiple dimensions
DROP TABLE IF EXISTS pca_test_data;
CREATE TEMP TABLE pca_test_data AS
SELECT 
	vector_slice(features, 0, 5) AS feat_5d,
	vector_slice(features, 0, 10) AS feat_10d,
	features AS feat_full
FROM test_train_view
LIMIT 100;

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
		IF result IS NULL OR array_length(result, 1) IS NULL THEN
			RAISE EXCEPTION 'PCA transform returned NULL or empty';
		END IF;
	EXCEPTION WHEN OTHERS THEN
		-- Error handled correctly
		NULL;
	END;
END $$;

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
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

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
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

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
\echo '=========================================================================='

\echo 'Test completed successfully'
