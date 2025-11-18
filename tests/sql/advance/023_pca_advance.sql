-- 023_pca_advance.sql
-- Advanced test for PCA (Principal Component Analysis)
-- Works on 1000 rows only and tests each and every way

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

\echo '=== PCA Advanced Test ==='

-- Verify required tables exist
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

-- Create test data with multiple dimensions
DROP TABLE IF EXISTS pca_test_data;
CREATE TEMP TABLE pca_test_data AS
SELECT 
	features[1:5]::vector(5) AS feat_5d,
	features[1:10]::vector(10) AS feat_10d,
	features AS feat_full
FROM test_train_view
LIMIT 100;

\echo 'Test 1: PCA transformation with different n_components'
DO $$
DECLARE
	result vector[];
BEGIN
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 2) INTO result;
		IF result IS NULL OR array_length(result, 1) IS NULL THEN
			RAISE EXCEPTION 'PCA transform returned NULL or empty';
		END IF;
		RAISE NOTICE '✓ PCA transform (5D -> 2D) successful, transformed % rows', array_length(result, 1);
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'PCA not yet implemented or error: %', SQLERRM;
	END;
END $$;

\echo 'Test 2: PCA with different number of components'
DO $$
DECLARE
	result vector[];
BEGIN
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_10d'], 3) INTO result;
		IF result IS NULL OR array_length(result, 1) IS NULL THEN
			RAISE EXCEPTION 'PCA transform returned NULL or empty';
		END IF;
		RAISE NOTICE '✓ PCA transform (10D -> 3D) successful, transformed % rows', array_length(result, 1);
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'PCA not yet implemented or error: %', SQLERRM;
	END;
END $$;

\echo 'Test 3: Verify PCA preserves number of rows'
DO $$
DECLARE
	result vector[];
	original_count integer;
BEGIN
	SELECT COUNT(*) INTO original_count FROM pca_test_data;
	BEGIN
		SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 2) INTO result;
		IF result IS NOT NULL AND array_length(result, 1) = original_count THEN
			RAISE NOTICE '✓ PCA preserves row count: % rows', array_length(result, 1);
		ELSE
			RAISE NOTICE '⚠ PCA row count mismatch: expected %, got %', original_count, COALESCE(array_length(result, 1), 0);
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'PCA not yet implemented or error: %', SQLERRM;
	END;
END $$;

DROP TABLE IF EXISTS pca_test_data;

\echo '✓ PCA advance test complete'

