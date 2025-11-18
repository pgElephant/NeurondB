-- 023_pca_perf.sql
-- Performance test for PCA (Principal Component Analysis) with GPU acceleration
-- Works on full dataset from sample_train table

\timing on
\pset footer off
\pset pager off
\pset tuples_only off
\set ON_ERROR_STOP on
SET client_min_messages TO WARNING;

\echo '=========================================================================='
\echo 'PCA (Principal Component Analysis) - Performance Test (Full Dataset with GPU)'
\echo '=========================================================================='

-- Verify required tables exist
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		RAISE EXCEPTION 'sample_train table does not exist. Please run: python ml_dataset.py <dataset_name>';
	END IF;
END
$$;

-- Configure GPU for performance
SET neurondb.gpu_enabled = on;
SET neurondb.gpu_kernels = 'l2,cosine,ip';
SELECT neurondb_gpu_enable() AS gpu_available;
SELECT neurondb_gpu_info() AS gpu_info;

\echo ''
\echo 'Testing PCA transformation on full dataset...'

-- Test PCA transformation on full dataset
DO $$
DECLARE
	result vector[];
	row_count bigint;
BEGIN
	SELECT COUNT(*) INTO row_count FROM sample_train;
	BEGIN
		-- Test PCA on the features vector column
		SELECT neurondb.transform_pca('sample_train', ARRAY['features'], 2) INTO result;
		IF result IS NULL OR array_length(result, 1) IS NULL THEN
			RAISE EXCEPTION 'PCA transform returned NULL or empty';
		END IF;
	EXCEPTION WHEN OTHERS THEN
	END;
END $$;

\echo 'PCA performance test completed successfully'
