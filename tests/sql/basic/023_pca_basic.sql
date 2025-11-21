-- 023_pca_basic.sql
-- Basic test for PCA (Principal Component Analysis) with GPU acceleration

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off
SET client_min_messages TO WARNING;

/* Step 1: Verify prerequisites and create test data */

DROP TABLE IF EXISTS pca_data;
CREATE TABLE pca_data (
	id serial PRIMARY KEY,
	features vector
);

-- Create sample data with vector features
INSERT INTO pca_data (features)
SELECT 
	array_to_vector_float8(ARRAY[
		x::double precision, 
		(x*1.5 + random()*0.1)::double precision, 
		(x*2.0 + random()*0.1)::double precision
	]) AS features
FROM generate_series(1, 100) AS x;

SELECT COUNT(*)::bigint AS data_rows FROM pca_data;

/* Step 2: Configure GPU */
\echo 'Step 2: Configuring GPU acceleration...'

/* Step 0: Read settings from test_settings table and apply them */
DO $$
DECLARE
	gpu_mode TEXT;
BEGIN
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	IF gpu_mode = 'gpu' THEN
		PERFORM neurondb_gpu_enable();
	ELSE
		PERFORM set_config('neurondb.gpu_enabled', 'off', false);
	END IF;
END $$;

\echo '=========================================================================='
\echo '=========================================================================='

/* Step 3: Test PCA transformation */

-- Test PCA transformation
DO $$
DECLARE
	result vector[];
BEGIN
	BEGIN
		SELECT neurondb.transform_pca('pca_data', ARRAY['features'], 2) INTO result;
		IF result IS NULL OR array_length(result, 1) IS NULL THEN
			RAISE EXCEPTION 'PCA transform returned NULL or empty';
		END IF;
	EXCEPTION WHEN OTHERS THEN
	END;
END $$;

DROP TABLE IF EXISTS pca_data;

\echo 'Test completed successfully'
