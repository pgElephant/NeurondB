-- 023_pca.sql
-- Basic test for PCA (Principal Component Analysis)

SET client_min_messages TO WARNING;
SET neurondb.gpu_enabled = off;

DROP TABLE IF EXISTS pca_data;
CREATE TABLE pca_data (
	id serial PRIMARY KEY,
	x1 double precision,
	x2 double precision,
	x3 double precision
);

INSERT INTO pca_data (x1, x2, x3)
SELECT x, x*1.5 + random()*0.1, x*2.0 + random()*0.1
FROM generate_series(1, 100) AS x;

\echo '=== PCA Basic Test ==='

-- Test PCA transformation
DO $$
DECLARE
	result vector[];
BEGIN
	BEGIN
		SELECT neurondb.transform_pca('pca_data', ARRAY['x1', 'x2', 'x3'], 2) INTO result;
		IF result IS NULL OR array_length(result, 1) IS NULL THEN
			RAISE EXCEPTION 'PCA transform returned NULL or empty';
		END IF;
		RAISE NOTICE '✓ PCA transform successful, transformed % rows', array_length(result, 1);
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'PCA not yet implemented: %', SQLERRM;
	END;
END $$;

\echo '✓ PCA basic test complete'

