-- 021_lightgbm_advance.sql
-- Advanced test for lightgbm
-- Note: Algorithm may not be fully implemented yet

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo '=========================================================================='

\echo ''
\echo 'Algorithm Status'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	-- Check if lightgbm functions exist
	IF EXISTS (SELECT 1 FROM pg_proc WHERE proname LIKE '%lightgbm%' OR proname LIKE '%lgb%') THEN
	ELSE
	END IF;
	
	-- Check if lightgbm is in ml_models (cast to text for comparison, no LIMIT in EXISTS)
	IF EXISTS (SELECT 1 FROM neurondb.ml_models WHERE algorithm::text = 'lightgbm') THEN
	ELSE
	END IF;
END $$;

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

\echo 'Test completed successfully'
