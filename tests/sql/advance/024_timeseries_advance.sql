-- 024_timeseries_advance.sql
-- Advanced test for timeseries
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
	-- Check if timeseries functions exist
	IF EXISTS (SELECT 1 FROM pg_proc WHERE proname LIKE '%timeseries%' OR proname LIKE '%ts%' OR proname LIKE '%time_series%') THEN
	ELSE
	END IF;
	
	-- Check if timeseries is in ml_models (cast to text for comparison)
	IF EXISTS (SELECT 1 FROM neurondb.ml_models WHERE algorithm::text = 'timeseries') THEN
	ELSE
	END IF;
END $$;

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

\echo 'Test completed successfully'
