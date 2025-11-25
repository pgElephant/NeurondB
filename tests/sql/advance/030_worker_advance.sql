-- 030_worker_advance.sql
-- Comprehensive advanced test for ALL worker module functions
-- Tests queue operations, tuner operations, defragmentation, LLM worker

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'Worker Module: Exhaustive Worker Operations Coverage'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- QUEUE WORKER OPERATIONS ----
 * Test queue worker with various job types and scenarios
 *------------------------------------------------------------------*/
\echo ''
\echo 'Queue Worker Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Queue worker status check'
SELECT 
	COUNT(*) FILTER (WHERE status = 'pending') AS pending_jobs,
	COUNT(*) FILTER (WHERE status = 'completed') AS completed_jobs,
	COUNT(*) FILTER (WHERE status = 'failed') AS failed_jobs,
	COUNT(*) FILTER (WHERE status = 'processing') AS processing_jobs
FROM neurondb.neurondb_job_queue;

\echo 'Test 2: Queue worker execution (multiple runs)'
SELECT neuranq_run_once() AS run1;
SELECT neuranq_run_once() AS run2;
SELECT neuranq_run_once() AS run3;

\echo 'Test 3: Queue worker with various job types'
-- Check if different job types exist
SELECT 
	job_type,
	COUNT(*) AS job_count,
	COUNT(*) FILTER (WHERE status = 'completed') AS completed,
	COUNT(*) FILTER (WHERE status = 'failed') AS failed
FROM neurondb.neurondb_job_queue
GROUP BY job_type
ORDER BY job_type;

\echo 'Test 4: Queue worker retry logic'
SELECT 
	job_id,
	job_type,
	retry_count,
	max_retries,
	status,
	CASE 
		WHEN retry_count >= max_retries THEN 'Max retries reached'
		ELSE 'Can retry'
	END AS retry_status
FROM neurondb.neurondb_job_queue
WHERE status IN ('pending', 'failed')
ORDER BY retry_count DESC
LIMIT 10;

/*-------------------------------------------------------------------
 * ---- TUNER WORKER OPERATIONS ----
 * Test tuner worker sampling and optimization
 *------------------------------------------------------------------*/
\echo ''
\echo 'Tuner Worker Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 5: Tuner worker configuration'
SELECT 
	name,
	setting,
	unit,
	short_desc
FROM pg_settings
WHERE name LIKE 'neurondb.neuranmon%'
ORDER BY name;

\echo 'Test 6: Tuner worker sampling'
SELECT neuranmon_sample() AS sampled_queries;

\echo 'Test 7: Query metrics collection'
SELECT 
	COUNT(*) AS total_metrics,
	COUNT(DISTINCT query_hash) AS unique_queries,
	AVG(execution_time_ms) AS avg_execution_time,
	MIN(execution_time_ms) AS min_execution_time,
	MAX(execution_time_ms) AS max_execution_time
FROM neurondb.neurondb_query_metrics
WHERE created_at > NOW() - INTERVAL '1 hour';

\echo 'Test 8: Tuner worker multiple samples'
SELECT neuranmon_sample() AS sample1;
SELECT neuranmon_sample() AS sample2;
SELECT neuranmon_sample() AS sample3;

/*-------------------------------------------------------------------
 * ---- DEFRAG WORKER OPERATIONS ----
 * Test defragmentation worker
 *------------------------------------------------------------------*/
\echo ''
\echo 'Defrag Worker Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 9: Defrag worker execution'
SELECT neurandefrag_run() AS defrag_result;

\echo 'Test 10: Defrag worker multiple runs'
SELECT neurandefrag_run() AS defrag_run1;
SELECT neurandefrag_run() AS defrag_run2;

/*-------------------------------------------------------------------
 * ---- LLM WORKER OPERATIONS ----
 * Test LLM worker job processing
 *------------------------------------------------------------------*/
\echo ''
\echo 'LLM Worker Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 11: LLM job status'
SELECT 
	operation,
	status,
	COUNT(*) AS job_count,
	AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) AS avg_duration_seconds
FROM neurondb.neurondb_llm_jobs
GROUP BY operation, status
ORDER BY operation, status;

\echo 'Test 12: LLM job retry statistics'
SELECT 
	operation,
	AVG(retry_count) AS avg_retries,
	MAX(retry_count) AS max_retries,
	COUNT(*) FILTER (WHERE retry_count > 0) AS jobs_with_retries
FROM neurondb.neurondb_llm_jobs
GROUP BY operation
ORDER BY operation;

/*-------------------------------------------------------------------
 * ---- WORKER CONFIGURATION ----
 * Test worker configuration parameters
 *------------------------------------------------------------------*/
\echo ''
\echo 'Worker Configuration Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 13: All worker GUC settings'
SELECT 
	name,
	setting,
	unit,
	short_desc
FROM pg_settings
WHERE name LIKE 'neurondb.neuran%'
ORDER BY name;

\echo 'Test 14: Worker enabled/disabled status'
SELECT 
	name,
	setting,
	CASE 
		WHEN setting = 'on' OR setting = 'true' THEN 'Enabled'
		ELSE 'Disabled'
	END AS status
FROM pg_settings
WHERE name IN (
	'neurondb.neuranq_enabled',
	'neurondb.neuranmon_enabled',
	'neurondb.neurandefrag_enabled',
	'neurondb.neuranllm_enabled'
)
ORDER BY name;

/*-------------------------------------------------------------------
 * ---- WORKER PERFORMANCE METRICS ----
 * Test worker performance and statistics
 *------------------------------------------------------------------*/
\echo ''
\echo 'Worker Performance Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 15: Queue worker processing rate'
SELECT 
	DATE_TRUNC('minute', created_at) AS minute,
	COUNT(*) AS jobs_created,
	COUNT(*) FILTER (WHERE status = 'completed') AS jobs_completed,
	COUNT(*) FILTER (WHERE status = 'failed') AS jobs_failed
FROM neurondb.neurondb_job_queue
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY DATE_TRUNC('minute', created_at)
ORDER BY minute DESC
LIMIT 10;

\echo 'Test 16: Worker execution timing'
-- Measure execution time for workers
DO $$
DECLARE
	start_time timestamp;
	end_time timestamp;
	duration interval;
BEGIN
	start_time := clock_timestamp();
	PERFORM neuranq_run_once();
	end_time := clock_timestamp();
	duration := end_time - start_time;
	RAISE NOTICE 'Queue worker execution time: %', duration;
END$$;

\echo ''
\echo '=========================================================================='
\echo '✓ Worker Module: Full exhaustive code-path test complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




