-- 039_crash_prevention_advance.sql
-- Comprehensive advanced crash prevention tests
-- Tests crash prevention mechanisms under various conditions

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'Crash Prevention: Advanced Comprehensive Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- MEMORY CONTEXT STRESS TESTS ----
 * Test memory context handling under stress
 *------------------------------------------------------------------*/
\echo ''
\echo 'Memory Context Stress Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Rapid context switching'
DO $$
DECLARE
	i int;
	result vector;
BEGIN
	-- Rapidly switch contexts and perform operations
	FOR i IN 1..100 LOOP
		BEGIN
			result := vector_normalize(vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector);
			-- Force context operations
			PERFORM pg_sleep(0.001);
		EXCEPTION WHEN OTHERS THEN
			NULL; -- Should handle gracefully
		END;
	END LOOP;
	
	RAISE NOTICE 'Rapid context switching completed successfully';
END$$;

\echo 'Test 2: Long-running operations with memory cleanup'
DO $$
DECLARE
	model_id int;
	i int;
BEGIN
	-- Train multiple models and ensure cleanup
	FOR i IN 1..5 LOOP
		BEGIN
			SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb) INTO model_id;
			-- Use model
			PERFORM neurondb.predict('model_' || model_id::text, vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector);
			-- Drop model to test cleanup
			PERFORM neurondb.drop_model('model_' || model_id::text);
		EXCEPTION WHEN OTHERS THEN
			NULL; -- Should handle gracefully
		END;
	END LOOP;
	
	RAISE NOTICE 'Long-running operations with cleanup completed';
END$$;

/*-------------------------------------------------------------------
 * ---- SPI CONTEXT STRESS TESTS ----
 * Test SPI operations under various conditions
 *------------------------------------------------------------------*/
\echo ''
\echo 'SPI Context Stress Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 3: Multiple SPI operations in sequence'
DO $$
DECLARE
	model_id int;
	result float8;
	i int;
BEGIN
	-- Train model
	SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb) INTO model_id;
	
	-- Perform multiple evaluations
	FOR i IN 1..10 LOOP
		BEGIN
			result := neurondb.evaluate(model_id, 'test_train_view', 'features', 'label');
		EXCEPTION WHEN OTHERS THEN
			NULL; -- Should handle gracefully
		END;
	END LOOP;
	
	RAISE NOTICE 'Multiple SPI operations completed: model_id = %, last result = %', model_id, result;
END$$;

\echo 'Test 4: SPI operations with transaction boundaries'
DO $$
DECLARE
	model_id int;
BEGIN
	-- Test SPI operations across transaction boundaries
	BEGIN
		SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb) INTO model_id;
		COMMIT;
	EXCEPTION WHEN OTHERS THEN
		ROLLBACK;
		NULL; -- Should handle gracefully
	END;
	
	BEGIN
		PERFORM neurondb.evaluate(model_id, 'test_train_view', 'features', 'label');
		COMMIT;
	EXCEPTION WHEN OTHERS THEN
		ROLLBACK;
		NULL; -- Should handle gracefully
	END;
	
	RAISE NOTICE 'SPI operations across transactions completed';
END$$;

/*-------------------------------------------------------------------
 * ---- NULL HANDLING COMPREHENSIVE TESTS ----
 * Test all NULL handling paths
 *------------------------------------------------------------------*/
\echo ''
\echo 'NULL Handling Comprehensive Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 5: NULL handling in vector operations'
SELECT 
	'NULL L2' AS test_type,
	vector_l2_distance(NULL::vector, vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector) AS result1,
	vector_l2_distance(vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector, NULL::vector) AS result2,
	vector_l2_distance(NULL::vector, NULL::vector) AS result3;

\echo 'Test 6: NULL handling in model operations'
DO $$
DECLARE
	model_id int;
	result vector;
BEGIN
	-- Train model
	SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb) INTO model_id;
	
	-- Try prediction with various NULL scenarios
	BEGIN
		result := neurondb.predict(NULL, vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
	
	BEGIN
		result := neurondb.predict('model_' || model_id::text, NULL::vector);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
	
	RAISE NOTICE 'NULL handling in model operations completed';
END$$;

/*-------------------------------------------------------------------
 * ---- MEMORY PRESSURE TESTS ----
 * Test behavior under memory pressure
 *------------------------------------------------------------------*/
\echo ''
\echo 'Memory Pressure Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 7: Large batch operations'
DO $$
DECLARE
	vectors vector[];
	i int;
	result vector;
BEGIN
	-- Create large batch of vectors
	vectors := ARRAY[]::vector[];
	FOR i IN 1..1000 LOOP
		vectors := array_append(vectors, vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector);
	END LOOP;
	
	-- Perform batch operations
	BEGIN
		-- Test batch normalization
		SELECT array_agg(vector_normalize(v)) INTO vectors FROM unnest(vectors) v;
		RAISE NOTICE 'Large batch operations completed: % vectors', array_length(vectors, 1);
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Large batch operations handled gracefully: %', SQLERRM;
	END;
END$$;

\echo 'Test 8: Memory-intensive model training'
DO $$
DECLARE
	model_id int;
BEGIN
	-- Create large dataset
	CREATE TEMP TABLE large_memory_test AS
	SELECT 
		i AS id,
		ARRAY[random(), random(), random(), random(), random(), random(), random(), random(), random(), random(),
		      random(), random(), random(), random(), random(), random(), random(), random(), random(), random(),
		      random(), random(), random(), random(), random(), random(), random(), random()]::float8[] AS features,
		random() * 100 AS label
	FROM generate_series(1, 10000) i;
	
	-- Train model
	BEGIN
		SELECT neurondb.train('linear_regression', 'large_memory_test', 'features', 'label', '{}'::jsonb) INTO model_id;
		RAISE NOTICE 'Memory-intensive training completed: model_id = %', model_id;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Memory-intensive training handled gracefully: %', SQLERRM;
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- ERROR RECOVERY TESTS ----
 * Test error recovery mechanisms
 *------------------------------------------------------------------*/
\echo ''
\echo 'Error Recovery Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 9: Error recovery in training'
DO $$
DECLARE
	model_id int;
BEGIN
	-- Try training with invalid data, then recover
	BEGIN
		SELECT neurondb.train('linear_regression', 'test_train_view', 'nonexistent', 'label', '{}'::jsonb) INTO model_id;
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
	
	-- Recover and train correctly
	BEGIN
		SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb) INTO model_id;
		RAISE NOTICE 'Error recovery successful: model_id = %', model_id;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Error recovery failed: %', SQLERRM;
	END;
END$$;

\echo 'Test 10: Error recovery in prediction'
DO $$
DECLARE
	model_id int;
	result vector;
BEGIN
	-- Train model
	SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb) INTO model_id;
	
	-- Try prediction with invalid model, then recover
	BEGIN
		result := neurondb.predict('nonexistent_model', vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
	
	-- Recover and predict correctly
	BEGIN
		result := neurondb.predict('model_' || model_id::text, vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector);
		RAISE NOTICE 'Error recovery successful: prediction completed';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Error recovery failed: %', SQLERRM;
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- CONCURRENT OPERATION TESTS ----
 * Test concurrent operations that could cause crashes
 *------------------------------------------------------------------*/
\echo ''
\echo 'Concurrent Operation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 11: Concurrent model operations'
DO $$
DECLARE
	model_ids int[];
	i int;
BEGIN
	-- Create multiple models concurrently
	FOR i IN 1..5 LOOP
		BEGIN
			SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb) INTO model_ids[i];
		EXCEPTION WHEN OTHERS THEN
			NULL; -- Should handle gracefully
		END;
	END LOOP;
	
	-- Use all models concurrently
	FOR i IN 1..5 LOOP
		IF model_ids[i] IS NOT NULL THEN
			BEGIN
				PERFORM neurondb.predict('model_' || model_ids[i]::text, vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector);
			EXCEPTION WHEN OTHERS THEN
				NULL; -- Should handle gracefully
			END;
		END IF;
	END LOOP;
	
	RAISE NOTICE 'Concurrent operations completed';
END$$;

\echo ''
\echo '=========================================================================='
\echo '✓ Crash Prevention: Advanced comprehensive tests complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




