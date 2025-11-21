-- Template for reading test settings
-- Copy this block to the beginning of each test file after \set ON_ERROR_STOP on
-- Note: GPU settings are configured by test runner via ALTER SYSTEM SET and reload
-- This block only verifies the settings, it does not set them

/* Step 0: Read settings from test_settings table and verify GPU configuration */
DO $$
DECLARE
	gpu_mode TEXT;
	current_gpu_enabled TEXT;
	current_gpu_kernels TEXT;
	num_rows_val TEXT;
BEGIN
	-- Read GPU mode setting from test_settings
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	
	-- Verify GPU configuration matches test_settings
	SELECT current_setting('neurondb.gpu_enabled', true) INTO current_gpu_enabled;
	
	IF gpu_mode = 'gpu' THEN
		-- Verify GPU is enabled (should be set by test runner)
		IF current_gpu_enabled != 'on' THEN
			RAISE WARNING 'GPU mode expected but neurondb.gpu_enabled = % (expected: on)', current_gpu_enabled;
		END IF;
	ELSE
		-- Verify GPU is disabled (should be set by test runner)
		IF current_gpu_enabled != 'off' THEN
			RAISE WARNING 'CPU mode expected but neurondb.gpu_enabled = % (expected: off)', current_gpu_enabled;
		END IF;
	END IF;
	
	-- Read other settings if needed
	SELECT setting_value INTO num_rows_val FROM test_settings WHERE setting_key = 'num_rows';
	SELECT current_setting('neurondb.gpu_kernels', true) INTO current_gpu_kernels;
	
	-- Additional settings can be read and verified here as needed
END $$;

