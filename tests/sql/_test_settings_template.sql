-- Template for reading test settings
-- Copy this block to the beginning of each test file after \set ON_ERROR_STOP on

/* Step 0: Read settings from test_settings table and apply them */
DO $$
DECLARE
	gpu_mode TEXT;
	num_rows_val TEXT;
	gpu_kernels_val TEXT;
BEGIN
	-- Read GPU mode setting and enable/disable GPU accordingly
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	IF gpu_mode = 'gpu' THEN
		PERFORM neurondb_gpu_enable();
	ELSE
		-- GPU disabled or CPU mode - ensure GPU is off
		PERFORM set_config('neurondb.gpu_enabled', 'off', false);
	END IF;
	
	-- Read other settings if needed
	SELECT setting_value INTO num_rows_val FROM test_settings WHERE setting_key = 'num_rows';
	SELECT setting_value INTO gpu_kernels_val FROM test_settings WHERE setting_key = 'gpu_kernels';
	
	-- Additional settings can be read and applied here as needed
END $$;

