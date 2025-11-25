\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: FP8 Quantization Advance Tests'
\echo '=========================================================================='

-- Test 1: Quantization accuracy comparison
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Quantization accuracy comparison'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH original AS (
	SELECT '[1.0,2.0,3.0,4.0,5.0]'::vector AS v
),
quantized AS (
	SELECT quantize_fp8_e4m3(v) AS q_e4m3, quantize_fp8_e5m2(v) AS q_e5m2
	FROM original
),
dequantized AS (
	SELECT dequantize_fp8(q_e4m3) AS d_e4m3, dequantize_fp8(q_e5m2) AS d_e5m2
	FROM quantized
)
SELECT
	'Accuracy comparison' AS test_name,
	vector_dims(d_e4m3) AS dims_e4m3,
	vector_dims(d_e5m2) AS dims_e5m2
FROM dequantized, original;

-- Test 2: GPU quantization (if available)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: GPU quantization support'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'GPU support check' AS test_name,
	current_setting('neurondb.gpu_enabled', true) AS gpu_enabled;

\echo ''
\echo '✅ Advance FP8 quantization tests completed'

\echo 'Test completed successfully'
