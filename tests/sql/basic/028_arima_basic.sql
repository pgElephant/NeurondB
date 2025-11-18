/*-------------------------------------------------------------------------
 *
 * 028_arima_basic.sql
 *    ARIMA Time Series basic test
 *
 *    Basic functionality test with training, prediction, and evaluation
 *
 *-------------------------------------------------------------------------*/

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

/* Step 1: Create time series test data */
\echo 'Step 1: Creating time series test data...'

DROP TABLE IF EXISTS ts_data;
CREATE TABLE ts_data (
    time_idx INTEGER,
    value FLOAT8
);

-- Generate synthetic time series data with trend and seasonality
INSERT INTO ts_data (time_idx, value)
SELECT
    i as time_idx,
    10.0 + 0.5 * i + 3.0 * sin(2 * pi() * i / 12) + random() * 2.0 as value
FROM generate_series(1, 200) i;

SELECT
    COUNT(*) as samples,
    MIN(time_idx) as min_time,
    MAX(time_idx) as max_time,
    ROUND(AVG(value)::numeric, 3) as avg_value,
    ROUND(STDDEV(value)::numeric, 3) as std_value
FROM ts_data;

/* Step 2: Train ARIMA model */
\echo 'Step 2: Training ARIMA model...'

CREATE TEMP TABLE arima_model AS
SELECT train_arima('ts_data', 'time_idx', 'value', 1, 0, 1) as model_id;

SELECT * FROM arima_model;

/* Step 3: Test predictions */
\echo 'Step 3: Testing ARIMA predictions...'

-- Forecast next 5 time steps
CREATE TEMP TABLE arima_forecast AS
SELECT
    time_idx + 5 as forecast_time,
    predict_arima((SELECT model_id FROM arima_model), time_idx + 5) as predicted_value
FROM ts_data
ORDER BY time_idx DESC
LIMIT 5;

SELECT * FROM arima_forecast;

/* Step 4: Evaluate model */
\echo 'Step 4: Evaluating ARIMA model...'

CREATE TEMP TABLE arima_metrics AS
SELECT evaluate_arima_by_model_id(
    (SELECT model_id FROM arima_model),
    'ts_data',
    'time_idx',
    'value',
    5  -- forecast horizon
) as metrics;

SELECT
    'MSE' as metric, ROUND((metrics->>'mse')::numeric, 6)::text as value
FROM arima_metrics
UNION ALL
SELECT 'MAE', ROUND((metrics->>'mae')::numeric, 6)::text
FROM arima_metrics
UNION ALL
SELECT 'RMSE', ROUND((metrics->>'rmse')::numeric, 6)::text
FROM arima_metrics
ORDER BY metric;

/* Step 5: Summary */
\echo 'Step 5: Test summary...'

SELECT
    (SELECT model_id FROM arima_model) as model_id,
    (SELECT COUNT(*) FROM ts_data) as training_samples,
    5 as forecast_horizon,
    (SELECT ROUND((metrics->>'rmse')::numeric, 6) FROM arima_metrics) as rmse;

/* Cleanup */
DROP TABLE IF EXISTS ts_data;

/* Success message */
\echo 'ARIMA basic test completed successfully'
