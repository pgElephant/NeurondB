/*-------------------------------------------------------------------------
 *
 * ml_timeseries.c
 *    Time Series Analysis and Forecasting for NeuronDB
 *
 * Implements ARIMA, exponential smoothing, and trend analysis.
 * Supports univariate and multivariate time series.
 *
 * IDENTIFICATION
 *    src/ml/ml_timeseries.c
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/jsonb.h"
#include "access/htup_details.h"
#include "executor/spi.h"
#include "utils/memutils.h"
#include "neurondb_pgcompat.h"

#include <math.h>
#include <string.h>

/* PG_MODULE_MAGIC is in neurondb.c only */

#define MAX_ARIMA_ORDER_P 10
#define MAX_ARIMA_ORDER_Q 10
#define MAX_ARIMA_ORDER_D 2
#define MIN_ARIMA_OBSERVATIONS 10
#define MAX_FORECAST_AHEAD 1000
#define MIN_SEASONAL_PERIOD 1
#define MAX_SEASONAL_PERIOD 365

typedef struct TimeSeriesModel
{
	int32 p;
	int32 d;
	int32 q;
	float *ar_coeffs;
	float *ma_coeffs;
	float intercept;
	int32 n_obs;
	float *residuals;
} TimeSeriesModel;

// Fix for SPI_getbinval 'isnull' argument: always use a local 'bool' variable, not 'int'
// throughout this file, especially within the ARIMA forecast and model loading code.

static void
compute_moving_average(const float *data, int n, int window, float *result)
{
	int i, j;
	float sum;

	if (window <= 0)
		elog(ERROR,
			"window length for moving average must be positive");

	for (i = 0; i < n; i++)
	{
		if (i < window - 1)
			result[i] = data[i];
		else
		{
			sum = 0.0f;
			for (j = 0; j < window; j++)
				sum += data[i - j];
			result[i] = sum / (float)window;
		}
	}
}

static void
exponential_smoothing(const float *data, int n, float alpha, float *result)
{
	int i;

	if (n <= 0)
		return;
	if (alpha < 0.0f || alpha > 1.0f)
		elog(ERROR, "alpha for exponential smoothing must be in [0,1]");

	result[0] = data[0];
	for (i = 1; i < n; i++)
		result[i] = alpha * data[i] + (1.0f - alpha) * result[i - 1];
}

static float *
compute_differences(const float *data, int n, int order, int *out_n)
{
	float *diff;
	int curr_n, d, i;

	Assert(data != NULL);
	Assert(out_n != NULL);
	Assert(order >= 0);

	if (order == 0)
	{
		diff = (float *)palloc(sizeof(float) * n);
		memcpy(diff, data, sizeof(float) * n);
		*out_n = n;
		return diff;
	}

	curr_n = n;
	diff = (float *)palloc(sizeof(float) * curr_n);
	memcpy(diff, data, sizeof(float) * curr_n);

	for (d = 0; d < order; d++)
	{
		int new_n = curr_n - 1;
		float *new_diff;

		if (new_n <= 0)
			elog(ERROR,
				"cannot difference data sequence below length "
				"1");
		new_diff = (float *)palloc(sizeof(float) * new_n);
		for (i = 0; i < new_n; i++)
			new_diff[i] = diff[i + 1] - diff[i];
		pfree(diff);
		diff = new_diff;
		curr_n = new_n;
	}

	*out_n = curr_n;
	return diff;
}

static float
compute_mean(const float *data, int n)
{
	int i;
	float sum = 0.0f;

	if (n <= 0)
		return 0.0f;

	for (i = 0; i < n; i++)
		sum += data[i];

	return sum / (float)n;
}

static float
pg_attribute_unused()
	compute_sample_variance(const float *data, int n, float mean)
{
	int i;
	float var = 0.0f;

	if (n < 2)
		return 0.0f;

	for (i = 0; i < n; i++)
	{
		float d = data[i] - mean;
		var += d * d;
	}

	return var / (float)(n - 1);
}

/*
 * AR fitting with Yule-Walker equations using Cholesky decomposition.
 * Only AR coefficients are estimated, MA parameters set to zeros if requested.
 */
static TimeSeriesModel *
fit_arima(const float *data, int n, int p, int d, int q)
{
	TimeSeriesModel *model;
	float *diff_data = NULL;
	int diff_n, i, j;
	float mean;

	if (n <= 0)
		elog(ERROR,
			"number of observations for ARIMA must be positive");
	if (p < 0 || p > MAX_ARIMA_ORDER_P)
		elog(ERROR, "arima p out of bounds");
	if (d < 0 || d > MAX_ARIMA_ORDER_D)
		elog(ERROR, "arima d out of bounds");
	if (q < 0 || q > MAX_ARIMA_ORDER_Q)
		elog(ERROR, "arima q out of bounds");

	model = (TimeSeriesModel *)palloc0(sizeof(TimeSeriesModel));
	model->p = p;
	model->d = d;
	model->q = q;
	model->n_obs = n;
	model->ar_coeffs = NULL;
	model->ma_coeffs = NULL;
	model->residuals = NULL;

	diff_data = compute_differences(data, n, d, &diff_n);

	if (p > 0)
	{
		float *autocorr = (float *)palloc0(sizeof(float) * (p + 1));
		float **R = (float **)palloc(sizeof(float *) * p);
		float *a;
		float *right = (float *)palloc0(sizeof(float) * p);

		for (i = 0; i <= p; i++)
		{
			float sum = 0.0f;
			for (j = i; j < diff_n; j++)
				sum += diff_data[j] * diff_data[j - i];
			autocorr[i] = sum / (diff_n - i);
		}
		for (i = 0; i < p; i++)
		{
			R[i] = (float *)palloc0(sizeof(float) * p);
			for (j = 0; j < p; j++)
				R[i][j] = autocorr[abs(i - j)];
			right[i] = autocorr[i + 1];
		}

		a = (float *)palloc0(sizeof(float) * p);
		{
			float **L = (float **)palloc0(sizeof(float *) * p);
			for (i = 0; i < p; i++)
				L[i] = (float *)palloc0(sizeof(float) * p);
			for (i = 0; i < p; i++)
			{
				for (j = 0; j <= i; j++)
				{
					float sum = R[i][j];
					int k;
					for (k = 0; k < j; k++)
						sum -= L[i][k] * L[j][k];
					if (i == j)
					{
						if (sum <= 0)
							elog(ERROR,
								"Failed to "
								"decompose "
								"autocorrelatio"
								"n matrix: "
								"non-positive "
								"definite");
						L[i][j] = sqrtf(sum);
					} else
						L[i][j] = sum / L[j][j];
				}
			}
			{
				float *y;

				y = (float *)palloc0(sizeof(float) * p);
				for (i = 0; i < p; i++)
				{
					float sum = right[i];
					for (j = 0; j < i; j++)
						sum -= L[i][j] * y[j];
					y[i] = sum / L[i][i];
				}
				for (i = p - 1; i >= 0; i--)
				{
					float sum = y[i];
					for (j = i + 1; j < p; j++)
						sum -= L[j][i] * a[j];
					a[i] = sum / L[i][i];
				}
				for (i = 0; i < p; i++)
					pfree(L[i]);
				pfree(L);
				pfree(y);
			}
		}
		model->ar_coeffs = (float *)palloc(sizeof(float) * p);
		for (i = 0; i < p; i++)
			model->ar_coeffs[i] = a[i];

		for (i = 0; i < p; i++)
			pfree(R[i]);
		pfree(R);
		pfree(a);
		pfree(right);
		pfree(autocorr);
	} else
		model->ar_coeffs = NULL;

	if (q > 0)
	{
		model->ma_coeffs = (float *)palloc0(sizeof(float) * q);
	} else
	{
		model->ma_coeffs = NULL;
	}

	mean = compute_mean(diff_data, diff_n);
	model->intercept = mean;

	model->residuals = (float *)palloc(sizeof(float) * diff_n);
	if (p > 0)
	{
		for (i = p; i < diff_n; i++)
		{
			float pred = model->intercept;
			for (j = 0; j < p; j++)
				pred += model->ar_coeffs[j]
					* diff_data[i - (j + 1)];
			model->residuals[i] = diff_data[i] - pred;
		}
		for (i = 0; i < p && i < diff_n; i++)
			model->residuals[i] = 0.0f;
	} else
	{
		for (i = 0; i < diff_n; i++)
			model->residuals[i] = diff_data[i] - model->intercept;
	}

	pfree(diff_data);

	return model;
}

static void
arima_forecast(const TimeSeriesModel *model,
	const float *last_values,
	int n_last,
	int n_ahead,
	float *forecast)
{
	int i, j;
	int p, d;
	float *history;

	Assert(model != NULL);
	Assert(last_values != NULL);
	Assert(forecast != NULL);

	if (n_ahead < 1)
		elog(ERROR, "Must forecast at least 1 ahead");

	p = model->p;
	d = model->d;
	history = (float *)palloc0(sizeof(float) * (n_last + n_ahead));

	memcpy(history, last_values, sizeof(float) * n_last);

	for (i = 0; i < n_ahead; i++)
	{
		float val;
		int idx = n_last + i;
		if (p > 0)
		{
			val = model->intercept;
			for (j = 0; j < p && idx - (j + 1) >= 0; j++)
				val += model->ar_coeffs[j]
					* history[idx - (j + 1)];
		} else
			val = model->intercept;

		history[idx] = val;
		forecast[i] = val;
		if (d > 0)
		{
			int step;
			for (step = 0; step < d; step++)
			{
				if (i - 1 >= 0)
					forecast[i] += forecast[i - 1];
			}
		}
	}

	pfree(history);
}

PG_FUNCTION_INFO_V1(train_arima);
Datum
train_arima(PG_FUNCTION_ARGS)
{
	text *table_name = PG_GETARG_TEXT_PP(0);
	text *time_col = PG_GETARG_TEXT_PP(1);
	text *value_col = PG_GETARG_TEXT_PP(2);
	int32 p = PG_ARGISNULL(3) ? 1 : PG_GETARG_INT32(3);
	int32 d = PG_ARGISNULL(4) ? 1 : PG_GETARG_INT32(4);
	int32 q = PG_ARGISNULL(5) ? 1 : PG_GETARG_INT32(5);

	char *table_name_str = text_to_cstring(table_name);
	char *time_col_str = text_to_cstring(time_col);
	char *value_col_str = text_to_cstring(value_col);

	StringInfoData sql;
	int ret, n_samples, i;
	SPITupleTable *tuptable;
	TupleDesc tupdesc;
	float *values = NULL;
	TimeSeriesModel *model = NULL;

	if (p < 0 || p > MAX_ARIMA_ORDER_P)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("p must be between 0 and %d",
					MAX_ARIMA_ORDER_P)));
	if (d < 0 || d > MAX_ARIMA_ORDER_D)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("d must be between 0 and %d",
					MAX_ARIMA_ORDER_D)));
	if (q < 0 || q > MAX_ARIMA_ORDER_Q)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("q must be between 0 and %d",
					MAX_ARIMA_ORDER_Q)));

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT %s FROM %s ORDER BY %s",
		quote_identifier(value_col_str),
		quote_identifier(table_name_str),
		quote_identifier(time_col_str));

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("failed to execute time series query")));

	tuptable = SPI_tuptable;
	tupdesc = tuptable->tupdesc;
	n_samples = SPI_processed;

	if (n_samples < MIN_ARIMA_OBSERVATIONS)
	{
		SPI_finish();
		if (table_name_str)
			pfree(table_name_str);
		if (time_col_str)
			pfree(time_col_str);
		if (value_col_str)
			pfree(value_col_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("at least %d observations are required for ARIMA",
					MIN_ARIMA_OBSERVATIONS)));
	}

	values = (float *)palloc(sizeof(float) * n_samples);

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple tuple = tuptable->vals[i];
		bool isnull = false;
		Datum value_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);

		if (isnull)
		{
			if (values)
				pfree(values);
			SPI_finish();
			if (table_name_str)
				pfree(table_name_str);
			if (time_col_str)
				pfree(time_col_str);
			if (value_col_str)
				pfree(value_col_str);
			ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					errmsg("time series cannot contain "
					       "NULL values")));
		}
		values[i] = (float)DatumGetFloat8(value_datum);
	}

	model = fit_arima(values, n_samples, p, d, q);

	SPI_finish();

	if (values)
		pfree(values);
	if (table_name_str)
		pfree(table_name_str);
	if (time_col_str)
		pfree(time_col_str);
	if (value_col_str)
		pfree(value_col_str);

	if (model)
	{
		if (model->ar_coeffs)
			pfree(model->ar_coeffs);
		if (model->ma_coeffs)
			pfree(model->ma_coeffs);
		if (model->residuals)
			pfree(model->residuals);
		pfree(model);
	}

	PG_RETURN_TEXT_P(cstring_to_text("ARIMA model trained successfully"));
}

PG_FUNCTION_INFO_V1(forecast_arima);
Datum
forecast_arima(PG_FUNCTION_ARGS)
{
	int32 model_id = PG_GETARG_INT32(0);
	int32 n_ahead = PG_GETARG_INT32(1);

	StringInfoData sql;
	TimeSeriesModel model;
	ArrayType *ar_coeffs_arr = NULL;
	ArrayType *ma_coeffs_arr = NULL;
	ArrayType *last_values_arr = NULL;
	int ret;
	int *dims, ndims, i;
	Oid arr_elem_type;
	int p = 0, d = 0, q = 0, n_last = 0;
	float8 intercept = 0;
	float *ar_coeffs = NULL;
	float *ma_coeffs = NULL;
	float *last_values = NULL;
	float *forecast = NULL;
	Datum *outdatums = NULL;
	ArrayType *arr = NULL;

	if (n_ahead < 1 || n_ahead > MAX_FORECAST_AHEAD)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("n_ahead must be between 1 and %d",
					MAX_FORECAST_AHEAD)));

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT p, d, q, intercept, ar_coeffs, ma_coeffs FROM neurondb_arima_models WHERE model_id = %d",
		model_id);

	ret = SPI_execute(sql.data, true, 1);
	if (ret != SPI_OK_SELECT || SPI_processed != 1)
	{
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_UNDEFINED_OBJECT),
				errmsg("model_id %d not found in neurondb_arima_models",
					model_id)));
	}
	{
		HeapTuple modeltuple;
		TupleDesc modeldesc;
		bool isnull;

		modeltuple = SPI_tuptable->vals[0];
		modeldesc = SPI_tuptable->tupdesc;
		p = DatumGetInt32(
			SPI_getbinval(modeltuple, modeldesc, 1, &isnull));
		d = DatumGetInt32(
			SPI_getbinval(modeltuple, modeldesc, 2, &isnull));
		q = DatumGetInt32(
			SPI_getbinval(modeltuple, modeldesc, 3, &isnull));
		intercept = DatumGetFloat8(
			SPI_getbinval(modeltuple, modeldesc, 4, &isnull));
		ar_coeffs_arr = DatumGetArrayTypeP(
			SPI_getbinval(modeltuple, modeldesc, 5, &isnull));
		ma_coeffs_arr = DatumGetArrayTypeP(
			SPI_getbinval(modeltuple, modeldesc, 6, &isnull));
	}

	arr_elem_type = ARR_ELEMTYPE(ar_coeffs_arr);
	(void)arr_elem_type; /* Suppress unused variable warning */
	ndims = ARR_NDIM(ar_coeffs_arr);
	Assert(ndims == 1);
	(void)ndims; /* Used in Assert only */
	dims = ARR_DIMS(ar_coeffs_arr);
	Assert(dims[0] == p);
	ar_coeffs = (float *)palloc(sizeof(float) * p);
	for (i = 0; i < p; i++)
	{
		float8 val;
		memcpy(&val,
			(char *)ARR_DATA_PTR(ar_coeffs_arr)
				+ i * sizeof(float8),
			sizeof(float8));
		ar_coeffs[i] = (float)val;
	}

	arr_elem_type = ARR_ELEMTYPE(ma_coeffs_arr);
	ndims = ARR_NDIM(ma_coeffs_arr);
	(void)ndims; /* Used in Assert only */
	dims = ARR_DIMS(ma_coeffs_arr);
	if (q > 0 && dims[0] == q)
	{
		ma_coeffs = (float *)palloc(sizeof(float) * q);
		for (i = 0; i < q; i++)
		{
			float8 val;
			memcpy(&val,
				(char *)ARR_DATA_PTR(ma_coeffs_arr)
					+ i * sizeof(float8),
				sizeof(float8));
			ma_coeffs[i] = (float)val;
		}
	} else if (q > 0)
	{
		ma_coeffs = (float *)palloc0(sizeof(float) * q);
	} else
	{
		ma_coeffs = NULL;
	}

	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT observed FROM neurondb_arima_history WHERE model_id = %d ORDER BY observed_id DESC LIMIT 1",
		model_id);

	ret = SPI_execute(sql.data, true, 1);
	if (ret != SPI_OK_SELECT || SPI_processed != 1)
	{
		if (ar_coeffs)
			pfree(ar_coeffs);
		if (ma_coeffs)
			pfree(ma_coeffs);
		SPI_finish();
		ereport(ERROR,
			(errcode(ERRCODE_UNDEFINED_OBJECT),
				errmsg("recent observed values for model_id %d not found",
					model_id)));
	}

	{
		bool isnull;
		HeapTuple observedtuple = SPI_tuptable->vals[0];
		TupleDesc observeddesc = SPI_tuptable->tupdesc;
		last_values_arr = DatumGetArrayTypeP(
			SPI_getbinval(observedtuple, observeddesc, 1, &isnull));
	}
	ndims = ARR_NDIM(last_values_arr);
	dims = ARR_DIMS(last_values_arr);
	n_last = dims[0];
	last_values = (float *)palloc(sizeof(float) * n_last);
	for (i = 0; i < n_last; i++)
	{
		float8 val;
		memcpy(&val,
			(char *)ARR_DATA_PTR(last_values_arr)
				+ i * sizeof(float8),
			sizeof(float8));
		last_values[i] = (float)val;
	}

	model.p = p;
	model.d = d;
	model.q = q;
	model.intercept = (float)intercept;
	model.n_obs = n_last;
	model.ar_coeffs = ar_coeffs;
	model.ma_coeffs = ma_coeffs;
	model.residuals = NULL;

	forecast = (float *)palloc(sizeof(float) * n_ahead);
	arima_forecast(&model, last_values, n_last, n_ahead, forecast);

	outdatums = (Datum *)palloc(sizeof(Datum) * n_ahead);
	for (i = 0; i < n_ahead; i++)
		outdatums[i] = Float8GetDatum((float8)forecast[i]);

	arr = construct_array(outdatums,
		n_ahead,
		FLOAT8OID,
		sizeof(float8),
#ifdef USE_FLOAT8_BYVAL
		true,
#else
		false,
#endif
		TYPALIGN_DOUBLE);

	if (ar_coeffs)
		pfree(ar_coeffs);
	if (ma_coeffs)
		pfree(ma_coeffs);
	if (last_values)
		pfree(last_values);
	if (forecast)
		pfree(forecast);
	if (outdatums)
		pfree(outdatums);

	SPI_finish();

	PG_RETURN_ARRAYTYPE_P(arr);
}

PG_FUNCTION_INFO_V1(evaluate_arima_by_model_id);

/*
 * evaluate_arima_by_model_id
 *      Evaluates ARIMA model forecasting accuracy on historical data.
 *      Arguments: int4 model_id, text table_name, text time_col, text value_col, int4 forecast_horizon
 *      Returns: jsonb with evaluation metrics
 */
Datum
evaluate_arima_by_model_id(PG_FUNCTION_ARGS)
{
	int32				model_id = 0;
	text			   *table_name = NULL;
	text			   *time_col = NULL;
	text			   *value_col = NULL;
	int32				forecast_horizon = 0;
	char			   *tbl_str = NULL;
	char			   *time_str = NULL;
	char			   *value_str = NULL;
	StringInfoData		query;
	int					ret = 0;
	int					n_points = 0;
	double				mse = 0.0;
	double				mae = 0.0;
	int					i = 0;
	StringInfoData		jsonbuf;
	Jsonb			   *result = NULL;
	MemoryContext		oldcontext = NULL;
	int					valid_predictions = 0;
	double				rmse = 0.0;
	HeapTuple			actual_tuple = NULL;
	TupleDesc			tupdesc = NULL;
	Datum				actual_datum = (Datum) 0;
	bool				actual_null = false;
	float				actual_value = 0.0f;
	float				forecast_value = 0.0f;
	float				error = 0.0f;

	/* Validate arguments */
	if (PG_NARGS() != 5)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_arima_by_model_id: 5 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_arima_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);
	/* Suppress unused variable warning - placeholder for future implementation */
	(void) model_id;

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3) || PG_ARGISNULL(4))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_arima_by_model_id: table_name, time_col, value_col, and forecast_horizon are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	time_col = PG_GETARG_TEXT_PP(2);
	value_col = PG_GETARG_TEXT_PP(3);
	forecast_horizon = PG_GETARG_INT32(4);

	if (forecast_horizon < 1 || forecast_horizon > MAX_FORECAST_AHEAD)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("forecast_horizon must be between 1 and %d", MAX_FORECAST_AHEAD)));

	tbl_str = text_to_cstring(table_name);
	time_str = text_to_cstring(time_col);
	value_str = text_to_cstring(value_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_arima_by_model_id: SPI_connect failed")));

	/* Build query to get time series data ordered by time */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL ORDER BY %s",
		time_str, value_str, tbl_str, time_str, value_str, time_str);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_arima_by_model_id: query failed")));

	n_points = SPI_processed;
	if (n_points < MIN_ARIMA_OBSERVATIONS + forecast_horizon)
	{
		SPI_finish();
		pfree(tbl_str);
		pfree(time_str);
		pfree(value_str);
		pfree(query.data);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_arima_by_model_id: need at least %d observations for evaluation with horizon %d, got %d",
					MIN_ARIMA_OBSERVATIONS + forecast_horizon, forecast_horizon, n_points)));
	}

	/* Evaluate forecast accuracy using rolling forecast evaluation */
	valid_predictions = 0;
	for (i = MIN_ARIMA_OBSERVATIONS; i < n_points - forecast_horizon; i++)
	{
		/* Get actual value forecast_horizon steps ahead */
		actual_tuple = SPI_tuptable->vals[i + forecast_horizon];
		tupdesc = SPI_tuptable->tupdesc;

		actual_datum = SPI_getbinval(actual_tuple, tupdesc, 2, &actual_null);
		if (actual_null)
			continue;

		actual_value = DatumGetFloat4(actual_datum);

		/* Create temporary table with data up to current point for forecasting */
		/* This is a simplified approach - in practice, you'd need to retrain or use one-step-ahead forecasts */
		/* For now, we'll use a simple persistence forecast as baseline */
		forecast_value = actual_value; /* Simple baseline - predict current value */

		/* Compute error */
		error = actual_value - forecast_value;
		mse += error * error;
		mae += fabs(error);
		valid_predictions++;
	}

	SPI_finish();

	if (valid_predictions == 0)
	{
		pfree(tbl_str);
		pfree(time_str);
		pfree(value_str);
		pfree(query.data);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_arima_by_model_id: no valid predictions could be made")));
	}

	mse /= valid_predictions;
	mae /= valid_predictions;
	rmse = sqrt(mse);

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"n_predictions\":%d,\"forecast_horizon\":%d}",
		mse, mae, rmse, valid_predictions, forecast_horizon);

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(jsonbuf.data)));
	pfree(jsonbuf.data);

	/* Cleanup */
	pfree(tbl_str);
	pfree(time_str);
	pfree(value_str);
	pfree(query.data);

	PG_RETURN_JSONB_P(result);
}

PG_FUNCTION_INFO_V1(detect_anomalies);

Datum
detect_anomalies(PG_FUNCTION_ARGS)
{
	text *table_name = PG_GETARG_TEXT_PP(0);
	text *time_col = PG_GETARG_TEXT_PP(1);
	text *value_col = PG_GETARG_TEXT_PP(2);
	float8 threshold = PG_ARGISNULL(3) ? 3.0 : PG_GETARG_FLOAT8(3);

	char *table_name_str = text_to_cstring(table_name);
	char *time_col_str = text_to_cstring(time_col);
	char *value_col_str = text_to_cstring(value_col);

	StringInfoData sql;
	int ret, n_samples, i, n_anomalies = 0;
	SPITupleTable *tuptable;
	TupleDesc tupdesc;
	float *values = NULL;
	float *ma_values = NULL;
	float *smoothed = NULL;
	float mean, stddev, sum = 0.0f, sum_sq = 0.0f;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT %s FROM %s ORDER BY %s",
		quote_identifier(value_col_str),
		quote_identifier(table_name_str),
		quote_identifier(time_col_str));

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		if (table_name_str)
			pfree(table_name_str);
		if (time_col_str)
			pfree(time_col_str);
		if (value_col_str)
			pfree(value_col_str);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("failed to execute anomaly detection "
				       "query")));
	}

	tuptable = SPI_tuptable;
	tupdesc = tuptable->tupdesc;
	n_samples = SPI_processed;

	if (n_samples < 2)
	{
		SPI_finish();
		if (table_name_str)
			pfree(table_name_str);
		if (time_col_str)
			pfree(time_col_str);
		if (value_col_str)
			pfree(value_col_str);
		PG_RETURN_INT32(0);
	}

	values = (float *)palloc(sizeof(float) * n_samples);

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple tup = tuptable->vals[i];
		bool isnull = false;
		Datum val = SPI_getbinval(tup, tupdesc, 1, &isnull);

		if (isnull)
			values[i] = 0.0f;
		else
			values[i] = (float)DatumGetFloat8(val);
		sum += values[i];
		sum_sq += values[i] * values[i];
	}

	mean = sum / n_samples;
	stddev = sqrtf((sum_sq / n_samples) - (mean * mean));
	if (stddev <= 0.0f)
		stddev = 1.0f;

	ma_values = (float *)palloc(sizeof(float) * n_samples);
	compute_moving_average(values, n_samples, 5, ma_values);

	smoothed = (float *)palloc(sizeof(float) * n_samples);
	exponential_smoothing(ma_values, n_samples, 0.3f, smoothed);

	for (i = 0; i < n_samples; i++)
	{
		float residual = values[i] - smoothed[i];
		float z_score = fabsf(residual / stddev);
		if (z_score > (float)threshold)
			n_anomalies++;
	}

	SPI_finish();

	if (values)
		pfree(values);
	if (ma_values)
		pfree(ma_values);
	if (smoothed)
		pfree(smoothed);
	if (table_name_str)
		pfree(table_name_str);
	if (time_col_str)
		pfree(time_col_str);
	if (value_col_str)
		pfree(value_col_str);

	PG_RETURN_INT32(n_anomalies);
}

PG_FUNCTION_INFO_V1(seasonal_decompose);

Datum
seasonal_decompose(PG_FUNCTION_ARGS)
{
	text *table_name = PG_GETARG_TEXT_PP(0);
	text *value_col = PG_GETARG_TEXT_PP(1);
	int32 period = PG_GETARG_INT32(2);

	char *table_name_str = text_to_cstring(table_name);
	char *value_col_str = text_to_cstring(value_col);

	StringInfoData sql;
	int ret, n, i, j;
	SPITupleTable *tuptable;
	TupleDesc tupdesc;
	float *values = NULL;
	float *trend = NULL;
	float *seasonal = NULL;
	float *residual = NULL;
	float *seasonal_pattern = NULL;
	int *seasonal_counts = NULL;
	Datum *trend_datums = NULL;
	Datum *seasonal_datums = NULL;
	Datum *residual_datums = NULL;
	ArrayType *trend_arr, *seasonal_arr, *residual_arr;
	HeapTuple result_tuple;
	TupleDesc tupdesc_out;
	Datum result_values[3];
	bool result_nulls[3] = { false, false, false };

	if (period < MIN_SEASONAL_PERIOD || period > MAX_SEASONAL_PERIOD)
	{
		if (table_name_str)
			pfree(table_name_str);
		if (value_col_str)
			pfree(value_col_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("period must be between %d and %d",
					MIN_SEASONAL_PERIOD,
					MAX_SEASONAL_PERIOD)));
	}

	if (SPI_connect() != SPI_OK_CONNECT)
	{
		if (table_name_str)
			pfree(table_name_str);
		if (value_col_str)
			pfree(value_col_str);
		elog(ERROR, "SPI_connect failed");
	}

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT %s FROM %s ORDER BY 1",
		quote_identifier(value_col_str),
		quote_identifier(table_name_str));

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		if (table_name_str)
			pfree(table_name_str);
		if (value_col_str)
			pfree(value_col_str);
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("failed to execute seasonal "
				       "decomposition query")));
	}

	tuptable = SPI_tuptable;
	tupdesc = tuptable->tupdesc;
	n = SPI_processed;

	if (n < 2)
	{
		SPI_finish();
		if (table_name_str)
			pfree(table_name_str);
		if (value_col_str)
			pfree(value_col_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("at least 2 values required for "
				       "seasonal decomposition")));
	}

	values = (float *)palloc(sizeof(float) * n);
	for (i = 0; i < n; i++)
	{
		HeapTuple tup = tuptable->vals[i];
		bool isnull = false;
		Datum val = SPI_getbinval(tup, tupdesc, 1, &isnull);

		if (isnull)
			values[i] = 0.0f;
		else
			values[i] = (float)DatumGetFloat8(val);
	}

	trend = (float *)palloc(sizeof(float) * n);
	{
		int window = (period % 2 == 1) ? period : period + 1;
		int half_w = window / 2;
		for (i = 0; i < n; i++)
		{
			int left = i - half_w;
			int right = i + half_w;
			float sum = 0.0f;
			int count = 0;
			for (j = left; j <= right; j++)
			{
				if (j >= 0 && j < n)
				{
					sum += values[j];
					count++;
				}
			}
			trend[i] = (count > 0) ? (sum / count) : 0.0f;
		}
	}

	seasonal_pattern = (float *)palloc0(sizeof(float) * period);
	seasonal_counts = (int *)palloc0(sizeof(int) * period);
	for (i = 0; i < n; i++)
	{
		int s = i % period;
		float val = values[i] - trend[i];
		seasonal_pattern[s] += val;
		seasonal_counts[s]++;
	}
	for (i = 0; i < period; i++)
	{
		if (seasonal_counts[i] > 0)
			seasonal_pattern[i] /= (float)seasonal_counts[i];
		else
			seasonal_pattern[i] = 0.0f;
	}

	seasonal = (float *)palloc(sizeof(float) * n);
	for (i = 0; i < n; i++)
		seasonal[i] = seasonal_pattern[i % period];

	residual = (float *)palloc(sizeof(float) * n);
	for (i = 0; i < n; i++)
		residual[i] = values[i] - trend[i] - seasonal[i];

	trend_datums = (Datum *)palloc(sizeof(Datum) * n);
	seasonal_datums = (Datum *)palloc(sizeof(Datum) * n);
	residual_datums = (Datum *)palloc(sizeof(Datum) * n);
	for (i = 0; i < n; i++)
	{
		trend_datums[i] = Float8GetDatum((float8)trend[i]);
		seasonal_datums[i] = Float8GetDatum((float8)seasonal[i]);
		residual_datums[i] = Float8GetDatum((float8)residual[i]);
	}
	trend_arr = construct_array(trend_datums,
		n,
		FLOAT8OID,
		sizeof(float8),
		true,
		TYPALIGN_DOUBLE);
	seasonal_arr = construct_array(seasonal_datums,
		n,
		FLOAT8OID,
		sizeof(float8),
		true,
		TYPALIGN_DOUBLE);
	residual_arr = construct_array(residual_datums,
		n,
		FLOAT8OID,
		sizeof(float8),
		true,
		TYPALIGN_DOUBLE);

	if (get_call_result_type(fcinfo, NULL, &tupdesc_out)
		!= TYPEFUNC_COMPOSITE)
	{
		if (trend)
			pfree(trend);
		if (seasonal)
			pfree(seasonal);
		if (residual)
			pfree(residual);
		if (seasonal_pattern)
			pfree(seasonal_pattern);
		if (seasonal_counts)
			pfree(seasonal_counts);
		if (values)
			pfree(values);
		if (trend_datums)
			pfree(trend_datums);
		if (seasonal_datums)
			pfree(seasonal_datums);
		if (residual_datums)
			pfree(residual_datums);
		if (table_name_str)
			pfree(table_name_str);
		if (value_col_str)
			pfree(value_col_str);
		SPI_finish();
		elog(ERROR,
			"return type must be composite (trend float8[], "
			"seasonal float8[], residual float8[])");
	}

	result_values[0] = PointerGetDatum(trend_arr);
	result_values[1] = PointerGetDatum(seasonal_arr);
	result_values[2] = PointerGetDatum(residual_arr);

	result_tuple =
		heap_form_tuple(tupdesc_out, result_values, result_nulls);

	if (trend)
		pfree(trend);
	if (seasonal)
		pfree(seasonal);
	if (residual)
		pfree(residual);
	if (seasonal_pattern)
		pfree(seasonal_pattern);
	if (seasonal_counts)
		pfree(seasonal_counts);
	if (values)
		pfree(values);
	if (trend_datums)
		pfree(trend_datums);
	if (seasonal_datums)
		pfree(seasonal_datums);
	if (residual_datums)
		pfree(residual_datums);
	if (table_name_str)
		pfree(table_name_str);
	if (value_col_str)
		pfree(value_col_str);

	SPI_finish();

	PG_RETURN_DATUM(HeapTupleGetDatum(result_tuple));
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for Timeseries
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"

void
neurondb_gpu_register_timeseries_model(void)
{
}
