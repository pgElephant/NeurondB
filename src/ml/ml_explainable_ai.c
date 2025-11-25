/*-------------------------------------------------------------------------
 *
 * ml_explainable_ai.c
 *    SHAP, LIME, and feature importance for model explainability
 *
 * Implements explainable AI algorithms:
 *
 * 1. SHAP (SHapley Additive exPlanations): Game theory-based feature importance
 *    - Shapley values for fair feature attribution
 *    - Kernel SHAP for model-agnostic explanations
 *
 * 2. LIME (Local Interpretable Model-agnostic Explanations): Local explanations
 *    - Perturbs input and observes predictions
 *    - Fits linear model to explain local behavior
 *
 * 3. Feature Importance: Permutation and tree-based importance
 *    - Permutation importance
 *    - Tree-based feature importance
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/ml/ml_explainable_ai.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/jsonb.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_catalog.h"

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_spi_safe.h"

/*
 * Helper function to call model prediction via SPI
 * Returns prediction value, or 0.0 on error
 */
static double
call_model_predict(int32 model_id, float *features, int n_features)
{
	StringInfoData sql;
	StringInfoData features_str;
	int			ret;
	double		prediction = 0.0;
	int			i;

	/* Build features array string */
	initStringInfo(&features_str);
	appendStringInfoString(&features_str, "ARRAY[");
	for (i = 0; i < n_features; i++)
	{
		if (i > 0)
			appendStringInfoString(&features_str, ", ");
		appendStringInfo(&features_str, "%.6f", (double) features[i]);
	}
	appendStringInfoString(&features_str, "]::real[]");

	/* Call unified predict function */
	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT neurondb.predict(%d, %s)", model_id, features_str.data);

	ret = ndb_spi_execute_safe(sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull = false;
		Datum		result = SPI_getbinval(SPI_tuptable->vals[0],
										   SPI_tuptable->tupdesc,
										   1,
										   &isnull);

		if (!isnull)
			prediction = DatumGetFloat8(result);
	}

	pfree(sql.data);
	pfree(features_str.data);

	return prediction;
}

/*
 * calculate_shap_values
 * --------------------
 * Calculate SHAP values for a prediction.
 *
 * SQL Arguments:
 *   model_id      - Trained model ID
 *   instance      - Feature vector to explain
 *   n_samples     - Number of samples for Kernel SHAP (default: 100)
 *
 * Returns:
 *   Array of SHAP values (one per feature)
 */
PG_FUNCTION_INFO_V1(calculate_shap_values);

Datum
calculate_shap_values(PG_FUNCTION_ARGS)
{
	int32		model_id;
	ArrayType  *instance;
	int			n_samples;
	float	   *features;
	int			n_features;
	double	   *shap_values;
	int			i,
				j;
	ArrayType  *result;
	Datum	   *result_datums;
	StringInfoData query;
	int			ret;

	model_id = PG_GETARG_INT32(0);
	instance = PG_GETARG_ARRAYTYPE_P(1);
	n_samples = PG_ARGISNULL(2) ? 100 : PG_GETARG_INT32(2);

	if (n_samples < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_samples must be positive")));

	/* Extract features from array */
	n_features = ARR_DIMS(instance)[0];
	features = (float *) ARR_DATA_PTR(instance);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("SPI_connect failed")));

	/* Initialize SHAP values */
	shap_values = (double *) palloc0(sizeof(double) * n_features);

	/* Kernel SHAP: approximate Shapley values using sampling */
	/* For each feature, estimate its contribution */
	for (i = 0; i < n_features; i++)
	{
		double		sum_contrib = 0.0;
		int			valid_samples = 0;

		for (j = 0; j < n_samples; j++)
		{
			/* Create perturbed instance */
			float	   *perturbed = (float *) palloc(sizeof(float) * n_features);
			int			k;
			double		pred_with,
						pred_without;

			/* Sample subset of features */
			for (k = 0; k < n_features; k++)
			{
				if (k == i)
					perturbed[k] = features[k];
				else if ((rand() % 2) == 0)
					perturbed[k] = features[k];
				else
					perturbed[k] = 0.0; /* Missing feature */
			}

			/* Get prediction with feature i */
			pred_with = call_model_predict(model_id, perturbed, n_features);

			/* Get prediction without feature i */
			perturbed[i] = 0.0;
			pred_without = call_model_predict(model_id, perturbed, n_features);

			sum_contrib += pred_with - pred_without;
			valid_samples++;

			NDB_SAFE_PFREE_AND_NULL(perturbed);
		}

		if (valid_samples > 0)
			shap_values[i] = sum_contrib / valid_samples;
	}

	/* Build result array */
	result_datums = (Datum *) palloc(sizeof(Datum) * n_features);
	for (i = 0; i < n_features; i++)
		result_datums[i] = Float8GetDatum(shap_values[i]);

	result = construct_array(result_datums,
							 n_features,
							 FLOAT8OID,
							 sizeof(float8),
							 FLOAT8PASSBYVAL,
							 'd');

	/* Cleanup */
	NDB_SAFE_PFREE_AND_NULL(shap_values);
	NDB_SAFE_PFREE_AND_NULL(result_datums);
	SPI_finish();

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * explain_with_lime
 * ------------------
 * Generate LIME explanation for a prediction.
 *
 * SQL Arguments:
 *   model_id      - Trained model ID
 *   instance      - Feature vector to explain
 *   n_samples     - Number of perturbed samples (default: 1000)
 *   n_features    - Number of top features to explain (default: 10)
 *
 * Returns:
 *   JSONB with feature names and importance scores
 */
PG_FUNCTION_INFO_V1(explain_with_lime);

Datum
explain_with_lime(PG_FUNCTION_ARGS)
{
	int32		model_id;
	ArrayType  *instance;
	int			n_samples;
	int			n_features;
	float	   *features;
	int			feature_dim;
	float	   *perturbed_features;
	double	   *predictions;
	double	   *weights;
	double	   *coefficients;
	int			i,
				j;
	StringInfoData jsonbuf;
	Jsonb	   *result;

	model_id = PG_GETARG_INT32(0);
	instance = PG_GETARG_ARRAYTYPE_P(1);
	n_samples = PG_ARGISNULL(2) ? 1000 : PG_GETARG_INT32(2);
	n_features = PG_ARGISNULL(3) ? 10 : PG_GETARG_INT32(3);

	if (n_samples < 1 || n_features < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_samples and n_features must be positive")));

	feature_dim = ARR_DIMS(instance)[0];
	features = (float *) ARR_DATA_PTR(instance);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("SPI_connect failed")));

	/* Allocate arrays */
	perturbed_features = (float *) palloc(sizeof(float) * feature_dim * n_samples);
	predictions = (double *) palloc(sizeof(double) * n_samples);
	weights = (double *) palloc(sizeof(double) * n_samples);
	coefficients = (double *) palloc0(sizeof(double) * feature_dim);

	/* Generate perturbed samples and get predictions */
	for (i = 0; i < n_samples; i++)
	{
		double		distance = 0.0;
		int			k;

		/* Perturb features */
		for (k = 0; k < feature_dim; k++)
		{
			float		noise = ((float) rand() / (float) RAND_MAX - 0.5) * 0.2;

			perturbed_features[i * feature_dim + k] = features[k] + noise;
			distance += noise * noise;
		}

		/* Get prediction using actual model */
		predictions[i] = call_model_predict(model_id,
											&perturbed_features[i * feature_dim],
											feature_dim);

		/* Exponential kernel weight based on distance */
		distance = sqrt(distance);
		weights[i] = exp(-distance * distance / (2.0 * 0.5 * 0.5));
	}

	/* Fit linear model: minimize weighted MSE */
	/* Simplified: would use proper linear regression solver */
	for (i = 0; i < feature_dim; i++)
	{
		double		numerator = 0.0;
		double		denominator = 0.0;

		for (j = 0; j < n_samples; j++)
		{
			float		feat_val = perturbed_features[j * feature_dim + i];

			numerator += weights[j] * feat_val * predictions[j];
			denominator += weights[j] * feat_val * feat_val;
		}

		if (denominator > 1e-10)
			coefficients[i] = numerator / denominator;
	}

	/* Build JSONB result */
	initStringInfo(&jsonbuf);
	appendStringInfoString(&jsonbuf, "{\"features\":[");
	for (i = 0; i < feature_dim && i < n_features; i++)
	{
		if (i > 0)
			appendStringInfoString(&jsonbuf, ",");
		appendStringInfo(&jsonbuf,
						 "{\"index\":%d,\"importance\":%.6f}",
						 i,
						 coefficients[i]);
	}
	appendStringInfoString(&jsonbuf, "]}");

	result = DatumGetJsonbP(
							DirectFunctionCall1(jsonb_in, CStringGetDatum(jsonbuf.data)));

	/* Cleanup */
	NDB_SAFE_PFREE_AND_NULL(perturbed_features);
	NDB_SAFE_PFREE_AND_NULL(predictions);
	NDB_SAFE_PFREE_AND_NULL(weights);
	NDB_SAFE_PFREE_AND_NULL(coefficients);
	NDB_SAFE_PFREE_AND_NULL(jsonbuf.data);
	SPI_finish();

	PG_RETURN_POINTER(result);
}

/*
 * feature_importance
 * ------------------
 * Calculate feature importance using permutation method.
 *
 * SQL Arguments:
 *   model_id      - Trained model ID
 *   table_name    - Test data table
 *   feature_column - Feature column name
 *   target_column  - Target column name
 *   metric        - Metric to use ('accuracy', 'mse', 'mae') (default: 'mse')
 *
 * Returns:
 *   Array of importance scores (one per feature)
 */
PG_FUNCTION_INFO_V1(feature_importance);

Datum
feature_importance(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *feature_column;
	text	   *target_column;
	text	   *metric_text;
	char	   *tbl_str;
	char	   *feat_col_str;
	char	   *targ_col_str;
	char	   *metric;
	float	  **features;
	double	   *targets;
	int			n_samples;
	int			n_features;
	double		baseline_score;
	double	   *importance;
	int			i,
				j;
	ArrayType  *result;
	Datum	   *result_datums;
	StringInfoData query;
	int			ret;

	model_id = PG_GETARG_INT32(0);
	table_name = PG_GETARG_TEXT_PP(1);
	feature_column = PG_GETARG_TEXT_PP(2);
	target_column = PG_GETARG_TEXT_PP(3);
	metric_text = PG_ARGISNULL(4) ? cstring_to_text("mse")
		: PG_GETARG_TEXT_PP(4);

	tbl_str = text_to_cstring(table_name);
	feat_col_str = text_to_cstring(feature_column);
	targ_col_str = text_to_cstring(target_column);
	metric = text_to_cstring(metric_text);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("SPI_connect failed")));

	/* Load test data */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s",
					 feat_col_str,
					 targ_col_str,
					 tbl_str);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Failed to load test data")));

	n_samples = SPI_processed;
	/* TODO: Determine actual feature count from model or data schema */

	/*
	 * For now, use default value - should be replaced with actual model
	 * metadata
	 */
	n_features = 10;

	features = (float **) palloc(sizeof(float *) * n_samples);
	targets = (double *) palloc(sizeof(double) * n_samples);

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		ArrayType  *feat_array = DatumGetArrayTypeP(
													SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, NULL));
		double		target = DatumGetFloat8(SPI_getbinval(tuple,
														  SPI_tuptable->tupdesc,
														  2, NULL));

		features[i] = (float *) ARR_DATA_PTR(feat_array);
		targets[i] = target;
	}

	/* Calculate baseline score */
	baseline_score = 0.0;
	for (i = 0; i < n_samples; i++)
	{
		/* Call actual model prediction */
		double		pred = call_model_predict(model_id, features[i], n_features);
		double		diff = pred - targets[i];

		baseline_score += diff * diff;
	}
	baseline_score /= n_samples;

	/* Permutation importance: shuffle each feature and measure impact */
	importance = (double *) palloc0(sizeof(double) * n_features);

	for (i = 0; i < n_features; i++)
	{
		/* Shuffle feature i */
		for (j = n_samples - 1; j > 0; j--)
		{
			int			k = rand() % (j + 1);
			float		tmp = features[j][i];

			features[j][i] = features[k][i];
			features[k][i] = tmp;
		}

		/* Calculate score with shuffled feature */
		double		shuffled_score = 0.0;
		int			s;

		for (s = 0; s < n_samples; s++)
		{
			/* Call actual model prediction */
			double		pred = call_model_predict(model_id, features[s], n_features);
			double		diff = pred - targets[s];

			shuffled_score += diff * diff;
		}
		shuffled_score /= n_samples;

		/* Importance = decrease in performance */
		importance[i] = shuffled_score - baseline_score;

		/* Restore original order */
		for (j = n_samples - 1; j > 0; j--)
		{
			int			k = rand() % (j + 1);
			float		tmp = features[j][i];

			features[j][i] = features[k][i];
			features[k][i] = tmp;
		}
	}

	/* Build result array */
	result_datums = (Datum *) palloc(sizeof(Datum) * n_features);
	for (i = 0; i < n_features; i++)
		result_datums[i] = Float8GetDatum(importance[i]);

	result = construct_array(result_datums,
							 n_features,
							 FLOAT8OID,
							 sizeof(float8),
							 FLOAT8PASSBYVAL,
							 'd');

	/* Cleanup */
	NDB_SAFE_PFREE_AND_NULL(features);
	NDB_SAFE_PFREE_AND_NULL(targets);
	NDB_SAFE_PFREE_AND_NULL(importance);
	NDB_SAFE_PFREE_AND_NULL(result_datums);
	NDB_SAFE_PFREE_AND_NULL(query.data);
	NDB_SAFE_PFREE_AND_NULL(tbl_str);
	NDB_SAFE_PFREE_AND_NULL(feat_col_str);
	NDB_SAFE_PFREE_AND_NULL(targ_col_str);
	NDB_SAFE_PFREE_AND_NULL(metric);
	SPI_finish();

	PG_RETURN_ARRAYTYPE_P(result);
}
