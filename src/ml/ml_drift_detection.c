/*-------------------------------------------------------------------------
 *
 * ml_drift_detection.c
 *    Embedding drift detection via centroid shift monitoring
 *
 * Embedding drift occurs when the distribution of embeddings changes over time,
 * indicating potential model degradation, data shift, or system issues.
 *
 * Detection Methods:
 *
 * 1. Centroid Shift: Compare mean vectors between time periods
 *    - Simple, interpretable
 *    - O(n*d) complexity
 *    - Good for detecting overall distribution shift
 *
 * 2. Covariance Change: Compare covariance matrices
 *    - Detects variance and correlation changes
 *    - More sensitive than centroid alone
 *
 * 3. KL Divergence (approximate): Estimate distribution divergence
 *    - Assumes roughly Gaussian distributions
 *    - Quantifies "surprise" of new data
 *
 * Metrics Returned:
 *   - Euclidean distance between centroids
 *   - Normalized shift (as fraction of baseline std)
 *   - P-value from Hotelling's T² test (if applicable)
 *
 * Use Cases:
 *   - Monitor embedding model health
 *   - Detect data distribution changes
 *   - Trigger model retraining
 *   - Alert on anomalous batches
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/ml/ml_drift_detection.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"

#include "neurondb.h"
#include "neurondb_ml.h"

#include <math.h>
#include <float.h>

/*
 * Compute Euclidean distance between two vectors
 */
static inline double
vector_distance(const double *a, const double *b, int dim)
{
	double sum = 0.0;
	int i;

	for (i = 0; i < dim; i++)
	{
		double diff = a[i] - b[i];
		sum += diff * diff;
	}
	return sqrt(sum);
}

/*
 * detect_centroid_drift
 * ---------------------
 * Detect embedding drift by comparing centroids between two datasets.
 *
 * SQL Arguments:
 *   baseline_table    - Baseline/reference table
 *   baseline_column   - Baseline vector column
 *   current_table     - Current/test table
 *   current_column    - Current vector column
 *
 * Returns:
 *   Record with (drift_distance FLOAT8, normalized_drift FLOAT8, is_significant BOOLEAN)
 *
 * Interpretation:
 *   - drift_distance: Euclidean distance between centroids
 *   - normalized_drift: Distance / baseline_std (dimensionless)
 *   - is_significant: TRUE if normalized_drift > 3.0 (rule of thumb)
 *
 * Thresholds (normalized_drift):
 *   - < 1.0: No significant drift
 *   - 1.0-2.0: Minor drift (monitor)
 *   - 2.0-3.0: Moderate drift (investigate)
 *   - > 3.0: Significant drift (action required)
 *
 * Example Usage:
 *   -- Daily drift monitoring:
 *   SELECT detect_centroid_drift(
 *     'embeddings_baseline',    -- Last week's data
 *     'embedding',
 *     'embeddings_today',        -- Today's data
 *     'embedding'
 *   );
 *
 *   -- Continuous monitoring:
 *   SELECT date, (detect_centroid_drift(
 *     'embeddings_jan',
 *     'embedding',
 *     'embeddings_' || to_char(date, 'Mon'),
 *     'embedding'
 *   )).* FROM generate_series('2025-02-01'::date, '2025-12-31'::date, '1 month') date;
 *
 * Notes:
 *   - Requires at least 10 vectors in each dataset
 *   - Dimensions must match
 *   - Best used with regular monitoring (daily/weekly)
 */
PG_FUNCTION_INFO_V1(detect_centroid_drift);

Datum
detect_centroid_drift(PG_FUNCTION_ARGS)
{
	text *baseline_table;
	text *baseline_column;
	text *current_table;
	text *current_column;
	char *baseline_tbl;
	char *baseline_col;
	char *current_tbl;
	char *current_col;
	float **baseline_vecs;
	float **current_vecs;
	int n_baseline, n_current;
	int dim_baseline, dim_current;
	double *baseline_mean;
	double *current_mean;
	double *baseline_std;
	double drift_distance;
	double normalized_drift;
	double avg_std;
	bool is_significant;
	int i, d;
	TupleDesc tupdesc;
	Datum values[3];
	bool nulls[3];
	HeapTuple tuple;

	/* Parse arguments */
	baseline_table = PG_GETARG_TEXT_PP(0);
	baseline_column = PG_GETARG_TEXT_PP(1);
	current_table = PG_GETARG_TEXT_PP(2);
	current_column = PG_GETARG_TEXT_PP(3);

	baseline_tbl = text_to_cstring(baseline_table);
	baseline_col = text_to_cstring(baseline_column);
	current_tbl = text_to_cstring(current_table);
	current_col = text_to_cstring(current_column);

	elog(DEBUG1, "neurondb: Drift detection: baseline=%s.%s, current=%s.%s",
		 baseline_tbl, baseline_col, current_tbl, current_col);

	/* Fetch vectors */
	baseline_vecs = neurondb_fetch_vectors_from_table(baseline_tbl, baseline_col,
													   &n_baseline, &dim_baseline);
	current_vecs = neurondb_fetch_vectors_from_table(current_tbl, current_col,
													  &n_current, &dim_current);

	if (n_baseline < 10 || n_current < 10)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 10 vectors in each dataset (baseline=%d, current=%d)",
						n_baseline, n_current)));

	if (dim_baseline != dim_current)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Dimension mismatch: baseline=%d, current=%d",
						dim_baseline, dim_current)));

	/* Compute baseline centroid and std deviation */
	baseline_mean = (double *) palloc0(sizeof(double) * dim_baseline);
	baseline_std = (double *) palloc0(sizeof(double) * dim_baseline);

	for (i = 0; i < n_baseline; i++)
		for (d = 0; d < dim_baseline; d++)
			baseline_mean[d] += (double)baseline_vecs[i][d];
	
	for (d = 0; d < dim_baseline; d++)
		baseline_mean[d] /= n_baseline;

	/* Compute baseline standard deviation per dimension */
	for (i = 0; i < n_baseline; i++)
		for (d = 0; d < dim_baseline; d++)
		{
			double diff = (double)baseline_vecs[i][d] - baseline_mean[d];
			baseline_std[d] += diff * diff;
		}
	
	for (d = 0; d < dim_baseline; d++)
		baseline_std[d] = sqrt(baseline_std[d] / n_baseline);

	/* Compute average std across dimensions */
	avg_std = 0.0;
	for (d = 0; d < dim_baseline; d++)
		avg_std += baseline_std[d];
	avg_std /= dim_baseline;

	if (avg_std < 1e-10)
		avg_std = 1.0;  /* Avoid division by zero */

	/* Compute current centroid */
	current_mean = (double *) palloc0(sizeof(double) * dim_current);
	for (i = 0; i < n_current; i++)
		for (d = 0; d < dim_current; d++)
			current_mean[d] += (double)current_vecs[i][d];
	
	for (d = 0; d < dim_current; d++)
		current_mean[d] /= n_current;

	/* Compute drift distance */
	drift_distance = vector_distance(baseline_mean, current_mean, dim_baseline);
	normalized_drift = drift_distance / avg_std;
	is_significant = (normalized_drift > 3.0);

	elog(DEBUG1, "neurondb: Drift distance=%.4f, normalized=%.4f, significant=%s",
		 drift_distance, normalized_drift, is_significant ? "YES" : "NO");

	/* Build result tuple */
	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Function returning record called in context that cannot accept type record")));

	tupdesc = BlessTupleDesc(tupdesc);

	values[0] = Float8GetDatum(drift_distance);
	values[1] = Float8GetDatum(normalized_drift);
	values[2] = BoolGetDatum(is_significant);
	nulls[0] = false;
	nulls[1] = false;
	nulls[2] = false;

	tuple = heap_form_tuple(tupdesc, values, nulls);

	/* Cleanup */
	for (i = 0; i < n_baseline; i++)
		pfree(baseline_vecs[i]);
	for (i = 0; i < n_current; i++)
		pfree(current_vecs[i]);
	pfree(baseline_vecs);
	pfree(current_vecs);
	pfree(baseline_mean);
	pfree(current_mean);
	pfree(baseline_std);
	pfree(baseline_tbl);
	pfree(baseline_col);
	pfree(current_tbl);
	pfree(current_col);

	PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/*
 * compute_distribution_divergence
 * --------------------------------
 * Approximate KL divergence between two embedding distributions.
 *
 * Assumes multivariate Gaussian distributions and computes simplified divergence.
 * Returns positive value where higher = more divergence.
 */
PG_FUNCTION_INFO_V1(compute_distribution_divergence);

Datum
compute_distribution_divergence(PG_FUNCTION_ARGS)
{
	text *baseline_table;
	text *baseline_column;
	text *current_table;
	text *current_column;
	char *baseline_tbl;
	char *baseline_col;
	char *current_tbl;
	char *current_col;
	float **baseline_vecs;
	float **current_vecs;
	int n_baseline, n_current;
	int dim_baseline, dim_current;
	double *baseline_mean;
	double *current_mean;
	double *baseline_var;
	double *current_var;
	double divergence;
	int i, d;

	/* Parse arguments */
	baseline_table = PG_GETARG_TEXT_PP(0);
	baseline_column = PG_GETARG_TEXT_PP(1);
	current_table = PG_GETARG_TEXT_PP(2);
	current_column = PG_GETARG_TEXT_PP(3);

	baseline_tbl = text_to_cstring(baseline_table);
	baseline_col = text_to_cstring(baseline_column);
	current_tbl = text_to_cstring(current_table);
	current_col = text_to_cstring(current_column);

	/* Fetch vectors */
	baseline_vecs = neurondb_fetch_vectors_from_table(baseline_tbl, baseline_col,
													   &n_baseline, &dim_baseline);
	current_vecs = neurondb_fetch_vectors_from_table(current_tbl, current_col,
													  &n_current, &dim_current);

	if (n_baseline < 10 || n_current < 10)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 10 vectors in each dataset")));

	if (dim_baseline != dim_current)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Dimension mismatch")));

	/* Compute means */
	baseline_mean = (double *) palloc0(sizeof(double) * dim_baseline);
	current_mean = (double *) palloc0(sizeof(double) * dim_current);

	for (i = 0; i < n_baseline; i++)
		for (d = 0; d < dim_baseline; d++)
			baseline_mean[d] += (double)baseline_vecs[i][d];
	for (d = 0; d < dim_baseline; d++)
		baseline_mean[d] /= n_baseline;

	for (i = 0; i < n_current; i++)
		for (d = 0; d < dim_current; d++)
			current_mean[d] += (double)current_vecs[i][d];
	for (d = 0; d < dim_current; d++)
		current_mean[d] /= n_current;

	/* Compute variances */
	baseline_var = (double *) palloc0(sizeof(double) * dim_baseline);
	current_var = (double *) palloc0(sizeof(double) * dim_current);

	for (i = 0; i < n_baseline; i++)
		for (d = 0; d < dim_baseline; d++)
		{
			double diff = (double)baseline_vecs[i][d] - baseline_mean[d];
			baseline_var[d] += diff * diff;
		}
	for (d = 0; d < dim_baseline; d++)
		baseline_var[d] /= n_baseline;

	for (i = 0; i < n_current; i++)
		for (d = 0; d < dim_current; d++)
		{
			double diff = (double)current_vecs[i][d] - current_mean[d];
			current_var[d] += diff * diff;
		}
	for (d = 0; d < dim_current; d++)
		current_var[d] /= n_current;

	/* Approximate KL divergence (simplified, per-dimension) */
	divergence = 0.0;
	for (d = 0; d < dim_baseline; d++)
	{
		double mean_diff = baseline_mean[d] - current_mean[d];
		double var_ratio;
		
		if (baseline_var[d] < 1e-10 || current_var[d] < 1e-10)
			continue;

		var_ratio = current_var[d] / baseline_var[d];
		
		/* KL(P||Q) ≈ 0.5 * [log(σ_q²/σ_p²) + σ_p²/σ_q² + (μ_p-μ_q)²/σ_q² - 1] */
		divergence += 0.5 * (log(var_ratio) + 1.0/var_ratio + 
							 mean_diff * mean_diff / current_var[d] - 1.0);
	}

	/* Cleanup */
	for (i = 0; i < n_baseline; i++)
		pfree(baseline_vecs[i]);
	for (i = 0; i < n_current; i++)
		pfree(current_vecs[i]);
	pfree(baseline_vecs);
	pfree(current_vecs);
	pfree(baseline_mean);
	pfree(current_mean);
	pfree(baseline_var);
	pfree(current_var);
	pfree(baseline_tbl);
	pfree(baseline_col);
	pfree(current_tbl);
	pfree(current_col);

	PG_RETURN_FLOAT8(divergence);
}

