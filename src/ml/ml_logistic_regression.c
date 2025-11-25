/*-------------------------------------------------------------------------
 *
 * ml_logistic_regression.c
 *	  Logistic Regression implementation for binary classification
 *
 * This module implements logistic regression for binary classification tasks
 * within the NeuronDB extension. The implementation uses gradient descent
 * optimization with L2 regularization to find optimal weights and bias that
 * minimize binary cross-entropy loss. The sigmoid activation function transforms
 * linear combinations of features into probability estimates between 0 and 1.
 *
 * The training process employs stochastic gradient descent, processing data in
 * chunks to handle large datasets efficiently. For each training iteration, the
 * algorithm computes gradients of the loss function with respect to weights and
 * bias, then updates parameters using a learning rate and L2 regularization term
 * to prevent overfitting. The gradient computation uses the chain rule, deriving
 * gradients from the sigmoid function and binary cross-entropy loss.
 *
 * The module supports both CPU and GPU-accelerated training. GPU training
 * leverages optimized kernels for matrix operations and parallel gradient
 * computation, enabling faster convergence on large datasets. The implementation
 * automatically falls back to CPU training when GPU is unavailable or when
 * dataset characteristics favor CPU processing.
 *
 * Model prediction computes the sigmoid of the linear combination (weights·features + bias),
 * producing probability estimates that can be thresholded for binary classification.
 * Evaluation metrics include accuracy, precision, recall, F1 score, and log loss,
 * computed by comparing predictions against actual labels.
 *
 * The module computes gradients using the chain rule from sigmoid derivatives and
 * binary cross-entropy loss, enabling efficient weight updates through backpropagation
 * across training iterations. For numerical stability, the implementation clamps
 * probabilities to avoid log(0) errors in loss computation.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *	  src/ml/ml_logistic_regression.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/memutils.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_logistic_regression_internal.h"
#include "ml_catalog.h"
#include "neurondb_gpu_bridge.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "neurondb_gpu_backend.h"
#include "ml_gpu_registry.h"
#include "ml_gpu_logistic_regression.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_sql.h"
#include "utils/elog.h"
#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_lr.h"
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif

#include "utils/builtins.h"

#include <math.h>
#include <float.h>

typedef struct LRDataset
{
	float	   *features;
	double	   *labels;
	int			n_samples;
	int			feature_dim;
}			LRDataset;

/*
 * Streaming accumulator for chunked gradient descent
 * Accumulates gradients across chunks without loading all data
 */
typedef struct LRStreamAccum
{
	double	   *grad_w;
	double		grad_b;
	int			feature_dim;
	int			n_samples;
	bool		initialized;
}			LRStreamAccum;

static void lr_dataset_init(LRDataset * dataset);
static void lr_dataset_free(LRDataset * dataset);
static void lr_dataset_load_limited(const char *quoted_tbl,
									const char *quoted_feat,
									const char *quoted_label,
									LRDataset * dataset,
									int max_rows);
static void lr_stream_accum_init(LRStreamAccum * accum, int dim);
static void lr_stream_accum_free(LRStreamAccum * accum);
static void lr_stream_accum_reset(LRStreamAccum * accum);
static void lr_stream_process_chunk(const char *quoted_tbl,
									const char *quoted_feat,
									const char *quoted_label,
									LRStreamAccum * accum,
									double *weights,
									double bias,
									int chunk_size,
									int offset,
									int *rows_processed);
static bytea * lr_model_serialize(const LRModel * model);
static LRModel * lr_model_deserialize(const bytea * data);
static bool lr_metadata_is_gpu(Jsonb * metadata);
static bool lr_try_gpu_predict_catalog(int32 model_id,
									   const Vector *feature_vec,
									   double *result_out);
static bool lr_load_model_from_catalog(int32 model_id, LRModel * *out);

/*
 * Sigmoid function: 1 / (1 + exp(-z))
 */
static inline double
sigmoid(double z)
{
	if (z > 500)
		return 1.0;
	if (z < -500)
		return 0.0;
	return 1.0 / (1.0 + exp(-z));
}

/*
 * Compute log loss (binary cross-entropy)
 * Note: Currently unused as we compute loss directly in chunked metrics
 */
static double
			compute_log_loss(double *y_true, double *y_pred, int n)
			pg_attribute_unused();
static double
compute_log_loss(double *y_true, double *y_pred, int n)
{
	double		loss = 0.0;
	int			i;

	for (i = 0; i < n; i++)
	{
		double		pred = y_pred[i];

		pred = fmax(1e-15, fmin(1.0 - 1e-15, pred));

		if (y_true[i] > 0.5)
			loss -= log(pred);
		else
			loss -= log(1.0 - pred);
	}

	return loss / n;
}

/*
 * train_logistic_regression
 *
 * Trains a logistic regression model using gradient descent
 * Returns coefficients (intercept + feature weights)
 */
PG_FUNCTION_INFO_V1(train_logistic_regression);

Datum
train_logistic_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	int			max_iters = PG_GETARG_INT32(3);
	double		learning_rate = PG_GETARG_FLOAT8(4);
	double		lambda = PG_GETARG_FLOAT8(5);
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	int			nvec = 0;
	int			dim = 0;
	double	   *weights = NULL;
	double		bias = 0.0;
	int			iter;
	int			j;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_label;
	MLGpuTrainResult gpu_result;
	NDB_DECLARE (char *, gpu_err);
	NDB_DECLARE (Jsonb *, gpu_hyperparams);
	StringInfoData hyperbuf = {0};
	int32		model_id = 0;
	double		final_loss = 0.0;
	int			correct = 0;
	LRStreamAccum stream_accum;
	int			chunk_size;
	int			offset;
	int			rows_in_chunk;
	int			total_rows_processed;
	double		accuracy_val = 0.0;

	if (PG_NARGS() != 6)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: train_logistic_regression: 6 arguments are required"),
				 errdetail("Usage: train_logistic_regression(table_name, feature_col, target_col, max_iters, learning_rate, lambda)"),
				 errhint("Provide table name, feature column, target column, maximum iterations, learning rate, and regularization lambda.")));
	}

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_label = quote_identifier(targ_str);

	{
		StringInfoData count_query = {0};
		int			ret;
		Oid			feat_type_oid = InvalidOid;
		NDB_DECLARE (NdbSpiSession *, check_spi_session);
		MemoryContext oldcontext;

		oldcontext = CurrentMemoryContext;
		Assert(oldcontext != NULL);
		NDB_SPI_SESSION_BEGIN(check_spi_session, oldcontext);

		initStringInfo(&count_query);
		appendStringInfo(&count_query,
						 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT 1",
						 quoted_feat,
						 quoted_label,
						 quoted_tbl,
						 quoted_feat,
						 quoted_label);

		ret = ndb_spi_execute(check_spi_session, count_query.data, true, 0);
		pfree(count_query.data);
		count_query.data = NULL;
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			NDB_SPI_SESSION_END(check_spi_session);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_logistic_regression: no valid rows found"),
					 errdetail("SPI execution returned code %d (expected %d), processed %lu rows", ret, SPI_OK_SELECT, (unsigned long) SPI_processed),
					 errhint("Verify the table exists and contains valid feature and target columns.")));
		}

		if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		{
			HeapTuple	first_tuple = SPI_tuptable->vals[0];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			Datum		feat_datum;
			bool		feat_null;

			feat_type_oid = SPI_gettypeid(tupdesc, 1);
			feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &feat_null);
			if (!feat_null)
			{
				if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
				{
					ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

					if (ARR_NDIM(arr) == 1)
						dim = ARR_DIMS(arr)[0];
				}
				else
				{
					Vector	   *vec = DatumGetVector(feat_datum);

					dim = vec->dim;
				}
			}
		}

		if (dim <= 0)
		{
			NDB_SPI_SESSION_END(check_spi_session);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_logistic_regression: could not determine feature dimension"),
					 errdetail("Feature dimension is %d (must be > 0)", dim),
					 errhint("Ensure feature column contains valid vector or array data.")));
		}

		initStringInfo(&count_query);
		appendStringInfo(&count_query,
						 "SELECT COUNT(*) FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
						 quoted_tbl,
						 quoted_feat,
						 quoted_label);

		ret = ndb_spi_execute(check_spi_session, count_query.data, true, 0);
		pfree(count_query.data);
		count_query.data = NULL;
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool		count_null;
			HeapTuple	tuple = SPI_tuptable->vals[0];
			Datum		count_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, &count_null);

			if (!count_null)
				nvec = DatumGetInt32(count_datum);
		}

		NDB_SPI_SESSION_END(check_spi_session);

		if (nvec < 10)
		{
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_logistic_regression: need at least 10 samples, got %d",
							nvec),
					 errdetail("Dataset contains %d rows, minimum required is 10", nvec),
					 errhint("Add more data rows to the training table.")));
		}
	}

	{
		int			max_samples = 500000;

		if (nvec > max_samples)
		{
			elog(INFO,
				 "neurondb: logistic_regression: dataset has %d rows, limiting to %d samples for training",
				 nvec,
				 max_samples);
			nvec = max_samples;
		}

		if (neurondb_gpu_enabled)
			ndb_gpu_init_if_needed();

		elog(DEBUG1,
			 "neurondb: logistic_regression: checking GPU - enabled=%d, available=%d, kernel_enabled=%d",
			 neurondb_gpu_enabled ? 1 : 0,
			 neurondb_gpu_is_available() ? 1 : 0,
			 ndb_gpu_kernel_enabled("lr_train") ? 1 : 0);

		if (neurondb_gpu_is_available() && nvec > 0 && dim > 0 && nvec <= 1000000 && dim <= 10000
			&& ndb_gpu_kernel_enabled("lr_train"))
		{
			int			gpu_sample_limit = nvec;
			LRDataset	dataset;
			bool		gpu_training_succeeded = false;

			if (gpu_sample_limit <= 0 || gpu_sample_limit > 10000000)
			{
				elog(DEBUG1,
					 "neurondb: logistic_regression: invalid gpu_sample_limit %d, skipping GPU",
					 gpu_sample_limit);
			}
			else if (dim <= 0 || dim > 10000)
			{
				elog(DEBUG1,
					 "neurondb: logistic_regression: invalid feature_dim %d, skipping GPU",
					 dim);
			}
			else if (tbl_str == NULL || feat_str == NULL || targ_str == NULL)
			{
				elog(DEBUG1,
					 "neurondb: logistic_regression: NULL table/column names, skipping GPU");
			}
			else
			{
				lr_dataset_init(&dataset);

				PG_TRY();
				{
					elog(INFO,
						 "neurondb: logistic_regression: attempting GPU training with %d samples",
						 gpu_sample_limit);

					lr_dataset_load_limited(quoted_tbl,
											quoted_feat,
											quoted_label,
											&dataset,
											gpu_sample_limit);

					if (dataset.features == NULL)
					{
						elog(DEBUG1,
							 "neurondb: logistic_regression: dataset.features is NULL after loading");
						lr_dataset_free(&dataset);
					}
					else if (dataset.labels == NULL)
					{
						elog(DEBUG1,
							 "neurondb: logistic_regression: dataset.labels is NULL after loading");
						lr_dataset_free(&dataset);
					}
					else if (dataset.n_samples <= 0 || dataset.n_samples > 10000000)
					{
						elog(DEBUG1,
							 "neurondb: logistic_regression: invalid n_samples %d after loading",
							 dataset.n_samples);
						lr_dataset_free(&dataset);
					}
					else if (dataset.feature_dim <= 0 || dataset.feature_dim > 10000)
					{
						elog(DEBUG1,
							 "neurondb: logistic_regression: invalid feature_dim %d after loading",
							 dataset.feature_dim);
						lr_dataset_free(&dataset);
					}
					else if (dataset.n_samples * dataset.feature_dim > 100000000)
					{
						elog(DEBUG1,
							 "neurondb: logistic_regression: dataset too large (%d x %d), skipping GPU",
							 dataset.n_samples,
							 dataset.feature_dim);
						lr_dataset_free(&dataset);
					}
					else
					{
						initStringInfo(&hyperbuf);
						if (hyperbuf.data == NULL)
						{
							elog(DEBUG1,
								 "neurondb: logistic_regression: failed to allocate hyperbuf");
							lr_dataset_free(&dataset);
						}
						else
						{
							elog(DEBUG1,
								 "neurondb: logistic_regression: building hyperparams JSON");
							appendStringInfo(&hyperbuf,
											 "{\"max_iters\":%d,\"learning_rate\":%.6f,\"lambda\":%.6f}",
											 max_iters,
											 learning_rate,
											 lambda);

							if (hyperbuf.data == NULL || hyperbuf.len <= 0)
							{
								elog(DEBUG1,
									 "neurondb: logistic_regression: failed to build hyperparams JSON");
								NDB_FREE(hyperbuf.data);
								lr_dataset_free(&dataset);
							}
							else
							{
								gpu_hyperparams = DatumGetJsonbP(DirectFunctionCall1(
																					 jsonb_in, CStringGetDatum(hyperbuf.data)));

								if (gpu_hyperparams == NULL)
								{
									elog(DEBUG1,
										 "neurondb: logistic_regression: gpu_hyperparams is NULL");
									NDB_FREE(hyperbuf.data);
									lr_dataset_free(&dataset);
								}
								else
								{
									if (!neurondb_gpu_is_available())
									{
										elog(DEBUG1,
											 "neurondb: logistic_regression: GPU no longer available");
										if (gpu_hyperparams != NULL)
											NDB_FREE(gpu_hyperparams);
										NDB_FREE(hyperbuf.data);
										lr_dataset_free(&dataset);
									}
									else
									{
										if (tbl_str == NULL || targ_str == NULL)
										{
											elog(DEBUG1,
												 "neurondb: logistic_regression: NULL table/target strings before GPU call");
											NDB_FREE(gpu_hyperparams);
											NDB_FREE(hyperbuf.data);
											lr_dataset_free(&dataset);
										}
										else if (dataset.features == NULL || dataset.labels == NULL)
										{
											elog(DEBUG1,
												 "neurondb: logistic_regression: NULL dataset arrays before GPU call");
											NDB_FREE(gpu_hyperparams);
											NDB_FREE(hyperbuf.data);
											lr_dataset_free(&dataset);
										}
										else if (dataset.n_samples <= 0 || dataset.feature_dim <= 0)
										{
											elog(DEBUG1,
												 "neurondb: logistic_regression: invalid dataset dimensions before GPU call");
											NDB_FREE(gpu_hyperparams);
											NDB_FREE(hyperbuf.data);
											lr_dataset_free(&dataset);
										}
										else if (!neurondb_gpu_is_available())
										{
											elog(DEBUG1,
												 "neurondb: logistic_regression: GPU no longer available before training call");
											NDB_FREE(gpu_hyperparams);
											NDB_FREE(hyperbuf.data);
											lr_dataset_free(&dataset);
										}
										else
										{
											memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
											gpu_err = NULL;

											gpu_training_succeeded = ndb_gpu_try_train_model("logistic_regression",
																							 NULL,
																							 NULL,
																							 tbl_str,
																							 targ_str,
																							 NULL,
																							 0,
																							 gpu_hyperparams,
																							 dataset.features,
																							 dataset.labels,
																							 dataset.n_samples,
																							 dataset.feature_dim,
																							 2,
																							 &gpu_result,
																							 &gpu_err);

											if (gpu_training_succeeded && gpu_result.spec.model_data != NULL)
											{
												/* Validate model_data size */
												size_t		model_size = VARSIZE(gpu_result.spec.model_data) - VARHDRSZ;

												if (model_size == 0 || model_size > 100000000)
												{
													elog(DEBUG1,
														 "neurondb: logistic_regression: invalid model_data size %zu",
														 model_size);
													gpu_training_succeeded = false;
												}
											}
											else
											{
												gpu_training_succeeded = false;
											}
										}
									}
								}
							}
						}
					}
				}
				PG_CATCH();
				{
					/* Defensive: cleanup on error */
					elog(DEBUG1,
						 "neurondb: logistic_regression: exception during GPU dataset loading or training");
					if (dataset.features != NULL || dataset.labels != NULL)
						lr_dataset_free(&dataset);
					if (gpu_hyperparams != NULL)
					{
						NDB_FREE(gpu_hyperparams);
						gpu_hyperparams = NULL;
					}
					if (hyperbuf.data != NULL)
						NDB_FREE(hyperbuf.data);
					memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
					gpu_training_succeeded = false;
					PG_RE_THROW();
				}
				PG_END_TRY();

				if (gpu_training_succeeded && gpu_result.spec.model_data != NULL)
				{
					MLCatalogModelSpec spec;

					elog(DEBUG1,
						 "neurondb: logistic_regression: GPU training succeeded");
					spec = gpu_result.spec;

					if (spec.training_table == NULL)
						spec.training_table = tbl_str;
					if (spec.training_column == NULL)
						spec.training_column = targ_str;
					if (spec.parameters == NULL)
					{
						spec.parameters = gpu_hyperparams;
						gpu_hyperparams = NULL;
					}

					spec.algorithm = "logistic_regression";
					spec.model_type = "classification";

					model_id = ml_catalog_register_model(&spec);

					if (gpu_err != NULL)
						NDB_FREE(gpu_err);
					if (gpu_hyperparams != NULL)
						NDB_FREE(gpu_hyperparams);
					ndb_gpu_free_train_result(&gpu_result);
					lr_dataset_free(&dataset);
					if (hyperbuf.data != NULL)
						NDB_FREE(hyperbuf.data);
					NDB_FREE(tbl_str);
					tbl_str = NULL;
					NDB_FREE(feat_str);
					feat_str = NULL;
					NDB_FREE(targ_str);
					targ_str = NULL;

					SPI_finish();
					PG_RETURN_INT32(model_id);
				}
				else
				{
					/* GPU training failed - cleanup and fall back to CPU */
					if (gpu_err != NULL)
					{
						elog(DEBUG1,
							 "neurondb: logistic_regression: GPU training failed: %s",
							 gpu_err);
						NDB_FREE(gpu_err);
						gpu_err = NULL;
					}
					else
					{
						elog(DEBUG1,
							 "neurondb: logistic_regression: GPU training failed, falling back to CPU");
					}

					/* Defensive: cleanup GPU result structure */
					ndb_gpu_free_train_result(&gpu_result);
					memset(&gpu_result, 0, sizeof(MLGpuTrainResult));

					/* Defensive: cleanup dataset */
					if (dataset.features != NULL || dataset.labels != NULL)
						lr_dataset_free(&dataset);

					/* Defensive: cleanup hyperparams */
					if (gpu_hyperparams != NULL)
					{
						NDB_FREE(gpu_hyperparams);
						gpu_hyperparams = NULL;
					}

					/* Defensive: cleanup hyperbuf */
					if (hyperbuf.data != NULL)
						NDB_FREE(hyperbuf.data);
				}
			}
		}
	}

	/* CPU training path using chunked streaming */
	/* Use larger chunks for better performance */
	if (nvec > 1000000)
		chunk_size = 100000;	/* 100k chunks for very large datasets */
	else if (nvec > 100000)
		chunk_size = 50000;		/* 50k chunks for large datasets */
	else
		chunk_size = 10000;		/* 10k chunks for smaller datasets */

	/* Initialize streaming accumulator */
	lr_stream_accum_init(&stream_accum, dim);

	/* Initialize weights and bias */
	{
		int			k;

		NDB_ALLOC(weights, double, dim);
		for (k = 0; k < dim; k++)
			weights[k] = 0.0;
	}

	elog(DEBUG1,
		 "neurondb: logistic_regression: processing %d rows in chunks of %d",
		 nvec,
		 chunk_size);

	{
		NDB_DECLARE (NdbSpiSession *, stream_spi_session);
		MemoryContext stream_oldcontext;

		stream_oldcontext = CurrentMemoryContext;
		Assert(stream_oldcontext != NULL);
		NDB_SPI_SESSION_BEGIN(stream_spi_session, stream_oldcontext);

	/* Gradient descent with chunked processing */
	for (iter = 0; iter < max_iters; iter++)
	{
		/* Reset accumulator for this iteration */
		lr_stream_accum_reset(&stream_accum);

		/* Process all data in chunks */
		offset = 0;
		total_rows_processed = 0;
		while (offset < nvec)
		{
			lr_stream_process_chunk(quoted_tbl,
									quoted_feat,
									quoted_label,
									&stream_accum,
									weights,
									bias,
									chunk_size,
									offset,
									&rows_in_chunk);

			if (rows_in_chunk == 0)
				break;

			offset += rows_in_chunk;
			total_rows_processed += rows_in_chunk;

			/* Log progress for large datasets */
			if (offset % 100000 == 0 || offset >= nvec)
			{
				elog(DEBUG1,
					 "neurondb: logistic_regression: iteration %d, processed %d/%d rows (%.1f%%)",
					 iter,
					 offset,
					 nvec,
					 (offset * 100.0) / nvec);
			}
		}

		if (total_rows_processed == 0)
		{
			NDB_SPI_SESSION_END(stream_spi_session);
			lr_stream_accum_free(&stream_accum);
			NDB_FREE(weights);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_logistic_regression: no rows processed in iteration %d",
							iter),
					 errdetail("Iteration %d processed 0 rows, training cannot proceed", iter),
					 errhint("Verify the training table contains valid data and is accessible.")));
		}

		/* Average gradients and add L2 regularization */
		stream_accum.grad_b /= total_rows_processed;
		for (j = 0; j < dim; j++)
		{
			stream_accum.grad_w[j] = stream_accum.grad_w[j] / total_rows_processed + lambda * weights[j];
		}

		/* Update weights and bias */
		bias -= learning_rate * stream_accum.grad_b;
		for (j = 0; j < dim; j++)
			weights[j] -= learning_rate * stream_accum.grad_w[j];

		/* Log progress every 100 iterations */
		if (iter % 100 == 0)
		{
			elog(DEBUG1,
				 "neurondb: logistic_regression: iteration %d/%d completed",
				 iter,
				 max_iters);
			}
		}

	NDB_SPI_SESSION_END(stream_spi_session);
	}

	/* Compute final metrics by processing data in chunks */
	/* Note: For very large datasets, we may want to sample for metrics */
	correct = 0;
	final_loss = 0.0;
	{
		NDB_DECLARE (NdbSpiSession *, metrics_spi_session);
		MemoryContext metrics_oldcontext;

		metrics_oldcontext = CurrentMemoryContext;
		Assert(metrics_oldcontext != NULL);
		NDB_SPI_SESSION_BEGIN(metrics_spi_session, metrics_oldcontext);

	/* Process chunks for final metrics (limit to reasonable size) */
	{
		int			max_samples_for_metrics = (nvec > 100000) ? 100000 : nvec;
		int			metrics_chunk_size = (max_samples_for_metrics > 10000) ? 10000 : max_samples_for_metrics;
		int			metrics_offset = 0;
		int			metrics_rows = 0;
		double		loss_sum = 0.0;
		int			metrics_n = 0;

		while (metrics_offset < max_samples_for_metrics && metrics_n < max_samples_for_metrics)
		{
			StringInfoData metrics_query = {0};
			int			ret;
			int			metrics_i;
			int			metrics_j;
			Oid			feat_type_oid = InvalidOid;
			bool		feat_is_array = false;
			TupleDesc	metrics_tupdesc;
			NDB_DECLARE (float *, row_buffer);

			initStringInfo(&metrics_query);
			appendStringInfo(&metrics_query,
							 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL "
							 "LIMIT %d OFFSET %d",
							 quoted_feat,
							 quoted_label,
							 quoted_tbl,
							 quoted_feat,
							 quoted_label,
							 metrics_chunk_size,
							 metrics_offset);

			ret = ndb_spi_execute(metrics_spi_session, metrics_query.data, true, 0);
			pfree(metrics_query.data);
			metrics_query.data = NULL;
			if (ret != SPI_OK_SELECT)
			{
				break;
			}

			metrics_rows = SPI_processed;
			if (metrics_rows == 0)
			{
				break;
			}

			/* Determine feature type */
			if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
			{
				metrics_tupdesc = SPI_tuptable->tupdesc;
				feat_type_oid = SPI_gettypeid(metrics_tupdesc, 1);
				if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
					feat_is_array = true;
			}

			NDB_ALLOC(row_buffer, float, dim);

			for (metrics_i = 0; metrics_i < metrics_rows && metrics_n < max_samples_for_metrics; metrics_i++)
			{
				HeapTuple	tuple = SPI_tuptable->vals[metrics_i];
				TupleDesc	metrics_row_tupdesc = SPI_tuptable->tupdesc;
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				Vector	   *vec;
				ArrayType  *arr;
				double		y_true;
				double		z;
				double		prediction;
				double		clipped_pred;

				feat_datum = SPI_getbinval(tuple, metrics_row_tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, metrics_row_tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				/* Extract feature vector */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						float8	   *data = (float8 *) ARR_DATA_PTR(arr);

						for (metrics_j = 0; metrics_j < dim; metrics_j++)
							row_buffer[metrics_j] = (float) data[metrics_j];
					}
					else
					{
						float4	   *data = (float4 *) ARR_DATA_PTR(arr);

						memcpy(row_buffer, data, sizeof(float) * dim);
					}
				}
				else
				{
					vec = DatumGetVector(feat_datum);
					if (vec->dim != dim)
						continue;
					memcpy(row_buffer, vec->data, sizeof(float) * dim);
				}

				/* Extract target */
				{
					Oid			targ_type = SPI_gettypeid(metrics_row_tupdesc, 2);

					if (targ_type == INT2OID || targ_type == INT4OID
						|| targ_type == INT8OID)
						y_true = (double) DatumGetInt32(targ_datum);
					else
						y_true = DatumGetFloat8(targ_datum);
				}

				if (y_true != 0.0 && y_true != 1.0)
					continue;

				/* Compute prediction */
				z = bias;
				for (metrics_j = 0; metrics_j < dim; metrics_j++)
					z += weights[metrics_j] * row_buffer[metrics_j];
				prediction = sigmoid(z);

				/* Compute loss */
				clipped_pred = fmax(1e-15, fmin(1.0 - 1e-15, prediction));
				if (y_true > 0.5)
					loss_sum -= log(clipped_pred);
				else
					loss_sum -= log(1.0 - clipped_pred);

				/* Count correct predictions */
				if ((prediction >= 0.5 && y_true > 0.5)
					|| (prediction < 0.5 && y_true <= 0.5))
					correct++;

				metrics_n++;
			}

			NDB_FREE(row_buffer);
			row_buffer = NULL;
			NDB_FREE(metrics_query.data);
			metrics_query.data = NULL;
			metrics_offset += metrics_rows;
		}

		if (metrics_n > 0)
		{
			final_loss = loss_sum / metrics_n;
			/* Scale correct count to full dataset */
			correct = (int) ((double) correct * (double) nvec / (double) metrics_n);
		}
		else
		{
			final_loss = 0.0;
			correct = 0;
		}
	}

	NDB_SPI_SESSION_END(metrics_spi_session);
	}

	lr_stream_accum_free(&stream_accum);

	accuracy_val = (nvec > 0) ? ((double) correct / (double) nvec) : 0.0;

	/* Build LRModel and register in catalog */
	{
		LRModel		model;
		bytea	   *serialized;
		MLCatalogModelSpec spec;
		Jsonb	   *params_jsonb;
		Jsonb	   *metrics_jsonb;
		StringInfoData metricsbuf = {0};

		/* Build model struct */
		/* Validate dim before using it - defensive check */
		if (dim <= 0 || dim > 10000)
		{
			NDB_FREE(weights);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: train_logistic_regression: invalid dim %d before model save",
							dim),
					 errdetail("Feature dimension is %d, must be between 1 and 10000", dim),
					 errhint("This indicates corrupted state. Verify training data.")));
		}

		memset(&model, 0, sizeof(model));
		model.n_features = dim;
		model.n_samples = nvec;
		model.bias = bias;
		{
			int			k;

			NDB_ALLOC(model.weights, double, dim);
			for (k = 0; k < dim; k++)
				model.weights[k] = weights[k];
		}

		elog(DEBUG1,
			 "neurondb: logistic_regression: saving model with n_features=%d, n_samples=%d",
			 model.n_features,
			 model.n_samples);
		model.learning_rate = learning_rate;
		model.lambda = lambda;
		model.max_iters = max_iters;
		model.final_loss = final_loss;
		model.accuracy = accuracy_val;

		/* Serialize model */
		serialized = lr_model_serialize(&model);

		/* Build hyperparameters JSON */
		pfree(hyperbuf.data);
		hyperbuf.data = NULL;
		initStringInfo(&hyperbuf);
		appendStringInfo(&hyperbuf,
						 "{\"max_iters\":%d,\"learning_rate\":%.6f,\"lambda\":%.6f}",
						 max_iters,
						 learning_rate,
						 lambda);
		params_jsonb = DatumGetJsonbP(DirectFunctionCall1(
														  jsonb_in, CStringGetDatum(hyperbuf.data)));
		pfree(hyperbuf.data);
		hyperbuf.data = NULL;

		/* Build metrics JSON - match linear regression format */
		/* Use model.n_features instead of dim to ensure consistency */
		initStringInfo(&metricsbuf);
		appendStringInfo(&metricsbuf,
						 "{\"algorithm\":\"logistic_regression\","
						 "\"storage\":\"cpu\","
						 "\"n_features\":%d,"
						 "\"n_samples\":%d,"
						 "\"final_loss\":%.6f,"
						 "\"accuracy\":%.6f}",
						 model.n_features,
						 model.n_samples,
						 final_loss,
						 model.accuracy);
		metrics_jsonb = DatumGetJsonbP(DirectFunctionCall1(
														   jsonb_in, CStringGetDatum(metricsbuf.data)));
		pfree(metricsbuf.data);
		metricsbuf.data = NULL;

		/* Register in catalog */
		memset(&spec, 0, sizeof(spec));
		spec.algorithm = "logistic_regression";
		spec.model_type = "classification";
		spec.training_table = tbl_str;
		spec.training_column = targ_str;
		spec.parameters = params_jsonb;
		spec.metrics = metrics_jsonb;
		spec.model_data = serialized;
		spec.training_time_ms = -1;
		spec.num_samples = nvec;
		spec.num_features = dim;

		model_id = ml_catalog_register_model(&spec);

		elog(DEBUG1,
			 "logistic_regression: CPU training completed, model_id=%d",
			 model_id);

		/* Cleanup */
		NDB_FREE(model.weights);

		/*
		 * Note: serialized, params_jsonb, metrics_jsonb are owned by catalog
		 * now
		 */
	}

	NDB_FREE(weights);
	NDB_FREE(tbl_str);
	NDB_FREE(feat_str);
	NDB_FREE(targ_str);
	PG_RETURN_INT32(model_id);
}

/*
 * predict_logistic_regression
 *
 * Predicts probability for logistic regression
 */
PG_FUNCTION_INFO_V1(predict_logistic_regression);

Datum
predict_logistic_regression(PG_FUNCTION_ARGS)
{
	ArrayType  *coef_array;
	Vector	   *features;
	int			ncoef;
	float8	   *coef;
	float	   *x;
	int			dim;
	double		z;
	double		probability;
	int			i;

	coef_array = PG_GETARG_ARRAYTYPE_P(0);
	features = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(features);

	/* Extract coefficients */
	if (ARR_NDIM(coef_array) != 1)
		ereport(ERROR,
				(errmsg("Coefficients must be 1-dimensional array")));

	ncoef = ARR_DIMS(coef_array)[0];
	coef = (float8 *) ARR_DATA_PTR(coef_array);

	x = features->data;
	dim = features->dim;

	if (ncoef != dim + 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Coefficient dimension mismatch: expected %d, got %d",
						dim + 1,
						ncoef)));

	/* Compute z = bias + w1*x1 + w2*x2 + ... */
	z = coef[0];				/* bias */
	for (i = 0; i < dim; i++)
		z += coef[i + 1] * x[i];

	/* Apply sigmoid */
	probability = sigmoid(z);

	PG_RETURN_FLOAT8(probability);
}

/*
 * predict_logistic_regression(model_id, features)
 *
 * Predicts using a model loaded from the catalog by model_id.
 * Supports both CPU and GPU models.
 */
PG_FUNCTION_INFO_V1(predict_logistic_regression_model_id);

Datum
predict_logistic_regression_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	Vector	   *features;
	LRModel    *model = NULL;
	double		probability;
	double		z;
	int			i;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("logistic_regression: model_id is "
						"required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("logistic_regression: features vector "
						"is required")));

	features = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(features);

	/* Try GPU prediction first */
	if (lr_try_gpu_predict_catalog(model_id, features, &probability))
	{
		elog(DEBUG1,
			 "logistic_regression: GPU prediction succeeded, probability=%.6f",
			 probability);
		PG_RETURN_FLOAT8(probability);
	}
	else
	{
		elog(DEBUG1,
			 "logistic_regression: GPU prediction failed or not available, trying CPU");
	}

	/* Load model from catalog - match linear regression error handling */
	if (!lr_load_model_from_catalog(model_id, &model))
	{
		/* Check if model is GPU-only */
		bytea	   *payload = NULL;
		Jsonb	   *metrics = NULL;
		bool		is_gpu = false;

		if (ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		{
			is_gpu = lr_metadata_is_gpu(metrics);
			if (payload)
			{
				NDB_FREE(payload);
				payload = NULL;
			}
			if (metrics)
			{
				NDB_FREE(metrics);
				metrics = NULL;
			}
		}

		if (is_gpu)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: logistic_regression: model %d is GPU-only and GPU prediction failed. "
							"Please ensure GPU is available and properly configured.",
							model_id)));
		else
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: logistic_regression: model %d not found",
							model_id)));
	}

	/* Validate feature dimension - match linear regression */
	if (model->n_features > 0 && features->dim != model->n_features)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: logistic_regression: feature dimension mismatch (expected %d, got %d)",
						model->n_features,
						features->dim)));

	/* Compute z = bias + w1*x1 + w2*x2 + ... */
	z = model->bias;
	for (i = 0; i < model->n_features && i < features->dim; i++)
		z += model->weights[i] * features->data[i];

	/* Apply sigmoid */
	probability = sigmoid(z);

	/* Cleanup */
	if (model != NULL)
	{
		if (model->weights != NULL)
		{
			NDB_FREE(model->weights);
			model->weights = NULL;
		}
		NDB_FREE(model);
		model = NULL;
	}

	PG_RETURN_FLOAT8(probability);
}

/*
 * evaluate_logistic_regression
 *
 * Evaluates model performance (accuracy, precision, recall, F1, AUC)
 * Returns: [accuracy, precision, recall, f1_score, log_loss]
 */
PG_FUNCTION_INFO_V1(evaluate_logistic_regression);

Datum
evaluate_logistic_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	ArrayType  *coef_array;
	double		threshold = PG_GETARG_FLOAT8(4);
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			ncoef;
	float8	   *coef;
	int			tp = 0,
				tn = 0,
				fp = 0,
				fn = 0;
	double		log_loss = 0.0;
	double		accuracy,
				precision,
				recall,
				f1_score;
	int			i,
				j;
	Datum	   *result_datums;
	ArrayType  *result_array;
	MemoryContext oldcontext;
	NDB_DECLARE (NdbSpiSession *, eval_spi_session);

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);
	coef_array = PG_GETARG_ARRAYTYPE_P(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	/* Extract coefficients */
	if (ARR_NDIM(coef_array) != 1)
		ereport(ERROR,
				(errmsg("Coefficients must be 1-dimensional array")));

	ncoef = ARR_DIMS(coef_array)[0];
	(void) ncoef;				/* Suppress unused variable warning */
	coef = (float8 *) ARR_DATA_PTR(coef_array);

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(eval_spi_session, oldcontext);

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str,
					 targ_str,
					 tbl_str,
					 feat_str,
					 targ_str);

	ret = ndb_spi_execute(eval_spi_session, query.data, true, 0);
	pfree(query.data);
	query.data = NULL;
	if (ret != SPI_OK_SELECT)
	{
		NDB_SPI_SESSION_END(eval_spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_logistic_regression: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table and columns exist and are accessible.")));
	}

	nvec = SPI_processed;

	/* Compute predictions and metrics */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec;
		double		y_true;
		double		z;
		double		probability;
		int			y_pred;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		vec = DatumGetVector(feat_datum);
		y_true = DatumGetFloat8(targ_datum);

		/* Compute probability */
		z = coef[0];			/* bias */
		for (j = 0; j < vec->dim; j++)
			z += coef[j + 1] * vec->data[j];
		probability = sigmoid(z);

		/* Apply threshold */
		y_pred = (probability >= threshold) ? 1 : 0;

		/* Update confusion matrix */
		if (y_true > 0.5)
		{
			if (y_pred == 1)
				tp++;
			else
				fn++;
		}
		else
		{
			if (y_pred == 1)
				fp++;
			else
				tn++;
		}

		/* Update log loss */
		probability = fmax(1e-15, fmin(1.0 - 1e-15, probability));
		if (y_true > 0.5)
			log_loss -= log(probability);
		else
			log_loss -= log(1.0 - probability);
	}

	log_loss /= nvec;

	NDB_SPI_SESSION_END(eval_spi_session);

	/* Compute metrics */
	accuracy = (double) (tp + tn) / (tp + tn + fp + fn);
	precision = (tp + fp > 0) ? (double) tp / (tp + fp) : 0.0;
	recall = (tp + fn > 0) ? (double) tp / (tp + fn) : 0.0;
	f1_score = (precision + recall > 0)
		? 2.0 * precision * recall / (precision + recall)
		: 0.0;

	/* Build result array: [accuracy, precision, recall, f1_score, log_loss] */
	MemoryContextSwitchTo(oldcontext);

	NDB_ALLOC(result_datums, Datum, 5);
	result_datums[0] = Float8GetDatum(accuracy);
	result_datums[1] = Float8GetDatum(precision);
	result_datums[2] = Float8GetDatum(recall);
	result_datums[3] = Float8GetDatum(f1_score);
	result_datums[4] = Float8GetDatum(log_loss);

	result_array = construct_array(result_datums,
								   5,
								   FLOAT8OID,
								   sizeof(float8),
								   FLOAT8PASSBYVAL,
								   'd');

	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(feat_str);
	NDB_FREE(targ_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * evaluate_logistic_regression_by_model_id
 *
 * One-shot evaluation function: loads model, fetches all data in one query,
 * loops through rows in C, computes predictions and metrics, returns jsonb.
 * This is much more efficient than calling predict() for each row in SQL.
 * Supports both CPU and GPU models.
 */
PG_FUNCTION_INFO_V1(evaluate_logistic_regression_by_model_id);

Datum
evaluate_logistic_regression_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	double		threshold = PG_ARGISNULL(4) ? 0.5 : PG_GETARG_FLOAT8(4);
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	double		log_loss = 0.0;

	/* tp, tn, fp, fn not used in GPU path - metrics computed by GPU kernel */
	double		accuracy,
				precision,
				recall,
				f1_score;
	int			i;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;
	Jsonb	   *result_jsonb;
	StringInfoData jsonbuf;
	bytea	   *gpu_payload = NULL;
	Jsonb	   *gpu_metrics = NULL;
	bool		is_gpu_model = false;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_logistic_regression_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_logistic_regression_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	/* Load model from catalog - check GPU model directly */
	{
		if (ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
		{
			is_gpu_model = lr_metadata_is_gpu(gpu_metrics);
			if (!is_gpu_model)
			{
				/*
				 * CPU model - would need to load here if we had CPU model
				 * loading
				 */
				NDB_FREE(gpu_payload);
				NDB_FREE(gpu_metrics);
				gpu_payload = NULL;
				gpu_metrics = NULL;
			}
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_logistic_regression_by_model_id: model %d not found",
							model_id)));
		}
	}

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		if (ret != SPI_OK_CONNECT)
		{
			SPI_finish();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: SPI_connect failed")));
		}
	{
		NDB_FREE(gpu_payload);
		NDB_FREE(gpu_metrics);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_logistic_regression_by_model_id: SPI_connect failed")));
	}

	/* Build query - single query to fetch all data */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quote_identifier(feat_str),
					 quote_identifier(targ_str),
					 quote_identifier(tbl_str),
					 quote_identifier(feat_str),
					 quote_identifier(targ_str));

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(gpu_payload);
		NDB_FREE(gpu_metrics);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_logistic_regression_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		NDB_FREE(gpu_payload);
		NDB_FREE(gpu_metrics);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_logistic_regression_by_model_id: no valid rows found")));
	}

	/* Determine feature column type */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/*
	 * GPU batch evaluation path for GPU models - uses optimized evaluation
	 * kernel
	 */
	if (is_gpu_model && neurondb_gpu_is_available())
	{
#ifdef NDB_GPU_CUDA
		const		NdbCudaLrModelHeader *gpu_hdr;
		double	   *h_labels = NULL;
		float	   *h_features = NULL;
		int			feat_dim = 0;
		int			valid_rows = 0;
		size_t		payload_size;

		/* Defensive check: validate payload size */
		payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
		if (payload_size < sizeof(NdbCudaLrModelHeader))
		{
			elog(DEBUG1,
				 "evaluate_logistic_regression_by_model_id: GPU payload too small (%zu bytes), falling back to CPU",
				 payload_size);
			goto cpu_evaluation_path;
		}

		/* Load GPU model header with defensive checks */
		gpu_hdr = (const NdbCudaLrModelHeader *) VARDATA(gpu_payload);
		if (gpu_hdr == NULL)
		{
			elog(DEBUG1,
				 "evaluate_logistic_regression_by_model_id: NULL GPU header, falling back to CPU");
			goto cpu_evaluation_path;
		}

		feat_dim = gpu_hdr->feature_dim;
		if (feat_dim <= 0 || feat_dim > 100000)
		{
			elog(DEBUG1,
				 "evaluate_logistic_regression_by_model_id: invalid feature_dim (%d), falling back to CPU",
				 feat_dim);
			goto cpu_evaluation_path;
		}

		/* Allocate host buffers for features and labels with size checks */
		{
			size_t		features_size = sizeof(float) * (size_t) nvec * (size_t) feat_dim;
			size_t		labels_size = sizeof(double) * (size_t) nvec;

			if (features_size > MaxAllocSize || labels_size > MaxAllocSize)
			{
				elog(DEBUG1,
					 "evaluate_logistic_regression_by_model_id: allocation size too large (features=%zu, labels=%zu), falling back to CPU",
					 features_size, labels_size);
				goto cpu_evaluation_path;
			}

			NDB_ALLOC(h_features, float, (size_t) nvec * (size_t) feat_dim);
			NDB_ALLOC(h_labels, double, (size_t) nvec);
		}

		/*
		 * Extract features and labels from SPI results - optimized batch
		 * extraction
		 */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			if (tupdesc == NULL)
			{
				elog(DEBUG1,
					 "evaluate_logistic_regression_by_model_id: NULL TupleDesc, falling back to CPU");
				NDB_FREE(h_features);
				h_features = NULL;
				NDB_FREE(h_labels);
				h_labels = NULL;
				goto cpu_evaluation_path;
			}

			for (i = 0; i < nvec; i++)
			{
				HeapTuple	tuple;
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				Vector	   *vec;
				ArrayType  *arr;
				float	   *feat_row;

				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || i >= SPI_processed)
					break;

				tuple = SPI_tuptable->vals[i];
				if (tuple == NULL)
					continue;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				/* Bounds check */
				if (valid_rows >= nvec)
				{
					elog(DEBUG1,
						 "evaluate_logistic_regression_by_model_id: valid_rows overflow, breaking");
					break;
				}

				feat_row = h_features + (valid_rows * feat_dim);
				if (feat_row == NULL || feat_row < h_features || feat_row >= h_features + (nvec * feat_dim))
				{
					elog(DEBUG1,
						 "evaluate_logistic_regression_by_model_id: feat_row out of bounds, skipping row");
					continue;
				}

				h_labels[valid_rows] = DatumGetFloat8(targ_datum);

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						/* Optimized: bulk conversion with loop unrolling hint */
						float8	   *data = (float8 *) ARR_DATA_PTR(arr);
						int			j;
						int			j_remain = feat_dim % 4;
						int			j_end = feat_dim - j_remain;

						/*
						 * Process 4 elements at a time for better cache
						 * locality
						 */
						for (j = 0; j < j_end; j += 4)
						{
							feat_row[j] = (float) data[j];
							feat_row[j + 1] = (float) data[j + 1];
							feat_row[j + 2] = (float) data[j + 2];
							feat_row[j + 3] = (float) data[j + 3];
						}
						/* Handle remaining elements */
						for (j = j_end; j < feat_dim; j++)
							feat_row[j] = (float) data[j];
					}
					else
					{
						/* FLOAT4ARRAYOID: direct memcpy (already optimal) */
						float4	   *data = (float4 *) ARR_DATA_PTR(arr);

						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					/* Vector type: direct memcpy (already optimal) */
					vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_rows++;
			}
		}

		if (valid_rows == 0)
		{
			NDB_FREE(h_features);
			h_features = NULL;
			NDB_FREE(h_labels);
			h_labels = NULL;
			if (gpu_payload)
			{
				NDB_FREE(gpu_payload);
				gpu_payload = NULL;
			}
			if (gpu_metrics)
			{
				NDB_FREE(gpu_metrics);
				gpu_metrics = NULL;
			}
			NDB_FREE(tbl_str);
			tbl_str = NULL;
			NDB_FREE(feat_str);
			feat_str = NULL;
			NDB_FREE(targ_str);
			targ_str = NULL;
			SPI_finish();
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_logistic_regression_by_model_id: no valid rows found")));
		}

		/* Use optimized GPU batch evaluation */
		{
			int			rc;
			char	   *gpu_errstr = NULL;

			/* Defensive checks before GPU call */
			if (h_features == NULL || h_labels == NULL || valid_rows <= 0 || feat_dim <= 0)
			{
				elog(DEBUG1,
					 "evaluate_logistic_regression_by_model_id: invalid inputs for GPU evaluation (features=%p, labels=%p, rows=%d, dim=%d), falling back to CPU",
					 (void *) h_features, (void *) h_labels, valid_rows, feat_dim);
				NDB_FREE(h_features);
				h_features = NULL;
				NDB_FREE(h_labels);
				h_labels = NULL;
				goto cpu_evaluation_path;
			}

			PG_TRY();
			{
				rc = ndb_cuda_lr_evaluate(gpu_payload,
										  h_features,
										  h_labels,
										  valid_rows,
										  feat_dim,
										  threshold,
										  &accuracy,
										  &precision,
										  &recall,
										  &f1_score,
										  &log_loss,
										  &gpu_errstr);

				if (rc == 0)
				{
					/* Success - build result and return */
					nvec = valid_rows;
					initStringInfo(&jsonbuf);
					appendStringInfo(&jsonbuf,
									 "{\"accuracy\":%.6f,\"precision\":%.6f,\"recall\":%.6f,\"f1_score\":%.6f,\"log_loss\":%.6f,\"n_samples\":%d}",
									 accuracy,
									 precision,
									 recall,
									 f1_score,
									 log_loss,
									 nvec);

					{
						Jsonb	   *temp_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
																					CStringGetDatum(jsonbuf.data)));
						MemoryContext oldcontext = CurrentMemoryContext;

						MemoryContextSwitchTo(oldcontext);
						result_jsonb = (Jsonb *) PG_DETOAST_DATUM_COPY((Datum) temp_jsonb);
						if (temp_jsonb != (Jsonb *) DatumGetPointer((Datum) temp_jsonb))
						{
							NDB_FREE(temp_jsonb);
							temp_jsonb = NULL;
						}
					}

					NDB_FREE(jsonbuf.data);
					jsonbuf.data = NULL;
					NDB_FREE(h_features);
					h_features = NULL;
					NDB_FREE(h_labels);
					h_labels = NULL;
					if (gpu_payload)
					{
						NDB_FREE(gpu_payload);
						gpu_payload = NULL;
					}
					if (gpu_metrics)
					{
						NDB_FREE(gpu_metrics);
						gpu_metrics = NULL;
					}
					if (gpu_errstr)
					{
						NDB_FREE(gpu_errstr);
						gpu_errstr = NULL;
					}
					NDB_FREE(tbl_str);
					tbl_str = NULL;
					NDB_FREE(feat_str);
					feat_str = NULL;
					NDB_FREE(targ_str);
					targ_str = NULL;
					SPI_finish();
					PG_RETURN_JSONB_P(result_jsonb);
				}
				else
				{
					/* GPU evaluation failed - fall back to CPU */
					elog(DEBUG1,
						 "evaluate_logistic_regression_by_model_id: GPU batch evaluation failed: %s, falling back to CPU",
						 gpu_errstr ? gpu_errstr : "unknown error");
					if (gpu_errstr)
					{
						NDB_FREE(gpu_errstr);
						gpu_errstr = NULL;
					}
					NDB_FREE(h_features);
					h_features = NULL;
					NDB_FREE(h_labels);
					h_labels = NULL;
					goto cpu_evaluation_path;
				}
			}
			PG_CATCH();
			{
				elog(DEBUG1,
					 "evaluate_logistic_regression_by_model_id: exception during GPU evaluation, falling back to CPU");
				if (h_features)
				{
					NDB_FREE(h_features);
					h_features = NULL;
				}
				if (h_labels)
				{
					NDB_FREE(h_labels);
					h_labels = NULL;
				}
				goto cpu_evaluation_path;
			}
			PG_END_TRY();
		}
#endif							/* NDB_GPU_CUDA */
	}
#ifndef NDB_GPU_CUDA
	/* When CUDA is not available, always use CPU path */
	if (false)
	{
	}
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-label"
cpu_evaluation_path:
#pragma GCC diagnostic pop
	/* CPU evaluation path */
	{
		LRModel    *model = NULL;
		int			tp = 0,
					tn = 0,
					fp = 0,
					fn = 0;
		int			correct_predictions = 0;
		int			total_predictions = 0;

		/* Load model from catalog - try CPU first, then GPU if CPU fails */
		if (!lr_load_model_from_catalog(model_id, &model))
		{
			/*
			 * If CPU model loading failed, try to load GPU model for CPU
			 * evaluation
			 */
			bytea	   *gpu_model_payload = NULL;
			Jsonb	   *gpu_model_metrics = NULL;

			if (ml_catalog_fetch_model_payload(model_id, &gpu_model_payload, NULL, &gpu_model_metrics))
			{
				if (gpu_model_payload != NULL && lr_metadata_is_gpu(gpu_model_metrics))
				{
					/*
					 * This is a GPU model - we can't evaluate GPU models on
					 * CPU yet
					 */
					if (gpu_model_payload)
					{
						NDB_FREE(gpu_model_payload);
						gpu_model_payload = NULL;
					}
					if (gpu_model_metrics)
					{
						NDB_FREE(gpu_model_metrics);
						gpu_model_metrics = NULL;
					}
					if (gpu_payload)
					{
						NDB_FREE(gpu_payload);
						gpu_payload = NULL;
					}
					if (gpu_metrics)
					{
						NDB_FREE(gpu_metrics);
						gpu_metrics = NULL;
					}
					NDB_FREE(tbl_str);
					tbl_str = NULL;
					NDB_FREE(feat_str);
					feat_str = NULL;
					NDB_FREE(targ_str);
					targ_str = NULL;
					SPI_finish();
					ereport(ERROR,
							(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
							 errmsg("neurondb: evaluate_logistic_regression_by_model_id: model %d is GPU-trained and cannot be evaluated on CPU yet. Please ensure GPU is available for evaluation.",
									model_id)));
				}
				if (gpu_model_payload)
				{
					NDB_FREE(gpu_model_payload);
					gpu_model_payload = NULL;
				}
				if (gpu_model_metrics)
				{
					NDB_FREE(gpu_model_metrics);
					gpu_model_metrics = NULL;
				}
			}

			/* If we get here, the model doesn't exist at all */
			if (gpu_payload)
			{
				NDB_FREE(gpu_payload);
				gpu_payload = NULL;
			}
			if (gpu_metrics)
			{
				NDB_FREE(gpu_metrics);
				gpu_metrics = NULL;
			}
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			SPI_finish();
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_logistic_regression_by_model_id: model %d not found",
							model_id)));
		}

		/* Re-execute query for CPU evaluation */
		ret = ndb_spi_execute_safe(query.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT)
		{
			NDB_FREE(model->weights);
			NDB_FREE(model);
			NDB_FREE(gpu_payload);
			NDB_FREE(gpu_metrics);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			SPI_finish();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: evaluate_logistic_regression_by_model_id: query failed for CPU evaluation")));
		}

		nvec = SPI_processed;
		if (nvec < 2)
		{
			if (model->weights)
			{
				NDB_FREE(model->weights);
				model->weights = NULL;
			}
			NDB_FREE(model);
			model = NULL;
			if (gpu_payload)
			{
				NDB_FREE(gpu_payload);
				gpu_payload = NULL;
			}
			if (gpu_metrics)
			{
				NDB_FREE(gpu_metrics);
				gpu_metrics = NULL;
			}
			NDB_FREE(tbl_str);
			tbl_str = NULL;
			NDB_FREE(feat_str);
			feat_str = NULL;
			NDB_FREE(targ_str);
			SPI_finish();
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_logistic_regression_by_model_id: need at least 2 samples for CPU evaluation, got %d",
							nvec)));
		}

		/* Process each row for CPU evaluation */
		for (i = 0; i < nvec; i++)
		{
			HeapTuple	tuple = SPI_tuptable->vals[i];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			Datum		feat_datum;
			Datum		targ_datum;
			bool		feat_null;
			bool		targ_null;
			Vector	   *vec;
			ArrayType  *arr;
			int			actual_dim;
			double		y_true;
			double		y_pred_prob;
			int			y_pred_class;
			int			y_true_class;
			double		z;
			int			eval_j;

			feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
			targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

			if (feat_null || targ_null)
				continue;

			y_true = DatumGetFloat8(targ_datum);

			/* Extract features and determine dimension */
			if (feat_is_array)
			{
				arr = DatumGetArrayTypeP(feat_datum);
				if (ARR_NDIM(arr) != 1)
				{
					NDB_FREE(model->weights);
					NDB_FREE(model);
					NDB_FREE(gpu_payload);
					NDB_FREE(gpu_metrics);
					NDB_FREE(tbl_str);
					NDB_FREE(feat_str);
					NDB_FREE(targ_str);
					SPI_finish();
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: evaluate_logistic_regression_by_model_id: features array must be 1-D")));
				}
				actual_dim = ARR_DIMS(arr)[0];
			}
			else
			{
				vec = DatumGetVector(feat_datum);
				actual_dim = vec->dim;
			}

			/* Validate feature dimension */
			if (actual_dim != model->n_features)
			{
				NDB_FREE(model->weights);
				NDB_FREE(model);
				NDB_FREE(gpu_payload);
				NDB_FREE(gpu_metrics);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);
				SPI_finish();
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_logistic_regression_by_model_id: feature dimension mismatch (expected %d, got %d)",
								model->n_features,
								actual_dim)));
			}

			/* Compute prediction using model */
			z = model->bias;

			if (feat_is_array)
			{
				if (feat_type_oid == FLOAT8ARRAYOID)
				{
					double	   *feat_data = (double *) ARR_DATA_PTR(arr);

					for (eval_j = 0; eval_j < model->n_features; eval_j++)
						z += model->weights[eval_j] * feat_data[eval_j];
				}
				else
				{
					float	   *feat_data = (float *) ARR_DATA_PTR(arr);

					for (eval_j = 0; eval_j < model->n_features; eval_j++)
						z += model->weights[eval_j] * (double) feat_data[eval_j];
				}
			}
			else
			{
				for (eval_j = 0; eval_j < model->n_features && eval_j < vec->dim; eval_j++)
					z += model->weights[eval_j] * vec->data[eval_j];
			}

			/* Apply sigmoid to get probability */
			y_pred_prob = 1.0 / (1.0 + exp(-z));

			/* Convert probability to class prediction */
			y_pred_class = (y_pred_prob >= threshold) ? 1 : 0;

			/* Convert true label to class */
			y_true_class = (y_true >= 0.5) ? 1 : 0;

			/* Update confusion matrix */
			if (y_true_class == 1 && y_pred_class == 1)
				tp++;
			else if (y_true_class == 0 && y_pred_class == 0)
				tn++;
			else if (y_true_class == 0 && y_pred_class == 1)
				fp++;
			else if (y_true_class == 1 && y_pred_class == 0)
				fn++;

			/* Compute log loss with numerical stability */
			if (y_pred_prob > 1e-15 && y_pred_prob < 1.0 - 1e-15)
			{
				log_loss += -(y_true * log(y_pred_prob) + (1.0 - y_true) * log(1.0 - y_pred_prob));
			}
			else if (y_pred_prob <= 1e-15)
			{
				/* Prediction is essentially 0, log loss for positive class */
				log_loss += -(y_true * -30.0 + (1.0 - y_true) * 0.0);	/* log(1e-15) ≈ -30 */
			}
			else if (y_pred_prob >= 1.0 - 1e-15)
			{
				/* Prediction is essentially 1, log loss for negative class */
				log_loss += -(y_true * 0.0 + (1.0 - y_true) * -30.0);	/* log(1e-15) ≈ -30 */
			}

			total_predictions++;
		}

		/* Compute final metrics */
		correct_predictions = tp + tn;
		accuracy = (total_predictions > 0) ? (double) correct_predictions / total_predictions : 0.0;
		precision = (tp + fp > 0) ? (double) tp / (tp + fp) : 0.0;
		recall = (tp + fn > 0) ? (double) tp / (tp + fn) : 0.0;
		f1_score = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
		log_loss = (total_predictions > 0) ? log_loss / total_predictions : 0.0;

		/* Cleanup before creating JSONB */
		if (model->weights)
		{
			NDB_FREE(model->weights);
			model->weights = NULL;
		}
		NDB_FREE(model);
		model = NULL;
		NDB_FREE(gpu_payload);
		gpu_payload = NULL;
		NDB_FREE(gpu_metrics);
		gpu_metrics = NULL;
		NDB_FREE(tbl_str);
		tbl_str = NULL;
		NDB_FREE(feat_str);
		feat_str = NULL;
		NDB_FREE(targ_str);
		targ_str = NULL;
		SPI_finish();

		/* Build result JSON AFTER SPI_finish in parent context */
		initStringInfo(&jsonbuf);
		appendStringInfo(&jsonbuf,
						 "{\"accuracy\":%.6f,\"precision\":%.6f,\"recall\":%.6f,\"f1_score\":%.6f,\"log_loss\":%.6f,\"n_samples\":%d,\"tp\":%d,\"tn\":%d,\"fp\":%d,\"fn\":%d}",
						 accuracy, precision, recall, f1_score, log_loss, total_predictions, tp, tn, fp, fn);

		result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(jsonbuf.data)));
		NDB_FREE(jsonbuf.data);
		jsonbuf.data = NULL;

		PG_RETURN_JSONB_P(result_jsonb);
	}
}

static void
lr_dataset_init(LRDataset * dataset)
{
	if (dataset == NULL)
		return;
	memset(dataset, 0, sizeof(LRDataset));
}

static void
lr_dataset_free(LRDataset * dataset)
{
	if (dataset == NULL)
		return;
	NDB_FREE(dataset->features);
	NDB_FREE(dataset->labels);
	lr_dataset_init(dataset);
}

/*
 * lr_dataset_load_limited
 *
 * Load dataset with LIMIT clause to avoid loading too much data for GPU
 */
static void
lr_dataset_load_limited(const char *quoted_tbl,
						const char *quoted_feat,
						const char *quoted_label,
						LRDataset * dataset,
						int max_rows)
{
	StringInfoData query = {0};
	MemoryContext oldcontext;
	int			ret = 0;
	int			n_samples = 0;
	int			feature_dim = 0;
	int			i;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;
	NDB_DECLARE (NdbSpiSession *, load_spi_session);

	if (dataset == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_dataset_load_limited: dataset is NULL")));

	if (max_rows <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_dataset_load_limited: max_rows must be positive")));

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);

	NDB_SPI_SESSION_BEGIN(load_spi_session, oldcontext);

	/* Initialize query - use ndb_spi_stringinfo_init for proper context handling */
	ndb_spi_stringinfo_init(load_spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %d",
					 quoted_feat,
					 quoted_label,
					 quoted_tbl,
					 quoted_feat,
					 quoted_label,
					 max_rows);

	ret = ndb_spi_execute(load_spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(load_spi_session, &query);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_dataset_load_limited: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table exists and contains valid feature and label columns.")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		ndb_spi_stringinfo_free(load_spi_session, &query);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_dataset_load_limited: need at least 10 samples, got %d",
						n_samples),
				 errdetail("Dataset contains %d rows, minimum required is 10", n_samples),
				 errhint("Add more data rows to the training table.")));
	}

	/* Determine feature column type and dimension before allocating */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/* Get feature dimension from first row before allocating */
	if (SPI_processed > 0)
	{
		HeapTuple	first_tuple = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		bool		feat_null;
		Vector	   *vec;
		ArrayType  *arr;

		feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &feat_null);
		if (!feat_null)
		{
			if (feat_is_array)
			{
				arr = DatumGetArrayTypeP(feat_datum);
				if (ARR_NDIM(arr) == 1)
					feature_dim = ARR_DIMS(arr)[0];
			}
			else
			{
				vec = DatumGetVector(feat_datum);
				feature_dim = vec->dim;
			}
		}
	}

	if (feature_dim <= 0)
	{
		ndb_spi_stringinfo_free(load_spi_session, &query);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_dataset_load_limited: could not determine feature dimension"),
				 errdetail("Feature dimension is %d (must be > 0)", feature_dim),
				 errhint("Ensure feature column contains valid vector or array data.")));
	}

	MemoryContextSwitchTo(oldcontext);
	NDB_ALLOC(dataset->features, float, (size_t) n_samples * (size_t) feature_dim);
	NDB_ALLOC(dataset->labels, double, (size_t) n_samples);

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec;
		ArrayType  *arr;
		float	   *row;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		if (feat_null)
			continue;

		row = dataset->features + (i * feature_dim);
		if (feat_is_array)
		{
			arr = DatumGetArrayTypeP(feat_datum);
				if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feature_dim)
			{
				ndb_spi_stringinfo_free(load_spi_session, &query);
				NDB_SPI_SESSION_END(load_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: lr_dataset_load_limited: inconsistent array feature dimensions"),
						 errdetail("Array has %d dimensions, first row had %d features", ARR_NDIM(arr), feature_dim),
						 errhint("Ensure all feature arrays have consistent dimensions.")));
			}
			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				float8	   *data = (float8 *) ARR_DATA_PTR(arr);
				int			j;

				for (j = 0; j < feature_dim; j++)
					row[j] = (float) data[j];
			}
			else
			{
				float4	   *data = (float4 *) ARR_DATA_PTR(arr);

				memcpy(row, data, sizeof(float) * feature_dim);
			}
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			if (vec->dim != feature_dim)
			{
				ndb_spi_stringinfo_free(load_spi_session, &query);
				NDB_SPI_SESSION_END(load_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: lr_dataset_load_limited: inconsistent vector dimensions"),
						 errdetail("Vector has %d dimensions, first row had %d features", vec->dim, feature_dim),
						 errhint("Ensure all feature vectors have consistent dimensions.")));
			}
			memcpy(row, vec->data, sizeof(float) * feature_dim);
		}

		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (targ_null)
			continue;

		{
			Oid			targ_type = SPI_gettypeid(tupdesc, 2);

			if (targ_type == INT2OID || targ_type == INT4OID
				|| targ_type == INT8OID)
				dataset->labels[i] =
					(double) DatumGetInt32(targ_datum);
			else
				dataset->labels[i] = DatumGetFloat8(targ_datum);
		}

		if (dataset->labels[i] != 0.0 && dataset->labels[i] != 1.0)
		{
			ndb_spi_stringinfo_free(load_spi_session, &query);
			NDB_SPI_SESSION_END(load_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: lr_dataset_load_limited: binary target required (0 or 1)"),
					 errdetail("Row %d has target value %.6f, must be 0.0 or 1.0", i, dataset->labels[i]),
					 errhint("Ensure all target values are binary (0 or 1) for logistic regression.")));
		}
	}

	dataset->n_samples = n_samples;
	dataset->feature_dim = feature_dim;

	ndb_spi_stringinfo_free(load_spi_session, &query);
	NDB_SPI_SESSION_END(load_spi_session);
}

/*
 * lr_stream_accum_init
 *
 * Initialize streaming accumulator for chunked gradient descent
 */
static void
lr_stream_accum_init(LRStreamAccum * accum, int dim)
{
	if (accum == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_stream_accum_init: accum is NULL")));

	if (dim <= 0 || dim > 10000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_stream_accum_init: invalid feature dimension %d",
						dim)));

	memset(accum, 0, sizeof(LRStreamAccum));

	accum->feature_dim = dim;
	accum->n_samples = 0;
	accum->initialized = false;

	/* Allocate gradient vector */
	accum->grad_w = (double *) palloc0(sizeof(double) * dim);
	if (accum->grad_w == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("neurondb: lr_stream_accum_init: failed to allocate gradient vector")));

	accum->grad_b = 0.0;
	accum->initialized = true;
}

/*
 * lr_stream_accum_free
 *
 * Free memory allocated for streaming accumulator
 */
static void
lr_stream_accum_free(LRStreamAccum * accum)
{
	if (accum == NULL)
		return;

	if (accum->grad_w != NULL)
	{
		NDB_FREE(accum->grad_w);
		accum->grad_w = NULL;
	}

	memset(accum, 0, sizeof(LRStreamAccum));
}

/*
 * lr_stream_accum_reset
 *
 * Reset accumulator for next iteration (keeps memory allocated)
 */
static void
lr_stream_accum_reset(LRStreamAccum * accum)
{
	int			i;

	if (accum == NULL || !accum->initialized)
		return;

	for (i = 0; i < accum->feature_dim; i++)
		accum->grad_w[i] = 0.0;
	accum->grad_b = 0.0;
	accum->n_samples = 0;
}

/*
 * lr_stream_process_chunk
 *
 * Process a chunk of data, computing gradients for gradient descent
 * Returns number of rows processed in this chunk
 */
static void
lr_stream_process_chunk(const char *quoted_tbl,
						const char *quoted_feat,
						const char *quoted_label,
						LRStreamAccum * accum,
						double *weights,
						double bias,
						int chunk_size,
						int offset,
						int *rows_processed)
{
	StringInfoData query = {0};
	int			ret;
	int			i;
	int			j;
	int			n_rows;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;
	NDB_DECLARE (float *, row_buffer);

	if (quoted_tbl == NULL || quoted_feat == NULL || quoted_label == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_stream_process_chunk: NULL parameter")));

	if (accum == NULL || !accum->initialized)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_stream_process_chunk: accumulator not initialized")));

	if (weights == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_stream_process_chunk: weights is NULL")));

	if (chunk_size <= 0 || chunk_size > 100000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_stream_process_chunk: invalid chunk_size %d",
						chunk_size)));

	if (rows_processed == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_stream_process_chunk: rows_processed is NULL")));

	*rows_processed = 0;

	/* Build query with LIMIT and OFFSET for chunking */
	/* Note: This function must be called within an active SPI session */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL "
					 "LIMIT %d OFFSET %d",
					 quoted_feat,
					 quoted_label,
					 quoted_tbl,
					 quoted_feat,
					 quoted_label,
					 chunk_size,
					 offset);

	ret = SPI_execute(query.data, true, 0);
	pfree(query.data);
	query.data = NULL;
	if (ret != SPI_OK_SELECT)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_stream_process_chunk: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table and columns exist and are accessible.")));
	}

	n_rows = SPI_processed;
	if (n_rows == 0)
	{
		*rows_processed = 0;
		return;
	}

	/* Determine feature type from first row */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
	{
		TupleDesc	chunk_tupdesc = SPI_tuptable->tupdesc;

		feat_type_oid = SPI_gettypeid(chunk_tupdesc, 1);
		if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
			feat_is_array = true;
	}

	/* Allocate temporary buffer for one row */
	NDB_ALLOC(row_buffer, float, accum->feature_dim);

	/* Process each row in chunk */
	for (i = 0; i < n_rows; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	row_tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec;
		ArrayType  *arr;
		double		y_true;
		double		z;
		double		prediction;
		double		error;

		feat_datum = SPI_getbinval(tuple, row_tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, row_tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		/* Extract feature vector */
		if (feat_is_array)
		{
			arr = DatumGetArrayTypeP(feat_datum);
			if (ARR_NDIM(arr) != 1)
				continue;
			if (ARR_DIMS(arr)[0] != accum->feature_dim)
				continue;
			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				float8	   *data = (float8 *) ARR_DATA_PTR(arr);

				for (j = 0; j < accum->feature_dim; j++)
					row_buffer[j] = (float) data[j];
			}
			else
			{
				float4	   *data = (float4 *) ARR_DATA_PTR(arr);

				memcpy(row_buffer, data, sizeof(float) * accum->feature_dim);
			}
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			if (vec->dim != accum->feature_dim)
				continue;
			memcpy(row_buffer, vec->data, sizeof(float) * accum->feature_dim);
		}

		/* Extract target */
		{
			Oid			targ_type = SPI_gettypeid(row_tupdesc, 2);

			if (targ_type == INT2OID || targ_type == INT4OID
				|| targ_type == INT8OID)
				y_true = (double) DatumGetInt32(targ_datum);
			else
				y_true = DatumGetFloat8(targ_datum);
		}

		if (y_true != 0.0 && y_true != 1.0)
			continue;

		/* Compute prediction: z = bias + weights * features */
		z = bias;
		for (j = 0; j < accum->feature_dim; j++)
			z += weights[j] * row_buffer[j];
		prediction = sigmoid(z);

		/* Compute error and accumulate gradients */
		error = prediction - y_true;
		accum->grad_b += error;
		for (j = 0; j < accum->feature_dim; j++)
			accum->grad_w[j] += error * row_buffer[j];

		(*rows_processed)++;
		accum->n_samples++;
	}

	NDB_FREE(row_buffer);
	row_buffer = NULL;
}

/* GPU model state for Logistic Regression */
typedef struct LRGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			feature_dim;
	int			n_samples;
}			LRGpuModelState;

static void
lr_gpu_release_state(LRGpuModelState * state)
{
	if (state == NULL)
		return;
	if (state->model_blob)
	{
		NDB_FREE(state->model_blob);
		state->model_blob = NULL;
	}
	if (state->metrics)
	{
		NDB_FREE(state->metrics);
		state->metrics = NULL;
	}
	NDB_FREE(state);
	state = NULL;
}

static bool
lr_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	LRGpuModelState *state;
	bytea	   *payload;
	Jsonb	   *metrics;
	int			rc;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
		return false;
	if (!neurondb_gpu_is_available())
		return false;
	if (spec->feature_matrix == NULL || spec->label_vector == NULL)
		return false;
	if (spec->sample_count <= 0 || spec->feature_dim <= 0)
		return false;

	payload = NULL;
	metrics = NULL;

	rc = ndb_gpu_lr_train(spec->feature_matrix,
						  spec->label_vector,
						  spec->sample_count,
						  spec->feature_dim,
						  spec->hyperparameters,
						  &payload,
						  &metrics,
						  errstr);
	if (rc != 0 || payload == NULL)
	{
		if (payload != NULL)
		{
			NDB_FREE(payload);
			payload = NULL;
		}
		if (metrics != NULL)
		{
			NDB_FREE(metrics);
			metrics = NULL;
		}
		return false;
	}

	if (model->backend_state != NULL)
	{
		lr_gpu_release_state((LRGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

	state = (LRGpuModelState *) palloc0(sizeof(LRGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;

	/* Store metrics in model state for later retrieval */
	if (metrics != NULL)
	{
		/* Copy metrics to ensure they're in the correct memory context */
		state->metrics = (Jsonb *) PG_DETOAST_DATUM_COPY(
														 PointerGetDatum(metrics));
		elog(DEBUG1,
			 "lr_gpu_train: stored metrics in state: %p",
			 (void *) state->metrics);
	}
	else
	{
		state->metrics = NULL;
		elog(WARNING,
			 "lr_gpu_train: metrics is NULL, cannot store in state!");
	}

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
lr_gpu_predict(const MLGpuModel * model,
			   const float *input,
			   int input_dim,
			   float *output,
			   int output_dim,
			   char **errstr)
{
	const		LRGpuModelState *state;
	double		probability;
	int			rc;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = -1.0f;
	if (model == NULL || input == NULL || output == NULL)
		return false;
	if (output_dim <= 0)
		return false;
	if (!model->gpu_ready || model->backend_state == NULL)
		return false;

	state = (const LRGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	rc = ndb_gpu_lr_predict(state->model_blob,
							input,
							state->feature_dim > 0 ? state->feature_dim : input_dim,
							&probability,
							errstr);
	if (rc != 0)
		return false;

	output[0] = (float) probability;

	return true;
}

static bool
lr_gpu_evaluate(const MLGpuModel * model,
				const MLGpuEvalSpec * spec,
				MLGpuMetrics * out,
				char **errstr)
{
	const		LRGpuModelState *state;
	Jsonb	   *metrics_json;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const LRGpuModelState *) model->backend_state;

	/* Build metrics JSON from stored model metadata */
	{
		StringInfoData buf;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"logistic_regression\","
						 "\"storage\":\"gpu\","
						 "\"n_features\":%d,"
						 "\"n_samples\":%d}",
						 state->feature_dim > 0 ? state->feature_dim : 0,
						 state->n_samples > 0 ? state->n_samples : 0);

		metrics_json = DatumGetJsonbP(DirectFunctionCall1(
														  jsonb_in, CStringGetDatum(buf.data)));
		NDB_FREE(buf.data);
		buf.data = NULL;
	}

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
lr_gpu_serialize(const MLGpuModel * model,
				 bytea * *payload_out,
				 Jsonb * *metadata_out,
				 char **errstr)
{
	const		LRGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
		return false;

	state = (const LRGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	payload_size = VARSIZE(state->model_blob);
	NDB_ALLOC(payload_copy, bytea, payload_size);
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
	{
		NDB_FREE(payload_copy);
		payload_copy = NULL;
	}

	/* Return stored metrics */
	if (metadata_out != NULL && state->metrics != NULL)
	{
		/* Copy metrics to ensure they're in the correct memory context */
		*metadata_out = (Jsonb *) PG_DETOAST_DATUM_COPY(
														PointerGetDatum(state->metrics));
		elog(DEBUG1,
			 "lr_gpu_serialize: returning metrics: %p",
			 (void *) *metadata_out);
	}
	else if (metadata_out != NULL)
	{
		*metadata_out = NULL;
		elog(WARNING,
			 "lr_gpu_serialize: state->metrics is NULL, cannot return metrics!");
	}

	return true;
}

static bool
lr_gpu_deserialize(MLGpuModel * model,
				   const bytea * payload,
				   const Jsonb * metadata,
				   char **errstr)
{
	LRGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

	payload_size = VARSIZE(payload);
	payload_copy = (bytea *) palloc(payload_size);
	NDB_CHECK_ALLOC(payload_copy, "payload_copy");
	memcpy(payload_copy, payload, payload_size);

	state = (LRGpuModelState *) palloc0(sizeof(LRGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = payload_copy;
	state->feature_dim = -1;
	state->n_samples = -1;

	if (model->backend_state != NULL)
		lr_gpu_release_state((LRGpuModelState *) model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static void
lr_gpu_destroy(MLGpuModel * model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		lr_gpu_release_state((LRGpuModelState *) model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps lr_gpu_model_ops = {
	.algorithm = "logistic_regression",
	.train = lr_gpu_train,
	.predict = lr_gpu_predict,
	.evaluate = lr_gpu_evaluate,
	.serialize = lr_gpu_serialize,
	.deserialize = lr_gpu_deserialize,
	.destroy = lr_gpu_destroy,
};

static bytea *
lr_model_serialize(const LRModel * model)
{
	StringInfoData buf;
	int			i;

	if (model == NULL)
		return NULL;

	/* Validate model before serialization - match linear regression */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_model_serialize: invalid n_features %d (corrupted model?)",
						model->n_features)));
	}

	pq_begintypsend(&buf);

	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendfloat8(&buf, model->bias);
	pq_sendfloat8(&buf, model->learning_rate);
	pq_sendfloat8(&buf, model->lambda);
	pq_sendint32(&buf, model->max_iters);
	pq_sendfloat8(&buf, model->final_loss);
	pq_sendfloat8(&buf, model->accuracy);

	if (model->weights != NULL && model->n_features > 0)
	{
		for (i = 0; i < model->n_features; i++)
			pq_sendfloat8(&buf, model->weights[i]);
	}

	return pq_endtypsend(&buf);
}

static LRModel *
lr_model_deserialize(const bytea * data)
{
	LRModel    *model;
	StringInfoData buf;
	int			i;

	if (data == NULL)
		return NULL;

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	model = (LRModel *) palloc0(sizeof(LRModel));
	NDB_CHECK_ALLOC(model, "model");

	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->bias = pq_getmsgfloat8(&buf);
	model->learning_rate = pq_getmsgfloat8(&buf);
	model->lambda = pq_getmsgfloat8(&buf);
	model->max_iters = pq_getmsgint(&buf, 4);
	model->final_loss = pq_getmsgfloat8(&buf);
	model->accuracy = pq_getmsgfloat8(&buf);

	/* Validate deserialized values - match linear regression */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		NDB_FREE(model);
		model = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr: invalid n_features in deserialized model (corrupted data?)")));
	}
	if (model != NULL && (model->n_samples < 0 || model->n_samples > 100000000))
	{
		NDB_FREE(model);
		model = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr: invalid n_samples in deserialized model (corrupted data?)")));
	}

	if (model->n_features > 0)
	{
		model->weights =
			(double *) palloc(sizeof(double) * model->n_features);
		for (i = 0; i < model->n_features; i++)
			model->weights[i] = pq_getmsgfloat8(&buf);
	}

	return model;
}

/*
 * lr_metadata_is_gpu
 *
 * Checks if a model's metadata indicates it's a GPU-backed model.
 */
static bool
lr_metadata_is_gpu(Jsonb * metadata)
{
	char	   *meta_txt;
	bool		is_gpu = false;

	if (metadata == NULL)
		return false;

	PG_TRY();
	{
		meta_txt = DatumGetCString(DirectFunctionCall1(
													   jsonb_out, JsonbPGetDatum(metadata)));
		if (meta_txt != NULL)
		{
			if (strstr(meta_txt, "\"storage\":\"gpu\"") != NULL
				|| strstr(meta_txt, "\"storage\": \"gpu\"")
				!= NULL)
				is_gpu = true;
			NDB_FREE(meta_txt);
			meta_txt = NULL;
		}
	}
	PG_CATCH();
	{
		/* Invalid JSONB, assume CPU */
		is_gpu = false;
	}
	PG_END_TRY();

	return is_gpu;
}

/*
 * lr_try_gpu_predict_catalog
 *
 * Attempts GPU prediction for a model loaded from the catalog.
 * Returns true if GPU prediction succeeded, false otherwise.
 */
static bool
lr_try_gpu_predict_catalog(int32 model_id,
						   const Vector *feature_vec,
						   double *result_out)
{
	bytea	   *payload = NULL;
	Jsonb	   *metrics = NULL;
	char	   *gpu_err = NULL;
	double		probability = 0.0;
	bool		success = false;

	elog(DEBUG1,
		 "lr_try_gpu_predict_catalog: entry, model_id=%d, feature_dim=%d",
		 model_id,
		 feature_vec ? feature_vec->dim : -1);

	if (!neurondb_gpu_is_available())
	{
		return false;
	}
	if (feature_vec == NULL)
		return false;
	if (feature_vec->dim <= 0)
		return false;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
	{
		elog(DEBUG1,
			 "lr_try_gpu_predict_catalog: failed to fetch model payload for model %d",
			 model_id);
		return false;
	}

	if (payload == NULL)
	{
		elog(DEBUG1,
			 "lr_try_gpu_predict_catalog: payload is NULL for model %d",
			 model_id);
		goto cleanup;
	}

	if (!lr_metadata_is_gpu(metrics))
	{
		elog(DEBUG1,
			 "lr_try_gpu_predict_catalog: model %d is not a GPU model",
			 model_id);
		goto cleanup;
	}

	elog(DEBUG1,
		 "lr_try_gpu_predict_catalog: model %d is GPU model, payload size=%d",
		 model_id,
		 VARSIZE(payload) - VARHDRSZ);

	if (ndb_gpu_lr_predict(payload,
						   feature_vec->data,
						   feature_vec->dim,
						   &probability,
						   &gpu_err)
		== 0)
	{
		if (result_out != NULL)
			*result_out = probability;
		elog(DEBUG1,
			 "logistic_regression: GPU prediction used for model %d probability=%.6f",
			 model_id,
			 probability);
		success = true;
	}
	else if (gpu_err != NULL)
	{
		elog(WARNING,
			 "logistic_regression: GPU prediction failed for model %d (%s)",
			 model_id,
			 gpu_err);
	}

cleanup:
	if (payload != NULL)
	{
		NDB_FREE(payload);
		payload = NULL;
	}
	if (metrics != NULL)
	{
		NDB_FREE(metrics);
		metrics = NULL;
	}
	if (gpu_err != NULL)
	{
		NDB_FREE(gpu_err);
		gpu_err = NULL;
	}

	return success;
}

/*
 * lr_load_model_from_catalog
 *
 * Loads a Logistic Regression model from the catalog by model_id.
 * Returns true on success, false on failure.
 */
static bool
lr_load_model_from_catalog(int32 model_id, LRModel * *out)
{
	bytea	   *payload = NULL;
	Jsonb	   *metrics = NULL;
	LRModel    *decoded;

	if (out == NULL)
		return false;

	*out = NULL;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
	{
		if (metrics != NULL)
		{
			NDB_FREE(metrics);
			metrics = NULL;
		}
		return false;
	}

	/*
	 * For CPU evaluation, we can try to load GPU models too if GPU evaluation
	 * failed
	 */

	decoded = lr_model_deserialize(payload);

	NDB_FREE(payload);
	payload = NULL;
	if (metrics != NULL)
	{
		NDB_FREE(metrics);
		metrics = NULL;
	}

	if (decoded == NULL)
		return false;

	*out = decoded;
	return true;
}

void
neurondb_gpu_register_lr_model(void)
{
	static bool registered = false;

	if (registered)
		return;

	ndb_gpu_register_model_ops(&lr_gpu_model_ops);
	registered = true;
}
