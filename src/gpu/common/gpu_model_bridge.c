/*-------------------------------------------------------------------------
 *
 * gpu_model_bridge.c
 *    Connects SQL ML entry points with GPU model registry.
 *
 * Provides helper routines that attempt to run training fully on the GPU
 * when a registered model implementation exists. Falls back silently when
 * GPU support is unavailable.
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "ml_catalog.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"
#include "ml_gpu_random_forest.h"
#include "ml_gpu_logistic_regression.h"
#include "neurondb_gpu_bridge.h"

#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include <string.h>

static void
ndb_gpu_init_train_result(MLGpuTrainResult *result)
{
	if (result == NULL)
		return;
	memset(result, 0, sizeof(MLGpuTrainResult));
}

static char *
ndb_gpu_strdup_or_null(const char *src)
{
	if (src == NULL)
		return NULL;
	return pstrdup(src);
}

bool
ndb_gpu_try_train_model(const char *algorithm,
	const char *project_name,
	const char *model_name,
	const char *training_table,
	const char *training_column,
	const char *const *feature_columns,
	int feature_count,
	Jsonb *hyperparameters,
	const float *feature_matrix,
	const double *label_vector,
	int sample_count,
	int feature_dim,
	int class_count,
	MLGpuTrainResult *result,
	char **errstr)
{
	const MLGpuModelOps *ops;
	MLGpuModel model;
	MLGpuTrainSpec spec;
	MLGpuContext ctx;
	bytea *payload = NULL;
	Jsonb *metadata = NULL;
	bool trained = false;

	if (errstr)
		*errstr = NULL;
	if (result)
		ndb_gpu_init_train_result(result);
	if (!neurondb_gpu_is_available())
		return false;

	ops = ndb_gpu_lookup_model_ops(algorithm);
	if (ops == NULL || ops->train == NULL || ops->serialize == NULL)
		return false;

	memset(&ctx, 0, sizeof(MLGpuContext));
	ctx.backend = ndb_gpu_get_active_backend();
	ctx.backend_name = (ctx.backend && ctx.backend->name)
		? ctx.backend->name
		: "unknown";
	ctx.device_id = neurondb_gpu_device;
	ctx.stream_handle = NULL;
	ctx.scratch_pool = NULL;
	ctx.memory_ctx = CurrentMemoryContext;

	memset(&model, 0, sizeof(MLGpuModel));
	model.ops = ops;
	model.backend_state = NULL;
	model.catalog_id = InvalidOid;
	model.model_name = pstrdup(model_name ? model_name : algorithm);
	model.is_gpu_resident = true;
	model.gpu_ready = false;

	memset(&spec, 0, sizeof(MLGpuTrainSpec));
	spec.algorithm = algorithm;
	spec.project_name = project_name;
	spec.model_name = model_name;
	spec.training_table = training_table;
	spec.training_column = training_column;
	spec.feature_columns = feature_columns;
	spec.feature_count = feature_count;
	spec.hyperparameters = hyperparameters;
	spec.context = &ctx;
	spec.expected_features = -1;
	spec.expected_classes = -1;
	spec.feature_matrix = feature_matrix;
	spec.label_vector = label_vector;
	spec.sample_count = sample_count;
	spec.feature_dim = feature_dim;
	spec.class_count = class_count;

	if (ops != NULL && ops->train != NULL && ops->serialize != NULL)
	{
		TimestampTz train_start = GetCurrentTimestamp();
		TimestampTz train_end;
		bool ops_trained = false;
		bool ops_serialized = false;
		long secs = 0;
		int usecs = 0;
		double elapsed_ms;

		if (ops->train(&model, &spec, errstr))
		{
			model.gpu_ready = true;
			ops_trained = true;
			if (ops->serialize(&model, &payload, &metadata, errstr))
			{
				ops_serialized = true;
				trained = true;
			}
		}

		train_end = GetCurrentTimestamp();
		TimestampDifference(train_start, train_end, &secs, &usecs);
		elapsed_ms = ((double)secs * 1000.0) + ((double)usecs / 1000.0);

		if (trained)
		{
			ndb_gpu_stats_record(true, elapsed_ms, 0.0, false);
			ereport(NOTICE,
				(errmsg("%s: GPU training succeeded",
					algorithm ? algorithm : "unknown"),
				 errdetail("Elapsed %.3f ms on backend %s",
					 elapsed_ms,
					 ctx.backend_name ? ctx.backend_name : "unknown"),
				 errhidestmt(true)));
		}
		else if (ops_trained || ops_serialized)
		{
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
			ereport(DEBUG1,
				(errmsg("%s: GPU training fell back to CPU",
					algorithm ? algorithm : "unknown"),
				 errdetail("GPU stage elapsed %.3f ms before fallback",
					 elapsed_ms),
				 errhidestmt(true)));
		}
	}

	if (!trained && algorithm != NULL
		&& strcmp(algorithm, "random_forest") == 0
		&& feature_matrix != NULL && label_vector != NULL
		&& sample_count > 0 && feature_dim > 0)
	{
		TimestampTz train_start = GetCurrentTimestamp();
		TimestampTz train_end;
		int gpu_rc = ndb_gpu_rf_train(feature_matrix,
			label_vector,
			sample_count,
			feature_dim,
			class_count,
			hyperparameters,
			&payload,
			&metadata,
			errstr);
		long secs = 0;
		int usecs = 0;
		double elapsed_ms;

		train_end = GetCurrentTimestamp();
		TimestampDifference(train_start, train_end, &secs, &usecs);
		elapsed_ms = ((double) secs * 1000.0) + ((double) usecs / 1000.0);

		if (gpu_rc == 0)
		{
			trained = true;
			ndb_gpu_stats_record(true, elapsed_ms, 0.0, false);
			ereport(NOTICE,
				(errmsg("random_forest: GPU training succeeded (direct)"),
				 errdetail("Elapsed %.3f ms on backend %s",
					 elapsed_ms,
					 ctx.backend_name ? ctx.backend_name : "unknown"),
				 errhidestmt(true)));
		} else
		{
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
			ereport(NOTICE,
				(errmsg("random_forest: GPU training unavailable, using CPU"),
				 errdetail("GPU attempt elapsed %.3f ms (%s)",
					 elapsed_ms,
					 (errstr && *errstr) ? *errstr : "no error"),
				 errhidestmt(true)));
		}
	}

	if (!trained && algorithm != NULL
		&& strcmp(algorithm, "logistic_regression") == 0
		&& feature_matrix != NULL && label_vector != NULL
		&& sample_count > 0 && feature_dim > 0)
	{
		TimestampTz train_start = GetCurrentTimestamp();
		TimestampTz train_end;
		const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
		int gpu_rc;
		long secs = 0;
		int usecs = 0;
		double elapsed_ms;

		elog(DEBUG1,
			"logistic_regression: attempting direct GPU training: "
			"backend=%s, lr_train=%p, samples=%d, dim=%d",
			backend ? (backend->name ? backend->name : "unknown") : "NULL",
			backend ? (void *)backend->lr_train : NULL,
			sample_count,
			feature_dim);

		if (backend == NULL || backend->lr_train == NULL)
		{
			elog(DEBUG1,
				"logistic_regression: backend or lr_train is NULL, skipping GPU");
			goto lr_fallback;
		}

		gpu_rc = ndb_gpu_lr_train(feature_matrix,
			label_vector,
			sample_count,
			feature_dim,
			hyperparameters,
			&payload,
			&metadata,
			errstr);
		train_end = GetCurrentTimestamp();
		TimestampDifference(train_start, train_end, &secs, &usecs);
		elapsed_ms = ((double) secs * 1000.0) + ((double) usecs / 1000.0);

		if (gpu_rc == 0)
		{
			trained = true;
			ndb_gpu_stats_record(true, elapsed_ms, 0.0, false);
			if (metadata != NULL)
			{
				char *meta_txt = DatumGetCString(
					DirectFunctionCall1(jsonb_out, JsonbPGetDatum(metadata)));
				elog(DEBUG1, "gpu_model_bridge: LR direct path metadata: %s", meta_txt);
				pfree(meta_txt);
			}
			else
			{
				elog(WARNING, "gpu_model_bridge: LR direct path metadata is NULL!");
			}
			ereport(NOTICE,
				(errmsg("logistic_regression: GPU training succeeded (direct)"),
				 errdetail("Elapsed %.3f ms on backend %s",
					 elapsed_ms,
					 ctx.backend_name ? ctx.backend_name : "unknown"),
				 errhidestmt(true)));
		} else
		{
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
			ereport(NOTICE,
				(errmsg("logistic_regression: GPU training unavailable, using CPU"),
				 errdetail("GPU attempt elapsed %.3f ms (%s)",
					 elapsed_ms,
					 (errstr && *errstr) ? *errstr : "no error"),
				 errhidestmt(true)));
		}
lr_fallback:
		;
	}

	if (trained && result != NULL)
	{
		result->spec.algorithm = ndb_gpu_strdup_or_null(algorithm);
		result->spec.model_type = NULL;
		result->spec.training_table =
			ndb_gpu_strdup_or_null(training_table);
		result->spec.training_column =
			ndb_gpu_strdup_or_null(training_column);
		result->spec.project_name =
			ndb_gpu_strdup_or_null(project_name);
		result->spec.model_name = ndb_gpu_strdup_or_null(model_name);
		result->spec.parameters = hyperparameters;
		
		/* Ensure metrics Jsonb is in the correct memory context */
		if (metadata != NULL)
		{
			Jsonb *metrics_copy = (Jsonb *)PG_DETOAST_DATUM_COPY(PointerGetDatum(metadata));
			result->spec.metrics = metrics_copy;
			result->metadata = metrics_copy;
			result->metrics = metrics_copy;
			elog(DEBUG1, "gpu_model_bridge: copied metrics to result->spec.metrics: %p", 
				(void *)result->spec.metrics);
		}
		else
		{
			result->spec.metrics = NULL;
			result->metadata = NULL;
			result->metrics = NULL;
			elog(WARNING, "gpu_model_bridge: metadata is NULL, cannot set result->spec.metrics!");
		}
		
		result->spec.model_data = payload;
		result->payload = payload;
	}

	if (ops != NULL && ops->destroy != NULL)
		ops->destroy(&model);

	if (!trained)
	{
		if (payload)
			pfree(payload);
		if (metadata)
			pfree(metadata);
		if (result != NULL)
			ndb_gpu_free_train_result(result);
	}

	return trained;
}

void
ndb_gpu_free_train_result(MLGpuTrainResult *result)
{
	if (result == NULL)
		return;

	if (result->spec.algorithm)
		pfree((void *)result->spec.algorithm);
	if (result->spec.training_table)
		pfree((void *)result->spec.training_table);
	if (result->spec.training_column)
		pfree((void *)result->spec.training_column);
	if (result->spec.project_name)
		pfree((void *)result->spec.project_name);
	if (result->spec.model_name)
		pfree((void *)result->spec.model_name);
	if (result->spec.model_data)
		pfree(result->spec.model_data);
	if (result->spec.metrics)
		pfree(result->spec.metrics);

	memset(result, 0, sizeof(MLGpuTrainResult));
}
