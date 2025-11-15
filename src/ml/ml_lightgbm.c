/*-------------------------------------------------------------------------
 *
 * ml_lightgbm.c
 *    LightGBM Machine Learning Integration for NeuronDB
 *
 * IDENTIFICATION
 *    src/ml/ml_lightgbm.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/array.h"
#include "access/htup_details.h"
#include "utils/memutils.h"

PG_FUNCTION_INFO_V1(train_lightgbm_classifier);
PG_FUNCTION_INFO_V1(train_lightgbm_regressor);
PG_FUNCTION_INFO_V1(predict_lightgbm);

Datum
train_lightgbm_classifier(PG_FUNCTION_ARGS)
{
	text *table_name = PG_GETARG_TEXT_PP(0);
	text *feature_col = PG_GETARG_TEXT_PP(1);
	text *label_col = PG_GETARG_TEXT_PP(2);
	int32 n_estimators = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);
	int32 num_leaves = PG_ARGISNULL(4) ? 31 : PG_GETARG_INT32(4);
	float8 learning_rate = PG_ARGISNULL(5) ? 0.1 : PG_GETARG_FLOAT8(5);

	CHECK_NARGS_RANGE(3, 6);

	/* Defensive: Check NULL inputs */
	if (table_name == NULL || feature_col == NULL || label_col == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				errmsg("train_lightgbm_classifier: table_name, feature_col, and label_col cannot be NULL")));

	/* Defensive: Validate parameters */
	if (n_estimators < 1 || n_estimators > 100000)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("n_estimators must be between 1 and 100000, got %d", n_estimators)));

	if (num_leaves < 1 || num_leaves > 32768)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("num_leaves must be between 1 and 32768, got %d", num_leaves)));

	if (isnan(learning_rate) || isinf(learning_rate) || learning_rate <= 0.0 || learning_rate > 1.0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("learning_rate must be between 0.0 and 1.0, got %f", learning_rate)));

	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("LightGBM library not available"),
			errhint("Rebuild NeuronDB with LightGBM support.")));
	PG_RETURN_INT32(0);
}

Datum
train_lightgbm_regressor(PG_FUNCTION_ARGS)
{
	text *table_name = PG_GETARG_TEXT_PP(0);
	text *feature_col = PG_GETARG_TEXT_PP(1);
	text *target_col = PG_GETARG_TEXT_PP(2);
	int32 n_estimators = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);

	(void)table_name;
	(void)feature_col;
	(void)target_col;
	(void)n_estimators;

	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("LightGBM library not available"),
			errhint("Rebuild NeuronDB with LightGBM support.")));
	PG_RETURN_INT32(0);
}

Datum
predict_lightgbm(PG_FUNCTION_ARGS)
{
	int32 model_id = PG_GETARG_INT32(0);
	ArrayType *features = PG_GETARG_ARRAYTYPE_P(1);

	(void)model_id;
	(void)features;

	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("LightGBM library not available"),
			errhint("Rebuild NeuronDB with LightGBM support.")));
	PG_RETURN_FLOAT8(0.0);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for Lightgbm
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"

void
neurondb_gpu_register_lightgbm_model(void)
{
	elog(DEBUG1, "Lightgbm GPU Model Ops registration skipped - not yet implemented");
}
