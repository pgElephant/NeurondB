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

	(void)table_name;
	(void)feature_col;
	(void)label_col;
	(void)n_estimators;
	(void)num_leaves;
	(void)learning_rate;

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
