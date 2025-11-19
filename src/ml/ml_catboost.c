/*-------------------------------------------------------------------------
 *
 * ml_catboost.c
 *    CatBoost Integration for NeuronDB
 *
 *    Provides full integration of the Yandex CatBoost gradient boosting library.
 *    Supports categorical features and can use GPU acceleration if enabled.
 *    Requires CatBoost C library (libcatboostmodel.so) and headers.
 *
 *    Copyright (c) 2024-2025, pgElephant, Inc.
 *
 *    IDENTIFICATION
 *        src/ml/ml_catboost.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "storage/fd.h"
#include "miscadmin.h"
#include "utils/memutils.h"

#ifdef __has_include
#if __has_include(<catboost/c_api.h>)
#include <catboost/c_api.h>
#define CATBOOST_AVAILABLE
#endif
#endif

#include <stdio.h>

/*
 * PG_MODULE_MAGIC is in neurondb.c only.
 */

PG_FUNCTION_INFO_V1(train_catboost_classifier);
PG_FUNCTION_INFO_V1(train_catboost_regressor);
PG_FUNCTION_INFO_V1(predict_catboost);
PG_FUNCTION_INFO_V1(evaluate_catboost_by_model_id);

/*
 * get_column_index
 *      Helper function to get column index in a query result by name.
 */
__attribute__((unused)) static int
get_column_index(SPITupleTable *tuptable,
                 TupleDesc tupdesc,
                 const char *colname)
{
    int i;

    for (i = 0; i < tupdesc->natts; i++)
    {
        if (strcmp(NameStr(TupleDescAttr(tupdesc, i)->attname), colname) == 0)
            return i;
    }
    ereport(ERROR,
		(errcode(ERRCODE_UNDEFINED_COLUMN),
			errmsg("neurondb: column \"%s\" does not exist in tuple", colname)));

    /* not reached */
    return -1;
}

/*
 * write_csv_from_spi
 *      Helper to write training data to CSV for CatBoost.
 */
__attribute__((unused)) static void
write_csv_from_spi(char *csv_path,
                   SPITupleTable *tuptable,
                   TupleDesc tupdesc,
                   int *feature_idxs,
                   int n_features,
                   int label_idx)
{
    FILE   *fp;
    uint64  row;
    int     i;

    fp = AllocateFile(csv_path, "w");
    if (fp == NULL)
        ereport(ERROR,
                (errcode_for_file_access(),
                 errmsg("could not open file \"%s\" for writing: %m", csv_path)));

    /* Write header */
    for (i = 0; i < n_features; i++)
        fprintf(fp, "f%d,", i);
    fprintf(fp, "label\n");

    for (row = 0; row < tuptable->numvals; row++)
    {
        HeapTuple   tuple = tuptable->vals[row];

        for (i = 0; i < n_features; i++)
        {
            Datum   val;
            bool    isnull;
            Oid     typid;
            Oid     typoutput;
            bool    typisvarlena;
            char   *valstr;

            val = heap_getattr(tuple, feature_idxs[i] + 1, tupdesc, &isnull);

            if (isnull)
                fprintf(fp, ",");
            else
            {
                typid = TupleDescAttr(tupdesc, feature_idxs[i])->atttypid;
                getTypeOutputInfo(typid, &typoutput, &typisvarlena);
                valstr = OidOutputFunctionCall(typoutput, val);
                fprintf(fp, "%s,", valstr);
            }
        }

        /* Output label column */
        {
            Datum   label_val;
            bool    label_isnull;
            Oid     typid;
            Oid     typoutput;
            bool    typisvarlena;
            char   *labelstr;

            label_val = heap_getattr(tuple, label_idx + 1, tupdesc, &label_isnull);
            if (label_isnull)
                fprintf(fp, "\n");
            else
            {
                typid = TupleDescAttr(tupdesc, label_idx)->atttypid;
                getTypeOutputInfo(typid, &typoutput, &typisvarlena);
                labelstr = OidOutputFunctionCall(typoutput, label_val);
                fprintf(fp, "%s\n", labelstr);
            }
        }
    }
    FreeFile(fp);
}

/*
 * check_catboost_error
 *      Helper function for error translation from CatBoost return codes.
 */
__attribute__((unused)) static void
check_catboost_error(int err_code)
{
#ifdef CATBOOST_AVAILABLE
    if (err_code != 0)
    {
        const char *err_msg = CatBoostGetErrorString();

        if (err_msg == NULL)
            err_msg = "Unknown error from CatBoost";
        ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: CatBoost error: %s (code=%d)", err_msg, err_code)));
    }
#else
    if (err_code != 0)
        ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				errmsg("neurondb: CatBoost error: code=%d (library not available)", err_code)));
#endif
}

/*
 * train_catboost_classifier
 *      Trains a CatBoost classifier model using the provided table, feature columns, and label column.
 *      Returns integer model_id on successful training and storage.
 */
Datum
train_catboost_classifier(PG_FUNCTION_ARGS)
{
#ifdef CATBOOST_AVAILABLE
    text       *table_name_text = PG_GETARG_TEXT_PP(0);
    text       *feature_col_text = PG_GETARG_TEXT_PP(1);
    text       *label_col_text = PG_GETARG_TEXT_PP(2);
    int32       iterations;
    float8      learning_rate;
    int32       depth;

    char       *table_name;
    char       *feature_col_list;
    char       *label_col;

    StringInfoData sql;
    int         ret;
    int         n_features;
    int         i;
    int         model_id;

    char      **features = NULL;
    char       *token;
    MemoryContext oldctx;
    MemoryContext per_query_ctx;

    /* Assign parameters with defaults if NULL */
    iterations = PG_ARGISNULL(3) ? 1000 : PG_GETARG_INT32(3);
    learning_rate = PG_ARGISNULL(4) ? 0.03 : PG_GETARG_FLOAT8(4);
    depth = PG_ARGISNULL(5) ? 6 : PG_GETARG_INT32(5);

    table_name = text_to_cstring(table_name_text);
    feature_col_list = text_to_cstring(feature_col_text);
    label_col = text_to_cstring(label_col_text);

    /* Parse feature_col_list into features[] */
    n_features = 1;
    for (i = 0; feature_col_list[i]; i++)
    {
        if (feature_col_list[i] == ',')
            n_features++;
    }
    features = (char **) palloc0(sizeof(char *) * n_features);

    i = 0;
    token = strtok(feature_col_list, ",");
    while (token != NULL)
    {
        while (*token == ' ' || *token == '\t')
            token++;
        features[i++] = pstrdup(token);
        token = strtok(NULL, ",");
    }

    initStringInfo(&sql);
    appendStringInfo(&sql, "SELECT ");
    for (i = 0; i < n_features; i++)
    {
        if (i > 0)
            appendStringInfoChar(&sql, ',');
        appendStringInfoString(&sql, features[i]);
    }
    appendStringInfo(&sql, ",%s FROM %s", label_col, table_name);


    per_query_ctx = AllocSetContextCreate(CurrentMemoryContext,
                                          elog(DEBUG1,
                                          	"catboost_spi_ctx",
                                          ALLOCSET_DEFAULT_SIZES);
    oldctx = MemoryContextSwitchTo(per_query_ctx);

    if ((ret = SPI_connect()) != SPI_OK_CONNECT)
        ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: could not connect to SPI")));

    ret = SPI_execute(sql.data, true, 0);
    if (ret != SPI_OK_SELECT)
    {
        SPI_finish();
        ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: could not execute SQL for training set")));
    }

    /* Get feature and label indexes */
    {
        TupleDesc   tupdesc = SPI_tuptable->tupdesc;
        int        *feature_idxs = (int *) palloc0(sizeof(int) * n_features);
        int         label_idx;

        for (i = 0; i < n_features; i++)
            feature_idxs[i] = get_column_index(SPI_tuptable, tupdesc, features[i]);
        label_idx = get_column_index(SPI_tuptable, tupdesc, label_col);

        /* Write training data to CSV */
        {
            char    tmp_csv_path[MAXPGPATH];
            char    tmp_model_path[MAXPGPATH];

            snprintf(tmp_csv_path, sizeof(tmp_csv_path),
                     "%s/catboost_train_%d.csv", PG_TEMP_FILES_DIR, MyProcPid);
            snprintf(tmp_model_path, sizeof(tmp_model_path),
                     "%s/catboost_model_%d.cbm", PG_TEMP_FILES_DIR, MyProcPid);

            write_csv_from_spi(tmp_csv_path,
                               SPI_tuptable,
                               tupdesc,
                               feature_idxs,
                               n_features,
                               label_idx);

            /* Setup CatBoost C API options and train */
            {
                ModelCalcerHandle           *model_handle = NULL;
                CatBoostModelTrainingOptions *opts;

                opts = CatBoostCreateModelTrainingOptions();

                check_catboost_error(CatBoostSetTrainingOptionInt(opts, "iterations", iterations));
                check_catboost_error(CatBoostSetTrainingOptionDouble(opts, "learning_rate", learning_rate));
                check_catboost_error(CatBoostSetTrainingOptionInt(opts, "depth", depth));

                check_catboost_error(CatBoostTrainModelFromFile(opts,
                                                                tmp_csv_path,
                                                                n_features,
                                                                label_idx,
                                                                true,   /* has header */
                                                                &model_handle));

                check_catboost_error(CatBoostSaveModelToFile(model_handle, tmp_model_path));

                model_id = MyProcPid;

                CatBoostDestroyModelTrainingOptions(opts);
                CatBoostFreeModelCalcer(model_handle);
            }
        }
    }

    SPI_finish();
    MemoryContextSwitchTo(oldctx);
    MemoryContextDelete(per_query_ctx);

    PG_RETURN_INT32(model_id);
#else
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("CatBoost library not available. Please install CatBoost to use this function.")));
    PG_RETURN_NULL();
#endif
}

/*
 * train_catboost_regressor
 *      Trains a CatBoost regressor model. Arguments similar to classifier.
 *      Returns integer model_id on success.
 */
Datum
train_catboost_regressor(PG_FUNCTION_ARGS)
{
#ifdef CATBOOST_AVAILABLE
    text       *table_name_text = PG_GETARG_TEXT_PP(0);
    text       *feature_col_text = PG_GETARG_TEXT_PP(1);
    text       *target_col_text = PG_GETARG_TEXT_PP(2);
    int32       iterations;
    char       *table_name;
    char       *feature_col_list;
    char       *target_col;

    StringInfoData sql;
    int         ret;
    int         n_features;
    int         i;
    int         model_id;

    char      **features = NULL;
    char       *token;
    MemoryContext oldctx;
    MemoryContext per_query_ctx;

    iterations = PG_ARGISNULL(3) ? 1000 : PG_GETARG_INT32(3);

    table_name = text_to_cstring(table_name_text);
    feature_col_list = text_to_cstring(feature_col_text);
    target_col = text_to_cstring(target_col_text);

    /* Parse feature list */
    n_features = 1;
    for (i = 0; feature_col_list[i]; i++)
    {
        if (feature_col_list[i] == ',')
            n_features++;
    }
    features = (char **) palloc0(sizeof(char *) * n_features);

    i = 0;
    token = strtok(feature_col_list, ",");
    while (token != NULL)
    {
        while (*token == ' ' || *token == '\t')
            token++;
        features[i++] = pstrdup(token);
        token = strtok(NULL, ",");
    }

    initStringInfo(&sql);
    appendStringInfo(&sql, "SELECT ");
    for (i = 0; i < n_features; i++)
    {
        if (i > 0)
            appendStringInfoChar(&sql, ',');
        appendStringInfoString(&sql, features[i]);
    }
    appendStringInfo(&sql, ",%s FROM %s", target_col, table_name);

    per_query_ctx = AllocSetContextCreate(CurrentMemoryContext,
                                          elog(DEBUG1,
                                          	"catboost_spi_ctx",
                                          ALLOCSET_DEFAULT_SIZES);
    oldctx = MemoryContextSwitchTo(per_query_ctx);

    if ((ret = SPI_connect()) != SPI_OK_CONNECT)
        ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: could not connect to SPI")));

    ret = SPI_execute(sql.data, true, 0);
    if (ret != SPI_OK_SELECT)
    {
        SPI_finish();
        ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: could not execute SQL for training set")));
    }

    {
        TupleDesc   tupdesc = SPI_tuptable->tupdesc;
        int        *feature_idxs = (int *) palloc0(sizeof(int) * n_features);
        int         target_idx;

        for (i = 0; i < n_features; i++)
            feature_idxs[i] = get_column_index(SPI_tuptable, tupdesc, features[i]);
        target_idx = get_column_index(SPI_tuptable, tupdesc, target_col);

        {
            char    tmp_csv_path[MAXPGPATH];
            char    tmp_model_path[MAXPGPATH];

            snprintf(tmp_csv_path, sizeof(tmp_csv_path),
                     "%s/catboost_train_%d.csv", PG_TEMP_FILES_DIR, MyProcPid);
            snprintf(tmp_model_path, sizeof(tmp_model_path),
                     "%s/catboost_model_%d.cbm", PG_TEMP_FILES_DIR, MyProcPid);

            write_csv_from_spi(tmp_csv_path,
                               SPI_tuptable,
                               tupdesc,
                               feature_idxs,
                               n_features,
                               target_idx);

            {
                ModelCalcerHandle           *model_handle = NULL;
                CatBoostModelTrainingOptions *opts;

                opts = CatBoostCreateModelTrainingOptions();
                check_catboost_error(CatBoostSetTrainingOptionInt(opts, "iterations", iterations));
                check_catboost_error(CatBoostSetTrainingOptionInt(opts, "task_type", 0)); /* 0 for regression */

                check_catboost_error(CatBoostTrainModelFromFile(opts,
                                                                tmp_csv_path,
                                                                n_features,
                                                                target_idx,
                                                                true,
                                                                &model_handle));

                check_catboost_error(CatBoostSaveModelToFile(model_handle, tmp_model_path));

                model_id = MyProcPid;

                CatBoostDestroyModelTrainingOptions(opts);
                CatBoostFreeModelCalcer(model_handle);
            }
        }
    }

    SPI_finish();
    MemoryContextSwitchTo(oldctx);
    MemoryContextDelete(per_query_ctx);

    PG_RETURN_INT32(model_id);
#else
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("CatBoost library not available. Please install CatBoost to use this function.")));
    PG_RETURN_NULL();
#endif
}

/*
 * predict_catboost
 *      Predicts using a persisted CatBoost model and a feature array.
 *      Arguments: int4 model_id, float8[] features (or text[] for categorical support).
 *      Returns float8 predicted value.
 */
Datum
predict_catboost(PG_FUNCTION_ARGS)
{
#ifdef CATBOOST_AVAILABLE
    int32       model_id;
    ArrayType  *features_array;
    Oid         elmtype;
    int16       typlen;
    bool        typbyval;
    char        typalign;
    int         n_features;
    int         i;
    int         ndim;
    Datum      *elems;
    bool       *nulls;
    float64     result = 0.0;

    ModelCalcerHandle *model_handle = NULL;
    char        model_path[MAXPGPATH];

    model_id = PG_GETARG_INT32(0);
    features_array = PG_GETARG_ARRAYTYPE_P(1);

    ndim = ARR_NDIM(features_array);
    if (ndim != 1)
        ereport(ERROR,
                (errmsg("features array must be one-dimensional")));

    n_features = ArrayGetNItems(ARR_NDIM(features_array), ARR_DIMS(features_array));
    elmtype = ARR_ELEMTYPE(features_array);
    get_typlenbyvalalign(elmtype, &typlen, &typbyval, &typalign);

    deconstruct_array(features_array, elmtype, typlen, typbyval, typalign,
                      &elems, &nulls, &n_features);

    snprintf(model_path, sizeof(model_path),
             elog(DEBUG1,
             	"%s/catboost_model_%d.cbm",
             PG_TEMP_FILES_DIR, model_id);

    check_catboost_error(CatBoostLoadModelFromFile(model_path, &model_handle));

    if (elmtype == TEXTOID)
    {
        /* categorical/text features */
        char    **input_features;

        input_features = (char **) palloc(sizeof(char *) * n_features);

        for (i = 0; i < n_features; i++)
        {
            if (nulls[i])
                ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: catboost feature %d is NULL", i)));

            input_features[i] = text_to_cstring(DatumGetTextPP(elems[i]));
        }

        check_catboost_error(CatBoostModelCalcerPredictText(model_handle,
                                                           (const char * const *) input_features,
                                                           n_features,
                                                           &result,
                                                           1));
        pfree(input_features);
    }
    else
    {
        /* numeric features */
        double  *features;

        features = (double *) palloc(sizeof(double) * n_features);
        for (i = 0; i < n_features; i++)
        {
            if (nulls[i])
                ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: catboost feature %d is NULL", i)));
            if (elmtype == FLOAT8OID)
                features[i] = DatumGetFloat8(elems[i]);
            else if (elmtype == FLOAT4OID)
                features[i] = (double) DatumGetFloat4(elems[i]);
            else if (elmtype == INT4OID)
                features[i] = (double) DatumGetInt32(elems[i]);
            else if (elmtype == INT8OID)
                features[i] = (double) DatumGetInt64(elems[i]);
            else if (elmtype == INT2OID)
                features[i] = (double) DatumGetInt16(elems[i]);
            else
                ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						errmsg("neurondb: unsupported feature element type for CatBoost: %u", elmtype)));
        }

        check_catboost_error(CatBoostModelCalcerPredict(model_handle,
                                                       features, n_features, &result, 1));
        pfree(features);
    }

    CatBoostFreeModelCalcer(model_handle);

    PG_RETURN_FLOAT8(result);
#else
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("CatBoost library not available. Please install CatBoost to use this function.")));
    PG_RETURN_NULL();
#endif
}

/*
 * evaluate_catboost_by_model_id
 *
 * Evaluates a CatBoost model on a dataset and returns performance metrics.
 * Arguments: int4 model_id, text table_name, text feature_col, text label_col
 * Returns: jsonb with metrics
 */
Datum
evaluate_catboost_by_model_id(PG_FUNCTION_ARGS)
{
#ifdef CATBOOST_AVAILABLE
    int32 model_id;
    text *table_name;
    text *feature_col;
    text *label_col;
    char *tbl_str;
    char *feat_str;
    char *targ_str;
    StringInfoData query;
    int ret;
    int nvec = 0;
    double mse = 0.0;
    double mae = 0.0;
    double ss_tot = 0.0;
    double ss_res = 0.0;
    double y_mean = 0.0;
    double r_squared;
    double rmse;
    int i;
    StringInfoData jsonbuf;
    Jsonb *result;
    MemoryContext oldcontext;

    /* Validate arguments */
    if (PG_NARGS() != 4)
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("neurondb: evaluate_catboost_by_model_id: 4 arguments are required")));

    if (PG_ARGISNULL(0))
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("neurondb: evaluate_catboost_by_model_id: model_id is required")));

    model_id = PG_GETARG_INT32(0);

    if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("neurondb: evaluate_catboost_by_model_id: table_name, feature_col, and label_col are required")));

    table_name = PG_GETARG_TEXT_PP(1);
    feature_col = PG_GETARG_TEXT_PP(2);
    label_col = PG_GETARG_TEXT_PP(3);

    tbl_str = text_to_cstring(table_name);
    feat_str = text_to_cstring(feature_col);
    targ_str = text_to_cstring(label_col);

    oldcontext = CurrentMemoryContext;

    /* Connect to SPI */
    if ((ret = SPI_connect()) != SPI_OK_CONNECT)
        ereport(ERROR,
            (errcode(ERRCODE_INTERNAL_ERROR),
                errmsg("neurondb: evaluate_catboost_by_model_id: SPI_connect failed")));

    /* Build query */
    initStringInfo(&query);
    appendStringInfo(&query,
        "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
        feat_str, targ_str, tbl_str, feat_str, targ_str);

    ret = SPI_execute(query.data, true, 0);
    if (ret != SPI_OK_SELECT)
        ereport(ERROR,
            (errcode(ERRCODE_INTERNAL_ERROR),
                errmsg("neurondb: evaluate_catboost_by_model_id: query failed")));

    nvec = SPI_processed;
    if (nvec < 2)
    {
        SPI_finish();
        pfree(tbl_str);
        pfree(feat_str);
        pfree(targ_str);
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("neurondb: evaluate_catboost_by_model_id: need at least 2 samples, got %d",
                    nvec)));
    }

    /* First pass: compute mean of y */
    for (i = 0; i < nvec; i++)
    {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        Datum targ_datum;
        bool targ_null;

        targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
        if (!targ_null)
            y_mean += DatumGetFloat8(targ_datum);
    }
    y_mean /= nvec;

    /* Determine feature type from first row */
    Oid feat_type_oid = InvalidOid;
    bool feat_is_array = false;
    if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
    {
        feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
        if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
            feat_is_array = true;
    }

    /* Second pass: compute predictions and metrics */
    for (i = 0; i < nvec; i++)
    {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        Datum feat_datum;
        Datum targ_datum;
        bool feat_null;
        bool targ_null;
        ArrayType *arr;
        Vector *vec;
        double y_true;
        double y_pred;
        double error;
        int actual_dim;
        int j;

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
                SPI_finish();
                pfree(tbl_str);
                pfree(feat_str);
                pfree(targ_str);
                ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                        errmsg("catboost: features array must be 1-D")));
            }
            actual_dim = ARR_DIMS(arr)[0];
        }
        else
        {
            vec = DatumGetVector(feat_datum);
            actual_dim = vec->dim;
        }

        /* Make prediction using CatBoost model */
        if (feat_is_array)
        {
            /* Create a temporary array for prediction */
            Datum features_datum = feat_datum;
            y_pred = DatumGetFloat8(DirectFunctionCall2(predict_catboost,
                                                       Int32GetDatum(model_id),
                                                       features_datum));
        }
        else
        {
            /* Convert vector to array for prediction */
            int ndims = 1;
            int dims[1] = {actual_dim};
            int lbs[1] = {1};
            Datum *elems = palloc(sizeof(Datum) * actual_dim);

            for (j = 0; j < actual_dim; j++)
                elems[j] = Float8GetDatum(vec->data[j]);

            ArrayType *feature_array = construct_md_array(elems, NULL, ndims, dims, lbs,
                                                        FLOAT8OID, sizeof(float8), true, 'd');
            Datum features_datum = PointerGetDatum(feature_array);

            y_pred = DatumGetFloat8(DirectFunctionCall2(predict_catboost,
                                                       Int32GetDatum(model_id),
                                                       features_datum));

            pfree(elems);
            pfree(feature_array);
        }

        /* Compute errors */
        error = y_true - y_pred;
        mse += error * error;
        mae += fabs(error);
        ss_res += error * error;
        ss_tot += (y_true - y_mean) * (y_true - y_mean);
    }

    SPI_finish();

    mse /= nvec;
    mae /= nvec;
    rmse = sqrt(mse);

    /* Handle R² calculation - if ss_tot is zero (no variance in y), R² is undefined */
    if (ss_tot == 0.0)
        r_squared = 0.0; /* Convention: set to 0 when there's no variance to explain */
    else
        r_squared = 1.0 - (ss_res / ss_tot);

    /* Build result JSON */
    MemoryContextSwitchTo(oldcontext);
    initStringInfo(&jsonbuf);
    appendStringInfo(&jsonbuf,
        "{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
        mse, mae, rmse, r_squared, nvec);

    result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(jsonbuf.data)));
    pfree(jsonbuf.data);

    /* Cleanup */
    pfree(tbl_str);
    pfree(feat_str);
    pfree(targ_str);

    PG_RETURN_JSONB_P(result);
#else
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("CatBoost library not available. Please install CatBoost to use evaluation.")));
    PG_RETURN_NULL();
#endif
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for Catboost
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"

void
neurondb_gpu_register_catboost_model(void)
{
}
