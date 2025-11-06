/*-------------------------------------------------------------------------
 *
 * ml_recommender.c
 *    Recommender Systems for NeuronDB
 *
 * Implements collaborative filtering (ALS matrix factorization), content-based
 * filtering (vector similarity), and hybrid approaches.
 *
 * IDENTIFICATION
 *    src/ml/ml_recommender.c
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include "utils/elog.h"
#include "utils/lsyscache.h"
#include "neurondb_pgcompat.h"

#include <math.h>
#include <string.h>
#include <sys/time.h>

/* Random number generation */
#define MAXRAND ((double) 0x7fffffff)

/* ALS (Alternating Least Squares) matrix factorization parameters */
#define ALS_DEFAULT_NFACTORS 20
#define ALS_MAX_NFACTORS     1000
#define ALS_MIN_NFACTORS     1
#define ALS_DEFAULT_EPOCHS   10
#define ALS_DEFAULT_LAMBDA   0.1
#define ALS_MAX_ITER         100

#define RECO_MIN_RESULT      1
#define RECO_MAX_RESULT      1000

/* Alloc and zero new matrix (float4**), nrow x ncol */
static float **
als_alloc_matrix(int nrow, int ncol)
{
    float **mat = (float **) palloc(sizeof(float *) * nrow);
    int i;
    for (i = 0; i < nrow; ++i)
    {
        mat[i] = (float *) palloc0(sizeof(float) * ncol);
    }
    return mat;
}

/* Free nrow rows from als_alloc_matrix */
static void
als_free_matrix(float **mat, int nrow)
{
    int i;
    for (i = 0; i < nrow; ++i)
        pfree(mat[i]);
    pfree(mat);
}

/* Dot product of two float vectors */
static float
dot_product(const float *v1, const float *v2, int n)
{
    float s = 0.0f;
    int i;
    for (i = 0; i < n; ++i)
        s += v1[i] * v2[i];
    return s;
}

/*
 * train_collaborative_filter
 * Trains ALS matrix factorization from a ratings table.
 * Returns a model id. The model is saved in-pg as two tables: user_factors, item_factors.
 */
PG_FUNCTION_INFO_V1(train_collaborative_filter);

Datum
train_collaborative_filter(PG_FUNCTION_ARGS)
{
    text       *table_name = PG_GETARG_TEXT_PP(0);
    text       *user_col = PG_GETARG_TEXT_PP(1);
    text       *item_col = PG_GETARG_TEXT_PP(2);
    text       *rating_col = PG_GETARG_TEXT_PP(3);
    int32       n_factors = PG_ARGISNULL(4) ? ALS_DEFAULT_NFACTORS : PG_GETARG_INT32(4);

    char       *table_name_str = text_to_cstring(table_name);
    char       *user_col_str = text_to_cstring(user_col);
    char       *item_col_str = text_to_cstring(item_col);
    char       *rating_col_str = text_to_cstring(rating_col);

    int         ret;
    MemoryContext oldcontext, model_mcxt;
    int         n_row, i;
    int         max_user_id = 0;
    int         max_item_id = 0;

    int        *user_ids = NULL;
    int        *item_ids = NULL;
    float      *ratings = NULL;
    int         n_ratings = 0;

    float     **P = NULL; /* user factors: user_id x n_factors */
    float     **Q = NULL; /* item factors: item_id x n_factors */

    StringInfoData sql;
    SPIPlanPtr    plan;

    /* Parameter checking */
    if (n_factors < ALS_MIN_NFACTORS || n_factors > ALS_MAX_NFACTORS)
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
             errmsg("n_factors must be between %d and %d", ALS_MIN_NFACTORS, ALS_MAX_NFACTORS)));

    initStringInfo(&sql);

    appendStringInfo(&sql,
        "SELECT %s, %s, %s FROM %s ORDER BY %s, %s",
        quote_identifier(user_col_str),
        quote_identifier(item_col_str),
        quote_identifier(rating_col_str),
        quote_identifier(table_name_str),
        quote_identifier(user_col_str),
        quote_identifier(item_col_str));

    if ((ret = SPI_connect()) != SPI_OK_CONNECT)
        elog(ERROR, "SPI_connect failed");

    ret = SPI_execute(sql.data, true, 0);
    if (ret != SPI_OK_SELECT)
    {
        SPI_finish();
        ereport(ERROR, (errmsg("Failed to execute SQL for ratings: %s", sql.data)));
    }

    n_row = SPI_processed;
    if (n_row <= 1)
    {
        SPI_finish();
        ereport(ERROR, (errmsg("Not enough ratings found to train model")));
    }

    user_ids = (int *) palloc(sizeof(int) * n_row);
    item_ids = (int *) palloc(sizeof(int) * n_row);
    ratings = (float *) palloc(sizeof(float) * n_row);
    n_ratings = n_row;
    max_user_id = 0;
    max_item_id = 0;

    for (i = 0; i < n_row; ++i)
    {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        bool isnull[3];
        int32 user, item;
        float r;

        user = DatumGetInt32(SPI_getbinval(tuple, tupdesc, 1, &isnull[0]));
        item = DatumGetInt32(SPI_getbinval(tuple, tupdesc, 2, &isnull[1]));

        if (isnull[0] || isnull[1])
        {
            SPI_finish();
            ereport(ERROR, (errmsg("user_col or item_col contains NULL at row %d", i+1)));
        }

        if (SPI_gettypeid(tupdesc, 3) == FLOAT8OID)
            r = (float) DatumGetFloat8(SPI_getbinval(tuple, tupdesc, 3, &isnull[2]));
        else
            r = (float) DatumGetFloat4(SPI_getbinval(tuple, tupdesc, 3, &isnull[2]));

        if (isnull[2])
        {
            SPI_finish();
            ereport(ERROR, (errmsg("rating_col contains NULL at row %d", i+1)));
        }
        user_ids[i] = user;
        item_ids[i] = item;
        ratings[i] = r;
        if (user > max_user_id)
            max_user_id = user;
        if (item > max_item_id)
            max_item_id = item;
    }

    /* Memory context for ALS model factors */
    model_mcxt = AllocSetContextCreate(CurrentMemoryContext,
                        "ALS factors context",
                        ALLOCSET_SMALL_SIZES);

    oldcontext = MemoryContextSwitchTo(model_mcxt);

    /* alloc: max_user_id+1 for 1-based indices */
    P = als_alloc_matrix(max_user_id+1, n_factors);
    Q = als_alloc_matrix(max_item_id+1, n_factors);

    /* Random initialize for stability */
    for (i = 0; i <= max_user_id; ++i)
    {
        int f;
        for (f = 0; f < n_factors; ++f)
            P[i][f] = ((float) random()/(float) MAXRAND) * 0.1f;
    }
    for (i = 0; i <= max_item_id; ++i)
    {
        int f;
        for (f = 0; f < n_factors; ++f)
            Q[i][f] = ((float) random()/(float) MAXRAND) * 0.1f;
    }

    /* ALS SGD: fixed number of iterations, no learning rate; very simple and robust */
    int epoch, u, v, k;
    float lambda = ALS_DEFAULT_LAMBDA;
    for (epoch = 0; epoch < ALS_DEFAULT_EPOCHS; ++epoch)
    {
        for (i = 0; i < n_ratings; ++i)
        {
            u = user_ids[i];
            v = item_ids[i];
            float r_ui = ratings[i];
            float pred = dot_product(P[u], Q[v], n_factors);
            float err = r_ui - pred;
            for (k = 0; k < n_factors; ++k)
            {
                float pu = P[u][k];
                float qi = Q[v][k];
                P[u][k] += 0.01f * (err * qi - lambda * pu);
                Q[v][k] += 0.01f * (err * pu - lambda * qi);
            }
        }
    }

    MemoryContextSwitchTo(oldcontext);

    /*
     * Save the model: create tables neurondb_cf_user_factors and neurondb_cf_item_factors.
     * Model id is current timestamp in ms.
     */
    int64 model_id;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    model_id = (((int64)tv.tv_sec) * 1000 + tv.tv_usec / 1000);

    resetStringInfo(&sql);

    appendStringInfo(&sql, "CREATE TABLE IF NOT EXISTS neurondb_cf_user_factors (model_id bigint, user_id int, factors float4[])");
    SPI_execute(sql.data, false, 0);
    resetStringInfo(&sql);
    appendStringInfo(&sql, "CREATE TABLE IF NOT EXISTS neurondb_cf_item_factors (model_id bigint, item_id int, factors float4[])");
    SPI_execute(sql.data, false, 0);

    /* Remove old rows for this model id (shouldn't exist, but for paranoia) */
    resetStringInfo(&sql);
    appendStringInfo(&sql, "DELETE FROM neurondb_cf_user_factors WHERE model_id = %ld", (long) model_id);
    SPI_execute(sql.data, false, 0);
    resetStringInfo(&sql);
    appendStringInfo(&sql, "DELETE FROM neurondb_cf_item_factors WHERE model_id = %ld", (long) model_id);
    SPI_execute(sql.data, false, 0);

    /* Insert factors into tables */
    Oid    arg_types[3] = {INT8OID, INT4OID, 1021}; /* float4[] is 1021 */
    Datum  values[3];
    char   nulls[3] = {false, false, false};
    ArrayType *array;
    int     j;

    plan = SPI_prepare(
            "INSERT INTO neurondb_cf_user_factors (model_id, user_id, factors) VALUES ($1,$2,$3)", 3, arg_types);
    if (plan == NULL)
        elog(ERROR, "Failed to prepare user_factors insert");

    for (i = 0; i <= max_user_id; ++i)
    {
        /* In this trivial ALS, skip users with P=0 (not seen in rating table) */
        bool seen = false;
        for (j = 0; j < n_ratings; ++j)
        {
            if (user_ids[j] == i)
            {
                seen = true;
                break;
            }
        }
        if (!seen)
            continue;

        array = construct_array(
                (Datum *) P[i],
                n_factors,
                FLOAT4OID,
                sizeof(float4),
                true,
                TYPALIGN_INT
        );
        values[0] = Int64GetDatum(model_id);
        values[1] = Int32GetDatum(i);
        values[2] = PointerGetDatum(array);
        ret = SPI_execute_plan(plan, values, nulls, false, 1);
        if (ret != SPI_OK_INSERT)
            elog(ERROR, "Failed to insert user_factors for user %d (model_id %ld)", i, (long) model_id);
    }

    plan = SPI_prepare(
            "INSERT INTO neurondb_cf_item_factors (model_id, item_id, factors) VALUES ($1,$2,$3)", 3, arg_types);
    if (plan == NULL)
        elog(ERROR, "Failed to prepare item_factors insert");

    for (i = 0; i <= max_item_id; ++i)
    {
        bool seen = false;
        for (j = 0; j < n_ratings; ++j)
        {
            if (item_ids[j] == i)
            {
                seen = true;
                break;
            }
        }
        if (!seen)
            continue;

        array = construct_array(
                (Datum *) Q[i],
                n_factors,
                FLOAT4OID,
                sizeof(float4),
                true,
                TYPALIGN_INT
        );
        values[0] = Int64GetDatum(model_id);
        values[1] = Int32GetDatum(i);
        values[2] = PointerGetDatum(array);
        ret = SPI_execute_plan(plan, values, nulls, false, 1);
        if (ret != SPI_OK_INSERT)
            elog(ERROR, "Failed to insert item_factors for item %d (model_id %ld)", i, (long) model_id);
    }

    als_free_matrix(P, max_user_id+1);
    als_free_matrix(Q, max_item_id+1);

    pfree(user_ids);
    pfree(item_ids);
    pfree(ratings);
    pfree(table_name_str);
    pfree(user_col_str);
    pfree(item_col_str);
    pfree(rating_col_str);
    MemoryContextDelete(model_mcxt);
    SPI_finish();

    resetStringInfo(&sql);
    appendStringInfo(&sql, "Collaborative filter model created, model_id=%ld", (long)model_id);

    PG_RETURN_TEXT_P(cstring_to_text(sql.data));
}

/*
 * recommend_items
 * Generate recommendations for given user_id from previously trained model.
 * Returns array of item_id (int4[]).
 */
PG_FUNCTION_INFO_V1(recommend_items);

Datum
recommend_items(PG_FUNCTION_ARGS)
{
    int32       model_id = PG_GETARG_INT32(0);
    int32       user_id = PG_GETARG_INT32(1);
    int32       n_items = PG_ARGISNULL(2) ? 10 : PG_GETARG_INT32(2);

    float      *user_factors = NULL;
    int         n_factors = 0;
	int        *item_ids = NULL;
	float     **item_factors = NULL;
	int         n_items_total = 0;
	int         i, j;
	int32      *top_items;
	float      *top_scores;
	StringInfoData sql;
	int         ret;

    ArrayType  *result_array;
    Datum      *elems;

    if (n_items < RECO_MIN_RESULT || n_items > RECO_MAX_RESULT)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("n_items must be between %d and %d", RECO_MIN_RESULT, RECO_MAX_RESULT)));

    if ((ret = SPI_connect()) != SPI_OK_CONNECT)
        elog(ERROR, "SPI_connect failed");

    /* Load user factors */
    initStringInfo(&sql);
    appendStringInfo(&sql,
        "SELECT factors FROM neurondb_cf_user_factors WHERE model_id = %d AND user_id = %d",
        model_id, user_id);

    ret = SPI_execute(sql.data, true, 1);
    if (ret != SPI_OK_SELECT)
    {
        SPI_finish();
        ereport(ERROR, (errmsg("Failed to load user factors")));
    }

    if (SPI_processed != 1)
    {
        SPI_finish();
        ereport(ERROR, (errmsg("No user_factors found for user %d in model %d", user_id, model_id)));
    }

    HeapTuple tuple = SPI_tuptable->vals[0];
    TupleDesc tupdesc = SPI_tuptable->tupdesc;
    bool isnull;
    Datum arr = SPI_getbinval(tuple, tupdesc, 1, &isnull);
    if (isnull)
    {
        SPI_finish();
        ereport(ERROR, (errmsg("NULL user factors array")));
    }

    ArrayType *user_vec = DatumGetArrayTypeP(arr);
    n_factors = ArrayGetNItems(ARR_NDIM(user_vec), ARR_DIMS(user_vec));
    user_factors = (float *) palloc(sizeof(float) * n_factors);
    memcpy(user_factors, ARR_DATA_PTR(user_vec), sizeof(float) * n_factors);

    /* Load all item factors */
    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "SELECT item_id, factors FROM neurondb_cf_item_factors WHERE model_id = %d", model_id);

    ret = SPI_execute(sql.data, true, 0);
    if (ret != SPI_OK_SELECT)
    {
        pfree(user_factors);
        SPI_finish();
        ereport(ERROR, (errmsg("Failed to load item factors")));
    }

    n_items_total = SPI_processed;
    if (n_items_total < 1)
    {
        pfree(user_factors);
        SPI_finish();
        ereport(ERROR, (errmsg("No items found for model")));
    }

    item_ids = (int *) palloc(sizeof(int) * n_items_total);
    item_factors = (float **) palloc(sizeof(float *) * n_items_total);
    for (i = 0; i < n_items_total; ++i)
    {
        HeapTuple itup = SPI_tuptable->vals[i];
        int item_id;
        float *fac;
        int item_n_factors;
        ArrayType *arr;
        bool isnull_item, isnull_fac;
        item_id = DatumGetInt32(SPI_getbinval(itup, SPI_tuptable->tupdesc, 1, &isnull_item));
        Datum facdatum = SPI_getbinval(itup, SPI_tuptable->tupdesc, 2, &isnull_fac);
        arr = DatumGetArrayTypeP(facdatum);
        item_n_factors = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
        if (item_n_factors != n_factors)
        {
            for (j = 0; j < i; ++j) pfree(item_factors[j]);
            pfree(item_ids);
            pfree(item_factors);
            pfree(user_factors);
            SPI_finish();
            ereport(ERROR, (errmsg("Factor dimension mismatch for item %d", item_id)));
        }
        fac = (float *) palloc(sizeof(float) * n_factors);
        memcpy(fac, ARR_DATA_PTR(arr), sizeof(float) * n_factors);
        item_ids[i] = item_id;
        item_factors[i] = fac;
    }

    /* Score all items, keep top n_items */
    top_items = (int32 *) palloc(sizeof(int32) * n_items);
    top_scores = (float *) palloc(sizeof(float) * n_items);

    /* initialize: lowest scores */
    for (i = 0; i < n_items; ++i)
    {
        top_scores[i] = -INFINITY;
        top_items[i] = -1;
    }

    for (i = 0; i < n_items_total; ++i)
    {
        float score = dot_product(user_factors, item_factors[i], n_factors);
        /* insert in top list if better than lowest */
        int minidx = 0;
        for (j = 1; j < n_items; ++j)
        {
            if (top_scores[j] < top_scores[minidx])
                minidx = j;
        }
        if (score > top_scores[minidx])
        {
            top_scores[minidx] = score;
            top_items[minidx] = item_ids[i];
        }
    }

    pfree(user_factors);
    for (i = 0; i < n_items_total; ++i)
        pfree(item_factors[i]);
    pfree(item_factors);
    pfree(item_ids);

    /* Return result as array (sorted by score descending) */
    /* Bubble down, for small N it's fine */
    for (i = 0; i < n_items-1; ++i)
    {
        for (j = i+1; j < n_items; ++j)
        {
            if (top_scores[j] > top_scores[i])
            {
                float tswap = top_scores[i];
                int32 iswap = top_items[i];
                top_scores[i] = top_scores[j];
                top_items[i] = top_items[j];
                top_scores[j] = tswap;
                top_items[j] = iswap;
            }
        }
    }

    elems = (Datum *) palloc(sizeof(Datum) * n_items);
    for (i = 0; i < n_items; ++i)
    {
        elems[i] = Int32GetDatum(top_items[i]);
    }

    result_array = construct_array(elems, n_items, INT4OID, sizeof(int32), true, 'i');

    pfree(elems);
    pfree(top_items);
    pfree(top_scores);
    SPI_finish();

    PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * recommend_content_based(item_id int, features_table text, n_recommendations int)
 * Uses cosine similarity between item_id and other items in features_table (column: item_id INT, features FLOAT4[])
 */
PG_FUNCTION_INFO_V1(recommend_content_based);

Datum
recommend_content_based(PG_FUNCTION_ARGS)
{
    int32       item_id = PG_GETARG_INT32(0);
    text       *features_table = PG_GETARG_TEXT_PP(1);
    int32       n_recommendations = PG_ARGISNULL(2) ? 10 : PG_GETARG_INT32(2);

    char       *features_table_str = text_to_cstring(features_table);
    int         ret, i, j, item_count, n_factors;
    int32      *other_ids = NULL;
    float     **other_factors = NULL;
    int         target_idx = -1;
    float      *target_vec = NULL;

    ArrayType  *result_array;
    Datum      *elems;
    int32      *top_items;
    float      *top_sims;

    if (n_recommendations < RECO_MIN_RESULT || n_recommendations > RECO_MAX_RESULT)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("n_recommendations must be between %d and %d", RECO_MIN_RESULT, RECO_MAX_RESULT)));

    if ((ret = SPI_connect()) != SPI_OK_CONNECT)
        elog(ERROR, "SPI_connect failed");

    StringInfoData sql;
    initStringInfo(&sql);
    appendStringInfo(&sql, "SELECT item_id, features FROM %s", quote_identifier(features_table_str));
    ret = SPI_execute(sql.data, true, 0);
    if (ret != SPI_OK_SELECT)
    {
        pfree(features_table_str);
        SPI_finish();
        ereport(ERROR, (errmsg("Could not SELECT item features")));
    }
    item_count = SPI_processed;
    if (item_count < 2)
    {
        pfree(features_table_str);
        SPI_finish();
        ereport(ERROR, (errmsg("Not enough items for content-based recommendations")));
    }
    other_ids = (int32 *) palloc(sizeof(int32) * item_count);
    other_factors = (float **) palloc(sizeof(float *) * item_count);

    for (i = 0; i < item_count; ++i)
    {
        HeapTuple tup = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        bool isnull_item, isnull_feat;

        int id = DatumGetInt32(SPI_getbinval(tup, tupdesc, 1, &isnull_item));
        Datum arr_datum = SPI_getbinval(tup, tupdesc, 2, &isnull_feat);
        if (isnull_item || isnull_feat)
        {
            pfree(features_table_str);
            for (j=0; j < i; ++j) pfree(other_factors[j]);
            pfree(other_factors);
            pfree(other_ids);
            SPI_finish();
            ereport(ERROR, (errmsg("NULL item or features at row %d", i+1)));
        }
        ArrayType *arr = DatumGetArrayTypeP(arr_datum);
        int nf = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
        if (i==0) n_factors = nf;
        if (nf != n_factors)
        {
            pfree(features_table_str);
            for (j=0; j < i; ++j) pfree(other_factors[j]);
            pfree(other_factors);
            pfree(other_ids);
            SPI_finish();
            ereport(ERROR, (errmsg("Feature length mismatch at row %d", i+1)));
        }
        float *vec = (float *) palloc(sizeof(float) * n_factors);
        memcpy(vec, ARR_DATA_PTR(arr), sizeof(float) * n_factors);
        other_ids[i] = id;
        other_factors[i] = vec;
        if (id == item_id)
        {
            target_idx = i;
            target_vec = vec;
        }
    }
    if (target_idx == -1)
    {
        pfree(features_table_str);
        for (j=0; j<item_count; ++j) pfree(other_factors[j]);
        pfree(other_factors);
        pfree(other_ids);
        SPI_finish();
        ereport(ERROR, (errmsg("item_id %d not found in features table", item_id)));
    }

    /* Calculate cosine similarities (skip self!) */
    top_items = (int32 *) palloc(sizeof(int32) * n_recommendations);
    top_sims = (float *) palloc(sizeof(float) * n_recommendations);
    for (i = 0; i < n_recommendations; ++i)
    {
        top_items[i] = -1;
        top_sims[i] = -INFINITY;
    }

    float target_len = sqrtf(dot_product(target_vec, target_vec, n_factors));
    for (i = 0; i < item_count; ++i)
    {
        if (i == target_idx)
            continue;
        float dot = dot_product(target_vec, other_factors[i], n_factors);
        float len = sqrtf(dot_product(other_factors[i], other_factors[i], n_factors));
        float sim = (len > 0 && target_len > 0) ? (dot/(len*target_len)) : 0.0f;
        /* keep top n_recommendations */
        int minidx = 0;
        for (j=1; j<n_recommendations; ++j)
            if (top_sims[j] < top_sims[minidx]) minidx = j;
        if (sim > top_sims[minidx])
        {
            top_sims[minidx] = sim;
            top_items[minidx] = other_ids[i];
        }
    }

    /* Sort descending */
    for (i = 0; i < n_recommendations-1; ++i)
        for(j=i+1;j<n_recommendations;++j)
            if (top_sims[j] > top_sims[i])
            {
                float t = top_sims[i]; top_sims[i]=top_sims[j]; top_sims[j]=t;
                int32 t2 = top_items[i]; top_items[i]=top_items[j]; top_items[j]=t2;
            }

    elems = (Datum *) palloc(sizeof(Datum) * n_recommendations);
    for (i=0;i<n_recommendations;++i)
        elems[i] = Int32GetDatum(top_items[i]);

    result_array = construct_array(elems, n_recommendations, INT4OID, sizeof(int32), true, 'i');

    for (i=0; i<item_count; ++i)
        pfree(other_factors[i]);
    pfree(other_factors);
    pfree(other_ids);
    pfree(features_table_str);
    pfree(elems);
    pfree(top_items);
    pfree(top_sims);

    SPI_finish();
    PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * user_similarity(user1_id int, user2_id int, ratings_table text)
 * Computes Pearson correlation coefficient between two users
 * in ratings table: (user_col integer, item_col integer, rating_col float)
 */
PG_FUNCTION_INFO_V1(user_similarity);

Datum
user_similarity(PG_FUNCTION_ARGS)
{
    int32 user1_id = PG_GETARG_INT32(0);
    int32 user2_id = PG_GETARG_INT32(1);
    text *ratings_table = PG_GETARG_TEXT_PP(2);

    char *ratings_table_str = text_to_cstring(ratings_table);
    int ret, count = 0;
    float sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
    float r = 0.0;

    if ((ret = SPI_connect()) != SPI_OK_CONNECT)
        elog(ERROR, "SPI_connect failed");

    StringInfoData sql;
    initStringInfo(&sql);

    /* Find items both users have rated */
    appendStringInfo(&sql,
        "SELECT a.rating AS x, b.rating AS y "
        "FROM %s a JOIN %s b ON a.item_col = b.item_col "
        "WHERE a.user_col = %d AND b.user_col = %d",
        quote_identifier(ratings_table_str), quote_identifier(ratings_table_str),
        user1_id, user2_id);

    ret = SPI_execute(sql.data, true, 0);
    if (ret != SPI_OK_SELECT)
    {
        pfree(ratings_table_str);
        SPI_finish();
        ereport(ERROR, (errmsg("Could not fetch user ratings")));
    }

    count = SPI_processed;
    if (count < 2)
    {
        pfree(ratings_table_str);
        SPI_finish();
        ereport(ERROR, (errmsg("Users must have at least two items in common")));
    }

    for (int i=0; i<count; ++i)
    {
        HeapTuple tup = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        float x = DatumGetFloat4(SPI_getbinval(tup, tupdesc, 1, NULL));
        float y = DatumGetFloat4(SPI_getbinval(tup, tupdesc, 2, NULL));

        sx += x;
        sy += y;
        sxx += x*x;
        syy += y*y;
        sxy += x*y;
    }

    float num = sxy - sx*sy/count;
    float den = sqrtf(sxx - sx*sx/count) * sqrtf(syy - sy*sy/count);
    if (den != 0)
        r = num / den;
    else
        r = 0.0f;

    pfree(ratings_table_str);
    SPI_finish();

    PG_RETURN_FLOAT8(r);
}

/*
 * recommend_hybrid(user_id int, cf_model_id int, content_table text, cf_weight float8, n_items int)
 * Combines CF model score and content similarity
 */
PG_FUNCTION_INFO_V1(recommend_hybrid);

Datum
recommend_hybrid(PG_FUNCTION_ARGS)
{
    int32 user_id = PG_GETARG_INT32(0);
    int32 cf_model_id = PG_GETARG_INT32(1);
    text *content_table = PG_GETARG_TEXT_PP(2);
    float8 cf_weight = PG_ARGISNULL(3) ? 0.7 : PG_GETARG_FLOAT8(3);
    int32 n_items = PG_ARGISNULL(4) ? 10 : PG_GETARG_INT32(4);

    if (cf_weight < 0.0 || cf_weight > 1.0)
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
            errmsg("cf_weight must be between 0.0 and 1.0")));
    if (n_items < RECO_MIN_RESULT || n_items > RECO_MAX_RESULT)
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
            errmsg("n_items must be between %d and %d", RECO_MIN_RESULT, RECO_MAX_RESULT)));

    char *content_table_str = text_to_cstring(content_table);

    int ret;
    if ((ret = SPI_connect()) != SPI_OK_CONNECT)
        elog(ERROR, "SPI_connect failed");

    StringInfoData sql;
    initStringInfo(&sql);

    /* 1. Load user factors */
    appendStringInfo(&sql,
        "SELECT factors FROM neurondb_cf_user_factors WHERE model_id = %d AND user_id = %d",
        cf_model_id, user_id);
    ret = SPI_execute(sql.data, true, 1);
    if (ret != SPI_OK_SELECT)
    {
        pfree(content_table_str);
        SPI_finish();
        ereport(ERROR, (errmsg("Failed to load user factors")));
    }
    if (SPI_processed != 1)
    {
        pfree(content_table_str);
        SPI_finish();
        ereport(ERROR, (errmsg("No user_factors found for user %d in model %d", user_id, cf_model_id)));
    }
    HeapTuple tuple = SPI_tuptable->vals[0];
    TupleDesc tupdesc = SPI_tuptable->tupdesc;
    bool isnull;
    Datum arr = SPI_getbinval(tuple, tupdesc, 1, &isnull);
    if (isnull)
    {
        pfree(content_table_str);
        SPI_finish();
        ereport(ERROR, (errmsg("NULL user factors array")));
    }
    ArrayType *user_vec = DatumGetArrayTypeP(arr);
    int n_factors = ArrayGetNItems(ARR_NDIM(user_vec), ARR_DIMS(user_vec));
    float *user_factors = (float *) palloc(sizeof(float) * n_factors);
    memcpy(user_factors, ARR_DATA_PTR(user_vec), sizeof(float) * n_factors);

    /* 2. Load all item factors */
    resetStringInfo(&sql);
    appendStringInfo(&sql, "SELECT item_id, factors FROM neurondb_cf_item_factors WHERE model_id = %d", cf_model_id);
    ret = SPI_execute(sql.data, true, 0);
    if (ret != SPI_OK_SELECT)
    {
        pfree(user_factors); pfree(content_table_str); SPI_finish();
        ereport(ERROR, (errmsg("Failed to load item factors")));
    }
    int n_items_total = SPI_processed, i, j;
    if (n_items_total < 1)
    {
        pfree(user_factors); pfree(content_table_str); SPI_finish();
        ereport(ERROR, (errmsg("No items found for model")));
    }
    int32 *item_ids = (int32 *) palloc(sizeof(int32) * n_items_total);
    float **item_factors = (float **) palloc(sizeof(float *) * n_items_total);
    for (i=0; i<n_items_total; ++i)
    {
        HeapTuple tup = SPI_tuptable->vals[i];
        TupleDesc desc = SPI_tuptable->tupdesc;
        bool isnull_id, isnull_fac;
        int id = DatumGetInt32(SPI_getbinval(tup, desc, 1, &isnull_id));
        Datum facdatum = SPI_getbinval(tup, desc, 2, &isnull_fac);
        ArrayType *facarr = DatumGetArrayTypeP(facdatum);
        int nf = ArrayGetNItems(ARR_NDIM(facarr), ARR_DIMS(facarr));
        if (nf != n_factors)
        {
            for (j=0;j<i;++j) pfree(item_factors[j]);
            pfree(item_factors); pfree(item_ids); pfree(user_factors); pfree(content_table_str);
            SPI_finish();
            ereport(ERROR, (errmsg("Factor dimension mismatch for item %d", id)));
        }
        float *vec = (float *) palloc(sizeof(float) * n_factors);
        memcpy(vec, ARR_DATA_PTR(facarr), sizeof(float) * n_factors);
        item_ids[i] = id; item_factors[i] = vec;
    }

    /* 3. Load all content feature vectors */
    resetStringInfo(&sql);
    appendStringInfo(&sql, "SELECT item_id, features FROM %s", quote_identifier(content_table_str));
    ret = SPI_execute(sql.data, true, 0);
    if (ret != SPI_OK_SELECT)
    {
        for (i=0;i<n_items_total;++i) pfree(item_factors[i]);
        pfree(item_factors); pfree(item_ids); pfree(user_factors); pfree(content_table_str);
        SPI_finish(); ereport(ERROR, (errmsg("Could not load item features")));
    }
    int n_feat_items = SPI_processed;
    int32 *feat_item_ids = (int32 *) palloc(sizeof(int32) * n_feat_items);
    float **content_factors = (float **) palloc(sizeof(float *) * n_feat_items);
    int nf_content = 0;

    for (i=0; i<n_feat_items; ++i)
    {
        HeapTuple tup = SPI_tuptable->vals[i];
        TupleDesc desc = SPI_tuptable->tupdesc;
        bool isnull_id, isnull_feat;
        int id = DatumGetInt32(SPI_getbinval(tup, desc, 1, &isnull_id));
        Datum arr = SPI_getbinval(tup, desc, 2, &isnull_feat);
        ArrayType *a = DatumGetArrayTypeP(arr);
        int nf = ArrayGetNItems(ARR_NDIM(a), ARR_DIMS(a));
        if (i == 0)
            nf_content = nf;
        if (nf != nf_content)
        {
            for (j=0;j<i;++j) pfree(content_factors[j]);
            for (j=0;j<n_items_total;++j) pfree(item_factors[j]);
            pfree(content_factors); pfree(feat_item_ids); pfree(item_factors); pfree(item_ids); pfree(user_factors); pfree(content_table_str);
            SPI_finish(); ereport(ERROR, (errmsg("Content vector dimension mismatch")));
        }
        float *vec = (float *) palloc(sizeof(float) * nf_content);
        memcpy(vec, ARR_DATA_PTR(a), sizeof(float) * nf_content);
        feat_item_ids[i] = id;
        content_factors[i] = vec;
    }

    /* 4. For every item_id that is in both, score as weighted sum of cf and content similarity */
    int ntop = Min(n_items, n_items_total);
    int32 *top_items = (int32 *) palloc(sizeof(int32) * ntop);
    float *top_scores = (float *) palloc(sizeof(float) * ntop);

    for (i=0; i<ntop; ++i)
    {
        top_items[i] = -1; top_scores[i]=-INFINITY;
    }

    for (i=0;i<n_items_total;++i)
    {
        int32 item = item_ids[i];
        /* find corresponding content vector */
        int featidx = -1;
        for (j=0;j<n_feat_items;++j)
            if (feat_item_ids[j]==item) {featidx=j; break;}

        if (featidx==-1)
            continue; /* skip if not present in features */

        float cf_score = dot_product(user_factors, item_factors[i], n_factors);
        float c_dot = dot_product(content_factors[featidx], content_factors[featidx], nf_content);
        float c_len = sqrtf(c_dot);
        float c_score = 1.0;
        if (c_len > 0.0)
            c_score = dot_product(content_factors[featidx], content_factors[featidx], nf_content) / (c_len * c_len);
        float score = (float)(cf_weight * cf_score + (1.0-cf_weight)*c_score);

        /* insert in top */
        int minidx=0;
        for (j=1;j<ntop;++j)
            if (top_scores[j] < top_scores[minidx]) minidx = j;
        if (score > top_scores[minidx])
        {
            top_scores[minidx]=score;
            top_items[minidx]=item;
        }
    }
    /* Sort top_items by score */
    for (i=0; i<ntop-1;++i)
        for (j=i+1; j<ntop; ++j)
            if (top_scores[j]>top_scores[i])
            {
                float t=top_scores[i]; top_scores[i]=top_scores[j]; top_scores[j]=t;
                int32 u=top_items[i]; top_items[i]=top_items[j]; top_items[j]=u;
            }

    Datum *elems = (Datum *) palloc(sizeof(Datum) * ntop);
    for (i=0;i<ntop;++i)
        elems[i]=Int32GetDatum(top_items[i]);
    ArrayType *result_array = construct_array(elems, ntop, INT4OID, sizeof(int32), true, 'i');
    /* cleanup */
    pfree(top_items); pfree(top_scores);
    for (i=0;i<n_feat_items;++i) pfree(content_factors[i]);
    for (i=0;i<n_items_total;++i) pfree(item_factors[i]);
    pfree(content_factors); pfree(feat_item_ids); pfree(item_factors); pfree(item_ids); pfree(user_factors); pfree(content_table_str); pfree(elems);

    SPI_finish();
    PG_RETURN_ARRAYTYPE_P(result_array);
}
