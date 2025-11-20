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
#include "utils/jsonb.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include "utils/elog.h"
#include "utils/lsyscache.h"
#include "neurondb_pgcompat.h"
#include "ml_catalog.h"

#include <math.h>
#include <string.h>
#include <sys/time.h>

/* Random number generation */
#define MAXRAND ((double)0x7fffffff)

/* ALS matrix factorization parameters */
#define ALS_DEFAULT_NFACTORS 20
#define ALS_MAX_NFACTORS 1000
#define ALS_MIN_NFACTORS 1
#define ALS_DEFAULT_EPOCHS 10
#define ALS_DEFAULT_LAMBDA 0.1
#define ALS_MAX_ITER 100

#define RECO_MIN_RESULT 1
#define RECO_MAX_RESULT 1000

/* Allocate and zero new matrix (float4 **), nrow x ncol */
static float **
als_alloc_matrix(int nrow, int ncol)
{
	float **mat;
	int i;
	mat = (float **)palloc(sizeof(float *) * nrow);
	for (i = 0; i < nrow; ++i)
	{
		mat[i] = (float *)palloc0(sizeof(float) * ncol);
	}
	return mat;
}

static void
als_free_matrix(float **mat, int nrow)
{
	int i;
	if (mat == NULL)
		return;
	for (i = 0; i < nrow; ++i)
	{
		if (mat[i])
			pfree(mat[i]);
	}
	pfree(mat);
}

/* Dot product of two float arrays */
static float
dot_product(const float *v1, const float *v2, int n)
{
	float s = 0.0f;
	int i;
	if (!v1 || !v2 || n <= 0)
		return 0.0f;
	for (i = 0; i < n; ++i)
		s += v1[i] * v2[i];
	return s;
}

/*
 * train_collaborative_filter
 * Trains ALS matrix factorization from a ratings table.
 * Returns a model id. The model is saved as two tables: user_factors and item_factors.
 */
PG_FUNCTION_INFO_V1(train_collaborative_filter);
PG_FUNCTION_INFO_V1(predict_collaborative_filter);
PG_FUNCTION_INFO_V1(evaluate_collaborative_filter_by_model_id);

Datum
train_collaborative_filter(PG_FUNCTION_ARGS)
{
	text *table_name = PG_GETARG_TEXT_PP(0);
	text *user_col = PG_GETARG_TEXT_PP(1);
	text *item_col = PG_GETARG_TEXT_PP(2);
	text *rating_col = PG_GETARG_TEXT_PP(3);
	int32 n_factors =
		PG_ARGISNULL(4) ? ALS_DEFAULT_NFACTORS : PG_GETARG_INT32(4);

	char *table_name_str = text_to_cstring(table_name);
	char *user_col_str = text_to_cstring(user_col);
	char *item_col_str = text_to_cstring(item_col);
	char *rating_col_str = text_to_cstring(rating_col);

	int ret;
	MemoryContext oldcontext = NULL, model_mcxt = NULL;
	int n_row, i, max_user_id = 0, max_item_id = 0;
	int *user_ids = NULL, *item_ids = NULL, n_ratings = 0;
	float *ratings = NULL;
	float **P = NULL, **Q = NULL;
	StringInfoData sql;
	SPIPlanPtr plan = NULL;

	if (n_factors < ALS_MIN_NFACTORS || n_factors > ALS_MAX_NFACTORS)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("n_factors must be between %d and %d",
					ALS_MIN_NFACTORS,
					ALS_MAX_NFACTORS)));

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
		ereport(ERROR,
			(errmsg("Failed to execute SQL for ratings: %s",
				sql.data)));
	}

	n_row = SPI_processed;
	if (n_row <= 1)
	{
		SPI_finish();
		ereport(ERROR,
			(errmsg("Not enough ratings found to train model")));
	}

	user_ids = (int *)palloc(sizeof(int) * n_row);
	item_ids = (int *)palloc(sizeof(int) * n_row);
	ratings = (float *)palloc(sizeof(float) * n_row);
	n_ratings = n_row;

	for (i = 0; i < n_row; ++i)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		bool isnull[3] = { false, false, false };
		int32 user = 0, item = 0;
		float r = 0.0f;

		user = DatumGetInt32(
			SPI_getbinval(tuple, tupdesc, 1, &isnull[0]));
		item = DatumGetInt32(
			SPI_getbinval(tuple, tupdesc, 2, &isnull[1]));

		if (isnull[0] || isnull[1])
		{
			SPI_finish();
			ereport(ERROR,
				(errmsg("user_col or item_col contains NULL at row %d",
					i + 1)));
		}

		if (SPI_gettypeid(tupdesc, 3) == FLOAT8OID)
			r = (float)DatumGetFloat8(
				SPI_getbinval(tuple, tupdesc, 3, &isnull[2]));
		else
			r = (float)DatumGetFloat4(
				SPI_getbinval(tuple, tupdesc, 3, &isnull[2]));

		if (isnull[2])
		{
			SPI_finish();
			ereport(ERROR,
				(errmsg("rating_col contains NULL at row %d",
					i + 1)));
		}
		user_ids[i] = user;
		item_ids[i] = item;
		ratings[i] = r;
		if (user > max_user_id)
			max_user_id = user;
		if (item > max_item_id)
			max_item_id = item;
	}

	model_mcxt = AllocSetContextCreate(CurrentMemoryContext,
		"ALS factors context",
		ALLOCSET_SMALL_SIZES);
	oldcontext = MemoryContextSwitchTo(model_mcxt);

	P = als_alloc_matrix(max_user_id + 1, n_factors);
	Q = als_alloc_matrix(max_item_id + 1, n_factors);

	/* Random initialize */
	for (i = 0; i <= max_user_id; ++i)
	{
		int f;
		for (f = 0; f < n_factors; ++f)
			P[i][f] = ((float)random() / (float)MAXRAND) * 0.1f;
	}
	for (i = 0; i <= max_item_id; ++i)
	{
		int f;
		for (f = 0; f < n_factors; ++f)
			Q[i][f] = ((float)random() / (float)MAXRAND) * 0.1f;
	}

	{
		int epoch, u, v, k;
		float lambda = ALS_DEFAULT_LAMBDA;
		for (epoch = 0; epoch < ALS_DEFAULT_EPOCHS; ++epoch)
		{
			for (i = 0; i < n_ratings; ++i)
			{
				float r_ui;
				float pred;
				float err;

				u = user_ids[i];
				v = item_ids[i];
				r_ui = ratings[i];
				pred = dot_product(P[u], Q[v], n_factors);
				err = r_ui - pred;
				for (k = 0; k < n_factors; ++k)
				{
					float pu = P[u][k];
					float qi = Q[v][k];
					P[u][k] += 0.01f
						* (err * qi - lambda * pu);
					Q[v][k] += 0.01f
						* (err * pu - lambda * qi);
				}
			}
		}
	}

	MemoryContextSwitchTo(oldcontext);

	/* Register model in catalog */
	{
		int32 model_id;
		MLCatalogModelSpec spec;
		memset(&spec, 0, sizeof(spec));
		spec.algorithm = "custom";  /* collaborative_filtering not in enum, use custom */
		spec.model_type = "regression";
		spec.training_table = table_name_str;
		spec.training_column = NULL;
		spec.project_name = NULL;
		spec.model_name = NULL;
		spec.parameters = NULL;
		spec.metrics = NULL;
		spec.model_data = NULL;
		spec.training_time_ms = -1;
		spec.num_samples = n_ratings;
		spec.num_features = n_factors;

		model_id = ml_catalog_register_model(&spec);
		elog(NOTICE, "DEBUG: ml_catalog returned model_id=%d (as int32)", model_id);
		if (model_id <= 0)
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
					errmsg("Failed to register collaborative filter model in catalog")));

		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"CREATE TABLE IF NOT EXISTS neurondb_cf_user_factors "
			"(model_id integer, user_id int, factors float4[])");
		SPI_execute(sql.data, false, 0);
		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"CREATE TABLE IF NOT EXISTS neurondb_cf_item_factors "
			"(model_id integer, item_id int, factors float4[])");
		SPI_execute(sql.data, false, 0);

		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"DELETE FROM neurondb_cf_user_factors WHERE model_id = %d",
			model_id);
		SPI_execute(sql.data, false, 0);
		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"DELETE FROM neurondb_cf_item_factors WHERE model_id = %d",
			model_id);
		SPI_execute(sql.data, false, 0);

		{
			Oid arg_types[3] = { INT4OID, INT4OID, 1021 };
			Datum values[3];
			char nulls[3] = { false, false, false };
			ArrayType *array = NULL;
			int j;
			plan = SPI_prepare(
				"INSERT INTO neurondb_cf_user_factors "
				"(model_id, user_id, factors) VALUES ($1,$2,$3)",
				3,
				arg_types);
			if (plan == NULL)
				elog(ERROR,
					"Failed to prepare user_factors "
					"insert");
			for (i = 0; i <= max_user_id; ++i)
			{
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
				array = construct_array((Datum *)P[i],
					n_factors,
					FLOAT4OID,
					sizeof(float4),
					true,
					TYPALIGN_INT);
				values[0] = Int32GetDatum(model_id);
				values[1] = Int32GetDatum(i);
				values[2] = PointerGetDatum(array);
				ret = SPI_execute_plan(
					plan, values, nulls, false, 1);
				if (ret != SPI_OK_INSERT)
					elog(ERROR,
						"Failed to insert user_factors for user %d (model_id %d)",
						i,
						model_id);
			}
		}

		{
			Oid arg_types[3] = { INT4OID, INT4OID, 1021 };
			Datum values[3];
			char nulls[3] = { false, false, false };
			ArrayType *array = NULL;
			int j;

			plan = SPI_prepare(
				"INSERT INTO neurondb_cf_item_factors "
				"(model_id, item_id, factors) VALUES ($1,$2,$3)",
				3,
				arg_types);
			if (plan == NULL)
				elog(ERROR,
					"Failed to prepare item_factors "
					"insert");
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
				array = construct_array((Datum *)Q[i],
					n_factors,
					FLOAT4OID,
					sizeof(float4),
					true,
					TYPALIGN_INT);
				values[0] = Int32GetDatum(model_id);
				values[1] = Int32GetDatum(i);
				values[2] = PointerGetDatum(array);
				ret = SPI_execute_plan(
					plan, values, nulls, false, 1);
				if (ret != SPI_OK_INSERT)
					elog(ERROR,
						"Failed to insert item_factors for item %d (model_id %d)",
						i,
						model_id);
			}
		}

		als_free_matrix(P, max_user_id + 1);
		als_free_matrix(Q, max_item_id + 1);

		if (user_ids)
			pfree(user_ids);
		if (item_ids)
			pfree(item_ids);
		if (ratings)
			pfree(ratings);
		if (table_name_str)
			pfree(table_name_str);
		if (user_col_str)
			pfree(user_col_str);
		if (item_col_str)
			pfree(item_col_str);
		if (rating_col_str)
			pfree(rating_col_str);
		if (model_mcxt)
			MemoryContextDelete(model_mcxt);
		
		SPI_finish();
		elog(INFO, "Collaborative filter model created, model_id=%d", model_id);
		PG_RETURN_INT32(model_id);
	}
}

/* Helper functions for loading factors from database */
static bool
als_load_user_factors(int32 model_id, int32 user_id, float **factors, int *n_factors)
{
	StringInfoData query;
	int ret;
	int n_rows;

	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT factors FROM neurondb_cf_user_factors WHERE model_id = %d AND user_id = %d",
		model_id, user_id);

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		return false;

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		return false;
	}

	n_rows = SPI_processed;
	if (n_rows != 1)
	{
		SPI_finish();
		return false;
	}

	/* Extract factors array */
	{
		HeapTuple tuple = SPI_tuptable->vals[0];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum factors_datum;
		bool factors_null;
		ArrayType *factors_array;
		Oid elmtype;
		int16 typlen;
		bool typbyval;
		char typalign;
		int n_dims;
		Datum *elems;
		bool *nulls;
		int n_elems;
		int i;

		factors_datum = SPI_getbinval(tuple, tupdesc, 1, &factors_null);
		if (factors_null)
		{
			SPI_finish();
			return false;
		}

		factors_array = DatumGetArrayTypeP(factors_datum);
		n_dims = ARR_NDIM(factors_array);
		if (n_dims != 1)
		{
			SPI_finish();
			return false;
		}

		elmtype = ARR_ELEMTYPE(factors_array);
		get_typlenbyvalalign(elmtype, &typlen, &typbyval, &typalign);

		deconstruct_array(factors_array, elmtype, typlen, typbyval, typalign,
						 &elems, &nulls, &n_elems);

		/* Allocate in parent context before SPI_finish() */
		*n_factors = n_elems;
		SPI_finish();
		
		*factors = palloc(sizeof(float) * n_elems);
		for (i = 0; i < n_elems; i++)
			(*factors)[i] = DatumGetFloat4(elems[i]);
	}

	return true;
}

static bool
als_load_item_factors(int32 model_id, int32 item_id, float ***factors, int *n_items_total, int *n_factors)
{
	StringInfoData query;
	int ret;
	int n_rows;

	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT factors FROM neurondb_cf_item_factors WHERE model_id = %d AND item_id = %d",
		model_id, item_id);

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		return false;

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		return false;
	}

	n_rows = SPI_processed;
	if (n_rows != 1)
	{
		SPI_finish();
		return false;
	}

	/* Extract factors array */
	{
		HeapTuple tuple = SPI_tuptable->vals[0];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum factors_datum;
		bool factors_null;
		ArrayType *factors_array;
		Oid elmtype;
		int16 typlen;
		bool typbyval;
		char typalign;
		int n_dims;
		Datum *elems;
		bool *nulls;
		int n_elems;
		int i;

		factors_datum = SPI_getbinval(tuple, tupdesc, 1, &factors_null);
		if (factors_null)
		{
			SPI_finish();
			return false;
		}

		factors_array = DatumGetArrayTypeP(factors_datum);
		n_dims = ARR_NDIM(factors_array);
		if (n_dims != 1)
		{
			SPI_finish();
			return false;
		}

		elmtype = ARR_ELEMTYPE(factors_array);
		get_typlenbyvalalign(elmtype, &typlen, &typbyval, &typalign);

		deconstruct_array(factors_array, elmtype, typlen, typbyval, typalign,
						 &elems, &nulls, &n_elems);

		/* Allocate in parent context before SPI_finish() */
		*n_items_total = 1;
		*n_factors = n_elems;
		SPI_finish();
		
		*factors = palloc(sizeof(float *) * 1); /* Only one item */
		(*factors)[0] = palloc(sizeof(float) * n_elems);

		for (i = 0; i < n_elems; i++)
			(*factors)[0][i] = DatumGetFloat4(elems[i]);
	}

	return true;
}

/*
 * predict_collaborative_filter
 *      Predicts rating for a specific user-item pair using trained ALS model.
 *      Arguments: int4 model_id, int4 user_id, int4 item_id
 *      Returns: float8 predicted rating
 */
Datum
predict_collaborative_filter(PG_FUNCTION_ARGS)
{
	int32 model_id = PG_GETARG_INT32(0);
	int32 user_id = PG_GETARG_INT32(1);
	int32 item_id = PG_GETARG_INT32(2);

	float *user_factors = NULL;
	float **item_factors = NULL;
	int n_factors = 0;
	int n_items_total = 0;
	float prediction = 0.0;
	int i;

	/* Load user factors */
	if (!als_load_user_factors(model_id, user_id, &user_factors, &n_factors))
	{
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("collaborative_filter: no factors found for user %d in model %d",
					user_id, model_id)));
	}

	/* Load item factors */
	if (!als_load_item_factors(model_id, item_id, &item_factors, &n_items_total, &n_factors))
	{
		pfree(user_factors);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("collaborative_filter: no factors found for item %d in model %d",
					item_id, model_id)));
	}

	/* Compute dot product of user and item factors */
	for (i = 0; i < n_factors; i++)
		prediction += user_factors[i] * item_factors[0][i];

	/* Clamp prediction to valid rating range (assuming 1-5 scale) */
	if (prediction < 1.0) prediction = 1.0;
	if (prediction > 5.0) prediction = 5.0;

	pfree(user_factors);
	for (i = 0; i < n_items_total; i++)
		pfree(item_factors[i]);
	pfree(item_factors);

	PG_RETURN_FLOAT8(prediction);
}

/*
 * evaluate_collaborative_filter_by_model_id
 *      Evaluates collaborative filtering model on a test dataset.
 *      Arguments: int4 model_id, text table_name, text user_col, text item_col, text rating_col
 *      Returns: jsonb with evaluation metrics
 */
Datum
evaluate_collaborative_filter_by_model_id(PG_FUNCTION_ARGS)
{
	int32 model_id;
	text *table_name;
	text *user_col;
	text *item_col;
	text *rating_col;
	char *tbl_str;
	char *user_str;
	char *item_str;
	char *rating_str;
	StringInfoData query;
	int ret;
	int n_ratings = 0;
	double mse = 0.0;
	double mae = 0.0;
	int i;
	StringInfoData jsonbuf;
	Jsonb *result;
	MemoryContext oldcontext;
	double rmse;

	/* Validate arguments */
	if (PG_NARGS() != 5)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_collaborative_filter_by_model_id: 5 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_collaborative_filter_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3) || PG_ARGISNULL(4))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_collaborative_filter_by_model_id: table_name, user_col, item_col, and rating_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	user_col = PG_GETARG_TEXT_PP(2);
	item_col = PG_GETARG_TEXT_PP(3);
	rating_col = PG_GETARG_TEXT_PP(4);

	tbl_str = text_to_cstring(table_name);
	user_str = text_to_cstring(user_col);
	item_str = text_to_cstring(item_col);
	rating_str = text_to_cstring(rating_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_collaborative_filter_by_model_id: SPI_connect failed")));

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
		"SELECT %s, %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL AND %s IS NOT NULL",
		user_str, item_str, rating_str, tbl_str, user_str, item_str, rating_str);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: evaluate_collaborative_filter_by_model_id: query failed")));

	n_ratings = SPI_processed;
	if (n_ratings < 2)
	{
		SPI_finish();
		pfree(tbl_str);
		pfree(user_str);
		pfree(item_str);
		pfree(rating_str);
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: evaluate_collaborative_filter_by_model_id: need at least 2 ratings, got %d",
					n_ratings)));
	}

	/* Evaluate each rating */
	for (i = 0; i < n_ratings; i++)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		Datum user_datum;
		Datum item_datum;
		Datum rating_datum;
		bool user_null;
		bool item_null;
		bool rating_null;
		int32 user_id;
		int32 item_id;
		float true_rating;
		float pred_rating;
		float error;

		user_datum = SPI_getbinval(tuple, tupdesc, 1, &user_null);
		item_datum = SPI_getbinval(tuple, tupdesc, 2, &item_null);
		rating_datum = SPI_getbinval(tuple, tupdesc, 3, &rating_null);

		if (user_null || item_null || rating_null)
			continue;

		user_id = DatumGetInt32(user_datum);
		item_id = DatumGetInt32(item_datum);
		true_rating = DatumGetFloat4(rating_datum);

		/* Get prediction */
		pred_rating = DatumGetFloat8(DirectFunctionCall3(predict_collaborative_filter,
														Int32GetDatum(model_id),
														Int32GetDatum(user_id),
														Int32GetDatum(item_id)));

		/* Compute error */
		error = true_rating - pred_rating;
		mse += error * error;
		mae += fabs(error);
	}

	SPI_finish();

	mse /= n_ratings;
	mae /= n_ratings;
	rmse = sqrt(mse);

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"n_ratings\":%d}",
		mse, mae, rmse, n_ratings);

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(jsonbuf.data)));
	pfree(jsonbuf.data);

	/* Cleanup */
	pfree(tbl_str);
	pfree(user_str);
	pfree(item_str);
	pfree(rating_str);

	PG_RETURN_JSONB_P(result);
}

/*
 * recommend_items
 */
PG_FUNCTION_INFO_V1(recommend_items);

Datum
recommend_items(PG_FUNCTION_ARGS)
{
	int32 model_id = PG_GETARG_INT32(0);
	int32 user_id = PG_GETARG_INT32(1);
	int32 n_items = PG_ARGISNULL(2) ? 10 : PG_GETARG_INT32(2);

	float *user_factors = NULL;
	int n_factors = 0;
	int *item_ids = NULL;
	float **item_factors = NULL;
	int n_items_total = 0;
	int i, j;
	int32 *top_items = NULL;
	float *top_scores = NULL;
	StringInfoData sql;
	int ret = 0;
	ArrayType *result_array = NULL;
	Datum *elems = NULL;

	if (n_items < RECO_MIN_RESULT || n_items > RECO_MAX_RESULT)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("n_items must be between %d and %d",
					RECO_MIN_RESULT,
					RECO_MAX_RESULT)));

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT factors FROM neurondb_cf_user_factors WHERE model_id = %d AND user_id = %d",
		model_id,
		user_id);

	ret = SPI_execute(sql.data, true, 1);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		ereport(ERROR, (errmsg("Failed to load user factors")));
	}
	if (SPI_processed != 1)
	{
		SPI_finish();
		ereport(ERROR,
			(errmsg("No user_factors found for user %d in model %d",
				user_id,
				model_id)));
	}

	{
		HeapTuple tuple = SPI_tuptable->vals[0];
		TupleDesc tupdesc = SPI_tuptable->tupdesc;
		bool isnull = false;
		Datum arr = SPI_getbinval(tuple, tupdesc, 1, &isnull);
		if (isnull)
		{
			SPI_finish();
			ereport(ERROR, (errmsg("NULL user factors array")));
		}
		{
			ArrayType *user_vec = DatumGetArrayTypeP(arr);
			n_factors = ArrayGetNItems(
				ARR_NDIM(user_vec), ARR_DIMS(user_vec));
			user_factors =
				(float *)palloc(sizeof(float) * n_factors);
			memcpy(user_factors,
				ARR_DATA_PTR(user_vec),
				sizeof(float) * n_factors);
		}
	}

	resetStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT item_id, factors FROM neurondb_cf_item_factors WHERE model_id = %d",
		model_id);

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		if (user_factors)
			pfree(user_factors);
		SPI_finish();
		ereport(ERROR, (errmsg("Failed to load item factors")));
	}
	n_items_total = SPI_processed;
	if (n_items_total < 1)
	{
		if (user_factors)
			pfree(user_factors);
		SPI_finish();
		ereport(ERROR, (errmsg("No items found for model")));
	}
	item_ids = (int *)palloc(sizeof(int) * n_items_total);
	item_factors = (float **)palloc(sizeof(float *) * n_items_total);

	for (i = 0; i < n_items_total; ++i)
	{
		HeapTuple itup = SPI_tuptable->vals[i];
		int item_id = 0;
		float *fac = NULL;
		int item_n_factors = 0;
		ArrayType *arr_f = NULL;
		bool isnull_item = false;
		bool isnull_fac = false;
		Datum facdatum;
		item_id = DatumGetInt32(SPI_getbinval(
			itup, SPI_tuptable->tupdesc, 1, &isnull_item));
		facdatum = SPI_getbinval(
			itup, SPI_tuptable->tupdesc, 2, &isnull_fac);
		arr_f = DatumGetArrayTypeP(facdatum);
		item_n_factors =
			ArrayGetNItems(ARR_NDIM(arr_f), ARR_DIMS(arr_f));
		if (item_n_factors != n_factors)
		{
			for (j = 0; j < i; ++j)
				pfree(item_factors[j]);
			pfree(item_ids);
			pfree(item_factors);
			pfree(user_factors);
			SPI_finish();
			ereport(ERROR,
				(errmsg("Factor dimension mismatch for item %d",
					item_id)));
		}
		fac = (float *)palloc(sizeof(float) * n_factors);
		memcpy(fac, ARR_DATA_PTR(arr_f), sizeof(float) * n_factors);
		item_ids[i] = item_id;
		item_factors[i] = fac;
	}

	top_items = (int32 *)palloc(sizeof(int32) * n_items);
	top_scores = (float *)palloc(sizeof(float) * n_items);

	for (i = 0; i < n_items; ++i)
	{
		top_scores[i] = -INFINITY;
		top_items[i] = -1;
	}

	for (i = 0; i < n_items_total; ++i)
	{
		float score =
			dot_product(user_factors, item_factors[i], n_factors);
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

	if (user_factors)
		pfree(user_factors);
	for (i = 0; i < n_items_total; ++i)
		if (item_factors[i])
			pfree(item_factors[i]);
	if (item_factors)
		pfree(item_factors);
	if (item_ids)
		pfree(item_ids);

	for (i = 0; i < n_items - 1; ++i)
	{
		for (j = i + 1; j < n_items; ++j)
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

	elems = (Datum *)palloc(sizeof(Datum) * n_items);
	for (i = 0; i < n_items; ++i)
	{
		elems[i] = Int32GetDatum(top_items[i]);
	}

	result_array = construct_array(
		elems, n_items, INT4OID, sizeof(int32), true, 'i');

	pfree(elems);
	pfree(top_items);
	pfree(top_scores);
	SPI_finish();

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * recommend_content_based
 */
PG_FUNCTION_INFO_V1(recommend_content_based);

Datum
recommend_content_based(PG_FUNCTION_ARGS)
{
	int32 item_id = PG_GETARG_INT32(0);
	text *features_table = PG_GETARG_TEXT_PP(1);
	int32 n_recommendations = PG_ARGISNULL(2) ? 10 : PG_GETARG_INT32(2);

	char *features_table_str = text_to_cstring(features_table);
	int ret, i, j, item_count, n_factors;
	int32 *other_ids = NULL;
	float **other_factors = NULL;
	int target_idx = -1;
	float *target_vec = NULL;

	ArrayType *result_array = NULL;
	Datum *elems = NULL;
	int32 *top_items = NULL;
	float *top_sims = NULL;

	if (n_recommendations < RECO_MIN_RESULT
		|| n_recommendations > RECO_MAX_RESULT)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("n_recommendations must be between %d and %d",
					RECO_MIN_RESULT,
					RECO_MAX_RESULT)));

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	{
		StringInfoData sql;
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT item_id, features FROM %s",
			quote_identifier(features_table_str));
		ret = SPI_execute(sql.data, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			if (features_table_str)
				pfree(features_table_str);
			SPI_finish();
			ereport(ERROR,
				(errmsg("Could not SELECT item features")));
		}
		item_count = SPI_processed;
		if (item_count < 2)
		{
			if (features_table_str)
				pfree(features_table_str);
			SPI_finish();
			ereport(ERROR,
				(errmsg("Not enough items for content-based "
					"recommendations")));
		}
		other_ids = (int32 *)palloc(sizeof(int32) * item_count);
		other_factors = (float **)palloc(sizeof(float *) * item_count);

		n_factors = 0;
		for (i = 0; i < item_count; ++i)
		{
			HeapTuple tup = SPI_tuptable->vals[i];
			TupleDesc tupdesc = SPI_tuptable->tupdesc;
			bool isnull_item = false, isnull_feat = false;

			int id = DatumGetInt32(
				SPI_getbinval(tup, tupdesc, 1, &isnull_item));
			Datum arr_datum =
				SPI_getbinval(tup, tupdesc, 2, &isnull_feat);
			if (isnull_item || isnull_feat)
			{
				if (features_table_str)
					pfree(features_table_str);
				for (j = 0; j < i; ++j)
					if (other_factors[j])
						pfree(other_factors[j]);
				if (other_factors)
					pfree(other_factors);
				if (other_ids)
					pfree(other_ids);
				SPI_finish();
				ereport(ERROR,
					(errmsg("NULL item or features at row %d",
						i + 1)));
			}
			{
				ArrayType *arr = DatumGetArrayTypeP(arr_datum);
				int nf = ArrayGetNItems(
					ARR_NDIM(arr), ARR_DIMS(arr));
				if (i == 0)
					n_factors = nf;
				if (nf != n_factors)
				{
					if (features_table_str)
						pfree(features_table_str);
					for (j = 0; j < i; ++j)
						if (other_factors[j])
							pfree(other_factors[j]);
					if (other_factors)
						pfree(other_factors);
					if (other_ids)
						pfree(other_ids);
					SPI_finish();
					ereport(ERROR,
						(errmsg("Feature length mismatch at row %d",
							i + 1)));
				}
				{
					float *vec = (float *)palloc(
						sizeof(float) * n_factors);
					memcpy(vec,
						ARR_DATA_PTR(arr),
						sizeof(float) * n_factors);
					other_ids[i] = id;
					other_factors[i] = vec;
					if (id == item_id)
					{
						target_idx = i;
						target_vec = vec;
					}
				}
			}
		}
		if (target_idx == -1)
		{
			if (features_table_str)
				pfree(features_table_str);
			for (j = 0; j < item_count; ++j)
				if (other_factors[j])
					pfree(other_factors[j]);
			if (other_factors)
				pfree(other_factors);
			if (other_ids)
				pfree(other_ids);
			SPI_finish();
			ereport(ERROR,
				(errmsg("item_id %d not found in features table",
					item_id)));
		}

		top_items = (int32 *)palloc(sizeof(int32) * n_recommendations);
		top_sims = (float *)palloc(sizeof(float) * n_recommendations);
		for (i = 0; i < n_recommendations; ++i)
		{
			top_items[i] = -1;
			top_sims[i] = -INFINITY;
		}

		{
			float target_len = sqrtf(
				dot_product(target_vec, target_vec, n_factors));
			for (i = 0; i < item_count; ++i)
			{
				if (i == target_idx)
					continue;
				{
					float dot = dot_product(target_vec,
						other_factors[i],
						n_factors);
					float len = sqrtf(
						dot_product(other_factors[i],
							other_factors[i],
							n_factors));
					float sim = (len > 0 && target_len > 0)
						? (dot / (len * target_len))
						: 0.0f;
					int minidx = 0;
					for (j = 1; j < n_recommendations; ++j)
						if (top_sims[j]
							< top_sims[minidx])
							minidx = j;
					if (sim > top_sims[minidx])
					{
						top_sims[minidx] = sim;
						top_items[minidx] =
							other_ids[i];
					}
				}
			}
		}

		for (i = 0; i < n_recommendations - 1; ++i)
			for (j = i + 1; j < n_recommendations; ++j)
				if (top_sims[j] > top_sims[i])
				{
					float t;
					int32 t2;

					t = top_sims[i];
					top_sims[i] = top_sims[j];
					top_sims[j] = t;
					t2 = top_items[i];
					top_items[i] = top_items[j];
					top_items[j] = t2;
				}
		elems = (Datum *)palloc(sizeof(Datum) * n_recommendations);
		for (i = 0; i < n_recommendations; ++i)
			elems[i] = Int32GetDatum(top_items[i]);
		result_array = construct_array(elems,
			n_recommendations,
			INT4OID,
			sizeof(int32),
			true,
			'i');

		for (i = 0; i < item_count; ++i)
			if (other_factors[i])
				pfree(other_factors[i]);
		if (other_factors)
			pfree(other_factors);
		if (other_ids)
			pfree(other_ids);
		if (features_table_str)
			pfree(features_table_str);
		if (elems)
			pfree(elems);
		if (top_items)
			pfree(top_items);
		if (top_sims)
			pfree(top_sims);
		SPI_finish();
		PG_RETURN_ARRAYTYPE_P(result_array);
	}
}

/*
 * user_similarity
 */
PG_FUNCTION_INFO_V1(user_similarity);

Datum
user_similarity(PG_FUNCTION_ARGS)
{
	int32 user1_id = PG_GETARG_INT32(0);
	int32 user2_id = PG_GETARG_INT32(1);
	text *ratings_table = PG_GETARG_TEXT_PP(2);

	char *ratings_table_str = text_to_cstring(ratings_table);
	int ret, count, i;
	float sx = 0.0f, sy = 0.0f, sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
	float r = 0.0f;

	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	{
		StringInfoData sql;
		initStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT a.rating AS x, b.rating AS y "
			"FROM %s a JOIN %s b ON a.item_col = b.item_col "
			"WHERE a.user_col = %d AND b.user_col = %d",
			quote_identifier(ratings_table_str),
			quote_identifier(ratings_table_str),
			user1_id,
			user2_id);

		ret = SPI_execute(sql.data, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			if (ratings_table_str)
				pfree(ratings_table_str);
			SPI_finish();
			ereport(ERROR,
				(errmsg("Could not fetch user ratings")));
		}

		count = SPI_processed;
		if (count < 2)
		{
			if (ratings_table_str)
				pfree(ratings_table_str);
			SPI_finish();
			ereport(ERROR,
				(errmsg("Users must have at least two items in "
					"common")));
		}
		for (i = 0; i < count; ++i)
		{
			HeapTuple tup = SPI_tuptable->vals[i];
			TupleDesc tupdesc = SPI_tuptable->tupdesc;
			float x = DatumGetFloat4(
				SPI_getbinval(tup, tupdesc, 1, NULL));
			float y = DatumGetFloat4(
				SPI_getbinval(tup, tupdesc, 2, NULL));
			sx += x;
			sy += y;
			sxx += x * x;
			syy += y * y;
			sxy += x * y;
		}

		{
			float num = sxy - sx * sy / count;
			float den = sqrtf(sxx - sx * sx / count)
				* sqrtf(syy - sy * sy / count);
			if (den != 0.0f)
				r = num / den;
			else
				r = 0.0f;
		}

		if (ratings_table_str)
			pfree(ratings_table_str);
		SPI_finish();
		PG_RETURN_FLOAT8((double)r);
	}
}

/*
 * recommend_hybrid
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
	{
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("cf_weight must be between 0.0 and "
				       "1.0")));
	}
	if (n_items < RECO_MIN_RESULT || n_items > RECO_MAX_RESULT)
	{
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("n_items must be between %d and %d",
					RECO_MIN_RESULT,
					RECO_MAX_RESULT)));
	}

	{
		char *content_table_str;
		int ret;
		int i;
		int j;
		StringInfoData sql;
		HeapTuple tuple;
		TupleDesc tupdesc;
		bool isnull;
		Datum arr;
		ArrayType *user_vec;
		int n_factors;
		float *user_factors;
		int n_items_total;
		int32 *item_ids;
		float **item_factors;
		int n_feat_items;
		int32 *feat_item_ids;
		float **content_factors;
		int nf_content;
		int ntop;
		int32 *top_items;
		float *top_scores;
		Datum *elems;
		ArrayType *result_array;

		content_table_str = text_to_cstring(content_table);

		if ((ret = SPI_connect()) != SPI_OK_CONNECT)
			elog(ERROR, "SPI_connect failed");

		initStringInfo(&sql);

		/* load user factors */
		appendStringInfo(&sql,
			"SELECT factors FROM neurondb_cf_user_factors WHERE model_id = %d AND user_id = %d",
			cf_model_id,
			user_id);
		ret = SPI_execute(sql.data, true, 1);
		if (ret != SPI_OK_SELECT)
		{
			if (content_table_str)
				pfree(content_table_str);
			SPI_finish();
			ereport(ERROR, (errmsg("Failed to load user factors")));
		}
		if (SPI_processed != 1)
		{
			if (content_table_str)
				pfree(content_table_str);
			SPI_finish();
			ereport(ERROR,
				(errmsg("No user_factors found for user %d in model %d",
					user_id,
					cf_model_id)));
		}
		tuple = SPI_tuptable->vals[0];
		tupdesc = SPI_tuptable->tupdesc;
		isnull = false;
		arr = SPI_getbinval(tuple, tupdesc, 1, &isnull);
		if (isnull)
		{
			if (content_table_str)
				pfree(content_table_str);
			SPI_finish();
			ereport(ERROR, (errmsg("NULL user factors array")));
		}
		user_vec = DatumGetArrayTypeP(arr);
		n_factors =
			ArrayGetNItems(ARR_NDIM(user_vec), ARR_DIMS(user_vec));
		user_factors = (float *)palloc(sizeof(float) * n_factors);
		memcpy(user_factors,
			ARR_DATA_PTR(user_vec),
			sizeof(float) * n_factors);

		/* load item factors */
		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT item_id, factors FROM neurondb_cf_item_factors WHERE model_id = %d",
			cf_model_id);
		ret = SPI_execute(sql.data, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			if (user_factors)
				pfree(user_factors);
			if (content_table_str)
				pfree(content_table_str);
			SPI_finish();
			ereport(ERROR, (errmsg("Failed to load item factors")));
		}
		n_items_total = SPI_processed;
		if (n_items_total < 1)
		{
			if (user_factors)
				pfree(user_factors);
			if (content_table_str)
				pfree(content_table_str);
			SPI_finish();
			ereport(ERROR, (errmsg("No items found for model")));
		}
		item_ids = (int32 *)palloc(sizeof(int32) * n_items_total);
		item_factors =
			(float **)palloc(sizeof(float *) * n_items_total);
		for (i = 0; i < n_items_total; ++i)
		{
			HeapTuple tup2 = SPI_tuptable->vals[i];
			TupleDesc desc = SPI_tuptable->tupdesc;
			bool isnull_id = false;
			bool isnull_fac = false;
			int id;
			Datum facdatum;
			ArrayType *facarr;
			int nf;
			float *vec;

			id = DatumGetInt32(
				SPI_getbinval(tup2, desc, 1, &isnull_id));
			facdatum = SPI_getbinval(tup2, desc, 2, &isnull_fac);
			facarr = DatumGetArrayTypeP(facdatum);
			nf = ArrayGetNItems(ARR_NDIM(facarr), ARR_DIMS(facarr));
			if (nf != n_factors)
			{
				for (j = 0; j < i; ++j)
				{
					if (item_factors[j])
						pfree(item_factors[j]);
				}
				if (item_factors)
					pfree(item_factors);
				if (item_ids)
					pfree(item_ids);
				if (user_factors)
					pfree(user_factors);
				if (content_table_str)
					pfree(content_table_str);
				SPI_finish();
				ereport(ERROR,
					(errmsg("Factor dimension mismatch for item %d",
						id)));
			}
			vec = (float *)palloc(sizeof(float) * n_factors);
			memcpy(vec,
				ARR_DATA_PTR(facarr),
				sizeof(float) * n_factors);
			item_ids[i] = id;
			item_factors[i] = vec;
		}

		resetStringInfo(&sql);
		appendStringInfo(&sql,
			"SELECT item_id, features FROM %s",
			quote_identifier(content_table_str));
		ret = SPI_execute(sql.data, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			for (i = 0; i < n_items_total; ++i)
				if (item_factors[i])
					pfree(item_factors[i]);
			if (item_factors)
				pfree(item_factors);
			if (item_ids)
				pfree(item_ids);
			if (user_factors)
				pfree(user_factors);
			if (content_table_str)
				pfree(content_table_str);
			SPI_finish();
			ereport(ERROR,
				(errmsg("Could not load item features")));
		}
		n_feat_items = SPI_processed;
		feat_item_ids = (int32 *)palloc(sizeof(int32) * n_feat_items);
		content_factors =
			(float **)palloc(sizeof(float *) * n_feat_items);
		nf_content = 0;

		for (i = 0; i < n_feat_items; ++i)
		{
			HeapTuple tupf = SPI_tuptable->vals[i];
			TupleDesc descf = SPI_tuptable->tupdesc;
			bool isnull_id = false, isnull_feat = false;
			int id = DatumGetInt32(
				SPI_getbinval(tupf, descf, 1, &isnull_id));
			Datum arr_tmp =
				SPI_getbinval(tupf, descf, 2, &isnull_feat);
			ArrayType *a = DatumGetArrayTypeP(arr_tmp);
			int nf = ArrayGetNItems(ARR_NDIM(a), ARR_DIMS(a));
			float *vec;
			if (i == 0)
				nf_content = nf;
			if (nf != nf_content)
			{
				for (j = 0; j < i; ++j)
					if (content_factors[j])
						pfree(content_factors[j]);
				for (j = 0; j < n_items_total; ++j)
					if (item_factors[j])
						pfree(item_factors[j]);
				if (content_factors)
					pfree(content_factors);
				if (feat_item_ids)
					pfree(feat_item_ids);
				if (item_factors)
					pfree(item_factors);
				if (item_ids)
					pfree(item_ids);
				if (user_factors)
					pfree(user_factors);
				if (content_table_str)
					pfree(content_table_str);
				SPI_finish();
				ereport(ERROR,
					(errmsg("Content vector dimension "
						"mismatch")));
			}
			vec = (float *)palloc(sizeof(float) * nf_content);
			memcpy(vec,
				ARR_DATA_PTR(a),
				sizeof(float) * nf_content);
			feat_item_ids[i] = id;
			content_factors[i] = vec;
		}

		ntop = Min(n_items, n_items_total);
		top_items = (int32 *)palloc(sizeof(int32) * ntop);
		top_scores = (float *)palloc(sizeof(float) * ntop);

		for (i = 0; i < ntop; ++i)
		{
			top_items[i] = -1;
			top_scores[i] = -INFINITY;
		}

		for (i = 0; i < n_items_total; ++i)
		{
			int32 item = item_ids[i];
			int featidx = -1;
			for (j = 0; j < n_feat_items; ++j)
				if (feat_item_ids[j] == item)
				{
					featidx = j;
					break;
				}
			if (featidx == -1)
				continue;
			{
				float cf_score = dot_product(user_factors,
					item_factors[i],
					n_factors);
				float c_dot =
					dot_product(content_factors[featidx],
						content_factors[featidx],
						nf_content);
				float c_len = sqrtf(c_dot);
				float c_score = (c_len > 0.0f)
					? dot_product(content_factors[featidx],
						  content_factors[featidx],
						  nf_content)
						/ (c_len * c_len)
					: 1.0f;
				float score = (float)(cf_weight * cf_score
					+ (1.0 - cf_weight) * c_score);
				int minidx = 0;
				for (j = 1; j < ntop; ++j)
					if (top_scores[j] < top_scores[minidx])
						minidx = j;
				if (score > top_scores[minidx])
				{
					top_scores[minidx] = score;
					top_items[minidx] = item;
				}
			}
		}

		for (i = 0; i < ntop - 1; ++i)
		{
			for (j = i + 1; j < ntop; ++j)
			{
				if (top_scores[j] > top_scores[i])
				{
					float t;
					int32 u;

					t = top_scores[i];
					top_scores[i] = top_scores[j];
					top_scores[j] = t;
					u = top_items[i];
					top_items[i] = top_items[j];
					top_items[j] = u;
				}
			}
		}

		elems = (Datum *)palloc(sizeof(Datum) * ntop);
		for (i = 0; i < ntop; ++i)
			elems[i] = Int32GetDatum(top_items[i]);
		result_array = construct_array(
			elems, ntop, INT4OID, sizeof(int32), true, 'i');
		if (top_items)
			pfree(top_items);
		if (top_scores)
			pfree(top_scores);
		for (i = 0; i < n_feat_items; ++i)
			if (content_factors[i])
				pfree(content_factors[i]);
		for (i = 0; i < n_items_total; ++i)
			if (item_factors[i])
				pfree(item_factors[i]);
		if (content_factors)
			pfree(content_factors);
		if (feat_item_ids)
			pfree(feat_item_ids);
		if (item_factors)
			pfree(item_factors);
		if (item_ids)
			pfree(item_ids);
		if (user_factors)
			pfree(user_factors);
		if (content_table_str)
			pfree(content_table_str);
		if (elems)
			pfree(elems);
		SPI_finish();
		PG_RETURN_ARRAYTYPE_P(result_array);
	}
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for Recommender
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"

void
neurondb_gpu_register_recommender_model(void)
{
}
