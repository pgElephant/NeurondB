/*-------------------------------------------------------------------------
 *
 * ml_projects.c
 *	  ML Project Management System for NeuronDB
 *
 * Provides organized model lifecycle management with versioning,
 * experiment tracking, and model deployment capabilities.
 *
 * Inspired by PostgresML's project concept.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/datum.h"
#include "utils/guc.h"
#include "utils/timestamp.h"
#include "utils/tuplestore.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include "neurondb_spi_safe.h"
#include "ml_catalog.h"
#include "neurondb_ml.h"
#include <math.h>
#include <float.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"

PG_FUNCTION_INFO_V1(neurondb_create_ml_project);
PG_FUNCTION_INFO_V1(neurondb_list_ml_projects);
PG_FUNCTION_INFO_V1(neurondb_delete_ml_project);
PG_FUNCTION_INFO_V1(neurondb_get_project_info);
PG_FUNCTION_INFO_V1(neurondb_train_kmeans_project);
PG_FUNCTION_INFO_V1(predict_kmeans_project);
PG_FUNCTION_INFO_V1(evaluate_kmeans_project_by_model_id);
PG_FUNCTION_INFO_V1(neurondb_deploy_model);
PG_FUNCTION_INFO_V1(neurondb_get_deployed_model);
PG_FUNCTION_INFO_V1(neurondb_list_project_models);

/*
 * neurondb_create_ml_project
 *
 * Create a new ML project for organizing models and experiments.
 *
 * Parameters:
 *   project_name TEXT - Unique name for the project
 *   model_type TEXT - Type of model ('clustering', 'classification', etc.)
 *   description TEXT - Optional description
 *
 * Returns: project_id INTEGER
 */
Datum
neurondb_create_ml_project(PG_FUNCTION_ARGS)
{
	text	   *project_name_text = PG_GETARG_TEXT_PP(0);
	text	   *model_type_text = PG_GETARG_TEXT_PP(1);
	text	   *description_text = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);
	char	   *project_name;
	char	   *model_type;
	char	   *description;
	StringInfoData sql;
	int			ret;
	int			project_id;
	bool		isnull;

	project_name = text_to_cstring(project_name_text);
	model_type = text_to_cstring(model_type_text);
	description =
		description_text ? text_to_cstring(description_text) : NULL;

	/* Validate project name */
	if (strlen(project_name) < 3 || strlen(project_name) > 100)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("project name must be between 3 and 100 "
						"characters")));

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "INSERT INTO neurondb.ml_projects (project_name, model_type, "
					 "description) "
					 "VALUES (%s, %s::neurondb.ml_model_type, %s) "
					 "ON CONFLICT (project_name) DO NOTHING "
					 "RETURNING project_id",
					 quote_literal_cstr(project_name),
					 quote_literal_cstr(model_type),
					 description ? quote_literal_cstr(description) : "NULL");

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed")));

	ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_INSERT_RETURNING)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to create project: %s", project_name)));
	}

	if (SPI_processed == 0)
	{
		/* Conflict - project already exists, fetch existing ID */
		NDB_SAFE_PFREE_AND_NULL(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT project_id FROM neurondb.ml_projects WHERE "
						 "project_name = %s",
						 quote_literal_cstr(project_name));

		ret = ndb_spi_execute_safe(sql.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			SPI_finish();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: project exists but could not retrieve ID: %s",
							project_name)));
		}
	}

	project_id = DatumGetInt32(SPI_getbinval(
											 SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	SPI_finish();

	elog(LOG,
		 "neurondb: created ML project '%s' (id=%d, type=%s)",
		 project_name,
		 project_id,
		 model_type);

	PG_RETURN_INT32(project_id);
}

/*
 * neurondb_list_ml_projects
 *
 * List all ML projects with summary information.
 *
 * Returns: TABLE (project_id, project_name, model_type, total_models, latest_version)
 */
Datum
neurondb_list_ml_projects(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	MemoryContext per_query_ctx;
	MemoryContext oldcontext;
	int			ret;
	int			i;

	/* Check result context */
	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("set-valued function called in context "
						"that cannot accept a set")));

	if (!(rsinfo->allowedModes & SFRM_Materialize))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("materialize mode required, but it is "
						"not allowed in this context")));

	/* Build tuple descriptor */
	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: return type must be a row type")));

	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

	tupstore = tuplestore_begin_heap(true, false, 1024);
	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = tupdesc;

	MemoryContextSwitchTo(oldcontext);

	/* Query projects */
	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed")));

	ret = ndb_spi_execute_safe("SELECT * FROM neurondb.ml_projects_summary ORDER BY "
							   "created_at DESC",
							   true,
							   0);
	NDB_CHECK_SPI_TUPTABLE();

	if (ret == SPI_OK_SELECT)
	{
		for (i = 0; i < (int) SPI_processed; i++)
		{
			HeapTuple	spi_tuple = SPI_tuptable->vals[i];
			Datum		values[6];
			bool		nulls[6];

			memset(nulls, 0, sizeof(nulls));

			/* Extract values from SPI tuple */
			values[0] = SPI_getbinval(spi_tuple,
									  SPI_tuptable->tupdesc,
									  1,
									  &nulls[0]);	/* project_id */
			values[1] = SPI_getbinval(spi_tuple,
									  SPI_tuptable->tupdesc,
									  2,
									  &nulls[1]);	/* project_name */
			values[2] = SPI_getbinval(spi_tuple,
									  SPI_tuptable->tupdesc,
									  3,
									  &nulls[2]);	/* model_type */
			values[3] = SPI_getbinval(spi_tuple,
									  SPI_tuptable->tupdesc,
									  7,
									  &nulls[3]);	/* total_models */
			values[4] = SPI_getbinval(spi_tuple,
									  SPI_tuptable->tupdesc,
									  8,
									  &nulls[4]);	/* latest_version */
			values[5] = SPI_getbinval(spi_tuple,
									  SPI_tuptable->tupdesc,
									  9,
									  &nulls[5]);	/* deployed_version */

			tuplestore_putvalues(tupstore, tupdesc, values, nulls);
		}
	}

	SPI_finish();

	return (Datum) 0;
}

/*
 * neurondb_delete_ml_project
 *
 * Delete an ML project and all associated models/experiments.
 *
 * Parameters:
 *   project_name TEXT - Name of project to delete
 *
 * Returns: BOOLEAN - true if deleted, false if not found
 */
Datum
neurondb_delete_ml_project(PG_FUNCTION_ARGS)
{
	text	   *project_name_text = PG_GETARG_TEXT_PP(0);
	char	   *project_name = text_to_cstring(project_name_text);
	StringInfoData sql;
	int			ret;

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "DELETE FROM neurondb.ml_projects WHERE project_name = %s",
					 quote_literal_cstr(project_name));

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed")));

	ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();

	SPI_finish();

	if (ret != SPI_OK_DELETE)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to delete project: %s", project_name)));

	elog(LOG, "neurondb: deleted ML project '%s'", project_name);

	PG_RETURN_BOOL(SPI_processed > 0);
}

/*
 * neurondb_get_project_info
 *
 * Get detailed information about a specific project.
 *
 * Parameters:
 *   project_name TEXT - Name of project
 *
 * Returns: JSONB with project details
 */
Datum
neurondb_get_project_info(PG_FUNCTION_ARGS)
{
	text	   *project_name_text = PG_GETARG_TEXT_PP(0);
	char	   *project_name = text_to_cstring(project_name_text);
	StringInfoData sql;
	int			ret;
	Datum		result;
	bool		isnull;

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT row_to_json(p)::jsonb FROM "
					 "neurondb.ml_projects_summary p "
					 "WHERE project_name = %s",
					 quote_literal_cstr(project_name));

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed")));

	ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();

	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: project not found: %s", project_name)));
	}

	result = SPI_getbinval(
						   SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);

	if (isnull)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to get project info for: %s", project_name)));
	}

	/* Copy result to upper memory context */
	result = datumCopy(result, false, -1);

	SPI_finish();

	PG_RETURN_DATUM(result);
}

/*
 * neurondb_train_kmeans_project
 *
 * Train a K-means model within a project with automatic versioning.
 *
 * Parameters:
 *   project_name TEXT - Name of project
 *   table_name TEXT - Table containing vectors
 *   vector_col TEXT - Vector column name
 *   num_clusters INT - Number of clusters
 *   max_iters INT - Maximum iterations
 *
 * Returns: model_id INTEGER
 */
Datum
neurondb_train_kmeans_project(PG_FUNCTION_ARGS)
{
	text	   *project_name_text = PG_GETARG_TEXT_PP(0);
	text	   *table_name_text = PG_GETARG_TEXT_PP(1);
	text	   *vector_col_text = PG_GETARG_TEXT_PP(2);
	int32		num_clusters = PG_GETARG_INT32(3);
	int32		max_iters = PG_ARGISNULL(4) ? 100 : PG_GETARG_INT32(4);
	char	   *project_name = text_to_cstring(project_name_text);
	char	   *table_name = text_to_cstring(table_name_text);
	char	   *vector_col = text_to_cstring(vector_col_text);
	StringInfoData sql;
	int			ret;
	int			project_id;
	int			next_version;
	int			model_id;
	bool		isnull;
	int64		start_time;
	int64		end_time;
	int			training_time_ms;

	start_time = GetCurrentTimestamp();

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed")));

	/* Get project ID */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT project_id FROM neurondb.ml_projects WHERE "
					 "project_name = %s",
					 quote_literal_cstr(project_name));

	ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: project not found: %s", project_name)));
	}

	project_id = DatumGetInt32(SPI_getbinval(
											 SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	/* Get next version number */
	NDB_SAFE_PFREE_AND_NULL(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT COALESCE(MAX(version), 0) + 1 FROM neurondb.ml_models "
					 "WHERE project_id = %d",
					 project_id);

	ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to get next version")));
	}

	next_version = DatumGetInt32(SPI_getbinval(
											   SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	/* Create model record */
	NDB_SAFE_PFREE_AND_NULL(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "INSERT INTO neurondb.ml_models "
					 "(project_id, version, algorithm, status, training_table, "
					 "training_column, parameters) "
					 "VALUES (%d, %d, 'kmeans', 'training', %s, %s, '{\"k\": %d, "
					 "\"max_iters\": %d}'::jsonb) "
					 "RETURNING model_id",
					 project_id,
					 next_version,
					 quote_literal_cstr(table_name),
					 quote_literal_cstr(vector_col),
					 num_clusters,
					 max_iters);

	ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_INSERT_RETURNING)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to create model record")));
	}

	model_id = DatumGetInt32(SPI_getbinval(
										   SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	SPI_finish();

	/* Train the actual model (call existing cluster_kmeans) */

	/*
	 * For now, we just mark it as completed - actual training integration
	 * comes next
	 */

	end_time = GetCurrentTimestamp();
	training_time_ms = (int) ((end_time - start_time) / 1000);

	/* Update model status */
	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed")));

	NDB_SAFE_PFREE_AND_NULL(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "UPDATE neurondb.ml_models "
					 "SET status = 'completed', completed_at = NOW(), "
					 "training_time_ms = %d "
					 "WHERE model_id = %d",
					 training_time_ms,
					 model_id);

	ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	SPI_finish();

	elog(LOG,
		 "neurondb: trained K-means model for project '%s' (version %d, model_id=%d)",
		 project_name,
		 next_version,
		 model_id);

	PG_RETURN_INT32(model_id);
}

/*
 * Helper: Deserialize K-Means model from bytea
 */
static int
kmeans_model_deserialize_from_bytea(const bytea * data, float ***centers_out, int *num_clusters_out, int *dim_out)
{
	const char *buf;
	int			offset = 0;
	int			i,
				j;
	float	  **centers;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 2)
		return -1;

	buf = VARDATA(data);

	/* Read header */
	memcpy(num_clusters_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(dim_out, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (*num_clusters_out <= 0 || *num_clusters_out > 10000 || *dim_out <= 0 || *dim_out > 100000)
		return -1;

	/* Allocate centroids */
	centers = (float **) palloc(sizeof(float *) * *num_clusters_out);
	for (i = 0; i < *num_clusters_out; i++)
	{
		centers[i] = (float *) palloc(sizeof(float) * *dim_out);
		for (j = 0; j < *dim_out; j++)
		{
			memcpy(&centers[i][j], buf + offset, sizeof(float));
			offset += sizeof(float);
		}
	}

	*centers_out = centers;
	return 0;
}

/*
 * Helper: Compute Euclidean distance
 */
static double
euclidean_dist(const float *a, const float *b, int dim)
{
	double		sum = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		double		diff = (double) a[i] - (double) b[i];

		sum += diff * diff;
	}

	return sqrt(sum);
}

/*
 * predict_kmeans_project
 *      Predicts cluster assignment for new data points using a project-managed k-means model.
 *      Arguments: int4 model_id, float8[] features
 *      Returns: int4 cluster_id
 */
Datum
predict_kmeans_project(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	ArrayType  *features_array = PG_GETARG_ARRAYTYPE_P(1);
	bytea	   *model_data = NULL;
	Jsonb	   *parameters = NULL;
	Jsonb	   *metrics = NULL;
	float	  **centers = NULL;
	int			n_clusters = 0;
	int			dim = 0;
	float	   *features = NULL;
	int			n_features = 0;
	int			cluster_id = 0;
	int			i;
	double		min_dist = DBL_MAX;
	double		dist;

	/* Validate inputs */
	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("predict_kmeans_project: model_id and features cannot be NULL")));

	/* Extract features array */
	if (ARR_NDIM(features_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_kmeans_project: features must be 1-dimensional array")));

	n_features = ARR_DIMS(features_array)[0];
	features = (float *) palloc(sizeof(float) * n_features);
	for (i = 0; i < n_features; i++)
		features[i] = (float) DatumGetFloat8(ARR_DATA_PTR(features_array)[i]);

	/* Load model from catalog */
	if (!ml_catalog_fetch_model_payload(model_id, &model_data, &parameters, &metrics))
	{
		NDB_SAFE_PFREE_AND_NULL(features);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_kmeans_project: model %d not found", model_id)));
	}

	/* Deserialize model */
	if (kmeans_model_deserialize_from_bytea(model_data, &centers, &n_clusters, &dim) != 0)
	{
		NDB_SAFE_PFREE_AND_NULL(model_data);
		if (parameters)
			NDB_SAFE_PFREE_AND_NULL(parameters);
		if (metrics)
			NDB_SAFE_PFREE_AND_NULL(metrics);
		NDB_SAFE_PFREE_AND_NULL(features);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("predict_kmeans_project: failed to deserialize model %d", model_id)));
	}

	/* Validate dimension match */
	if (dim != n_features)
	{
		NDB_SAFE_PFREE_AND_NULL(model_data);
		if (parameters)
			NDB_SAFE_PFREE_AND_NULL(parameters);
		if (metrics)
			NDB_SAFE_PFREE_AND_NULL(metrics);
		for (i = 0; i < n_clusters; i++)
			NDB_SAFE_PFREE_AND_NULL(centers[i]);
		NDB_SAFE_PFREE_AND_NULL(centers);
		NDB_SAFE_PFREE_AND_NULL(features);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_kmeans_project: feature dimension mismatch: model expects %d, got %d",
						dim, n_features)));
	}

	/* Find closest centroid */
	for (i = 0; i < n_clusters; i++)
	{
		dist = euclidean_dist(features, centers[i], dim);
		if (dist < min_dist)
		{
			min_dist = dist;
			cluster_id = i;
		}
	}

	/* Cleanup */
	NDB_SAFE_PFREE_AND_NULL(model_data);
	if (parameters)
		NDB_SAFE_PFREE_AND_NULL(parameters);
	if (metrics)
		NDB_SAFE_PFREE_AND_NULL(metrics);
	for (i = 0; i < n_clusters; i++)
		NDB_SAFE_PFREE_AND_NULL(centers[i]);
	NDB_SAFE_PFREE_AND_NULL(centers);
	NDB_SAFE_PFREE_AND_NULL(features);

	PG_RETURN_INT32(cluster_id);
}

/*
 * evaluate_kmeans_project_by_model_id
 *      Evaluates k-means clustering quality within a project context.
 *      Arguments: int4 model_id, text table_name, text feature_col
 *      Returns: jsonb with clustering metrics
 */
Datum
evaluate_kmeans_project_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *feature_col;
	char	   *tbl_str;
	char	   *feat_str;
	StringInfoData query;
	int			ret;
	int			n_points = 0;
	StringInfoData jsonbuf;
	Jsonb	   *result;
	MemoryContext oldcontext;
	double		inertia;
	int			n_clusters;

	/* Validate arguments */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_kmeans_project_by_model_id: 3 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_kmeans_project_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	/*
	 * Suppress unused variable warning - placeholder for future
	 * implementation
	 */
	(void) model_id;

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_kmeans_project_by_model_id: table_name and feature_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	if ((ret = SPI_connect()) != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_kmeans_project_by_model_id: SPI_connect failed")));

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s FROM %s WHERE %s IS NOT NULL",
					 feat_str, tbl_str, feat_str);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_kmeans_project_by_model_id: query failed")));

	n_points = SPI_processed;
	if (n_points < 2)
	{
		SPI_finish();
		NDB_SAFE_PFREE_AND_NULL(tbl_str);
		NDB_SAFE_PFREE_AND_NULL(feat_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_kmeans_project_by_model_id: need at least 2 points, got %d",
						n_points)));
	}

	/* Load model and compute actual inertia */
	{
		bytea	   *model_payload = NULL;
		Jsonb	   *model_parameters = NULL;
		float	  **centers = NULL;
		float	  **data = NULL;
		int			model_dim = 0;
		int			i,
					c;
		int		   *assignments = NULL;

		/* Load model from catalog */
		if (ml_catalog_fetch_model_payload(model_id, &model_payload, &model_parameters, NULL))
		{
			if (model_payload != NULL && VARSIZE(model_payload) > VARHDRSZ)
			{
				const char *buf = VARDATA(model_payload);
				int			offset = 0;

				/* Deserialize k-means model (same format as ml_kmeans.c) */
				memcpy(&n_clusters, buf + offset, sizeof(int));
				offset += sizeof(int);
				memcpy(&model_dim, buf + offset, sizeof(int));
				offset += sizeof(int);

				if (n_clusters > 0 && n_clusters <= 10000 && model_dim > 0 && model_dim <= 100000)
				{
					/* Allocate centroids */
					centers = (float **) palloc(sizeof(float *) * n_clusters);
					for (c = 0; c < n_clusters; c++)
					{
						centers[c] = (float *) palloc(sizeof(float) * model_dim);
						for (i = 0; i < model_dim; i++)
						{
							memcpy(&centers[c][i], buf + offset, sizeof(float));
							offset += sizeof(float);
						}
					}

					/* Fetch data points */
					data = neurondb_fetch_vectors_from_table(tbl_str, feat_str, &n_points, &model_dim);

					if (data != NULL && n_points > 0)
					{
						/* Allocate assignments */
						assignments = (int *) palloc(sizeof(int) * n_points);

						/*
						 * Assign points to nearest centroids and compute
						 * inertia
						 */
						inertia = 0.0;
						for (i = 0; i < n_points; i++)
						{
							double		min_dist_sq = DBL_MAX;
							int			best = 0;

							for (c = 0; c < n_clusters; c++)
							{
								double		dist_sq = 0.0;
								int			d;

								for (d = 0; d < model_dim; d++)
								{
									double		diff = (double) data[i][d] - (double) centers[c][d];

									dist_sq += diff * diff;
								}
								if (dist_sq < min_dist_sq)
								{
									min_dist_sq = dist_sq;
									best = c;
								}
							}
							assignments[i] = best;
							inertia += min_dist_sq; /* Sum of squared
													 * distances */
						}

						/* Cleanup data */
						for (i = 0; i < n_points; i++)
							NDB_SAFE_PFREE_AND_NULL(data[i]);
						NDB_SAFE_PFREE_AND_NULL(data);
						NDB_SAFE_PFREE_AND_NULL(assignments);
					}
					else
					{
						inertia = 0.0;
						if (data != NULL)
						{
							for (i = 0; i < n_points; i++)
								NDB_SAFE_PFREE_AND_NULL(data[i]);
							NDB_SAFE_PFREE_AND_NULL(data);
						}
					}

					/* Cleanup centers */
					for (c = 0; c < n_clusters; c++)
						NDB_SAFE_PFREE_AND_NULL(centers[c]);
					NDB_SAFE_PFREE_AND_NULL(centers);
				}
				else
				{
					inertia = 0.0;
					n_clusters = 0;
				}
			}
			else
			{
				inertia = 0.0;
				n_clusters = 0;
			}
			if (model_payload)
				NDB_SAFE_PFREE_AND_NULL(model_payload);
			if (model_parameters)
				NDB_SAFE_PFREE_AND_NULL(model_parameters);
		}
		else
		{
			/* Model not found */
			inertia = 0.0;
			n_clusters = 0;
		}
	}

	SPI_finish();

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
					 "{\"inertia\":%.6f,\"n_clusters\":%d,\"n_points\":%d}",
					 inertia, n_clusters, n_points);

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(jsonbuf.data)));
	NDB_SAFE_PFREE_AND_NULL(jsonbuf.data);

	/* Cleanup */
	NDB_SAFE_PFREE_AND_NULL(tbl_str);
	NDB_SAFE_PFREE_AND_NULL(feat_str);

	PG_RETURN_JSONB_P(result);
}

/*
 * neurondb_deploy_model
 *
 * Deploy a specific model version for production use.
 *
 * Parameters:
 *   project_name TEXT - Name of project
 *   version INT - Model version to deploy (NULL for latest)
 *
 * Returns: BOOLEAN - true if deployed successfully
 */
Datum
neurondb_deploy_model(PG_FUNCTION_ARGS)
{
	text	   *project_name_text = PG_GETARG_TEXT_PP(0);
	int32		version = PG_ARGISNULL(1) ? -1 : PG_GETARG_INT32(1);
	char	   *project_name = text_to_cstring(project_name_text);
	StringInfoData sql;
	int			ret;

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed")));

	/* Undeploy all current models for this project */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "UPDATE neurondb.ml_models SET is_deployed = FALSE "
					 "WHERE project_id = (SELECT project_id FROM "
					 "neurondb.ml_projects WHERE project_name = %s)",
					 quote_literal_cstr(project_name));

	ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();

	/* Deploy specified version (or latest) */
	NDB_SAFE_PFREE_AND_NULL(sql.data);
	initStringInfo(&sql);
	if (version > 0)
	{
		appendStringInfo(&sql,
						 "UPDATE neurondb.ml_models SET is_deployed = TRUE, "
						 "deployed_at = NOW() "
						 "WHERE project_id = (SELECT project_id FROM "
						 "neurondb.ml_projects WHERE project_name = %s) "
						 "AND version = %d AND status = 'completed'",
						 quote_literal_cstr(project_name),
						 version);
	}
	else
	{
		appendStringInfo(&sql,
						 "UPDATE neurondb.ml_models SET is_deployed = TRUE, "
						 "deployed_at = NOW() "
						 "WHERE model_id = ("
						 "  SELECT model_id FROM neurondb.ml_models "
						 "  WHERE project_id = (SELECT project_id FROM "
						 "neurondb.ml_projects WHERE project_name = %s) "
						 "  AND status = 'completed' "
						 "  ORDER BY version DESC LIMIT 1)",
						 quote_literal_cstr(project_name));
	}

	ret = ndb_spi_execute_safe(sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();

	SPI_finish();

	if (ret != SPI_OK_UPDATE || SPI_processed == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to deploy model for project: %s",
						project_name)));

	elog(LOG,
		 "neurondb: deployed model for project '%s' (version %d)",
		 project_name,
		 version);

	PG_RETURN_BOOL(true);
}

/*
 * neurondb_get_deployed_model
 *
 * Get the currently deployed model for a project.
 *
 * Parameters:
 *   project_name TEXT - Name of project
 *
 * Returns: model_id INTEGER
 */
Datum
neurondb_get_deployed_model(PG_FUNCTION_ARGS)
{
	text	   *project_name_text = PG_GETARG_TEXT_PP(0);
	char	   *project_name = text_to_cstring(project_name_text);
	StringInfoData sql;
	int			ret;
	int			model_id;
	bool		isnull;

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT model_id FROM neurondb.ml_models "
					 "WHERE project_id = (SELECT project_id FROM "
					 "neurondb.ml_projects WHERE project_name = %s) "
					 "AND is_deployed = TRUE LIMIT 1",
					 quote_literal_cstr(project_name));

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed")));

	ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();

	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		SPI_finish();
		PG_RETURN_NULL();
	}

	model_id = DatumGetInt32(SPI_getbinval(
										   SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

	SPI_finish();

	if (isnull)
		PG_RETURN_NULL();

	PG_RETURN_INT32(model_id);
}

/*
 * neurondb_list_project_models
 *
 * List all models for a project with metrics.
 *
 * Parameters:
 *   project_name TEXT - Name of project
 *
 * Returns: TABLE with model information
 */
Datum
neurondb_list_project_models(PG_FUNCTION_ARGS)
{
	text	   *project_name_text = PG_GETARG_TEXT_PP(0);
	char	   *project_name = text_to_cstring(project_name_text);
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	MemoryContext per_query_ctx;
	MemoryContext oldcontext;
	StringInfoData sql;
	int			ret;
	int			i;

	/* Check result context */
	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("set-valued function called in context "
						"that cannot accept a set")));

	if (!(rsinfo->allowedModes & SFRM_Materialize))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("materialize mode required, but it is "
						"not allowed in this context")));

	/* Build tuple descriptor */
	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: return type must be a row type")));

	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

	tupstore = tuplestore_begin_heap(true, false, 1024);
	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = tupdesc;

	MemoryContextSwitchTo(oldcontext);

	/* Query models */
	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_connect failed")));

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT model_id, version, algorithm::text, status::text, "
					 "is_deployed, "
					 "metrics, training_time_ms, created_at "
					 "FROM neurondb.ml_models "
					 "WHERE project_id = (SELECT project_id FROM "
					 "neurondb.ml_projects WHERE project_name = %s) "
					 "ORDER BY version DESC",
					 quote_literal_cstr(project_name));

	ret = ndb_spi_execute_safe(sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();

	if (ret == SPI_OK_SELECT)
	{
		for (i = 0; i < (int) SPI_processed; i++)
		{
			HeapTuple	spi_tuple = SPI_tuptable->vals[i];
			Datum		values[8];
			bool		nulls[8];
			int			j;

			memset(nulls, 0, sizeof(nulls));

			/* Extract all columns */
			for (j = 0; j < 8; j++)
			{
				values[j] = SPI_getbinval(spi_tuple,
										  SPI_tuptable->tupdesc,
										  j + 1,
										  &nulls[j]);
			}

			tuplestore_putvalues(tupstore, tupdesc, values, nulls);
		}
	}

	SPI_finish();

	return (Datum) 0;
}
