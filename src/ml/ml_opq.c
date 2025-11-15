/*-------------------------------------------------------------------------
 *
 * ml_opq.c
 *    Optimized Product Quantization (OPQ) with rotation matrix
 *
 * OPQ improves upon standard PQ by learning a rotation matrix R that
 * minimizes quantization error. The process:
 *
 * 1. Learn rotation matrix R using iterative optimization
 * 2. Rotate vectors: x' = R·x
 * 3. Apply standard PQ to rotated vectors
 *
 * Benefits over PQ:
 *   - Lower quantization error (better recall)
 *   - More balanced subspace partitioning
 *   - Better for clustered/correlated dimensions
 *
 * Complexity: Training is O(n·d²) for rotation learning
 *
 * Reference: "Optimized Product Quantization for Approximate Nearest 
 * Neighbor Search" (Ge et al., CVPR 2013)
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/ml/ml_opq.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"

#include "neurondb.h"
#include "neurondb_ml.h"

#include <math.h>
#include <float.h>

/*
 * train_opq_rotation
 * ------------------
 * Learn OPQ rotation matrix (simplified version).
 *
 * Full OPQ training involves iterating between:
 *   1. Fixing R, optimizing codebooks
 *   2. Fixing codebooks, optimizing R via SVD
 *
 * This simplified version initializes R to identity (PCA-like preprocessing).
 * For production, consider training externally with full OPQ algorithm.
 *
 * SQL Arguments:
 *   table_name    - Training data table
 *   vector_column - Vector column
 *   num_subspaces - Number of PQ subspaces (default: 8)
 *
 * Returns:
 *   Rotation matrix flattened as 1D array [d×d]
 *
 * Example:
 *   SELECT train_opq_rotation('vectors', 'embedding', 8);
 *
 * Notes:
 *   - This is a placeholder for identity/PCA rotation
 *   - Full OPQ requires iterative refinement (complex)
 *   - For best results, train offline with sklearn/faiss
 */
PG_FUNCTION_INFO_V1(train_opq_rotation);

Datum
train_opq_rotation(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *vector_column;
	int num_subspaces;
	char *tbl_str;
	char *col_str;
	float **data;
	int nvec, dim;
	double *rotation_matrix;
	int i;
	ArrayType *result;
	Datum *result_datums;

	/* Parse arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_subspaces = PG_ARGISNULL(2) ? 8 : PG_GETARG_INT32(2);

	if (num_subspaces < 2 || num_subspaces > 64)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("num_subspaces must be between 2 and "
				       "64")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);

	elog(DEBUG1,
		"neurondb: Training OPQ rotation (m=%d subspaces)",
		num_subspaces);

	/* Fetch training data */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (dim % num_subspaces != 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Vector dimension %d must be divisible "
				       "by num_subspaces %d",
					dim,
					num_subspaces)));

	/* Initialize rotation matrix to identity (simplified OPQ) */
	/* Full OPQ would iterate: optimize codebook -> optimize R (SVD) -> repeat */
	rotation_matrix = (double *)palloc0(sizeof(double) * dim * dim);

	for (i = 0; i < dim; i++)
		rotation_matrix[i * dim + i] = 1.0; /* Identity matrix */

	elog(NOTICE,
		"OPQ: Using identity rotation (simplified). "
		"For optimal results, train with full OPQ offline.");

	/* Build result array */
	result_datums = (Datum *)palloc(sizeof(Datum) * dim * dim);
	for (i = 0; i < dim * dim; i++)
		result_datums[i] = Float8GetDatum(rotation_matrix[i]);

	result = construct_array(result_datums,
		dim * dim,
		FLOAT8OID,
		sizeof(float8),
		FLOAT8PASSBYVAL,
		'd');

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		pfree(data[i]);
	pfree(data);
	pfree(rotation_matrix);
	pfree(result_datums);
	pfree(tbl_str);
	pfree(col_str);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * apply_opq_rotation
 * ------------------
 * Apply learned OPQ rotation to a vector.
 *
 * SQL Arguments:
 *   vector          - Input vector
 *   rotation_matrix - Rotation matrix from train_opq_rotation()
 *
 * Returns:
 *   Rotated vector
 *
 * Example:
 *   WITH rotation AS (
 *     SELECT train_opq_rotation('train_data', 'vec', 8) AS R
 *   )
 *   SELECT apply_opq_rotation(my_vector, R) FROM rotation, my_table;
 */
PG_FUNCTION_INFO_V1(apply_opq_rotation);

Datum
apply_opq_rotation(PG_FUNCTION_ARGS)
{
	ArrayType *vector_array;
	ArrayType *rotation_array;
	float8 *vector;
	float8 *rotation;
	int dim;
	int rotation_dim;
	double *rotated;
	int i, j;
	ArrayType *result;
	Datum *result_datums;

	vector_array = PG_GETARG_ARRAYTYPE_P(0);
	rotation_array = PG_GETARG_ARRAYTYPE_P(1);

	if (ARR_NDIM(vector_array) != 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Vector must be 1-dimensional")));

	dim = ARR_DIMS(vector_array)[0];
	rotation_dim = (int)sqrt((double)ARR_DIMS(rotation_array)[0]);

	if (rotation_dim * rotation_dim != ARR_DIMS(rotation_array)[0])
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Rotation matrix must be square")));

	if (dim != rotation_dim)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Dimension mismatch: vector=%d, "
				       "rotation=%d",
					dim,
					rotation_dim)));

	vector = (float8 *)ARR_DATA_PTR(vector_array);
	rotation = (float8 *)ARR_DATA_PTR(rotation_array);

	/* Apply rotation: rotated = R · vector */
	rotated = (double *)palloc0(sizeof(double) * dim);
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			rotated[i] += rotation[i * dim + j] * vector[j];

	/* Build result */
	result_datums = (Datum *)palloc(sizeof(Datum) * dim);
	for (i = 0; i < dim; i++)
		result_datums[i] = Float8GetDatum(rotated[i]);

	result = construct_array(result_datums,
		dim,
		FLOAT8OID,
		sizeof(float8),
		FLOAT8PASSBYVAL,
		'd');

	pfree(rotated);
	pfree(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for OPQ
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"

void
neurondb_gpu_register_opq_model(void)
{
	elog(DEBUG1, "OPQ GPU Model Ops registration skipped - not yet implemented");
}
