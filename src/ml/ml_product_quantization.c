/*-------------------------------------------------------------------------
 *
 * ml_product_quantization.c
 *    Detailed implementation of Product Quantization (PQ) for vector compression.
 *
 * Product Quantization (PQ) is a lossy compression technique for high-dimensional
 * vectors, commonly employed for efficient approximate nearest neighbor search.
 * 
 * The PQ scheme splits a d-dimensional vector into m contiguous subspaces of
 * equal dimension dsub = d / m. Each subspace is quantized independently using
 * k-means clustering to learn a codebook of ksub centroids per subspace.
 * Encoding replaces each subvector by the index of its nearest centroid. 
 * This provides significant compression (e.g., 8-32x), enabling large-scale search.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/ml/ml_product_quantization.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_gpu_registry.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

/*
 * Product Quantization Codebook Data Structure
 * --------------------------------------------
 * This structure describes the learned centroids for each subquantizer (subspace).
 *  - m:      the number of subspaces (subquantizers)
 *  - ksub:   number of centroids for each subspace (each code is in [0, ksub-1])
 *  - dsub:   subspace dimensionality, i.e., dsub = total_dim / m (must divide evenly)
 *  - centroids: a 3-level array: centroids[m][ksub][dsub]
 */
typedef struct PQCodebook
{
	int m; /* Number of subspaces/subquantizers */
	int ksub; /* Number of centroids for each subspace */
	int dsub; /* Subspace dimension (must be d/m) */
	float ***centroids; /* Shape: [m][ksub][dsub] */
} PQCodebook;

/*
 * train_subspace_kmeans
 * ---------------------
 * Internal function for fitting a k-means codebook on a batch of subspace vectors.
 * 
 * Args:
 *   subspace_data: 2D array of nvec vectors, each of dimension dsub, for this subspace.
 *   nvec:          Number of training vectors.
 *   dsub:          Dimension of each vector in this subspace.
 *   k:             Number of clusters/centroids for k-means (ksub).
 *   centroids:     Output: centroids[k][dsub].
 *   max_iters:     Maximum number of Lloyd iterations.
 *
 * Returns:
 *   void (centroids array is modified in-place)
 */
static void
train_subspace_kmeans(float **subspace_data,
	int nvec,
	int dsub,
	int k,
	float **centroids,
	int max_iters)
{
	int *assignments =
		NULL; /* Assignment for each vector (index of centroid) */
	int *counts = NULL; /* Number of points assigned to each centroid */
	bool changed = true;
	int iter, i, c, d;

	/* Allocate assignment array, initialize to zeros (or later random if desired) */
	assignments = (int *)palloc0(sizeof(int) * nvec);

	/* Random initialization of centroids from dataset */
	for (c = 0; c < k; c++)
	{
		int idx = rand() % nvec;
		memcpy(centroids[c], subspace_data[idx], sizeof(float) * dsub);
	}

	/* Lloyd's k-means main loop */
	for (iter = 0; iter < max_iters; iter++)
	{
		changed = false;

		/* Assignment step: assign each vector to the nearest centroid */
		for (i = 0; i < nvec; i++)
		{
			double min_dist = DBL_MAX;
			int best = -1;

			for (c = 0; c < k; c++)
			{
				double dist = 0.0;
				/* Compute squared L2 distance */
				for (d = 0; d < dsub; d++)
				{
					double diff =
						(double)subspace_data[i][d]
						- (double)centroids[c][d];
					dist += diff * diff;
				}
				if (dist < min_dist)
				{
					min_dist = dist;
					best = c;
				}
			}
			/* Only record if assignment has changed */
			if (assignments[i] != best)
			{
				assignments[i] = best;
				changed = true;
			}
		}

		if (!changed)
			break; /* Converged */

		/* Update step: recompute centroids as mean of assigned points */
		counts = (int *)palloc0(sizeof(int) * k);
		for (c = 0; c < k; c++)
			memset(centroids[c], 0, sizeof(float) * dsub);

		for (i = 0; i < nvec; i++)
		{
			c = assignments[i];
			for (d = 0; d < dsub; d++)
				centroids[c][d] += subspace_data[i][d];
			counts[c]++;
		}

		for (c = 0; c < k; c++)
		{
			if (counts[c] > 0)
			{
				for (d = 0; d < dsub; d++)
					centroids[c][d] /= counts[c];
			}
			/* If a centroid got no points, it remains from old pass */
		}
		pfree(counts);
	}

	pfree(assignments);
}

/*
 * train_pq_codebook
 * -----------------
 * Trains a PQ codebook from a table/column containing real-valued vector data.
 *
 * Args (SQL):
 *    table_name  - Name of source table containing the training vectors (text)
 *    column_name - Name of the vector column (text)
 *    m           - Number of subspaces/subquantizers (typically a divisor of dim)
 *    ksub        - Number of clusters per subspace (usually 256 for 8-bit PQ)
 *
 * Returns: PQ codebook as a bytea object (serialized in host order).
 *
 * Validation:
 *    - Number of subspaces (m) must be >=1, <=128, and divide dimensionality.
 *    - Number of centroids per subspace (ksub) must be >=2 and <=65536.
 * 
 * Serialization format (bytea):
 *    - int m
 *    - int ksub
 *    - int dsub
 *    - float centroids[m][ksub][dsub] (in order)
 */
PG_FUNCTION_INFO_V1(train_pq_codebook);

Datum
train_pq_codebook(PG_FUNCTION_ARGS)
{
	text *table_name;
	text *column_name;
	int m, ksub;
	char *tbl_str, *col_str;
	float **data = NULL; /* Training data array [nvec][dim] */
	int nvec = 0, dim = 0;
	PQCodebook codebook;
	int sub, i;
	bytea *result;
	int result_size;
	char *result_ptr;

	/* Argument parsing and validation */
	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	m = PG_GETARG_INT32(2);
	ksub = PG_GETARG_INT32(3);

	if (m < 1 || m > 128)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("m (number of subspaces) must be "
				       "between 1 and 128")));
	if (ksub < 2 || ksub > 65536)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("ksub (centroids per subspace) must be "
				       "between 2 and 65536")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
		"neurondb: Starting PQ training for table = %s.%s, m=%d, "
		"ksub=%d",
		tbl_str,
		col_str,
		m,
		ksub);

	/* Pull training vectors from the table: neurondb_fetch_vectors_from_table 
	   is expected to allocate a 2D array [nvec][dim] and fill nvec, dim */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (nvec <= 0)
		ereport(ERROR,
			(errcode(ERRCODE_DATA_EXCEPTION),
				errmsg("No training vectors found in table "
				       "'%s' column '%s'",
					tbl_str,
					col_str)));

	if (dim % m != 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Vector dimension %d must be divisible "
				       "by number of subspaces m=%d",
					dim,
					m)));

	/* Prepare codebook structure */
	codebook.m = m;
	codebook.ksub = ksub;
	codebook.dsub = dim / m;

	/* Allocate centroids 3D array: [m][ksub][dsub], each float* properly allocated */
	codebook.centroids = (float ***)palloc(sizeof(float **) * m);
	for (sub = 0; sub < m; sub++)
	{
		codebook.centroids[sub] =
			(float **)palloc(sizeof(float *) * ksub);
		for (i = 0; i < ksub; i++)
			codebook.centroids[sub][i] =
				(float *)palloc(sizeof(float) * codebook.dsub);
	}

	/* Begin PQ codebook training: fit a k-means on each subspace independently */
	for (sub = 0; sub < m; sub++)
	{
		float **subspace_data = NULL;
		int start_dim = sub * codebook.dsub;

		elog(DEBUG1,
			"neurondb: Training PQ subspace %d of %d (dims %d-%d)",
			sub + 1,
			m,
			start_dim,
			start_dim + codebook.dsub - 1);

		/* Extract subspace vectors from data (copy block of contiguous floats) */
		subspace_data = (float **)palloc(sizeof(float *) * nvec);
		for (i = 0; i < nvec; i++)
		{
			subspace_data[i] =
				(float *)palloc(sizeof(float) * codebook.dsub);
			memcpy(subspace_data[i],
				&data[i][start_dim],
				sizeof(float) * codebook.dsub);
		}

		/* Train k-means for this subspace. 100 iterations is typical. */
		train_subspace_kmeans(subspace_data,
			nvec,
			codebook.dsub,
			ksub,
			codebook.centroids[sub],
			100);

		/* Release subspace data for this subspace */
		for (i = 0; i < nvec; i++)
			pfree(subspace_data[i]);
		pfree(subspace_data);
	}

	/* Serialization step: compute total serialized length for bytea */
	result_size = sizeof(int) * 3 + /* header: m, ksub, dsub */
		m * ksub * codebook.dsub * sizeof(float); /* centroids */

	result = (bytea *)palloc(VARHDRSZ + result_size);
	SET_VARSIZE(result, VARHDRSZ + result_size);
	result_ptr = VARDATA(result);

	/* Write codebook header fields (all host order: m, ksub, dsub) */
	memcpy(result_ptr, &m, sizeof(int));
	result_ptr += sizeof(int);
	memcpy(result_ptr, &ksub, sizeof(int));
	result_ptr += sizeof(int);
	memcpy(result_ptr, &codebook.dsub, sizeof(int));
	result_ptr += sizeof(int);

	/* Write centroids sequentially: for each subspace, for each centroid, write full dsub floats */
	for (sub = 0; sub < m; sub++)
		for (i = 0; i < ksub; i++)
		{
			memcpy(result_ptr,
				codebook.centroids[sub][i],
				sizeof(float) * codebook.dsub);
			result_ptr += sizeof(float) * codebook.dsub;
		}

	/* Clean up all allocated memory for centroids and input data */
	for (sub = 0; sub < m; sub++)
	{
		for (i = 0; i < ksub; i++)
			pfree(codebook.centroids[sub][i]);
		pfree(codebook.centroids[sub]);
	}
	pfree(codebook.centroids);

	for (i = 0; i < nvec; i++)
		pfree(data[i]);
	pfree(data);

	pfree(tbl_str);
	pfree(col_str);

	PG_RETURN_BYTEA_P(result);
}

/*
 * pq_encode_vector
 * ----------------
 * Encode a query vector using a learned PQ codebook.
 * 
 * Args (SQL):
 *   vector   - Input vector (float4[])
 *   codebook - PQ codebook (bytea serialization format, see train_pq_codebook).
 *
 * Returns:
 *   Smallint array (int2[]) of length m, one code [0, ksub-1] per subspace.
 *   Each code identifies the closest centroid in the corresponding codebook subspace.
 */
PG_FUNCTION_INFO_V1(pq_encode_vector);

Datum
pq_encode_vector(PG_FUNCTION_ARGS)
{
	ArrayType *vec_array;
	bytea *codebook_bytea;
	float4 *vec_data;
	int dim, m, ksub, dsub;
	char *cb_ptr;
	float ***centroids = NULL;
	int16 *codes = NULL; /* array of length m */
	ArrayType *result;
	Datum *result_datums;
	int sub, c, d;
	int16 typlen;
	bool typbyval;
	char typalign;

	/* ----- Parse Inputs ----- */
	vec_array = PG_GETARG_ARRAYTYPE_P(0);
	codebook_bytea = PG_GETARG_BYTEA_PP(1);

	dim = ARR_DIMS(
		vec_array)[0]; /* Expect one-dimensional packed float4[] */
	vec_data = (float4 *)ARR_DATA_PTR(vec_array);

	/* ----- Deserialize Codebook ----- */
	cb_ptr = VARDATA(codebook_bytea);
	memcpy(&m, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);
	memcpy(&ksub, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);
	memcpy(&dsub, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);

	if (dim != m * dsub)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Vector dimension (%d) does not match "
				       "codebook definition (m=%d * dsub=%d).",
					dim,
					m,
					dsub)));

	/* ----- Reconstruct centroids 3D array from codebook bytea ----- */
	centroids = (float ***)palloc(sizeof(float **) * m);
	for (sub = 0; sub < m; sub++)
	{
		centroids[sub] = (float **)palloc(sizeof(float *) * ksub);
		for (c = 0; c < ksub; c++)
		{
			centroids[sub][c] =
				(float *)palloc(sizeof(float) * dsub);
			memcpy(centroids[sub][c], cb_ptr, sizeof(float) * dsub);
			cb_ptr += sizeof(float) * dsub;
		}
	}

	/* ----- Compute PQ codes for each subspace ----- */
	codes = (int16 *)palloc(sizeof(int16) * m);
	for (sub = 0; sub < m; sub++)
	{
		int start_dim = sub * dsub;
		double min_dist = DBL_MAX;
		int best = -1;

		for (c = 0; c < ksub; c++)
		{
			double dist = 0.0;
			for (d = 0; d < dsub; d++)
			{
				double diff = (double)vec_data[start_dim + d]
					- (double)centroids[sub][c][d];
				dist += diff * diff;
			}
			if (dist < min_dist)
			{
				min_dist = dist;
				best = c;
			}
		}
		codes[sub] = (int16)best;
	}

	/* ----- Build int2[] result array ----- */
	result_datums = (Datum *)palloc(sizeof(Datum) * m);
	for (sub = 0; sub < m; sub++)
		result_datums[sub] = Int16GetDatum(codes[sub]);

	get_typlenbyvalalign(INT2OID, &typlen, &typbyval, &typalign);
	result = construct_array(
		result_datums, m, INT2OID, typlen, typbyval, typalign);

	/* ----- Clean up c-allocated memory ----- */
	for (sub = 0; sub < m; sub++)
	{
		for (c = 0; c < ksub; c++)
			pfree(centroids[sub][c]);
		pfree(centroids[sub]);
	}
	pfree(centroids);
	pfree(codes);
	pfree(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * pq_asymmetric_distance
 * ----------------------
 * Compute approximate (asymmetric) L2 distance between a query (float4[])
 * and a PQ-compressed vector (codes), given a codebook.
 *
 * Args (SQL):
 *   query     - Query vector (float4[]), must match PQ codebook dim
 *   pq_codes  - PQ codes as int2[] (length m, one code per subspace)
 *   codebook  - PQ codebook (bytea)
 *
 * Returns:
 *   Float4: Approximate Euclidean distance between query and reconstructed database vector.
 *
 * Safety: raises error if dimensions mismatch or if a PQ code is out of valid range.
 */
PG_FUNCTION_INFO_V1(pq_asymmetric_distance);

Datum
pq_asymmetric_distance(PG_FUNCTION_ARGS)
{
	ArrayType *query_array, *codes_array;
	bytea *codebook_bytea;
	float4 *query_data;
	int16 *codes;
	int dim, m, ksub, dsub;
	char *cb_ptr;
	float ***centroids = NULL;
	double total_dist = 0.0;
	int sub, d;
	int16 code;

	/* ----- Parse arguments and extract data ----- */
	query_array = PG_GETARG_ARRAYTYPE_P(0);
	codes_array = PG_GETARG_ARRAYTYPE_P(1);
	codebook_bytea = PG_GETARG_BYTEA_PP(2);

	dim = ARR_DIMS(query_array)[0];
	query_data = (float4 *)ARR_DATA_PTR(query_array);
	codes = (int16 *)ARR_DATA_PTR(codes_array);

	/* ----- Deserialize codebook header and centroids ----- */
	cb_ptr = VARDATA(codebook_bytea);
	memcpy(&m, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);
	memcpy(&ksub, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);
	memcpy(&dsub, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);

	if (dim != m * dsub)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Query vector dimension (%d) does not "
				       "match codebook (m=%d * dsub=%d)",
					dim,
					m,
					dsub)));

	centroids = (float ***)palloc(sizeof(float **) * m);
	for (sub = 0; sub < m; sub++)
	{
		centroids[sub] = (float **)palloc(sizeof(float *) * ksub);
		for (int c = 0; c < ksub; c++)
		{
			centroids[sub][c] =
				(float *)palloc(sizeof(float) * dsub);
			memcpy(centroids[sub][c], cb_ptr, sizeof(float) * dsub);
			cb_ptr += sizeof(float) * dsub;
		}
	}

	/* ----- Compute asymmetric (query-to-product) distance ----- */
	total_dist = 0.0;
	for (sub = 0; sub < m; sub++)
	{
		int start_dim = sub * dsub;
		code = codes[sub];

		if (code < 0 || code >= ksub)
			ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
					errmsg("Invalid PQ code %d at subspace "
					       "%d (valid: 0-%d)",
						code,
						sub,
						ksub - 1)));

		for (d = 0; d < dsub; d++)
		{
			double diff = (double)query_data[start_dim + d]
				- (double)centroids[sub][code][d];
			total_dist += diff * diff;
		}
	}

	/* ----- Free memory for reconstructed centroids ----- */
	for (sub = 0; sub < m; sub++)
	{
		for (int c = 0; c < ksub; c++)
			pfree(centroids[sub][c]);
		pfree(centroids[sub]);
	}
	pfree(centroids);

	PG_RETURN_FLOAT4((float4)sqrt(total_dist));
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for ProductQuantization
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"

void
neurondb_gpu_register_product_quantization_model(void)
{
	elog(DEBUG1, "ProductQuantization GPU Model Ops registration skipped - not yet implemented");
}
