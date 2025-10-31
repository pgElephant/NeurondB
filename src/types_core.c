/*-------------------------------------------------------------------------
 *
 * types_core.c
 *		Core enterprise data types implementation
 *
 * Implements vectorp (packed SIMD), vecmap (sparse), rtext (retrievable
 * text), and vgraph (compact graph) data types with I/O functions.
 *
 * Copyright (c) 2024-2025, NeuronDB Development Group
 *
 * IDENTIFICATION
 *	  src/types_core.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "varatt.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include <zlib.h>

/*
 * vectorp_in: Parse vectorp from text
 * Format: "[1.0,2.0,3.0]"
 */
PG_FUNCTION_INFO_V1(vectorp_in);
Datum
vectorp_in(PG_FUNCTION_ARGS)
{
	char		   *str = PG_GETARG_CSTRING(0);
	VectorPacked   *result;
	float4		   *temp_data;
	int				dim;
	int				capacity;
	char		   *ptr;
	char		   *endptr;
	uint32			fingerprint;
	int				size;
	
	dim = 0;
	capacity = 16;
	
	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;
	
	if (*ptr == '[')
		ptr++;
	
	temp_data = (float4 *) palloc(sizeof(float4) * capacity);
	
	while (*ptr && *ptr != ']')
	{
		while (isspace((unsigned char) *ptr) || *ptr == ',')
			ptr++;
		
		if (*ptr == ']' || *ptr == '\0')
			break;
		
		if (dim >= capacity)
		{
			capacity *= 2;
			temp_data = (float4 *) repalloc(temp_data, sizeof(float4) * capacity);
		}
		
		temp_data[dim] = strtof(ptr, &endptr);
		if (ptr == endptr)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input for vectorp")));
		
		ptr = endptr;
		dim++;
	}
	
	if (dim == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vectorp must have at least 1 dimension")));
	
	size = VECTORP_SIZE(dim);
	result = (VectorPacked *) palloc0(size);
	SET_VARSIZE(result, size);
	
	/* Compute fingerprint (CRC32 of dimension count) */
	fingerprint = crc32(0L, Z_NULL, 0);
	fingerprint = crc32(fingerprint, (unsigned char *) &dim, sizeof(dim));
	
	result->fingerprint = fingerprint;
	result->version = 1;
	result->dim = dim;
	result->endian_guard = 0x01;	/* Little endian */
	result->flags = 0;
	
	memcpy(result->data, temp_data, sizeof(float4) * dim);
	pfree(temp_data);
	
	PG_RETURN_POINTER(result);
}

/*
 * vectorp_out: Convert vectorp to text
 */
PG_FUNCTION_INFO_V1(vectorp_out);
Datum
vectorp_out(PG_FUNCTION_ARGS)
{
	VectorPacked   *vec = (VectorPacked *) PG_GETARG_POINTER(0);
	StringInfoData	buf;
	int				i;
	
	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');
	
	for (i = 0; i < vec->dim; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%g", vec->data[i]);
	}
	
	appendStringInfoChar(&buf, ']');
	PG_RETURN_CSTRING(buf.data);
}

/*
 * vecmap_in: Parse sparse vector map
 * Format: "{dim:1000,nnz:5,indices:[0,10,20],values:[1.0,2.0,3.0]}"
 */
PG_FUNCTION_INFO_V1(vecmap_in);
Datum
vecmap_in(PG_FUNCTION_ARGS)
{
	char	   *str = PG_GETARG_CSTRING(0);
	
	(void) str;
	
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("vecmap_in not yet implemented")));
	
	PG_RETURN_NULL();
}

/*
 * vecmap_out: Convert sparse vector map to text
 */
PG_FUNCTION_INFO_V1(vecmap_out);
Datum
vecmap_out(PG_FUNCTION_ARGS)
{
	(void) fcinfo;
	
	PG_RETURN_CSTRING("{}");
}

/*
 * rtext_in: Parse retrievable text
 */
PG_FUNCTION_INFO_V1(rtext_in);
Datum
rtext_in(PG_FUNCTION_ARGS)
{
	char			*str = PG_GETARG_CSTRING(0);
	RetrievableText	*result;
	int				 text_len;
	int				 size;
	
	text_len = strlen(str);
	
	/* Basic implementation: store text, tokenize later */
	size = sizeof(RetrievableText) + text_len + 1;
	result = (RetrievableText *) palloc0(size);
	SET_VARSIZE(result, size);
	
	result->text_len = text_len;
	result->num_tokens = 0;		/* Will be computed on first access */
	result->lang_tag = 0;		/* Auto-detect */
	result->flags = 0;
	
	memcpy(RTEXT_DATA(result), str, text_len);
	
	PG_RETURN_POINTER(result);
}

/*
 * rtext_out: Convert retrievable text to string
 */
PG_FUNCTION_INFO_V1(rtext_out);
Datum
rtext_out(PG_FUNCTION_ARGS)
{
	RetrievableText *rt = (RetrievableText *) PG_GETARG_POINTER(0);
	char			*result;
	
	result = (char *) palloc(rt->text_len + 1);
	memcpy(result, RTEXT_DATA(rt), rt->text_len);
	result[rt->text_len] = '\0';
	
	PG_RETURN_CSTRING(result);
}

/*
 * vgraph_in: Parse graph structure
 */
PG_FUNCTION_INFO_V1(vgraph_in);
Datum
vgraph_in(PG_FUNCTION_ARGS)
{
	char	   *str = PG_GETARG_CSTRING(0);
	
	(void) str;
	
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("vgraph_in not yet implemented")));
	
	PG_RETURN_NULL();
}

/*
 * vgraph_out: Convert graph to text
 */
PG_FUNCTION_INFO_V1(vgraph_out);
Datum
vgraph_out(PG_FUNCTION_ARGS)
{
	(void) fcinfo;
	
	PG_RETURN_CSTRING("{}");
}

/*
 * vectorp_dims: Get dimensions of packed vector
 */
PG_FUNCTION_INFO_V1(vectorp_dims);
Datum
vectorp_dims(PG_FUNCTION_ARGS)
{
	VectorPacked *vec = (VectorPacked *) PG_GETARG_POINTER(0);
	
	PG_RETURN_INT32(vec->dim);
}

/*
 * vectorp_validate: Validate fingerprint and endianness
 */
PG_FUNCTION_INFO_V1(vectorp_validate);
Datum
vectorp_validate(PG_FUNCTION_ARGS)
{
	VectorPacked   *vec = (VectorPacked *) PG_GETARG_POINTER(0);
	uint32			expected_fp;
	uint32			dim;
	
	dim = vec->dim;
	
	/* Recompute fingerprint */
	expected_fp = crc32(0L, Z_NULL, 0);
	expected_fp = crc32(expected_fp, (unsigned char *) &dim, sizeof(dim));
	
	if (vec->fingerprint != expected_fp)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("vectorp fingerprint mismatch: corrupted data")));
	
	if (vec->endian_guard != 0x01 && vec->endian_guard != 0x10)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("vectorp endianness guard invalid")));
	
	PG_RETURN_BOOL(true);
}

