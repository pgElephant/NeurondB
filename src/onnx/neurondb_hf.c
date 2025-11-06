/*-------------------------------------------------------------------------
 *
 * neurondb_hf.c
 *	  HuggingFace model SQL interface via ONNX Runtime
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 * SPDX-License-Identifier: PostgreSQL
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "fmgr.h"
#include "funcapi.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"

#include "neurondb_onnx.h"
#include "neurondb.h"

extern char *neurondb_onnx_model_path;
extern bool neurondb_onnx_use_gpu;
extern int neurondb_onnx_threads;
extern int neurondb_onnx_cache_size;

PG_FUNCTION_INFO_V1(neurondb_hf_embedding);
PG_FUNCTION_INFO_V1(neurondb_hf_classify);
PG_FUNCTION_INFO_V1(neurondb_hf_ner);
PG_FUNCTION_INFO_V1(neurondb_hf_qa);
PG_FUNCTION_INFO_V1(neurondb_onnx_info);

Datum
neurondb_hf_embedding(PG_FUNCTION_ARGS)
{
	text *t1;
	text *t2;
	char *mname;
	char *txt;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		PG_RETURN_NULL();

	t1 = PG_GETARG_TEXT_PP(0);
	t2 = PG_GETARG_TEXT_PP(1);
	mname = text_to_cstring(t1);
	txt = text_to_cstring(t2);

	pfree(mname);
	pfree(txt);

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("HuggingFace embedding function requires ONNX models"),
			 errhint("Export models first using export_hf_to_onnx.py")));

	PG_RETURN_NULL();
}

Datum
neurondb_hf_classify(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("HuggingFace classify function requires ONNX models")));
	PG_RETURN_NULL();
}

Datum
neurondb_hf_ner(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("HuggingFace NER function requires ONNX models")));
	PG_RETURN_NULL();
}

Datum
neurondb_hf_qa(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("HuggingFace QA function requires ONNX models")));
	PG_RETURN_NULL();
}

Datum
neurondb_onnx_info(PG_FUNCTION_ARGS)
{
	StringInfoData buf;

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"available\": %s, \"version\": \"%s\"}",
					 neurondb_onnx_available() ? "true" : "false",
					 neurondb_onnx_version());

	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}
