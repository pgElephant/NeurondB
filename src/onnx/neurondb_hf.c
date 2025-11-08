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

#include <math.h>

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
	text	   *model_name_text;
	text	   *input_text;
	char	   *model_name;
	char	   *txt;
	ONNXModelSession *session;
	int32	   *token_ids;
	int32		token_length;
	ONNXTensor *input_tensor;
	ONNXTensor *output_tensor;
	Vector	   *result;
	int32		embedding_dim;
	int			i;
	int			j;
	int64		input_shape[2];
	float	   *input_data;
	float		sum;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		PG_RETURN_NULL();

	model_name_text = PG_GETARG_TEXT_P(0);
	input_text = PG_GETARG_TEXT_P(1);

	model_name = text_to_cstring(model_name_text);
	txt = text_to_cstring(input_text);

	/* Load or get cached model */
	session = neurondb_onnx_get_or_load_model(model_name, ONNX_MODEL_EMBEDDING);

	/* Tokenize input text */
	token_ids = neurondb_tokenize(txt, 128, &token_length);

	/* Convert token IDs to float array for ONNX */
	input_data = (float *) palloc(token_length * sizeof(float));
	for (i = 0; i < token_length; i++)
		input_data[i] = (float) token_ids[i];

	/* Create input tensor */
	input_shape[0] = 1;
	input_shape[1] = token_length;

	input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 2);

	/* Run inference */
	PG_TRY();
	{
		output_tensor = neurondb_onnx_run_inference(session, input_tensor);
	}
	PG_CATCH();
	{
		neurondb_onnx_free_tensor(input_tensor);
		pfree(input_data);
		pfree(token_ids);
		pfree(model_name);
		pfree(txt);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Calculate embedding dimension */
	embedding_dim = output_tensor->size / token_length;

	/* Pool embeddings (mean pooling across sequence dimension) */
	result = (Vector *) palloc0(VECTOR_SIZE(embedding_dim));
	SET_VARSIZE(result, VECTOR_SIZE(embedding_dim));
	result->dim = embedding_dim;

	for (i = 0; i < embedding_dim; i++)
	{
		sum = 0.0f;
		for (j = 0; j < token_length; j++)
			sum += output_tensor->data[j * embedding_dim + i];

		result->data[i] = sum / token_length;
	}

	/* Cleanup */
	neurondb_onnx_free_tensor(input_tensor);
	neurondb_onnx_free_tensor(output_tensor);
	pfree(input_data);
	pfree(token_ids);
	pfree(model_name);
	pfree(txt);

	PG_RETURN_POINTER(result);
}

Datum
neurondb_hf_classify(PG_FUNCTION_ARGS)
{
	text	   *model_name_text;
	text	   *input_text;
	char	   *model_name;
	char	   *txt;
	ONNXModelSession *session;
	int32	   *token_ids;
	int32		token_length;
	ONNXTensor *input_tensor;
	ONNXTensor *output_tensor;
	StringInfoData buf;
	int64		input_shape[2];
	float	   *input_data;
	float		positive_score;
	float		negative_score;
	int			i;
	char	   *predicted_label;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		PG_RETURN_NULL();

	model_name_text = PG_GETARG_TEXT_P(0);
	input_text = PG_GETARG_TEXT_P(1);

	model_name = text_to_cstring(model_name_text);
	txt = text_to_cstring(input_text);

	/* Load or get cached model */
	session = neurondb_onnx_get_or_load_model(model_name, ONNX_MODEL_CLASSIFICATION);

	/* Tokenize input text */
	token_ids = neurondb_tokenize(txt, 128, &token_length);

	/* Convert token IDs to float array for ONNX */
	input_data = (float *) palloc(token_length * sizeof(float));
	for (i = 0; i < token_length; i++)
		input_data[i] = (float) token_ids[i];

	/* Create input tensor */
	input_shape[0] = 1;
	input_shape[1] = token_length;

	input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 2);

	/* Run inference */
	PG_TRY();
	{
		output_tensor = neurondb_onnx_run_inference(session, input_tensor);
	}
	PG_CATCH();
	{
		neurondb_onnx_free_tensor(input_tensor);
		pfree(input_data);
		pfree(token_ids);
		pfree(model_name);
		pfree(txt);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Extract logits (assume binary classification: [negative, positive]) */
	if (output_tensor->size >= 2)
	{
		float		max_score;
		float		sum;

		negative_score = output_tensor->data[0];
		positive_score = output_tensor->data[1];

		/* Apply softmax */
		max_score = (positive_score > negative_score) ? positive_score : negative_score;
		negative_score = exp(negative_score - max_score);
		positive_score = exp(positive_score - max_score);
		sum = negative_score + positive_score;
		negative_score /= sum;
		positive_score /= sum;

		/* Determine predicted label */
		predicted_label = (positive_score > negative_score) ? "POSITIVE" : "NEGATIVE";

		/* Build JSON result */
		initStringInfo(&buf);
		appendStringInfo(&buf, "{\"label\": \"%s\", \"score\": %.4f}", 
						 predicted_label, 
						 (positive_score > negative_score) ? positive_score : negative_score);
	}
	else
	{
		/* Fallback for unexpected output */
		initStringInfo(&buf);
		appendStringInfo(&buf, "{\"label\": \"UNKNOWN\", \"score\": 0.0}");
	}

	/* Cleanup */
	neurondb_onnx_free_tensor(input_tensor);
	neurondb_onnx_free_tensor(output_tensor);
	pfree(input_data);
	pfree(token_ids);
	pfree(model_name);
	pfree(txt);

	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}

Datum
neurondb_hf_ner(PG_FUNCTION_ARGS)
{
	text	   *model_name_text;
	text	   *input_text;
	char	   *model_name;
	char	   *txt;
	ONNXModelSession *session;
	int32	   *token_ids;
	int32		token_length;
	ONNXTensor *input_tensor;
	ONNXTensor *output_tensor;
	StringInfoData buf;
	int64		input_shape[2];
	float	   *input_data;
	int			i;
	int			max_class;
	float		max_score;
	const char *entity_labels[] = {"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"};
	const int	num_labels = 7;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		PG_RETURN_NULL();

	model_name_text = PG_GETARG_TEXT_P(0);
	input_text = PG_GETARG_TEXT_P(1);

	model_name = text_to_cstring(model_name_text);
	txt = text_to_cstring(input_text);

	/* Load or get cached model */
	session = neurondb_onnx_get_or_load_model(model_name, ONNX_MODEL_NER);

	/* Tokenize input text */
	token_ids = neurondb_tokenize(txt, 128, &token_length);

	/* Convert token IDs to float array for ONNX */
	input_data = (float *) palloc(token_length * sizeof(float));
	for (i = 0; i < token_length; i++)
		input_data[i] = (float) token_ids[i];

	/* Create input tensor */
	input_shape[0] = 1;
	input_shape[1] = token_length;

	input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 2);

	/* Run inference */
	PG_TRY();
	{
		output_tensor = neurondb_onnx_run_inference(session, input_tensor);
	}
	PG_CATCH();
	{
		neurondb_onnx_free_tensor(input_tensor);
		pfree(input_data);
		pfree(token_ids);
		pfree(model_name);
		pfree(txt);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Build JSON result with entities */
	initStringInfo(&buf);
	appendStringInfo(&buf, "{\"entities\": [");

	/* Extract entities (simplified - just first token for demo) */
	if (output_tensor->size >= (size_t) num_labels)
	{
		max_class = 0;
		max_score = output_tensor->data[0];
		
		for (i = 1; i < num_labels && (size_t) i < output_tensor->size; i++)
		{
			if (output_tensor->data[i] > max_score)
			{
				max_score = output_tensor->data[i];
				max_class = i;
			}
		}

		if (max_class > 0)  /* Not 'O' (outside) */
		{
			appendStringInfo(&buf, "{\"entity\": \"%s\", \"label\": \"%s\", \"score\": %.4f}", 
							 txt, entity_labels[max_class], max_score);
		}
	}

	appendStringInfo(&buf, "]}");

	/* Cleanup */
	neurondb_onnx_free_tensor(input_tensor);
	neurondb_onnx_free_tensor(output_tensor);
	pfree(input_data);
	pfree(token_ids);
	pfree(model_name);
	pfree(txt);

	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}

Datum
neurondb_hf_qa(PG_FUNCTION_ARGS)
{
	text	   *model_name_text;
	text	   *question_text;
	text	   *context_text;
	char	   *model_name;
	char	   *question;
	char	   *context;
	ONNXModelSession *session;
	int32	   *token_ids;
	int32		token_length;
	ONNXTensor *input_tensor;
	ONNXTensor *output_tensor;
	StringInfoData buf;
	StringInfoData combined_text;
	int64		input_shape[2];
	float	   *input_data;
	volatile int start_idx = 0;
	volatile int end_idx = 0;
	volatile float start_score = -1000.0f;
	volatile float end_score = -1000.0f;
	int			i;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();

	model_name_text = PG_GETARG_TEXT_P(0);
	question_text = PG_GETARG_TEXT_P(1);
	context_text = PG_GETARG_TEXT_P(2);

	model_name = text_to_cstring(model_name_text);
	question = text_to_cstring(question_text);
	context = text_to_cstring(context_text);

	/* Combine question and context for QA models */
	initStringInfo(&combined_text);
	appendStringInfo(&combined_text, "%s [SEP] %s", question, context);

	/* Load or get cached model */
	session = neurondb_onnx_get_or_load_model(model_name, ONNX_MODEL_QA);

	/* Tokenize input */
	token_ids = neurondb_tokenize(combined_text.data, 256, &token_length);

	/* Convert token IDs to float array for ONNX */
	input_data = (float *) palloc(token_length * sizeof(float));
	for (i = 0; i < token_length; i++)
		input_data[i] = (float) token_ids[i];

	/* Create input tensor */
	input_shape[0] = 1;
	input_shape[1] = token_length;

	input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 2);

	/* Run inference */
	PG_TRY();
	{
		output_tensor = neurondb_onnx_run_inference(session, input_tensor);
	}
	PG_CATCH();
	{
		neurondb_onnx_free_tensor(input_tensor);
		pfree(input_data);
		pfree(token_ids);
		pfree(model_name);
		pfree(question);
		pfree(context);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Extract start and end positions (QA model outputs start/end logits) */
	/* Simplified: find max start and end positions */
	if (output_tensor->size >= (size_t) token_length * 2)
	{
		/* Find start position */
		for (i = 0; i < token_length; i++)
		{
			if (output_tensor->data[i] > start_score)
			{
				start_score = output_tensor->data[i];
				start_idx = i;
			}
		}

		/* Find end position */
		for (i = token_length; i < token_length * 2 && (size_t) i < output_tensor->size; i++)
		{
			if (output_tensor->data[i] > end_score)
			{
				end_score = output_tensor->data[i];
				end_idx = i - token_length;
			}
		}
	}

	/* Build JSON result */
	initStringInfo(&buf);
	appendStringInfo(&buf, "{\"answer\": \"Answer extracted from position %d to %d\", \"score\": %.4f, \"start\": %d, \"end\": %d}",
					 start_idx, end_idx, start_score, start_idx, end_idx);

	/* Cleanup */
	neurondb_onnx_free_tensor(input_tensor);
	neurondb_onnx_free_tensor(output_tensor);
	pfree(input_data);
	pfree(token_ids);
	pfree(model_name);
	pfree(question);
	pfree(context);

	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
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
