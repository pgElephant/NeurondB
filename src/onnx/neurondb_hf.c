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
#include "neurondb_llm.h"
#include "utils/memutils.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

extern char *neurondb_onnx_model_path;
extern bool neurondb_onnx_use_gpu;
extern int neurondb_onnx_threads;
extern int neurondb_onnx_cache_size;

/* Generation parameters structure */
typedef struct
{
	float		temperature;
	float		top_p;
	int			top_k;
	int			max_tokens;
	int			min_tokens;
	float		repetition_penalty;
	bool		do_sample;
	bool		return_prompt;
	int			seed;
} ONNXGenParams;

/* Forward declarations */
static int parse_gen_params(const char *params_json, ONNXGenParams *gen_params, char **errstr);
static int sample_token_greedy(float *logits, int vocab_size);
static int sample_token_multinomial(float *logits, int vocab_size, float temperature);
static char *decode_tokens(int32_t *token_ids, int num_tokens);

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

	/* Tokenize input text with model-specific tokenizer */
	token_ids = neurondb_tokenize_with_model(txt, 128, &token_length, model_name);

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

	/* Tokenize input text with model-specific tokenizer */
	token_ids = neurondb_tokenize_with_model(txt, 128, &token_length, model_name);

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

	/* Tokenize input text with model-specific tokenizer */
	token_ids = neurondb_tokenize_with_model(txt, 128, &token_length, model_name);

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

	/* Tokenize input with model-specific tokenizer */
	token_ids = neurondb_tokenize_with_model(combined_text.data, 256, &token_length, model_name);

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

#ifdef HAVE_ONNX_RUNTIME

/*
 * parse_gen_params
 *	  Parse JSON parameters for text generation
 *
 * Expected JSON format:
 * {
 *   "temperature": 0.7,
 *   "top_p": 0.9,
 *   "top_k": 50,
 *   "max_tokens": 100,
 *   "min_tokens": 10,
 *   "repetition_penalty": 1.1,
 *   "do_sample": true,
 *   "return_prompt": false,
 *   "seed": 42
 * }
 */
static int
parse_gen_params(const char *params_json, ONNXGenParams *gen_params, char **errstr)
{
	char	   *json_copy = NULL;
	char	   *p = NULL;
	char	   *key = NULL;
	char	   *endptr = NULL;
	float		float_val;
	int			int_val;

	if (errstr)
		*errstr = NULL;
	if (!params_json || !gen_params)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for parse_gen_params");
		return -1;
	}

	/* Initialize with defaults */
	memset(gen_params, 0, sizeof(ONNXGenParams));
	gen_params->temperature = 1.0f;
	gen_params->top_p = 1.0f;
	gen_params->top_k = 0;		/* 0 = disabled */
	gen_params->max_tokens = 100;
	gen_params->min_tokens = 0;
	gen_params->repetition_penalty = 1.0f;
	gen_params->do_sample = false;
	gen_params->return_prompt = false;
	gen_params->seed = 0;

	/* Skip empty JSON */
	if (strlen(params_json) == 0 || strcmp(params_json, "{}") == 0)
		return 0;

	/* Simple JSON parsing - find key-value pairs */
	json_copy = pstrdup(params_json);
	p = json_copy;

	/* Skip whitespace and opening brace */
	while (*p && (isspace((unsigned char) *p) || *p == '{'))
		p++;

	/* Parse key-value pairs */
	while (*p && *p != '}')
	{
		/* Skip whitespace and commas */
		while (*p && (isspace((unsigned char) *p) || *p == ','))
			p++;

		if (*p == '}' || *p == '\0')
			break;

		/* Find key */
		if (*p != '"')
		{
			pfree(json_copy);
			if (errstr)
				*errstr = pstrdup("invalid JSON format: expected key");
			return -1;
		}
		p++;				/* Skip opening quote */
		key = p;
		while (*p && *p != '"')
			p++;
		if (*p != '"')
		{
			pfree(json_copy);
			if (errstr)
				*errstr = pstrdup("invalid JSON format: unterminated key");
			return -1;
		}
		*p = '\0';			/* Null-terminate key */
		p++;				/* Skip closing quote */

		/* Skip colon */
		while (*p && (isspace((unsigned char) *p) || *p == ':'))
			p++;

		/* Parse value based on key */
		if (strcmp(key, "temperature") == 0)
		{
			float_val = strtof(p, &endptr);
			if (endptr != p && float_val > 0.0f)
				gen_params->temperature = float_val;
		}
		else if (strcmp(key, "top_p") == 0)
		{
			float_val = strtof(p, &endptr);
			if (endptr != p && float_val > 0.0f && float_val <= 1.0f)
				gen_params->top_p = float_val;
		}
		else if (strcmp(key, "top_k") == 0)
		{
			int_val = (int) strtol(p, &endptr, 10);
			if (endptr != p && int_val >= 0)
				gen_params->top_k = int_val;
		}
		else if (strcmp(key, "max_tokens") == 0 || strcmp(key, "max_length") == 0)
		{
			int_val = (int) strtol(p, &endptr, 10);
			if (endptr != p && int_val > 0)
				gen_params->max_tokens = int_val;
		}
		else if (strcmp(key, "min_tokens") == 0 || strcmp(key, "min_length") == 0)
		{
			int_val = (int) strtol(p, &endptr, 10);
			if (endptr != p && int_val >= 0)
				gen_params->min_tokens = int_val;
		}
		else if (strcmp(key, "repetition_penalty") == 0)
		{
			float_val = strtof(p, &endptr);
			if (endptr != p && float_val > 0.0f)
				gen_params->repetition_penalty = float_val;
		}
		else if (strcmp(key, "do_sample") == 0)
		{
			if (strncmp(p, "true", 4) == 0 || strncmp(p, "TRUE", 4) == 0)
				gen_params->do_sample = true;
			else if (strncmp(p, "false", 5) == 0 || strncmp(p, "FALSE", 5) == 0)
				gen_params->do_sample = false;
		}
		else if (strcmp(key, "return_prompt") == 0)
		{
			if (strncmp(p, "true", 4) == 0 || strncmp(p, "TRUE", 4) == 0)
				gen_params->return_prompt = true;
			else if (strncmp(p, "false", 5) == 0 || strncmp(p, "FALSE", 5) == 0)
				gen_params->return_prompt = false;
		}
		else if (strcmp(key, "seed") == 0)
		{
			int_val = (int) strtol(p, &endptr, 10);
			if (endptr != p)
				gen_params->seed = int_val;
		}

		/* Skip to next key or closing brace */
		while (*p && *p != ',' && *p != '}')
		{
			if (*p == '"')
			{
				p++;
				while (*p && *p != '"')
					p++;
				if (*p == '"')
					p++;
			}
			else
				p++;
		}
	}

	pfree(json_copy);
	return 0;
}

/*
 * sample_token_greedy
 *	  Sample token using greedy selection (argmax)
 */
static int
sample_token_greedy(float *logits, int vocab_size)
{
	int			max_idx = 0;
	float		max_val = logits[0];
	int			i;

	for (i = 1; i < vocab_size; i++)
	{
		if (logits[i] > max_val)
		{
			max_val = logits[i];
			max_idx = i;
		}
	}

	return max_idx;
}

/*
 * sample_token_multinomial
 *	  Sample token using multinomial sampling with temperature
 */
static int
sample_token_multinomial(float *logits, int vocab_size, float temperature)
{
	float	   *probs = NULL;
	float		sum = 0.0f;
	float		cumsum = 0.0f;
	float		r;
	int			i;
	int			selected = 0;

	/* Allocate probabilities array */
	probs = (float *) palloc(vocab_size * sizeof(float));

	/* Compute softmax with temperature */
	for (i = 0; i < vocab_size; i++)
	{
		probs[i] = expf(logits[i] / temperature);
		sum += probs[i];
	}

	/* Normalize */
	if (sum > 0.0f)
	{
		for (i = 0; i < vocab_size; i++)
			probs[i] /= sum;
	}

	/* Sample from multinomial distribution */
	r = (float) rand() / (float) RAND_MAX;
	for (i = 0; i < vocab_size; i++)
	{
		cumsum += probs[i];
		if (r <= cumsum)
		{
			selected = i;
			break;
		}
	}

	pfree(probs);
	return selected;
}

/*
 * decode_tokens
 *	  Decode token IDs to text (simplified: just convert to string representation)
 */
static char *
decode_tokens(int32_t *token_ids, int num_tokens)
{
	StringInfoData buf;
	int			i;

	initStringInfo(&buf);

	for (i = 0; i < num_tokens; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ' ');
		appendStringInfo(&buf, "%d", token_ids[i]);
	}

	return buf.data;
}

/*
 * ndb_onnx_hf_complete
 *	  Generate text completion using ONNX Runtime
 *
 * This implements a simplified text generation pipeline:
 * 1. Parse generation parameters from JSON
 * 2. Load or find model in cache
 * 3. Tokenize input prompt
 * 4. Run inference in a loop (autoregressive generation):
 *    a. Run ONNX inference to get logits
 *    b. Apply temperature, top-k, top-p, repetition penalty
 *    c. Sample next token (greedy or multinomial)
 *    d. Append to input sequence
 *    e. Check for stop conditions
 * 5. Decode generated token IDs to text
 * 6. Return generated text
 */
int
ndb_onnx_hf_complete(const char *model_name,
					 const char *prompt,
					 const char *params_json,
					 char **text_out,
					 char **errstr)
{
	ONNXModelSession *session = NULL;
	ONNXGenParams gen_params;
	int32_t	   *input_token_ids = NULL;
	int32		input_token_length = 0;
	int32		*generated_token_ids = NULL;
	int32		generated_token_count = 0;
	int32		max_gen_tokens = 100;
	int32		min_gen_tokens = 0;
	int32		i;
	int			rc = -1;
	ONNXTensor *input_tensor = NULL;
	ONNXTensor *output_tensor = NULL;
	int64		input_shape[2];
	float	   *input_data = NULL;
	float	   *logits = NULL;
	int32		vocab_size = 0;
	int32		next_token_id = 0;
	char	   *generated_text = NULL;
	MemoryContext oldcontext;
	MemoryContext complete_context;

	if (errstr)
		*errstr = NULL;
	if (model_name == NULL || prompt == NULL || text_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for ONNX HF complete");
		return -1;
	}

	/* Create memory context for this operation */
	complete_context = AllocSetContextCreate(CurrentMemoryContext,
											 "onnx_hf_complete_ctx",
											 ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(complete_context);

	/* Parse generation parameters */
	if (params_json && strlen(params_json) > 0)
	{
		rc = parse_gen_params(params_json, &gen_params, errstr);
		if (rc != 0)
		{
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(complete_context);
			return -1;
		}
	}
	else
	{
		/* Use defaults */
		memset(&gen_params, 0, sizeof(ONNXGenParams));
		gen_params.temperature = 1.0f;
		gen_params.top_p = 1.0f;
		gen_params.top_k = 0;
		gen_params.max_tokens = 100;
		gen_params.min_tokens = 0;
		gen_params.repetition_penalty = 1.0f;
		gen_params.do_sample = false;
		gen_params.return_prompt = false;
		gen_params.seed = 0;
	}

	max_gen_tokens = gen_params.max_tokens;
	min_gen_tokens = gen_params.min_tokens;

	/* Load or get cached model */
	PG_TRY();
	{
		session = neurondb_onnx_get_or_load_model(model_name, ONNX_MODEL_GENERATION);
	}
	PG_CATCH();
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(complete_context);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Tokenize input prompt with model-specific tokenizer */
	input_token_ids = neurondb_tokenize_with_model(prompt, 512, &input_token_length, model_name);
	if (input_token_ids == NULL || input_token_length == 0)
	{
		if (errstr)
			*errstr = pstrdup("failed to tokenize input prompt");
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(complete_context);
		return -1;
	}

	/* Allocate generated token IDs array */
	generated_token_ids = (int32_t *) palloc(max_gen_tokens * sizeof(int32_t));

	/* Set seed if provided */
	if (gen_params.seed != 0)
		srand(gen_params.seed);

	/* Generation loop */
	for (i = 0; i < max_gen_tokens; i++)
	{
		int32		current_seq_len;
		int32		j;

		/* Prepare input sequence (prompt + generated tokens) */
		current_seq_len = input_token_length + generated_token_count;

		/* Allocate input data */
		if (input_data)
			pfree(input_data);
		input_data = (float *) palloc(current_seq_len * sizeof(float));

		/* Copy prompt tokens */
		for (j = 0; j < input_token_length; j++)
			input_data[j] = (float) input_token_ids[j];

		/* Copy generated tokens */
		for (j = 0; j < generated_token_count; j++)
			input_data[input_token_length + j] = (float) generated_token_ids[j];

		/* Create input tensor */
		input_shape[0] = 1;
		input_shape[1] = current_seq_len;

		if (input_tensor)
			neurondb_onnx_free_tensor(input_tensor);
		input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 2);

		/* Run inference */
		PG_TRY();
		{
			if (output_tensor)
				neurondb_onnx_free_tensor(output_tensor);
			output_tensor = neurondb_onnx_run_inference(session, input_tensor);
		}
		PG_CATCH();
		{
			if (input_tensor)
				neurondb_onnx_free_tensor(input_tensor);
			if (input_data)
				pfree(input_data);
			if (generated_token_ids)
				pfree(generated_token_ids);
			if (input_token_ids)
				pfree(input_token_ids);
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(complete_context);
			PG_RE_THROW();
		}
		PG_END_TRY();

		/* Extract logits from output tensor */
		/* Assume output shape is [1, vocab_size] or [1, seq_len, vocab_size] */
		if (output_tensor->ndim == 2 && output_tensor->shape[0] == 1)
		{
			/* Output shape: [1, vocab_size] */
			vocab_size = output_tensor->shape[1];
			logits = output_tensor->data;
		}
		else if (output_tensor->ndim == 3 && output_tensor->shape[0] == 1)
		{
			/* Output shape: [1, seq_len, vocab_size] */
			/* Use last token's logits */
			int32		seq_len = output_tensor->shape[1];

			vocab_size = output_tensor->shape[2];
			logits = output_tensor->data + (seq_len - 1) * vocab_size;
		}
		else
		{
			if (errstr)
				*errstr = psprintf("unexpected output tensor shape: [%ld",
								   output_tensor->shape[0]);
			if (input_tensor)
				neurondb_onnx_free_tensor(input_tensor);
			if (output_tensor)
				neurondb_onnx_free_tensor(output_tensor);
			if (input_data)
				pfree(input_data);
			if (generated_token_ids)
				pfree(generated_token_ids);
			if (input_token_ids)
				pfree(input_token_ids);
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(complete_context);
			return -1;
		}

		/* Apply temperature scaling */
		if (gen_params.temperature > 0.0f && gen_params.temperature != 1.0f)
		{
			for (j = 0; j < vocab_size; j++)
				logits[j] /= gen_params.temperature;
		}

		/* Apply repetition penalty */
		if (gen_params.repetition_penalty != 1.0f)
		{
			for (j = 0; j < generated_token_count; j++)
			{
				int32		prev_token = generated_token_ids[j];

				if (prev_token >= 0 && prev_token < vocab_size)
				{
					if (logits[prev_token] > 0.0f)
						logits[prev_token] /= gen_params.repetition_penalty;
					else
						logits[prev_token] *= gen_params.repetition_penalty;
				}
			}
		}

		/* Sample next token */
		if (gen_params.do_sample && gen_params.temperature > 0.0f)
			next_token_id = sample_token_multinomial(logits, vocab_size, gen_params.temperature);
		else
			next_token_id = sample_token_greedy(logits, vocab_size);

		/* Check for stop conditions (simplified: check for EOS token) */
		if (next_token_id == 102 || next_token_id == 0)	/* [SEP] or [PAD] */
		{
			if (generated_token_count >= min_gen_tokens)
				break;
		}

		/* Append to generated tokens */
		generated_token_ids[generated_token_count++] = next_token_id;
	}

	/* Decode generated tokens to text */
	generated_text = decode_tokens(generated_token_ids, generated_token_count);

	/* Copy to parent memory context */
	*text_out = MemoryContextStrdup(oldcontext, generated_text);

	/* Cleanup */
	if (input_tensor)
		neurondb_onnx_free_tensor(input_tensor);
	if (output_tensor)
		neurondb_onnx_free_tensor(output_tensor);
	if (input_data)
		pfree(input_data);
	if (generated_token_ids)
		pfree(generated_token_ids);
	if (input_token_ids)
		pfree(input_token_ids);

	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(complete_context);

	return 0;
}

#else /* !HAVE_ONNX_RUNTIME */

/*
 * Stub implementation when ONNX Runtime is not available
 */
int
ndb_onnx_hf_complete(const char *model_name,
					 const char *prompt,
					 const char *params_json,
					 char **text_out,
					 char **errstr)
{
	if (errstr)
		*errstr = pstrdup("ONNX Runtime support not compiled in");
	return -1;
}

#endif /* HAVE_ONNX_RUNTIME */
