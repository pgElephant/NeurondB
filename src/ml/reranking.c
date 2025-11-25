/*-------------------------------------------------------------------------
 *
 * reranking.c
 *    Advanced reranking functions for semantic search result refinement and ordering.
 *
 * This module provides integration points for a rich set of reranking functions,
 * supporting LLMs, cross-encoders, Cohere API, ColBERT, Learning-to-Rank, and
 * ensemble approaches. It enables precise, reconfigurable, and extensible reranking
 * by leveraging both local and remote AI models, including interfacing with the Hugging Face
 * API, custom LLM endpoints, and future model types.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/ml/reranking.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_llm.h"
#include "neurondb_gpu.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "access/htup_details.h"
#include <string.h>
#include <float.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"

/* NOTE: PG_MODULE_MAGIC should reside only in the main extension file, not here. */

/*-------------------------------------------------------------------------
 * RerankState - Holds state across multi-call SRF invocations.
 *-------------------------------------------------------------------------
 */
typedef struct RerankState
{
	char	   *query;			/* User query string for conditioning reranker */
	Datum	   *candidates;		/* Array of candidate Datum (text) values */
	bool	   *nulls;			/* Which candidates are NULL */
	float	   *scores;			/* Output: reranked scores [0,1], descending
								 * order */
	int		   *indices;		/* Output: indices of reranked elements */
	int			ncandidates;	/* Number of provided candidates */
}			RerankState;

/*-------------------------------------------------------------------------
 * Internal: Utility to perform descending sort of scores/indices in tandem.
 *-------------------------------------------------------------------------
 */
static void
sort_rerank_desc(float *scores, int *indices, int n)
{
	int			i,
				j;

	for (i = 0; i < n - 1; i++)
	{
		for (j = i + 1; j < n; j++)
		{
			if (scores[j] > scores[i])
			{
				float		tmp_s = scores[i];
				int			tmp_i = indices[i];

				scores[i] = scores[j];
				indices[i] = indices[j];
				scores[j] = tmp_s;
				indices[j] = tmp_i;
			}
		}
	}
}

/*-------------------------------------------------------------------------
 * rerank_cross_encoder
 *    Perform cross-encoder reranking given (query, candidate_array, model_name, top_k).
 *    Returns SRF: (idx INT, score FLOAT4)
 *-------------------------------------------------------------------------
 *   query     = user query (TEXT)
 *   candidates = array of candidate TEXT documents
 *   model     = cross-encoder model name/identifier (TEXT, optional)
 *   top_k     = number of results to return (INT)
 *-------------------------------------------------------------------------
 */
PG_FUNCTION_INFO_V1(rerank_cross_encoder);
Datum
rerank_cross_encoder(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	int			call_cntr;
	int			max_calls;
	RerankState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		text	   *query_text;
		ArrayType  *candidates_array;
		text	   *model_text;
		int			top_k;
		char	   *query_str;
		Datum	   *candidate_datums;
		bool	   *candidate_nulls;
		int			ncandidates;
		float	   *scores = NULL;

		/*-- Prepare multi-call context --*/
		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/*--- 1. Parse Inputs ---*/
		query_text = PG_GETARG_TEXT_PP(0);
		candidates_array = PG_GETARG_ARRAYTYPE_P(1);
		model_text = (PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2));
		top_k = PG_GETARG_INT32(3);

		query_str = text_to_cstring(query_text);

		deconstruct_array(candidates_array,
						  TEXTOID,
						  -1,
						  false,
						  'i',
						  &candidate_datums,
						  &candidate_nulls,
						  &ncandidates);

		/* Robustly limit for top_k edge cases */
		if (top_k < 1)
			ereport(ERROR,
					(errmsg("top_k must be positive (got %d)",
							top_k)));
		if (ncandidates <= 0)
			ereport(ERROR,
					(errmsg("candidate array cannot be empty")));
		max_calls = (ncandidates < top_k) ? ncandidates : top_k;

		/*--- 2. Allocate Rerank State ---*/
		state = (RerankState *) palloc0(sizeof(RerankState));
		state->query = query_str;
		state->candidates = candidate_datums;
		state->nulls = candidate_nulls;
		state->scores = (float *) palloc0(ncandidates * sizeof(float));
		state->indices = (int *) palloc0(ncandidates * sizeof(int));
		state->ncandidates = ncandidates;

		/*--- 3. Rerank via API or fallback ---*/
		if (model_text)
		{
			char	   *model_str;
			NdbLLMConfig cfg;
			NdbLLMCallOptions call_opts;
			int			i;
			const char **docs;
			int			api_result;

			model_str = text_to_cstring(model_text);

			cfg.provider = neurondb_llm_provider
				? neurondb_llm_provider
				: "huggingface";
			cfg.endpoint = neurondb_llm_endpoint
				? neurondb_llm_endpoint
				: "https://router.huggingface.co";
			cfg.model = model_str
				? model_str
				: (neurondb_llm_model ? neurondb_llm_model
				   : "sentence-transformers/"
				   "all-MiniLM-L6-v2");
			cfg.api_key = neurondb_llm_api_key;
			cfg.timeout_ms = neurondb_llm_timeout_ms;
			cfg.prefer_gpu = neurondb_gpu_enabled;
			cfg.require_gpu = false;
			if (cfg.provider != NULL
				&& (pg_strcasecmp(
								  cfg.provider, "huggingface-local")
					== 0
					|| pg_strcasecmp(
									 cfg.provider, "hf-local")
					== 0)
				&& !neurondb_llm_fail_open)
				cfg.require_gpu = true;

			call_opts.task = "rerank";
			call_opts.prefer_gpu = cfg.prefer_gpu;
			call_opts.require_gpu = cfg.require_gpu;
			call_opts.fail_open = neurondb_llm_fail_open;

			/* --- Prepare docs array for API call --- */
			docs = (const char **) palloc0(
										   ncandidates * sizeof(char *));
			for (i = 0; i < ncandidates; i++)
			{
				if (!candidate_nulls[i]
					&& DatumGetPointer(candidate_datums[i]))
					docs[i] =
						text_to_cstring(DatumGetTextPP(
													   candidate_datums[i]));
				else
					docs[i] = "";
			}

			/* --- Try remote rerank using external API --- */
			api_result = ndb_llm_route_rerank(&cfg,
											  &call_opts,
											  query_str,
											  docs,
											  ncandidates,
											  &scores);
			if (api_result == 0 && scores)
			{
				memcpy(state->scores,
					   scores,
					   sizeof(float) * ncandidates);
				for (i = 0; i < ncandidates; i++)
					state->indices[i] = i;

				sort_rerank_desc(state->scores,
								 state->indices,
								 ncandidates);
			}
			else
			{
				/*
				 * If Hugging Face (or model) is unavailable, fallback to
				 * sequential dummy scores. This prevents user errors from
				 * causing null results.
				 */
				for (i = 0; i < ncandidates; i++)
				{
					state->indices[i] = i;
					state->scores[i] = 1.0f
						- ((float) i
						   / (float)
						   ncandidates);	/* [1.0, 0.0] linear scores */
				}
			}

			/* --- Free allocated (detached) strings in docs[] --- */
			for (i = 0; i < ncandidates; i++)
			{
				if (docs[i][0] != '\0')
				{
					void	   *ptr = (void *) docs[i];

					ndb_safe_pfree(ptr);
					docs[i] = NULL;
				}
			}
			NDB_SAFE_PFREE_AND_NULL(docs);
			if (scores)
				NDB_SAFE_PFREE_AND_NULL(scores);
			NDB_SAFE_PFREE_AND_NULL(model_str);
		}
		else
		{
			/* No model (default): preserve order, assign perfect scores */
			int			i;

			for (i = 0; i < ncandidates; i++)
			{
				state->indices[i] = i;
				state->scores[i] = 1.0f;
			}
		}

		/*--- 4. Build SRF output tuple descriptor: (idx int, score float4) ---*/
		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 1, "idx", INT4OID, -1, 0);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
		BlessTupleDesc(tupdesc);

		funcctx->max_calls = max_calls;
		funcctx->user_fctx = state;
		funcctx->tuple_desc = tupdesc;

		MemoryContextSwitchTo(oldcontext);
	}

	/*--- Per-call: emit one tuple per result, ranked in descending order ---*/
	funcctx = SRF_PERCALL_SETUP();
	state = (RerankState *) funcctx->user_fctx;
	call_cntr = funcctx->call_cntr;
	max_calls = funcctx->max_calls;

	if (call_cntr < max_calls)
	{
		HeapTuple	tuple;
		Datum		values[2];
		bool		nulls[2] = {false, false};
		int			idx_ranked =
			state->indices
			[call_cntr];		/* sorted index in candidate array */

		values[0] = Int32GetDatum(idx_ranked);
		values[1] = Float4GetDatum(state->scores[idx_ranked]);
		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		/*
		 * Memory context cleanup: All allocations (state, query, arrays) are
		 * in funcctx->multi_call_memory_ctx and will be automatically freed
		 * when the context is destroyed. No explicit cleanup needed here.
		 */
		SRF_RETURN_DONE(funcctx);
	}
}

/*-------------------------------------------------------------------------
 * rerank_llm
 *    Advanced: LLM-completion-based reranking (zero-/few-shot, instruction prompt).
 *    Synthesizes a prompt with query and candidates, invokes LLM completion,
 *    parses response to extract scores/indices, returns reranked results.
 *-------------------------------------------------------------------------
 */
PG_FUNCTION_INFO_V1(rerank_llm);
Datum
rerank_llm(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	int			call_cntr;
	int			max_calls;
	RerankState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		text	   *query_text;
		ArrayType  *candidates_array;
		text	   *model_text;
		text	   *prompt_template_text;
		int			top_k;
		float4		temperature;
		char	   *query_str;
		char	   *model_str = NULL;
		char	   *prompt_template = NULL;
		Datum	   *candidate_datums;
		bool	   *candidate_nulls;
		int			ncandidates;
		int			i;
		StringInfoData prompt;
		StringInfoData params_json;
		char	   *params_json_str;
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;
		NdbLLMResp	resp;
		char	   *llm_response = NULL;
		int			api_result;

		/*-- Prepare multi-call context --*/
		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/*--- 1. Parse Inputs ---*/
		query_text = PG_GETARG_TEXT_PP(0);
		candidates_array = PG_GETARG_ARRAYTYPE_P(1);
		model_text = (PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2));
		top_k = PG_ARGISNULL(3) ? -1 : PG_GETARG_INT32(3);
		prompt_template_text = (PG_ARGISNULL(4) ? NULL : PG_GETARG_TEXT_PP(4));
		temperature = PG_ARGISNULL(5) ? 0.0f : PG_GETARG_FLOAT4(5);

		query_str = text_to_cstring(query_text);

		deconstruct_array(candidates_array,
						  TEXTOID,
						  -1,
						  false,
						  'i',
						  &candidate_datums,
						  &candidate_nulls,
						  &ncandidates);

		if (ncandidates <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("candidate array cannot be empty")));

		max_calls = (top_k < 1 || top_k > ncandidates) ? ncandidates : top_k;

		/*--- 2. Allocate Rerank State ---*/
		state = (RerankState *) palloc0(sizeof(RerankState));
		state->query = query_str;
		state->candidates = candidate_datums;
		state->nulls = candidate_nulls;
		state->scores = (float *) palloc0(ncandidates * sizeof(float));
		state->indices = (int *) palloc0(ncandidates * sizeof(int));
		state->ncandidates = ncandidates;

		/* Initialize indices */
		for (i = 0; i < ncandidates; i++)
			state->indices[i] = i;

		/*--- 3. Build prompt with query and candidates ---*/
		initStringInfo(&prompt);
		if (prompt_template_text)
		{
			prompt_template = text_to_cstring(prompt_template_text);
			appendStringInfo(&prompt, "%s\n\n", prompt_template);
		}
		else
		{
			/* Default prompt template */
			appendStringInfo(&prompt,
							 "Rank the following documents by relevance to the query. "
							 "Return a JSON array of scores (0.0 to 1.0) in the same order.\n\n");
		}

		appendStringInfo(&prompt, "Query: %s\n\nDocuments:\n", query_str);
		for (i = 0; i < ncandidates; i++)
		{
			char	   *doc_str;

			if (!candidate_nulls[i] && DatumGetPointer(candidate_datums[i]))
			{
				doc_str = text_to_cstring(DatumGetTextPP(candidate_datums[i]));
				appendStringInfo(&prompt, "%d. %s\n", i + 1, doc_str);
				NDB_SAFE_PFREE_AND_NULL(doc_str);
			}
			else
			{
				appendStringInfo(&prompt, "%d. [NULL]\n", i + 1);
			}
		}

		appendStringInfo(&prompt,
						 "\nReturn JSON array of scores: [score1, score2, ...]");

		/*--- 4. Configure LLM ---*/
		model_str = model_text ? text_to_cstring(model_text) : NULL;

		cfg.provider = neurondb_llm_provider
			? neurondb_llm_provider
			: "openai";
		cfg.endpoint = neurondb_llm_endpoint
			? neurondb_llm_endpoint
			: "https://api.openai.com/v1";
		cfg.model = model_str
			? model_str
			: (neurondb_llm_model ? neurondb_llm_model : "gpt-3.5-turbo");
		cfg.api_key = neurondb_llm_api_key;
		cfg.timeout_ms = neurondb_llm_timeout_ms;
		cfg.prefer_gpu = neurondb_gpu_enabled;
		cfg.require_gpu = false;

		call_opts.task = "complete";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = false;
		call_opts.fail_open = neurondb_llm_fail_open;

		/* Build params JSON with temperature */
		initStringInfo(&params_json);
		appendStringInfo(&params_json,
						 "{\"temperature\":%.2f}",
						 temperature);
		params_json_str = params_json.data;

		/*--- 5. Call LLM completion API ---*/
		memset(&resp, 0, sizeof(resp));
		api_result = ndb_llm_route_complete(&cfg,
											&call_opts,
											prompt.data,
											params_json_str,
											&resp);

		if (api_result == 0 && resp.text)
		{
			llm_response = resp.text;

			/*--- 6. Parse LLM response (JSON array of scores) ---*/
			/* Simple JSON array parsing - look for [score1, score2, ...] */
			{
				const char *p = llm_response;
				int			score_idx = 0;
				double		score_val;

				/* Find opening bracket */
				while (*p && *p != '[')
					p++;
				if (*p == '[')
					p++;

				/* Parse scores */
				while (*p && score_idx < ncandidates)
				{
					char	   *endptr;

					/* Skip whitespace and commas */
					while (*p && (isspace((unsigned char) *p) || *p == ','))
						p++;
					if (*p == ']' || !*p)
						break;

					errno = 0;
					score_val = strtod(p, &endptr);
					if (endptr != p && errno == 0)
					{
						/* Clamp to [0, 1] */
						if (score_val < 0.0)
							score_val = 0.0;
						if (score_val > 1.0)
							score_val = 1.0;
						state->scores[score_idx] = (float) score_val;
						p = endptr;
					}
					else
					{
						/* Parse error, use default score */
						state->scores[score_idx] = 0.5f;
						break;
					}
					score_idx++;
				}

				/*
				 * Fill remaining with default scores if LLM didn't return
				 * enough
				 */
				for (; score_idx < ncandidates; score_idx++)
					state->scores[score_idx] = 0.5f;
			}
		}
		else
		{
			/* API call failed, use default scores */
			for (i = 0; i < ncandidates; i++)
				state->scores[i] = 0.5f;
		}

		/*--- 7. Sort by score (descending) ---*/
		sort_rerank_desc(state->scores, state->indices, ncandidates);

		/*--- 8. Setup SRF return ---*/
		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "index", INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		funcctx->max_calls = max_calls;
		funcctx->user_fctx = state;

		/* Cleanup */
		if (llm_response && llm_response != resp.text)
			NDB_SAFE_PFREE_AND_NULL(llm_response);
		if (resp.text)
			NDB_SAFE_PFREE_AND_NULL(resp.text);
		if (resp.json)
			NDB_SAFE_PFREE_AND_NULL(resp.json);
		if (params_json_str)
			NDB_SAFE_PFREE_AND_NULL(params_json_str);
		if (prompt.data)
			NDB_SAFE_PFREE_AND_NULL(prompt.data);
		if (model_str)
			NDB_SAFE_PFREE_AND_NULL(model_str);
		if (prompt_template)
			NDB_SAFE_PFREE_AND_NULL(prompt_template);

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (RerankState *) funcctx->user_fctx;
	call_cntr = funcctx->call_cntr;
	max_calls = funcctx->max_calls;

	if (call_cntr < max_calls)
	{
		HeapTuple	tuple;
		Datum		values[2];
		bool		nulls[2] = {false, false};
		int			idx_ranked = state->indices[call_cntr];

		values[0] = Int32GetDatum(idx_ranked);
		values[1] = Float4GetDatum(state->scores[idx_ranked]);
		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

/*-------------------------------------------------------------------------
 * rerank_cohere
 *    Cohere API-style reranking (external API or compatible endpoint, not yet implemented).
 *-------------------------------------------------------------------------
 */
PG_FUNCTION_INFO_V1(rerank_cohere);
Datum
rerank_cohere(PG_FUNCTION_ARGS)
{
	PG_RETURN_NULL();
}

/*-------------------------------------------------------------------------
 * rerank_colbert
 *    ColBERT architecture reranking (not yet implemented).
 *-------------------------------------------------------------------------
 */
PG_FUNCTION_INFO_V1(rerank_colbert);
Datum
rerank_colbert(PG_FUNCTION_ARGS)
{
	PG_RETURN_NULL();
}

/*-------------------------------------------------------------------------
 * rerank_ltr
 *    Learning-to-Rank model reranking (LambdaMART, RankNet, etc., not yet implemented).
 *-------------------------------------------------------------------------
 */
PG_FUNCTION_INFO_V1(rerank_ltr);
Datum
rerank_ltr(PG_FUNCTION_ARGS)
{
	PG_RETURN_NULL();
}

/*-------------------------------------------------------------------------
 * rerank_ensemble
 *    Ensemble reranking across multiple models/strategies (not yet implemented).
 *-------------------------------------------------------------------------
 */
PG_FUNCTION_INFO_V1(rerank_ensemble);
Datum
rerank_ensemble(PG_FUNCTION_ARGS)
{
	PG_RETURN_NULL();
}
