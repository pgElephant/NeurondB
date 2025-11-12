#include "postgres.h"

#include "utils/builtins.h"
#include <ctype.h>
#include <math.h>
#include <string.h>

#include "neurondb_llm.h"
#include "neurondb_gpu.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_hf.h"
#endif

#ifdef HAVE_ONNX_RUNTIME
#include "neurondb_onnx.h"
#endif

static bool provider_is(const char *provider, const char *name);
static int fallback_complete(const NdbLLMConfig *cfg,
			     const NdbLLMCallOptions *opts,
			     const char *reason,
			     const char *prompt,
			     const char *params_json,
			     NdbLLMResp *out);
static int fallback_embed(const NdbLLMConfig *cfg,
			  const NdbLLMCallOptions *opts,
			  const char *reason,
			  const char *text,
			  float **vec_out,
			  int *dim_out);
static int fallback_rerank(const NdbLLMConfig *cfg,
			   const NdbLLMCallOptions *opts,
			   const char *reason,
			   const char *query,
			   const char **docs,
			   int ndocs,
			   float **scores_out);

static bool
provider_is(const char *provider, const char *name)
{
	if (provider == NULL)
		return false;
	return pg_strcasecmp(provider, name) == 0;
}

static int
fallback_complete(const NdbLLMConfig *cfg,
		  const NdbLLMCallOptions *opts,
		  const char *reason,
		  const char *prompt,
		  const char *params_json,
		  NdbLLMResp *out)
{
	if (opts != NULL && opts->require_gpu)
	{
		ereport(ERROR,
			(errmsg("neurondb: LLM provider \"%s\" requires GPU but is unavailable locally (%s)",
				cfg->provider ? cfg->provider : "huggingface-local",
				reason ? reason : "not supported")));
		return NDB_LLM_ROUTE_ERROR;
	}
	if (opts != NULL && opts->fail_open)
	{
		NdbLLMConfig fallback = *cfg;

		fallback.provider = "huggingface";
		ereport(WARNING,
			(errmsg("neurondb: LLM provider \"%s\" unavailable locally (%s); falling back to remote Hugging Face",
				cfg->provider ? cfg->provider : "huggingface-local",
				reason ? reason : "not supported")));
		return ndb_hf_complete(&fallback, prompt, params_json, out);
	}

	ereport(ERROR,
		(errmsg("neurondb: LLM provider \"%s\" unavailable locally (%s)",
			cfg->provider ? cfg->provider : "huggingface-local",
			reason ? reason : "not supported")));
	return NDB_LLM_ROUTE_ERROR; /* unreachable */
}

static int
fallback_embed(const NdbLLMConfig *cfg,
	       const NdbLLMCallOptions *opts,
	       const char *reason,
	       const char *text,
	       float **vec_out,
	       int *dim_out)
{
	if (opts != NULL && opts->require_gpu)
	{
		ereport(ERROR,
			(errmsg("neurondb: LLM provider \"%s\" requires GPU but is unavailable locally (%s)",
				cfg->provider ? cfg->provider : "huggingface-local",
				reason ? reason : "not supported")));
		return NDB_LLM_ROUTE_ERROR;
	}
	if (opts != NULL && opts->fail_open)
	{
		NdbLLMConfig fallback = *cfg;

		fallback.provider = "huggingface";
		ereport(WARNING,
			(errmsg("neurondb: LLM provider \"%s\" unavailable locally (%s); falling back to remote Hugging Face",
				cfg->provider ? cfg->provider : "huggingface-local",
				reason ? reason : "not supported")));
		return ndb_hf_embed(&fallback, text, vec_out, dim_out);
	}

	ereport(ERROR,
		(errmsg("neurondb: LLM provider \"%s\" unavailable locally (%s)",
			cfg->provider ? cfg->provider : "huggingface-local",
			reason ? reason : "not supported")));
	return NDB_LLM_ROUTE_ERROR; /* unreachable */
}

static int
fallback_rerank(const NdbLLMConfig *cfg,
		const NdbLLMCallOptions *opts,
		const char *reason,
		const char *query,
		const char **docs,
		int ndocs,
		float **scores_out)
{
	if (opts != NULL && opts->require_gpu)
	{
		ereport(ERROR,
			(errmsg("neurondb: LLM provider \"%s\" requires GPU but is unavailable locally (%s)",
				cfg->provider ? cfg->provider : "huggingface-local",
				reason ? reason : "not supported")));
		return NDB_LLM_ROUTE_ERROR;
	}
	if (opts != NULL && opts->fail_open)
	{
		NdbLLMConfig fallback = *cfg;

		fallback.provider = "huggingface";
		ereport(WARNING,
			(errmsg("neurondb: LLM provider \"%s\" unavailable locally (%s); falling back to remote Hugging Face",
				cfg->provider ? cfg->provider : "huggingface-local",
				reason ? reason : "not supported")));
		return ndb_hf_rerank(&fallback, query, docs, ndocs, scores_out);
	}

	ereport(ERROR,
		(errmsg("neurondb: LLM provider \"%s\" unavailable locally (%s)",
			cfg->provider ? cfg->provider : "huggingface-local",
			reason ? reason : "not supported")));
	return NDB_LLM_ROUTE_ERROR; /* unreachable */
}

int
ndb_llm_route_complete(const NdbLLMConfig *cfg,
		       const NdbLLMCallOptions *opts,
		       const char *prompt,
		       const char *params_json,
		       NdbLLMResp *out)
{
	if (cfg == NULL || prompt == NULL || out == NULL)
		return NDB_LLM_ROUTE_ERROR;

	if (cfg->provider == NULL ||
	    provider_is(cfg->provider, "huggingface") ||
	    provider_is(cfg->provider, "hf-http"))
		return ndb_hf_complete(cfg, prompt, params_json, out);

	if (provider_is(cfg->provider, "huggingface-local") ||
	    provider_is(cfg->provider, "hf-local"))
	{
		/* Try GPU-accelerated inference first if GPU is available and preferred */
		if (neurondb_gpu_is_available() &&
		    (opts == NULL || opts->prefer_gpu || opts->require_gpu))
		{
			char *gpu_text = NULL;
			char *gpu_err = NULL;
			int rc;

			rc = neurondb_gpu_hf_complete(cfg->model, prompt, params_json,
						      &gpu_text, &gpu_err);
			if (rc == 0 && gpu_text != NULL)
			{
				int32 token_length;
				int32 *token_ids;

				/* Count input tokens */
				token_ids = neurondb_tokenize_with_model(prompt, 2048, &token_length, cfg->model);
				if (token_ids && token_length > 0)
					out->tokens_in = token_length;
				else
					out->tokens_in = 0;	/* Fallback: estimate from word count */
				if (token_ids)
					pfree(token_ids);

				/* Count output tokens using tokenizer when available */
#ifdef HAVE_ONNX_RUNTIME
				PG_TRY();
				{
					int32 output_token_length;
					int32 *output_token_ids = neurondb_tokenize_with_model(gpu_text, 2048, &output_token_length, cfg->model);
					if (output_token_ids && output_token_length > 0)
					{
						out->tokens_out = output_token_length;
					}
					else
					{
						/* Fallback: estimate from word count */
						if (gpu_text && strlen(gpu_text) > 0)
						{
							const char *ptr = gpu_text;
							int word_count = 0;
							int in_word = 0;

							while (*ptr)
							{
								if (!isspace((unsigned char) *ptr))
								{
									if (!in_word)
									{
										word_count++;
										in_word = 1;
									}
								}
								else
								{
									in_word = 0;
								}
								ptr++;
							}
							out->tokens_out = word_count > 0 ? word_count : 1;
						}
						else
						{
							out->tokens_out = 0;
						}
					}
					if (output_token_ids)
						pfree(output_token_ids);
				}
				PG_CATCH();
				{
					/* On error, use word count fallback */
					EmitErrorReport();
					FlushErrorState();

					if (gpu_text && strlen(gpu_text) > 0)
					{
						const char *ptr = gpu_text;
						int word_count = 0;
						int in_word = 0;

						while (*ptr)
						{
							if (!isspace((unsigned char) *ptr))
							{
								if (!in_word)
								{
									word_count++;
									in_word = 1;
								}
							}
							else
							{
								in_word = 0;
							}
							ptr++;
						}
						out->tokens_out = word_count > 0 ? word_count : 1;
					}
					else
					{
						out->tokens_out = 0;
					}
				}
				PG_END_TRY();
#else
				/* ONNX runtime not available, use word count fallback */
				if (gpu_text && strlen(gpu_text) > 0)
				{
					const char *ptr = gpu_text;
					int word_count = 0;
					int in_word = 0;

					while (*ptr)
					{
						if (!isspace((unsigned char) *ptr))
						{
							if (!in_word)
							{
								word_count++;
								in_word = 1;
							}
						}
						else
						{
							in_word = 0;
						}
						ptr++;
					}
					out->tokens_out = word_count > 0 ? word_count : 1;
				}
				else
				{
					out->tokens_out = 0;
				}
#endif

				out->text = gpu_text;
				out->json = NULL;
				out->http_status = 200;
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_SUCCESS;
			}
			if (opts != NULL && opts->require_gpu)
			{
				if (gpu_err)
					ereport(ERROR,
						(errmsg("neurondb: GPU HF completion failed: %s",
							gpu_err)));
				ereport(ERROR,
					(errmsg("neurondb: GPU HF completion failed")));
				if (gpu_text)
					pfree(gpu_text);
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_ERROR;
			}
			if (gpu_text)
				pfree(gpu_text);
			if (gpu_err)
				pfree(gpu_err);
		}
#ifdef HAVE_ONNX_RUNTIME
		if (!neurondb_onnx_available())
			return fallback_complete(cfg, opts,
						 "ONNX runtime not available",
						 prompt, params_json, out);
		{
			char *onnx_text = NULL;
			char *onnx_err = NULL;
			int rc;

			rc = ndb_onnx_hf_complete(cfg->model, prompt, params_json,
						  &onnx_text, &onnx_err);
			if (rc == 0 && onnx_text != NULL)
			{
				int32 token_length;
				int32 *token_ids;

				/* Count input tokens */
				token_ids = neurondb_tokenize_with_model(prompt, 2048, &token_length, cfg->model);
				if (token_ids && token_length > 0)
					out->tokens_in = token_length;
				else
					out->tokens_in = 0;	/* Fallback: estimate from word count */
				if (token_ids)
					pfree(token_ids);

				/* Count output tokens using tokenizer when available */
#ifdef HAVE_ONNX_RUNTIME
				PG_TRY();
				{
					int32 output_token_length;
					int32 *output_token_ids = neurondb_tokenize_with_model(onnx_text, 2048, &output_token_length, cfg->model);
					if (output_token_ids && output_token_length > 0)
					{
						out->tokens_out = output_token_length;
					}
					else
					{
						/* Fallback: estimate from word count */
						if (onnx_text && strlen(onnx_text) > 0)
						{
							const char *ptr = onnx_text;
							int word_count = 0;
							int in_word = 0;

							while (*ptr)
							{
								if (!isspace((unsigned char) *ptr))
								{
									if (!in_word)
									{
										word_count++;
										in_word = 1;
									}
								}
								else
								{
									in_word = 0;
								}
								ptr++;
							}
							out->tokens_out = word_count > 0 ? word_count : 1;
						}
						else
						{
							out->tokens_out = 0;
						}
					}
					if (output_token_ids)
						pfree(output_token_ids);
				}
				PG_CATCH();
				{
					/* On error, use word count fallback */
					EmitErrorReport();
					FlushErrorState();

					if (onnx_text && strlen(onnx_text) > 0)
					{
						const char *ptr = onnx_text;
						int word_count = 0;
						int in_word = 0;

						while (*ptr)
						{
							if (!isspace((unsigned char) *ptr))
							{
								if (!in_word)
								{
									word_count++;
									in_word = 1;
								}
							}
							else
							{
								in_word = 0;
							}
							ptr++;
						}
						out->tokens_out = word_count > 0 ? word_count : 1;
					}
					else
					{
						out->tokens_out = 0;
					}
				}
				PG_END_TRY();
#else
				/* ONNX runtime not available, use word count fallback */
				if (onnx_text && strlen(onnx_text) > 0)
				{
					const char *ptr = onnx_text;
					int word_count = 0;
					int in_word = 0;

					while (*ptr)
					{
						if (!isspace((unsigned char) *ptr))
						{
							if (!in_word)
							{
								word_count++;
								in_word = 1;
							}
						}
						else
						{
							in_word = 0;
						}
						ptr++;
					}
					out->tokens_out = word_count > 0 ? word_count : 1;
				}
				else
				{
					out->tokens_out = 0;
				}
#endif

				out->text = onnx_text;
				out->json = NULL;
				out->http_status = 200;
				if (onnx_err)
					pfree(onnx_err);
				return NDB_LLM_ROUTE_SUCCESS;
			}
			if (opts != NULL && opts->require_gpu)
			{
				if (onnx_err)
					ereport(ERROR,
						(errmsg("neurondb: ONNX HF completion failed: %s",
							onnx_err)));
				ereport(ERROR,
					(errmsg("neurondb: ONNX HF completion failed")));
				if (onnx_text)
					pfree(onnx_text);
				if (onnx_err)
					pfree(onnx_err);
				return NDB_LLM_ROUTE_ERROR;
			}
			if (onnx_text)
				pfree(onnx_text);
			if (onnx_err)
				pfree(onnx_err);
			return fallback_complete(cfg, opts,
						 "ONNX completion failed",
						 prompt, params_json, out);
		}
#else
		return fallback_complete(cfg, opts,
					 "compiled without ONNX runtime",
					 prompt, params_json, out);
#endif
	}

	ereport(WARNING,
		(errmsg("neurondb: unknown LLM provider \"%s\", using remote Hugging Face",
			cfg->provider)));
	return ndb_hf_complete(cfg, prompt, params_json, out);
}

int
ndb_llm_route_embed(const NdbLLMConfig *cfg,
		    const NdbLLMCallOptions *opts,
		    const char *text,
		    float **vec_out,
		    int *dim_out)
{
	if (cfg == NULL || text == NULL || vec_out == NULL || dim_out == NULL)
		return NDB_LLM_ROUTE_ERROR;

	if (cfg->provider == NULL ||
	    provider_is(cfg->provider, "huggingface") ||
	    provider_is(cfg->provider, "hf-http"))
		return ndb_hf_embed(cfg, text, vec_out, dim_out);

	if (provider_is(cfg->provider, "huggingface-local") ||
	    provider_is(cfg->provider, "hf-local"))
	{
		/* Try GPU-accelerated inference first if GPU is available and preferred */
		if (neurondb_gpu_is_available() &&
		    (opts == NULL || opts->prefer_gpu || opts->require_gpu))
		{
			float *gpu_vec = NULL;
			int gpu_dim = 0;
			char *gpu_err = NULL;
			int rc;

			rc = neurondb_gpu_hf_embed(cfg->model, text, &gpu_vec, &gpu_dim, &gpu_err);
			if (rc == 0 && gpu_vec != NULL && gpu_dim > 0)
			{
				*vec_out = gpu_vec;
				*dim_out = gpu_dim;
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_SUCCESS;
			}
			if (opts != NULL && opts->require_gpu)
			{
				if (gpu_err)
					ereport(ERROR,
						(errmsg("neurondb: GPU HF embedding failed: %s",
							gpu_err)));
				ereport(ERROR,
					(errmsg("neurondb: GPU HF embedding failed")));
				if (gpu_vec)
					pfree(gpu_vec);
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_ERROR;
			}
			if (gpu_vec)
				pfree(gpu_vec);
			if (gpu_err)
				pfree(gpu_err);
		}
#ifdef HAVE_ONNX_RUNTIME
		if (!neurondb_onnx_available())
			return fallback_embed(cfg, opts,
					      "ONNX runtime not available",
					      text, vec_out, dim_out);
		/* Use ONNX embedding implementation */
		{
			ONNXModelSession *session;
			int32 *token_ids = NULL;
			int32 token_length;
			ONNXTensor *input_tensor = NULL;
			ONNXTensor *output_tensor = NULL;
			float *input_data = NULL;
			int i;
			int64 input_shape[2];
			int rc = NDB_LLM_ROUTE_ERROR;

			/* Validate input */
			if (!text || strlen(text) == 0)
				return fallback_embed(cfg, opts,
						      "empty input text",
						      text, vec_out, dim_out);

			/* Load or get cached model */
			PG_TRY();
			{
				session = neurondb_onnx_get_or_load_model(cfg->model, ONNX_MODEL_EMBEDDING);
				if (!session)
				{
					rc = fallback_embed(cfg, opts,
							    "failed to load ONNX model",
							    text, vec_out, dim_out);
				}
				else
				{
					/* Tokenize input text with model-specific tokenizer */
					token_ids = neurondb_tokenize_with_model(text, 128, &token_length, cfg->model);
					if (!token_ids)
					{
						rc = fallback_embed(cfg, opts,
								    "tokenization failed: returned NULL",
								    text, vec_out, dim_out);
					}
					else if (token_length <= 0)
					{
						pfree(token_ids);
						token_ids = NULL;
						rc = fallback_embed(cfg, opts,
								    "tokenization failed: empty result",
								    text, vec_out, dim_out);
					}
					else
					{
						/* Convert token IDs to float array for ONNX */
						input_data = (float *) palloc(token_length * sizeof(float));
						for (i = 0; i < token_length; i++)
							input_data[i] = (float) token_ids[i];

						/* Create input tensor */
						input_shape[0] = 1;
						input_shape[1] = token_length;
						input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 2);
						/* Note: neurondb_onnx_create_tensor uses Assert() and palloc(),
						 * so it will either succeed or throw an error. No NULL check needed. */

						/* Run inference (may throw PostgreSQL error) */
						output_tensor = neurondb_onnx_run_inference(session, input_tensor);
						if (!output_tensor || output_tensor->size <= 0)
						{
							if (input_tensor)
								neurondb_onnx_free_tensor(input_tensor);
							if (input_data)
								pfree(input_data);
							if (token_ids)
								pfree(token_ids);
							rc = fallback_embed(cfg, opts,
									    "ONNX inference failed: invalid output",
									    text, vec_out, dim_out);
						}
						else
						{
							/* Extract embedding vector */
							*dim_out = output_tensor->size;
							*vec_out = (float *) palloc(*dim_out * sizeof(float));
							memcpy(*vec_out, output_tensor->data, *dim_out * sizeof(float));

							/* Normalize embedding (L2 normalization) */
							{
								float sum = 0.0f;
								for (i = 0; i < *dim_out; i++)
									sum += (*vec_out)[i] * (*vec_out)[i];
								sum = sqrtf(sum);
								if (sum > 0.0f)
								{
									for (i = 0; i < *dim_out; i++)
										(*vec_out)[i] /= sum;
								}
							}

							/* Cleanup */
							if (input_tensor)
								neurondb_onnx_free_tensor(input_tensor);
							if (output_tensor)
								neurondb_onnx_free_tensor(output_tensor);
							if (input_data)
								pfree(input_data);
							if (token_ids)
								pfree(token_ids);

							rc = NDB_LLM_ROUTE_SUCCESS;
						}
					}
				}
			}
			PG_CATCH();
			{
				/* Cleanup on error */
				EmitErrorReport();
				FlushErrorState();

				if (input_tensor)
					neurondb_onnx_free_tensor(input_tensor);
				if (output_tensor)
					neurondb_onnx_free_tensor(output_tensor);
				if (input_data)
					pfree(input_data);
				if (token_ids)
					pfree(token_ids);
				if (*vec_out)
				{
					pfree(*vec_out);
					*vec_out = NULL;
					*dim_out = 0;
				}

				/* Fall back to HTTP or error */
				rc = fallback_embed(cfg, opts,
						    "ONNX inference error",
						    text, vec_out, dim_out);
			}
			PG_END_TRY();

			return rc;
		}
#else
		return fallback_embed(cfg, opts,
				      "compiled without ONNX runtime",
				      text, vec_out, dim_out);
#endif
	}

	ereport(WARNING,
		(errmsg("neurondb: unknown LLM provider \"%s\", using remote Hugging Face",
			cfg->provider)));
	return ndb_hf_embed(cfg, text, vec_out, dim_out);
}

int
ndb_llm_route_rerank(const NdbLLMConfig *cfg,
		     const NdbLLMCallOptions *opts,
		     const char *query,
		     const char **docs,
		     int ndocs,
		     float **scores_out)
{
	if (cfg == NULL || query == NULL || docs == NULL || scores_out == NULL)
		return NDB_LLM_ROUTE_ERROR;

	if (cfg->provider == NULL ||
	    provider_is(cfg->provider, "huggingface") ||
	    provider_is(cfg->provider, "hf-http"))
		return ndb_hf_rerank(cfg, query, docs, ndocs, scores_out);

	if (provider_is(cfg->provider, "huggingface-local") ||
	    provider_is(cfg->provider, "hf-local"))
	{
		/* Try GPU-accelerated inference first if GPU is available and preferred */
		if (neurondb_gpu_is_available() &&
		    (opts == NULL || opts->prefer_gpu || opts->require_gpu))
		{
			float *gpu_scores = NULL;
			char *gpu_err = NULL;
			int rc;

			rc = neurondb_gpu_hf_rerank(cfg->model, query, docs, ndocs,
						    &gpu_scores, &gpu_err);
			if (rc == 0 && gpu_scores != NULL)
			{
				*scores_out = gpu_scores;
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_SUCCESS;
			}
			if (opts != NULL && opts->require_gpu)
			{
				if (gpu_err)
					ereport(ERROR,
						(errmsg("neurondb: GPU HF reranking failed: %s",
							gpu_err)));
				ereport(ERROR,
					(errmsg("neurondb: GPU HF reranking failed")));
				if (gpu_scores)
					pfree(gpu_scores);
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_ERROR;
			}
			if (gpu_scores)
				pfree(gpu_scores);
			if (gpu_err)
				pfree(gpu_err);
		}
#ifdef HAVE_ONNX_RUNTIME
		if (!neurondb_onnx_available())
			return fallback_rerank(cfg, opts,
					       "ONNX runtime not available",
					       query, docs, ndocs, scores_out);
		/* ONNX reranking not yet implemented - fall back to HTTP */
		return fallback_rerank(cfg, opts,
				       "ONNX reranking not yet implemented - use HTTP or GPU",
				       query, docs, ndocs, scores_out);
#else
		return fallback_rerank(cfg, opts,
				       "compiled without ONNX runtime",
				       query, docs, ndocs, scores_out);
#endif
	}

	ereport(WARNING,
		(errmsg("neurondb: unknown LLM provider \"%s\", using remote Hugging Face",
			cfg->provider)));
	return ndb_hf_rerank(cfg, query, docs, ndocs, scores_out);
}

/*
 * ndb_llm_route_complete_batch
 *	  Route batch completion requests to appropriate backend (GPU, ONNX, HTTP)
 */
int
ndb_llm_route_complete_batch(const NdbLLMConfig *cfg,
			     const NdbLLMCallOptions *opts,
			     const char **prompts,
			     int num_prompts,
			     const char *params_json,
			     NdbLLMBatchResp *out)
{
	int i;
	int num_success = 0;

	if (cfg == NULL || prompts == NULL || out == NULL || num_prompts <= 0)
		return NDB_LLM_ROUTE_ERROR;

	/* Initialize output */
	out->num_items = num_prompts;
	out->num_success = 0;
	out->texts = (char **) palloc0(num_prompts * sizeof(char *));
	out->tokens_in = (int *) palloc0(num_prompts * sizeof(int));
	out->tokens_out = (int *) palloc0(num_prompts * sizeof(int));
	out->http_status = (int *) palloc0(num_prompts * sizeof(int));

	if (provider_is(cfg->provider, "huggingface-local") ||
	    provider_is(cfg->provider, "hf-local"))
	{
		/* Try GPU-accelerated batch inference first if GPU is available */
		if (neurondb_gpu_is_available() &&
		    (opts == NULL || opts->prefer_gpu || opts->require_gpu))
		{
#ifdef NDB_GPU_CUDA
			NdbCudaHfBatchResult *batch_results;
			int rc;
			char *gpu_err = NULL;

			batch_results = (NdbCudaHfBatchResult *) palloc0(num_prompts * sizeof(NdbCudaHfBatchResult));
			rc = neurondb_gpu_hf_complete_batch(cfg->model, prompts, num_prompts,
							    params_json, batch_results, &gpu_err);
			if (rc == 0)
			{
				/* Copy results to output */
				for (i = 0; i < num_prompts; i++)
				{
					if (batch_results[i].status == 0 && batch_results[i].text)
					{
						/* Count input tokens */
						int32 token_length;
						int32 *token_ids = neurondb_tokenize_with_model(prompts[i], 2048, &token_length, cfg->model);
						if (token_ids && token_length > 0)
							out->tokens_in[i] = token_length;
						else
							out->tokens_in[i] = 0;	/* Fallback: estimate from word count */
						if (token_ids)
							pfree(token_ids);

						out->texts[i] = batch_results[i].text;
						out->tokens_out[i] = batch_results[i].num_tokens;
						out->http_status[i] = 200;
						num_success++;
					}
					else
					{
						out->texts[i] = NULL;
						out->tokens_in[i] = 0;
						out->tokens_out[i] = 0;
						out->http_status[i] = 500;
						if (batch_results[i].error)
							pfree(batch_results[i].error);
					}
				}
				out->num_success = num_success;
				if (gpu_err)
					pfree(gpu_err);
				pfree(batch_results);
				return NDB_LLM_ROUTE_SUCCESS;
			}
			if (opts != NULL && opts->require_gpu)
			{
				if (gpu_err)
					ereport(ERROR,
						(errmsg("neurondb: GPU HF batch completion failed: %s",
							gpu_err)));
				ereport(ERROR,
					(errmsg("neurondb: GPU HF batch completion failed")));
				if (gpu_err)
					pfree(gpu_err);
				pfree(batch_results);
				return NDB_LLM_ROUTE_ERROR;
			}
			if (gpu_err)
				pfree(gpu_err);
			pfree(batch_results);
#endif
		}
		/* Fall back to sequential processing or ONNX */
		/* For now, process sequentially */
		for (i = 0; i < num_prompts; i++)
		{
			NdbLLMResp resp;
			int rc;

			memset(&resp, 0, sizeof(NdbLLMResp));
			rc = ndb_llm_route_complete(cfg, opts, prompts[i], params_json, &resp);
			if (rc == 0 && resp.text)
			{
				out->texts[i] = resp.text;
				out->tokens_in[i] = resp.tokens_in;
				out->tokens_out[i] = resp.tokens_out;
				out->http_status[i] = resp.http_status;
				num_success++;
			}
			else
			{
				out->texts[i] = NULL;
				out->tokens_in[i] = 0;
				out->tokens_out[i] = 0;
				out->http_status[i] = 500;
			}
		}
		out->num_success = num_success;
		return (num_success > 0) ? NDB_LLM_ROUTE_SUCCESS : NDB_LLM_ROUTE_ERROR;
	}

	/* For remote providers, process sequentially */
	for (i = 0; i < num_prompts; i++)
	{
		NdbLLMResp resp;
		int rc;

		memset(&resp, 0, sizeof(NdbLLMResp));
		rc = ndb_hf_complete(cfg, prompts[i], params_json, &resp);
		if (rc == 0 && resp.text)
		{
			out->texts[i] = resp.text;
			out->tokens_in[i] = resp.tokens_in;
			out->tokens_out[i] = resp.tokens_out;
			out->http_status[i] = resp.http_status;
			num_success++;
		}
		else
		{
			out->texts[i] = NULL;
			out->tokens_in[i] = 0;
			out->tokens_out[i] = 0;
			out->http_status[i] = 500;
		}
	}
	out->num_success = num_success;
	return (num_success > 0) ? NDB_LLM_ROUTE_SUCCESS : NDB_LLM_ROUTE_ERROR;
}

/*
 * ndb_llm_route_rerank_batch
 *	  Route batch reranking requests to appropriate backend (GPU, ONNX, HTTP)
 */
int
ndb_llm_route_rerank_batch(const NdbLLMConfig *cfg,
			   const NdbLLMCallOptions *opts,
			   const char **queries,
			   const char ***docs_array,
			   int *ndocs_array,
			   int num_queries,
			   float ***scores_out,
			   int **nscores_out)
{
	int i;
	int num_success = 0;

	if (cfg == NULL || queries == NULL || docs_array == NULL || 
	    ndocs_array == NULL || scores_out == NULL || nscores_out == NULL || 
	    num_queries <= 0)
		return NDB_LLM_ROUTE_ERROR;

	/* Initialize output */
	*scores_out = (float **) palloc0(num_queries * sizeof(float *));
	*nscores_out = (int *) palloc0(num_queries * sizeof(int));

	/* For now, process sequentially */
	for (i = 0; i < num_queries; i++)
	{
		float *scores = NULL;
		int rc;

		rc = ndb_llm_route_rerank(cfg, opts, queries[i], docs_array[i],
					  ndocs_array[i], &scores);
		if (rc == 0 && scores != NULL)
		{
			(*scores_out)[i] = scores;
			(*nscores_out)[i] = ndocs_array[i];
			num_success++;
		}
		else
		{
			(*scores_out)[i] = NULL;
			(*nscores_out)[i] = 0;
		}
	}

	return (num_success > 0) ? NDB_LLM_ROUTE_SUCCESS : NDB_LLM_ROUTE_ERROR;
}

