/*-------------------------------------------------------------------------
 *
 * ml_text.c
 *    Text ML for NeuronDB
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_text.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/memutils.h"

#include <ctype.h>
#include <string.h>

/* PG_MODULE_MAGIC is in neurondb.c only */

PG_FUNCTION_INFO_V1(neurondb_text_classify);
PG_FUNCTION_INFO_V1(neurondb_sentiment_analysis);
PG_FUNCTION_INFO_V1(neurondb_named_entity_recognition);
PG_FUNCTION_INFO_V1(neurondb_text_summarize);

#define MAX_TOKENS          4096
#define MAX_TOKEN_LEN       128
#define MAX_CATEGORY        128
#define MAX_CATEGORIES      16
#define MAX_ENTITIES        256
#define MAX_ENTITY_TYPE     32
#define MAX_ENTITY_LEN      128
#define MAX_SENTIMENT       12
#define MAX_SUMMARY         4096
#define MAX_SENTENCES       512

typedef struct ClassifyResult
{
    char    category[MAX_CATEGORY];
    float4  confidence;
} ClassifyResult;

typedef struct NERResult
{
    char    entity[MAX_ENTITY_LEN];
    char    entity_type[MAX_ENTITY_TYPE];
    float4  confidence;
    int32   entity_position;
} NERResult;

typedef struct SentimentResult
{
    float4  positive;
    float4  negative;
    float4  neutral;
    char    sentiment[MAX_SENTIMENT];
} SentimentResult;

/*
 * Helper: Tokenize input into lowercase word tokens only, using PostgreSQL memory context allocations.
 */
static void
simple_tokenize(const char *input, char **tokens, int *num_tokens)
{
    int     i = 0;
    int     input_len = (int) strlen(input);
    int     t = 0;

    while (i < input_len && t < MAX_TOKENS)
    {
        /* Skip non-alphanumeric */
        while (i < input_len && !isalnum((unsigned char) input[i]))
            i++;
        if (i >= input_len)
            break;

        char wordbuf[MAX_TOKEN_LEN];
        int j = 0;

        memset(wordbuf, 0, sizeof(wordbuf));
        while (i < input_len && isalnum((unsigned char) input[i]) && j < MAX_TOKEN_LEN - 1)
        {
            wordbuf[j++] = (char) tolower((unsigned char) input[i]);
            i++;
        }
        wordbuf[j] = '\0';
        tokens[t++] = pstrdup(wordbuf);
    }
    *num_tokens = t;
}

/*
 * Text Classification (Bag-of-words + SPI model table support)
 *
 * Args:
 *    INT4   model_id
 *    TEXT   input
 * Returns SETOF (category text, confidence float4)
 */
Datum
neurondb_text_classify(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    int32           model_id = PG_GETARG_INT32(0);
    text           *input_text = PG_GETARG_TEXT_PP(1);

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext        oldcontext;
        char               *input_str;
        char               *tokens[MAX_TOKENS];
        int                 num_tokens = 0;
        ClassifyResult     *results = NULL;
        int                 n_categories = 0;
        int                 ret;
        char                qry[256];
        char              **categories;
        int                *category_counts;
        int                 i, t, r;

        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        if (model_id <= 0)
            ereport(ERROR,
                    (errmsg("model_id must be positive integer")));

        input_str = text_to_cstring(input_text);
        simple_tokenize(input_str, tokens, &num_tokens);

        snprintf(qry, sizeof(qry),
                "SELECT c.category, w.word "
                "FROM neurondb_textclass_words w "
                "JOIN neurondb_textclass_categories c ON (w.cat_id = c.id) "
                "WHERE w.model_id = %d", model_id);

        if ((ret = SPI_connect()) != SPI_OK_CONNECT)
            ereport(ERROR,
                    (errmsg("SPI_connect failed: %d", ret)));

        ret = SPI_exec(qry, 0);
        if (ret != SPI_OK_SELECT)
        {
            SPI_finish();
            ereport(ERROR,
                    (errmsg("Could not fetch model word lists")));
        }

        /* Prepare category lists */
        categories = (char **) palloc0(sizeof(char *) * MAX_CATEGORIES);
        category_counts = (int *) palloc0(sizeof(int) * MAX_CATEGORIES);

        /* Fill category names (de-duplication) */
        for (r = 0; r < (int)SPI_processed; r++)
        {
            HeapTuple   tuple = SPI_tuptable->vals[r];
            TupleDesc   tupdesc = SPI_tuptable->tupdesc;
            char       *cat = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 1, NULL));
            bool        found = false;
            for (i = 0; i < n_categories; i++)
            {
                if (strcmp(categories[i], cat) == 0)
                {
                    found = true;
                    break;
                }
            }
            if (!found && n_categories < MAX_CATEGORIES)
            {
                categories[n_categories++] = pstrdup(cat);
            }
            pfree(cat);
        }

        if (n_categories == 0)
        {
            SPI_finish();
            for (t = 0; t < num_tokens; t++) pfree(tokens[t]);
            ereport(ERROR,
                    (errmsg("No categories found for model_id %d", model_id)));
        }

        /* Tally up word matches for each category */
        for (r = 0; r < (int)SPI_processed; r++)
        {
            HeapTuple   tuple = SPI_tuptable->vals[r];
            TupleDesc   tupdesc = SPI_tuptable->tupdesc;
            char       *cat = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 1, NULL));
            char       *word = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 2, NULL));

            for (t = 0; t < num_tokens; t++)
            {
                if (strcmp(tokens[t], word) == 0)
                {
                    for (i = 0; i < n_categories; i++)
                    {
                        if (strcmp(categories[i], cat) == 0)
                        {
                            category_counts[i]++;
                            break;
                        }
                    }
                }
            }
            pfree(cat);
            pfree(word);
        }

        for (t = 0; t < num_tokens; t++)
            pfree(tokens[t]);

        SPI_finish();

        /* Calculate confidences */
        results = (ClassifyResult *) palloc0(n_categories * sizeof(ClassifyResult));
        int total_count = 0;

        for (i = 0; i < n_categories; i++)
            total_count += category_counts[i];

        for (i = 0; i < n_categories; i++)
        {
            strlcpy(results[i].category, categories[i], MAX_CATEGORY);
            if (total_count)
                results[i].confidence = ((float4) category_counts[i]) / total_count;
            else
                results[i].confidence = (1.0f / (float4) n_categories);
            pfree(categories[i]);
        }
        pfree(categories);
        pfree(category_counts);

        funcctx->user_fctx = results;
        funcctx->max_calls = n_categories;
        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    {
        ClassifyResult *results = (ClassifyResult *) funcctx->user_fctx;

        if (funcctx->call_cntr < funcctx->max_calls)
        {
            Datum     values[2];
            bool      nulls[2] = {false, false};
            HeapTuple tuple;

            values[0] = CStringGetTextDatum(results[funcctx->call_cntr].category);
            values[1] = Float4GetDatum(results[funcctx->call_cntr].confidence);

            if (funcctx->tuple_desc == NULL)
            {
                TupleDesc desc = CreateTemplateTupleDesc(2);
                TupleDescInitEntry(desc, (AttrNumber) 1, "category", TEXTOID, -1, 0);
                TupleDescInitEntry(desc, (AttrNumber) 2, "confidence", FLOAT4OID, -1, 0);
                funcctx->tuple_desc = BlessTupleDesc(desc);
            }
            tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
            SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
        }
        else
        {
            if (funcctx->max_calls > 0 && funcctx->user_fctx)
                pfree(results);
            SRF_RETURN_DONE(funcctx);
        }
    }
}

/* Sentiment analysis based on VADER-like lexicon (from SPI table) */
Datum
neurondb_sentiment_analysis(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    text           *input_text = PG_GETARG_TEXT_PP(0);
    text           *model_text = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext        oldcontext;
        char               *input_str;
        char               *model_name = NULL;
        char               *tokens[MAX_TOKENS];
        int                 num_tokens = 0;
        int                 pos = 0, neg = 0, neu = 0;
        SentimentResult    *result;
        int                 ret;
        char                qry[256];
        int                 t, r;

        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        input_str = text_to_cstring(input_text);

        if (model_text)
            model_name = text_to_cstring(model_text);
        else
            model_name = pstrdup("vader");

        simple_tokenize(input_str, tokens, &num_tokens);

        snprintf(qry, sizeof(qry),
                 "SELECT word, polarity FROM neurondb_sentiment_lexicon WHERE model = '%s'",
                 model_name);

        if ((ret = SPI_connect()) != SPI_OK_CONNECT)
        {
            for (t = 0; t < num_tokens; t++) pfree(tokens[t]);
            pfree(model_name);
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }
        ret = SPI_exec(qry, 0);
        if (ret != SPI_OK_SELECT)
        {
            SPI_finish();
            for (t = 0; t < num_tokens; t++) pfree(tokens[t]);
            pfree(model_name);
            ereport(ERROR, (errmsg("Could not execute sentiment lexicon fetch")));
        }

        for (t = 0; t < num_tokens; t++)
        {
            bool found = false;
            for (r = 0; r < (int)SPI_processed; r++)
            {
                HeapTuple   tuple = SPI_tuptable->vals[r];
                TupleDesc   tupdesc = SPI_tuptable->tupdesc;
                char       *w = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 1, NULL));
                char       *pol = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 2, NULL));
                if (strcmp(tokens[t], w) == 0)
                {
                    found = true;
                    if (strcmp(pol, "positive") == 0)
                        pos++;
                    else if (strcmp(pol, "negative") == 0)
                        neg++;
                    else
                        neu++; /* treat unknown as neutral */
                }
                pfree(w);
                pfree(pol);
                if (found)
                    break;
            }
            if (!found)
                neu++;
            pfree(tokens[t]);
        }
        pfree(model_name);

        SPI_finish();

        if (num_tokens == 0)
            num_tokens = 1;

        result = (SentimentResult *) palloc0(sizeof(SentimentResult));
        result->positive = ((float4) pos) / num_tokens;
        result->negative = ((float4) neg) / num_tokens;
        result->neutral  = ((float4) neu) / num_tokens;

        if (result->positive >= result->negative &&
            result->positive >= result->neutral)
        {
            strlcpy(result->sentiment, "positive", MAX_SENTIMENT);
        }
        else if (result->negative >= result->positive &&
                 result->negative >= result->neutral)
        {
            strlcpy(result->sentiment, "negative", MAX_SENTIMENT);
        }
        else
        {
            strlcpy(result->sentiment, "neutral", MAX_SENTIMENT);
        }

        funcctx->user_fctx = result;
        funcctx->max_calls = 1;
        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    {
        SentimentResult *result = (SentimentResult *) funcctx->user_fctx;
        if (funcctx->call_cntr < funcctx->max_calls)
        {
            Datum     values[4];
            bool      nulls[4] = {false, false, false, false};
            HeapTuple tuple;

            values[0] = CStringGetTextDatum(result->sentiment);
            values[1] = Float4GetDatum(result->positive);
            values[2] = Float4GetDatum(result->negative);
            values[3] = Float4GetDatum(result->neutral);

            if (funcctx->tuple_desc == NULL)
            {
                TupleDesc desc = CreateTemplateTupleDesc(4);
                TupleDescInitEntry(desc, (AttrNumber) 1, "sentiment", TEXTOID, -1, 0);
                TupleDescInitEntry(desc, (AttrNumber) 2, "positive", FLOAT4OID, -1, 0);
                TupleDescInitEntry(desc, (AttrNumber) 3, "negative", FLOAT4OID, -1, 0);
                TupleDescInitEntry(desc, (AttrNumber) 4, "neutral", FLOAT4OID, -1, 0);
                funcctx->tuple_desc = BlessTupleDesc(desc);
            }
            tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
            pfree(result);
            SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
        }
        else
        {
            SRF_RETURN_DONE(funcctx);
        }
    }
}

/*
 * Named Entity Recognition using SPI table rules
 * (Any entity in neurondb_ner_entities with matching tokens)
 */
Datum
neurondb_named_entity_recognition(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    text           *input_text = PG_GETARG_TEXT_PP(0);
    ArrayType      *entity_types_array = PG_ARGISNULL(1) ? NULL : PG_GETARG_ARRAYTYPE_P(1);

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext        oldcontext;
        char               *input_str;
        char               *tokens[MAX_TOKENS];
        int                 num_tokens = 0;
        NERResult          *entities;
        int                 n_entities = 0;
        int                 ret, t, r;

        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        input_str = text_to_cstring(input_text);
        simple_tokenize(input_str, tokens, &num_tokens);

        if ((ret = SPI_connect()) != SPI_OK_CONNECT)
        {
            for (t = 0; t < num_tokens; t++) pfree(tokens[t]);
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }
        ret = SPI_exec("SELECT entity, entity_type, confidence FROM neurondb_ner_entities", 0);
        if (ret != SPI_OK_SELECT)
        {
            SPI_finish();
            for (t = 0; t < num_tokens; t++) pfree(tokens[t]);
            ereport(ERROR, (errmsg("NER entity table fetch failed")));
        }

        entities = (NERResult *) palloc0(MAX_ENTITIES * sizeof(NERResult));

        for (t = 0; t < num_tokens && n_entities < MAX_ENTITIES; t++)
        {
            for (r = 0; r < (int)SPI_processed && n_entities < MAX_ENTITIES; r++)
            {
                HeapTuple   tuple = SPI_tuptable->vals[r];
                TupleDesc   tupdesc = SPI_tuptable->tupdesc;
                char       *dbent = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 1, NULL));
                char       *type = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 2, NULL));
                float4      conf = DatumGetFloat4(SPI_getbinval(tuple, tupdesc, 3, NULL));
                if (strcmp(tokens[t], dbent) == 0)
                {
                    strlcpy(entities[n_entities].entity, dbent, MAX_ENTITY_LEN);
                    strlcpy(entities[n_entities].entity_type, type, MAX_ENTITY_TYPE);
                    entities[n_entities].confidence = conf;
                    entities[n_entities].entity_position = t + 1;
                    n_entities++;
                }
                pfree(dbent);
                pfree(type);
            }
            pfree(tokens[t]);
        }
        SPI_finish();

        /* Optional: filter by entity_type list arg */
        if (entity_types_array != NULL && n_entities > 0)
        {
            Datum      *datum_array;
            bool       *nulls;
            int         n_types;
            int         k = 0, e, i;
            NERResult  *filtered;

            deconstruct_array(entity_types_array, TEXTOID, -1, false, 'i', &datum_array, &nulls, &n_types);
            filtered = (NERResult *) palloc0(MAX_ENTITIES * sizeof(NERResult));
            for (e = 0; e < n_entities; e++)
            {
                bool keep = false;
                for (i = 0; i < n_types; i++)
                {
                    char *etype = TextDatumGetCString(datum_array[i]);
                    if (pg_strcasecmp(etype, entities[e].entity_type) == 0)
                    {
                        keep = true;
                    }
                    pfree(etype);
                    if (keep)
                        break;
                }
                if (keep && k < MAX_ENTITIES)
                {
                    filtered[k++] = entities[e];
                }
            }
            pfree(entities);
            entities = filtered;
            n_entities = k;
        }

        funcctx->user_fctx = entities;
        funcctx->max_calls = n_entities;
        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    {
        NERResult *entities = (NERResult *) funcctx->user_fctx;
        if (funcctx->call_cntr < funcctx->max_calls)
        {
            Datum     values[4];
            bool      nulls[4] = {false, false, false, false};
            HeapTuple tuple;

            values[0] = CStringGetTextDatum(entities[funcctx->call_cntr].entity);
            values[1] = CStringGetTextDatum(entities[funcctx->call_cntr].entity_type);
            values[2] = Float4GetDatum(entities[funcctx->call_cntr].confidence);
            values[3] = Int32GetDatum(entities[funcctx->call_cntr].entity_position);

            if (funcctx->tuple_desc == NULL)
            {
                TupleDesc desc = CreateTemplateTupleDesc(4);
                TupleDescInitEntry(desc, (AttrNumber) 1, "entity", TEXTOID, -1, 0);
                TupleDescInitEntry(desc, (AttrNumber) 2, "entity_type", TEXTOID, -1, 0);
                TupleDescInitEntry(desc, (AttrNumber) 3, "confidence", FLOAT4OID, -1, 0);
                TupleDescInitEntry(desc, (AttrNumber) 4, "entity_position", INT4OID, -1, 0);
                funcctx->tuple_desc = BlessTupleDesc(desc);
            }
            tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
            SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
        }
        else
        {
            if (funcctx->max_calls > 0 && funcctx->user_fctx)
                pfree(entities);
            SRF_RETURN_DONE(funcctx);
        }
    }
}

/*
 * Text Summarization (extractive: top sentences by non-stopword count, uses SPI stopwords)
 * Args:
 *    text input
 *    int4 max_length (optional, default 128)
 *    text method (optional, default "extractive")
 */
Datum
neurondb_text_summarize(PG_FUNCTION_ARGS)
{
    text   *input_text = PG_GETARG_TEXT_PP(0);
    int32   max_length = PG_ARGISNULL(1) ? 128 : PG_GETARG_INT32(1);
    text   *method_text = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);
    char   *text_str;
    char   *method;
    int     len;
    char    summary[MAX_SUMMARY];
    int     i;

    text_str = text_to_cstring(input_text);
    len = (int) strlen(text_str);

    if (method_text)
        method = text_to_cstring(method_text);
    else
        method = pstrdup("extractive");

    memset(summary, 0, sizeof(summary));

    if (pg_strcasecmp(method, "extractive") == 0)
    {
        /* Split text to sentences (by '.', '?', '!') */
        int         sstart = 0, send = 0, slen;
        char       *sentence_ptrs[MAX_SENTENCES];
        int         scores[MAX_SENTENCES];
        int         n_sentences = 0, written = 0;
        int         used[MAX_SENTENCES];
        int         ret, s;

        memset(used, 0, sizeof(used));
        while (send < len && n_sentences < MAX_SENTENCES)
        {
            sstart = send;
            while (send < len && text_str[send] != '.' && text_str[send] != '?' && text_str[send] != '!')
                send++;
            if (send < len)
                send++;
            slen = send - sstart;
            if (slen > 0 && n_sentences < MAX_SENTENCES)
            {
                sentence_ptrs[n_sentences] = pnstrdup(text_str + sstart, slen);
                scores[n_sentences] = 0;
                n_sentences++;
            }
            while (send < len && isspace((unsigned char) text_str[send]))
                send++;
        }

        if ((ret = SPI_connect()) != SPI_OK_CONNECT)
        {
            for (s = 0; s < n_sentences; s++)
                pfree(sentence_ptrs[s]);
            pfree(method);
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }
        ret = SPI_exec("SELECT stopword FROM neurondb_summarizer_stopwords", 0);
        if (ret != SPI_OK_SELECT)
        {
            SPI_finish();
            for (s = 0; s < n_sentences; s++)
                pfree(sentence_ptrs[s]);
            pfree(method);
            ereport(ERROR, (errmsg("could not query stopword table")));
        }
        int n_stopwords = SPI_processed;
        char **stopwords = (char **) palloc0(n_stopwords * sizeof(char *));
        for (i = 0; i < n_stopwords; i++)
        {
            HeapTuple tup = SPI_tuptable->vals[i];
            TupleDesc desc = SPI_tuptable->tupdesc;
            stopwords[i] = TextDatumGetCString(SPI_getbinval(tup, desc, 1, NULL));
        }

        /* Score: count of non-stopword tokens per sentence */
        for (s = 0; s < n_sentences; s++)
        {
            char   *sentence = sentence_ptrs[s];
            char   *stoks[128];
            int     stok_ct = 0, tokidx;
            simple_tokenize(sentence, stoks, &stok_ct);
            for (tokidx = 0; tokidx < stok_ct; tokidx++)
            {
                bool is_stop = false;
                int sw;
                for (sw = 0; sw < n_stopwords; sw++)
                {
                    if (strcmp(stoks[tokidx], stopwords[sw]) == 0)
                    {
                        is_stop = true;
                        break;
                    }
                }
                if (!is_stop)
                    scores[s]++;
                pfree(stoks[tokidx]);
            }
        }

        /* Assemble top sentences into summary */
        written = 0;
        while (written < max_length - 1)
        {
            int     maxscore = -1, maxi = -1;
            for (s = 0; s < n_sentences; s++)
            {
                if (!used[s] && scores[s] > maxscore)
                {
                    maxscore = scores[s];
                    maxi = s;
                }
            }
            if (maxi == -1 || maxscore == 0)
                break;
            int sl = (int) strlen(sentence_ptrs[maxi]);
            int tocopy = (sl > max_length - 1 - written) ? (max_length - 1 - written) : sl;
            if (tocopy > 0)
            {
                memcpy(summary + written, sentence_ptrs[maxi], tocopy);
                written += tocopy;
                if (written < max_length - 1)
                {
                    summary[written] = ' ';
                    written++;
                }
            }
            used[maxi] = 1;
            if (written >= max_length - 1)
                break;
        }
        if (written > 0)
            summary[written - 1] = '\0';
        else
            summary[0] = '\0';

        for (i = 0; i < n_sentences; i++)
            pfree(sentence_ptrs[i]);
        for (i = 0; i < n_stopwords; i++)
            pfree(stopwords[i]);
        pfree(stopwords);
        SPI_finish();
    }
    else
    {
        /* Abstractive: copy first max_length-8 bytes, append marker */
        int j = 0;
        for (i = 0; i < len && j < max_length - 8; i++)
        {
            summary[j++] = text_str[i];
        }
        summary[j] = '\0';
        if (j > 0 && summary[j - 1] == ' ')
            summary[j - 1] = '\0';
        strlcat(summary, " [abs]", sizeof(summary));
    }
    pfree(method);
    PG_RETURN_TEXT_P(cstring_to_text(summary));
}
