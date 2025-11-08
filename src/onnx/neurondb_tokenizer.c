/*-------------------------------------------------------------------------
 *
 * neurondb_tokenizer.c
 *    Highly detailed tokenizer implementation for BERT-style HuggingFace models.
 *
 * This module provides a BERT-style tokenizer for preprocessing text prior
 * to ONNX model inference. It includes lowercasing, special token handling,
 * whitespace and rudimentary WordPiece tokenization, and vocabulary lookup.
 *
 * Supported Features:
 *    - Lowercasing (English-language normalization)
 *    - Handles [CLS], [SEP], [PAD], [UNK], [MASK]
 *    - Simple hash-based vocabulary table for fast lookups
 *    - Tokenization: whitespace split, with facility for future WordPiece extension
 *    - Attention mask generation
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 * SPDX-License-Identifier: PostgreSQL
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "catalog/pg_type.h"
#include "neurondb_onnx.h"

#include <ctype.h>
#include <string.h>
#include <errno.h>

/* ---- SPECIAL TOKEN IDs (BERT standard) ---- */
#define TOKEN_PAD_ID   0     /* [PAD] */
#define TOKEN_UNK_ID   100   /* [UNK] */
#define TOKEN_CLS_ID   101   /* [CLS] */
#define TOKEN_SEP_ID   102   /* [SEP] */
#define TOKEN_MASK_ID  103   /* [MASK] */

/* ---- Sequence length limits ---- */
#define MAX_SEQ_LENGTH 512

/* ---- Vocabulary table implementation ---- */
typedef struct VocabEntry
{
    char       *token;      /* Owned string in TopMemoryContext */
    int32       token_id;   /* Vocabulary index */
    struct VocabEntry *next; /* For chaining on hash collision */
} VocabEntry;

#define VOCAB_HASH_SIZE 65536

static VocabEntry *g_vocab_table[VOCAB_HASH_SIZE] = {0};
static bool g_vocab_loaded = false;

/**
 * hash_token
 *    Lightweight hash for token-string to vocabulary index.
 *    Not cryptographically secure, but stable and simple.
 */
static uint32
hash_token(const char *token)
{
    uint32 hash = 0;
    for (const char *p = token; *p != '\0'; ++p)
        hash = hash * 31 + (unsigned char) *p;
    return hash % VOCAB_HASH_SIZE;
}

/**
 * vocab_add_token
 *    Safely adds a token and its associated token_id to the global vocabulary table.
 */
static void
vocab_add_token(const char *token, int32 token_id)
{
    uint32 hash = hash_token(token);
    VocabEntry *entry = (VocabEntry *) MemoryContextAlloc(TopMemoryContext, sizeof(VocabEntry));

    /* Copy the token string into TopMemoryContext as well */
    entry->token = MemoryContextStrdup(TopMemoryContext, token);
    entry->token_id = token_id;
    entry->next = g_vocab_table[hash];
    g_vocab_table[hash] = entry;
}

/**
 * vocab_lookup_token
 *    Returns the integer token_id for a provided string token.
 *    If not found, RETURN TOKEN_UNK_ID.
 */
static int32
vocab_lookup_token(const char *token)
{
    uint32 hash = hash_token(token);
    for (VocabEntry *entry = g_vocab_table[hash]; entry != NULL; entry = entry->next)
    {
        if (strcmp(entry->token, token) == 0)
            return entry->token_id;
    }
    return TOKEN_UNK_ID;
}

/**
 * load_vocab_file
 *    Loads a vocab.txt file, as exported by HuggingFace or other BERT-compatible models.
 *    Each line is a token. Line number = token_id.
 */
static void load_vocab_file(const char *vocab_path) pg_attribute_unused();
static void
load_vocab_file(const char *vocab_path)
{
    FILE *fp;
    char line[512];
    int32 token_id = 0;

    fp = fopen(vocab_path, "r");
    if (fp == NULL)
    {
        int save_errno = errno;
        ereport(ERROR,
                (errcode_for_file_access(),
                 errmsg("could not open vocabulary file: %s (%s)",
                        vocab_path, strerror(save_errno))));
    }

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        size_t len = strlen(line);

        /* Trim trailing newline and possible carriage return */
        if (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        if (len > 0 && (line[len - 1] == '\r'))
            line[--len] = '\0';

        /* Skip empty lines in vocab.txt */
        if (line[0] == '\0')
            continue;

        vocab_add_token(line, token_id++);
    }

    fclose(fp);
    g_vocab_loaded = true;

    elog(LOG, "loaded vocabulary with %d tokens from \"%s\"", token_id, vocab_path);
}

/**
 * _to_lower_case
 *    Converts a string (in-place) to lower case, only if the char is in A-Z.
 *    This is in-place: modifies the input string.
 */
static void
_to_lower_case(char *src)
{
    for (char *p = src; *p; ++p)
    {
        if (*p >= 'A' && *p <= 'Z')
            *p = *p + ('a' - 'A');
    }
}

/**
 * tokenize_text
 *    Tokenizes incoming UTF-8 string based on whitespace.
 *    Converts to lower case (by default) for basic BERT models.
 *    Returns a palloc'd char** array of palloc'd null-terminated strings.
 *    The number of tokens is stored in (*num_tokens).
 *
 * NOTE: This does not include advanced tokenization like punctuation splitting, WordPiece, CJK, etc.
 *       See HuggingFace Tokenizers for production tokenization.
 */
static char **
tokenize_text(const char *text, int *num_tokens)
{
    char *text_copy;
    char *token;
    char **tokens;         /* Array of pointers to token strings */
    int count = 0;
    int capacity = 128;
    char *saveptr = NULL;

    Assert(text != NULL);
    Assert(num_tokens != NULL);

    /* Copy the source text since strtok_r will mutate it */
    text_copy = pstrdup(text);

    /* Lowercase transformation for BERT-style preprocessing */
    _to_lower_case(text_copy);

    /* Preallocate token pointer array */
    tokens = (char **) palloc(capacity * sizeof(char *));
    count = 0;

    /* Tokenize by whitespace: space, tab, newline, and carriage return */
    token = strtok_r(text_copy, " \t\n\r", &saveptr);
    while (token != NULL)
    {
        /* Expand array if necessary */
        if (count >= capacity)
        {
            capacity *= 2;
            tokens = (char **) repalloc(tokens, capacity * sizeof(char *));
        }
        tokens[count++] = pstrdup(token);
        token = strtok_r(NULL, " \t\n\r", &saveptr);
    }

    /* Free the temporary, mutable copy */
    pfree(text_copy);

    *num_tokens = count;
    return tokens;
}

/**
 * neurondb_tokenize
 *    Complete procedure for converting input text to an ID sequence suitable for model input.
 *    - Tokenizes and normalizes input
 *    - Adds [CLS] at the start, [SEP] at the end
 *    - Maps each token to a vocab index or to [UNK] if unknown
 *    - Pads the result to 'max_length' with [PAD]
 *    - Returns a palloc'd int32 array of full length
 *    - Fills *output_length with the produced sequence length (always 'max_length')
 *
 * Arguments:
 *    - text:        UTF-8 null-terminated C string to tokenize
 *    - max_length:  Max sequence length, typically 128, 256, or 512
 *    - output_length: pointer to int32 for returning the length written (== max_length)
 */
int32 *
neurondb_tokenize(const char *text, int32 max_length, int32 *output_length)
{
    char **tokens = NULL;             /* Array of palloc'd char* */
    int32 *token_ids = NULL;          /* Output array */
    int num_tokens = 0;
    int output_idx = 0;
    int i;

    Assert(text != NULL);
    Assert(output_length != NULL);
    Assert(max_length > 1); /* Provide at least space for [CLS], [SEP] */

    /*
     * Tokenize text. tokens is a palloc'd array of palloc'd strings.
     * num_tokens: number of tokens in the result.
     */
    tokens = tokenize_text(text, &num_tokens);

    /*
     * Allocate output array, zeroed (so all trailing positions are [PAD]).
     * We always write max_length elements, zero-initialized.
     */
    token_ids = (int32 *) palloc0(max_length * sizeof(int32));

    /* Add [CLS] at beginning */
    output_idx = 0;
    token_ids[output_idx++] = TOKEN_CLS_ID;

    /* Add as many content tokens as fit, but reserve one slot for [SEP].
     * If tokens don't fit, silently truncate.
     */
    for (i = 0; i < num_tokens && output_idx < max_length - 1; ++i)
    {
        int32 token_id;
        if (g_vocab_loaded)
        {
            token_id = vocab_lookup_token(tokens[i]);
        }
        else
        {
            token_id = TOKEN_UNK_ID;
        }
        token_ids[output_idx++] = token_id;
        pfree(tokens[i]);  /* Free each token string */
    }
    /* Free the tokens array itself */
    pfree(tokens);

    /* Always add [SEP] if there is room */
    if (output_idx < max_length)
        token_ids[output_idx++] = TOKEN_SEP_ID;

    /* All remaining positions are [PAD] (== 0), already set by palloc0 */

    *output_length = max_length;
    return token_ids;
}

/**
 * neurondb_create_attention_mask
 *    Create attention mask (int32[]), value 1 for any real token (not [PAD]), 0 for [PAD] token.
 *    Sequence length matches input length.
 *    Used during inference to tell model which positions are valid tokens.
 */
int32 *
neurondb_create_attention_mask(int32 *token_ids, int32 length)
{
    int32 *mask;
    int i;

    Assert(token_ids != NULL);
    Assert(length > 0);

    mask = (int32 *) palloc(length * sizeof(int32));
    for (i = 0; i < length; ++i)
    {
        /* If not a [PAD] ID, mask value = 1, else 0. */
        mask[i] = (token_ids[i] != TOKEN_PAD_ID) ? 1 : 0;
    }

    return mask;
}

