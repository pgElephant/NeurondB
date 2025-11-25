/*-------------------------------------------------------------------------
 *
 * neurondb_compat.h
 *		PostgreSQL version compatibility macros for NeurondB
 *
 * This header provides compatibility macros for differences between
 * PostgreSQL versions 16, 17, and 18, including format specifiers
 * for int64/uint64 types.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  include/neurondb_compat.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_COMPAT_H
#define NEURONDB_COMPAT_H

#include "postgres.h"

/*
 * Format specifiers for int64/uint64 types
 *
 * PostgreSQL 16/17: int64 is 'long', uint64 is 'unsigned long'
 * PostgreSQL 18+:   int64 is 'long long', uint64 is 'unsigned long long'
 *
 * Use NDB_INT64_FMT and NDB_UINT64_FMT in format strings instead of
 * hardcoded %lld or %lu to ensure compatibility across all versions.
 */
#if PG_VERSION_NUM >= 180000
/* PostgreSQL 18+: Use long long format specifiers */
#define NDB_INT64_FMT "%lld"
#define NDB_UINT64_FMT "%llu"
#else
/* PostgreSQL 16/17: Use long format specifiers */
#define NDB_INT64_FMT "%ld"
#define NDB_UINT64_FMT "%lu"
#endif

/*
 * Cast macros for consistent type handling
 */
#if PG_VERSION_NUM >= 180000
#define NDB_INT64_CAST(x) ((long long)(x))
#define NDB_UINT64_CAST(x) ((unsigned long long)(x))
#else
#define NDB_INT64_CAST(x) ((long)(x))
#define NDB_UINT64_CAST(x) ((unsigned long)(x))
#endif

/*
 * Compatibility declarations for PostgreSQL 14
 * These symbols may not be exported but are referenced by the linker
 */
#if PG_VERSION_NUM < 150000
#include "libpq/pqsignal.h"
#include "port.h"

/* pqsignal_be may be an alias or macro for pqsignal in some contexts */
#ifndef pqsignal_be
/* Define pqsignal_be as an alias to pqsignal if not already defined */
#define pqsignal_be(signo, func) pqsignal(signo, func)
#endif

/* shmem_request_hook type and variable for PG14 compatibility */
typedef void (*shmem_request_hook_type)(void);
/* Declare as weak symbol to avoid linker errors if not used */
__attribute__((weak)) extern shmem_request_hook_type shmem_request_hook;

#endif /* PG_VERSION_NUM < 150000 */

#endif /* NEURONDB_COMPAT_H */
