/*-------------------------------------------------------------------------
 *
 * neurondb_safe_memory.h
 *    Safe memory management utilities for NeuronDB crash prevention
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_safe_memory.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_SAFE_MEMORY_H
#define NEURONDB_SAFE_MEMORY_H

#include "postgres.h"
#include "utils/memutils.h"
#include <stdarg.h>

/* Safe pfree wrapper */
extern bool ndb_safe_pfree(void *ptr);
extern int ndb_safe_pfree_multi(int count, ...);
extern void ndb_safe_pfree_array(void **array, int count);
extern void ndb_safe_pfree_string_array(char **array, int count);

/* Memory context validation */
extern bool ndb_memory_context_validate(MemoryContext context);
extern bool ndb_ensure_memory_context(MemoryContext context);
extern void ndb_safe_context_cleanup(MemoryContext context, MemoryContext oldcontext);

/* Cleanup helpers */
extern void ndb_cleanup_on_error(MemoryContext oldcontext,
								  MemoryContext callcontext,
								  bool finish_spi,
								  int n_ptrs,
								  ...);

#ifdef NDB_DEBUG_MEMORY
extern void ndb_track_allocation(void *ptr, const char *alloc_func);
extern void ndb_untrack_allocation(void *ptr);
#endif

#endif	/* NEURONDB_SAFE_MEMORY_H */

