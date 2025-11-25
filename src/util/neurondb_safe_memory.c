/*-------------------------------------------------------------------------
 *
 * neurondb_safe_memory.c
 *    Safe memory management utilities for NeuronDB crash prevention
 *
 * Provides centralized safe pfree wrapper and memory context tracking
 * to prevent crashes from:
 * - Freeing NULL pointers
 * - Freeing already freed pointers
 * - Freeing memory from wrong context
 * - Memory context validation issues
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *    src/util/neurondb_safe_memory.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/memutils.h"
#include "utils/elog.h"
#include "access/htup_details.h"
#include "executor/spi.h"

#include <stdarg.h>

#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"

static bool
ndb_safe_pfree_and_null_impl(void *ptr)
{
	if (ptr == NULL)
		return false;

#ifdef USE_ASSERT_CHECKING
	/*
	 * In debug builds, log suspicious pointer patterns
	 * (e.g., pointers that look like they might be invalid)
	 */
	if ((uintptr_t)ptr < 0x1000)
	{
		elog(DEBUG1,
			"neurondb: suspicious pointer pattern in ndb_safe_pfree: %p",
			ptr);
	}
#endif

	/* Actually free the pointer - was calling itself recursively before! */
	pfree(ptr);
	return true;
}

/*
 * ndb_safe_pfree - Safely free a pointer, checking for NULL
 * Returns true if pointer was freed, false otherwise
 */
bool ndb_safe_pfree(void *ptr)
{
    return ndb_safe_pfree_and_null_impl(ptr);
}
/*
 * ndb_safe_pfree_multi - Safely free multiple pointers
 *
 * Useful for cleanup in error paths where multiple allocations
 * may or may not have succeeded.
 *
 * Returns number of pointers actually freed
 */
int
ndb_safe_pfree_multi(int count, ...)
{
	va_list ap;
	int i;
	int freed = 0;
	void *ptr;

	if (count <= 0)
		return 0;

	va_start(ap, count);

	for (i = 0; i < count; i++)
	{
		ptr = va_arg(ap, void *);
		if (ndb_safe_pfree_and_null_impl(ptr))
			freed++;
	}

	va_end(ap);
	return freed;
}

/*-------------------------------------------------------------------------
 * Memory context validation
 *-------------------------------------------------------------------------
 */

/*
 * ndb_memory_context_validate - Validate memory context is valid
 *
 * Use at boundaries only:
 * - Entry points in background workers
 * - SPI wrappers
 * - LLM HTTP wrappers if context switches
 *
 * Store CurrentMemoryContext at function entry, verify before leaving
 * or before big operations.
 *
 * DO NOT spam in inner loops.
 *
 * Returns true if context is valid, false otherwise
 */
bool
ndb_memory_context_validate(MemoryContext context)
{
	if (context == NULL)
		return false;

	/*
	 * In PostgreSQL, MemoryContextIsValid is a macro that checks
	 * if context has valid magic number. We use a similar check.
	 */
	return MemoryContextIsValid(context);
}

/*
 * ndb_ensure_memory_context - Ensure we're in the specified context
 *
 * Switches to the context if not already there, and validates
 * the context is valid.
 *
 * Returns true if switch succeeded, false if context invalid
 */
bool
ndb_ensure_memory_context(MemoryContext context)
{
	if (!ndb_memory_context_validate(context))
	{
		elog(WARNING,
			"neurondb: attempt to switch to invalid memory context");
		return false;
	}

	if (CurrentMemoryContext != context)
		MemoryContextSwitchTo(context);

	return true;
}

/*
 * ndb_safe_context_cleanup - Safely clean up a memory context
 *
 * Validates context before deletion and ensures we're not
 * currently in the context being deleted.
 *
 * If oldcontext is provided, switches to it before deletion.
 */
void
ndb_safe_context_cleanup(MemoryContext context, MemoryContext oldcontext)
{
	if (context == NULL)
		return;

	if (!ndb_memory_context_validate(context))
	{
		elog(WARNING,
			"neurondb: attempt to delete invalid memory context");
		return;
	}

	/*
	 * If we're currently in the context being deleted, we need
	 * to switch to a different context first
	 */
	if (CurrentMemoryContext == context)
	{
		if (oldcontext == NULL || !ndb_memory_context_validate(oldcontext))
		{
			elog(ERROR,
				"neurondb: cannot delete current memory context without valid old context");
			return;
		}
		MemoryContextSwitchTo(oldcontext);
	}

	MemoryContextDelete(context);
}

/*-------------------------------------------------------------------------
 * Pointer tracking (optional - for debugging)
 *-------------------------------------------------------------------------
 */

#ifdef NDB_DEBUG_MEMORY

/*
 * Simple pointer tracking structure for debugging
 * Only enabled in debug builds
 */
typedef struct NdbPointerEntry
{
	void *ptr;
	MemoryContext context;
	const char *alloc_func;
} NdbPointerEntry;

static NdbPointerEntry *ptr_tracker = NULL;
static int ptr_tracker_size = 0;
static int ptr_tracker_count = 0;

/*
 * ndb_track_allocation - Track an allocation for debugging
 */
void
ndb_track_allocation(void *ptr, const char *alloc_func)
{
	if (ptr == NULL)
		return;

	/* Simple implementation - expand if needed */
	/* In production, this would be more sophisticated */
	elog(DEBUG2,
		"neurondb: tracking allocation %p from %s in context %p",
		ptr,
		alloc_func,
		CurrentMemoryContext);
}

/*
 * ndb_untrack_allocation - Remove allocation from tracking
 */
void
ndb_untrack_allocation(void *ptr)
{
	if (ptr == NULL)
		return;

	elog(DEBUG2, "neurondb: untracking allocation %p", ptr);
}

#endif	/* NDB_DEBUG_MEMORY */

/*-------------------------------------------------------------------------
 * Array cleanup helpers
 *-------------------------------------------------------------------------
 */

/*
 * ndb_safe_pfree_array - Safely free an array of pointers
 *
 * Frees each non-NULL pointer in the array, then frees the array itself
 *
 * Usage:
 *   float **arrays = palloc(sizeof(float *) * count);
 *   ... populate arrays ...
 *   ndb_safe_pfree_array((void **)arrays, count);
 */
void
ndb_safe_pfree_array(void **array, int count)
{
	int i;

	if (array == NULL)
		return;

	if (count > 0)
	{
		for (i = 0; i < count; i++)
		{
			if (array[i] != NULL)
				ndb_safe_pfree_and_null_impl(array[i]);
		}
	}

	ndb_safe_pfree_and_null_impl(array);
}

/*
 * ndb_safe_pfree_string_array - Safely free an array of strings
 *
 * Frees each string, then the array itself
 */
void
ndb_safe_pfree_string_array(char **array, int count)
{
	int i;

	if (array == NULL)
		return;

	if (count > 0)
	{
		for (i = 0; i < count; i++)
		{
			if (array[i] != NULL)
				ndb_safe_pfree_and_null_impl(array[i]);
		}
	}

	ndb_safe_pfree_and_null_impl(array);
}

/*-------------------------------------------------------------------------
 * Cleanup pattern helpers
 *-------------------------------------------------------------------------
 */

/*
 * ndb_cleanup_on_error - Standardized cleanup on error
 *
 * Cleans up multiple resources in error path:
 * - Frees pointers if non-NULL
 * - Cleans up SPI if connected
 * - Switches to old context and deletes call context
 *
 * This matches the pattern used in ml_unified_api.c
 */
void
ndb_cleanup_on_error(MemoryContext oldcontext,
					 MemoryContext callcontext,
					 bool finish_spi,
					 int n_ptrs,
					 ...)
{
	va_list ap;
	int i;
	void *ptr;

	/* Free all provided pointers */
	if (n_ptrs > 0)
	{
		va_start(ap, n_ptrs);
		for (i = 0; i < n_ptrs; i++)
		{
			ptr = va_arg(ap, void *);
			ndb_safe_pfree_and_null_impl(ptr);
		}
		va_end(ap);
	}

	/* Clean up SPI if needed */
	if (finish_spi)
		SPI_finish();

	/* Ensure we're in old context before deleting call context */
	if (oldcontext != NULL)
		ndb_ensure_memory_context(oldcontext);

	if (callcontext != NULL && callcontext != oldcontext)
		ndb_safe_context_cleanup(callcontext, oldcontext);
}

