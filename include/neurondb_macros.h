/*-------------------------------------------------------------------------
 *
 * neurondb_macros.h
 *	  Strict pointer lifetime helpers for NeurondDB
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_MACROS_H
#define NEURONDB_MACROS_H

#include "postgres.h"
#include "utils/memutils.h"

/*
 * NDB_DECLARE
 * Declare pointer and initialize to NULL.
 *
 * Usage:
 *	 NDB_DECLARE(char *, buf);
 *	 NDB_DECLARE(LinRegModel *, model);
 */
#define NDB_DECLARE(type, name) \
	type name = NULL

/*
 * NDB_ALLOC
 * Allocate with palloc0, require previous value NULL.
 *
 * Usage:
 *	 NDB_ALLOC(model, LinRegModel, 1);
 *	 NDB_ALLOC(coeffs, double, n_features);
 */
#define NDB_ALLOC(ptr, type, count)				\
	Assert((ptr) == NULL);						\
	do {									\
		Assert((ptr) == NULL);				\
		(ptr) = (type *) palloc0(sizeof(type) * (count));	\
	} while (0)

/*
 * NDB_FREE
 * Free pointer if not NULL and then set to NULL.
 *
 * Usage:
 *	 NDB_FREE(model);
 *	 NDB_FREE(coeffs);
 */
#define NDB_FREE(ptr)					\
	do {								\
		if ((ptr) != NULL)				\
		{								\
			pfree(ptr);					\
			(ptr) = NULL;				\
		}								\
	} while (0)

#endif	/* NEURONDB_MACROS_H */
