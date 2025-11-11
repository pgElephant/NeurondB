/*-------------------------------------------------------------------------
 *
 * ml_gpu_registry.c
 *    Registers GPU-capable ML model implementations.
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "ml_gpu_registry.h"

/* Declarations for per-algorithm registration routines */
extern void neurondb_gpu_register_rf_model(void);

void
neurondb_gpu_register_models(void)
{
	static bool registered = false;

	if (registered)
		return;

	neurondb_gpu_register_rf_model();

	registered = true;
}
