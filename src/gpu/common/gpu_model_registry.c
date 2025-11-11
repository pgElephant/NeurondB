/*-------------------------------------------------------------------------
 *
 * gpu_model_registry.c
 *    Registry of GPU model operators.
 *
 * Allows individual algorithms to register GPU-native implementations.
 * The registry is consulted by the unified ML entry points when routing
 * training and prediction to GPU.
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "neurondb_gpu_model.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"

typedef struct MLGpuModelEntry
{
	const char *algorithm;
	const MLGpuModelOps *ops;
} MLGpuModelEntry;

static HTAB *gpu_model_registry = NULL;

static void
ndb_gpu_init_model_registry(void)
{
	HASHCTL ctl;

	if (gpu_model_registry != NULL)
		return;

	MemSet(&ctl, 0, sizeof(HASHCTL));
	ctl.keysize = sizeof(char *);
	ctl.entrysize = sizeof(MLGpuModelEntry);
	ctl.hcxt = TopMemoryContext;
	gpu_model_registry = hash_create("neurondb GPU model registry",
		16,
		&ctl,
		HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
}

bool
ndb_gpu_register_model_ops(const MLGpuModelOps *ops)
{
	MLGpuModelEntry *entry;
	bool found;

	if (ops == NULL || ops->algorithm == NULL)
		return false;

	ndb_gpu_init_model_registry();

	entry = (MLGpuModelEntry *)hash_search(
		gpu_model_registry, &ops->algorithm, HASH_ENTER, &found);
	if (entry == NULL)
		return false;
	entry->algorithm = ops->algorithm;
	entry->ops = ops;
	return !found;
}

const MLGpuModelOps *
ndb_gpu_lookup_model_ops(const char *algorithm)
{
	MLGpuModelEntry *entry;

	if (algorithm == NULL || gpu_model_registry == NULL)
		return NULL;

	entry = (MLGpuModelEntry *)hash_search(
		gpu_model_registry, &algorithm, HASH_FIND, NULL);
	if (entry == NULL)
		return NULL;
	return entry->ops;
}

void
ndb_gpu_clear_model_registry(void)
{
	if (gpu_model_registry != NULL)
	{
		hash_destroy(gpu_model_registry);
		gpu_model_registry = NULL;
	}
}
