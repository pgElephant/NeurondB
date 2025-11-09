/*-------------------------------------------------------------------------
 *
 * gpu_backend_registry.c
 *     GPU backend registration and selection system
 *
 * This module manages the registry of available GPU backends and provides
 * automatic backend selection based on system capabilities.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "utils/builtins.h"
#include "utils/elog.h"

#include "neurondb_gpu_backend.h"

#include <string.h>

typedef struct NDBGpuBackendRegistry
{
    const ndb_gpu_backend *backends[NDB_GPU_MAX_BACKENDS];
    int                     count;
} NDBGpuBackendRegistry;

static NDBGpuBackendRegistry registry = {
    .backends = {NULL},
    .count = 0,
};

static const ndb_gpu_backend *active_backend = NULL;

static int
ndb_backend_priority(NDBGpuBackendKind kind)
{
    switch (kind)
    {
        case NDB_GPU_BACKEND_METAL:
            return 100;
        case NDB_GPU_BACKEND_CUDA:
            return 90;
        case NDB_GPU_BACKEND_ROCM:
            return 80;
        default:
            return 0;
    }
}

static int
ndb_backend_is_available(const ndb_gpu_backend *backend)
{
    if (backend == NULL)
        return 0;

    if (backend->is_available == NULL)
        return 1;

    return backend->is_available();
}

int
ndb_gpu_register_backend(const ndb_gpu_backend *backend)
{
    int i;

    if (backend == NULL)
    {
        elog(WARNING, "neurondb: attempted to register NULL GPU backend");
        return -1;
    }

    if (registry.count >= NDB_GPU_MAX_BACKENDS)
    {
        elog(WARNING, "neurondb: GPU backend registry full; ignoring '%s'",
             backend->name ? backend->name : "unknown");
        return -1;
    }

    for (i = 0; i < registry.count; i++)
    {
        if (registry.backends[i]->kind == backend->kind)
        {
            elog(DEBUG1,
                 "neurondb: GPU backend kind %d already registered; keeping existing entry",
                 backend->kind);
            return 0;
        }
    }

    registry.backends[registry.count++] = backend;

    elog(DEBUG1, "neurondb: GPU backend registered: %s (%s)",
         backend->name ? backend->name : "unnamed",
         backend->provider ? backend->provider : "unknown");

    return 0;
}

static const ndb_gpu_backend *
ndb_gpu_select_best_internal(void)
{
    const ndb_gpu_backend *best = NULL;
    int best_priority = -1;
    int i;

    for (i = 0; i < registry.count; i++)
    {
        const ndb_gpu_backend *candidate = registry.backends[i];
        int priority;

        if (!ndb_backend_is_available(candidate))
            continue;

        priority = candidate->priority != 0
                        ? candidate->priority
                        : ndb_backend_priority(candidate->kind);

        if (priority > best_priority)
        {
            best = candidate;
            best_priority = priority;
        }
    }

    return best;
}

int
ndb_gpu_set_active_backend(const ndb_gpu_backend *backend)
{
    if (active_backend == backend)
        return 0;

    if (active_backend && active_backend->shutdown)
        active_backend->shutdown();

    active_backend = backend;

    if (backend && backend->init)
    {
        int rc = backend->init();

        if (rc != 0)
        {
            elog(WARNING,
                 "neurondb: failed to initialise GPU backend '%s' (rc=%d)",
                 backend->name ? backend->name : "unknown", rc);

            if (backend->shutdown)
                backend->shutdown();

            active_backend = NULL;
            return rc;
        }
    }

    return 0;
}

const ndb_gpu_backend *
ndb_gpu_get_active_backend(void)
{
    return active_backend;
}

const ndb_gpu_backend *
ndb_gpu_select_backend(const char *name)
{
    const ndb_gpu_backend *chosen = NULL;
    int i;

    if (registry.count == 0)
    {
        elog(DEBUG1, "neurondb: no GPU backends registered");
        return NULL;
    }

    if (name == NULL || name[0] == '\0' || pg_strcasecmp(name, "auto") == 0)
    {
        chosen = ndb_gpu_select_best_internal();
        if (!chosen)
        {
            elog(DEBUG1, "neurondb: no suitable GPU backend available");
            return NULL;
        }
    }
    else
    {
        for (i = 0; i < registry.count; i++)
        {
            const ndb_gpu_backend *candidate = registry.backends[i];

            if (pg_strcasecmp(candidate->name, name) != 0)
                continue;

            if (!ndb_backend_is_available(candidate))
            {
                elog(WARNING, "neurondb: GPU backend '%s' not available", name);
                return NULL;
            }

            chosen = candidate;
            break;
        }

        if (chosen == NULL)
        {
            elog(WARNING, "neurondb: GPU backend '%s' not found", name);
            return NULL;
        }
    }

    if (ndb_gpu_set_active_backend(chosen) != 0)
        return NULL;

    elog(LOG, "neurondb: selected GPU backend: %s", chosen->name);

    return active_backend;
}

void
ndb_gpu_list_backends(void)
{
    int i;

    elog(LOG, "neurondb: GPU backends registered: %d", registry.count);

    for (i = 0; i < registry.count; i++)
    {
        const ndb_gpu_backend *backend = registry.backends[i];
        const char *name = backend->name ? backend->name : "unknown";
        const char *provider = backend->provider ? backend->provider : "unknown";
        bool available = ndb_backend_is_available(backend) != 0;

        elog(LOG, "  [%d] %s (%s) - %s", i + 1, name, provider,
             available ? "available" : "unavailable");
    }
}

