#ifndef NEURONDB_AUTOML_H
#define NEURONDB_AUTOML_H

#include "postgres.h"

/* AutoML backend indicator */
typedef enum AutoMLBackendType
{
	AUTOML_BACKEND_CPU = 0,
	AUTOML_BACKEND_GPU = 1
} AutoMLBackendType;

extern PGDLLIMPORT bool neurondb_automl_use_gpu;

extern void neurondb_automl_define_gucs(void);
extern AutoMLBackendType neurondb_automl_choose_backend(const char *algorithm);

#endif /* NEURONDB_AUTOML_H */
