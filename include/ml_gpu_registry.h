/*-------------------------------------------------------------------------
 *
 * ml_gpu_registry.h
 *    Registration helper for GPU-capable ML model implementations.
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_GPU_REGISTRY_H
#define NEURONDB_ML_GPU_REGISTRY_H

extern void neurondb_gpu_register_models(void);
extern void neurondb_gpu_register_rf_model(void);
extern void neurondb_gpu_register_lr_model(void);
extern void neurondb_gpu_register_linreg_model(void);
extern void neurondb_gpu_register_svm_model(void);

#endif /* NEURONDB_ML_GPU_REGISTRY_H */

