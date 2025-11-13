#ifndef NEURONDB_GPU_BACKEND_H
#define NEURONDB_GPU_BACKEND_H

#include "neurondb_gpu_types.h"
#include "utils/jsonb.h"

struct RFModel;
struct LRModel;
struct LinRegModel;
struct SVMModel;
struct DTModel;
struct RidgeModel;
struct LassoModel;

#define NDB_GPU_MAX_BACKENDS 8

/*
 * Core backend interface. Each GPU implementation exposes a single instance of
 * this structure and registers it with the common registry at module load.
 *
 * Return codes follow PostgreSQL conventions: 0 indicates success while
 * negative values represent backend-specific failures. Callers should translate
 * errors to elog/ereport as appropriate.
 */
typedef struct ndb_gpu_backend
{
	/* Identity */
	const char *name;
	const char *provider; /* e.g., "NVIDIA", "AMD", "Apple" */
	NDBGpuBackendKind kind;
	unsigned int features; /* Bitmask of NDBGpuBackendFeature */
	int priority; /* Higher wins when auto-selecting */

	/* Lifecycle */
	int (*init)(void);
	void (*shutdown)(void);
	int (*is_available)(void);

	/* Device management */
	int (*device_count)(void);
	int (*device_info)(int device_id, NDBGpuDeviceInfo *info);
	int (*set_device)(int device_id);

	/* Memory helpers */
	int (*mem_alloc)(void **ptr, size_t bytes);
	int (*mem_free)(void *ptr);
	int (*memcpy_h2d)(void *dst, const void *src, size_t bytes);
	int (*memcpy_d2h)(void *dst, const void *src, size_t bytes);

	/* Launchers */
	int (*launch_l2_distance)(const float *A,
		const float *B,
		float *out,
		int n,
		int d,
		ndb_stream_t stream);
	int (*launch_cosine)(const float *A,
		const float *B,
		float *out,
		int n,
		int d,
		ndb_stream_t stream);
	int (*launch_kmeans_assign)(const float *X,
		const float *C,
		int *idx,
		int n,
		int d,
		int k,
		ndb_stream_t stream);
	int (*launch_kmeans_update)(const float *X,
		const int *idx,
		float *C,
		int n,
		int d,
		int k,
		ndb_stream_t stream);
	int (*launch_quant_fp16)(const float *in,
		void *out,
		int n,
		ndb_stream_t stream);
	int (*launch_quant_int8)(const float *in,
		int8_t *out,
		int n,
		float scale,
		ndb_stream_t stream);
	int (*launch_quant_binary)(const float *in,
		uint8_t *out,
		int n,
		ndb_stream_t stream);
	int (*launch_pq_encode)(const float *X,
		const float *codebooks,
		uint8_t *codes,
		int n,
		int d,
		int m,
		int ks,
		ndb_stream_t stream);

	/* Random forest */
	int (*rf_train)(const float *features,
		const double *labels,
		int n_samples,
		int feature_dim,
		int class_count,
		const Jsonb *hyperparams,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);
	int (*rf_predict)(const bytea *model_data,
		const float *input,
		int feature_dim,
		int *class_out,
		char **errstr);
	int (*rf_pack)(const struct RFModel *model,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);

	/* Logistic regression */
	int (*lr_train)(const float *features,
		const double *labels,
		int n_samples,
		int feature_dim,
		const Jsonb *hyperparams,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);
	int (*lr_predict)(const bytea *model_data,
		const float *input,
		int feature_dim,
		double *probability_out,
		char **errstr);
	int (*lr_pack)(const struct LRModel *model,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);

	/* Linear regression */
	int (*linreg_train)(const float *features,
		const double *targets,
		int n_samples,
		int feature_dim,
		const Jsonb *hyperparams,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);
	int (*linreg_predict)(const bytea *model_data,
		const float *input,
		int feature_dim,
		double *prediction_out,
		char **errstr);
	int (*linreg_pack)(const struct LinRegModel *model,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);

	/* Support Vector Machine */
	int (*svm_train)(const float *features,
		const double *labels,
		int n_samples,
		int feature_dim,
		const Jsonb *hyperparams,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);
	int (*svm_predict)(const bytea *model_data,
		const float *input,
		int feature_dim,
		int *class_out,
		double *confidence_out,
		char **errstr);
	int (*svm_pack)(const struct SVMModel *model,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);

	/* Decision Tree */
	int (*dt_train)(const float *features,
		const double *labels,
		int n_samples,
		int feature_dim,
		const Jsonb *hyperparams,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);
	int (*dt_predict)(const bytea *model_data,
		const float *input,
		int feature_dim,
		double *prediction_out,
		char **errstr);
	int (*dt_pack)(const struct DTModel *model,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);

	/* Ridge Regression */
	int (*ridge_train)(const float *features,
		const double *targets,
		int n_samples,
		int feature_dim,
		const Jsonb *hyperparams,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);
	int (*ridge_predict)(const bytea *model_data,
		const float *input,
		int feature_dim,
		double *prediction_out,
		char **errstr);
	int (*ridge_pack)(const struct RidgeModel *model,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);

	/* Lasso Regression */
	int (*lasso_train)(const float *features,
		const double *targets,
		int n_samples,
		int feature_dim,
		const Jsonb *hyperparams,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);
	int (*lasso_predict)(const bytea *model_data,
		const float *input,
		int feature_dim,
		double *prediction_out,
		char **errstr);
	int (*lasso_pack)(const struct LassoModel *model,
		bytea **model_data,
		Jsonb **metrics,
		char **errstr);

	/* Stream utilities */
	int (*stream_create)(ndb_stream_t *stream);
	int (*stream_destroy)(ndb_stream_t stream);
	int (*stream_synchronize)(ndb_stream_t stream);
} ndb_gpu_backend;

/* Registry hooks used by backend implementations */
int ndb_gpu_register_backend(const ndb_gpu_backend *backend);
int ndb_gpu_set_active_backend(const ndb_gpu_backend *backend);
const ndb_gpu_backend *ndb_gpu_get_active_backend(void);
const ndb_gpu_backend *ndb_gpu_select_backend(const char *name);
void ndb_gpu_list_backends(void);

void neurondb_gpu_register_cuda_backend(void);
void neurondb_gpu_register_rocm_backend(void);
void neurondb_gpu_register_metal_backend(void);

#endif /* NEURONDB_GPU_BACKEND_H */
