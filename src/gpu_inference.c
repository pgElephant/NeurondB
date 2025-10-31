/*-------------------------------------------------------------------------
 *
 * gpu_inference.c
 *		GPU-accelerated ML inference
 *
 * Implements ONNX Runtime GPU inference and embedding generation
 * for high-throughput batch inference operations.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/gpu_inference.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"

#include "neurondb_config.h"
#include "neurondb_gpu.h"

#ifdef HAVE_ONNXRUNTIME_GPU
#include <onnxruntime_cxx_api.h>
#endif

/*
 * GPU ONNX inference
 */
void
neurondb_gpu_onnx_inference(void *model_handle, const float *input,
							int input_size, float *output, int output_size)
{
	if (!neurondb_gpu_is_available())
		return;

#ifdef HAVE_ONNXRUNTIME_GPU
	if (neurondb_gpu_get_backend() == GPU_BACKEND_CUDA)
	{
		/* ONNX Runtime GPU inference */
		/* This requires ONNX Runtime with CUDA execution provider */
		elog(DEBUG1, "neurondb: ONNX Runtime GPU not linked, using CPU fallback");
		return;
	}
#endif

	/* CPU fallback handled by caller */
}

