/*-------------------------------------------------------------------------
 *
 * ml_gpu_buffer.c
 *	  Shared GPU buffer abstraction for NeurondB ML pipelines.
 *
 * The initial implementation focuses on consistent bookkeeping for host
 * buffers and graceful CPU fallback. Device allocation and transfers are
 * stubbed for now; future steps will wire these through backend-specific
 * kernels.
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "ml_gpu_buffer.h"

#include "utils/memutils.h"

Size
ml_gpu_dtype_size(MLGpuDType dtype)
{
	switch (dtype)
	{
		case MLGPU_DTYPE_FLOAT32:
			return sizeof(float);
		case MLGPU_DTYPE_FLOAT64:
			return sizeof(double);
		case MLGPU_DTYPE_INT32:
			return sizeof(int32);
		case MLGPU_DTYPE_UINT8:
			return sizeof(uint8);
		case MLGPU_DTYPE_INVALID:
		default:
			break;
	}

	elog(ERROR, "ml_gpu_buffer: unknown dtype %d", dtype);
	pg_unreachable();
}

static void
ml_gpu_buffer_reset(MLGpuBuffer *buf)
{
	Assert(buf != NULL);

	buf->context = NULL;
	buf->host_ptr = NULL;
	buf->host_bytes = 0;
	buf->host_owner = false;
	buf->device_ptr = NULL;
	buf->device_bytes = 0;
	buf->device_owner = false;
	buf->elem_count = 0;
	buf->dtype = MLGPU_DTYPE_INVALID;
	buf->host_valid = false;
	buf->device_valid = false;
}

void
ml_gpu_buffer_init_owner(MLGpuBuffer *buf,
						 MLGpuContext *ctx,
						 MLGpuDType dtype,
						 int64 elem_count,
						 bool zero)
{
	MemoryContext oldcx;
	Size elem_size;
	Size total_bytes;

	Assert(buf != NULL);
	Assert(ctx != NULL);
	Assert(elem_count >= 0);

	ml_gpu_buffer_reset(buf);

	elem_size = ml_gpu_dtype_size(dtype);
	total_bytes = (Size) elem_count * elem_size;

	oldcx = MemoryContextSwitchTo(ml_gpu_context_memory(ctx));

	buf->context = ctx;
	buf->dtype = dtype;
	buf->elem_count = elem_count;
	buf->host_bytes = total_bytes;
	buf->host_owner = true;
	buf->host_valid = true;

	if (total_bytes > 0)
	{
		buf->host_ptr = zero ?
			palloc0(total_bytes) :
			palloc(total_bytes);
	}

	MemoryContextSwitchTo(oldcx);
}

void
ml_gpu_buffer_bind_host(MLGpuBuffer *buf,
						MLGpuContext *ctx,
						void *host_ptr,
						Size host_bytes,
						int64 elem_count,
						MLGpuDType dtype)
{
	Assert(buf != NULL);

	ml_gpu_buffer_reset(buf);

	buf->context = ctx;
	buf->host_ptr = host_ptr;
	buf->host_bytes = host_bytes;
	buf->host_owner = false;
	buf->host_valid = (host_ptr != NULL);
	buf->elem_count = elem_count;
	buf->dtype = dtype;
}

void
ml_gpu_buffer_invalidate_device(MLGpuBuffer *buf)
{
	if (buf == NULL)
		return;

	buf->device_valid = false;
}

bool
ml_gpu_buffer_ensure_device(MLGpuBuffer *buf, bool copy_from_host)
{
	if (buf == NULL || buf->context == NULL)
		return false;

	if (!ml_gpu_context_ready(buf->context))
		return false;

	/* Placeholder for future device allocation logic. */
	(void) copy_from_host;
	return false;
}

bool
ml_gpu_buffer_copy_to_host(MLGpuBuffer *buf)
{
	if (buf == NULL)
		return false;

	if (!ml_gpu_context_ready(buf->context))
		return false;

	if (!buf->device_valid || buf->host_ptr == NULL)
		return false;

	/* Device copies not implemented yet. */
	return false;
}

void
ml_gpu_buffer_release(MLGpuBuffer *buf)
{
	if (buf == NULL)
		return;

	if (buf->host_owner && buf->host_ptr != NULL)
		pfree(buf->host_ptr);

	/* Device resource cleanup will be added in a later phase. */

	ml_gpu_buffer_reset(buf);
}


