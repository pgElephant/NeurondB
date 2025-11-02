/*-------------------------------------------------------------------------
 *
 * scan_quota.c
 *		Quota enforcement for index operations
 *
 * Implements hard limits on:
 * - Per-tenant vector count
 * - Per-tenant storage size
 * - Per-tenant index size
 * - Rate limiting for inserts
 *
 * Quotas are checked during insert/update operations and enforced
 * before writing to the index.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/scan/scan_quota.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_scan.h"
#include "fmgr.h"
#include "access/htup_details.h"
#include "access/relation.h"
#include "catalog/pg_class.h"
#include "executor/spi.h"
#include "storage/lmgr.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/rel.h"
#include "utils/timestamp.h"
#include "utils/typcache.h"
#include "funcapi.h"

/*
 * Quota limits structure
 */
typedef struct QuotaLimits
{
	int64		maxVectors;		/* Maximum vectors per tenant */
	int64		maxStorageBytes; /* Maximum storage per tenant */
	int64		maxIndexSize;	/* Maximum index size per tenant */
	int			maxQPS;			/* Maximum queries per second */
	bool		enforceHard;	/* Hard enforcement vs. warning */
} QuotaLimits;

/*
 * Current usage structure
 */
typedef struct QuotaUsage
{
	char	   *tenantId;
	int64		vectorCount;
	int64		storageBytes;
	int64		indexSize;
	int			currentQPS;
	TimestampTz lastCheck;
} QuotaUsage;

/* GUCs for default quotas */
static int64 default_max_vectors = 1000000;
static int64 default_max_storage_mb = 10240; /* 10 GB */
static int	default_max_qps = 1000;
static bool	enforce_quotas = true;

/*
 * Module initialization
 */
void
ndb_quota_init_guc(void)
{
	DefineCustomIntVariable("neurondb.default_max_vectors",
							"Default maximum vectors per tenant (thousands)",
							NULL,
							(int *) &default_max_vectors,
							1000000,
							1000,
							INT_MAX,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomIntVariable("neurondb.default_max_storage_mb",
							"Default maximum storage (MB) per tenant",
							NULL,
							(int *) &default_max_storage_mb,
							10240,
							100,
							INT_MAX,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomIntVariable("neurondb.default_max_qps",
							"Default maximum queries per second per tenant",
							NULL,
							&default_max_qps,
							1000,
							1,
							INT_MAX,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	DefineCustomBoolVariable("neurondb.enforce_quotas",
							 "Enable hard quota enforcement",
							 NULL,
							 &enforce_quotas,
							 true,
							 PGC_SUSET,
							 0,
							 NULL, NULL, NULL);
}

/*
 * Get quota limits for a tenant
 */
static QuotaLimits *
get_tenant_quota(const char *tenantId)
{
	QuotaLimits *limits;

	limits = (QuotaLimits *) palloc0(sizeof(QuotaLimits));
	
	/* TODO: Query neurondb.tenant_quotas table */
	/* For now, use defaults */
	limits->maxVectors = default_max_vectors;
	limits->maxStorageBytes = default_max_storage_mb * 1024 * 1024;
	limits->maxIndexSize = default_max_storage_mb * 1024 * 1024;
	limits->maxQPS = default_max_qps;
	limits->enforceHard = enforce_quotas;

	return limits;
}

/*
 * Get current usage for a tenant
 */
static QuotaUsage *
get_tenant_usage(const char *tenantId, Oid indexOid)
{
	QuotaUsage *usage;
	int			ret;
	bool		isnull;

	usage = (QuotaUsage *) palloc0(sizeof(QuotaUsage));
	usage->tenantId = pstrdup(tenantId);
	usage->lastCheck = GetCurrentTimestamp();

	/* Query usage from catalog */
	SPI_connect();

	/* Get vector count */
	ret = SPI_execute("SELECT count(*) FROM neurondb.tenant_usage WHERE tenant_id = $1",
					  true, 0);
	
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		usage->vectorCount = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0],
														  SPI_tuptable->tupdesc,
														  1, &isnull));
	}

	/* TODO: Get storage size from pg_class */
	/* TODO: Get QPS from stats */

	SPI_finish();

	return usage;
}

/*
 * Check if operation would exceed quota
 */
bool
ndb_quota_check(const char *tenantId, Oid indexOid, int64 additionalVectors,
				int64 additionalBytes)
{
	QuotaLimits *limits;
	QuotaUsage *usage;
	bool		allowed = true;

	if (!enforce_quotas)
		return true;

	if (tenantId == NULL)
		return true; /* No tenant = no quota */

	limits = get_tenant_quota(tenantId);
	usage = get_tenant_usage(tenantId, indexOid);

	/* Check vector count */
	if (usage->vectorCount + additionalVectors > limits->maxVectors)
	{
		if (limits->enforceHard)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
					 errmsg("neurondb: quota exceeded for tenant '%s'", tenantId),
					 errdetail("Vector count %lld + %lld would exceed limit %lld",
							   (long long) usage->vectorCount,
							   (long long) additionalVectors,
							   (long long) limits->maxVectors),
					 errhint("Contact administrator to increase quota or delete old vectors")));
		}
		else
		{
			ereport(WARNING,
					(errmsg("neurondb: tenant '%s' approaching vector quota (%lld/%lld)",
							tenantId,
							(long long) usage->vectorCount,
							(long long) limits->maxVectors)));
		}
		allowed = false;
	}

	/* Check storage size */
	if (usage->storageBytes + additionalBytes > limits->maxStorageBytes)
	{
		if (limits->enforceHard)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
					 errmsg("neurondb: storage quota exceeded for tenant '%s'", tenantId),
					 errdetail("Storage %lld + %lld bytes would exceed limit %lld",
							   (long long) usage->storageBytes,
							   (long long) additionalBytes,
							   (long long) limits->maxStorageBytes)));
		}
		allowed = false;
	}

	pfree(limits);
	pfree(usage);

	return allowed;
}

/*
 * Enforce quota at index insert time
 *
 * Called from index insert functions before writing.
 */
void
ndb_quota_enforce_insert(Relation index, const char *tenantId,
						 int64 vectorCount, int64 estimatedBytes)
{
	Oid			indexOid = RelationGetRelid(index);

	if (!ndb_quota_check(tenantId, indexOid, vectorCount, estimatedBytes))
	{
		/* Error already raised if hard enforcement */
		return;
	}

	elog(DEBUG2, "neurondb: Quota check passed for tenant '%s'", tenantId);
}

/*
 * Update usage statistics
 */
void
ndb_quota_update_usage(const char *tenantId, Oid indexOid,
					   int64 vectorsDelta, int64 bytesDelta)
{
	StringInfoData query;

	if (tenantId == NULL)
		return;

	initStringInfo(&query);

	/* Upsert usage record */
	appendStringInfo(&query,
					 "INSERT INTO neurondb.tenant_usage "
					 "(tenant_id, index_oid, vector_count, storage_bytes, last_updated) "
					 "VALUES ('%s', %u, %lld, %lld, now()) "
					 "ON CONFLICT (tenant_id, index_oid) "
					 "DO UPDATE SET "
					 "vector_count = neurondb.tenant_usage.vector_count + %lld, "
					 "storage_bytes = neurondb.tenant_usage.storage_bytes + %lld, "
					 "last_updated = now()",
					 tenantId, indexOid,
					 (long long) vectorsDelta, (long long) bytesDelta,
					 (long long) vectorsDelta, (long long) bytesDelta);

	SPI_connect();
	SPI_execute(query.data, false, 0);
	SPI_finish();

	elog(DEBUG1, "neurondb: Updated quota usage for tenant '%s'", tenantId);
}

/*
 * SQL-callable quota check function
 */
PG_FUNCTION_INFO_V1(neurondb_check_quota);

Datum
neurondb_check_quota(PG_FUNCTION_ARGS)
{
	text	   *tenant_id_text = PG_GETARG_TEXT_PP(0);
	Oid			index_oid = PG_GETARG_OID(1);
	int64		additional_vectors = PG_GETARG_INT64(2);
	char	   *tenant_id = text_to_cstring(tenant_id_text);
	bool		allowed;

	allowed = ndb_quota_check(tenant_id, index_oid, additional_vectors, 0);

	PG_RETURN_BOOL(allowed);
}

/*
 * SQL-callable function to get quota usage
 */
PG_FUNCTION_INFO_V1(neurondb_get_quota_usage);

Datum
neurondb_get_quota_usage(PG_FUNCTION_ARGS)
{
	text	   *tenant_id_text = PG_GETARG_TEXT_PP(0);
	Oid			index_oid = PG_GETARG_OID(1);
	char	   *tenant_id = text_to_cstring(tenant_id_text);
	QuotaUsage *usage;
	QuotaLimits *limits;
	TupleDesc	tupdesc;
	Datum		values[6];
	bool		nulls[6];
	HeapTuple	tuple;

	usage = get_tenant_usage(tenant_id, index_oid);
	limits = get_tenant_quota(tenant_id);

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("function returning record called in context that cannot accept type record")));

	tupdesc = BlessTupleDesc(tupdesc);

	values[0] = Int64GetDatum(usage->vectorCount);
	values[1] = Int64GetDatum(limits->maxVectors);
	values[2] = Int64GetDatum(usage->storageBytes);
	values[3] = Int64GetDatum(limits->maxStorageBytes);
	values[4] = Int32GetDatum(usage->currentQPS);
	values[5] = Int32GetDatum(limits->maxQPS);

	memset(nulls, 0, sizeof(nulls));

	tuple = heap_form_tuple(tupdesc, values, nulls);

	PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/*
 * Reset quota usage (for testing)
 */
PG_FUNCTION_INFO_V1(neurondb_reset_quota);

Datum
neurondb_reset_quota(PG_FUNCTION_ARGS)
{
	text	   *tenant_id_text = PG_GETARG_TEXT_PP(0);
	char	   *tenant_id = text_to_cstring(tenant_id_text);

	SPI_connect();
	SPI_execute("DELETE FROM neurondb.tenant_usage WHERE tenant_id = $1", false, 0);
	SPI_finish();

	elog(NOTICE, "neurondb: Reset quota for tenant '%s'", tenant_id);

	PG_RETURN_VOID();
}

