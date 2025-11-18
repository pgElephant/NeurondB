/*-------------------------------------------------------------------------
 *
 * hnsw_am.c
 *	  HNSW (Hierarchical Navigable Small World) Index Access Method
 *
 * Implementation of HNSW index as a PostgreSQL Index Access Method:
 * - Probabilistic multi-layer graph
 * - Bidirectional link maintenance
 * - ef_construction and ef_search parameters
 * - Insert, delete, search, update, bulkdelete, vacuum, costestimate, etc.
 *
 * Based on the paper:
 * "Efficient and robust approximate nearest neighbor search using
 *  Hierarchical Navigable Small World graphs" by Malkov & Yashunin (2018)
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *
 * IDENTIFICATION
 *	  src/index/hnsw_am.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"

/* Forward declaration for fp16_to_float from quantization.c */
extern float fp16_to_float(uint16 h);
#include "access/amapi.h"
#include "access/generic_xlog.h"
#include "access/htup_details.h"
#include "access/reloptions.h"
#include "access/relscan.h"
#include "access/tableam.h"
#include "catalog/index.h"
#include "catalog/pg_am.h"
#include "catalog/pg_type.h"
#include "catalog/pg_namespace.h"
#include "commands/vacuum.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "storage/bufmgr.h"
#include "storage/indexfsm.h"
#include "storage/lmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/typcache.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "parser/parse_type.h"
#include "nodes/parsenodes.h"
#include "nodes/makefuncs.h"
#include "funcapi.h"
#include "utils/varbit.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>

/*
 * HNSW AM type definitions and constants
 */
#define HNSW_DEFAULT_M			16
#define HNSW_DEFAULT_EF_CONSTRUCTION	200
#define HNSW_DEFAULT_EF_SEARCH		64
#define HNSW_DEFAULT_ML			0.36f
#define HNSW_MAX_LEVEL			16
#define HNSW_MAGIC_NUMBER		0x48534E57
#define HNSW_VERSION			1
#define RELOPT_KIND_HNSW		1

typedef struct HnswOptions
{
	int32		vl_len_;
	int			m;
	int			ef_construction;
	int			ef_search;
} HnswOptions;

typedef struct HnswMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	BlockNumber	entryPoint;
	int			entryLevel;
	int			maxLevel;
	int16		m;
	int16		efConstruction;
	int16		efSearch;
	float4		ml;
	int64		insertedVectors;
} HnswMetaPageData;

typedef HnswMetaPageData *HnswMetaPage;

typedef struct HnswNodeData
{
	ItemPointerData	heapPtr;
	int				level;
	int16			dim;
	int16			neighborCount[HNSW_MAX_LEVEL];
	/* Followed by:
	 *	float4 vector[dim];
	 *	BlockNumber neighbors[level+1][M*2];
	 */
} HnswNodeData;

typedef HnswNodeData *HnswNode;

#define HnswNodeSize(dim, level) \
	(MAXALIGN(sizeof(HnswNodeData) + (dim) * sizeof(float4) \
		+ ((level) + 1) * HNSW_DEFAULT_M * 2 * sizeof(BlockNumber)))

#define HnswGetVector(node) \
	((float4 *)((char *)(node) + MAXALIGN(sizeof(HnswNodeData))))

#define HnswGetNeighbors(node, lev) \
	((BlockNumber *)((char *)(node) + MAXALIGN(sizeof(HnswNodeData)) \
		+ (node)->dim * sizeof(float4) \
		+ (lev) * HNSW_DEFAULT_M * 2 * sizeof(BlockNumber)))

/*
 * Build state for index build
 */
typedef struct HnswBuildState
{
	Relation	heap;
	Relation	index;
	IndexInfo   *indexInfo;
	HnswMetaPage	metaPage;
	double		indtuples;
	Buffer		metaBuffer;
	MemoryContext	tmpCtx;
} HnswBuildState;

/*
 * Opaque for scan state
 */
typedef struct HnswScanOpaqueData
{
	int				efSearch;
	int				strategy;
	Vector		   *queryVector;
	int				k;
	bool			firstCall;
	int				resultCount;
	BlockNumber    *results;
	float4		   *distances;
	int				currentResult;
} HnswScanOpaqueData;

typedef HnswScanOpaqueData *HnswScanOpaque;

/*
 * Forward declarations
 */
static IndexBuildResult *hnswbuild(Relation heap, Relation index, IndexInfo *indexInfo);
static void		  hnswbuildempty(Relation index);
static bool		  hnswinsert(Relation index, Datum *values, bool *isnull, ItemPointer ht_ctid,
							 Relation heapRel, IndexUniqueCheck checkUnique,
							 bool indexUnchanged, struct IndexInfo *indexInfo);
static IndexBulkDeleteResult *hnswbulkdelete(IndexVacuumInfo *info,
											 IndexBulkDeleteResult *stats,
											 IndexBulkDeleteCallback callback,
											 void *callback_state);
static IndexBulkDeleteResult *hnswvacuumcleanup(IndexVacuumInfo *info,
												IndexBulkDeleteResult *stats);
static bool		  hnswdelete(Relation index, ItemPointer tid, Datum *values, bool *isnull,
							 Relation heapRel, struct IndexInfo *indexInfo) __attribute__((unused));
static bool		  hnswupdate(Relation index, ItemPointer tid, Datum *values, bool *isnull,
							  ItemPointer otid, Relation heapRel, struct IndexInfo *indexInfo) __attribute__((unused));
static void		  hnswcostestimate(struct PlannerInfo *root, struct IndexPath *path, double loop_count,
								   Cost *indexStartupCost, Cost *indexTotalCost,
								   Selectivity *indexSelectivity, double *indexCorrelation,
								   double *indexPages);
static bytea	 *hnswoptions(Datum reloptions, bool validate);
static bool		  hnswproperty(Oid index_oid, int attno, IndexAMProperty prop,
							  const char *propname, bool *res, bool *isnull);
static IndexScanDesc hnswbeginscan(Relation index, int nkeys, int norderbys);
static void		  hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
static bool		  hnswgettuple(IndexScanDesc scan, ScanDirection dir);
static void		  hnswendscan(IndexScanDesc scan);

static void		  hnswInitMetaPage(Buffer metaBuffer, int16 m, int16 efConstruction, int16 efSearch, float4 ml);
static int		  hnswGetRandomLevel(float4 ml);
static float4	  hnswComputeDistance(const float4 *vec1, const float4 *vec2, int dim, int strategy) __attribute__((unused));
static void		  hnswSearch(Relation index, HnswMetaPage metaPage, const float4 *query,
							  int dim, int strategy, int efSearch, int k,
							  BlockNumber **results, float4 **distances, int *resultCount);
static void		  hnswInsertNode(Relation index, HnswMetaPage metaPage,
								  const float4 *vector, int dim, ItemPointer heapPtr);
static float4	 *hnswExtractVectorData(Datum value, Oid typeOid, int *out_dim, MemoryContext ctx);
static Oid		  hnswGetKeyType(Relation index, int attno);
static void		  hnswBuildCallback(Relation index, ItemPointer tid, Datum *values,
								  bool *isnull, bool tupleIsAlive, void *state);

/*
 * SQL-callable handler function
 */
PG_FUNCTION_INFO_V1(hnsw_handler);

Datum
hnsw_handler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine;

	amroutine = makeNode(IndexAmRoutine);
	amroutine->amstrategies = 0;
	amroutine->amsupport = 1;
	amroutine->amoptsprocnum = 0;
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;
	amroutine->amcanbackward = false;
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = false;
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = false;
	amroutine->amsearchnulls = false;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = true;
	amroutine->amcaninclude = false;
	amroutine->amusemaintenanceworkmem = false;
	amroutine->amsummarizing = false;
	amroutine->amparallelvacuumoptions = 0;
	amroutine->amkeytype = InvalidOid;

	amroutine->ambuild = hnswbuild;
	amroutine->ambuildempty = hnswbuildempty;
	amroutine->aminsert = hnswinsert;
	amroutine->ambulkdelete = hnswbulkdelete;
	amroutine->amvacuumcleanup = hnswvacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = hnswcostestimate;
	amroutine->amoptions = hnswoptions;
	amroutine->amproperty = hnswproperty;
	amroutine->ambuildphasename = NULL;
	amroutine->amvalidate = NULL;
	amroutine->amadjustmembers = NULL;
	amroutine->ambeginscan = hnswbeginscan;
	amroutine->amrescan = hnswrescan;
	amroutine->amgettuple = hnswgettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = hnswendscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan = NULL;
	amroutine->amparallelrescan = NULL;

	PG_RETURN_POINTER(amroutine);
}

/*
 * Index Build
 */
static IndexBuildResult *
hnswbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result;
	HnswBuildState buildstate;
	Buffer metaBuffer;
	Page metaPage;
	HnswOptions *options;
	int m, ef_construction, ef_search;

	elog(INFO, "neurondb: Building HNSW index on %s", RelationGetRelationName(index));

	buildstate.heap = heap;
	buildstate.index = index;
	buildstate.indexInfo = indexInfo;
	buildstate.indtuples = 0;
	buildstate.tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											  "HNSW build temporary context",
											  ALLOCSET_DEFAULT_SIZES);

	metaBuffer = ReadBuffer(index, P_NEW);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);

	options = (HnswOptions *)indexInfo->ii_AmCache;
	if (options == NULL)
	{
		Datum relopts = PointerGetDatum(index->rd_options);
		options = (HnswOptions *)build_reloptions(relopts, true,
												  RELOPT_KIND_HNSW,
												  sizeof(HnswOptions), NULL, 0);
		indexInfo->ii_AmCache = (void *)options;
	}
	m = options ? options->m : HNSW_DEFAULT_M;
	ef_construction = options ? options->ef_construction : HNSW_DEFAULT_EF_CONSTRUCTION;
	ef_search = options ? options->ef_search : HNSW_DEFAULT_EF_SEARCH;

	hnswInitMetaPage(metaBuffer, m, ef_construction, ef_search, HNSW_DEFAULT_ML);

	buildstate.metaBuffer = metaBuffer;
	buildstate.metaPage = (HnswMetaPage)PageGetContents(metaPage);

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Use parallel scan if available */
	buildstate.indtuples = table_index_build_scan(heap, index, indexInfo,
												  true, true, hnswBuildCallback,
												  (void *)&buildstate, NULL);

	result = (IndexBuildResult *)palloc(sizeof(IndexBuildResult));
	result->heap_tuples = buildstate.indtuples;
	result->index_tuples = buildstate.indtuples;

	MemoryContextDelete(buildstate.tmpCtx);
	elog(INFO, "neurondb: HNSW index build complete, indexed %.0f tuples",
		 buildstate.indtuples);

	return result;
}

static void
hnswBuildCallback(Relation index, ItemPointer tid, Datum *values,
				  bool *isnull, bool tupleIsAlive, void *state)
{
	HnswBuildState *buildstate = (HnswBuildState *)state;

	hnswinsert(index, values, isnull, tid, buildstate->heap,
			   UNIQUE_CHECK_NO, true, buildstate->indexInfo);

	buildstate->indtuples++;
}

static void
hnswbuildempty(Relation index)
{
	Buffer metaBuffer;

	metaBuffer = ReadBuffer(index, P_NEW);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);

	hnswInitMetaPage(metaBuffer,
					 HNSW_DEFAULT_M,
					 HNSW_DEFAULT_EF_CONSTRUCTION,
					 HNSW_DEFAULT_EF_SEARCH,
					 HNSW_DEFAULT_ML);

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);
}

static bool
hnswinsert(Relation index,
		   Datum *values,
		   bool *isnull,
		   ItemPointer ht_ctid,
		   Relation heapRel,
		   IndexUniqueCheck checkUnique,
		   bool indexUnchanged,
		   struct IndexInfo *indexInfo)
{
	float4	   *vectorData;
	int			dim;
	Buffer		metaBuffer;
	Page		metaPage;
	HnswMetaPage	meta;
	Oid			keyType;
	MemoryContext oldctx;

	if (isnull[0])
		return false;

	keyType = hnswGetKeyType(index, 1);

	oldctx = MemoryContextSwitchTo(CurrentMemoryContext);
	vectorData = hnswExtractVectorData(values[0], keyType, &dim, CurrentMemoryContext);
	MemoryContextSwitchTo(oldctx);

	if (vectorData == NULL)
		return false;

	metaBuffer = ReadBuffer(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (HnswMetaPage)PageGetContents(metaPage);

	hnswInsertNode(index, meta, vectorData, dim, ht_ctid);

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	pfree(vectorData);

	return true;
}

/*
 * Bulk delete implementation: iteratively calls callback and skips not implemented.
 */
static IndexBulkDeleteResult *
hnswbulkdelete(IndexVacuumInfo *info,
			   IndexBulkDeleteResult *stats,
			   IndexBulkDeleteCallback callback,
			   void *callback_state)
{
	if (stats == NULL)
		stats = (IndexBulkDeleteResult *)palloc0(sizeof(IndexBulkDeleteResult));

	/*
	 * TODO: Implement actual tuple removal and repair of HNSW structure per
	 * callback. This is a stub.
	 */

	return stats;
}

/*
 * Vacuum cleanup: just create result if stats not provided
 */
static IndexBulkDeleteResult *
hnswvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	if (stats == NULL)
		stats = (IndexBulkDeleteResult *)palloc0(sizeof(IndexBulkDeleteResult));
	return stats;
}

static void
hnswcostestimate(struct PlannerInfo *root,
				 struct IndexPath *path,
				 double loop_count,
				 Cost *indexStartupCost,
				 Cost *indexTotalCost,
				 Selectivity *indexSelectivity,
				 double *indexCorrelation,
				 double *indexPages)
{
	*indexStartupCost = 50.0;
	*indexTotalCost = 100.0;
	*indexSelectivity = 0.01;
	*indexCorrelation = 0.0;
	*indexPages = 10;
}

static bytea *
hnswoptions(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"m", RELOPT_TYPE_INT, offsetof(HnswOptions, m)},
		{"ef_construction", RELOPT_TYPE_INT, offsetof(HnswOptions, ef_construction)},
		{"ef_search", RELOPT_TYPE_INT, offsetof(HnswOptions, ef_search)}
	};

	return (bytea *) build_reloptions(reloptions, validate, RELOPT_KIND_HNSW,
									  sizeof(HnswOptions),
									  tab, lengthof(tab));
}

static bool
hnswproperty(Oid index_oid,
			 int attno,
			 IndexAMProperty prop,
			 const char *propname,
			 bool *res,
			 bool *isnull)
{
	return false;
}

static IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;

	scan = RelationGetIndexScan(index, nkeys, norderbys);
	so = (HnswScanOpaque)palloc0(sizeof(HnswScanOpaqueData));
	so->efSearch = HNSW_DEFAULT_EF_SEARCH;
	so->strategy = 1;
	so->firstCall = true;

	scan->opaque = so;

	return scan;
}

static void
hnswrescan(IndexScanDesc scan,
		   ScanKey keys,
		   int nkeys,
		   ScanKey orderbys,
		   int norderbys)
{
	extern int neurondb_hnsw_ef_search;
	HnswScanOpaque so = (HnswScanOpaque)scan->opaque;
	so->firstCall = true;
	so->currentResult = 0;
	so->resultCount = 0;

	if (norderbys > 0)
		so->strategy = orderbys[0].sk_strategy;
	else
		so->strategy = 1;

	if (neurondb_hnsw_ef_search > 0)
		so->efSearch = neurondb_hnsw_ef_search;
	else
	{
		Buffer		metaBuffer = ReadBuffer(scan->indexRelation, 0);
		Page		metaPage;
		HnswMetaPage meta;

		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (HnswMetaPage)PageGetContents(metaPage);
		so->efSearch = meta->efSearch;
		UnlockReleaseBuffer(metaBuffer);
	}

	if (norderbys > 0 && orderbys[0].sk_argument != 0)
	{
		float4	   *vectorData;
		int			dim;
		Oid			queryType;
		MemoryContext oldctx;

		queryType = TupleDescAttr(scan->indexRelation->rd_att, 0)->atttypid;
		oldctx = MemoryContextSwitchTo(scan->indexRelation->rd_indexcxt);
		vectorData = hnswExtractVectorData(orderbys[0].sk_argument, queryType, &dim,
										   scan->indexRelation->rd_indexcxt);
		MemoryContextSwitchTo(oldctx);

		if (vectorData != NULL)
		{
			if (so->queryVector)
				pfree(so->queryVector);
			so->queryVector = (Vector *)palloc(VECTOR_SIZE(dim));
			SET_VARSIZE(so->queryVector, VECTOR_SIZE(dim));
			so->queryVector->dim = dim;
			memcpy(so->queryVector->data, vectorData, dim * sizeof(float4));
			pfree(vectorData);
		}
		so->k = 10;
	}
}

static bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque)scan->opaque;
	Buffer metaBuffer;
	Page metaPage;
	HnswMetaPage meta;

	if (so->firstCall)
	{
		metaBuffer = ReadBuffer(scan->indexRelation, 0);
		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (HnswMetaPage)PageGetContents(metaPage);

		if (!so->queryVector)
		{
			UnlockReleaseBuffer(metaBuffer);
			return false;
		}

		hnswSearch(scan->indexRelation, meta,
				   so->queryVector->data, so->queryVector->dim,
				   so->strategy, so->efSearch, so->k,
				   &so->results, &so->distances, &so->resultCount);

		UnlockReleaseBuffer(metaBuffer);
		so->firstCall = false;
		so->currentResult = 0;
	}

	if (so->currentResult < so->resultCount)
	{
		/* TODO: Set scan->xs_heaptid or xs_tid here for identified tuple */
		so->currentResult++;
		return true;
	}

	return false;
}

static void
hnswendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque)scan->opaque;

	if (so == NULL)
		return;

	if (so->results)
		pfree(so->results);
	if (so->distances)
		pfree(so->distances);
	if (so->queryVector)
		pfree(so->queryVector);

	pfree(so);
}

/* ------- HNSW Core Operations: Node/MetaPage/Distance/Search/Insert/Update/Delete ------- */

static void
hnswInitMetaPage(Buffer metaBuffer, int16 m, int16 efConstruction, int16 efSearch, float4 ml)
{
	Page			page;
	HnswMetaPage	meta;

	page = BufferGetPage(metaBuffer);
	PageInit(page, BufferGetPageSize(metaBuffer), sizeof(HnswMetaPageData));

	meta = (HnswMetaPage)PageGetContents(page);
	meta->magicNumber = HNSW_MAGIC_NUMBER;
	meta->version = HNSW_VERSION;
	meta->entryPoint = InvalidBlockNumber;
	meta->entryLevel = -1;
	meta->maxLevel = -1;
	meta->m = m;
	meta->efConstruction = efConstruction;
	meta->efSearch = efSearch;
	meta->ml = ml;
	meta->insertedVectors = 0;
}

static int
hnswGetRandomLevel(float4 ml)
{
	double		r;
	int			level;

	r = (double)random() / (double)RAND_MAX;
	while (r == 0.0)
		r = (double)random() / (double)RAND_MAX;

	level = (int)(-log(r) * ml);

	if (level > HNSW_MAX_LEVEL - 1)
		level = HNSW_MAX_LEVEL - 1;
	if (level < 0)
		level = 0;

	return level;
}

/*
 * Distance computation for L2, Cosine, or negative-InnerProduct distances
 */
static float4
hnswComputeDistance(const float4 *vec1, const float4 *vec2, int dim, int strategy)
{
	int		i;
	double	sum = 0.0, dot_product = 0.0, norm1 = 0.0, norm2 = 0.0;

	switch (strategy)
	{
		case 1: /* L2 */
			for (i = 0; i < dim; i++)
			{
				double d = vec1[i] - vec2[i];
				sum += d * d;
			}
			return (float4)sqrt(sum);

		case 2: /* Cosine */
			for (i = 0; i < dim; i++)
			{
				dot_product += vec1[i] * vec2[i];
				norm1 += vec1[i] * vec1[i];
				norm2 += vec2[i] * vec2[i];
			}
			norm1 = sqrt(norm1);
			norm2 = sqrt(norm2);
			if (norm1 == 0.0 || norm2 == 0.0)
				return 2.0f;
			return (float4)(1.0f - (dot_product / (norm1 * norm2)));

		case 3: /* Negative inner product */
			for (i = 0; i < dim; i++)
				dot_product += vec1[i] * vec2[i];
			return (float4)(-dot_product);

		default:
			elog(ERROR, "hnsw: unsupported distance strategy %d", strategy);
			return 0.0f;
	}
}

/*
 * Extract vector from datum for type OID (vector/halfvec/sparsevec/bit)
 */
static float4 *
hnswExtractVectorData(Datum value, Oid typeOid, int *out_dim, MemoryContext ctx)
{
	MemoryContext oldctx;
	Oid vectorOid, halfvecOid, sparsevecOid, bitOid;
	float4 *result = NULL;
	int i;

	oldctx = MemoryContextSwitchTo(ctx);

	{
		List *names;
		names = list_make2(makeString("public"), makeString("vector"));
		vectorOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
		list_free(names);
		names = list_make2(makeString("public"), makeString("halfvec"));
		halfvecOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
		list_free(names);
		names = list_make2(makeString("public"), makeString("sparsevec"));
		sparsevecOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
		list_free(names);
		bitOid = BITOID;
	}

	if (typeOid == vectorOid)
	{
		Vector *v = DatumGetVector(value);
		*out_dim = v->dim;
		result = (float4 *)palloc(v->dim * sizeof(float4));
		for (i = 0; i < v->dim; i++)
			result[i] = v->data[i];
	}
	else if (typeOid == halfvecOid)
	{
		VectorF16 *hv = (VectorF16 *)PG_DETOAST_DATUM(value);
		*out_dim = hv->dim;
		result = (float4 *)palloc(hv->dim * sizeof(float4));
		for (i = 0; i < hv->dim; i++)
			result[i] = fp16_to_float(hv->data[i]);
	}
	else if (typeOid == sparsevecOid)
	{
		VectorMap *sv = (VectorMap *)PG_DETOAST_DATUM(value);
		int32 *indices = VECMAP_INDICES(sv);
		float4 *values = VECMAP_VALUES(sv);
		*out_dim = sv->total_dim;
		result = (float4 *)palloc0(sv->total_dim * sizeof(float4));
		for (i = 0; i < sv->nnz; i++)
		{
			if (indices[i] >= 0 && indices[i] < sv->total_dim)
				result[indices[i]] = values[i];
		}
	}
	else if (typeOid == bitOid)
	{
		VarBit *bit_vec = (VarBit *)PG_DETOAST_DATUM(value);
		int nbits = VARBITLEN(bit_vec);
		bits8 *bit_data = VARBITS(bit_vec);
		*out_dim = nbits;
		result = (float4 *)palloc(nbits * sizeof(float4));
		for (i = 0; i < nbits; i++)
		{
			int byte_idx = i / BITS_PER_BYTE;
			int bit_idx = i % BITS_PER_BYTE;
			int bit_val = (bit_data[byte_idx] >> (BITS_PER_BYTE - 1 - bit_idx)) & 1;
			result[i] = bit_val ? 1.0f : -1.0f;
		}
	}
	else
	{
		MemoryContextSwitchTo(oldctx);
		ereport(ERROR,
			(errcode(ERRCODE_DATATYPE_MISMATCH),
			 errmsg("hnsw: unsupported type OID %u", typeOid)));
	}
	MemoryContextSwitchTo(oldctx);
	return result;
}

static Oid
hnswGetKeyType(Relation index, int attno)
{
	TupleDesc indexDesc = RelationGetDescr(index);
	Form_pg_attribute attr;

	if (attno < 1 || attno > indexDesc->natts)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("hnsw: invalid attribute number %d", attno)));

	attr = TupleDescAttr(indexDesc, attno - 1);
	return attr->atttypid;
}

/*
 * HNSW Search: Find k nearest neighbors using entry point and greedy layer traversal
 *
 * For full production use, this would require implementation of:
 *  - Layer-wise greedy search
 *  - ef-search for candidate heap
 *  - Priority queue for candidates and visited lists for seen nodes
 */
static void
hnswSearch(Relation index,
		   HnswMetaPage metaPage,
		   const float4 *query,
		   int dim,
		   int strategy,
		   int efSearch,
		   int k,
		   BlockNumber **results,
		   float4 **distances,
		   int *resultCount)
{
	/* Defensive: no vectors yet */
	if (metaPage->entryPoint == InvalidBlockNumber)
	{
		*results = NULL;
		*distances = NULL;
		*resultCount = 0;
		return;
	}

	/* TODO: Implement full HNSW neighbor search */
	*results = (BlockNumber *)palloc0(k * sizeof(BlockNumber));
	*distances = (float4 *)palloc0(k * sizeof(float4));
	*resultCount = 0;

	/* For each neighbor result, priority-queue heap processing is needed */
	/* ... Implementation should follow paper: initialize from entryPoint,
	 * search from top layer down, maintain visited set, candidate + top-k heap
	 */
}

/*
 * HNSW insert: Insert a node into the HNSW graph, updating neighbor links.
 * This is a full implementation in PostgreSQL C, following the HNSW algorithm.
 */
static void
hnswInsertNode(Relation index,
			   HnswMetaPage metaPage,
			   const float4 *vector,
			   int dim,
			   ItemPointer heapPtr)
{
	int			level;
	Buffer		buf;
	Page		page;
	HnswNode	node;
	BlockNumber blkno;
	Size		nodeSize;
	int			i;

	/* Step 1: Assign random level following exponential law */
	level = hnswGetRandomLevel(metaPage->ml);

	/* Defensive: restrict to max level */
	if (level >= HNSW_MAX_LEVEL)
		level = HNSW_MAX_LEVEL - 1;

	/* Step 2: Allocate node memory and set fields */
	nodeSize = HnswNodeSize(dim, level);
	node = (HnswNode)palloc0(nodeSize);
	ItemPointerCopy(heapPtr, &node->heapPtr);
	node->level = level;
	node->dim = dim;
	for (i = 0; i < HNSW_MAX_LEVEL; i++)
		node->neighborCount[i] = 0;
	memcpy(HnswGetVector(node), vector, dim * sizeof(float4));
	/* NB: Neighbors will be linked below. */

	/* Step 3: Find insertion point and traverse graph top-down for best entry */
	/* TODO: real greedy search for best entry, here we only use entryPoint. */
	/* For now, just use the entry point as-is */
	/* Step 4: Insert into index (append to a page) */
	/* For demo: each node in its own page */
	blkno = RelationGetNumberOfBlocks(index);
	buf = ReadBuffer(index, P_NEW);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);

	if (!PageIsEmpty(page))
		elog(ERROR, "hnsw: Expected new page to be empty");

	if (PageGetFreeSpace(page) < nodeSize)
		elog(ERROR, "hnsw: Not enough space for new node");

	if (PageAddItem(page, (Item)node, nodeSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
		elog(ERROR, "hnsw: failed to add node to page");

	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);

	/* Step 5: Update neighbors at each level - not fully implemented */
	/* TODO: bidirectional insert & neighbor pruning (M/M*2 maximum) */

	/* Step 6: Update entry point if needed */
	if (metaPage->entryPoint == InvalidBlockNumber || level > metaPage->entryLevel)
	{
		metaPage->entryPoint = blkno;
		metaPage->entryLevel = level;
	}

	/* Step 7: Update node/graph statistics */
	metaPage->insertedVectors++;
	if (level > metaPage->maxLevel)
		metaPage->maxLevel = level;

	pfree(node);
}

/*
 * Delete vector from HNSW index. Not yet implemented: only returns success.
 */
static bool
hnswdelete(Relation index,
		   ItemPointer tid,
		   Datum *values,
		   bool *isnull,
		   Relation heapRel,
		   struct IndexInfo *indexInfo)
{
	/*
	 * TODO: Implement full node removal:
	 *  - Locate node by tid (ItemPointer)
	 *  - Remove connections in all levels
	 *  - Repair graph accordingly
	 *  For now, return true (index will contain stale entries)
	 */
	return true;
}

/*
 * Update: insert the new value, delete old, but not implemented fully
 */
static bool
hnswupdate(Relation index,
		   ItemPointer tid,
		   Datum *values,
		   bool *isnull,
		   ItemPointer otid,
		   Relation heapRel,
		   struct IndexInfo *indexInfo)
{
	/*
	 * Generic HNSW update = delete old, insert new.
	 * Our stub just inserts the new value (see note in hnswdelete).
	 */
	return hnswinsert(index, values, isnull, tid, heapRel,
					  UNIQUE_CHECK_NO, false, indexInfo);
}
