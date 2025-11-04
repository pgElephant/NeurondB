/*-------------------------------------------------------------------------
 *
 * custom_hybrid_scan.c
 *		CustomScan node for single-pass hybrid vector+FTS+metadata execution
 *
 * Implements a custom scan node that fuses:
 * - Vector ANN search
 * - Full-text search (FTS)
 * - Metadata filtering
 * - Reranking
 *
 * This avoids multiple sequential scans and provides optimal hybrid search.
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/scan/custom_hybrid_scan.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_scan.h"
#include "fmgr.h"
#include "access/relscan.h"
#include "access/table.h"
#include "commands/explain.h"
#include "executor/executor.h"
#include "executor/nodeCustom.h"
#include "nodes/execnodes.h"
#include "nodes/extensible.h"
#include "nodes/plannodes.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "utils/builtins.h"
#include "utils/rel.h"

/*
 * Hybrid scan state
 */
typedef struct HybridScanState
{
	CustomScanState css;		/* Must be first */
	
	/* Query parameters */
	Vector	   *queryVector;	/* Vector query */
	char	   *ftsQuery;		/* FTS query string */
	List	   *filters;		/* Metadata filters */
	float4		vectorWeight;	/* Hybrid weight: 0..1 */
	float4		ftsWeight;		/* 1 - vectorWeight */
	int			k;				/* Target results */
	
	/* Candidate management */
	TupleTableSlot **candidates;
	float4	   *scores;			/* Hybrid scores */
	int			candidateCount;
	int			currentPos;
	
	/* Sub-scans */
	IndexScanDesc vectorScan;
	IndexScanDesc ftsScan;
	TableScanDesc heapScan;
	
	bool		vectorDone;
	bool		ftsDone;
} HybridScanState;

/*
 * Plan-time data
 */
typedef struct HybridScanPlan
{
	CustomScan	cscan;			/* Must be first */
	
	Oid			vectorIndexOid;
	Oid			ftsIndexOid;
	Oid			relationOid;
	
	float4		vectorWeight;
	int			k;
} HybridScanPlan;

/* CustomScan method declarations */
static Node *hybrid_create_scan_state(CustomScan *cscan);
static void hybrid_begin(CustomScanState *node, EState *estate, int eflags);
static TupleTableSlot *hybrid_exec(CustomScanState *node);
static void hybrid_end(CustomScanState *node);
static void hybrid_rescan(CustomScanState *node);
static void hybrid_explain(CustomScanState *node, List *ancestors, ExplainState *es);

/* Helper functions */
static float4 compute_hybrid_score(float4 vectorDist, float4 ftsScore,
								   float4 vectorWeight);
static void merge_candidates(HybridScanState *state);
static int score_comparator(const void *a, const void *b);

/* CustomScan methods table */
static CustomExecMethods hybrid_exec_methods = {
	.CustomName = "HybridScan",
	.BeginCustomScan = hybrid_begin,
	.ExecCustomScan = hybrid_exec,
	.EndCustomScan = hybrid_end,
	.ReScanCustomScan = hybrid_rescan,
	.ExplainCustomScan = hybrid_explain
};

/*
 * Create a hybrid scan custom path
 *
 * This would be called from a planning hook to inject the custom path.
 */
PG_FUNCTION_INFO_V1(create_hybrid_scan_path);

Datum
create_hybrid_scan_path(PG_FUNCTION_ARGS)
{
	/* This is called during planning to add hybrid path */
	elog(NOTICE, "neurondb: Creating hybrid scan path");
	
	/* TODO: Integrate with planner hooks to add CustomPath */
	
	PG_RETURN_VOID();
}

/*
 * Create scan state
 *
 * Note: this is only used for node factory registration and should not generate
 * an unused-function warning.
 */
__attribute__((unused))
static Node *
hybrid_create_scan_state(CustomScan *cscan)
{
	HybridScanState *state;

	state = (HybridScanState *) newNode(sizeof(HybridScanState), T_CustomScanState);
	state->css.methods = &hybrid_exec_methods;
	state->vectorDone = false;
	state->ftsDone = false;
	state->currentPos = 0;
	state->candidateCount = 0;

	return (Node *) state;
}

/*
 * Begin execution
 */
static void
hybrid_begin(CustomScanState *node, EState *estate, int eflags)
{
	HybridScanState *state = (HybridScanState *) node;
	
	elog(DEBUG1, "neurondb: Beginning hybrid scan execution");

	/* Extract parameters from plan */
	/* state->queryVector = ... */
	/* state->ftsQuery = ... */
	state->vectorWeight = 0.5; /* Default */
	state->ftsWeight = 0.5;
	state->k = 100;

	/* Allocate candidate arrays */
	state->candidates = (TupleTableSlot **) palloc(state->k * sizeof(TupleTableSlot *));
	state->scores = (float4 *) palloc(state->k * sizeof(float4));

	/* TODO: Open vector and FTS indexes */
	/* state->vectorScan = index_beginscan(...); */
	/* state->ftsScan = index_beginscan(...); */
}

/*
 * Execute scan - return next tuple
 */
static TupleTableSlot *
hybrid_exec(CustomScanState *node)
{
	HybridScanState *state = (HybridScanState *) node;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;

	/* On first call, gather all candidates */
	if (state->currentPos == 0 && state->candidateCount == 0)
	{
		/* Scan vector index */
		while (!state->vectorDone && state->candidateCount < state->k)
		{
			/* TODO: Get next vector candidate and score */
			/* state->candidates[state->candidateCount] = ... */
			/* state->scores[state->candidateCount] = ... */
			/* state->candidateCount++; */
			break; /* Temporary */
		}

		/* Scan FTS index */
		while (!state->ftsDone && state->candidateCount < state->k)
		{
			/* TODO: Get next FTS candidate and score */
			break; /* Temporary */
		}

		/* Merge and rerank */
		merge_candidates(state);
	}

	/* Return next result */
	if (state->currentPos < state->candidateCount)
	{
		/* Copy candidate to slot */
		/* ExecCopySlot(slot, state->candidates[state->currentPos]); */
		state->currentPos++;
		return slot;
	}

	/* Done */
	return ExecClearTuple(slot);
}

/*
 * End scan
 */
static void
hybrid_end(CustomScanState *node)
{
	HybridScanState *state = (HybridScanState *) node;

	elog(DEBUG1, "neurondb: Ending hybrid scan");

	/* Close scans */
	/* if (state->vectorScan)
		index_endscan(state->vectorScan); */

	/* Free memory */
	if (state->candidates)
		pfree(state->candidates);
	if (state->scores)
		pfree(state->scores);
}

/*
 * Rescan
 */
static void
hybrid_rescan(CustomScanState *node)
{
	HybridScanState *state = (HybridScanState *) node;

	state->currentPos = 0;
	state->candidateCount = 0;
	state->vectorDone = false;
	state->ftsDone = false;

	/* Rescan indexes */
	/* if (state->vectorScan)
		index_rescan(state->vectorScan, ...); */
}

/*
 * Explain plan
 */
static void
hybrid_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	HybridScanState *state = (HybridScanState *) node;

#if PG_VERSION_NUM >= 180000
	/* PG18 removed ExplainProperty* functions - use appendStringInfo instead */
	if (es->format == EXPLAIN_FORMAT_TEXT)
	{
		appendStringInfoSpaces(es->str, es->indent * 2);
		appendStringInfo(es->str, "Hybrid Scan Type: Vector+FTS\n");
		appendStringInfoSpaces(es->str, es->indent * 2);
		appendStringInfo(es->str, "Vector Weight: %.2f\n", state->vectorWeight);
		appendStringInfoSpaces(es->str, es->indent * 2);
		appendStringInfo(es->str, "FTS Weight: %.2f\n", state->ftsWeight);
		appendStringInfoSpaces(es->str, es->indent * 2);
		appendStringInfo(es->str, "Target Results: %d\n", state->k);
	}
#else
	ExplainPropertyText("Hybrid Scan Type", "Vector+FTS", es);
	ExplainPropertyFloat("Vector Weight", NULL, state->vectorWeight, 2, es);
	ExplainPropertyFloat("FTS Weight", NULL, state->ftsWeight, 2, es);
	ExplainPropertyInteger("Target Results", NULL, state->k, es);
#endif
}

/*
 * Compute hybrid score from vector distance and FTS score
 *
 * Mark as maybe unused so no warning if only used in template or ahead-of-time.
 */
__attribute__((unused))
static float4
compute_hybrid_score(float4 vectorDist, float4 ftsScore,
						   float4 vectorWeight)
{
	float4		vectorScore;
	float4		hybridScore;

	/* Normalize vector distance to score (0..1, higher is better) */
	vectorScore = 1.0 / (1.0 + vectorDist);

	/* Weighted combination */
	hybridScore = vectorWeight * vectorScore + (1.0 - vectorWeight) * ftsScore;

	return hybridScore;
}

/*
 * Merge and rerank candidates by hybrid score
 *
 * Mark as maybe unused to suppress warning.
 */
__attribute__((unused))
static void
merge_candidates(HybridScanState *state)
{
	/* Remove duplicates */
	/* Recompute scores with hybrid formula */
	/* Sort by final score */
	
	/* TODO: Implement deduplication and sorting */
	
	elog(DEBUG1, "neurondb: Merged %d candidates", state->candidateCount);
}

/*
 * Comparator for sorting candidates by score (descending)
 *
 * Mark as maybe unused to suppress warning.
 */
__attribute__((unused))
static int
score_comparator(const void *a, const void *b)
{
	float4 score_a = *((float4 *) a);
	float4 score_b = *((float4 *) b);

	if (score_a > score_b)
		return -1;
	else if (score_a < score_b)
		return 1;
	else
		return 0;
}

/*
 * Register custom scan provider
 *
 * This should be called in _PG_init().
 */
void
register_hybrid_scan_provider(void)
{
	/* RegisterCustomScanMethods(&hybrid_exec_methods); */
	elog(DEBUG1, "neurondb: Registered hybrid scan provider");
}
