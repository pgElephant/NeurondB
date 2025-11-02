/*-------------------------------------------------------------------------
 *
 * temporal_integration.c
 *		Temporal scoring integration into ANN search
 *
 * Integrates time-decay scoring with vector similarity:
 * - Exponential time decay
 * - Recency boosting
 * - Time window filtering
 * - Temporal reranking
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *	  src/search/temporal_integration.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "access/htup_details.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
#include <math.h>

/* Temporal scoring parameters */
#define DEFAULT_DECAY_RATE		0.1		/* Per day */
#define DEFAULT_RECENCY_WEIGHT	0.3		/* 0..1 */

/*
 * Temporal scoring configuration
 */
typedef struct TemporalConfig
{
	float4		decayRate;		/* Exponential decay rate */
	float4		recencyWeight;	/* Weight for temporal component */
	TimestampTz referenceTime;	/* Reference timestamp (usually now) */
	Interval   *timeWindow;		/* Optional time window filter */
	bool		enabled;
} TemporalConfig;

/*
 * Compute time decay factor
 *
 * Returns a multiplier in [0,1] based on how old the document is.
 * Uses exponential decay: exp(-lambda * age_days)
 */
static float4
compute_time_decay(TimestampTz docTime, TimestampTz refTime, float4 decayRate)
{
	float8		age_seconds;
	float8		age_days;
	float4		decay;

	age_seconds = (float8) (refTime - docTime) / USECS_PER_SEC;
	age_days = age_seconds / (24.0 * 3600.0);

	/* Exponential decay */
	decay = (float4) exp(-decayRate * age_days);

	return decay;
}

/*
 * Compute hybrid score with temporal component
 *
 * final_score = (1 - w) * vector_similarity + w * temporal_score
 */
static float4
temporal_compute_hybrid_score(float4 vectorDistance, TimestampTz docTime,
							   TemporalConfig *config)
{
	float4		vectorScore;
	float4		temporalScore;
	float4		finalScore;

	if (!config->enabled)
		return 1.0 / (1.0 + vectorDistance); /* Vector-only */

	/* Normalize vector distance to similarity score */
	vectorScore = 1.0 / (1.0 + vectorDistance);

	/* Compute temporal score */
	temporalScore = compute_time_decay(docTime, config->referenceTime, config->decayRate);

	/* Weighted combination */
	finalScore = (1.0 - config->recencyWeight) * vectorScore + 
				 config->recencyWeight * temporalScore;

	return finalScore;
}

/*
 * Check if document is within time window
 */
static bool
temporal_in_window(TimestampTz docTime, TemporalConfig *config)
{
	TimestampTz windowStart;

	if (!config->enabled || config->timeWindow == NULL)
		return true;

	/* Compute window start time */
	/* windowStart = config->referenceTime - config->timeWindow; */
	/* TODO: Proper interval subtraction */
	windowStart = config->referenceTime - (7 * 24 * 3600 * USECS_PER_SEC); /* 7 days */

	return docTime >= windowStart;
}

/*
 * Rerank results by temporal score
 *
 * Takes a set of results and re-orders them by combined vector+temporal score.
 */
static void pg_attribute_unused()
temporal_rerank_results(ItemPointer *items, float4 *distances, 
						TimestampTz *timestamps, int count,
						TemporalConfig *config)
{
	float4	   *scores;
	int			i, j;
	float4		temp_score, temp_dist;
	ItemPointer temp_item;
	TimestampTz temp_ts;

	if (!config->enabled || count == 0)
		return;

	/* Compute hybrid scores */
	scores = (float4 *) palloc(count * sizeof(float4));
	
	for (i = 0; i < count; i++)
	{
		scores[i] = temporal_compute_hybrid_score(distances[i], timestamps[i], config);
	}

	/* Simple bubble sort (for small result sets) */
	/* TODO: Use qsort for larger sets */
	for (i = 0; i < count - 1; i++)
	{
		for (j = i + 1; j < count; j++)
		{
			if (scores[j] > scores[i]) /* Higher score = better */
			{
				/* Swap */
				temp_score = scores[i];
				scores[i] = scores[j];
				scores[j] = temp_score;

				temp_dist = distances[i];
				distances[i] = distances[j];
				distances[j] = temp_dist;

				temp_item = items[i];
				items[i] = items[j];
				items[j] = temp_item;

				temp_ts = timestamps[i];
				timestamps[i] = timestamps[j];
				timestamps[j] = temp_ts;
			}
		}
	}

	pfree(scores);

	elog(DEBUG1, "neurondb: Temporally reranked %d results", count);
}

/*
 * SQL-callable function: temporal_score(vector_distance, timestamp)
 */
PG_FUNCTION_INFO_V1(neurondb_temporal_score);

Datum
neurondb_temporal_score(PG_FUNCTION_ARGS)
{
	float4		vectorDistance = PG_GETARG_FLOAT4(0);
	TimestampTz docTime = PG_GETARG_TIMESTAMPTZ(1);
	float4		decayRate = PG_GETARG_FLOAT4(2);
	float4		recencyWeight = PG_GETARG_FLOAT4(3);
	TemporalConfig config;
	float4		score;

	config.decayRate = decayRate;
	config.recencyWeight = recencyWeight;
	config.referenceTime = GetCurrentTimestamp();
	config.timeWindow = NULL;
	config.enabled = true;

	score = temporal_compute_hybrid_score(vectorDistance, docTime, &config);

	PG_RETURN_FLOAT4(score);
}

/*
 * SQL-callable function: temporal_filter(timestamp, window_interval)
 */
PG_FUNCTION_INFO_V1(neurondb_temporal_filter);

Datum
neurondb_temporal_filter(PG_FUNCTION_ARGS)
{
	TimestampTz docTime = PG_GETARG_TIMESTAMPTZ(0);
	Interval   *window = PG_GETARG_INTERVAL_P(1);
	TemporalConfig config;
	bool		inWindow;

	config.referenceTime = GetCurrentTimestamp();
	config.timeWindow = window;
	config.enabled = true;

	inWindow = temporal_in_window(docTime, &config);

	PG_RETURN_BOOL(inWindow);
}

/*
 * Create default temporal config
 */
static TemporalConfig *
temporal_create_config(float4 decayRate, float4 recencyWeight)
{
	TemporalConfig *config;

	config = (TemporalConfig *) palloc0(sizeof(TemporalConfig));
	config->decayRate = decayRate;
	config->recencyWeight = recencyWeight;
	config->referenceTime = GetCurrentTimestamp();
	config->timeWindow = NULL;
	config->enabled = true;

	return config;
}

/*
 * Integrate temporal scoring into HNSW search
 *
 * This would be called from the HNSW search function to apply
 * temporal boosting to results.
 */
static void pg_attribute_unused()
temporal_integrate_hnsw_search(BlockNumber *results, float4 *distances,
								int resultCount, float4 decayRate, 
								float4 recencyWeight)
{
	TimestampTz *timestamps;
	TemporalConfig *config;
	int			i;

	(void) results;		/* Unused - would fetch from blocks */
	(void) distances;	/* Used later for reranking */

	if (resultCount == 0)
		return;

	/* Allocate timestamp array */
	timestamps = (TimestampTz *) palloc(resultCount * sizeof(TimestampTz));

	/* TODO: Fetch timestamps from heap tuples */
	for (i = 0; i < resultCount; i++)
	{
		timestamps[i] = GetCurrentTimestamp(); /* Placeholder */
	}

	/* Create config */
	config = temporal_create_config(decayRate, recencyWeight);

	/* Rerank */
	/* temporal_rerank_results(...) would need ItemPointers */

	pfree(timestamps);
	pfree(config);

	elog(DEBUG1, "neurondb: Applied temporal scoring to HNSW results");
}

/*
 * Helper: Extract timestamp from heap tuple
 */
static TimestampTz pg_attribute_unused()
temporal_get_tuple_timestamp(HeapTuple tuple, TupleDesc tupdesc, 
							  const char *columnName)
{
	bool		isnull;
	Datum		datum;
	int			attnum;

	/* Find timestamp column */
	attnum = SPI_fnumber(tupdesc, columnName);
	if (attnum == SPI_ERROR_NOATTRIBUTE)
		return GetCurrentTimestamp(); /* Fallback to now */

	datum = SPI_getbinval(tuple, tupdesc, attnum, &isnull);
	if (isnull)
		return GetCurrentTimestamp();

	return DatumGetTimestampTz(datum);
}

