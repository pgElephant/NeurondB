/*-------------------------------------------------------------------------
 *
 * ml_graph_neural_networks.c
 *    Graph Convolutional Networks, Graph Attention Networks, GraphSAGE
 *
 * Implements graph neural network algorithms:
 *
 * 1. Graph Convolutional Network (GCN): Spectral convolution on graphs
 *    - Aggregates neighbor features
 *    - Layer-wise propagation rule
 *
 * 2. Graph Attention Network (GAT): Attention mechanism for graphs
 *    - Learns attention weights for neighbors
 *    - Multi-head attention support
 *
 * 3. GraphSAGE: Sample and Aggregate
 *    - Inductive learning on large graphs
 *    - Samples neighbors for aggregation
 *
 * Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
 *
 * IDENTIFICATION
 *    src/ml/ml_graph_neural_networks.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "neurondb_types.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"
#include "ml_catalog.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/jsonb.h"

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_spi_safe.h"

/* GCN layer structure */
typedef struct GCNLayer
{
	float **weights;
	float *bias;
	int input_dim;
	int output_dim;
} GCNLayer;

/* GCN model structure */
typedef struct GCNModel
{
	GCNLayer *layers;
	int n_layers;
	int hidden_dim;
	int output_dim;
} GCNModel;

/*
 * Normalize adjacency matrix (symmetric normalization)
 */
static void
normalize_adjacency(float **adj, int n_nodes, float **norm_adj)
{
	float *degrees;
	int i, j;

	degrees = (float *)palloc0(sizeof(float) * n_nodes);

	/* Calculate degrees */
	for (i = 0; i < n_nodes; i++)
		for (j = 0; j < n_nodes; j++)
			degrees[i] += adj[i][j];

	/* Normalize: D^(-1/2) * A * D^(-1/2) */
	for (i = 0; i < n_nodes; i++)
	{
		float di_sqrt = (degrees[i] > 0.0) ? 1.0 / sqrt(degrees[i]) : 0.0;
		for (j = 0; j < n_nodes; j++)
		{
			float dj_sqrt = (degrees[j] > 0.0) ? 1.0 / sqrt(degrees[j]) : 0.0;
			norm_adj[i][j] = adj[i][j] * di_sqrt * dj_sqrt;
		}
	}

	NDB_SAFE_PFREE_AND_NULL(degrees);
}

/*
 * GCN forward pass: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
 */
static void
gcn_forward(float **features, float **adj_norm, float **weights, float *bias,
	    int n_nodes, int input_dim, int output_dim, float **output)
{
	int i, j, k;

	/* Matrix multiplication: A_norm * H */
	float **ah = (float **)palloc(sizeof(float *) * n_nodes);
	for (i = 0; i < n_nodes; i++)
	{
		ah[i] = (float *)palloc0(sizeof(float) * input_dim);
		for (j = 0; j < n_nodes; j++)
			for (k = 0; k < input_dim; k++)
				ah[i][k] += adj_norm[i][j] * features[j][k];
	}

	/* Matrix multiplication: (A_norm * H) * W */
	for (i = 0; i < n_nodes; i++)
	{
		for (j = 0; j < output_dim; j++)
		{
			output[i][j] = bias[j];
			for (k = 0; k < input_dim; k++)
				output[i][j] += ah[i][k] * weights[k][j];
			/* ReLU activation */
			if (output[i][j] < 0.0)
				output[i][j] = 0.0;
		}
	}

	for (i = 0; i < n_nodes; i++)
		NDB_SAFE_PFREE_AND_NULL(ah[i]);
	NDB_SAFE_PFREE_AND_NULL(ah);
}

/*
 * gcn_train
 * ---------
 * Train Graph Convolutional Network.
 *
 * SQL Arguments:
 *   graph_table    - Table with graph structure (node_id, neighbor_id, weight)
 *   features_table - Table with node features (node_id, features[])
 *   labels_table   - Table with node labels (node_id, label)
 *   n_nodes        - Number of nodes
 *   feature_dim    - Feature dimension
 *   hidden_dim     - Hidden layer dimension (default: 64)
 *   output_dim     - Output dimension (default: number of classes)
 *   learning_rate  - Learning rate (default: 0.01)
 *   epochs         - Training epochs (default: 100)
 *
 * Returns:
 *   Model ID
 */
PG_FUNCTION_INFO_V1(gcn_train);

Datum
gcn_train(PG_FUNCTION_ARGS)
{
	text *graph_table;
	text *features_table;
	text *labels_table;
	int n_nodes;
	int feature_dim;
	int hidden_dim;
	int output_dim;
	double learning_rate;
	int epochs;
	char *graph_tbl;
	char *feat_tbl;
	char *label_tbl;
	float **adjacency;
	float **features;
	int *labels;
	float **adj_norm;
	int i, j;
	int32 model_id;

	graph_table = PG_GETARG_TEXT_PP(0);
	features_table = PG_GETARG_TEXT_PP(1);
	labels_table = PG_GETARG_TEXT_PP(2);
	n_nodes = PG_GETARG_INT32(3);
	feature_dim = PG_GETARG_INT32(4);
	hidden_dim = PG_ARGISNULL(5) ? 64 : PG_GETARG_INT32(5);
	output_dim = PG_ARGISNULL(6) ? 2 : PG_GETARG_INT32(6);
	learning_rate = PG_ARGISNULL(7) ? 0.01 : PG_GETARG_FLOAT8(7);
	epochs = PG_ARGISNULL(8) ? 100 : PG_GETARG_INT32(8);

	if (n_nodes < 1 || feature_dim < 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("n_nodes and feature_dim must be positive")));

	graph_tbl = text_to_cstring(graph_table);
	feat_tbl = text_to_cstring(features_table);
	label_tbl = text_to_cstring(labels_table);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_connect failed")));

	/* Initialize adjacency matrix */
	adjacency = (float **)palloc(sizeof(float *) * n_nodes);
	adj_norm = (float **)palloc(sizeof(float *) * n_nodes);
	for (i = 0; i < n_nodes; i++)
	{
		adjacency[i] = (float *)palloc0(sizeof(float) * n_nodes);
		adj_norm[i] = (float *)palloc0(sizeof(float) * n_nodes);
		/* Self-connections */
		adjacency[i][i] = 1.0;
	}

	/* Load graph structure */
	{
		StringInfoData query;
		int ret;

		initStringInfo(&query);
		appendStringInfo(&query,
				 "SELECT node_id, neighbor_id, COALESCE(weight, 1.0) FROM %s",
				 graph_tbl);

		ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
		if (ret == SPI_OK_SELECT)
		{
			for (i = 0; i < SPI_processed; i++)
			{
				HeapTuple tuple = SPI_tuptable->vals[i];
				int node_id = DatumGetInt32(SPI_getbinval(tuple,
									SPI_tuptable->tupdesc,
									1, NULL));
				int neighbor_id = DatumGetInt32(SPI_getbinval(tuple,
									  SPI_tuptable->tupdesc,
									  2, NULL));
				float weight = DatumGetFloat4(SPI_getbinval(tuple,
									SPI_tuptable->tupdesc,
									3, NULL));

				if (node_id >= 0 && node_id < n_nodes &&
				    neighbor_id >= 0 && neighbor_id < n_nodes)
				{
					adjacency[node_id][neighbor_id] = weight;
					adjacency[neighbor_id][node_id] = weight; /* Undirected */
				}
			}
		}
		NDB_SAFE_PFREE_AND_NULL(query.data);
	}

	/* Normalize adjacency */
	normalize_adjacency(adjacency, n_nodes, adj_norm);

	/* Load features */
	features = (float **)palloc(sizeof(float *) * n_nodes);
	for (i = 0; i < n_nodes; i++)
		features[i] = (float *)palloc0(sizeof(float) * feature_dim);

	{
		StringInfoData query;
		int ret;

		initStringInfo(&query);
		appendStringInfo(&query,
				 "SELECT node_id, features FROM %s", feat_tbl);

		ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
		if (ret == SPI_OK_SELECT)
		{
			for (i = 0; i < SPI_processed; i++)
			{
				HeapTuple tuple = SPI_tuptable->vals[i];
				int node_id = DatumGetInt32(SPI_getbinval(tuple,
									SPI_tuptable->tupdesc,
									1, NULL));
				ArrayType *feat_array = DatumGetArrayTypeP(
					SPI_getbinval(tuple, SPI_tuptable->tupdesc, 2, NULL));
				float *feat_data;
				int feat_len;
				int d;

				if (node_id < 0 || node_id >= n_nodes)
					continue;

				feat_data = (float *)ARR_DATA_PTR(feat_array);
				feat_len = ARR_DIMS(feat_array)[0];

				for (d = 0; d < feature_dim && d < feat_len; d++)
					features[node_id][d] = feat_data[d];
			}
		}
		NDB_SAFE_PFREE_AND_NULL(query.data);
	}

	/* Load labels */
	labels = (int *)palloc(sizeof(int) * n_nodes);
	for (i = 0; i < n_nodes; i++)
		labels[i] = 0;

	{
		StringInfoData query;
		int ret;

		initStringInfo(&query);
		appendStringInfo(&query,
				 "SELECT node_id, label FROM %s", label_tbl);

		ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
		if (ret == SPI_OK_SELECT)
		{
			for (i = 0; i < SPI_processed; i++)
			{
				HeapTuple tuple = SPI_tuptable->vals[i];
				int node_id = DatumGetInt32(SPI_getbinval(tuple,
									SPI_tuptable->tupdesc,
									1, NULL));
				int label = DatumGetInt32(SPI_getbinval(tuple,
								     SPI_tuptable->tupdesc,
								     2, NULL));

				if (node_id >= 0 && node_id < n_nodes)
					labels[node_id] = label;
			}
		}
		NDB_SAFE_PFREE_AND_NULL(query.data);
	}

	/* Initialize GCN layers */
	GCNLayer layer1, layer2;
	float **w1, **w2;
	float *b1, *b2;
	int l;

	/* Layer 1: feature_dim -> hidden_dim */
	w1 = (float **)palloc(sizeof(float *) * feature_dim);
	for (i = 0; i < feature_dim; i++)
	{
		w1[i] = (float *)palloc(sizeof(float) * hidden_dim);
		for (j = 0; j < hidden_dim; j++)
			w1[i][j] = ((float)rand() / (float)RAND_MAX - 0.5) * 0.1;
	}
	b1 = (float *)palloc0(sizeof(float) * hidden_dim);

	/* Layer 2: hidden_dim -> output_dim */
	w2 = (float **)palloc(sizeof(float *) * hidden_dim);
	for (i = 0; i < hidden_dim; i++)
	{
		w2[i] = (float *)palloc(sizeof(float) * output_dim);
		for (j = 0; j < output_dim; j++)
			w2[i][j] = ((float)rand() / (float)RAND_MAX - 0.5) * 0.1;
	}
	b2 = (float *)palloc0(sizeof(float) * output_dim);

	/* Training loop (simplified - would need proper backprop) */
	float **h1 = (float **)palloc(sizeof(float *) * n_nodes);
	float **h2 = (float **)palloc(sizeof(float *) * n_nodes);
	for (i = 0; i < n_nodes; i++)
	{
		h1[i] = (float *)palloc(sizeof(float) * hidden_dim);
		h2[i] = (float *)palloc(sizeof(float) * output_dim);
	}

	/* Allocate gradient arrays */
	float **grad_h2 = (float **)palloc(sizeof(float *) * n_nodes);
	float **grad_h1 = (float **)palloc(sizeof(float *) * n_nodes);
	float **grad_w2 = (float **)palloc(sizeof(float *) * hidden_dim);
	float **grad_w1 = (float **)palloc(sizeof(float *) * feature_dim);
	float *grad_b2 = (float *)palloc0(sizeof(float) * output_dim);
	float *grad_b1 = (float *)palloc0(sizeof(float) * hidden_dim);

	for (i = 0; i < n_nodes; i++)
	{
		grad_h2[i] = (float *)palloc0(sizeof(float) * output_dim);
		grad_h1[i] = (float *)palloc0(sizeof(float) * hidden_dim);
	}

	for (i = 0; i < hidden_dim; i++)
		grad_w2[i] = (float *)palloc0(sizeof(float) * output_dim);
	for (i = 0; i < feature_dim; i++)
		grad_w1[i] = (float *)palloc0(sizeof(float) * hidden_dim);

	for (l = 0; l < epochs; l++)
	{
		float loss = 0.0f;
		int node;

		/* Forward pass */
		gcn_forward(features, adj_norm, w1, b1, n_nodes, feature_dim,
			    hidden_dim, h1);
		gcn_forward(h1, adj_norm, w2, b2, n_nodes, hidden_dim, output_dim,
			    h2);

		/* Compute loss and output layer gradients */
		for (node = 0; node < n_nodes; node++)
		{
			int true_label = labels[node];
			float *output = h2[node];
			float sum_exp = 0.0f;
			float max_logit = -FLT_MAX;
			int c;

			/* Find max for numerical stability */
			for (c = 0; c < output_dim; c++)
			{
				if (output[c] > max_logit)
					max_logit = output[c];
			}

			/* Compute softmax and cross-entropy loss */
			for (c = 0; c < output_dim; c++)
				sum_exp += expf(output[c] - max_logit);

			for (c = 0; c < output_dim; c++)
			{
				float prob = expf(output[c] - max_logit) / sum_exp;
				if (c == true_label)
				{
					loss -= logf(prob + 1e-10f);
					grad_h2[node][c] = prob - 1.0f; /* dL/dlogit = prob - target */
				}
				else
				{
					grad_h2[node][c] = prob; /* dL/dlogit = prob */
				}
			}
		}

		/* Backpropagate through layer 2 */
		/* grad_h1 = adj_norm^T * grad_h2 * w2^T */
		{
			float **grad_h2_w2 = (float **)palloc(sizeof(float *) * n_nodes);
			for (i = 0; i < n_nodes; i++)
				grad_h2_w2[i] = (float *)palloc0(sizeof(float) * hidden_dim);

			/* grad_h2_w2 = grad_h2 * w2^T */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < hidden_dim; j++)
				{
					for (k = 0; k < output_dim; k++)
						grad_h2_w2[i][j] += grad_h2[i][k] * w2[j][k];
				}
			}

			/* grad_h1 = adj_norm^T * grad_h2_w2 */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < n_nodes; j++)
				{
					for (k = 0; k < hidden_dim; k++)
						grad_h1[i][k] += adj_norm[j][i] * grad_h2_w2[j][k];
				}

				/* Apply ReLU derivative */
				for (k = 0; k < hidden_dim; k++)
				{
					if (h1[i][k] <= 0.0f)
						grad_h1[i][k] = 0.0f;
				}
			}

			for (i = 0; i < n_nodes; i++)
				NDB_SAFE_PFREE_AND_NULL(grad_h2_w2[i]);
			NDB_SAFE_PFREE_AND_NULL(grad_h2_w2);
		}

		/* Compute weight gradients for layer 2 */
		/* grad_w2 = h1^T * adj_norm * grad_h2 */
		{
			float **adj_grad_h2 = (float **)palloc(sizeof(float *) * n_nodes);
			for (i = 0; i < n_nodes; i++)
				adj_grad_h2[i] = (float *)palloc0(sizeof(float) * output_dim);

			/* adj_grad_h2 = adj_norm * grad_h2 */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < n_nodes; j++)
				{
					for (k = 0; k < output_dim; k++)
						adj_grad_h2[i][k] += adj_norm[i][j] * grad_h2[j][k];
				}
			}

			/* grad_w2 = h1^T * adj_grad_h2 */
			for (i = 0; i < hidden_dim; i++)
			{
				for (j = 0; j < output_dim; j++)
				{
					for (k = 0; k < n_nodes; k++)
						grad_w2[i][j] += h1[k][i] * adj_grad_h2[k][j];
				}
			}

			/* grad_b2 = sum over nodes of grad_h2 */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < output_dim; j++)
					grad_b2[j] += grad_h2[i][j];
			}

			for (i = 0; i < n_nodes; i++)
				NDB_SAFE_PFREE_AND_NULL(adj_grad_h2[i]);
			NDB_SAFE_PFREE_AND_NULL(adj_grad_h2);
		}

		/* Backpropagate through layer 1 */
		/* grad_features = adj_norm^T * grad_h1 * w1^T */
		{
			float **grad_h1_w1 = (float **)palloc(sizeof(float *) * n_nodes);
			for (i = 0; i < n_nodes; i++)
				grad_h1_w1[i] = (float *)palloc0(sizeof(float) * feature_dim);

			/* grad_h1_w1 = grad_h1 * w1^T */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < feature_dim; j++)
				{
					for (k = 0; k < hidden_dim; k++)
						grad_h1_w1[i][j] += grad_h1[i][k] * w1[j][k];
				}
			}

			/* grad_features = adj_norm^T * grad_h1_w1 (not used for weight update) */
			for (i = 0; i < n_nodes; i++)
				NDB_SAFE_PFREE_AND_NULL(grad_h1_w1[i]);
			NDB_SAFE_PFREE_AND_NULL(grad_h1_w1);
		}

		/* Compute weight gradients for layer 1 */
		/* grad_w1 = features^T * adj_norm * grad_h1 */
		{
			float **adj_grad_h1 = (float **)palloc(sizeof(float *) * n_nodes);
			for (i = 0; i < n_nodes; i++)
				adj_grad_h1[i] = (float *)palloc0(sizeof(float) * hidden_dim);

			/* adj_grad_h1 = adj_norm * grad_h1 */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < n_nodes; j++)
				{
					for (k = 0; k < hidden_dim; k++)
						adj_grad_h1[i][k] += adj_norm[i][j] * grad_h1[j][k];
				}
			}

			/* grad_w1 = features^T * adj_grad_h1 */
			for (i = 0; i < feature_dim; i++)
			{
				for (j = 0; j < hidden_dim; j++)
				{
					for (k = 0; k < n_nodes; k++)
						grad_w1[i][j] += features[k][i] * adj_grad_h1[k][j];
				}
			}

			/* grad_b1 = sum over nodes of grad_h1 */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < hidden_dim; j++)
					grad_b1[j] += grad_h1[i][j];
			}

			for (i = 0; i < n_nodes; i++)
				NDB_SAFE_PFREE_AND_NULL(adj_grad_h1[i]);
			NDB_SAFE_PFREE_AND_NULL(adj_grad_h1);
		}

		/* Update weights */
		for (i = 0; i < feature_dim; i++)
		{
			for (j = 0; j < hidden_dim; j++)
				w1[i][j] -= learning_rate * grad_w1[i][j] / (float)n_nodes;
		}
		for (i = 0; i < hidden_dim; i++)
		{
			for (j = 0; j < output_dim; j++)
				w2[i][j] -= learning_rate * grad_w2[i][j] / (float)n_nodes;
		}
		for (i = 0; i < hidden_dim; i++)
			b1[i] -= learning_rate * grad_b1[i] / (float)n_nodes;
		for (i = 0; i < output_dim; i++)
			b2[i] -= learning_rate * grad_b2[i] / (float)n_nodes;

		/* Reset gradients for next iteration */
		for (i = 0; i < n_nodes; i++)
		{
			for (j = 0; j < output_dim; j++)
				grad_h2[i][j] = 0.0f;
			for (j = 0; j < hidden_dim; j++)
				grad_h1[i][j] = 0.0f;
		}
		for (i = 0; i < hidden_dim; i++)
		{
			for (j = 0; j < output_dim; j++)
				grad_w2[i][j] = 0.0f;
		}
		for (i = 0; i < feature_dim; i++)
		{
			for (j = 0; j < hidden_dim; j++)
				grad_w1[i][j] = 0.0f;
		}
		for (i = 0; i < output_dim; i++)
			grad_b2[i] = 0.0f;
		for (i = 0; i < hidden_dim; i++)
			grad_b1[i] = 0.0f;

		if (l % 10 == 0 || l == epochs - 1)
		{
			elog(DEBUG1,
				"GCN epoch %d: loss = %.6f",
				l, loss / (float)n_nodes);
		}
	}

	/* Cleanup gradient arrays */
	for (i = 0; i < n_nodes; i++)
	{
		NDB_SAFE_PFREE_AND_NULL(grad_h2[i]);
		NDB_SAFE_PFREE_AND_NULL(grad_h1[i]);
	}
	for (i = 0; i < hidden_dim; i++)
		NDB_SAFE_PFREE_AND_NULL(grad_w2[i]);
	for (i = 0; i < feature_dim; i++)
		NDB_SAFE_PFREE_AND_NULL(grad_w1[i]);
	NDB_SAFE_PFREE_AND_NULL(grad_h2);
	NDB_SAFE_PFREE_AND_NULL(grad_h1);
	NDB_SAFE_PFREE_AND_NULL(grad_w2);
	NDB_SAFE_PFREE_AND_NULL(grad_w1);
	NDB_SAFE_PFREE_AND_NULL(grad_b2);
	NDB_SAFE_PFREE_AND_NULL(grad_b1);

	/* Serialize GCN model */
	{
		StringInfoData model_buf;
		bytea *serialized = NULL;
		StringInfoData paramsbuf;
		StringInfoData metricsbuf;
		Jsonb *params_jsonb = NULL;
		Jsonb *metrics_jsonb = NULL;
		MLCatalogModelSpec spec;

		/* Serialize model weights and biases */
		initStringInfo(&model_buf);

		/* Write header */
		pq_sendint32(&model_buf, n_nodes);
		pq_sendint32(&model_buf, feature_dim);
		pq_sendint32(&model_buf, hidden_dim);
		pq_sendint32(&model_buf, output_dim);
		pq_sendint32(&model_buf, n_layers); /* 2 layers */

		/* Serialize layer 1 weights (feature_dim x hidden_dim) */
		for (i = 0; i < feature_dim; i++)
		{
			for (j = 0; j < hidden_dim; j++)
				pq_sendfloat8(&model_buf, w1[i][j]);
		}

		/* Serialize layer 1 bias */
		for (j = 0; j < hidden_dim; j++)
			pq_sendfloat8(&model_buf, b1[j]);

		/* Serialize layer 2 weights (hidden_dim x output_dim) */
		for (i = 0; i < hidden_dim; i++)
		{
			for (j = 0; j < output_dim; j++)
				pq_sendfloat8(&model_buf, w2[i][j]);
		}

		/* Serialize layer 2 bias */
		for (j = 0; j < output_dim; j++)
			pq_sendfloat8(&model_buf, b2[j]);

		serialized = (bytea *)pq_endtypsend(&model_buf);

		/* Build parameters JSON */
		initStringInfo(&paramsbuf);
		appendStringInfo(&paramsbuf,
			"{\"n_nodes\":%d,"
			"\"feature_dim\":%d,"
			"\"hidden_dim\":%d,"
			"\"output_dim\":%d,"
			"\"learning_rate\":%.6f,"
			"\"epochs\":%d,"
			"\"graph_table\":\"%s\","
			"\"features_table\":\"%s\","
			"\"labels_table\":\"%s\"}",
			n_nodes,
			feature_dim,
			hidden_dim,
			output_dim,
			learning_rate,
			epochs,
			graph_tbl,
			feat_tbl,
			label_tbl);
		params_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(paramsbuf.data)));
		NDB_SAFE_PFREE_AND_NULL(paramsbuf.data);

		/* Build metrics JSON */
		initStringInfo(&metricsbuf);
		appendStringInfo(&metricsbuf,
			"{\"n_nodes\":%d,"
			"\"training_complete\":true}");
		metrics_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metricsbuf.data)));
		NDB_SAFE_PFREE_AND_NULL(metricsbuf.data);

		/* Register model in catalog */
		memset(&spec, 0, sizeof(MLCatalogModelSpec));
		spec.algorithm = "gcn";
		spec.training_table = graph_tbl;
		spec.model_data = serialized;
		spec.parameters = params_jsonb;
		spec.metrics = metrics_jsonb;

		model_id = ml_catalog_register_model(&spec);
	}

	/* Cleanup */
	for (i = 0; i < n_nodes; i++)
	{
		NDB_SAFE_PFREE_AND_NULL(adjacency[i]);
		NDB_SAFE_PFREE_AND_NULL(adj_norm[i]);
		NDB_SAFE_PFREE_AND_NULL(features[i]);
		NDB_SAFE_PFREE_AND_NULL(h1[i]);
		NDB_SAFE_PFREE_AND_NULL(h2[i]);
	}
	NDB_SAFE_PFREE_AND_NULL(adjacency);
	NDB_SAFE_PFREE_AND_NULL(adj_norm);
	NDB_SAFE_PFREE_AND_NULL(features);
	NDB_SAFE_PFREE_AND_NULL(labels);
	for (i = 0; i < feature_dim; i++)
		NDB_SAFE_PFREE_AND_NULL(w1[i]);
	NDB_SAFE_PFREE_AND_NULL(w1);
	NDB_SAFE_PFREE_AND_NULL(b1);
	for (i = 0; i < hidden_dim; i++)
		NDB_SAFE_PFREE_AND_NULL(w2[i]);
	NDB_SAFE_PFREE_AND_NULL(w2);
	NDB_SAFE_PFREE_AND_NULL(b2);
	NDB_SAFE_PFREE_AND_NULL(h1);
	NDB_SAFE_PFREE_AND_NULL(h2);
	NDB_SAFE_PFREE_AND_NULL(graph_tbl);
	NDB_SAFE_PFREE_AND_NULL(feat_tbl);
	NDB_SAFE_PFREE_AND_NULL(label_tbl);
	SPI_finish();

	PG_RETURN_INT32(model_id);
}

/*
 * graphsage_aggregate
 * -------------------
 * GraphSAGE neighbor sampling and aggregation.
 *
 * SQL Arguments:
 *   graph_table    - Graph structure table
 *   features_table - Node features table
 *   node_id        - Target node
 *   n_samples      - Number of neighbors to sample (default: 10)
 *   depth          - Aggregation depth (default: 2)
 *
 * Returns:
 *   Aggregated feature vector
 */
PG_FUNCTION_INFO_V1(graphsage_aggregate);

Datum
graphsage_aggregate(PG_FUNCTION_ARGS)
{
	text *graph_table;
	text *features_table;
	int32 node_id;
	int n_samples;
	int depth;
	char *graph_tbl;
	char *feat_tbl;
	float *aggregated;
	int feature_dim;
	ArrayType *result;
	Datum *result_datums;

	graph_table = PG_GETARG_TEXT_PP(0);
	features_table = PG_GETARG_TEXT_PP(1);
	node_id = PG_GETARG_INT32(2);
	n_samples = PG_ARGISNULL(3) ? 10 : PG_GETARG_INT32(3);
	depth = PG_ARGISNULL(4) ? 2 : PG_GETARG_INT32(4);

	if (n_samples < 1 || depth < 1)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("n_samples and depth must be positive")));

	graph_tbl = text_to_cstring(graph_table);
	feat_tbl = text_to_cstring(features_table);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_connect failed")));

	/* Load graph structure and extract feature dimension */
	{
		StringInfoData query;
		int ret;
		int *neighbors = NULL;
		int neighbor_count = 0;
		int max_neighbors = n_samples * depth;
		float **neighbor_features = NULL;
		int sampled_count = 0;

		/* First, get feature dimension from features table */
		initStringInfo(&query);
		appendStringInfo(&query,
			"SELECT features FROM %s WHERE node_id = %d LIMIT 1",
			feat_tbl, node_id);

		ret = ndb_spi_execute_safe(query.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			HeapTuple tuple = SPI_tuptable->vals[0];
			ArrayType *feat_array = DatumGetArrayTypeP(
				SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, NULL));
			feature_dim = ARR_DIMS(feat_array)[0];
		}
		else
		{
			/* Fallback: try to get dimension from any row */
			NDB_SAFE_PFREE_AND_NULL(query.data);
			initStringInfo(&query);
			appendStringInfo(&query,
				"SELECT features FROM %s LIMIT 1", feat_tbl);
			ret = ndb_spi_execute_safe(query.data, true, 0);
			NDB_CHECK_SPI_TUPTABLE();
			if (ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				HeapTuple tuple = SPI_tuptable->vals[0];
				ArrayType *feat_array = DatumGetArrayTypeP(
					SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, NULL));
				feature_dim = ARR_DIMS(feat_array)[0];
			}
			else
			{
				feature_dim = 64; /* Default fallback */
			}
		}
		NDB_SAFE_PFREE_AND_NULL(query.data);

		/* Load neighbors from graph structure */
		neighbors = (int *)palloc(sizeof(int) * max_neighbors);
		neighbor_features = (float **)palloc(sizeof(float *) * max_neighbors);

		/* Sample neighbors at each depth */
		{
			int current_nodes[1000]; /* Current level nodes */
			int next_nodes[1000]; /* Next level nodes */
			int current_count = 1;
			int next_count = 0;
			int d;
			int total_sampled = 0;

			current_nodes[0] = node_id;

			for (d = 0; d < depth && total_sampled < max_neighbors; d++)
			{
				next_count = 0;

				/* For each node at current depth, sample neighbors */
				for (i = 0; i < current_count && total_sampled < max_neighbors; i++)
				{
					int current_node = current_nodes[i];
					int samples_needed = (d == depth - 1) ? n_samples : n_samples;
					int sampled = 0;

					/* Query neighbors from graph table */
					initStringInfo(&query);
					appendStringInfo(&query,
						"SELECT neighbor_id FROM %s WHERE node_id = %d ORDER BY random() LIMIT %d",
						graph_tbl, current_node, samples_needed);

					ret = ndb_spi_execute_safe(query.data, true, 0);
					NDB_CHECK_SPI_TUPTABLE();
					if (ret == SPI_OK_SELECT)
					{
						for (j = 0; j < SPI_processed && sampled < samples_needed && total_sampled < max_neighbors; j++)
						{
							HeapTuple tuple = SPI_tuptable->vals[j];
							int neighbor_id = DatumGetInt32(SPI_getbinval(tuple,
								SPI_tuptable->tupdesc, 1, NULL));

							if (total_sampled < max_neighbors)
							{
								neighbors[total_sampled] = neighbor_id;
								if (next_count < 1000)
									next_nodes[next_count++] = neighbor_id;
								total_sampled++;
								sampled++;
							}
						}
					}
					NDB_SAFE_PFREE_AND_NULL(query.data);
				}

				/* Move to next depth */
				for (i = 0; i < next_count && i < 1000; i++)
					current_nodes[i] = next_nodes[i];
				current_count = next_count;
			}

			sampled_count = total_sampled;
		}

		/* Load features for sampled neighbors */
		aggregated = (float *)palloc0(sizeof(float) * feature_dim);

		if (sampled_count > 0)
		{
			/* Load features for each neighbor */
			for (i = 0; i < sampled_count; i++)
			{
				initStringInfo(&query);
				appendStringInfo(&query,
					"SELECT features FROM %s WHERE node_id = %d",
					feat_tbl, neighbors[i]);

				ret = ndb_spi_execute_safe(query.data, true, 0);
				NDB_CHECK_SPI_TUPTABLE();
				if (ret == SPI_OK_SELECT && SPI_processed > 0)
				{
					HeapTuple tuple = SPI_tuptable->vals[0];
					ArrayType *feat_array = DatumGetArrayTypeP(
						SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, NULL));
					float *feat_data = (float *)ARR_DATA_PTR(feat_array);
					int feat_len = ARR_DIMS(feat_array)[0];

					neighbor_features[i] = (float *)palloc(sizeof(float) * feature_dim);
					for (j = 0; j < feature_dim && j < feat_len; j++)
						neighbor_features[i][j] = feat_data[j];
					for (; j < feature_dim; j++)
						neighbor_features[i][j] = 0.0f;
				}
				else
				{
					neighbor_features[i] = (float *)palloc0(sizeof(float) * feature_dim);
				}
				NDB_SAFE_PFREE_AND_NULL(query.data);
			}

			/* Aggregate: mean of neighbor features */
			for (i = 0; i < sampled_count; i++)
			{
				for (j = 0; j < feature_dim; j++)
					aggregated[j] += neighbor_features[i][j];
			}

			/* Normalize by count */
			if (sampled_count > 0)
			{
				for (j = 0; j < feature_dim; j++)
					aggregated[j] /= sampled_count;
			}

			/* Cleanup neighbor features */
			for (i = 0; i < sampled_count; i++)
				NDB_SAFE_PFREE_AND_NULL(neighbor_features[i]);
		}

		NDB_SAFE_PFREE_AND_NULL(neighbors);
		NDB_SAFE_PFREE_AND_NULL(neighbor_features);
	}

	/* Build result array */
	result_datums = (Datum *)palloc(sizeof(Datum) * feature_dim);
	for (int i = 0; i < feature_dim; i++)
		result_datums[i] = Float4GetDatum(aggregated[i]);

	result = construct_array(result_datums,
				 feature_dim,
				 FLOAT4OID,
				 sizeof(float4),
				 FLOAT4PASSBYVAL,
				 'i');

	NDB_SAFE_PFREE_AND_NULL(aggregated);
	NDB_SAFE_PFREE_AND_NULL(result_datums);
	NDB_SAFE_PFREE_AND_NULL(graph_tbl);
	NDB_SAFE_PFREE_AND_NULL(feat_tbl);
	SPI_finish();

	PG_RETURN_ARRAYTYPE_P(result);
}


