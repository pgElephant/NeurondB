/*-------------------------------------------------------------------------
 *
 * ml_neural_network.c
 *	  Basic Neural Network Implementation for NeuronDB
 *
 * Implements feedforward neural networks with backpropagation.
 * Supports multiple hidden layers and various activation functions.
 *
 * IDENTIFICATION
 *	  src/ml/ml_neural_network.c
 *
 * Copyright (c) 2024-2025, pgElephant, Inc.
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/jsonb.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include "neurondb_pgcompat.h"
#include "neurondb.h"
#include "ml_catalog.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Neural Network Structures */
typedef struct NeuralLayer
{
	int n_inputs;
	int n_outputs;
	float **weights; /* [n_outputs][n_inputs+1] (includes bias) */
	float *activations; /* Current layer activations */
	float *deltas; /* Backpropagation deltas */
} NeuralLayer;

typedef struct NeuralNetwork
{
	int n_layers;
	int n_inputs;
	int n_outputs;
	NeuralLayer *layers;
	char *activation_func; /* "relu", "sigmoid", "tanh" */
	float learning_rate;
} NeuralNetwork;

/* Activation functions */
static float
activation_relu(float x)
{
	return (x > 0.0f) ? x : 0.0f;
}

static float
activation_sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

static float
activation_tanh(float x)
{
	return tanhf(x);
}

static float
activation_derivative_relu(float x)
{
	return (x > 0.0f) ? 1.0f : 0.0f;
}

static float
activation_derivative_sigmoid(float x)
{
	float s = activation_sigmoid(x);
	return s * (1.0f - s);
}

static float
activation_derivative_tanh(float x)
{
	float t = tanhf(x);
	return 1.0f - t * t;
}

/* Forward pass through network */
static void
neural_network_forward(NeuralNetwork *net, float *input, float *output)
{
	int i, j, k;
	float *prev_activations = input;
	float *curr_activations;

	for (i = 0; i < net->n_layers; i++)
	{
		NeuralLayer *layer = &net->layers[i];
		curr_activations = layer->activations;

		for (j = 0; j < layer->n_outputs; j++)
		{
			float sum =
				layer->weights[j][layer->n_inputs]; /* bias */

			for (k = 0; k < layer->n_inputs; k++)
				sum += layer->weights[j][k]
					* prev_activations[k];

			/* Apply activation */
			if (strcmp(net->activation_func, "relu") == 0)
				curr_activations[j] = activation_relu(sum);
			else if (strcmp(net->activation_func, "sigmoid") == 0)
				curr_activations[j] = activation_sigmoid(sum);
			else if (strcmp(net->activation_func, "tanh") == 0)
				curr_activations[j] = activation_tanh(sum);
			else
				curr_activations[j] = sum; /* linear */
		}

		prev_activations = curr_activations;
	}

	/* Copy final layer to output */
	memcpy(output,
		net->layers[net->n_layers - 1].activations,
		net->n_outputs * sizeof(float));
}

/* Backward pass (backpropagation) */
static void
neural_network_backward(NeuralNetwork *net,
	float *input,
	float *target,
	float *predicted)
{
	int i, j, k;
	NeuralLayer *output_layer = &net->layers[net->n_layers - 1];

	/* Compute output layer deltas */
	for (j = 0; j < output_layer->n_outputs; j++)
	{
		float error = target[j] - predicted[j];
		float activation = output_layer->activations[j];
		float derivative;

		if (strcmp(net->activation_func, "relu") == 0)
			derivative = activation_derivative_relu(activation);
		else if (strcmp(net->activation_func, "sigmoid") == 0)
			derivative = activation_derivative_sigmoid(activation);
		else if (strcmp(net->activation_func, "tanh") == 0)
			derivative = activation_derivative_tanh(activation);
		else
			derivative = 1.0f;

		output_layer->deltas[j] = error * derivative;
	}

	/* Backpropagate through hidden layers */
	for (i = net->n_layers - 2; i >= 0; i--)
	{
		NeuralLayer *curr_layer = &net->layers[i];
		NeuralLayer *next_layer = &net->layers[i + 1];

		for (j = 0; j < curr_layer->n_outputs; j++)
		{
			float sum = 0.0f;
			float activation;
			float derivative;

			for (k = 0; k < next_layer->n_outputs; k++)
				sum += next_layer->weights[k][j]
					* next_layer->deltas[k];

			activation = curr_layer->activations[j];

			if (strcmp(net->activation_func, "relu") == 0)
				derivative =
					activation_derivative_relu(activation);
			else if (strcmp(net->activation_func, "sigmoid") == 0)
				derivative = activation_derivative_sigmoid(
					activation);
			else if (strcmp(net->activation_func, "tanh") == 0)
				derivative =
					activation_derivative_tanh(activation);
			else
				derivative = 1.0f;

			curr_layer->deltas[j] = sum * derivative;
		}
	}
}

/* Update weights using gradient descent */
static void
neural_network_update_weights(NeuralNetwork *net, float *input)
{
	int i, j, k;
	float *prev_activations = input;

	for (i = 0; i < net->n_layers; i++)
	{
		NeuralLayer *layer = &net->layers[i];

		for (j = 0; j < layer->n_outputs; j++)
		{
			/* Update bias */
			layer->weights[j][layer->n_inputs] +=
				net->learning_rate * layer->deltas[j];

			/* Update input weights */
			for (k = 0; k < layer->n_inputs; k++)
				layer->weights[j][k] += net->learning_rate
					* layer->deltas[j]
					* prev_activations[k];
		}

		prev_activations = layer->activations;
	}
}

/* Initialize neural network */
static NeuralNetwork *
neural_network_init(int n_inputs,
	int n_outputs,
	int *hidden_layers,
	int n_hidden,
	const char *activation,
	float learning_rate)
{
	int i;
	int j;
	int k;
	int prev_size;
	NeuralLayer *output_layer;
	NeuralNetwork *net = (NeuralNetwork *)palloc(sizeof(NeuralNetwork));

	net->n_inputs = n_inputs;
	net->n_outputs = n_outputs;
	net->n_layers = n_hidden + 1; /* hidden + output */
	net->activation_func = pstrdup(activation);
	net->learning_rate = learning_rate;

	net->layers =
		(NeuralLayer *)palloc(net->n_layers * sizeof(NeuralLayer));

	/* Initialize hidden layers */
	prev_size = n_inputs;

	for (i = 0; i < n_hidden; i++)
	{
		NeuralLayer *layer = &net->layers[i];

		layer->n_inputs = prev_size;
		layer->n_outputs = hidden_layers[i];
		layer->weights =
			(float **)palloc(layer->n_outputs * sizeof(float *));
		layer->activations =
			(float *)palloc(layer->n_outputs * sizeof(float));
		layer->deltas =
			(float *)palloc(layer->n_outputs * sizeof(float));

		for (j = 0; j < layer->n_outputs; j++)
		{
			layer->weights[j] = (float *)palloc(
				(layer->n_inputs + 1) * sizeof(float));
			/* Initialize weights randomly (small values) */
			for (k = 0; k <= layer->n_inputs; k++)
				layer->weights[j][k] =
					(float)(((double)rand() / (double)RAND_MAX)) * 0.1f
					- 0.05f;
		}

		prev_size = layer->n_outputs;
	}

	/* Initialize output layer */
	output_layer = &net->layers[n_hidden];
	output_layer->n_inputs = prev_size;
	output_layer->n_outputs = n_outputs;
	output_layer->weights =
		(float **)palloc(output_layer->n_outputs * sizeof(float *));
	output_layer->activations =
		(float *)palloc(output_layer->n_outputs * sizeof(float));
	output_layer->deltas =
		(float *)palloc(output_layer->n_outputs * sizeof(float));

	for (j = 0; j < output_layer->n_outputs; j++)
	{
		output_layer->weights[j] = (float *)palloc(
			(output_layer->n_inputs + 1) * sizeof(float));
		for (k = 0; k <= output_layer->n_inputs; k++)
			output_layer->weights[j][k] =
				(float)(((double)rand() / (double)RAND_MAX)) * 0.1f - 0.05f;
	}

	return net;
}

/* Free neural network */
static void
neural_network_free(NeuralNetwork *net)
{
	int i, j;

	if (!net)
		return;

	for (i = 0; i < net->n_layers; i++)
	{
		NeuralLayer *layer = &net->layers[i];

		for (j = 0; j < layer->n_outputs; j++)
			pfree(layer->weights[j]);

		pfree(layer->weights);
		pfree(layer->activations);
		pfree(layer->deltas);
	}

	pfree(net->layers);
	pfree(net->activation_func);
	pfree(net);
}

/*
 * Train neural network
 * 
 * train_neural_network(
 *   table_name text,
 *   feature_col text,
 *   label_col text,
 *   hidden_layers int[],
 *   activation text DEFAULT 'relu',
 *   learning_rate float8 DEFAULT 0.01,
 *   epochs int DEFAULT 100,
 *   batch_size int DEFAULT 32
 * )
 */
PG_FUNCTION_INFO_V1(train_neural_network);

Datum
train_neural_network(PG_FUNCTION_ARGS)
{
	text *table_name = PG_GETARG_TEXT_PP(0);
	text *feature_col = PG_GETARG_TEXT_PP(1);
	text *label_col = PG_GETARG_TEXT_PP(2);
	ArrayType *hidden_layers_array = PG_GETARG_ARRAYTYPE_P(3);
	text *activation_text = PG_ARGISNULL(4) ? NULL : PG_GETARG_TEXT_PP(4);
	float8 learning_rate = PG_ARGISNULL(5) ? 0.01 : PG_GETARG_FLOAT8(5);
	int32 epochs = PG_ARGISNULL(6) ? 100 : PG_GETARG_INT32(6);
	char *table_name_str;
	char *feature_col_str;
	char *label_col_str;
	char *activation;
	int n_hidden;
	int *hidden_layers;
	int n_inputs = 0;
	int n_outputs = 1;
	MemoryContext oldcontext;
	MemoryContext callcontext;
	StringInfoData sql;
	int ret;
	SPITupleTable *tuptable;
	TupleDesc tupdesc;
	int n_samples = 0;
	float **X = NULL;
	float *y = NULL;
	NeuralNetwork *net = NULL;
	int epoch, sample;
	float loss;
	int i;
	int j;

	table_name_str = text_to_cstring(table_name);
	feature_col_str = text_to_cstring(feature_col);
	label_col_str = text_to_cstring(label_col);
	activation = activation_text ? text_to_cstring(activation_text)
				     : pstrdup("relu");

	/* batch_size not yet used in training loop - will be implemented */
	if (PG_ARGISNULL(7))
		(void)32;
	else
		(void)PG_GETARG_INT32(7);

	/* Validate activation function */
	if (strcmp(activation, "relu") != 0
		&& strcmp(activation, "sigmoid") != 0
		&& strcmp(activation, "tanh") != 0
		&& strcmp(activation, "linear") != 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("activation must be 'relu', 'sigmoid', "
				       "'tanh', or 'linear'")));

	/* Extract hidden layers */
	n_hidden = ArrayGetNItems(
		ARR_NDIM(hidden_layers_array), ARR_DIMS(hidden_layers_array));
	hidden_layers = (int *)palloc(n_hidden * sizeof(int));

	for (i = 0; i < n_hidden; i++)
	{
		bool isnull;
		Datum elem;

		elem = array_ref(hidden_layers_array,
			1,
			&i,
			-1,
			-1,
			false,
			'i',
			&isnull);

		if (isnull)
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("hidden_layers array cannot "
					       "contain NULL")));

		hidden_layers[i] = DatumGetInt32(elem);
		if (hidden_layers[i] <= 0)
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("hidden layer sizes must be "
					       "positive")));
	}

	/* Create memory context */
	callcontext = AllocSetContextCreate(CurrentMemoryContext,
		"train_neural_network memory context",
		ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("SPI_connect failed")));

	/* Load training data */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		quote_identifier(feature_col_str),
		quote_identifier(label_col_str),
		quote_identifier(table_name_str),
		quote_identifier(feature_col_str),
		quote_identifier(label_col_str));

	ret = SPI_execute(sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("failed to load training data")));

	tuptable = SPI_tuptable;
	tupdesc = tuptable->tupdesc;
	n_samples = SPI_processed;

	if (n_samples == 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("no training data found")));

	/* Determine input/output dimensions will be set from first vector */
	n_inputs = 0; /* Will be determined from first vector */
	n_outputs = 1; /* Regression - single output */

	/* Determine actual feature dimension from first row */
	if (n_samples > 0)
	{
		HeapTuple first_tuple = tuptable->vals[0];
		bool		isnull;
		Datum		feat_datum;

		feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &isnull);
		if (isnull)
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("neural network training: feature vector cannot be NULL")));

		/* Extract dimension from vector type */
		/* Check if type is vector by trying to cast to Vector */
		{
			Vector	   *test_vec;

			test_vec = DatumGetVector(feat_datum);
			if (test_vec != NULL && test_vec->dim > 0)
			{
				n_inputs = test_vec->dim;
				if (n_inputs <= 0 || n_inputs > 10000)
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neural network training: invalid vector dimension %d",
									n_inputs)));
			}
			else
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neural network training: feature column must be vector type")));
			}
		}
	}
	else
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neural network training: no training data found")));
	}

	/* Allocate training data arrays with correct dimensions */
	X = (float **)palloc(n_samples * sizeof(float *));
	y = (float *)palloc(n_samples * sizeof(float));

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple tuple = tuptable->vals[i];
		bool		isnull;
		Datum		feat_datum;
		Datum		label_datum;
		Vector	   *vec;

		/* Extract feature vector */
		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
		if (isnull)
		{
			for (j = 0; j < i; j++)
				pfree(X[j]);
			pfree(X);
			pfree(y);
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("neural network training: feature vector at row %d cannot be NULL",
							i + 1)));
		}

		vec = DatumGetVector(feat_datum);
		if (vec == NULL || vec->dim != n_inputs)
		{
			for (j = 0; j < i; j++)
				pfree(X[j]);
			pfree(X);
			pfree(y);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neural network training: invalid vector at row %d (expected dim %d, got %d)",
							i + 1, n_inputs, vec ? vec->dim : 0)));
		}

		/* Copy vector data to feature matrix */
		X[i] = (float *)palloc(n_inputs * sizeof(float));
		for (j = 0; j < n_inputs; j++)
		{
			if (!isfinite(vec->data[j]))
			{
				for (j = 0; j <= i; j++)
					pfree(X[j]);
				pfree(X);
				pfree(y);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neural network training: non-finite value in feature vector at row %d, dimension %d",
								i + 1, j)));
			}
			X[i][j] = vec->data[j];
		}

		/* Extract label */
		label_datum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
		if (isnull)
		{
			for (j = 0; j <= i; j++)
				pfree(X[j]);
			pfree(X);
			pfree(y);
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("neural network training: label at row %d cannot be NULL",
							i + 1)));
		}

		{
			Oid			label_type = SPI_gettypeid(tupdesc, 2);

			if (label_type == FLOAT8OID)
				y[i] = (float)DatumGetFloat8(label_datum);
			else if (label_type == FLOAT4OID)
				y[i] = DatumGetFloat4(label_datum);
			else if (label_type == INT4OID)
				y[i] = (float)DatumGetInt32(label_datum);
			else if (label_type == INT8OID)
				y[i] = (float)DatumGetInt64(label_datum);
			else
			{
				for (j = 0; j <= i; j++)
					pfree(X[j]);
				pfree(X);
				pfree(y);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neural network training: unsupported label type")));
			}

			if (!isfinite(y[i]))
			{
				for (j = 0; j <= i; j++)
					pfree(X[j]);
				pfree(X);
				pfree(y);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neural network training: non-finite label value at row %d",
								i + 1)));
			}
		}
	}

	/* Initialize neural network */
	net = neural_network_init(n_inputs,
		n_outputs,
		hidden_layers,
		n_hidden,
		activation,
		(float)learning_rate);

	/* Training loop */
	for (epoch = 0; epoch < epochs; epoch++)
	{
		loss = 0.0f;

		for (sample = 0; sample < n_samples; sample++)
		{
			float predicted[1];
			float error;
			float target[1];

			/* Forward pass */
			neural_network_forward(net, X[sample], predicted);

			/* Compute loss */
			error = y[sample] - predicted[0];
			loss += error * error;

			/* Backward pass */
			target[0] = y[sample];
			neural_network_backward(
				net, X[sample], target, predicted);

			/* Update weights */
			neural_network_update_weights(net, X[sample]);
		}

		loss /= n_samples;

		if (epoch % 10 == 0)
			elog(DEBUG1, "epoch %d: loss = %.6f", epoch, loss);
	}

	/* Store model in database */
	/* Simplified: just return success message */
	SPI_finish();
	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(callcontext);

	/* Cleanup */
	for (i = 0; i < n_samples; i++)
		pfree(X[i]);
	pfree(X);
	pfree(y);
	neural_network_free(net);
	pfree(hidden_layers);
	pfree(table_name_str);
	pfree(feature_col_str);
	pfree(label_col_str);
	pfree(activation);

	PG_RETURN_TEXT_P(cstring_to_text("Neural network training completed"));
}

/*
 * Predict with neural network
 *
 * Loads trained neural network model and performs forward pass
 * to generate prediction.
 */
PG_FUNCTION_INFO_V1(predict_neural_network);

Datum
predict_neural_network(PG_FUNCTION_ARGS)
{
	Vector	   *features;
	int32		model_id;
	bytea	   *model_data = NULL;
	Jsonb	   *parameters = NULL;
	Jsonb	   *metrics = NULL;
	MemoryContext oldcontext;
	MemoryContext pred_context;

	/* Defensive: validate inputs */
	features = PG_GETARG_VECTOR_P(0);
	model_id = PG_GETARG_INT32(1);

	if (features == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("predict_neural_network: features cannot be NULL")));

	if (model_id <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_neural_network: model_id must be positive")));

	if (features->dim <= 0 || features->dim > 10000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_neural_network: invalid feature dimension %d",
						features->dim)));

	/* Create memory context */
	pred_context = AllocSetContextCreate(CurrentMemoryContext,
										"neural network prediction context",
										ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(pred_context);

	/* Load model from catalog */
	if (!ml_catalog_fetch_model_payload(model_id, &model_data,
										&parameters, &metrics))
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(pred_context);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_neural_network: model %d not found", model_id)));
	}

	/* Defensive: validate model_data */
	if (model_data == NULL)
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(pred_context);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_neural_network: model %d has no model data",
						model_id)));
	}

	/* TODO: Deserialize neural network from model_data */
	/* For now, return error indicating model deserialization needed */
	/* This requires implementing neural_network_deserialize function */
	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(pred_context);
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("predict_neural_network: model deserialization not yet implemented"),
			 errhint("Neural network model storage and deserialization needs to be implemented")));

	/* Placeholder code below - will be enabled after deserialization is implemented */
	/*
	 * net = neural_network_deserialize(model_data);
	 * if (net == NULL)
	 *     ereport(ERROR, ...);
	 *
	 * input_features = (float *)palloc(features->dim * sizeof(float));
	 * for (i = 0; i < features->dim; i++)
	 * {
	 *     if (!isfinite(features->data[i]))
	 *         ereport(ERROR, ...);
	 *     input_features[i] = features->data[i];
	 * }
	 *
	 * neural_network_forward(net, input_features, result);
	 * neural_network_free(net);
	 * pfree(input_features);
	 *
	 * MemoryContextSwitchTo(oldcontext);
	 * MemoryContextDelete(pred_context);
	 *
	 * PG_RETURN_FLOAT8(result[0]);
	 */
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration Stub for NeuralNetwork
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"

void
neurondb_gpu_register_neural_network_model(void)
{
	elog(DEBUG1, "NeuralNetwork GPU Model Ops registration skipped - not yet implemented");
}
