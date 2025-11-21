#!/usr/bin/env python3
"""
NeuronDB to ONNX Converter
Converts NeuronDB custom binary model formats to ONNX format for Hugging Face compatibility.

Usage:
    python neurondb_to_onnx.py <model_data_file> <algorithm> <output_path>
    
    algorithm: ridge, lasso, linear_regression, logistic_regression
"""

import struct
import sys
import os
import json
from typing import Dict, Any, Optional
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("ERROR: onnx library not installed. Install with: pip install onnx", file=sys.stderr)
    sys.exit(1)


class NeuronDBModelReader:
    """Reads NeuronDB custom binary model formats (PostgreSQL pq format, network byte order)"""
    
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
    
    def read_int32(self) -> int:
        """Read 32-bit integer (network byte order = big-endian)"""
        if self.pos + 4 > len(self.data):
            raise ValueError("Unexpected end of data")
        val = struct.unpack('>i', self.data[self.pos:self.pos+4])[0]
        self.pos += 4
        return val
    
    def read_float64(self) -> float:
        """Read 64-bit float (network byte order = big-endian)"""
        if self.pos + 8 > len(self.data):
            raise ValueError("Unexpected end of data")
        val = struct.unpack('>d', self.data[self.pos:self.pos+8])[0]
        self.pos += 8
        return val


def deserialize_ridge_model(data: bytes) -> Dict[str, Any]:
    """Deserialize Ridge regression model from NeuronDB binary format"""
    reader = NeuronDBModelReader(data)
    
    # Skip varlena header (4 bytes for length)
    if len(data) < 4:
        raise ValueError("Invalid data: too short")
    varlena_len = struct.unpack('>I', data[0:4])[0]
    reader.pos = 4  # Skip varlena header
    
    model = {
        'algorithm': 'ridge',
        'n_features': reader.read_int32(),
        'n_samples': reader.read_int32(),
        'intercept': reader.read_float64(),
        'lambda': reader.read_float64(),
        'r_squared': reader.read_float64(),
        'mse': reader.read_float64(),
        'mae': reader.read_float64(),
    }
    
    # Read coefficients
    n_features = model['n_features']
    if n_features > 0:
        coefficients = []
        for i in range(n_features):
            coefficients.append(reader.read_float64())
        model['coefficients'] = np.array(coefficients, dtype=np.float32)
    
    return model


def deserialize_lasso_model(data: bytes) -> Dict[str, Any]:
    """Deserialize Lasso regression model from NeuronDB binary format"""
    # Lasso format is identical to Ridge
    model = deserialize_ridge_model(data)
    model['algorithm'] = 'lasso'
    return model


def deserialize_linear_regression_model(data: bytes) -> Dict[str, Any]:
    """Deserialize Linear Regression model from NeuronDB binary format"""
    reader = NeuronDBModelReader(data)
    
    if len(data) < 4:
        raise ValueError("Invalid data: too short")
    reader.pos = 4  # Skip varlena header
    
    model = {
        'algorithm': 'linear_regression',
        'n_features': reader.read_int32(),
        'n_samples': reader.read_int32(),
        'intercept': reader.read_float64(),
        'r_squared': reader.read_float64(),
        'mse': reader.read_float64(),
        'mae': reader.read_float64(),
    }
    
    n_features = model['n_features']
    if n_features > 0:
        coefficients = []
        for i in range(n_features):
            coefficients.append(reader.read_float64())
        model['coefficients'] = np.array(coefficients, dtype=np.float32)
    
    return model


def deserialize_logistic_regression_model(data: bytes) -> Dict[str, Any]:
    """Deserialize Logistic Regression model from NeuronDB binary format"""
    reader = NeuronDBModelReader(data)
    
    if len(data) < 4:
        raise ValueError("Invalid data: too short")
    reader.pos = 4  # Skip varlena header
    
    model = {
        'algorithm': 'logistic_regression',
        'n_features': reader.read_int32(),
        'n_samples': reader.read_int32(),
        'bias': reader.read_float64(),
        'learning_rate': reader.read_float64(),
        'lambda': reader.read_float64(),
        'max_iters': reader.read_int32(),
        'final_loss': reader.read_float64(),
        'accuracy': reader.read_float64(),
    }
    
    n_features = model['n_features']
    if n_features > 0:
        weights = []
        for i in range(n_features):
            weights.append(reader.read_float64())
        model['weights'] = np.array(weights, dtype=np.float32)
    
    return model


def ridge_to_onnx(model: Dict[str, Any], output_path: str) -> str:
    """Convert Ridge model to ONNX format"""
    n_features = model['n_features']
    coefficients = model['coefficients']
    intercept = model['intercept']
    
    # Create ONNX graph
    # Input: [batch_size, n_features]
    # Output: [batch_size, 1]
    
    # Input tensor
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [None, n_features]
    )
    
    # Coefficients as initializer (transposed for MatMul: [n_features, 1])
    # Reshape coefficients from [n_features] to [n_features, 1]
    coeff_2d = coefficients.reshape(n_features, 1)
    coeff_tensor = helper.make_tensor(
        'coefficients',
        TensorProto.FLOAT,
        [n_features, 1],
        coeff_2d.flatten().tolist()
    )
    
    # Intercept as initializer
    intercept_tensor = helper.make_tensor(
        'intercept',
        TensorProto.FLOAT,
        [1],
        [float(intercept)]
    )
    
    # Output tensor
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [None, 1]
    )
    
    # Create graph: MatMul(input, coefficients) + intercept
    matmul_node = helper.make_node(
        'MatMul',
        ['input', 'coefficients'],
        ['matmul_output'],
        name='MatMul'
    )
    
    add_node = helper.make_node(
        'Add',
        ['matmul_output', 'intercept'],
        ['output'],
        name='Add'
    )
    
    graph = helper.make_graph(
        [matmul_node, add_node],
        'ridge_regression',
        [input_tensor],
        [output_tensor],
        [coeff_tensor, intercept_tensor]
    )
    
    model_proto = helper.make_model(graph, producer_name='NeuronDB')
    model_proto.opset_import[0].version = 14
    
    # Save ONNX model
    onnx.save(model_proto, output_path)
    return output_path


def linear_regression_to_onnx(model: Dict[str, Any], output_path: str) -> str:
    """Convert Linear Regression model to ONNX format"""
    # Same structure as Ridge
    return ridge_to_onnx(model, output_path)


def logistic_regression_to_onnx(model: Dict[str, Any], output_path: str) -> str:
    """Convert Logistic Regression model to ONNX format"""
    n_features = model['n_features']
    weights = model['weights']
    bias = model['bias']
    
    # Create ONNX graph with sigmoid activation
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [None, n_features]
    )
    
    # Weights as initializer (transposed for MatMul: [n_features, 1])
    # Reshape weights from [n_features] to [n_features, 1]
    weights_2d = weights.reshape(n_features, 1)
    weights_tensor = helper.make_tensor(
        'weights',
        TensorProto.FLOAT,
        [n_features, 1],
        weights_2d.flatten().tolist()
    )
    
    bias_tensor = helper.make_tensor(
        'bias',
        TensorProto.FLOAT,
        [1],
        [float(bias)]
    )
    
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [None, 1]
    )
    
    # Graph: MatMul -> Add -> Sigmoid
    matmul_node = helper.make_node(
        'MatMul',
        ['input', 'weights'],
        ['matmul_output'],
        name='MatMul'
    )
    
    add_node = helper.make_node(
        'Add',
        ['matmul_output', 'bias'],
        ['add_output'],
        name='Add'
    )
    
    sigmoid_node = helper.make_node(
        'Sigmoid',
        ['add_output'],
        ['output'],
        name='Sigmoid'
    )
    
    graph = helper.make_graph(
        [matmul_node, add_node, sigmoid_node],
        'logistic_regression',
        [input_tensor],
        [output_tensor],
        [weights_tensor, bias_tensor]
    )
    
    model_proto = helper.make_model(graph, producer_name='NeuronDB')
    model_proto.opset_import[0].version = 14
    
    onnx.save(model_proto, output_path)
    return output_path


def convert_model_to_onnx(model_data: bytes, algorithm: str, output_path: str) -> str:
    """Main conversion function"""
    # Deserialize based on algorithm
    if algorithm == 'ridge':
        model = deserialize_ridge_model(model_data)
        return ridge_to_onnx(model, output_path)
    elif algorithm == 'lasso':
        model = deserialize_lasso_model(model_data)
        return ridge_to_onnx(model, output_path)  # Same structure
    elif algorithm == 'linear_regression':
        model = deserialize_linear_regression_model(model_data)
        return linear_regression_to_onnx(model, output_path)
    elif algorithm == 'logistic_regression':
        model = deserialize_logistic_regression_model(model_data)
        return logistic_regression_to_onnx(model, output_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: ridge, lasso, linear_regression, logistic_regression")


def main():
    """CLI interface"""
    if len(sys.argv) < 4:
        print("Usage: neurondb_to_onnx.py <model_data_file> <algorithm> <output_path>", file=sys.stderr)
        print("  algorithm: ridge, lasso, linear_regression, logistic_regression", file=sys.stderr)
        sys.exit(1)
    
    model_data_file = sys.argv[1]
    algorithm = sys.argv[2]
    output_path = sys.argv[3]
    
    # Read model data
    if not os.path.exists(model_data_file):
        print(f"ERROR: Model data file not found: {model_data_file}", file=sys.stderr)
        sys.exit(1)
    
    with open(model_data_file, 'rb') as f:
        model_data = f.read()
    
    # Convert to ONNX
    try:
        result_path = convert_model_to_onnx(model_data, algorithm, output_path)
        print(f"SUCCESS: Model converted to {result_path}")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

