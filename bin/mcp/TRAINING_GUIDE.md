# NeuronDB MCP Server - Training Guide

Complete guide on how to train ML models using the NeuronDB MCP server.

> **💡 Don't need MCP?** You can also use SQL directly! See [USAGE_GUIDE.md](USAGE_GUIDE.md) for all usage methods.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Available Algorithms](#available-algorithms)
3. [Training Examples](#training-examples)
4. [GPU Training](#gpu-training)
5. [Model Management](#model-management)
6. [Making Predictions](#making-predictions)
7. [Best Practices](#best-practices)

## Quick Start

### 1. Prepare Your Data

First, ensure you have a table with features and labels:

```sql
-- Example: Create training table
CREATE TABLE sample_train (
    id SERIAL PRIMARY KEY,
    features vector(10),  -- Your feature vector
    label float8          -- Your target variable
);

-- Insert training data
INSERT INTO sample_train (features, label) VALUES
    ('[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]', 100.0),
    ('[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]', 110.0);
```

### 2. Train a Model via MCP

Use the `train_ml_model` tool:

```json
{
  "name": "train_ml_model",
  "arguments": {
    "table": "sample_train",
    "feature_col": "features",
    "label_col": "label",
    "algorithm": "linear_regression"
  }
}
```

**Response:**
```json
{
  "model_id": 1,
  "algorithm": "linear_regression",
  "status": "trained"
}
```

## Available Algorithms

The MCP server supports **9 ML algorithms**:

### Regression Algorithms
1. **linear_regression** - Simple linear regression
2. **ridge** - Ridge regression (L2 regularization)
3. **lasso** - Lasso regression (L1 regularization)

### Classification Algorithms
4. **logistic** - Logistic regression
5. **random_forest** - Random Forest classifier
6. **svm** - Support Vector Machine
7. **knn** - K-Nearest Neighbors
8. **decision_tree** - Decision Tree classifier
9. **naive_bayes** - Naive Bayes classifier

## Training Examples

### Example 1: Linear Regression

```json
{
  "name": "train_ml_model",
  "arguments": {
    "table": "sample_train",
    "feature_col": "features",
    "label_col": "label",
    "algorithm": "linear_regression"
  }
}
```

### Example 2: Ridge Regression with Hyperparameters

```json
{
  "name": "train_ml_model",
  "arguments": {
    "table": "sample_train",
    "feature_col": "features",
    "label_col": "label",
    "algorithm": "ridge",
    "params": {
      "alpha": 0.5
    }
  }
}
```

### Example 3: Random Forest with Custom Parameters

```json
{
  "name": "train_ml_model",
  "arguments": {
    "table": "sample_train",
    "feature_col": "features",
    "label_col": "label",
    "algorithm": "random_forest",
    "params": {
      "n_estimators": 200,
      "max_depth": 15,
      "min_samples_split": 5
    }
  }
}
```

### Example 4: Logistic Regression

```json
{
  "name": "train_ml_model",
  "arguments": {
    "table": "sample_train",
    "feature_col": "features",
    "label_col": "label",
    "algorithm": "logistic",
    "params": {
      "max_iter": 2000,
      "learning_rate": 0.01,
      "tolerance": 0.0001
    }
  }
}
```

### Example 5: SVM Classifier

```json
{
  "name": "train_ml_model",
  "arguments": {
    "table": "sample_train",
    "feature_col": "features",
    "label_col": "label",
    "algorithm": "svm",
    "params": {
      "C": 1.5,
      "max_iters": 2000
    }
  }
}
```

### Example 6: K-Nearest Neighbors

```json
{
  "name": "train_ml_model",
  "arguments": {
    "table": "sample_train",
    "feature_col": "features",
    "label_col": "label",
    "algorithm": "knn",
    "params": {
      "k": 7
    }
  }
}
```

## Algorithm-Specific Parameters

### Linear Regression
- No parameters required

### Ridge Regression
- `alpha` (default: 1.0) - Regularization strength

### Lasso Regression
- `alpha` (default: 1.0) - Regularization strength
- `max_iter` (default: 1000) - Maximum iterations

### Logistic Regression
- `max_iter` (default: 1000) - Maximum iterations
- `learning_rate` (default: 0.01) - Learning rate
- `tolerance` (default: 0.001) - Convergence tolerance

### Random Forest
- `n_estimators` (default: 100) - Number of trees
- `max_depth` (default: 10) - Maximum tree depth
- `min_samples_split` (default: 2) - Minimum samples to split

### SVM
- `C` (default: 1.0) - Regularization parameter
- `max_iters` (default: 1000) - Maximum iterations

### KNN
- `k` (default: 5) - Number of neighbors

### Decision Tree
- `max_depth` (default: 10) - Maximum tree depth
- `min_samples_split` (default: 2) - Minimum samples to split

### Naive Bayes
- No parameters required

## GPU Training

NeuronDB supports GPU-accelerated training. Enable GPU before training:

### Step 1: Enable GPU

```json
{
  "name": "gpu_enable",
  "arguments": {
    "enabled": true
  }
}
```

### Step 2: Check GPU Status

```json
{
  "name": "gpu_info",
  "arguments": {}
}
```

**Response:**
```json
{
  "backend": "metal",
  "device_name": "Apple M2",
  "available": true,
  "enabled": true
}
```

### Step 3: Train with GPU

Training will automatically use GPU if available:

```json
{
  "name": "train_ml_model",
  "arguments": {
    "table": "sample_train",
    "feature_col": "features",
    "label_col": "label",
    "algorithm": "linear_regression"
  }
}
```

### Step 4: Verify GPU Usage

Check model metrics to confirm GPU training:

```json
{
  "name": "get_model_info",
  "arguments": {
    "model_id": 1
  }
}
```

Look for `"storage": "gpu"` in the metrics.

## Model Management

### List All Models

```json
{
  "name": "get_model_info",
  "arguments": {}
}
```

**Response:**
```json
[
  {
    "model_id": 1,
    "algorithm": "linear_regression",
    "training_table": "sample_train",
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-15T10:30:00Z"
  },
  {
    "model_id": 2,
    "algorithm": "random_forest",
    "training_table": "sample_train",
    "created_at": "2025-01-15T11:00:00Z",
    "updated_at": "2025-01-15T11:00:00Z"
  }
]
```

### Get Specific Model Info

```json
{
  "name": "get_model_info",
  "arguments": {
    "model_id": 1
  }
}
```

### Access Model Metrics via SQL

You can also query model details directly:

```sql
SELECT 
    model_id,
    algorithm,
    metrics::jsonb->>'storage' as storage_type,
    metrics::jsonb->>'r_squared' as r_squared,
    metrics::jsonb->>'mse' as mse
FROM neurondb.ml_models
WHERE model_id = 1;
```

## Making Predictions

After training, use the model to make predictions:

### Single Prediction

```json
{
  "name": "predict_ml_model",
  "arguments": {
    "model_id": 1,
    "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  }
}
```

**Response:**
```json
{
  "prediction": 100.5
}
```

### Batch Predictions via SQL

For batch predictions, use SQL directly:

```sql
SELECT 
    id,
    features,
    neurondb_predict(1, features::real[]) AS prediction
FROM sample_test;
```

## Complete Training Workflow

### Step 1: Prepare Data

```sql
-- Create training and test sets
CREATE TABLE sample_train (
    id SERIAL PRIMARY KEY,
    features vector(10),
    label float8
);

CREATE TABLE sample_test (
    id SERIAL PRIMARY KEY,
    features vector(10),
    label float8
);
```

### Step 2: Train Model (via MCP)

```json
{
  "name": "train_ml_model",
  "arguments": {
    "table": "sample_train",
    "feature_col": "features",
    "label_col": "label",
    "algorithm": "linear_regression"
  }
}
```

Save the `model_id` from the response.

### Step 3: Evaluate Model

```sql
-- Evaluate on test set
SELECT 
    AVG(ABS(label - neurondb_predict(1, features::real[]))) AS mae,
    SQRT(AVG(POWER(label - neurondb_predict(1, features::real[]), 2))) AS rmse
FROM sample_test;
```

### Step 4: Make Predictions

```json
{
  "name": "predict_ml_model",
  "arguments": {
    "model_id": 1,
    "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  }
}
```

## Best Practices

### 1. Data Preparation
- Ensure feature vectors have consistent dimensions
- Handle missing values before training
- Normalize features for better performance
- Split data into train/test sets (80/20 or 70/30)

### 2. Algorithm Selection
- **Linear Regression**: Simple, fast, interpretable
- **Ridge/Lasso**: When you have many features (regularization)
- **Random Forest**: Good default, handles non-linearity
- **SVM**: For complex decision boundaries
- **KNN**: Simple, good for small datasets
- **Logistic Regression**: Binary classification
- **Naive Bayes**: Text classification, fast

### 3. Hyperparameter Tuning
- Start with default parameters
- Use cross-validation for parameter selection
- Monitor training metrics (R², accuracy, etc.)
- Use GPU for faster iteration

### 4. Model Evaluation
- Always evaluate on held-out test set
- Use appropriate metrics:
  - Regression: R², MSE, MAE, RMSE
  - Classification: Accuracy, Precision, Recall, F1
- Check for overfitting (train vs test performance)

### 5. GPU Usage
- Enable GPU for large datasets (>10k samples)
- Monitor GPU memory usage
- Use GPU stats to track performance
- Fallback to CPU if GPU unavailable

### 6. Model Management
- Track model versions
- Store training metadata
- Monitor model performance over time
- Retrain periodically with new data

## Troubleshooting

### Error: "Table does not exist"
- Ensure table name is correct
- Check database connection

### Error: "Column does not exist"
- Verify column names match exactly
- Check column types (features should be `vector`, label should be `float8`)

### Error: "No training data found"
- Ensure table has data
- Check for NULL values in features/labels

### GPU Not Available
- Check GPU backend (Metal/CUDA/ROCm)
- Verify GPU is enabled: `gpu_info` tool
- Training will fallback to CPU automatically

### Model Training Fails
- Check data quality (no NULLs, correct types)
- Verify feature dimensions match
- Reduce dataset size for testing
- Check algorithm parameters

## Advanced Usage

### Using SQL Directly

You can also train models using SQL directly:

```sql
-- Train linear regression
SELECT train_linear_regression('sample_train', 'features', 'label') AS model_id;

-- Train with GPU (if enabled)
SET neurondb.gpu_enabled = on;
SELECT train_linear_regression('sample_train', 'features', 'label') AS model_id;
```

### Model Metrics

Access detailed model metrics:

```sql
SELECT 
    model_id,
    algorithm,
    metrics::jsonb->>'storage' as storage,
    metrics::jsonb->>'r_squared' as r_squared,
    metrics::jsonb->>'mse' as mse,
    metrics::jsonb->>'mae' as mae
FROM neurondb.ml_models
WHERE model_id = 1;
```

## Resources

- [NeuronDB Documentation](https://pgelephant.com/neurondb)
- [MCP Server README](README.md)
- [Features Overview](FEATURES.md)

