# Complete Training Workflow Example

This document shows a complete end-to-end workflow for training ML models using the NeuronDB MCP server.

## Prerequisites

1. NeuronDB extension installed in PostgreSQL
2. MCP server running
3. Training data prepared

## Step-by-Step Workflow

### 1. Prepare Training Data

```sql
-- Create training table
CREATE TABLE sample_train (
    id SERIAL PRIMARY KEY,
    features vector(10),
    label float8
);

-- Insert sample data
INSERT INTO sample_train (features, label) VALUES
    ('[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]', 100.0),
    ('[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]', 110.0),
    ('[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]', 120.0);
```

### 2. Enable GPU (Optional but Recommended)

**MCP Tool Call:**
```json
{
  "name": "gpu_enable",
  "arguments": {
    "enabled": true
  }
}
```

**Verify GPU:**
```json
{
  "name": "gpu_info",
  "arguments": {}
}
```

### 3. Train Model

**MCP Tool Call:**
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

### 4. Check Model Information

**MCP Tool Call:**
```json
{
  "name": "get_model_info",
  "arguments": {
    "model_id": 1
  }
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
  }
]
```

### 5. Make Predictions

**MCP Tool Call:**
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

### 6. Evaluate Model (SQL)

```sql
-- Create test set
CREATE TABLE sample_test (
    id SERIAL PRIMARY KEY,
    features vector(10),
    label float8
);

-- Evaluate predictions
SELECT 
    id,
    label,
    neurondb_predict(1, features::real[]) AS prediction,
    ABS(label - neurondb_predict(1, features::real[])) AS error
FROM sample_test;
```

### 7. Monitor GPU Usage

**MCP Tool Call:**
```json
{
  "name": "gpu_stats",
  "arguments": {}
}
```

## Using Different Algorithms

### Ridge Regression
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

### Random Forest
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

### Logistic Regression
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

## Complete Example Script

See `training-example.js` for a complete Node.js example.

## Tips

1. **Always check GPU availability** before training large models
2. **Save model_id** after training for later predictions
3. **Use appropriate hyperparameters** for your use case
4. **Evaluate on test set** to avoid overfitting
5. **Monitor GPU stats** to track performance

