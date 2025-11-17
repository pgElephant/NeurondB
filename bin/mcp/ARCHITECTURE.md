# Neurondb MCP Server Architecture

## Overview

The Neurondb MCP server is built with a clean, modular architecture that separates concerns and makes the codebase maintainable and extensible.

## Module Structure

### Core Modules

#### `src/index.ts`
Main server entry point that:
- Initializes the MCP server
- Sets up tool and resource handlers
- Manages database connection
- Routes tool calls to appropriate handlers

#### `src/db.ts`
Database connection management:
- `Database` class encapsulates PostgreSQL connection
- Handles connection pooling
- Provides query execution interface
- Identifier escaping for SQL safety

#### `src/types.ts`
TypeScript type definitions:
- `NeurondbConfig` - Database configuration
- Tool parameter interfaces
- Request/response types

### Tool Modules (`src/tools/`)

#### `src/tools/vector.ts` - VectorTools
Vector operations and search:
- `vectorSearch()` - Similarity search with multiple distance metrics
- `generateEmbedding()` - Single text embedding
- `batchEmbedding()` - Batch embedding generation
- `createHNSWIndex()` - Index creation
- `hybridSearch()` - Combined vector + text search

#### `src/tools/ml.ts` - MLTools
Machine learning model operations:
- Training: `trainLinearRegression()`, `trainRidgeRegression()`, `trainLassoRegression()`, `trainLogisticRegression()`, `trainRandomForest()`, `trainSVM()`, `trainKNN()`, `trainDecisionTree()`, `trainNaiveBayes()`
- Prediction: `predict()`
- Model info: `getModelInfo()`

#### `src/tools/analytics.ts` - AnalyticsTools
Analytics and clustering:
- `clusterKMeans()` - K-means clustering
- `clusterMiniBatchKMeans()` - Mini-batch K-means
- `clusterGMM()` - Gaussian Mixture Model
- `detectOutliersZScore()` - Outlier detection
- `computePCA()` - Principal Component Analysis

#### `src/tools/rag.ts` - RAGTools
RAG pipeline operations:
- `chunkText()` - Text chunking
- `rerankCrossEncoder()` - Cross-encoder reranking
- `rerankLLM()` - LLM-based reranking

#### `src/tools/projects.ts` - ProjectTools
ML project management:
- `createProject()` - Create ML project
- `listProjects()` - List all projects
- `getProjectInfo()` - Get project details
- `trainKMeansProject()` - Train within project
- `deployModel()` - Deploy model version
- `listProjectModels()` - List project models

#### `src/tools/index.ts`
Central export for all tool classes

### Resource Module

#### `src/resources.ts` - Resources
Resource data providers:
- `getSchema()` - Database schema
- `getModels()` - ML models catalog
- `getIndexes()` - Vector indexes
- `getConfig()` - Configuration settings
- `getWorkerStatus()` - Background workers
- `getVectorStats()` - Vector statistics
- `getIndexHealth()` - Index health

## Design Principles

### 1. Separation of Concerns
- Each module has a single, well-defined responsibility
- Database operations isolated in `db.ts`
- Tool logic separated by domain (vector, ML, analytics, etc.)

### 2. Dependency Injection
- Tool classes receive `Database` instance via constructor
- Enables easy testing and mocking
- Loose coupling between modules

### 3. Type Safety
- Comprehensive TypeScript types
- Parameter validation through types
- Clear interfaces for all operations

### 4. Error Handling
- Consistent error handling across modules
- User-friendly error messages
- Proper error propagation

### 5. Extensibility
- Easy to add new tools (create new class in `tools/`)
- Easy to add new resources (extend `Resources` class)
- Modular structure supports growth

## Data Flow

```
MCP Client Request
    ↓
index.ts (route to handler)
    ↓
Tool Class (e.g., VectorTools)
    ↓
Database (execute query)
    ↓
PostgreSQL/Neurondb
    ↓
Response (JSON)
    ↓
MCP Client
```

## Adding New Features

### Adding a New Tool

1. Create method in appropriate tool class (or new class)
2. Add tool definition to `setupTools()` in `index.ts`
3. Add handler case in `setupToolHandlers()`
4. Export from `tools/index.ts` if new class

### Adding a New Resource

1. Add method to `Resources` class
2. Add resource definition to `setupResources()`
3. Add handler case in `ReadResourceRequestSchema` handler

## Testing Strategy

Each module can be tested independently:
- Mock `Database` class for tool tests
- Test tool classes in isolation
- Integration tests with real database

## Future Enhancements

- Add caching layer for frequently accessed data
- Add connection pooling configuration
- Add query result pagination
- Add batch operation support
- Add transaction support
- Add comprehensive logging

