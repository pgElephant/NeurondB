# MCP Server Reranking Tools Update

## Overview
This document describes the updates needed to add complete reranking tool support to the NeuronDB MCP server.

## Current Status
- ✅ `RerankingTools` class created with all 7 methods
- ⚠️ MCP server only registers 3 reranking tools (incomplete)
- ❌ Missing: ColBERT, RRF, Ensemble tools in MCP registration
- ❌ Missing: Full LLM parameter support (promptTemplate, temperature)

## Required Updates

### 1. Update `setupTools()` method in `index.ts`

Replace the existing reranking tools section (around line 640-680) with:

```typescript
// Reranking Tools - Use RerankingTools.getToolDefinitions() for all tools
if (features.reranking?.enabled !== false) {
    const rerankingToolDefs = this.rerankingTools.getToolDefinitions();
    tools.push(...rerankingToolDefs);
}
```

This will automatically register all 7 reranking tools:
1. `mmr_rerank`
2. `rerank_cross_encoder`
3. `rerank_llm` (with full parameters)
4. `rerank_colbert`
5. `rerank_rrf`
6. `rerank_ensemble_weighted`
7. `rerank_ensemble_borda`

### 2. Update `setupToolHandlers()` method in `index.ts`

Replace the existing reranking handler (around line 988-992) with:

```typescript
// Reranking tools - route all to RerankingTools.handleToolCall()
case "mmr_rerank":
case "rerank_cross_encoder":
case "rerank_llm":
case "rerank_colbert":
case "rerank_rrf":
case "rerank_ensemble_weighted":
case "rerank_ensemble_borda":
    if (features.reranking?.enabled === false) {
        throw new Error("Reranking features are disabled");
    }
    return await this.rerankingTools.handleToolCall(name, args);
```

### 3. Update Configuration

Add reranking feature flag to `config.ts`:

```typescript
features: {
    reranking: {
        enabled: true,
    },
    // ... other features
}
```

## Tool Definitions

All tool definitions are provided by `RerankingTools.getToolDefinitions()`:

### 1. mmr_rerank
- **Parameters**: table, queryVector, vectorColumn, lambda (default: 0.5), topK (default: 10)
- **Description**: Maximal Marginal Relevance for diversity-aware reranking

### 2. rerank_cross_encoder
- **Parameters**: query, documents, model (optional), topK (optional)
- **Description**: Neural reranking using cross-encoder models

### 3. rerank_llm
- **Parameters**: query, documents, model (optional), topK (optional), promptTemplate (optional), temperature (optional, 0.0-2.0)
- **Description**: GPT/Claude-powered reranking with full parameter support

### 4. rerank_colbert
- **Parameters**: query, documents, model (optional), topK (optional), nbits (optional, 1-8), kmeansNiters (optional, 1-100)
- **Description**: ColBERT late interaction model for token-level reranking

### 5. rerank_rrf
- **Parameters**: rankingTables, idColumn (optional), rankColumn (optional), k (optional, default: 60)
- **Description**: Reciprocal Rank Fusion for combining multiple rankings

### 6. rerank_ensemble_weighted
- **Parameters**: modelTables, weights (optional), idColumn (optional), scoreColumn (optional), normalize (optional, default: true)
- **Description**: Weighted ensemble combining multiple ranking models

### 7. rerank_ensemble_borda
- **Parameters**: modelTables, idColumn (optional), scoreColumn (optional)
- **Description**: Borda count ensemble for combining rankings

## Implementation Files

- **Source**: `bin/mcp/src/tools/reranking.ts` - Complete RerankingTools class
- **Update**: `bin/mcp/dist/index.js` (or `bin/mcp/src/index.ts` if source exists) - MCP server registration

## Testing

After updates, test all tools:

```bash
# Test MMR
# Test Cross-Encoder
# Test LLM (with all parameters)
# Test ColBERT
# Test RRF
# Test Ensemble Weighted
# Test Ensemble Borda
```

## Notes

- All tools use `neurondb.` schema prefix for SQL functions
- Error handling is built into RerankingTools methods
- All tools return arrays of result objects
- Parameter validation is handled by MCP schema validation

