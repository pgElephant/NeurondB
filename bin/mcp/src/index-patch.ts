/**
 * PATCH FILE: Updates needed to index.ts for complete reranking support
 * 
 * This file shows the exact changes needed in the main MCP server index.ts
 * to support all 7 reranking tools.
 */

// ============================================================================
// CHANGE 1: In setupTools() method, replace reranking tools section
// ============================================================================
// OLD CODE (around line 640-680):
/*
// Reranking Tools
tools.push({
    name: "mmr_rerank",
    ...
}, {
    name: "rerank_cross_encoder",
    ...
}, {
    name: "rerank_llm",
    ...
});
*/

// NEW CODE:
/*
// Reranking Tools - All 7 tools from RerankingTools.getToolDefinitions()
if (features.reranking?.enabled !== false) {
    const rerankingToolDefs = this.rerankingTools.getToolDefinitions();
    tools.push(...rerankingToolDefs);
}
*/

// ============================================================================
// CHANGE 2: In setupToolHandlers() method, replace reranking handler
// ============================================================================
// OLD CODE (around line 988-992):
/*
case "rerank_results":
    if (!features.rag?.enabled)
        throw new Error("RAG features are disabled");
    return this.rerankResults(args);
*/

// NEW CODE:
/*
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
*/

// ============================================================================
// CHANGE 3: Remove rerankResults() helper method (no longer needed)
// ============================================================================
// DELETE this method (around line 1030-1040):
/*
async rerankResults(args) {
    const { method = "cross_encoder", ...rest } = args;
    if (method === "cross_encoder") {
        return this.ragTools.rerankCrossEncoder(rest);
    } else {
        return this.ragTools.rerankLLM(rest);
    }
}
*/

// ============================================================================
// CHANGE 4: Update config.ts to add reranking feature flag
// ============================================================================
// In getDefaultConfig() or config interface, add:
/*
features: {
    reranking: {
        enabled: true,
    },
    // ... other features
}
*/

