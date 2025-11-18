#!/usr/bin/env node
/**
 * Script to update MCP server with complete reranking tools support
 * 
 * This script patches the compiled index.js to:
 * 1. Register all 7 reranking tools using RerankingTools.getToolDefinitions()
 * 2. Route all reranking tool calls to RerankingTools.handleToolCall()
 */

const fs = require('fs');
const path = require('path');

const indexJsPath = path.join(__dirname, 'dist', 'index.js');
const indexJsContent = fs.readFileSync(indexJsPath, 'utf8');

// Find the reranking tools section and replace it
const rerankingToolsStart = indexJsContent.indexOf('// Reranking Tools');
const rerankingToolsEnd = indexJsContent.indexOf('// Indexing Tools', rerankingToolsStart);

if (rerankingToolsStart === -1 || rerankingToolsEnd === -1) {
    console.error('Could not find reranking tools section');
    process.exit(1);
}

// New reranking tools registration using getToolDefinitions()
const newRerankingToolsSection = `            // Reranking Tools - All 7 tools from RerankingTools.getToolDefinitions()
            if (features.reranking?.enabled !== false) {
                const rerankingToolDefs = this.rerankingTools.getToolDefinitions();
                tools.push(...rerankingToolDefs);
            }`;

// Find and replace the reranking handler
const rerankResultsHandler = indexJsContent.indexOf('case "rerank_results":');
const rerankResultsHandlerEnd = indexJsContent.indexOf('return this.rerankResults(args);', rerankResultsHandler) + 30;

if (rerankResultsHandler === -1) {
    console.error('Could not find rerank_results handler');
    process.exit(1);
}

// New reranking handlers section
const newRerankingHandlers = `            // Reranking tools - route all to RerankingTools.handleToolCall()
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
                return await this.rerankingTools.handleToolCall(name, args);`;

// Create updated content
let updatedContent = indexJsContent.substring(0, rerankingToolsStart) +
    newRerankingToolsSection +
    indexJsContent.substring(rerankingToolsEnd);

// Update handlers
const handlersStart = updatedContent.indexOf('case "rerank_results":');
if (handlersStart !== -1) {
    const handlersEnd = updatedContent.indexOf('return this.rerankResults(args);', handlersStart) + 30;
    updatedContent = updatedContent.substring(0, handlersStart) +
        newRerankingHandlers +
        updatedContent.substring(handlersEnd);
}

// Write backup
const backupPath = indexJsPath + '.backup';
fs.writeFileSync(backupPath, indexJsContent);
console.log(`Backup created: ${backupPath}`);

// Write updated file
fs.writeFileSync(indexJsPath, updatedContent);
console.log(`Updated: ${indexJsPath}`);
console.log('Reranking tools update complete!');

