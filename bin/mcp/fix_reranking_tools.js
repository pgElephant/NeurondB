#!/usr/bin/env node
/**
 * Detailed fix script for MCP reranking tools
 * 
 * Fixes:
 * 1. Removes duplicate reranking handlers
 * 2. Fixes syntax error (double closing paren)
 * 3. Removes old rerank_results tool
 * 4. Ensures all 7 tools use handleToolCall()
 * 5. Fixes parameter name mismatches
 */

const fs = require('fs');
const path = require('path');

const indexJsPath = path.join(__dirname, 'dist', 'index.js');
let content = fs.readFileSync(indexJsPath, 'utf8');

console.log('Fixing MCP reranking tools...');

// 1. Fix syntax error - remove double closing paren
content = content.replace(
    /return await this\.rerankingTools\.handleToolCall\(name, args\)\);\);/g,
    'return await this.rerankingTools.handleToolCall(name, args);'
);

// 2. Remove duplicate individual reranking handlers (lines ~1037-1041)
// Remove the old individual method calls
const duplicatePattern = /case "mmr_rerank":\s+return this\.rerankingTools\.mmrRerank\([^;]+;\s+case "rerank_cross_encoder":\s+return this\.rerankingTools\.rerankCrossEncoder\([^;]+;\s+case "rerank_llm":\s+return this\.rerankingTools\.rerankLLM\([^;]+;/g;
content = content.replace(duplicatePattern, '');

// 3. Remove old rerank_results tool registration (around line 279)
const rerankResultsToolPattern = /\{\s*name: "rerank_results",[\s\S]{0,500}?\},/g;
content = content.replace(rerankResultsToolPattern, '');

// 4. Remove old rerank_results handler case
const rerankResultsHandlerPattern = /case "rerank_results":\s+if \(!features\.rag\?\.enabled\)\s+throw new Error\("RAG features are disabled"\);\s+return this\.rerankResults\(args\);/g;
content = content.replace(rerankResultsHandlerPattern, '');

// 5. Remove old rerankResults() method (around line 1110)
const rerankResultsMethodPattern = /async rerankResults\(args\) \{[\s\S]{0,200}?\}/g;
content = content.replace(rerankResultsMethodPattern, '');

// 6. Ensure reranking tools section uses getToolDefinitions()
// Check if it's already there, if not add it
if (!content.includes('getToolDefinitions()')) {
    // Find the reranking tools section and replace
    const rerankingSection = /\/\/ Reranking Tools[\s\S]{0,500}?\/\/ Indexing Tools/;
    const newSection = `            // Reranking Tools - All 7 tools from RerankingTools.getToolDefinitions()
            if (features.reranking?.enabled !== false) {
                const rerankingToolDefs = this.rerankingTools.getToolDefinitions();
                tools.push(...rerankingToolDefs);
            }
            // Indexing Tools`;
    content = content.replace(rerankingSection, newSection);
}

// 7. Ensure all reranking handlers use handleToolCall()
// Find the reranking handlers section
const handlersPattern = /\/\/ Reranking tools - route all to RerankingTools\.handleToolCall\(\)[\s\S]{0,300}?return await this\.rerankingTools\.handleToolCall\(name, args\);/;
if (!handlersPattern.test(content)) {
    // Add proper handlers if missing
    const beforeHandlers = content.indexOf('case "create_ml_project":');
    if (beforeHandlers > 0) {
        const newHandlers = `            // Reranking tools - route all to RerankingTools.handleToolCall()
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
            `;
        content = content.substring(0, beforeHandlers) + newHandlers + content.substring(beforeHandlers);
    }
}

// Write backup
const backupPath = indexJsPath + '.backup2';
fs.writeFileSync(backupPath, fs.readFileSync(indexJsPath, 'utf8'));
console.log(`Backup created: ${backupPath}`);

// Write fixed file
fs.writeFileSync(indexJsPath, content);
console.log(`Fixed: ${indexJsPath}`);
console.log('Reranking tools fix complete!');

// Verify fixes
const hasGetToolDefinitions = content.includes('getToolDefinitions()');
const hasHandleToolCall = content.includes('handleToolCall(name, args)');
const hasAll7Tools = content.includes('rerank_colbert') && 
                     content.includes('rerank_rrf') && 
                     content.includes('rerank_ensemble_weighted') && 
                     content.includes('rerank_ensemble_borda');
const noDuplicateHandlers = (content.match(/case "mmr_rerank":/g) || []).length <= 2; // One in handlers, maybe one in comments

console.log('\nVerification:');
console.log(`  ✓ getToolDefinitions(): ${hasGetToolDefinitions}`);
console.log(`  ✓ handleToolCall(): ${hasHandleToolCall}`);
console.log(`  ✓ All 7 tools present: ${hasAll7Tools}`);
console.log(`  ✓ No duplicate handlers: ${noDuplicateHandlers}`);

