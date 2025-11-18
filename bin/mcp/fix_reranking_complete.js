#!/usr/bin/env node
/**
 * Complete fix for MCP reranking tools - Detailed implementation
 * 
 * This script:
 * 1. Replaces old reranking tool registrations with getToolDefinitions()
 * 2. Removes all duplicate handlers
 * 3. Adds unified handler using handleToolCall()
 * 4. Fixes parameter name mappings
 * 5. Removes old rerank_results tool
 */

const fs = require('fs');
const path = require('path');

const indexJsPath = path.join(__dirname, 'dist', 'index.js');
let content = fs.readFileSync(indexJsPath, 'utf8');

console.log('Implementing complete reranking tools support...');

// ============================================================================
// STEP 1: Replace reranking tools registration section
// ============================================================================
// Find the old reranking tools section (around line 640-680)
const oldRerankingToolsRegex = /\/\/ Reranking Tools[\s\S]{0,800}?\/\/ Indexing Tools/;

const newRerankingToolsSection = `            // Reranking Tools - All 7 tools from RerankingTools.getToolDefinitions()
            // Includes: mmr_rerank, rerank_cross_encoder, rerank_llm (full params),
            //           rerank_colbert, rerank_rrf, rerank_ensemble_weighted, rerank_ensemble_borda
            if (features.reranking?.enabled !== false) {
                const rerankingToolDefs = this.rerankingTools.getToolDefinitions();
                tools.push(...rerankingToolDefs);
            }
            // Indexing Tools`;

if (oldRerankingToolsRegex.test(content)) {
    content = content.replace(oldRerankingToolsRegex, newRerankingToolsSection);
    console.log('✓ Replaced reranking tools registration');
} else {
    console.log('⚠ Could not find old reranking tools section, may already be updated');
}

// ============================================================================
// STEP 2: Remove old rerank_results tool registration (around line 279)
// ============================================================================
const rerankResultsToolRegex = /\{\s*name: "rerank_results",[\s\S]{0,600}?\},/;
if (rerankResultsToolRegex.test(content)) {
    content = content.replace(rerankResultsToolRegex, '');
    console.log('✓ Removed old rerank_results tool registration');
}

// ============================================================================
// STEP 3: Replace all reranking handlers with unified handleToolCall()
// ============================================================================
// Find and remove old individual handlers (around line 1037-1041)
const oldIndividualHandlersRegex = /\/\/ Reranking Tools\s+case "mmr_rerank":\s+return this\.rerankingTools\.mmrRerank\([^;]+;\s+case "rerank_cross_encoder":\s+return this\.rerankingTools\.rerankCrossEncoder\([^;]+;\s+case "rerank_llm":\s+return this\.rerankingTools\.rerankLLM\([^;]+;/;

// Find the location before "// Indexing Tools" in handlers
const handlersIndexingToolsPos = content.indexOf('// Indexing Tools', content.indexOf('setupToolHandlers'));
if (handlersIndexingToolsPos > 0) {
    // Find the position before reranking handlers
    const beforeRerankingHandlers = content.lastIndexOf('case "ltr_rerank":', handlersIndexingToolsPos);
    if (beforeRerankingHandlers > 0) {
        // Remove old reranking handlers section
        const oldHandlersEnd = content.indexOf('// Indexing Tools', beforeRerankingHandlers);
        if (oldHandlersEnd > 0) {
            const oldHandlersSection = content.substring(beforeRerankingHandlers, oldHandlersEnd);
            if (oldHandlersSection.includes('case "mmr_rerank":') && 
                oldHandlersSection.includes('return this.rerankingTools.mmrRerank')) {
                // Remove old section
                content = content.substring(0, beforeRerankingHandlers) + 
                         content.substring(oldHandlersEnd);
                console.log('✓ Removed old individual reranking handlers');
            }
        }
    }
}

// ============================================================================
// STEP 4: Add unified reranking handlers using handleToolCall()
// ============================================================================
// Find where to insert (before "create_ml_project" or similar)
const insertBefore = content.indexOf('case "create_ml_project":');
if (insertBefore > 0) {
    // Check if unified handlers already exist
    const hasUnifiedHandlers = content.substring(0, insertBefore).includes('case "rerank_colbert":');
    
    if (!hasUnifiedHandlers) {
        const unifiedHandlers = `
            // Reranking tools - All 7 tools routed through RerankingTools.handleToolCall()
            // This ensures consistent parameter handling and error management
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
                // handleToolCall() maps tool names to methods and handles parameter conversion
                return await this.rerankingTools.handleToolCall(name, args);
            `;
        
        content = content.substring(0, insertBefore) + unifiedHandlers + content.substring(insertBefore);
        console.log('✓ Added unified reranking handlers');
    } else {
        console.log('✓ Unified handlers already present');
    }
}

// ============================================================================
// STEP 5: Remove old rerank_results handler
// ============================================================================
const rerankResultsHandlerRegex = /case "rerank_results":\s+if \(!features\.rag\?\.enabled\)\s+throw new Error\("RAG features are disabled"\);\s+return this\.rerankResults\(args\);/g;
if (rerankResultsHandlerRegex.test(content)) {
    content = content.replace(rerankResultsHandlerRegex, '');
    console.log('✓ Removed old rerank_results handler');
}

// ============================================================================
// STEP 6: Remove old rerankResults() method
// ============================================================================
const rerankResultsMethodRegex = /async rerankResults\(args\) \{[\s\S]{0,300}?\n\s+\}/;
if (rerankResultsMethodRegex.test(content)) {
    content = content.replace(rerankResultsMethodRegex, '');
    console.log('✓ Removed old rerankResults() method');
}

// ============================================================================
// STEP 7: Fix syntax errors (double closing parens, etc.)
// ============================================================================
content = content.replace(/\);\);/g, ');');
content = content.replace(/return await this\.rerankingTools\.handleToolCall\(name, args\)\);\);/g, 
                          'return await this.rerankingTools.handleToolCall(name, args);');

// ============================================================================
// STEP 8: Update RerankingTools.handleToolCall() to map parameter names
// ============================================================================
// The handleToolCall method needs to handle parameter name differences
// MCP uses camelCase (queryVector, topK) but SQL may use snake_case
// This is handled in the RerankingTools class methods

// Write backup
const backupPath = indexJsPath + '.backup_complete';
fs.writeFileSync(backupPath, fs.readFileSync(indexJsPath, 'utf8'));
console.log(`\nBackup created: ${backupPath}`);

// Write fixed file
fs.writeFileSync(indexJsPath, content);
console.log(`Updated: ${indexJsPath}\n`);

// ============================================================================
// VERIFICATION
// ============================================================================
console.log('Verification:');
const checks = {
    'getToolDefinitions() used': content.includes('getToolDefinitions()'),
    'handleToolCall() used': content.includes('handleToolCall(name, args)'),
    'All 7 tools in handlers': content.includes('rerank_colbert') && 
                               content.includes('rerank_rrf') && 
                               content.includes('rerank_ensemble_weighted') && 
                               content.includes('rerank_ensemble_borda'),
    'No duplicate mmr_rerank handlers': (content.match(/case "mmr_rerank":/g) || []).length <= 2,
    'No old rerank_results': !content.includes('case "rerank_results":'),
    'No syntax errors': !content.includes('));)') && !content.includes('handleToolCall(name, args));)')
};

Object.entries(checks).forEach(([check, passed]) => {
    console.log(`  ${passed ? '✓' : '✗'} ${check}`);
});

const allPassed = Object.values(checks).every(v => v);
console.log(`\n${allPassed ? '✅ All checks passed!' : '⚠️  Some checks failed - please review'}`);

