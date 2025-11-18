#!/usr/bin/env node
/**
 * Final fix for MCP reranking tools - Clean up structure
 * 
 * Fixes:
 * 1. Ensures tool registration is only in setupTools()
 * 2. Ensures handlers are only in setupToolHandlers()
 * 3. Removes any misplaced code
 */

const fs = require('fs');
const path = require('path');

const indexJsPath = path.join(__dirname, 'dist', 'index.js');
let content = fs.readFileSync(indexJsPath, 'utf8');

console.log('Applying final fixes to reranking tools...');

// Find setupToolHandlers section
const handlersStart = content.indexOf('setupToolHandlers()');
const handlersEnd = content.indexOf('async connect()', handlersStart);

if (handlersStart > 0 && handlersEnd > 0) {
    const handlersSection = content.substring(handlersStart, handlersEnd);
    
    // Remove any tool registration code that got into handlers section
    // Look for "Reranking Tools - All 7 tools" in handlers section
    if (handlersSection.includes('Reranking Tools - All 7 tools')) {
        // Find and remove the misplaced registration
        const misplacedReg = /\/\/ Reranking Tools - All 7 tools[\s\S]{0,200}?\/\/ Indexing Tools/;
        content = content.replace(misplacedReg, '');
        console.log('✓ Removed misplaced tool registration from handlers section');
    }
    
    // Ensure handlers section has the unified reranking handlers
    if (!handlersSection.includes('case "rerank_colbert":')) {
        // Find where to insert (before create_ml_project)
        const insertPos = content.indexOf('case "create_ml_project":', handlersStart);
        if (insertPos > 0) {
            const unifiedHandlers = `
            // Reranking tools - All 7 tools routed through RerankingTools.handleToolCall()
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
            content = content.substring(0, insertPos) + unifiedHandlers + content.substring(insertPos);
            console.log('✓ Added unified reranking handlers');
        }
    }
}

// Ensure setupTools() has the registration
const toolsStart = content.indexOf('setupTools()');
const toolsEnd = content.indexOf('setupResources()', toolsStart);

if (toolsStart > 0 && toolsEnd > 0) {
    const toolsSection = content.substring(toolsStart, toolsEnd);
    
    // Check if getToolDefinitions() is used
    if (!toolsSection.includes('getToolDefinitions()')) {
        // Find the old reranking tools section and replace
        const oldRerankingRegex = /\/\/ Reranking Tools[\s\S]{0,800}?\/\/ Indexing Tools/;
        const newRegistration = `            // Reranking Tools - All 7 tools from RerankingTools.getToolDefinitions()
            // Includes: mmr_rerank, rerank_cross_encoder, rerank_llm (with full params),
            //           rerank_colbert, rerank_rrf, rerank_ensemble_weighted, rerank_ensemble_borda
            if (features.reranking?.enabled !== false) {
                const rerankingToolDefs = this.rerankingTools.getToolDefinitions();
                tools.push(...rerankingToolDefs);
            }
            // Indexing Tools`;
        
        if (oldRerankingRegex.test(toolsSection)) {
            content = content.substring(0, toolsStart) + 
                     toolsSection.replace(oldRerankingRegex, newRegistration) +
                     content.substring(toolsEnd);
            console.log('✓ Updated tool registration in setupTools()');
        }
    } else {
        console.log('✓ Tool registration already uses getToolDefinitions()');
    }
}

// Remove any remaining duplicate handlers
const duplicateHandlersRegex = /case "mmr_rerank":\s+return this\.rerankingTools\.mmrRerank\([^;]+;\s+case "rerank_cross_encoder":\s+return this\.rerankingTools\.rerankCrossEncoder\([^;]+;\s+case "rerank_llm":\s+return this\.rerankingTools\.rerankLLM\([^;]+;/g;
if (duplicateHandlersRegex.test(content)) {
    content = content.replace(duplicateHandlersRegex, '');
    console.log('✓ Removed duplicate individual handlers');
}

// Fix any syntax errors
content = content.replace(/\);\);/g, ');');
content = content.replace(/handleToolCall\(name, args\)\);\);/g, 'handleToolCall(name, args);');

// Write backup
const backupPath = indexJsPath + '.backup_final';
fs.writeFileSync(backupPath, fs.readFileSync(indexJsPath, 'utf8'));
console.log(`\nBackup: ${backupPath}`);

// Write fixed file
fs.writeFileSync(indexJsPath, content);
console.log(`Updated: ${indexJsPath}\n`);

// Final verification
console.log('Final Verification:');
const finalChecks = {
    'Tool registration uses getToolDefinitions()': content.includes('getToolDefinitions()') && 
                                                   content.indexOf('getToolDefinitions()') < content.indexOf('setupToolHandlers()'),
    'Handlers use handleToolCall()': content.includes('handleToolCall(name, args)') &&
                                      content.indexOf('handleToolCall') > content.indexOf('setupToolHandlers()'),
    'All 7 tools in handlers': content.includes('rerank_colbert') && 
                               content.includes('rerank_rrf') && 
                               content.includes('rerank_ensemble_weighted') && 
                               content.includes('rerank_ensemble_borda'),
    'LLM has promptTemplate in schema': content.includes('promptTemplate') || content.includes('prompt_template'),
    'LLM has temperature in schema': content.includes('temperature'),
    'No duplicate handlers': (content.match(/case "mmr_rerank":/g) || []).length <= 2,
    'No syntax errors': !content.includes('));)') && !content.includes('handleToolCall(name, args));)')
};

Object.entries(finalChecks).forEach(([check, passed]) => {
    console.log(`  ${passed ? '✓' : '✗'} ${check}`);
});

const allPassed = Object.values(finalChecks).every(v => v);
console.log(`\n${allPassed ? '✅ All checks passed! MCP reranking tools are complete.' : '⚠️  Some checks failed'}`);

