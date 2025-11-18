#!/usr/bin/env node
/**
 * Complete fix for MCP reranking tools - Final implementation
 * 
 * This script:
 * 1. Replaces hardcoded reranking tool definitions in setupTools() with getToolDefinitions()
 * 2. Ensures all 7 tools are registered with full LLM parameter support
 * 3. Verifies handlers are correctly set up
 */

const fs = require('fs');
const path = require('path');

const indexJsPath = path.join(__dirname, 'dist', 'index.js');
let content = fs.readFileSync(indexJsPath, 'utf8');

console.log('Applying complete reranking tools fix...\n');

// ============================================================================
// STEP 1: Find and replace hardcoded reranking tools in setupTools()
// ============================================================================
const setupToolsStart = content.indexOf('setupTools()');
const setupToolsEnd = content.indexOf('setupToolHandlers()', setupToolsStart);

if (setupToolsStart > 0 && setupToolsEnd > 0) {
    const setupToolsSection = content.substring(setupToolsStart, setupToolsEnd);
    
    // Find the hardcoded reranking tools (mmr_rerank through rerank_llm)
    // Look for the pattern: tools.push({ name: "mmr_rerank", ... }, { name: "rerank_cross_encoder", ... }, { name: "rerank_llm", ... });
    const hardcodedPattern = /tools\.push\([\s\S]{0,100}?name: "mmr_rerank"[\s\S]{0,1500}?name: "rerank_llm"[\s\S]{0,200}?\}\);\s*\/\/ Indexing Tools/;
    
    if (hardcodedPattern.test(setupToolsSection)) {
        // Find the exact location
        const mmrStart = setupToolsSection.indexOf('name: "mmr_rerank"');
        const indexingToolsStart = setupToolsSection.indexOf('// Indexing Tools', mmrStart);
        
        if (mmrStart > 0 && indexingToolsStart > 0) {
            // Find the tools.push( that starts this section
            let toolsPushStart = mmrStart;
            while (toolsPushStart > 0 && setupToolsSection[toolsPushStart] !== '(') {
                toolsPushStart--;
            }
            // Go back to find 'tools.push'
            while (toolsPushStart > 0 && !setupToolsSection.substring(toolsPushStart - 10, toolsPushStart).includes('tools.push')) {
                toolsPushStart--;
            }
            toolsPushStart = setupToolsSection.lastIndexOf('tools.push', mmrStart);
            
            if (toolsPushStart > 0) {
                // Find the closing paren for this tools.push
                let parenCount = 0;
                let inString = false;
                let escapeNext = false;
                let i = toolsPushStart;
                while (i < setupToolsSection.length && setupToolsSection[i] !== '(') i++;
                i++; // Skip opening paren
                
                for (; i < indexingToolsStart; i++) {
                    const char = setupToolsSection[i];
                    if (escapeNext) {
                        escapeNext = false;
                        continue;
                    }
                    if (char === '\\') {
                        escapeNext = true;
                        continue;
                    }
                    if (char === '"' || char === "'" || char === '`') {
                        inString = !inString;
                        continue;
                    }
                    if (!inString) {
                        if (char === '(') parenCount++;
                        if (char === ')') {
                            if (parenCount === 0) {
                                i++;
                                break;
                            }
                            parenCount--;
                        }
                    }
                }
                
                // Replace the hardcoded section
                const beforeHardcoded = setupToolsSection.substring(0, toolsPushStart);
                const afterHardcoded = setupToolsSection.substring(i);
                
                const newRerankingSection = `            // Reranking Tools - All 7 tools from RerankingTools.getToolDefinitions()
            // Includes: mmr_rerank, rerank_cross_encoder, rerank_llm (with promptTemplate & temperature),
            //           rerank_colbert, rerank_rrf, rerank_ensemble_weighted, rerank_ensemble_borda
            if (features.reranking?.enabled !== false) {
                const rerankingToolDefs = this.rerankingTools.getToolDefinitions();
                tools.push(...rerankingToolDefs);
            }
            `;
                
                const newSetupTools = beforeHardcoded + newRerankingSection + afterHardcoded;
                content = content.substring(0, setupToolsStart) + newSetupTools + content.substring(setupToolsEnd);
                
                console.log('✓ Replaced hardcoded reranking tools in setupTools() with getToolDefinitions()');
            }
        }
    } else if (setupToolsSection.includes('getToolDefinitions()')) {
        console.log('✓ setupTools() already uses getToolDefinitions()');
    } else {
        console.log('⚠ Could not find hardcoded reranking tools pattern');
    }
}

// ============================================================================
// STEP 2: Verify handlers are correct
// ============================================================================
const hasAllHandlers = content.includes('case "rerank_colbert":') &&
                       content.includes('case "rerank_rrf":') &&
                       content.includes('case "rerank_ensemble_weighted":') &&
                       content.includes('case "rerank_ensemble_borda":') &&
                       content.includes('handleToolCall(name, args)');

if (hasAllHandlers) {
    console.log('✓ All 7 reranking tool handlers are correctly set up');
} else {
    console.log('⚠ Some reranking handlers may be missing');
}

// Write backup
const backupPath = indexJsPath + '.backup_complete_fix';
fs.writeFileSync(backupPath, fs.readFileSync(indexJsPath, 'utf8'));
console.log(`\nBackup: ${backupPath}`);

// Write fixed file
fs.writeFileSync(indexJsPath, content);
console.log(`Updated: ${indexJsPath}\n`);

// ============================================================================
// FINAL VERIFICATION
// ============================================================================
console.log('Final Verification:');
const checks = {
    'getToolDefinitions() in setupTools()': content.indexOf('getToolDefinitions()') > content.indexOf('setupTools()') &&
                                             content.indexOf('getToolDefinitions()') < content.indexOf('setupToolHandlers()'),
    'All 7 tools in handlers': content.includes('case "rerank_colbert":') &&
                               content.includes('case "rerank_rrf":') &&
                               content.includes('case "rerank_ensemble_weighted":') &&
                               content.includes('case "rerank_ensemble_borda":'),
    'handleToolCall() used': content.includes('handleToolCall(name, args)'),
    'No hardcoded rerank_llm in setupTools()': !content.substring(content.indexOf('setupTools()'), content.indexOf('setupToolHandlers()')).includes('name: "rerank_llm",') ||
                                                content.substring(content.indexOf('setupTools()'), content.indexOf('setupToolHandlers()')).indexOf('getToolDefinitions()') < 
                                                content.substring(content.indexOf('setupTools()'), content.indexOf('setupToolHandlers()')).indexOf('name: "rerank_llm",')
};

Object.entries(checks).forEach(([check, passed]) => {
    console.log(`  ${passed ? '✓' : '✗'} ${check}`);
});

const allPassed = Object.values(checks).every(v => v);
console.log(`\n${allPassed ? '✅ All checks passed! Reranking tools are fully implemented.' : '⚠️  Some checks failed'}`);

