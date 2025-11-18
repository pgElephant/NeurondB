#!/usr/bin/env node
/**
 * Fix setupTools() to use getToolDefinitions() for reranking tools
 */

const fs = require('fs');
const path = require('path');

const indexJsPath = path.join(__dirname, 'dist', 'index.js');
let content = fs.readFileSync(indexJsPath, 'utf8');

console.log('Fixing setupTools() to use getToolDefinitions()...');

// Find setupTools() method
const setupToolsStart = content.indexOf('setupTools()');
const setupToolsEnd = content.indexOf('setupToolHandlers()', setupToolsStart);

if (setupToolsStart > 0 && setupToolsEnd > 0) {
    const setupToolsSection = content.substring(setupToolsStart, setupToolsEnd);
    
    // Check if it has hardcoded reranking tools
    if (setupToolsSection.includes('name: "mmr_rerank"') && 
        setupToolsSection.includes('name: "rerank_llm"') &&
        !setupToolsSection.includes('getToolDefinitions()')) {
        
        // Find the old reranking tools section
        const oldRerankingStart = setupToolsSection.indexOf('name: "mmr_rerank"');
        const oldRerankingEnd = setupToolsSection.indexOf('// Indexing Tools', oldRerankingStart);
        
        if (oldRerankingStart > 0 && oldRerankingEnd > 0) {
            // Extract the part before reranking tools
            const beforeReranking = setupToolsSection.substring(0, oldRerankingStart);
            // Find the tools.push( that starts the reranking section
            const toolsPushStart = beforeReranking.lastIndexOf('tools.push(');
            if (toolsPushStart > 0) {
                // Find the closing of that tools.push
                let parenCount = 0;
                let inString = false;
                let escapeNext = false;
                let i = toolsPushStart + 'tools.push('.length;
                for (; i < oldRerankingEnd; i++) {
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
                
                // Replace the old reranking tools section
                const newRerankingSection = `            // Reranking Tools - All 7 tools from RerankingTools.getToolDefinitions()
            // Includes: mmr_rerank, rerank_cross_encoder, rerank_llm (with promptTemplate & temperature),
            //           rerank_colbert, rerank_rrf, rerank_ensemble_weighted, rerank_ensemble_borda
            if (features.reranking?.enabled !== false) {
                const rerankingToolDefs = this.rerankingTools.getToolDefinitions();
                tools.push(...rerankingToolDefs);
            }
            // Indexing Tools`;
                
                const newSetupTools = setupToolsSection.substring(0, toolsPushStart) + 
                                    newRerankingSection + 
                                    setupToolsSection.substring(oldRerankingEnd);
                
                content = content.substring(0, setupToolsStart) + 
                         newSetupTools + 
                         content.substring(setupToolsEnd);
                
                console.log('✓ Replaced hardcoded reranking tools in setupTools()');
            }
        }
    } else if (setupToolsSection.includes('getToolDefinitions()')) {
        console.log('✓ setupTools() already uses getToolDefinitions()');
    } else {
        console.log('⚠ Could not find reranking tools in setupTools()');
    }
} else {
    console.log('⚠ Could not find setupTools() method');
}

// Write backup
const backupPath = indexJsPath + '.backup_setup_tools';
fs.writeFileSync(backupPath, fs.readFileSync(indexJsPath, 'utf8'));
console.log(`Backup: ${backupPath}`);

// Write fixed file
fs.writeFileSync(indexJsPath, content);
console.log(`Updated: ${indexJsPath}\n`);

// Verify
const setupToolsIdx = content.indexOf('setupTools()');
const setupHandlersIdx = content.indexOf('setupToolHandlers()');
const getToolDefsIdx = content.indexOf('getToolDefinitions()');

const hasGetToolDefs = getToolDefsIdx > 0 && 
                       getToolDefsIdx > setupToolsIdx && 
                       getToolDefsIdx < setupHandlersIdx;

console.log('Verification:');
console.log(`  ${hasGetToolDefs ? '✓' : '✗'} getToolDefinitions() is in setupTools() method`);

if (hasGetToolDefs) {
    console.log('\n✅ setupTools() fixed!');
} else {
    console.log('\n⚠️  May need manual inspection');
}

