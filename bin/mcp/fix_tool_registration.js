#!/usr/bin/env node
/**
 * Fix tool registration to use getToolDefinitions() instead of hardcoded definitions
 */

const fs = require('fs');
const path = require('path');

const indexJsPath = path.join(__dirname, 'dist', 'index.js');
let content = fs.readFileSync(indexJsPath, 'utf8');

console.log('Fixing tool registration to use getToolDefinitions()...');

// Find the old hardcoded reranking tools section (around line 640-680)
// It should have: name: "mmr_rerank", name: "rerank_cross_encoder", name: "rerank_llm"
const oldRerankingPattern = /\{\s*name: "mmr_rerank",[\s\S]{0,2000}?name: "rerank_llm",[\s\S]{0,200}?\}\s*\}\);\s*\/\/ Indexing Tools/;

if (oldRerankingPattern.test(content)) {
    // Replace with getToolDefinitions() call
    const newRegistration = `            // Reranking Tools - All 7 tools from RerankingTools.getToolDefinitions()
            // Includes: mmr_rerank, rerank_cross_encoder, rerank_llm (with promptTemplate & temperature),
            //           rerank_colbert, rerank_rrf, rerank_ensemble_weighted, rerank_ensemble_borda
            if (features.reranking?.enabled !== false) {
                const rerankingToolDefs = this.rerankingTools.getToolDefinitions();
                tools.push(...rerankingToolDefs);
            }
            // Indexing Tools`;
    
    content = content.replace(oldRerankingPattern, newRegistration);
    console.log('✓ Replaced hardcoded reranking tools with getToolDefinitions()');
} else {
    // Check if getToolDefinitions() is already used
    if (content.includes('getToolDefinitions()') && content.indexOf('getToolDefinitions()') < content.indexOf('setupToolHandlers()')) {
        console.log('✓ Tool registration already uses getToolDefinitions()');
    } else {
        console.log('⚠ Could not find old reranking tools pattern - may need manual fix');
    }
}

// Write backup
const backupPath = indexJsPath + '.backup_registration';
fs.writeFileSync(backupPath, fs.readFileSync(indexJsPath, 'utf8'));
console.log(`Backup: ${backupPath}`);

// Write fixed file
fs.writeFileSync(indexJsPath, content);
console.log(`Updated: ${indexJsPath}\n`);

// Verify
const hasGetToolDefinitions = content.includes('getToolDefinitions()') && 
                               content.indexOf('getToolDefinitions()') < content.indexOf('setupToolHandlers()');
const noHardcodedRerankLLM = !content.includes('name: "rerank_llm",') || 
                             content.indexOf('getToolDefinitions()') < content.indexOf('name: "rerank_llm",');

console.log('Verification:');
console.log(`  ${hasGetToolDefinitions ? '✓' : '✗'} Uses getToolDefinitions() in setupTools()`);
console.log(`  ${noHardcodedRerankLLM ? '✓' : '✗'} No hardcoded rerank_llm in tool registration`);

if (hasGetToolDefinitions && noHardcodedRerankLLM) {
    console.log('\n✅ Tool registration fixed!');
} else {
    console.log('\n⚠️  May need additional fixes');
}

