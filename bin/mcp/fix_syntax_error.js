#!/usr/bin/env node
/**
 * Fix syntax error in dist/index.js
 * Removes orphaned tool definition fragment
 */

const fs = require('fs');
const path = require('path');

const indexPath = path.join(__dirname, 'dist', 'index.js');
let content = fs.readFileSync(indexPath, 'utf8');

console.log('Fixing syntax error in dist/index.js...\n');

// Find and remove the orphaned tool definition fragment
// Pattern: }, \n                            documents: { type: "array"...
// This is a broken tool definition that needs to be removed

const brokenPattern = /},\s*\n\s+documents:\s*\{\s*type:\s*"array",\s*items:\s*\{\s*type:\s*"string"\s*\}\s*\},\s*\n\s+method:/;

if (brokenPattern.test(content)) {
    // Find the context - this should be part of a tool definition that's broken
    // Look for the pattern and remove the orphaned part
    content = content.replace(brokenPattern, '},\n                }, {');
    console.log('✓ Removed orphaned tool definition fragment');
} else {
    // Try a different pattern - look for the specific broken section
    const brokenSection = /},\s*\n\s+documents:\s*\{\s*type:\s*"array"/;
    if (brokenSection.test(content)) {
        // Find the exact location
        const lines = content.split('\n');
        let fixed = false;
        for (let i = 0; i < lines.length - 1; i++) {
            if (lines[i].trim() === '},' && lines[i+1].includes('documents: { type: "array"')) {
                // This is the broken section - remove the orphaned properties
                // Find where this broken section ends
                let endIdx = i + 1;
                while (endIdx < lines.length && !lines[endIdx].includes('});')) {
                    endIdx++;
                }
                // Remove the broken lines
                lines.splice(i + 1, endIdx - i);
                content = lines.join('\n');
                fixed = true;
                console.log(`✓ Removed broken tool definition at line ${i + 2}`);
                break;
            }
        }
        if (!fixed) {
            console.log('⚠ Could not automatically fix - may need manual inspection');
        }
    } else {
        console.log('⚠ Could not find the broken pattern');
    }
}

// Write backup
const backupPath = indexPath + '.backup_syntax_fix';
fs.writeFileSync(backupPath, fs.readFileSync(indexPath, 'utf8'));
console.log(`Backup: ${backupPath}`);

// Write fixed file
fs.writeFileSync(indexPath, content);
console.log(`Updated: ${indexPath}\n`);

// Verify syntax
const { execSync } = require('child_process');
try {
    execSync('node -c dist/index.js', { cwd: __dirname, stdio: 'pipe' });
    console.log('✅ Syntax check passed!');
} catch (error) {
    console.log('❌ Syntax check failed - file may still have errors');
    console.log('Error:', error.message);
}

