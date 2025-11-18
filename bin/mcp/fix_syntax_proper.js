#!/usr/bin/env node
/**
 * Properly fix the syntax error in dist/index.js
 * Removes the orphaned tool definition that's causing the error
 */

const fs = require('fs');
const path = require('path');

const indexPath = path.join(__dirname, 'dist', 'index.js');
let content = fs.readFileSync(indexPath, 'utf8');

console.log('Fixing syntax error properly...\n');

// The issue is at line ~279: there's an orphaned tool definition fragment
// Pattern: }, \n                            documents: { type: "array"...
// This needs to be completely removed as it's a broken tool definition

const lines = content.split('\n');
let fixed = false;
let inBrokenSection = false;
let braceCount = 0;
const newLines = [];

for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // Detect the start of the broken section
    if (line.trim() === '},' && i + 1 < lines.length && 
        lines[i + 1].includes('documents: { type: "array"') &&
        !lines[i - 1].includes('name:') && !lines[i - 1].includes('description:')) {
        // This is the broken section - skip it until we find the closing });
        inBrokenSection = true;
        braceCount = 0;
        console.log(`Found broken section starting at line ${i + 1}`);
        continue;
    }
    
    if (inBrokenSection) {
        // Count braces to find the end
        for (const char of line) {
            if (char === '{') braceCount++;
            if (char === '}') braceCount--;
        }
        
        // If we find }); and braceCount is balanced, we've reached the end
        if (line.includes('});') && braceCount <= 0) {
            inBrokenSection = false;
            console.log(`Removed broken section ending at line ${i + 1}`);
            // Don't include this line either
            continue;
        }
        
        // Skip lines in the broken section
        continue;
    }
    
    newLines.push(line);
}

if (newLines.length < lines.length) {
    content = newLines.join('\n');
    fixed = true;
    console.log(`✓ Removed ${lines.length - newLines.length} lines of broken code`);
} else {
    console.log('⚠ Could not find broken section to remove');
}

// Write backup
const backupPath = indexPath + '.backup_proper_fix';
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
    if (fixed) {
        console.log('\n✅ Syntax error fixed! MCP server should work now.');
    }
} catch (error) {
    console.log('❌ Syntax check failed');
    console.log('Error:', error.message);
    process.exit(1);
}

