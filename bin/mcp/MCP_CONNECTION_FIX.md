# MCP Server Connection Fix

## Issue
"Could not connect to neurondb mcp server" error in Claude Desktop.

## Root Cause
Syntax error in `dist/index.js` - orphaned tool definition fragment causing parse failure.

## Solution Applied
1. ✅ Removed broken tool definition fragment (lines 278-290)
2. ✅ Syntax check passes
3. ⚠️ Import still fails - may need to rebuild from source

## Next Steps

### Option 1: Rebuild from TypeScript Source (Recommended)
If TypeScript source exists:
```bash
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
npm run build
# or
tsc
```

### Option 2: Use Working Backup
Find a backup that works:
```bash
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
# Test backups
for backup in dist/index.js.backup*; do
  node -c "$backup" && echo "✅ $backup works"
done
```

### Option 3: Manual Fix
The broken section was at lines 278-290. If rebuilding doesn't work, manually verify the file structure around that area.

## Current Status
- ✅ Syntax error removed
- ✅ Node.js syntax check passes  
- ❌ ES module import still fails
- ⚠️ May need to rebuild from source

## Test After Fix
```bash
cd /Users/ibrarahmed/pgelephant/pge/Neurondb/bin/mcp
node test_mcp_server.js
```

If test passes, restart Claude Desktop.

