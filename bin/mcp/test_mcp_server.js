#!/usr/bin/env node
/**
 * Test script to verify MCP server can start and connect to database
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('Testing MCP Server...\n');

// Test 1: Check if dist/index.js exists
try {
    const fs = await import('fs');
    const indexPath = join(__dirname, 'dist', 'index.js');
    if (fs.existsSync(indexPath)) {
        console.log('✅ MCP server file exists:', indexPath);
    } else {
        console.log('❌ MCP server file not found:', indexPath);
        process.exit(1);
    }
} catch (error) {
    console.log('❌ Error checking file:', error.message);
    process.exit(1);
}

// Test 2: Try to import the server
try {
    console.log('\nAttempting to import MCP server...');
    const serverModule = await import('./dist/index.js');
    console.log('✅ MCP server module imported successfully');
} catch (error) {
    console.log('❌ Failed to import MCP server:', error.message);
    console.log('Stack:', error.stack);
    process.exit(1);
}

// Test 3: Check database connection
try {
    console.log('\nTesting database connection...');
    const { Database } = await import('./dist/db.js');
    const { ConfigManager } = await import('./dist/config.js');
    
    const configManager = new ConfigManager();
    const config = configManager.getConfig();
    const dbConfig = config.database;
    
    console.log('Database config:', {
        host: dbConfig.host,
        port: dbConfig.port,
        database: dbConfig.database,
        user: dbConfig.user
    });
    
    const db = new Database();
    db.connect(dbConfig);
    
    // Test connection
    await db.testConnection();
    console.log('✅ Database connection successful');
    
    await db.close();
} catch (error) {
    console.log('❌ Database connection failed:', error.message);
    console.log('Stack:', error.stack);
    process.exit(1);
}

console.log('\n✅ All tests passed! MCP server should work.');
process.exit(0);

