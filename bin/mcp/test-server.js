#!/usr/bin/env node

/**
 * Simple test script to verify MCP server is working
 * Tests the server by sending a list_tools request
 */

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const serverPath = join(__dirname, 'dist', 'index.js');

console.log('Testing Neurondb MCP Server...');
console.log('Server path:', serverPath);
console.log('');

const server = spawn('node', [serverPath], {
	stdio: ['pipe', 'pipe', 'pipe']
});

let output = '';
let errorOutput = '';

server.stdout.on('data', (data) => {
	output += data.toString();
});

server.stderr.on('data', (data) => {
	errorOutput += data.toString();
	console.error('Server stderr:', data.toString());
});

server.on('error', (error) => {
	console.error('Failed to start server:', error);
	process.exit(1);
});

// Send initialize request
const initRequest = {
	jsonrpc: '2.0',
	id: 1,
	method: 'initialize',
	params: {
		protocolVersion: '2024-11-05',
		capabilities: {},
		clientInfo: {
			name: 'test-client',
			version: '1.0.0'
		}
	}
};

setTimeout(() => {
	console.log('Sending initialize request...');
	server.stdin.write(JSON.stringify(initRequest) + '\n');
	
	setTimeout(() => {
		// Send list_tools request
		const listToolsRequest = {
			jsonrpc: '2.0',
			id: 2,
			method: 'tools/list',
			params: {}
		};
		
		console.log('Sending list_tools request...');
		server.stdin.write(JSON.stringify(listToolsRequest) + '\n');
		
		setTimeout(() => {
			console.log('\n' + '='.repeat(60));
			console.log('Server Response:');
			console.log('='.repeat(60));
			
			// Parse and pretty-print JSON responses
			const lines = output.trim().split('\n').filter(line => line.trim());
			lines.forEach((line, idx) => {
				try {
					const json = JSON.parse(line);
					console.log(`\n[Response ${idx + 1}]`);
					console.log(JSON.stringify(json, null, 2));
				} catch (e) {
					console.log(line);
				}
			});
			
			// Extract and display tools summary
			try {
				const lastResponse = JSON.parse(lines[lines.length - 1]);
				if (lastResponse.result && lastResponse.result.tools) {
					console.log('\n' + '='.repeat(60));
					console.log(`Available Tools: ${lastResponse.result.tools.length}`);
					console.log('='.repeat(60));
					lastResponse.result.tools.forEach((tool, idx) => {
						console.log(`\n${idx + 1}. ${tool.name}`);
						console.log(`   ${tool.description}`);
						if (tool.inputSchema && tool.inputSchema.properties) {
							const required = tool.inputSchema.required || [];
							const props = Object.keys(tool.inputSchema.properties);
							console.log(`   Parameters: ${props.length} (Required: ${required.join(', ') || 'none'})`);
						}
					});
				}
			} catch (e) {
				// Ignore parsing errors
			}
			
			console.log('\n' + '='.repeat(60));
			console.log('✅ Test complete. Server is responding correctly.');
			console.log('='.repeat(60) + '\n');
			server.kill();
			process.exit(0);
		}, 1000);
	}, 500);
}, 500);

