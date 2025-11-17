#!/usr/bin/env node

/**
 * Example: Training ML Models using NeuronDB MCP Server
 * 
 * This script demonstrates how to train various ML models
 * using the NeuronDB MCP server.
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

async function main() {
	// Initialize MCP client
	const transport = new StdioClientTransport({
		command: "node",
		args: ["dist/index.js"],
	});

	const client = new Client(
		{
			name: "training-example",
			version: "1.0.0",
		},
		{
			capabilities: {},
		}
	);

	await client.connect(transport);

	console.log("🚀 NeuronDB MCP Training Example\n");

	try {
		// Step 1: Check GPU availability
		console.log("Step 1: Checking GPU availability...");
		const gpuInfo = await client.callTool({
			name: "gpu_info",
			arguments: {},
		});
		console.log("GPU Info:", JSON.stringify(gpuInfo.content[0].text, null, 2));
		console.log();

		// Step 2: Enable GPU (optional)
		console.log("Step 2: Enabling GPU...");
		const gpuEnable = await client.callTool({
			name: "gpu_enable",
			arguments: { enabled: true },
		});
		console.log("GPU Status:", JSON.stringify(gpuEnable.content[0].text, null, 2));
		console.log();

		// Step 3: Train Linear Regression Model
		console.log("Step 3: Training Linear Regression Model...");
		const trainResult = await client.callTool({
			name: "train_ml_model",
			arguments: {
				table: "sample_train",
				feature_col: "features",
				label_col: "label",
				algorithm: "linear_regression",
			},
		});
		const modelResult = JSON.parse(trainResult.content[0].text);
		const modelId = modelResult.model_id;
		console.log(`✅ Model trained! Model ID: ${modelId}`);
		console.log("Training Result:", JSON.stringify(modelResult, null, 2));
		console.log();

		// Step 4: Get Model Information
		console.log("Step 4: Getting Model Information...");
		const modelInfo = await client.callTool({
			name: "get_model_info",
			arguments: { model_id: modelId },
		});
		console.log("Model Info:", JSON.stringify(modelInfo.content[0].text, null, 2));
		console.log();

		// Step 5: Make a Prediction
		console.log("Step 5: Making a Prediction...");
		const prediction = await client.callTool({
			name: "predict_ml_model",
			arguments: {
				model_id: modelId,
				features: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
			},
		});
		console.log("Prediction:", JSON.stringify(prediction.content[0].text, null, 2));
		console.log();

		// Step 6: Train Random Forest (Example with hyperparameters)
		console.log("Step 6: Training Random Forest Model...");
		const rfResult = await client.callTool({
			name: "train_ml_model",
			arguments: {
				table: "sample_train",
				feature_col: "features",
				label_col: "label",
				algorithm: "random_forest",
				params: {
					n_estimators: 100,
					max_depth: 10,
					min_samples_split: 2,
				},
			},
		});
		const rfModel = JSON.parse(rfResult.content[0].text);
		console.log(`✅ Random Forest trained! Model ID: ${rfModel.model_id}`);
		console.log();

		// Step 7: List All Models
		console.log("Step 7: Listing All Models...");
		const allModels = await client.callTool({
			name: "get_model_info",
			arguments: {},
		});
		console.log("All Models:", JSON.stringify(allModels.content[0].text, null, 2));
		console.log();

		// Step 8: Get GPU Statistics
		console.log("Step 8: Getting GPU Statistics...");
		const gpuStats = await client.callTool({
			name: "gpu_stats",
			arguments: {},
		});
		console.log("GPU Stats:", JSON.stringify(gpuStats.content[0].text, null, 2));
		console.log();

		console.log("✅ Training example completed successfully!");
	} catch (error) {
		console.error("❌ Error:", error);
		process.exit(1);
	} finally {
		await client.close();
	}
}

main().catch(console.error);

