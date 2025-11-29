import { Database } from "../db.js";
import {
	VectorSearchParams,
	EmbeddingParams,
	BatchEmbeddingParams,
	HNSWIndexParams,
	HybridSearchParams,
} from "../types.js";

export class VectorTools {
	constructor(private db: Database) {}

	async vectorSearch(params: VectorSearchParams) {
		const {
			table,
			vector_column,
			query_vector,
			limit = 10,
			distance_metric = "l2",
		} = params;

		const vectorStr = `[${query_vector.join(",")}]`;
		let operator = "<->";
		if (distance_metric === "cosine") {
			operator = "<=>";
		} else if (distance_metric === "inner_product") {
			operator = "<#>";
		}

		const query = `
			SELECT *, ${this.db.escapeIdentifier(vector_column)} ${operator} $1::vector AS distance
			FROM ${this.db.escapeIdentifier(table)}
			ORDER BY distance
			LIMIT $2
		`;

		const result = await this.db.query(query, [vectorStr, limit]);
		return result.rows;
	}

	async generateEmbedding(params: EmbeddingParams) {
		const { text, model } = params;

		let query = "SELECT embed_text($1) AS embedding";
		const queryParams: any[] = [text];

		if (model) {
			query = "SELECT embed_text($1, $2) AS embedding";
			queryParams.push(model);
		}

		const result = await this.db.query(query, queryParams);
		return result.rows[0];
	}

	async batchEmbedding(params: BatchEmbeddingParams) {
		const { texts, model } = params;

		let query = "SELECT embed_text_batch($1) AS embeddings";
		const queryParams: any[] = [texts];

		if (model) {
			query = "SELECT embed_text_batch($1, $2) AS embeddings";
			queryParams.push(model);
		}

		const result = await this.db.query(query, queryParams);
		return result.rows[0];
	}

	async createHNSWIndex(params: HNSWIndexParams) {
		const {
			table,
			vector_column,
			index_name,
			m = 16,
			ef_construction = 200,
		} = params;

		const query = `
			SELECT hnsw_create_index(
				$1::text,
				$2::text,
				$3::text,
				$4::integer,
				$5::integer
			) AS result
		`;

		const result = await this.db.query(query, [
			table,
			vector_column,
			index_name,
			m,
			ef_construction,
		]);

		return result.rows[0];
	}

	async hybridSearch(params: HybridSearchParams) {
		const {
			table,
			query_vector,
			query_text,
			text_column,
			vector_column,
			vector_weight = 0.7,
			limit = 10,
		} = params;

		const vectorStr = `[${query_vector.join(",")}]`;

		const query = `
			SELECT * FROM hybrid_search(
				$1::text,
				$2::vector,
				$3::text,
				$4::jsonb,
				$5::float,
				$6::integer
			)
		`;

		const result = await this.db.query(query, [
			table,
			vectorStr,
			query_text,
			JSON.stringify({}),
			vector_weight,
			limit,
		]);

		return result.rows;
	}
}

