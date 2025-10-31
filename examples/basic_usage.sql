-- NeurondB: Basic Usage Examples

-- Load extension
CREATE EXTENSION IF NOT EXISTS neurondb;

-- Create table with vectors
CREATE TABLE products (
    id serial PRIMARY KEY,
    name text,
    description text,
    price numeric(10,2),
    category text,
    embedding vector(384),
    created_at timestamp DEFAULT now()
);

-- Insert sample data
INSERT INTO products (name, description, price, category, embedding) VALUES
('Laptop', 'High-performance laptop for developers', 1299.99, 'Electronics',
 '[0.1, 0.2, 0.3, 0.4, 0.5]'::vector),
('Coffee Maker', 'Automatic coffee maker with timer', 89.99, 'Appliances',
 '[0.2, 0.3, 0.1, 0.5, 0.4]'::vector),
('Running Shoes', 'Comfortable running shoes', 129.99, 'Sports',
 '[0.3, 0.1, 0.4, 0.2, 0.5]'::vector);

-- Basic vector operations
SELECT name, vector_dims(embedding) as dims FROM products LIMIT 1;
SELECT name, vector_norm(embedding) as norm FROM products;
SELECT name, vector_normalize(embedding) FROM products;

-- Vector arithmetic
SELECT 
    embedding + '[0.1, 0.1, 0.1, 0.1, 0.1]'::vector as added,
    embedding - '[0.1, 0.1, 0.1, 0.1, 0.1]'::vector as subtracted,
    embedding * 2.0 as scaled
FROM products LIMIT 1;

-- Distance calculations
WITH query AS (
    SELECT '[0.15, 0.25, 0.2, 0.45, 0.45]'::vector as query_vec
)
SELECT 
    name,
    embedding <-> query_vec as l2_distance,
    embedding <=> query_vec as cosine_distance,
    embedding <#> query_vec as inner_product
FROM products, query
ORDER BY l2_distance
LIMIT 5;

-- Array conversion
SELECT 
    name,
    vector_to_array(embedding) as array_form,
    array_to_vector(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5]) as vec_form
FROM products LIMIT 1;

-- Quantization for memory savings
ALTER TABLE products 
    ADD COLUMN embedding_i8 bytea,
    ADD COLUMN embedding_binary bytea;

UPDATE products SET 
    embedding_i8 = vector_to_int8(embedding),
    embedding_binary = vector_to_binary(embedding);

-- Check size reduction
SELECT 
    pg_column_size(embedding) as original_size,
    pg_column_size(embedding_i8) as int8_size,
    pg_column_size(embedding_binary) as binary_size
FROM products LIMIT 1;

