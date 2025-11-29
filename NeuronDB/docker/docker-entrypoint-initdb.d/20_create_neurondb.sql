-- NeuronDB Extension Initialization Script
-- This script automatically creates the NeuronDB extension in the default database
-- when the container is first initialized.

-- Create the extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS neurondb;

-- Display extension information
SELECT extname, extversion 
FROM pg_extension 
WHERE extname = 'neurondb';
