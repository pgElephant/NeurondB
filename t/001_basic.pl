#!/usr/bin/perl
#
# TAP Test 001: Basic NeuronDB Functionality
# Tests extension creation, basic vector operations, and cleanup
#

use strict;
use warnings;
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# Create a new cluster
my $node = PostgreSQL::Test::Cluster->new('main');
$node->init;
$node->start;

# Test 1: Create extension
$node->safe_psql('postgres', 'CREATE EXTENSION neurondb;');
my $result = $node->safe_psql('postgres', 
    "SELECT COUNT(*) FROM pg_extension WHERE extname = 'neurondb';");
is($result, '1', 'neurondb extension created successfully');

# Test 2: Create vectorf32
$result = $node->safe_psql('postgres',
    "SELECT '[1.0, 2.0, 3.0]'::vectorf32;");
like($result, qr/\[1.*2.*3.*\]/, 'vectorf32 type works');

# Test 3: Create table with vectors
$node->safe_psql('postgres', 
    'CREATE TABLE test_vectors (id serial, vec vectorf32);');
$result = $node->safe_psql('postgres',
    "SELECT COUNT(*) FROM pg_class WHERE relname = 'test_vectors';");
is($result, '1', 'table with vectorf32 column created');

# Test 4: Insert and query vectors
$node->safe_psql('postgres',
    "INSERT INTO test_vectors (vec) VALUES ('[1.0, 2.0, 3.0]'::vectorf32);");
$result = $node->safe_psql('postgres',
    'SELECT COUNT(*) FROM test_vectors;');
is($result, '1', 'vector inserted successfully');

# Test 5: Distance calculation
$result = $node->safe_psql('postgres',
    "SELECT l2_distance('[1.0, 2.0]'::vectorf32, '[4.0, 6.0]'::vectorf32);");
ok($result > 0, 'l2_distance returns positive value');

# Test 6: Vector dimensions
$result = $node->safe_psql('postgres',
    "SELECT vector_dims('[1.0, 2.0, 3.0]'::vectorf32);");
is($result, '3', 'vector_dims returns correct dimension');

# Test 7: Quantization
$result = $node->safe_psql('postgres',
    "SELECT vectorf32_to_vectori8('[1.0, 2.0, 3.0]'::vectorf32);");
like($result, qr/\[.*\]/, 'quantization works');

# Test 8: HNSW index creation
$node->safe_psql('postgres',
    'CREATE INDEX test_hnsw_idx ON test_vectors USING hnsw (vec);');
$result = $node->safe_psql('postgres',
    "SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'test_hnsw_idx';");
is($result, '1', 'HNSW index created');

# Test 9: Cleanup
$node->safe_psql('postgres', 'DROP TABLE test_vectors CASCADE;');
$node->safe_psql('postgres', 'DROP EXTENSION neurondb CASCADE;');

# Stop the cluster
$node->stop;

done_testing();

