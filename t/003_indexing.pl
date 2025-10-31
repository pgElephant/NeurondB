#!/usr/bin/perl
#
# TAP Test 003: Indexing and ANN Search
# Tests HNSW, IVF, and hybrid indexes
#

use strict;
use warnings;
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# Create cluster
my $node = PostgreSQL::Test::Cluster->new('idx');
$node->init;
$node->start;

# Create extension
$node->safe_psql('postgres', 'CREATE EXTENSION neurondb;');

# Test 1: Create table with vectors
$node->safe_psql('postgres',
    'CREATE TABLE vec_idx_test (id serial PRIMARY KEY, embedding vectorf32);');
$node->safe_psql('postgres',
    "INSERT INTO vec_idx_test (embedding) SELECT ('[' || i || ',0,0]')::vectorf32 FROM generate_series(1,100) i;");

my $result = $node->safe_psql('postgres',
    'SELECT COUNT(*) FROM vec_idx_test;');
is($result, '100', '100 vectors inserted');

# Test 2: Create HNSW index
$node->safe_psql('postgres',
    'CREATE INDEX vec_hnsw_idx ON vec_idx_test USING hnsw (embedding);');
$result = $node->safe_psql('postgres',
    "SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'vec_hnsw_idx';");
is($result, '1', 'HNSW index created');

# Test 3: kNN search using index
$result = $node->safe_psql('postgres',
    "SELECT COUNT(*) FROM vec_idx_test ORDER BY embedding <-> '[50,0,0]'::vectorf32 LIMIT 5;");
ok($result > 0, 'kNN search returns results');

# Test 4: Create IVF index
$node->safe_psql('postgres',
    'CREATE INDEX vec_ivf_idx ON vec_idx_test USING ivf (embedding);');
$result = $node->safe_psql('postgres',
    "SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'vec_ivf_idx';");
is($result, '1', 'IVF index created');

# Test 5: Verify index is used
$result = $node->safe_psql('postgres',
    "EXPLAIN SELECT * FROM vec_idx_test ORDER BY embedding <-> '[50,0,0]'::vectorf32 LIMIT 5;");
like($result, qr/Index/, 'Query plan uses index');

# Test 6: Index build statistics
$result = $node->safe_psql('postgres',
    "SELECT index_name FROM hnsw_index_stats('vec_hnsw_idx');");
like($result, qr/vec_hnsw/, 'HNSW stats function works');

# Test 7: Concurrent index builds
$node->safe_psql('postgres',
    'CREATE INDEX CONCURRENTLY vec_hnsw_concurrent ON vec_idx_test USING hnsw (embedding);');
$result = $node->safe_psql('postgres',
    "SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'vec_idx_test';");
ok($result >= 3, 'Multiple indexes can coexist');

# Test 8: Drop indexes
$node->safe_psql('postgres', 'DROP INDEX vec_hnsw_idx;');
$node->safe_psql('postgres', 'DROP INDEX vec_ivf_idx;');
$node->safe_psql('postgres', 'DROP INDEX vec_hnsw_concurrent;');

# Cleanup
$node->safe_psql('postgres', 'DROP TABLE vec_idx_test CASCADE;');
$node->safe_psql('postgres', 'DROP EXTENSION neurondb CASCADE;');
$node->stop;

done_testing();

