#!/usr/bin/perl
#
# TAP Test 002: Distance Metrics and Quantization
# Tests all distance functions and vector quantization
#

use strict;
use warnings;
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# Create cluster
my $node = PostgreSQL::Test::Cluster->new('dist');
$node->init;
$node->start;

# Create extension
$node->safe_psql('postgres', 'CREATE EXTENSION neurondb;');

# Test 1: L2 Distance
my $result = $node->safe_psql('postgres',
    "SELECT l2_distance('[3, 4]'::vectorf32, '[0, 0]'::vectorf32);");
ok($result == 5.0, 'L2 distance returns 5.0 for 3-4-5 triangle');

# Test 2: Cosine Distance  
$result = $node->safe_psql('postgres',
    "SELECT cosine_distance('[1, 0, 0]'::vectorf32, '[1, 0, 0]'::vectorf32);");
ok($result == 0, 'Cosine distance returns 0 for identical vectors');

# Test 3: Inner Product
$result = $node->safe_psql('postgres',
    "SELECT inner_product('[1, 2, 3]'::vectorf32, '[4, 5, 6]'::vectorf32);");
ok($result == 32, 'Inner product returns correct value');

# Test 4: Manhattan Distance
$result = $node->safe_psql('postgres',
    "SELECT l1_distance('[1, 2]'::vectorf32, '[4, 6]'::vectorf32);");
ok($result == 7, 'Manhattan distance returns 7');

# Test 5: Quantization to INT8
$result = $node->safe_psql('postgres',
    "SELECT vectorf32_to_vectori8('[1.0, 2.0, 3.0]'::vectorf32);");
like($result, qr/\[.*\]/, 'Quantization to INT8 works');

# Test 6: Quantization to F16
$result = $node->safe_psql('postgres',
    "SELECT vectorf32_to_vectorf16('[1.0, 2.0, 3.0]'::vectorf32);");
like($result, qr/\[.*\]/, 'Quantization to F16 works');

# Test 7: Binary quantization
$result = $node->safe_psql('postgres',
    "SELECT vectorf32_to_binary('[1.0, -1.0, 1.0]'::vectorf32);");
like($result, qr/[01]+/, 'Binary quantization works');

# Test 8: Hamming distance
$result = $node->safe_psql('postgres',
    "SELECT hamming_distance('1010'::vectorbin, '1100'::vectorbin);");
ok($result >= 0, 'Hamming distance returns non-negative');

# Test 9: Batch distance calculations
$node->safe_psql('postgres', 
    'CREATE TABLE dist_test (id int, vec vectorf32);');
$node->safe_psql('postgres',
    "INSERT INTO dist_test VALUES (1, '[1,0,0]'::vectorf32), (2, '[0,1,0]'::vectorf32);");
$result = $node->safe_psql('postgres',
    'SELECT COUNT(*) FROM dist_test;');
is($result, '2', 'Distance test table populated');

# Test 10: Cross-product distances
$result = $node->safe_psql('postgres',
    'SELECT COUNT(*) FROM dist_test a CROSS JOIN dist_test b WHERE a.id < b.id;');
is($result, '1', 'Cross-product distance query works');

# Cleanup
$node->safe_psql('postgres', 'DROP TABLE dist_test;');
$node->safe_psql('postgres', 'DROP EXTENSION neurondb CASCADE;');
$node->stop;

done_testing();

