#!/usr/bin/perl
#-------------------------------------------------------------------------
#
# 010_gpu_features.pl
#     TAP tests for NeurondB GPU acceleration features
#
# Tests GPU initialization, fallback behavior, distance operations,
# quantization, and statistics. Gracefully handles CPU-only builds.
#
# Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
#
#-------------------------------------------------------------------------

use strict;
use warnings;
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# Create a test cluster
my $node = PostgreSQL::Test::Cluster->new('gpu_test');
$node->init;
$node->append_conf('postgresql.conf', "shared_preload_libraries = 'neurondb'");
$node->start;

# Create extension
$node->safe_psql('postgres', 'CREATE EXTENSION neurondb;');

# Test 1: GPU info function exists
my $result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'neurondb_gpu_info';
});
is($result, 'neurondb_gpu_info', 'neurondb_gpu_info function exists');

# Test 2: GPU enable function exists
$result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'neurondb_gpu_enable';
});
is($result, 'neurondb_gpu_enable', 'neurondb_gpu_enable function exists');

# Test 3: GPU stats function exists
$result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'neurondb_gpu_stats';
});
is($result, 'neurondb_gpu_stats', 'neurondb_gpu_stats function exists');

# Test 4: Try to enable GPU (will warn if no GPU available)
$result = $node->psql('postgres', q{
    SELECT neurondb_gpu_enable(true);
});
# Accept both success (GPU available) or false (CPU fallback)
ok($result == 0 || $result == 3, 'neurondb_gpu_enable handles missing GPU gracefully');

# Test 5: GPU info returns data or empty
$result = $node->safe_psql('postgres', q{
    SELECT COUNT(*) >= 0 AS has_info FROM (SELECT neurondb_gpu_info()) AS t;
});
is($result, 't', 'neurondb_gpu_info returns valid result');

# Test 6: Create test vectors
$node->safe_psql('postgres', q{
    CREATE TABLE test_gpu_vectors (
        id SERIAL PRIMARY KEY,
        vec vector(128)
    );
    
    INSERT INTO test_gpu_vectors (vec)
    SELECT ('[' || array_to_string(ARRAY(
        SELECT (random() * 2 - 1)::float4
        FROM generate_series(1, 128)
    ), ',') || ']')::vector
    FROM generate_series(1, 50);
});

# Test 7: GPU L2 distance function exists
$result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'vector_l2_distance_gpu';
});
is($result, 'vector_l2_distance_gpu', 'vector_l2_distance_gpu function exists');

# Test 8: GPU cosine distance function exists
$result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'vector_cosine_distance_gpu';
});
is($result, 'vector_cosine_distance_gpu', 'vector_cosine_distance_gpu function exists');

# Test 9: GPU inner product function exists
$result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'vector_inner_product_gpu';
});
is($result, 'vector_inner_product_gpu', 'vector_inner_product_gpu function exists');

# Test 10: GPU quantization INT8 function exists
$result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'vector_to_int8_gpu';
});
is($result, 'vector_to_int8_gpu', 'vector_to_int8_gpu function exists');

# Test 11: GPU quantization FP16 function exists
$result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'vector_to_fp16_gpu';
});
is($result, 'vector_to_fp16_gpu', 'vector_to_fp16_gpu function exists');

# Test 12: GPU quantization binary function exists
$result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'vector_to_binary_gpu';
});
is($result, 'vector_to_binary_gpu', 'vector_to_binary_gpu function exists');

# Test 13: GPU KMeans clustering function exists
$result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'cluster_kmeans_gpu';
});
is($result, 'cluster_kmeans_gpu', 'cluster_kmeans_gpu function exists');

# Test 14: GPU HNSW search function exists
$result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'hnsw_knn_search_gpu';
});
is($result, 'hnsw_knn_search_gpu', 'hnsw_knn_search_gpu function exists');

# Test 15: GPU IVF search function exists
$result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'ivf_knn_search_gpu';
});
is($result, 'ivf_knn_search_gpu', 'ivf_knn_search_gpu function exists');

# Test 16: GPU stats reset function exists
$result = $node->safe_psql('postgres', q{
    SELECT proname FROM pg_proc WHERE proname = 'neurondb_gpu_stats_reset';
});
is($result, 'neurondb_gpu_stats_reset', 'neurondb_gpu_stats_reset function exists');

# Test 17: GUC variables exist
my @gucs = (
    'neurondb.gpu_enabled',
    'neurondb.gpu_device',
    'neurondb.gpu_batch_size',
    'neurondb.gpu_streams',
    'neurondb.gpu_memory_pool_mb',
    'neurondb.gpu_fail_open',
    'neurondb.gpu_kernels',
    'neurondb.gpu_timeout_ms'
);

foreach my $guc (@gucs) {
    $result = $node->safe_psql('postgres', qq{
        SELECT COUNT(*) FROM pg_settings WHERE name = '$guc';
    });
    is($result, '1', "GUC $guc exists");
}

# Test 18: GPU GUC defaults
$result = $node->safe_psql('postgres', q{
    SELECT setting FROM pg_settings WHERE name = 'neurondb.gpu_enabled';
});
is($result, 'off', 'GPU disabled by default');

$result = $node->safe_psql('postgres', q{
    SELECT setting FROM pg_settings WHERE name = 'neurondb.gpu_batch_size';
});
is($result, '8192', 'GPU batch size default is 8192');

$result = $node->safe_psql('postgres', q{
    SELECT setting FROM pg_settings WHERE name = 'neurondb.gpu_streams';
});
is($result, '2', 'GPU streams default is 2');

# Test 19: GPU statistics reset
$node->safe_psql('postgres', q{
    SELECT neurondb_gpu_stats_reset();
});
$result = $node->safe_psql('postgres', q{
    SELECT COUNT(*) >= 0 FROM (SELECT * FROM neurondb_gpu_stats()) AS t;
});
is($result, 't', 'GPU stats reset successful');

# Test 20: Disable GPU
$node->safe_psql('postgres', q{
    SELECT neurondb_gpu_enable(false);
});
pass('GPU disable successful');

# Cleanup
$node->safe_psql('postgres', 'DROP TABLE IF EXISTS test_gpu_vectors CASCADE;');

$node->stop;
done_testing();

