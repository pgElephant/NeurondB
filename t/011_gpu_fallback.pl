use strict;
use warnings FATAL => 'all';
use Test::More;
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;

my $pg = PostgreSQL::Test::Cluster->new('neurondb_gpu_fallback');
$pg->init;
$pg->start;

$pg->safe_psql('postgres', q{CREATE EXTENSION neurondb;});

# Ensure GPU disabled; wrapper should still work via CPU
my $res = $pg->safe_psql('postgres', q{
  SET neurondb.gpu_enabled = off;
  SELECT vector_l2_distance_gpu('[1,2]'::vector, '[4,6]'::vector);
});

is($res, '5', 'GPU wrapper falls back to CPU and returns correct L2');

done_testing();

