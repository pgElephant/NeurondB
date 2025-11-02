use strict;
use warnings FATAL => 'all';
use Test::More;
use PostgreSQL::Test::Cluster;

my $pg = PostgreSQL::Test::Cluster->new('neurondb_llm_mock');
$pg->init;
$pg->start;

$pg->safe_psql('postgres', q{CREATE EXTENSION neurondb;});

$pg->safe_psql('postgres', q{
  SET neurondb.llm_endpoint = 'mock://hf';
  SET neurondb.llm_provider = 'huggingface';
  SET neurondb.llm_model = 'sentence-transformers/all-MiniLM-L6-v2';
});

my $c = $pg->safe_psql('postgres', q{SELECT llm_complete('hello','{}')});
is($c, 'mock-completion', 'llm_complete mock');

my $e = $pg->safe_psql('postgres', q{SELECT vector_dims(llm_embed('hello'))});
is($e, '4', 'llm_embed mock dim');

done_testing();

