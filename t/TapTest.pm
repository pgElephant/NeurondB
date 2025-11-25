package TapTest;

use strict;
use warnings;
use Test::More;
use Exporter 'import';

our @EXPORT = qw(
	neurondb_ok
	query_ok
	query_fails
	result_is
	result_matches
	vector_ok
	ml_result_ok
	extension_ok
	schema_ok
	table_ok
	function_ok
);

=head1 NAME

TapTest - TAP test utilities for NeuronDB

=head1 SYNOPSIS

  use TapTest;
  use PostgresNode;

  my $node = PostgresNode->new('test');
  $node->init();
  $node->start();

  query_ok($node, 'postgres', 'SELECT 1', 'Simple query');
  result_is($node, 'postgres', 'SELECT 1', '1', 'Result matches');
  extension_ok($node, 'neurondb', 'NeuronDB extension installed');

=head1 DESCRIPTION

Provides TAP test utilities and NeuronDB-specific assertions for
testing PostgreSQL extensions and NeuronDB functionality.

=cut

=head2 neurondb_ok

Check if NeuronDB extension is properly installed and configured.

=cut

sub neurondb_ok {
	my ($node, $dbname, $test_name) = @_;
	$test_name ||= 'NeuronDB extension is installed';
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, q{
		SELECT extname, extversion 
		FROM pg_extension 
		WHERE extname = 'neurondb';
	});
	
	ok($result->{success}, $test_name);
	
	if ($result->{success}) {
		like($result->{stdout}, qr/neurondb/, "Extension name found");
		like($result->{stdout}, qr/\d+\.\d+/, "Extension version found");
	}
	
	return $result->{success};
}

=head2 query_ok

Execute a SQL query and verify it succeeds.

=cut

sub query_ok {
	my ($node, $dbname, $sql, $test_name) = @_;
	$test_name ||= "Query executes successfully";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql);
	ok($result->{success}, $test_name);
	
	unless ($result->{success}) {
		diag("SQL: $sql");
		diag("Error: $result->{stderr}");
	}
	
	return $result->{success};
}

=head2 query_fails

Execute a SQL query and verify it fails (returns non-zero exit code).

=cut

sub query_fails {
	my ($node, $dbname, $sql, $test_name) = @_;
	$test_name ||= "Query fails as expected";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql);
	ok(!$result->{success}, $test_name);
	
	if ($result->{success}) {
		diag("SQL: $sql");
		diag("Expected query to fail but it succeeded");
	}
	
	return !$result->{success};
}

=head2 result_is

Execute a SQL query and verify the result matches expected value.

=cut

sub result_is {
	my ($node, $dbname, $sql, $expected, $test_name) = @_;
	$test_name ||= "Query result matches expected";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql, tuples_only => 1);
	
	unless ($result->{success}) {
		fail($test_name);
		diag("SQL: $sql");
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	my $actual = $result->{stdout};
	$actual =~ s/^\s+|\s+$//g;  # Trim leading/trailing whitespace
	chomp $expected;
	
	is($actual, $expected, $test_name);
	
	unless ($actual eq $expected) {
		diag("Expected: $expected");
		diag("Got: $actual");
	}
	
	return $actual eq $expected;
}

=head2 result_matches

Execute a SQL query and verify the result matches a regex pattern.

=cut

sub result_matches {
	my ($node, $dbname, $sql, $pattern, $test_name) = @_;
	$test_name ||= "Query result matches pattern";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql, tuples_only => 1);
	
	unless ($result->{success}) {
		fail($test_name);
		diag("SQL: $sql");
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	my $actual = $result->{stdout};
	chomp $actual;
	
	like($actual, $pattern, $test_name);
	
	unless ($actual =~ $pattern) {
		diag("Pattern: $pattern");
		diag("Got: $actual");
	}
	
	return $actual =~ $pattern;
}

=head2 vector_ok

Verify vector type operations work correctly.

=cut

sub vector_ok {
	my ($node, $dbname, $test_name) = @_;
	$test_name ||= 'Vector operations work';
	$dbname ||= 'postgres';
	
	# Test vector creation
	my $result = $node->psql($dbname, q{
		SELECT '[1,2,3]'::vector(3);
	});
	
	unless ($result->{success}) {
		fail($test_name);
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	# Test vector distance
	$result = $node->psql($dbname, q{
		SELECT '[1,2,3]'::vector(3) <-> '[4,5,6]'::vector(3) AS distance;
	});
	
	unless ($result->{success}) {
		fail($test_name);
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	ok(1, $test_name);
	return 1;
}

=head2 ml_result_ok

Verify ML function result structure and validity.

=cut

sub ml_result_ok {
	my ($node, $dbname, $sql, $test_name, %checks) = @_;
	$test_name ||= 'ML function result is valid';
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql);
	
	unless ($result->{success}) {
		fail($test_name);
		diag("SQL: $sql");
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	my $output = $result->{stdout};
	
	# Check for expected columns if specified
	if (exists $checks{columns}) {
		for my $col (@{$checks{columns}}) {
			like($output, qr/$col/, "Result contains column: $col");
		}
	}
	
	# Check for numeric results if specified
	if (exists $checks{numeric}) {
		like($output, qr/\d+\.?\d*/, "Result contains numeric values");
	}
	
	ok(1, $test_name);
	return 1;
}

=head2 extension_ok

Verify an extension is installed.

=cut

sub extension_ok {
	my ($node, $dbname, $extname, $test_name) = @_;
	$test_name ||= "Extension $extname is installed";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, 
		"SELECT extname FROM pg_extension WHERE extname = '$extname';",
		tuples_only => 1
	);
	
	unless ($result->{success}) {
		fail($test_name);
		return 0;
	}
	
	my $output = $result->{stdout};
	$output =~ s/^\s+|\s+$//g;  # Trim leading/trailing whitespace
	
	is($output, $extname, $test_name);
	return $output eq $extname;
}

=head2 schema_ok

Verify a schema exists.

=cut

sub schema_ok {
	my ($node, $dbname, $schemaname, $test_name) = @_;
	$test_name ||= "Schema $schemaname exists";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SELECT nspname FROM pg_namespace WHERE nspname = '$schemaname';",
		tuples_only => 1
	);
	
	unless ($result->{success}) {
		fail($test_name);
		return 0;
	}
	
	my $output = $result->{stdout};
	$output =~ s/^\s+|\s+$//g;  # Trim leading/trailing whitespace
	
	is($output, $schemaname, $test_name);
	return $output eq $schemaname;
}

=head2 table_ok

Verify a table exists.

=cut

sub table_ok {
	my ($node, $dbname, $schemaname, $tablename, $test_name) = @_;
	$test_name ||= "Table $schemaname.$tablename exists";
	$dbname ||= 'postgres';
	
	my $schema_part = $schemaname ? "$schemaname." : "";
	my $result = $node->psql($dbname,
		"SELECT tablename FROM pg_tables WHERE schemaname = '$schemaname' AND tablename = '$tablename';",
		tuples_only => 1
	);
	
	unless ($result->{success}) {
		fail($test_name);
		return 0;
	}
	
	my $output = $result->{stdout};
	chomp $output;
	$output =~ s/^\s+|\s+$//g;  # Trim leading/trailing whitespace
	
	is($output, $tablename, $test_name);
	return $output eq $tablename;
}

=head2 function_ok

Verify a function exists.

=cut

sub function_ok {
	my ($node, $dbname, $schemaname, $funcname, $test_name) = @_;
	$test_name ||= "Function $schemaname.$funcname exists";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SELECT proname FROM pg_proc p JOIN pg_namespace n ON p.pronamespace = n.oid WHERE n.nspname = '$schemaname' AND p.proname = '$funcname';",
		tuples_only => 1
	);
	
	unless ($result->{success}) {
		fail($test_name);
		return 0;
	}
	
	my $output = $result->{stdout};
	chomp $output;
	
	is($output, $funcname, $test_name);
	return $output eq $funcname;
}

1;


