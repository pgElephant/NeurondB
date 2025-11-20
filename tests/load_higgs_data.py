#!/usr/bin/env python3

"""
Load HIGGS.csv.gz into test_train and test_test tables.
HIGGS format: 29 columns (label + 28 features)
"""

import argparse
import csv
import gzip
import os
import sys
import psycopg2
from psycopg2.extras import execute_batch
import time

def get_conn(dbname="neurondb"):
	"""Get PostgreSQL connection."""
	host = os.getenv("PGHOST", "localhost")
	port = int(os.getenv("PGPORT", "5432"))
	user = os.getenv("PGUSER") or os.getenv("USER")
	password = os.getenv("PGPASSWORD")
	
	return psycopg2.connect(
		dbname=dbname,
		user=user,
		password=password,
		host=host,
		port=port
	)

def load_higgs_data(csv_path, dbname, train_split=0.8, limit=None):
	"""
	Load HIGGS data into higgs.test_train and higgs.test_test tables.
	The test runner will create views from these tables with LIMIT based on num_rows.
	Using higgs schema to avoid issues when dropping the neurondb extension.
	"""
	
	conn = get_conn(dbname)
	cur = conn.cursor()
	
	# Create higgs schema if it doesn't exist
	print("Creating higgs schema...")
	cur.execute("CREATE SCHEMA IF NOT EXISTS higgs")
	conn.commit()
	
	# Drop and recreate tables in higgs schema
	# Use REAL[] instead of vector to avoid dependency on neurondb extension
	print("Creating tables in higgs schema...")
	cur.execute("""
		DROP TABLE IF EXISTS higgs.test_train CASCADE;
		DROP TABLE IF EXISTS higgs.test_test CASCADE;
		
		CREATE TABLE higgs.test_train (
			features REAL[],
			label integer
		);
		
		CREATE TABLE higgs.test_test (
			features REAL[],
			label integer
		);
	""")
	conn.commit()
	
	# Prepare data arrays
	train_data = []
	test_data = []
	
	print(f"Reading {csv_path}...")
	start_time = time.time()
	row_count = 0
	
	# Open file (handle both .gz and regular files)
	if csv_path.endswith('.gz'):
		f = gzip.open(csv_path, 'rt')
	else:
		f = open(csv_path, 'r')
	
	try:
		reader = csv.reader(f)
		for row in reader:
			if limit and row_count >= limit:
				break
			
			if len(row) != 29:
				continue
			
			label = int(float(row[0].strip()))
			features = [float(x.strip()) for x in row[1:29]]
			
			# Convert features to PostgreSQL array format
			features_array = features  # Already a list, will be converted to array
			
			# Split into train/test (80/20)
			if row_count % 10 < int(train_split * 10):
				train_data.append((features, label))
			else:
				test_data.append((features, label))
			
			row_count += 1
			
			if row_count % 100000 == 0:
				elapsed = time.time() - start_time
				print(f"  Processed {row_count:,} rows ({elapsed:.1f}s)")
		
	finally:
		f.close()
	
	print(f"\nTotal rows processed: {row_count:,}")
	print(f"Train rows: {len(train_data):,}")
	print(f"Test rows: {len(test_data):,}")
	
	# Insert into database using batch inserts
	print("\nInserting into test_train...")
	start_time = time.time()
	
	batch_size = 1000
	for i in range(0, len(train_data), batch_size):
		batch = train_data[i:i+batch_size]
		values = []
		for features_str, label in batch:
			# Use proper vector casting
			values.append((features_str, label))
		
		# Use execute_batch for efficient inserts
		# Use REAL[] array type (native PostgreSQL, no extension dependency)
		execute_batch(
			cur,
			"INSERT INTO higgs.test_train (features, label) VALUES (%s::REAL[], %s)",
			values
		)
		conn.commit()
		
		if (i // batch_size) % 10 == 0:
			elapsed = time.time() - start_time
			print(f"  Inserted {i + len(batch):,} / {len(train_data):,} train rows ({elapsed:.1f}s)")
	
	train_elapsed = time.time() - start_time
	print(f"✓ Loaded {len(train_data):,} rows into test_train in {train_elapsed:.1f}s")
	
	# Load test data
	print("\nInserting into test_test...")
	start_time = time.time()
	
	for i in range(0, len(test_data), batch_size):
		batch = test_data[i:i+batch_size]
		values = []
		for features_str, label in batch:
			values.append((features_str, label))
		
		execute_batch(
			cur,
			"INSERT INTO higgs.test_test (features, label) VALUES (%s::REAL[], %s)",
			values
		)
		conn.commit()
		
		if (i // batch_size) % 10 == 0:
			elapsed = time.time() - start_time
			print(f"  Inserted {i + len(batch):,} / {len(test_data):,} test rows ({elapsed:.1f}s)")
	
	test_elapsed = time.time() - start_time
	print(f"✓ Loaded {len(test_data):,} rows into test_test in {test_elapsed:.1f}s")
	
	total_elapsed = train_elapsed + test_elapsed
	print(f"\n✓ Total load time: {total_elapsed:.1f}s")
	
	# Verify
	cur.execute("SELECT COUNT(*) FROM higgs.test_train")
	train_count = cur.fetchone()[0]
	cur.execute("SELECT COUNT(*) FROM higgs.test_test")
	test_count = cur.fetchone()[0]
	
	print(f"\nFinal counts:")
	print(f"  higgs.test_train: {train_count:,} rows")
	print(f"  higgs.test_test: {test_count:,} rows")
	
	cur.close()
	conn.close()

def main():
	parser = argparse.ArgumentParser(description="Load HIGGS.csv.gz into test_train and test_test")
	parser.add_argument("csv_path", help="Path to HIGGS.csv.gz file")
	parser.add_argument("--dbname", default="neurondb", help="Database name")
	parser.add_argument("--train-split", type=float, default=0.8, help="Train/test split ratio (default: 0.8)")
	parser.add_argument("--limit", type=int, help="Limit number of rows to load")
	
	args = parser.parse_args()
	
	if not os.path.exists(args.csv_path):
		print(f"Error: File not found: {args.csv_path}", file=sys.stderr)
		return 1
	
	load_higgs_data(args.csv_path, args.dbname, args.train_split, args.limit)
	return 0

if __name__ == "__main__":
	sys.exit(main())
