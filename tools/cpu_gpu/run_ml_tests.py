#!/usr/bin/env python3
"""
Run ML test scripts (001-007) after ensuring dataset is set up.
Supports: --ml, --gpu, --cpu, --hf, --all.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

def check_table_exists(pguser, pgdatabase, table_name):
    """Check if a table exists in the database."""
    try:
        import psycopg2
        try:
            conn = psycopg2.connect(user=pguser, database=pgdatabase, host="localhost")
        except:
            conn = psycopg2.connect(user=pguser, database=pgdatabase)
        cur = conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
        """, (table_name,))
        exists = cur.fetchone()[0]
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        print(f"Error checking table: {e}", file=sys.stderr)
        return False

def run_sql_file(pguser, pgdatabase, sql_file, prepend_sql=None):
    """
    Run a SQL file using psql and return success, output, and elapsed time.
    If `prepend_sql` is given, temporarily write a temp file with prepended SQL lines.
    """
    try:
        start_time = time.time()

        sql_to_run = sql_file
        temp_file = None
        cleanup = False

        if prepend_sql:
            with open(sql_file, "r") as orig:
                orig_content = orig.read()
            import tempfile
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".sql", mode="w", encoding="utf-8")
            if isinstance(prepend_sql, list):
                for line in prepend_sql:
                    temp.write(line.rstrip()+"\n")
            else:
                temp.write(prepend_sql.rstrip()+"\n")
            temp.write("\n")
            temp.write(orig_content)
            temp.flush()
            temp_file = temp.name
            temp.close()
            sql_to_run = temp_file
            cleanup = True

        cmd = ['psql', '-U', pguser, '-d', pgdatabase, '-f', sql_to_run]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        elapsed = time.time() - start_time

        if cleanup:
            os.unlink(temp_file)

        if result.returncode == 0:
            return True, result.stdout, elapsed

        # Fallback (without user arg)
        start_time = time.time()
        cmd = ['psql', '-d', pgdatabase, '-f', sql_to_run]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        elapsed = time.time() - start_time

        if cleanup and os.path.exists(sql_to_run):
            try: os.unlink(sql_to_run)
            except: pass

        output = result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        return result.returncode == 0, output, elapsed
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0.0
        return False, str(e), elapsed

def setup_database_and_extension(pguser, database_name='pmdb'):
    """Drop and create database, then drop and create extension neurondb."""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        print(f"Setting up database '{database_name}' and extension 'neurondb'...")
        
        # Connect to default 'postgres' database to drop/create target database
        try:
            conn = psycopg2.connect(user=pguser, database='postgres', host="localhost")
        except:
            conn = psycopg2.connect(user=pguser, database='postgres')
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Terminate existing connections to the database
        cur.execute(f"""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '{database_name}' AND pid <> pg_backend_pid();
        """)
        
        # Drop database if exists
        cur.execute(f"DROP DATABASE IF EXISTS {database_name};")
        print(f"  Dropped database '{database_name}' (if it existed)")
        
        # Create database
        cur.execute(f"CREATE DATABASE {database_name};")
        print(f"  Created database '{database_name}'")
        
        cur.close()
        conn.close()
        
        # Connect to the new database
        try:
            conn = psycopg2.connect(user=pguser, database=database_name, host="localhost")
        except:
            conn = psycopg2.connect(user=pguser, database=database_name)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Drop extension if exists
        cur.execute("DROP EXTENSION IF EXISTS neurondb CASCADE;")
        print(f"  Dropped extension 'neurondb' (if it existed)")
        
        # Create extension
        cur.execute("CREATE EXTENSION neurondb;")
        print(f"  Created extension 'neurondb'")
        
        cur.close()
        conn.close()
        
        print(f"✓ Database '{database_name}' and extension 'neurondb' setup complete")
        return True
    except Exception as e:
        print(f"Error setting up database and extension: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

def download_dataset(dataset, pguser, pgdatabase, script_dir):
    """Download and set up dataset using pmlb package."""
    try:
        import pmlb
        import pandas as pd
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        print(f"Downloading dataset '{dataset}' using pmlb package...")
        try:
            from pmlb import fetch_data
            df = fetch_data(dataset)
        except (ImportError, AttributeError):
            try:
                from pmlb import load_data
                df = load_data(dataset)
            except (ImportError, AttributeError):
                import pmlb
                if hasattr(pmlb, 'fetch_data'):
                    df = pmlb.fetch_data(dataset)
                elif hasattr(pmlb, 'load_data'):
                    df = pmlb.load_data(dataset)
                else:
                    raise ImportError("pmlb does not have fetch_data or load_data")
        if df is None or df.empty:
            raise ValueError(f"Failed to download dataset '{dataset}'")

        print(f"Dataset downloaded: {len(df)} rows, {len(df.columns)} columns")
        
        try:
            conn = psycopg2.connect(user=pguser, database=pgdatabase, host="localhost")
        except:
            conn = psycopg2.connect(user=pguser, database=pgdatabase)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        feature_cols = []
        label_col = None

        def quote_identifier(name):
            normalized = name.strip().strip('"').replace(' ', '_').replace('\n', '')
            if '-' in normalized or (normalized and normalized[0].isdigit()):
                return f'"{normalized}"'
            return normalized
        
        for col in df.columns:
            normalized = col.strip().strip('"').replace(' ', '_').replace('\n', '')
            if normalized.lower() in ('class', 'target', 'label'):
                label_col = quote_identifier(normalized)
            else:
                feature_cols.append(quote_identifier(normalized))
        if not label_col:
            raise ValueError("No label column found (expected 'class', 'target', or 'label')")

        table_name = f"pmlb_{dataset}"
        col_defs = ', '.join([f"{col} float4" for col in feature_cols] + [f"{label_col} int"])
        cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        cur.execute(f"CREATE TABLE {table_name} ({col_defs});")
        print(f"Importing data into {table_name}...")
        for idx, row in df.iterrows():
            values = [str(row[col]) for col in df.columns if quote_identifier(col) in feature_cols]
            values.append(str(int(row[label_col.strip('\"')])))
            cur.execute(
                f"INSERT INTO {table_name} VALUES ({', '.join(['%s'] * len(values))})",
                values
            )
        feat_array = "ARRAY[" + ", ".join(feature_cols) + "]::vector"
        cur.execute(f"""
            DROP TABLE IF EXISTS {table_name}_with_vectors CASCADE;
            CREATE TABLE {table_name}_with_vectors AS
            SELECT 
                {feat_array} AS features,
                {label_col} AS label
            FROM {table_name};
        """)
        cur.execute("DROP TABLE IF EXISTS sample_train CASCADE;")
        cur.execute("DROP TABLE IF EXISTS sample_test CASCADE;")
        cur.execute(f"""
            CREATE TABLE sample_train AS
            SELECT *
            FROM {table_name}_with_vectors
            WHERE random() < 0.8;
        """)
        cur.execute(f"""
            CREATE TABLE sample_test AS
            SELECT *
            FROM {table_name}_with_vectors
            WHERE random() >= 0.8;
        """)
        cur.execute("ANALYZE sample_train;")
        cur.execute("ANALYZE sample_test;")
        cur.execute("SELECT COUNT(*) FROM sample_train;")
        train_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM sample_test;")
        test_count = cur.fetchone()[0]
        print(f"Created tables: sample_train ({train_count} rows), sample_test ({test_count} rows)")
        cur.close()
        conn.close()
        return True
    except ImportError:
        print("Error: pmlb package not found. Please install: pip install pmlb", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error downloading dataset: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Run ML test scripts after ensuring dataset is set up'
    )
    parser.add_argument('--ml',
        choices=['test','ml','hf','all'],
        default='all',
        help='ML test mode: ml, hf, all (default: all)'
    )
    parser.add_argument('--gpu', action='store_true', help='Enable GPU mode')
    parser.add_argument('--cpu', action='store_true', help='Enable CPU-only mode (disable GPU)')
    parser.add_argument('--dataset',
        default='adult',
        help='Dataset name to use (default: adult)'
    )
    parser.add_argument('--pguser',
        default=os.environ.get('PGUSER', 'pge'),
        help='PostgreSQL user (default: pge or $PGUSER)'
    )
    parser.add_argument('--pgdatabase',
        default=os.environ.get('PGDATABASE', 'pmdb'),
        help='PostgreSQL database (default: pmdb or $PGDATABASE)'
    )
    parser.add_argument('--setup-db',
        action='store_true',
        default=True,
        help='Drop and create database and extension (default: True)'
    )
    parser.add_argument('--no-setup-db',
        action='store_false',
        dest='setup_db',
        help='Skip database and extension setup'
    )
    parser.add_argument('--skip-setup',
        action='store_true',
        help='Skip dataset setup (assume tables already exist)'
    )
    parser.add_argument('--verbose',
        action='store_true',
        default=False,
        help='Enable verbose output (default: False)'
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Setup database and extension first
    if args.setup_db:
        if not setup_database_and_extension(args.pguser, args.pgdatabase):
            print("Error: Failed to setup database and extension", file=sys.stderr)
            sys.exit(1)

    if not args.skip_setup:
        print("Checking for sample_train and sample_test tables...")
        if not check_table_exists(args.pguser, args.pgdatabase, 'sample_train') or \
           not check_table_exists(args.pguser, args.pgdatabase, 'sample_test'):
            print("Tables not found. Downloading dataset...")
            if not download_dataset(args.dataset, args.pguser, args.pgdatabase, script_dir):
                print("Error: Failed to download and set up dataset", file=sys.stderr)
                sys.exit(1)
        else:
            print("Tables found. Skipping dataset download.")
    else:
        print("Skipping dataset setup (--skip-setup specified)")

    # Test listing logic per mode
    test_tables = {
        "ml": "ml_serialsechudal.txt",
        "hf": "hf_serialsechudal.txt"
    }
    all_mlintests = [
        '001_linreg.sql',
        '002_logreg.sql',
        '003_rf.sql',
        '004_svm.sql',
        '005_dt.sql',
        '006_ridge.sql',
        '007_lasso.sql'
    ]
    all_hftests = [
        '008_hf_embed.sql',
        '009_hf_complete.sql',
        '010_hf_rerank.sql',
        '011_hf_tokenize.sql'
    ]
    schedule_files = {}
    for k, v in test_tables.items():
        schedule_files[k] = os.path.join(script_dir, v)

    # Mode logic per prompt
    test_files = []
    if args.ml == 'ml' or args.ml == 'test':
        schedule_file = schedule_files['ml']
        if os.path.exists(schedule_file):
            with open(schedule_file, 'r') as f:
                for line in f:
                    l = line.strip()
                    if l and not l.startswith('#'):
                        fname = l.split()[0] if l.split() else l
                        if not fname.endswith('.sql'):
                            fname += '.sql'
                        test_files.append(fname)
        else:
            test_files = all_mlintests
    elif args.ml == 'hf':
        schedule_file = schedule_files['hf']
        if os.path.exists(schedule_file):
            with open(schedule_file, 'r') as f:
                for line in f:
                    l = line.strip()
                    if l and not l.startswith('#'):
                        fname = l.split()[0] if l.split() else l
                        if not fname.endswith('.sql'):
                            fname += '.sql'
                        test_files.append(fname)
        else:
            test_files = all_hftests
    elif args.ml == 'all' or not args.ml:
        # load all from ml_serialsechudal.txt and hf_serialsechudal.txt, with fallback to hardcoded full lists
        seen = set()
        for group, schedule_file in schedule_files.items():
            if os.path.exists(schedule_file):
                with open(schedule_file, 'r') as f:
                    for line in f:
                        l = line.strip()
                        if l and not l.startswith('#'):
                            fname = l.split()[0] if l.split() else l
                            if not fname.endswith('.sql'):
                                fname += '.sql'
                            if fname not in seen:
                                seen.add(fname)
                                test_files.append(fname)
        # fallback if not found
        for f in all_mlintests+all_hftests:
            if f not in test_files:
                test_files.append(f)
    else:
        # default: all
        for f in all_mlintests+all_hftests:
            if f not in test_files:
                test_files.append(f)

    # Prepend SQL depending on --gpu/--cpu
    gpu_prelines = [
        "SET neurondb.gpu_enabled = on;",
        "SET neurondb.gpu_kernels = 'l2,cosine,ip,linreg_train,linreg_predict';",
        "SELECT neurondb_gpu_enable();"
    ]
    cpu_prelines = [
        "SET neurondb.gpu_enabled = off;",
        "SELECT neurondb_gpu_enable();"
    ]
    prepend_lines = None
    if args.gpu and args.cpu:
        print("Conflicting options: --gpu and --cpu. Specify only one.", file=sys.stderr)
        sys.exit(1)
    if args.gpu:
        prepend_lines = gpu_prelines
    elif args.cpu:
        prepend_lines = cpu_prelines
    # else, default: do NOT change GPU setting

    passed = []
    failed = []

    # Print header
    if args.verbose:
        print(f"\n{'='*80}")
        print(f"Running ML Tests (mode: {args.ml})")
        print(f"{'='*80}\n")
    else:
        print(f"{'Status':<8} {'Timestamp':<20} {'Test Name':<30} {'Time':<12}")
        print(f"{'-'*8} {'-'*20} {'-'*30} {'-'*12}")

    for test_file in test_files:
        test_path = os.path.join(script_dir, test_file)
        if not os.path.exists(test_path):
            if args.verbose:
                print(f"Warning: {test_file} not found, skipping")
            continue

        if args.verbose:
            mode_str = 'GPU' if args.gpu else ('CPU' if args.cpu else 'default')
            print(f"\n{'='*60}")
            print(f"Running {test_file} (mode: {mode_str})")
            print('='*60)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        success, output, elapsed = run_sql_file(
            args.pguser, args.pgdatabase, test_path, prepend_sql=prepend_lines
        )

        if elapsed < 1.0:
            time_str = f"{elapsed*1000:.0f}ms"
        else:
            time_str = f"{elapsed:.2f}s"

        if success:
            status = "✓ PASS"
            passed.append((test_file, elapsed))
            if not args.verbose:
                print(f"{status:<8} {timestamp:<20} {test_file:<30} {time_str:<12}")
            else:
                print(f"✓ {test_file} PASSED ({time_str})")
        else:
            status = "✗ FAIL"
            failed.append((test_file, elapsed))
            if not args.verbose:
                print(f"{status:<8} {timestamp:<20} {test_file:<30} {time_str:<12}")
            else:
                print(f"✗ {test_file} FAILED ({time_str})")
                print(output)

    total_time = sum(t for _, t in passed) + sum(t for _, t in failed)
    if not args.verbose:
        print(f"{'-'*8} {'-'*20} {'-'*30} {'-'*12}")

    print(f"\n{'='*80}")
    print("Summary:")
    print(f"  Passed: {len(passed)}/{len(test_files)}")
    print(f"  Failed: {len(failed)}/{len(test_files)}")
    print(f"  Total time: {total_time:.2f}s")

    if failed:
        print("\nFailed tests:")
        for test, elapsed in failed:
            time_str = f"{elapsed*1000:.0f}ms" if elapsed < 1.0 else f"{elapsed:.2f}s"
            print(f"  ✗ {test} ({time_str})")
        if args.verbose:
            print("\nDetailed error output:")
            for test_file, _ in failed:
                test_path = os.path.join(script_dir, test_file)
                _, output, _ = run_sql_file(args.pguser, args.pgdatabase, test_path, prepend_sql=prepend_lines)
                print(f"\n{test_file}:")
                print(output)
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
