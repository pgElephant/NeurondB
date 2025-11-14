#!/usr/bin/env python3
"""
Run tests from serial schedule files (basic, advance, negative).
Usage: python3 run_tests_by_schedule.py [basic|advance|negative] [--gpu|--cpu]
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

def run_sql_file(pguser, pgdatabase, sql_file):
    """Run a SQL file using psql and return success, output, and elapsed time."""
    try:
        start_time = time.time()
        cmd = ['psql', '-U', pguser, '-d', pgdatabase, '-f', sql_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        elapsed = time.time() - start_time

        if result.returncode == 0:
            return True, result.stdout, elapsed

        # Fallback (without user arg)
        start_time = time.time()
        cmd = ['psql', '-d', pgdatabase, '-f', sql_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        elapsed = time.time() - start_time

        output = result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        return result.returncode == 0, output, elapsed
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0.0
        return False, str(e), elapsed

def main():
    parser = argparse.ArgumentParser(
        description='Run tests from serial schedule files'
    )
    parser.add_argument('schedule',
        nargs='?',
        default='basic',
        choices=['basic', 'advance', 'negative'],
        help='Schedule file to use: basic, advance, or negative (default: basic)'
    )
    parser.add_argument('--gpu', action='store_true', default=True, help='Enable GPU mode (default: on)')
    parser.add_argument('--cpu', action='store_true', help='Enable CPU-only mode')
    parser.add_argument('--pguser',
        default=os.environ.get('PGUSER', 'pge'),
        help='PostgreSQL user (default: pge or $PGUSER)'
    )
    parser.add_argument('--pgdatabase',
        default=os.environ.get('PGDATABASE', 'pmdb'),
        help='PostgreSQL database (default: pmdb or $PGDATABASE)'
    )
    parser.add_argument('--verbose',
        action='store_true',
        default=False,
        help='Enable verbose output'
    )
    args = parser.parse_args()

    # If --gpu is not specified and --cpu is not specified, default to --gpu True
    # Since argparse default=True for --gpu, this holds unless explicitly --cpu
    if args.cpu:
        args.gpu = False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    schedule_file = os.path.join(script_dir, f'serialsechudal_{args.schedule}.txt')

    if not os.path.exists(schedule_file):
        print(f"Error: Schedule file not found: {schedule_file}", file=sys.stderr)
        sys.exit(1)

    # Read test files from schedule
    test_files = []
    with open(schedule_file, 'r') as f:
        for line in f:
            l = line.strip()
            if l and not l.startswith('#'):
                fname = l.split()[0] if l.split() else l
                if not fname.endswith('.sql'):
                    fname += '.sql'
                test_files.append(fname)

    passed = []
    failed = []

    print(f"\n{'='*80}")
    print(f"Running {args.schedule.upper()} Tests ({len(test_files)} tests)")
    print(f"{'='*80}\n")
    print(f"{'Status':<8} {'Timestamp':<20} {'Test Name':<30} {'Time':<12}")
    print(f"{'-'*8} {'-'*20} {'-'*30} {'-'*12}")

    for test_file in test_files:
        test_path = os.path.join(script_dir, test_file)
        if not os.path.exists(test_path):
            print(f"✗ SKIP   {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<20} {test_file:<30} {'N/A':<12}")
            failed.append((test_file, 0.0))
            continue

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        success, output, elapsed = run_sql_file(args.pguser, args.pgdatabase, test_path)

        if elapsed < 1.0:
            time_str = f"{elapsed*1000:.0f}ms"
        else:
            time_str = f"{elapsed:.2f}s"

        if success:
            status = "✓ PASS"
            passed.append((test_file, elapsed))
            print(f"{status:<8} {timestamp:<20} {test_file:<30} {time_str:<12}")
        else:
            status = "✗ FAIL"
            failed.append((test_file, elapsed))
            print(f"{status:<8} {timestamp:<20} {test_file:<30} {time_str:<12}")
            if args.verbose:
                print(f"\nError output for {test_file}:")
                print(output[:500])  # First 500 chars
                print("...")

    total_time = sum(t for _, t in passed) + sum(t for _, t in failed)
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
                if os.path.exists(test_path):
                    _, output, _ = run_sql_file(args.pguser, args.pgdatabase, test_path)
                    print(f"\n{test_file}:")
                    print(output)
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
