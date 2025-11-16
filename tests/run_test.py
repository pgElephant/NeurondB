#!/usr/bin/env python3

"""
run_test.py
Run SQL testcases in tests/sql by category and print concise results.

Output format per test:
  ✓ 2025-11-16 12:34:56  001_linreg.sql             2.34s
  ✗ 2025-11-16 12:35:01  002_logreg_negative.sql     0.18s

Categories:
  - basic:     files without `_advance` or `_negative`
  - advance:   files with `_advance.sql`
  - negative:  files with `_negative.sql`
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from typing import Iterable, List, Tuple


TESTS_SQL_DIR = os.path.join(os.path.dirname(__file__), "sql")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DEFAULT_ERROR_DIR = os.path.join(os.path.dirname(__file__), "error")
DEFAULT_DB = "neurondb"


def list_sql_files(category: str) -> List[str]:
	"""
	List SQL files by category from tests/sql.
	"""
	if not os.path.isdir(TESTS_SQL_DIR):
		raise FileNotFoundError(f"SQL directory not found: {TESTS_SQL_DIR}")

	all_files = [f for f in os.listdir(TESTS_SQL_DIR) if f.endswith(".sql")]
	all_files.sort()

	if category == "basic":
		return [
			os.path.join(TESTS_SQL_DIR, f)
			for f in all_files
			if "_advance.sql" not in f and "_negative.sql" not in f
		]
	elif category == "advance":
		return [
			os.path.join(TESTS_SQL_DIR, f)
			for f in all_files
			if f.endswith("_advance.sql")
		]
	elif category == "negative":
		return [
			os.path.join(TESTS_SQL_DIR, f)
			for f in all_files
			if f.endswith("_negative.sql")
		]
	elif category == "all":
		return [os.path.join(TESTS_SQL_DIR, f) for f in all_files]
	else:
		raise ValueError("category must be one of: basic, advance, negative, all")


def run_psql_file(dbname: str, sql_file: str, psql_path: str, verbose: bool = False) -> Tuple[bool, float, str, str]:
	"""
	Run a single SQL file through psql.
	Returns (success, elapsed_seconds, stdout, stderr).
	"""
	start = time.perf_counter()
	cmd = [
		psql_path,
		"-v", "ON_ERROR_STOP=1",
		"-d", dbname,
		"-f", sql_file,
	]
	proc = subprocess.Popen(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
	)
	out, err = proc.communicate()
	elapsed = time.perf_counter() - start
	success = proc.returncode == 0
	if verbose:
		# Echo captured output on verbose mode for visibility
		if out:
			sys.stdout.write(out)
			if not out.endswith("\n"):
				sys.stdout.write("\n")
		if err:
			sys.stderr.write(err)
			if not err.endswith("\n"):
				sys.stderr.write("\n")
	return success, elapsed, out or "", err or ""


def format_line(ok: bool, when: datetime, name: str, elapsed: float) -> str:
	"""
	Format a single result line: tick/cross, timestamp, name, elapsed.
	"""
	icon = "✓" if ok else "✗"
	ts = when.strftime("%Y-%m-%d %H:%M:%S")
	return f"{icon} {ts}  {name:<28} {elapsed:.2f}s"


def find_psql() -> str:
	"""
	Resolve psql executable in PATH.
	"""
	psql = os.environ.get("PSQL", "psql")
	return psql


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run SQL testcases by category with clean output."
	)
	parser.add_argument(
		"--category",
		choices=["basic", "advance", "negative", "all"],
		default="basic",
		help="Which test category to run (default: basic).",
	)
	parser.add_argument(
		"--db",
		default=DEFAULT_DB,
		help=f"Database name (default: {DEFAULT_DB}).",
	)
	parser.add_argument(
		"--psql",
		default=find_psql(),
		help="Path to psql executable (default: resolve from PATH).",
	)
	parser.add_argument(
		"-v", "--verbose",
		action="store_true",
		help="Verbose mode: print psql stdout/stderr for each test.",
	)
	return parser.parse_args()


def ensure_dir(path: str) -> None:
	if not os.path.isdir(path):
		os.makedirs(path, exist_ok=True)


def write_artifacts(name: str, ok: bool, out_dir: str, err_dir: str,
		    stdout_text: str, stderr_text: str) -> None:
	"""
	Write per-test artifacts. On success, write to output directory.
	On failure, write both stdout/stderr to error directory.
	"""
	target_dir = out_dir if ok else err_dir
	ensure_dir(target_dir)
	base = os.path.splitext(os.path.basename(name))[0]
	out_path = os.path.join(target_dir, f"{base}.out")
	err_path = os.path.join(target_dir, f"{base}.err")
	# Always write stdout
	with open(out_path, "w", encoding="utf8") as f:
		f.write(stdout_text)
	# Write stderr if present or on failure
	if (stderr_text and stderr_text.strip()) or not ok:
		with open(err_path, "w", encoding="utf8") as f:
			f.write(stderr_text)


def main() -> int:
	args = parse_args()
	# Extend parser with output/error dirs without breaking existing users
	# Backward-compatible defaults
	if not hasattr(args, "output_dir"):
		setattr(args, "output_dir", DEFAULT_OUTPUT_DIR)
	if not hasattr(args, "error_dir"):
		setattr(args, "error_dir", DEFAULT_ERROR_DIR)

	sql_files = list_sql_files(args.category)
	if not sql_files:
		print(f"No SQL files found for category '{args.category}' in {TESTS_SQL_DIR}")
		return 2

	total = len(sql_files)
	passed = 0
	failed = 0
	t0 = time.perf_counter()

	for path in sql_files:
		when = datetime.now()
		name = os.path.basename(path)
		# Show which test is starting
		start_ts = when.strftime("%Y-%m-%d %H:%M:%S")
		print(f"… {start_ts}  {name}")
		ok, elapsed, out_text, err_text = run_psql_file(args.db, path, args.psql, verbose=getattr(args, "verbose", False))
		# Persist artifacts
		write_artifacts(name, ok, DEFAULT_OUTPUT_DIR, DEFAULT_ERROR_DIR, out_text, err_text)
		if ok:
			passed += 1
		else:
			failed += 1
		print(format_line(ok, when, name, elapsed))
		if not ok:
			err_tail = "\n".join((err_text or "").strip().splitlines()[-5:])
			# brief context line on failure
			if err_tail:
				print(f"    {err_tail}")

	total_elapsed = time.perf_counter() - t0
	print("")
	print(f"Total: {total}, Passed: {passed}, Failed: {failed}, Elapsed: {total_elapsed:.2f}s")
	return 0 if failed == 0 else 1


if __name__ == "__main__":
	sys.exit(main())


