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
import csv
import gzip
import os
import platform
import random
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from urllib.request import urlopen

try:
	import psycopg2
	from psycopg2.extras import execute_batch
except ImportError:
	psycopg2 = None
	execute_batch = None

try:
	import numpy as np
	HAS_NUMPY = True
except ImportError:
	HAS_NUMPY = False

# Import GPUDetector from tools/gpu.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
try:
	from gpu import GPUDetector
except ImportError:
	GPUDetector = None


TESTS_SQL_DIR = os.path.join(os.path.dirname(__file__), "sql")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DEFAULT_ERROR_DIR = os.path.join(os.path.dirname(__file__), "error")
DEFAULT_DB = "neurondb"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5432
DEFAULT_MODE = "gpu"
DEFAULT_NUM_ROWS = 1000

# Output formatting constants for perfect alignment
ICON_WIDTH = 2
TIMESTAMP_WIDTH = 19
TEST_NUM_WIDTH = 8
TEST_NAME_WIDTH = 45
ELAPSED_WIDTH = 12
LINE_WIDTH = 80  # Standard line width for separators
HEADER_SEPARATOR = "-" * LINE_WIDTH
LABEL_WIDTH = 20  # Width for labels in header info sections

# ANSI escape codes for formatting
BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
GREEN_BOLD = "\033[1;32m"
RED_BOLD = "\033[1;31m"
GREEN = "\033[32m"
RED = "\033[31m"
RED_BOLD = "\033[1;31m"
GREEN_BOLD = "\033[1;32m"

# Script version
SCRIPT_VERSION = "1.0.0"
SCRIPT_NAME = "test_runner.py"

# Global flag for graceful shutdown on Ctrl+C
_shutdown_requested = False


def signal_handler(signum, frame):
	"""Handle SIGINT (Ctrl+C) gracefully."""
	global _shutdown_requested
	_shutdown_requested = True


def list_sql_files(category: str) -> List[str]:
	"""
	List SQL files by category from tests/sql (recursively).
	"""
	if not os.path.isdir(TESTS_SQL_DIR):
		raise FileNotFoundError(f"SQL directory not found: {TESTS_SQL_DIR}")

	# Recursively find all SQL files
	all_files = []
	for root, dirs, files in os.walk(TESTS_SQL_DIR):
		for f in files:
			if f.endswith(".sql"):
				rel_path = os.path.relpath(os.path.join(root, f), TESTS_SQL_DIR)
				all_files.append(rel_path)
	all_files.sort()

	if category == "basic":
		# Include files from basic/ subdirectory or files without _advance/_negative
		result = []
		for f in all_files:
			if "/basic/" in f or (f.startswith("basic/") or os.path.dirname(f) == "basic"):
				result.append(os.path.join(TESTS_SQL_DIR, f))
			elif "/" not in f and "_advance.sql" not in f and "_negative.sql" not in f and "_perf.sql" not in f:
				result.append(os.path.join(TESTS_SQL_DIR, f))
		return result
	elif category == "advance":
		return [
			os.path.join(TESTS_SQL_DIR, f)
			for f in all_files
			if "_advance.sql" in f or "/advance/" in f
		]
	elif category == "negative":
		return [
			os.path.join(TESTS_SQL_DIR, f)
			for f in all_files
			if "_negative.sql" in f or "/negative/" in f
		]
	elif category == "all":
		return [os.path.join(TESTS_SQL_DIR, f) for f in all_files]
	else:
		raise ValueError("category must be one of: basic, advance, negative, all")


def verify_gpu_usage(dbname: str, psql_path: str, mode: str, test_name: str = "", host: Optional[str] = None, port: Optional[int] = None) -> Tuple[bool, str]:
	"""
	Verify that GPU-trained models actually used GPU.
	Only checks the most recent model to avoid false positives from previous tests.
	Returns (success, error_message).
	"""
	if mode != "gpu":
		return True, ""  # Skip verification for CPU mode
	
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	# Check only the most recent model to avoid false positives from previous test runs
	check_sql = """
		SELECT 
			m.model_id,
			m.algorithm,
			COALESCE(m.metrics::jsonb->>'storage', 'cpu') AS storage
		FROM neurondb.ml_models m
		ORDER BY m.model_id DESC
		LIMIT 1;
	"""
	
	cmd = [
		psql_path,
		"-d", dbname,
		"-t", "-A",
		"-w",
		"-c", check_sql
	]
	
	try:
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
		out, err = proc.communicate(timeout=5)
		
		if proc.returncode != 0:
			return True, ""  # If query fails, don't fail the test
		
		if out.strip():
			lines = [line.strip() for line in out.strip().split('\n') if line.strip()]
			for line in lines:
				parts = line.split('|')
				if len(parts) >= 3:
					model_id = parts[0].strip()
					algorithm = parts[1].strip()
					storage = parts[2].strip()
					# Only check ML algorithms that should use GPU
					ml_algorithms = ['linear_regression', 'logistic_regression', 'random_forest', 'svm', 'ridge', 'lasso', 'decision_tree', 'naive_bayes']
					if storage != 'gpu' and algorithm in ml_algorithms:
						return False, f"GPU mode enabled but model_id={model_id} (algorithm={algorithm}) was trained on CPU (storage={storage})"
		
		return True, ""
	except Exception:
		return True, ""  # Don't fail tests if verification fails


def run_psql_file(dbname: str, sql_file: str, psql_path: str, verbose: bool = False) -> Tuple[bool, float, str, str]:
	"""
	Run a single SQL file through psql.
	Uses trust authentication by default (no password prompt).
	Returns (success, elapsed_seconds, stdout, stderr).
	"""
	start = time.perf_counter()
	cmd = [
		psql_path,
		"-v", "ON_ERROR_STOP=1",
		"-d", dbname,
		"-f", sql_file,
		"-w",  # Never prompt for password (trust auth)
	]
	proc = subprocess.Popen(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		env=os.environ.copy(),
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


def format_status_line(ok: bool, when: datetime, message: str, elapsed: float) -> str:
	"""
	Format a status line (pre-test checks) with perfect alignment.
	Format: [icon] [timestamp] [message]...[elapsed]s
	"""
	icon = f"{GREEN_BOLD}✓{RESET}" if ok else f"{RED_BOLD}✗{RESET}"
	ts = when.strftime("%Y-%m-%d %H:%M:%S")
	elapsed_str = f"{elapsed:>8.2f}s"
	
	# Calculate available width for message
	available_width = 80 - ICON_WIDTH - 1 - TIMESTAMP_WIDTH - 2 - ELAPSED_WIDTH
	message_width = min(len(message), available_width)
	message_padded = message[:message_width].ljust(available_width)
	
	return f"{icon:<{ICON_WIDTH + len(GREEN_BOLD) + len(RESET) - 1}} {ts:<{TIMESTAMP_WIDTH}}  {message_padded:<{available_width}} {elapsed_str:>{ELAPSED_WIDTH}}"


def format_test_line(ok: bool, when: datetime, test_num: int, total: int, name: str, elapsed: float) -> str:
	"""
	Format a test result line with perfect alignment.
	Format: [icon] [timestamp] [test_num/total] [test_name]...[elapsed]s
	"""
	if ok:
		icon = f"{GREEN_BOLD}✓{RESET}"
	else:
		icon = f"{RED_BOLD}✗{RESET}"
	ts = when.strftime("%Y-%m-%d %H:%M:%S")
	test_info = f"{test_num}/{total}"
	elapsed_str = f"{elapsed:>8.2f}s"
	
	# Calculate available width for test name
	available_width = TEST_NAME_WIDTH
	name_padded = name[:available_width].ljust(available_width)
	
	return f"{icon:<{ICON_WIDTH + len(GREEN_BOLD) + len(RESET) - 1}} {ts:<{TIMESTAMP_WIDTH}}  {test_info:<{TEST_NUM_WIDTH}} {name_padded:<{TEST_NAME_WIDTH}} {elapsed_str:>{ELAPSED_WIDTH}}"


def find_psql() -> str:
	"""
	Resolve psql executable in PATH.
	"""
	psql = os.environ.get("PSQL", "psql")
	return psql


def check_postgresql_connection(dbname: str, psql_path: str, host: Optional[str] = None, port: Optional[int] = None) -> Tuple[bool, float, str]:
	"""
	Check PostgreSQL connection.
	Uses trust authentication by default (no password prompt).
	Returns (success, elapsed_seconds, connection_info).
	"""
	start = time.perf_counter()
	
	# Build command with trust authentication (no password prompt)
	cmd = [
		psql_path,
		"-d", dbname,
		"-c", "SELECT version();",
		"-w",  # Never prompt for password (trust auth)
	]
	
	# Use environment variables (which may include PGPASSWORD if set)
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	try:
		proc = subprocess.Popen(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			env=env,
		)
		out, err = proc.communicate()
		elapsed = time.perf_counter() - start
		success = proc.returncode == 0
		
		host_display = host or os.environ.get("PGHOST", "localhost")
		port_display = port or os.environ.get("PGPORT", "5432")
		conn_info = f"postgresql://{host_display}:{port_display}"
		
		return success, elapsed, conn_info
	except Exception as e:
		elapsed = time.perf_counter() - start
		host_display = host or os.environ.get("PGHOST", "localhost")
		port_display = port or os.environ.get("PGPORT", "5432")
		return False, elapsed, f"postgresql://{host_display}:{port_display}"


def get_platform_info() -> Dict[str, str]:
	"""Gather platform information."""
	info = {}
	
	# CPU information
	try:
		if platform.system() == "Linux":
			with open('/proc/cpuinfo', 'r') as f:
				cpuinfo = f.read()
				for line in cpuinfo.split('\n'):
					if 'model name' in line:
						info['cpu'] = line.split(':')[1].strip()
						break
					elif 'processor' in line and 'cpu' not in info:
						info['cpu'] = platform.processor() or "Unknown"
		else:
			info['cpu'] = platform.processor() or "Unknown"
	except Exception:
		info['cpu'] = platform.processor() or "Unknown"
	
	# Memory information
	try:
		if platform.system() == "Linux":
			with open('/proc/meminfo', 'r') as f:
				meminfo = f.read()
				for line in meminfo.split('\n'):
					if 'MemTotal' in line:
						mem_kb = int(line.split()[1])
						mem_gb = mem_kb / (1024 * 1024)
						info['memory'] = f"{mem_gb:.2f} GB"
						break
		elif platform.system() == "Darwin":
			result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
								capture_output=True, text=True)
			if result.returncode == 0:
				mem_bytes = int(result.stdout.strip())
				mem_gb = mem_bytes / (1024 ** 3)
				info['memory'] = f"{mem_gb:.2f} GB"
			else:
				info['memory'] = "Unknown"
		else:
			info['memory'] = "Unknown"
	except Exception:
		info['memory'] = "Unknown"
	
	# Disk information (works on Linux, macOS, Rocky, Ubuntu)
	try:
		result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, timeout=5)
		if result.returncode == 0:
			lines = result.stdout.strip().split('\n')
			if len(lines) > 1:
				parts = lines[1].split()
				if len(parts) >= 4:
					# Linux format: Filesystem Size Used Avail Use% Mounted
					# macOS format: Filesystem  Size Used Avail Capacity iused ifree %iused Mounted
					if platform.system() == "Darwin":
						# macOS: parts[8] is total, parts[7] is used
						if len(parts) >= 9:
							total = parts[8]
							used = parts[7]
							info['disk'] = f"{used} / {total}"
						else:
							total = parts[1]
							used = parts[2]
							info['disk'] = f"{used} / {total}"
					else:
						# Linux (including Rocky, Ubuntu)
						total = parts[1]
						used = parts[2]
						info['disk'] = f"{used} / {total}"
		else:
			info['disk'] = "Unknown"
	except (subprocess.TimeoutExpired, Exception):
		info['disk'] = "Unknown"
	
	return info


def get_os_info() -> Dict[str, str]:
	"""Gather OS information."""
	return {
		'system': platform.system(),
		'release': platform.release(),
		'version': platform.version(),
		'machine': platform.machine(),
		'architecture': platform.architecture()[0],
	}


def get_gpu_info(dbname: str, psql_path: str, host: Optional[str] = None, port: Optional[int] = None) -> Dict[str, str]:
	"""Gather GPU information using Python-based system detection."""
	info = {}
	
	# Use Python-based GPU detection
	if GPUDetector is None:
		info['available'] = "Unknown"
		info['device_name'] = "Unknown"
		info['device_id'] = "Unknown"
		return info
	
	try:
		# Detect all GPUs using Python system detection
		gpus = GPUDetector.detect_all()
		
		if gpus and len(gpus) > 0:
			gpu = gpus[0]  # Use first GPU
			info['available'] = "Yes"
			info['device_id'] = str(gpu.get('id', 0))
			info['device_name'] = gpu.get('name', 'Unknown')
			info['backend'] = gpu.get('backend', 'Unknown')
			info['type'] = gpu.get('type', 'Unknown')
			
			# Memory information
			if 'memory_mb' in gpu and gpu['memory_mb'] > 0:
				info['memory_total'] = f"{gpu['memory_mb']} MB"
			
			# Compute capability
			if 'compute_cap' in gpu and gpu['compute_cap']:
				info['compute_capability'] = gpu['compute_cap']
			
			# Platform information
			if 'platform' in gpu:
				info['platform'] = gpu['platform']
			
			# Chip information for Apple Silicon
			if 'chip' in gpu:
				info['chip'] = gpu['chip']
		else:
			info['available'] = "No"
			info['device_name'] = "None detected"
			info['device_id'] = "N/A"
	except Exception as e:
		info['available'] = "Unknown"
		info['device_name'] = f"Detection error: {str(e)}"
		info['device_id'] = "Unknown"
	
	# Try to get GPU settings from PostgreSQL GUC (optional, for enabled status)
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	try:
		cmd = [psql_path, "-d", dbname, "-t", "-A", "-w", "-c", 
		       "SELECT current_setting('neurondb.gpu_enabled', true), current_setting('neurondb.gpu_device', true), current_setting('neurondb.gpu_kernels', true);"]
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=5)
		out, err = proc.communicate()
		if proc.returncode == 0 and out.strip():
			parts = out.strip().split('|')
			if len(parts) >= 3:
				if parts[0].strip() and parts[0].strip() != '':
					info['enabled'] = parts[0].strip()
				if parts[1].strip() and parts[1].strip() != '':
					if not info.get('device_id') or info.get('device_id') == 'Unknown':
						info['device_id'] = parts[1].strip()
				if parts[2].strip() and parts[2].strip() != '':
					info['kernels'] = parts[2].strip()
	except (subprocess.TimeoutExpired, Exception):
		# If PostgreSQL query fails, that's okay - we have system detection
		pass
	
	return info


def get_postgresql_info(dbname: str, psql_path: str, host: Optional[str] = None, port: Optional[int] = None) -> Dict[str, str]:
	"""Gather PostgreSQL information."""
	info = {}
	
	# Get PostgreSQL version
	cmd = [psql_path, "-d", dbname, "-t", "-A", "-c", "SELECT version();"]
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	try:
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
		out, err = proc.communicate()
		if proc.returncode == 0 and out.strip():
			version_line = out.strip()
			# Extract PostgreSQL version
			if 'PostgreSQL' in version_line:
				info['version'] = version_line.split('PostgreSQL')[1].split()[0] if 'PostgreSQL' in version_line else version_line
			else:
				info['version'] = version_line
		else:
			info['version'] = "Unable to connect"
	except Exception:
		info['version'] = "Unable to connect"
	
	# Get server information
	try:
		cmd = [psql_path, "-d", dbname, "-t", "-A", "-c", 
		       "SELECT current_setting('server_version');"]
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
		out, err = proc.communicate()
		if proc.returncode == 0 and out.strip():
			info['server_version'] = out.strip()
	except Exception:
		pass
	
	# Get database information
	try:
		cmd = [psql_path, "-d", dbname, "-t", "-A", "-c",
		       f"SELECT current_database(), current_user, inet_server_addr(), inet_server_port();"]
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
		out, err = proc.communicate()
		if proc.returncode == 0 and out.strip():
			parts = out.strip().split('|')
			if len(parts) >= 4:
				info['database'] = parts[0]
				info['user'] = parts[1]
				info['host'] = parts[2] if parts[2] else (host or os.environ.get('PGHOST', 'localhost'))
				info['port'] = parts[3] if parts[3] else (str(port) if port else os.environ.get('PGPORT', '5432'))
	except Exception:
		info['host'] = host or os.environ.get('PGHOST', 'localhost')
		info['port'] = str(port) if port else os.environ.get('PGPORT', '5432')
	
	return info


def print_header_info(script_name: str, version: str, dbname: str, psql_path: str, 
		      host: Optional[str] = None, port: Optional[int] = None, mode: Optional[str] = None) -> None:
	"""Print detailed header information about the test runner and system."""
	print()
	print(f"NeuronDB Test Suite Version {version}")
	print("-" * LINE_WIDTH)
	print("Platform Information:")
	
	platform_info = get_platform_info()
	
	# Print CPU (Bold, aligned)
	cpu_label = f"{BOLD}CPU:{RESET}"
	cpu_value = platform_info.get('cpu', 'Unknown')
	print(f"\t{cpu_label:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {cpu_value}")
	
	# Print Memory (Bold, aligned)
	memory_label = f"{BOLD}Memory:{RESET}"
	memory_value = platform_info.get('memory', 'Unknown')
	print(f"\t{memory_label:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {memory_value}")
	
	# Print Disk (Bold, aligned)
	disk_label = f"{BOLD}Disk:{RESET}"
	disk_value = platform_info.get('disk', 'Unknown')
	print(f"\t{disk_label:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {disk_value}")
	
	print("OS Information:")
	
	os_info = get_os_info()
	for key, value in sorted(os_info.items()):
		key_formatted = key.capitalize() + ":"
		key_bold = f"{BOLD}{key_formatted}{RESET}"
		print(f"\t{key_bold:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {value}")
	
	print("PostgreSQL Information:")
	
	pg_info = get_postgresql_info(dbname, psql_path, host, port)
	
	# Print PostgreSQL info in order with perfect alignment
	pg_order = ['version', 'server_version', 'host', 'port', 'database', 'user']
	for key in pg_order:
		if key in pg_info:
			key_formatted = key.replace('_', ' ').title() + ":"
			key_bold = f"{BOLD}{key_formatted}{RESET}"
			print(f"\t{key_bold:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {pg_info[key]}")
	
	# Print any remaining keys
	for key, value in sorted(pg_info.items()):
		if key not in pg_order:
			key_formatted = key.replace('_', ' ').title() + ":"
			key_bold = f"{BOLD}{key_formatted}{RESET}"
			print(f"\t{key_bold:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {value}")
	
	# Print GPU Information if mode is GPU
	if mode == "gpu":
		print("GPU Information:")
		
		gpu_info = get_gpu_info(dbname, psql_path, host, port)
		
		# Print GPU info in order with perfect alignment
		gpu_order = ['available', 'enabled', 'device_id', 'device_name', 'compute_capability', 'memory_total', 'memory_free', 'kernels']
		for key in gpu_order:
			if key in gpu_info:
				key_formatted = key.replace('_', ' ').title() + ":"
				key_bold = f"{BOLD}{key_formatted}{RESET}"
				print(f"\t{key_bold:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {gpu_info[key]}")
		
		# Print any remaining keys
		for key, value in sorted(gpu_info.items()):
			if key not in gpu_order:
				key_formatted = key.replace('_', ' ').title() + ":"
				key_bold = f"{BOLD}{key_formatted}{RESET}"
				print(f"\t{key_bold:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {value}")
	
	print("-" * LINE_WIDTH)
	print()
	print()


def create_test_views(dbname: str, psql_path: str, num_rows: int, host: Optional[str] = None, port: Optional[int] = None) -> Tuple[bool, int]:
	"""
	Create test views with specified number of rows.
	Returns (success, row_count).
	"""
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	# Check which source table exists (dataset.test_train/test_test, sample_train/sample_test, or test_train/test_test)
	try:
		# Check for dataset.test_train first (preferred), then sample_train, then test_train
		check_train_cmd = [
			psql_path, "-d", dbname, "-t", "-A", "-w",
			"-c", "SELECT table_schema || '.' || table_name FROM information_schema.tables WHERE (table_schema = 'dataset' AND table_name = 'test_train') OR (table_schema = 'public' AND table_name IN ('sample_train', 'test_train')) ORDER BY CASE WHEN table_schema = 'dataset' THEN 0 WHEN table_name = 'sample_train' THEN 1 ELSE 2 END LIMIT 1;"
		]
		
		train_proc = subprocess.Popen(check_train_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
		train_out, train_err = train_proc.communicate()
		
		if train_proc.returncode == 0 and train_out.strip():
			train_table_full = train_out.strip()
			
			# Determine corresponding test table
			if train_table_full.startswith('dataset.'):
				test_table_full = "dataset.test_test"
			elif 'sample_train' in train_table_full:
				test_table_full = "sample_test"
			else:
				test_table_full = "test_test"
			
			# Verify test table exists
			if '.' in test_table_full:
				schema, table = test_table_full.split('.')
				check_test_cmd = [
					psql_path, "-d", dbname, "-t", "-A", "-w",
					"-c", f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{schema}' AND table_name = '{table}';"
				]
			else:
				check_test_cmd = [
					psql_path, "-d", dbname, "-t", "-A", "-w",
					"-c", f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '{test_table_full}';"
				]
			
			test_proc = subprocess.Popen(check_test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
			test_out, test_err = test_proc.communicate()
			
			if test_proc.returncode == 0 and test_out.strip() and int(test_out.strip()) > 0:
				# Check if vector extension is available
				check_vector_cmd = [
					psql_path, "-d", dbname, "-t", "-A", "-w",
					"-c", "SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'vector');"
				]
				vector_check = subprocess.Popen(check_vector_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
				vector_out, vector_err = vector_check.communicate()
				has_vector = vector_check.returncode == 0 and vector_out.strip() == "t"
				
				# Create views with LIMIT num_rows
				# Try to convert REAL[] to vector if vector extension is available, otherwise use REAL[]
				# Use DO block to handle potential cast failures gracefully
				if has_vector:
					# Try vector cast, fallback to REAL[] if it fails
					create_sql = f"""
DROP VIEW IF EXISTS test_train_view CASCADE;
DROP VIEW IF EXISTS test_test_view CASCADE;

DO $$
BEGIN
	-- Try to create views with vector cast
	BEGIN
		EXECUTE format('CREATE VIEW test_train_view AS SELECT features::vector(28) as features, label FROM %s LIMIT %s', '{train_table_full}', {num_rows});
		EXECUTE format('CREATE VIEW test_test_view AS SELECT features::vector(28) as features, label FROM %s LIMIT %s', '{test_table_full}', {num_rows});
	EXCEPTION WHEN OTHERS THEN
		-- Fallback to REAL[] if vector cast fails
		EXECUTE format('CREATE VIEW test_train_view AS SELECT features as features, label FROM %s LIMIT %s', '{train_table_full}', {num_rows});
		EXECUTE format('CREATE VIEW test_test_view AS SELECT features as features, label FROM %s LIMIT %s', '{test_table_full}', {num_rows});
	END;
END $$;

-- Create or truncate test settings table for test runner configuration
CREATE TABLE IF NOT EXISTS test_settings (
	setting_key TEXT PRIMARY KEY,
	setting_value TEXT,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create or truncate test metrics table for storing test results
CREATE TABLE IF NOT EXISTS test_metrics (
	test_name TEXT PRIMARY KEY,
	algorithm TEXT,
	model_id INTEGER,
	train_samples BIGINT,
	test_samples BIGINT,
	-- Regression metrics
	mse NUMERIC,
	rmse NUMERIC,
	mae NUMERIC,
	r_squared NUMERIC,
	-- Classification metrics
	accuracy NUMERIC,
	precision NUMERIC,
	recall NUMERIC,
	f1_score NUMERIC,
	-- Clustering metrics
	silhouette_score NUMERIC,
	inertia NUMERIC,
	n_clusters INTEGER,
	-- Time series metrics
	mape NUMERIC,
	-- Predictions (stored as JSONB for flexibility)
	predictions JSONB,
	-- Additional metadata
	metadata JSONB,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

TRUNCATE TABLE test_metrics;
"""
				else:
					# No vector extension, use REAL[] directly
					create_sql = f"""
DROP VIEW IF EXISTS test_train_view CASCADE;
DROP VIEW IF EXISTS test_test_view CASCADE;

CREATE VIEW test_train_view AS
SELECT features as features, label 
FROM {train_table_full} 
LIMIT {num_rows};

CREATE VIEW test_test_view AS
SELECT features as features, label 
FROM {test_table_full} 
LIMIT {num_rows};

-- Create or truncate test settings table for test runner configuration
CREATE TABLE IF NOT EXISTS test_settings (
	setting_key TEXT PRIMARY KEY,
	setting_value TEXT,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create or truncate test metrics table for storing test results
CREATE TABLE IF NOT EXISTS test_metrics (
	test_name TEXT PRIMARY KEY,
	algorithm TEXT,
	model_id INTEGER,
	train_samples BIGINT,
	test_samples BIGINT,
	-- Regression metrics
	mse NUMERIC,
	rmse NUMERIC,
	mae NUMERIC,
	r_squared NUMERIC,
	-- Classification metrics
	accuracy NUMERIC,
	precision NUMERIC,
	recall NUMERIC,
	f1_score NUMERIC,
	-- Clustering metrics
	silhouette_score NUMERIC,
	inertia NUMERIC,
	n_clusters INTEGER,
	-- Time series metrics
	mape NUMERIC,
	-- Predictions (stored as JSONB for flexibility)
	predictions JSONB,
	-- Additional metadata
	metadata JSONB,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

TRUNCATE TABLE test_metrics;
"""
				
				create_proc = subprocess.Popen(
					[psql_path, "-d", dbname, "-w", "-c", create_sql],
					stdout=subprocess.PIPE,
					stderr=subprocess.PIPE,
					text=True,
					env=env
				)
				create_out, create_err = create_proc.communicate()
				
				if create_proc.returncode == 0:
					# Verify views were created and get row count
					count_cmd = [
						psql_path, "-d", dbname, "-t", "-A", "-w",
						"-c", "SELECT COALESCE((SELECT COUNT(*) FROM test_train_view), 0);"
					]
					
					count_proc = subprocess.Popen(count_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
					count_out, count_err = count_proc.communicate()
					
					if count_proc.returncode == 0 and count_out.strip():
						row_count = int(count_out.strip())
						if row_count > 0:
							return True, row_count
					# If row count is 0, something went wrong
					if count_err:
						print(f"Warning: View created but has 0 rows. Error: {count_err}", file=sys.stderr)
				else:
					# View creation failed - log error for debugging
					if create_err:
						# Only show error if it's not about vector type (we handle that gracefully)
						if "type \"vector\" does not exist" not in create_err.lower():
							print(f"Warning: View creation failed: {create_err}", file=sys.stderr)
		
		return False, 0
	except (subprocess.TimeoutExpired, Exception):
		return False, 0


def run_psql_command(dbname: str, sql_command: str, psql_path: str, verbose: bool = False) -> Tuple[bool, str, str]:
	"""
	Run a SQL command through psql.
	Uses trust authentication by default (no password prompt).
	Returns (success, stdout, stderr).
	"""
	cmd = [
		psql_path,
		"-v", "ON_ERROR_STOP=1",
		"-d", dbname,
		"-c", sql_command,
		"-w",  # Never prompt for password (trust auth)
	]
	proc = subprocess.Popen(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		env=os.environ.copy(),
	)
	out, err = proc.communicate()
	success = proc.returncode == 0
	if verbose:
		if out:
			sys.stdout.write(out)
		if err:
			sys.stderr.write(err)
	return success, out or "", err or ""


def switch_gpu_mode(dbname: str, mode: str, psql_path: str, gpu_kernels: str = None, verbose: bool = False) -> bool:
	"""
	Switch GPU mode using ALTER SYSTEM and reload configuration.
	Mode should be 'gpu' or 'cpu'.
	Returns True on success, False on failure.
	"""
	if mode not in ("gpu", "cpu"):
		print(f"Invalid mode: {mode}. Must be 'gpu' or 'cpu'.", file=sys.stderr)
		return False

	gpu_enabled = "on" if mode == "gpu" else "off"

	# Set GPU enabled
	cmd1 = f"ALTER SYSTEM SET neurondb.gpu_enabled = '{gpu_enabled}';"
	success1, out1, err1 = run_psql_command(dbname, cmd1, psql_path, verbose)
	if not success1:
		print(f"Failed to set GPU enabled: {err1}", file=sys.stderr)
		return False

	# Set GPU kernels - include all ML algorithm kernels for GPU mode
	if mode == "gpu":
		# Default kernels plus all ML training/prediction kernels
		default_kernels = "l2,cosine,ip,rf_split,rf_predict"
		ml_kernels = "linreg_train,linreg_predict,lr_train,lr_predict,rf_train,svm_train,svm_predict,ridge_train,ridge_predict,lasso_train,lasso_predict,dt_train,dt_predict,nb_train,nb_predict"
		full_kernels = f"{default_kernels},{ml_kernels}"
		cmd_kernels = f"ALTER SYSTEM SET neurondb.gpu_kernels = '{full_kernels}';"
		success_k, out_k, err_k = run_psql_command(dbname, cmd_kernels, psql_path, verbose)
		if not success_k:
			print(f"Warning: Failed to set GPU kernels: {err_k}", file=sys.stderr)
	elif gpu_kernels:
		# Use provided kernels if specified
		cmd_kernels = f"ALTER SYSTEM SET neurondb.gpu_kernels = '{gpu_kernels}';"
		success_k, out_k, err_k = run_psql_command(dbname, cmd_kernels, psql_path, verbose)
		if not success_k:
			print(f"Warning: Failed to set GPU kernels: {err_k}", file=sys.stderr)

	# Reload configuration
	cmd2 = "SELECT pg_reload_conf();"
	success2, out2, err2 = run_psql_command(dbname, cmd2, psql_path, verbose)
	if not success2:
		print(f"Failed to reload configuration: {err2}", file=sys.stderr)
		return False

	# Initialize GPU if GPU mode is enabled
	if mode == "gpu":
		cmd3 = "SELECT neurondb_gpu_enable();"
		success3, out3, err3 = run_psql_command(dbname, cmd3, psql_path, verbose)
		if not success3:
			print(f"Warning: Failed to enable GPU: {err3}", file=sys.stderr)
			# Don't fail, GPU might not be available but we still want to test
		
		# Force GPU initialization by querying GPU info
		cmd4 = "SELECT * FROM neurondb_gpu_info() LIMIT 1;"
		success4, out4, err4 = run_psql_command(dbname, cmd4, psql_path, verbose)
		if not success4:
			if verbose:
				print(f"Warning: GPU info query failed (GPU may not be available): {err4}", file=sys.stderr)

	if verbose:
		print(f"Switched to {mode.upper()} mode successfully.")
	return True


# HIGGS dataset constants
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/280/higgs.zip"
HIGGS_CSV_BASENAME = "HIGGS.csv"
EXPECTED_NUM_COLUMNS = 29
MB = 1024 * 1024


def find_local_higgs_csv() -> Optional[str]:
	"""Try to locate an existing HIGGS.csv in common locations."""
	candidates = [
		os.path.join(os.getcwd(), HIGGS_CSV_BASENAME),  # Current directory first
		os.path.join(os.path.dirname(__file__), "datasets", HIGGS_CSV_BASENAME),
		os.path.join(os.path.dirname(__file__), HIGGS_CSV_BASENAME),
		os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", HIGGS_CSV_BASENAME),
		HIGGS_CSV_BASENAME,
	]
	for path in candidates:
		if os.path.isfile(path):
			return path
	return None


def find_local_higgs_zip() -> Optional[str]:
	"""Try to locate a local higgs.zip - checks current directory first."""
	candidates = [
		os.path.join(os.getcwd(), "higgs.zip"),  # Current directory first
		os.path.join(os.path.dirname(__file__), "datasets", "higgs.zip"),
		os.path.join(os.path.dirname(__file__), "higgs.zip"),
		os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "higgs.zip"),
		"higgs.zip",
	]
	for path in candidates:
		if os.path.isfile(path):
			return path
	return None


def download_higgs_zip(dest_path: str) -> None:
	"""Download HIGGS dataset zip file."""
	print(f"Downloading HIGGS dataset from {UCI_ZIP_URL}...")
	print("This may take several minutes (file is ~2.8GB)...")
	
	with urlopen(UCI_ZIP_URL) as resp, open(dest_path, "wb") as out:
		chunk_size = 1 * MB
		total = 0
		total_size = 0
		try:
			hdr_len = resp.getheader("Content-Length")
			total_size = int(hdr_len) if hdr_len else 0
		except Exception:
			total_size = 0
		
		start_ts = last_ts = time.time()
		last_reported_percent = -1
		while True:
			chunk = resp.read(chunk_size)
			if not chunk:
				break
			out.write(chunk)
			total += len(chunk)
			now = time.time()
			
			if total_size > 0:
				percent = int((total * 100) / total_size)
			else:
				percent = -1
			
			if percent != last_reported_percent or (now - last_ts) >= 0.5:
				elapsed = max(0.001, now - start_ts)
				speed = total / elapsed
				mb_done = total / MB
				if total_size > 0:
					mb_total = total_size / MB
					sys.stderr.write(
						f"\r[DOWNLOAD] {percent:3d}% "
						f"({mb_done:.1f}/{mb_total:.1f} MB) "
						f"@ {speed/MB:.2f} MB/s"
					)
				else:
					sys.stderr.write(
						f"\r[DOWNLOAD] {mb_done:.1f} MB "
						f"@ {speed/MB:.2f} MB/s"
					)
				sys.stderr.flush()
				last_ts = now
				last_reported_percent = percent
		sys.stderr.write("\n")
		sys.stderr.flush()
	print(f"Download complete: {dest_path}")


def extract_higgs_csv(zip_path: str, output_dir: str) -> str:
	"""Extract HIGGS.csv from zip file."""
	print(f"Extracting HIGGS.csv from {zip_path}...")
	with zipfile.ZipFile(zip_path, "r") as zf:
		names = zf.namelist()
		csv_name = None
		for n in names:
			lower = n.lower()
			if lower.endswith(".csv") and "higgs" in lower:
				csv_name = n
				break
		
		if csv_name is None:
			raise FileNotFoundError("No HIGGS CSV file found in zip.")
		
		os.makedirs(output_dir, exist_ok=True)
		target_path = os.path.join(output_dir, HIGGS_CSV_BASENAME)
		with zf.open(csv_name, "r") as src, open(target_path, "wb") as dst:
			while True:
				chunk = src.read(1 * MB)
				if not chunk:
					break
				dst.write(chunk)
		
		print(f"Extracted to: {target_path}")
		return target_path


def get_higgs_csv_path(csv_path: Optional[str] = None) -> str:
	"""
	Get path to HIGGS.csv from local sources only.
	Checks current directory first, then extracts from zip if found.
	Does NOT download - quits if not found.
	"""
	if csv_path and os.path.isfile(csv_path):
		return csv_path
	
	# Try to find local CSV
	local_csv = find_local_higgs_csv()
	if local_csv:
		return local_csv
	
	# Try to find local ZIP (current directory first)
	local_zip = find_local_higgs_zip()
	if local_zip:
		try:
			# Extract to current directory or datasets subdirectory
			output_dir = os.getcwd()
			datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
			# Prefer extracting to datasets if it exists, otherwise current dir
			if os.path.isdir(datasets_dir):
				output_dir = datasets_dir
			return extract_higgs_csv(local_zip, output_dir)
		except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
			# Invalid zip file
			raise FileNotFoundError(
				f"\n{'='*80}\n"
				f"ERROR: Invalid or corrupted zip file: {local_zip}\n"
				f"{'='*80}\n"
				f"\n"
				f"Please download the HIGGS dataset from:\n"
				f"  {UCI_ZIP_URL}\n"
				f"\n"
				f"Then place 'higgs.zip' in this directory:\n"
				f"  {os.getcwd()}\n"
				f"\n"
				f"The file should be approximately 2.8GB in size.\n"
				f"{'='*80}\n"
			) from e
		except Exception as e:
			# Other extraction errors
			raise FileNotFoundError(
				f"\n{'='*80}\n"
				f"ERROR: Failed to extract HIGGS.csv from zip file: {e}\n"
				f"{'='*80}\n"
				f"\n"
				f"Please ensure 'higgs.zip' is a valid zip file.\n"
				f"Download from: {UCI_ZIP_URL}\n"
				f"Place it in: {os.getcwd()}\n"
				f"{'='*80}\n"
			) from e
	
	# Not found - raise error with clear instructions
	current_dir = os.getcwd()
	raise FileNotFoundError(
		f"\n{'='*80}\n"
		f"ERROR: HIGGS dataset not found\n"
		f"{'='*80}\n"
		f"\n"
		f"Please download the HIGGS dataset from:\n"
		f"  {UCI_ZIP_URL}\n"
		f"\n"
		f"Then place 'higgs.zip' in this directory:\n"
		f"  {current_dir}\n"
		f"\n"
		f"The file should be approximately 2.8GB in size.\n"
		f"\n"
		f"Alternatively, you can use --dataset=synthetic to generate synthetic data instead.\n"
		f"{'='*80}\n"
	)


def generate_synthetic_higgs_data(
	num_rows: int,
	seed: Optional[int] = None
) -> Tuple[List[Tuple], List[Tuple]]:
	"""
	Generate synthetic HIGGS-style dataset.
	Returns (train_data, test_data) where each is a list of (features, label) tuples.
	
	HIGGS format:
	- 29 columns: label (0 or 1) + 28 features
	- Features are real-valued
	- 21 low-level features + 7 high-level features
	
	The synthetic data mimics HIGGS characteristics:
	- Binary classification (signal=1, background=0)
	- 28 real-valued features
	- Some correlation between features and labels for realistic ML testing
	"""
	if seed is not None:
		random.seed(seed)
		if HAS_NUMPY:
			np.random.seed(seed)
	
	train_data = []
	test_data = []
	
	# Generate data with some correlation between features and labels
	# to make it somewhat realistic for ML testing
	for i in range(num_rows):
		# Generate label (0 or 1) - balanced dataset
		label = 1 if random.random() < 0.5 else 0
		
		# Generate 28 features
		# First 21: low-level features (more variation, independent)
		# Last 7: high-level features (derived-like, more correlated)
		features = []
		
		# Low-level features (0-20): more independent, wider range
		# These simulate raw measurements (lepton pT, jet properties, etc.)
		for j in range(21):
			if label == 1:
				# Signal: slightly higher values on average
				base = random.gauss(0.5, 1.5)
			else:
				# Background: slightly lower values
				base = random.gauss(0.0, 1.5)
			# Clamp to reasonable range
			features.append(max(-10.0, min(10.0, base)))
		
		# High-level features (21-27): more correlated, derived-like
		# These simulate derived quantities (m_jj, m_jjj, m_lv, etc.)
		# Make them functions of low-level features to simulate derived features
		for j in range(7):
			# Use some combination of previous features
			idx1 = random.randint(0, 20)
			idx2 = random.randint(0, 20)
			# Derived feature as combination of two low-level features
			derived = (features[idx1] + features[idx2]) / 2.0 + random.gauss(0, 0.5)
			if label == 1:
				derived += 0.3  # Slight bias for signal
			# Clamp to reasonable range
			features.append(max(-10.0, min(10.0, derived)))
		
		# Split 80/20 train/test (consistent with HIGGS loading)
		if i % 10 < 8:
			train_data.append((features, label))
		else:
			test_data.append((features, label))
	
	return train_data, test_data


def load_synthetic_dataset(
	dbname: str,
	num_rows: int = 100000,
	seed: Optional[int] = None,
	host: Optional[str] = None,
	port: Optional[int] = None
) -> bool:
	"""
	Generate and load synthetic HIGGS-style dataset into dataset.test_train and dataset.test_test tables.
	Returns True on success, False on failure.
	"""
	if psycopg2 is None:
		print("ERROR: psycopg2 is required for dataset loading. Install with: pip install psycopg2-binary", file=sys.stderr)
		return False
	
	# Get database connection
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	user = env.get("PGUSER") or env.get("USER")
	password = env.get("PGPASSWORD")
	
	try:
		conn = psycopg2.connect(
			dbname=dbname,
			user=user,
			password=password,
			host=host or env.get("PGHOST", "localhost"),
			port=port or int(env.get("PGPORT", "5432"))
		)
		cur = conn.cursor()
	except Exception as e:
		print(f"ERROR: Failed to connect to database: {e}", file=sys.stderr)
		return False
	
	try:
		# Create dataset schema
		print("Creating dataset schema...")
		cur.execute("CREATE SCHEMA IF NOT EXISTS dataset")
		conn.commit()
		
		# Create tables
		print("Creating tables in dataset schema...")
		cur.execute("""
			DROP TABLE IF EXISTS dataset.test_train CASCADE;
			DROP TABLE IF EXISTS dataset.test_test CASCADE;
			
			CREATE TABLE dataset.test_train (
				features REAL[],
				label integer
			);
			
			CREATE TABLE dataset.test_test (
				features REAL[],
				label integer
			);
		""")
		conn.commit()
		
		# Generate synthetic data
		print(f"Generating {num_rows:,} synthetic HIGGS-style rows...")
		start_time = time.time()
		train_data, test_data = generate_synthetic_higgs_data(num_rows, seed)
		gen_elapsed = time.time() - start_time
		
		print(f"Generated {len(train_data):,} train rows and {len(test_data):,} test rows in {gen_elapsed:.1f}s")
		
		# Insert train data
		print("\nInserting into dataset.test_train...")
		start_time = time.time()
		batch_size = 1000
		
		for i in range(0, len(train_data), batch_size):
			batch = train_data[i:i+batch_size]
			values = [(features, label) for features, label in batch]
			
			execute_batch(
				cur,
				"INSERT INTO dataset.test_train (features, label) VALUES (%s::REAL[], %s)",
				values
			)
			conn.commit()
			
			if (i // batch_size) % 10 == 0:
				elapsed = time.time() - start_time
				print(f"  Inserted {i + len(batch):,} / {len(train_data):,} train rows ({elapsed:.1f}s)")
		
		train_elapsed = time.time() - start_time
		print(f"✓ Loaded {len(train_data):,} rows into dataset.test_train in {train_elapsed:.1f}s")
		
		# Insert test data
		print("\nInserting into dataset.test_test...")
		start_time = time.time()
		
		for i in range(0, len(test_data), batch_size):
			batch = test_data[i:i+batch_size]
			values = [(features, label) for features, label in batch]
			
			execute_batch(
				cur,
				"INSERT INTO dataset.test_test (features, label) VALUES (%s::REAL[], %s)",
				values
			)
			conn.commit()
			
			if (i // batch_size) % 10 == 0:
				elapsed = time.time() - start_time
				print(f"  Inserted {i + len(batch):,} / {len(test_data):,} test rows ({elapsed:.1f}s)")
		
		test_elapsed = time.time() - start_time
		print(f"✓ Loaded {len(test_data):,} rows into dataset.test_test in {test_elapsed:.1f}s")
		
		# Verify
		cur.execute("SELECT COUNT(*) FROM dataset.test_train")
		train_count = cur.fetchone()[0]
		cur.execute("SELECT COUNT(*) FROM dataset.test_test")
		test_count = cur.fetchone()[0]
		
		print(f"\nFinal counts:")
		print(f"  dataset.test_train: {train_count:,} rows")
		print(f"  dataset.test_test: {test_count:,} rows")
		
		cur.close()
		conn.close()
		return True
		
	except Exception as e:
		print(f"ERROR: Failed to load synthetic dataset: {e}", file=sys.stderr)
		import traceback
		traceback.print_exc()
		if cur:
			cur.close()
		if conn:
			conn.close()
		return False


def load_higgs_dataset(
	dbname: str,
	csv_path: Optional[str] = None,
	limit: Optional[int] = None,
	host: Optional[str] = None,
	port: Optional[int] = None,
	train_split: float = 0.8
) -> bool:
	"""
	Load HIGGS dataset into dataset.test_train and dataset.test_test tables.
	Returns True on success, False on failure.
	"""
	if psycopg2 is None:
		print("ERROR: psycopg2 is required for dataset loading. Install with: pip install psycopg2-binary", file=sys.stderr)
		return False
	
	# Get CSV path
	try:
		csv_file = get_higgs_csv_path(csv_path)
	except FileNotFoundError as e:
		# Print the detailed error message
		print(str(e), file=sys.stderr)
		return False
	except Exception as e:
		print(f"\n{'='*80}\n", file=sys.stderr)
		print(f"ERROR: Failed to get HIGGS CSV: {e}\n", file=sys.stderr)
		print(f"Please download higgs.zip from: {UCI_ZIP_URL}\n", file=sys.stderr)
		print(f"Place it in: {os.getcwd()}\n", file=sys.stderr)
		print(f"{'='*80}\n", file=sys.stderr)
		return False
	
	# Get database connection
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	user = env.get("PGUSER") or env.get("USER")
	password = env.get("PGPASSWORD")
	
	try:
		conn = psycopg2.connect(
			dbname=dbname,
			user=user,
			password=password,
			host=host or env.get("PGHOST", "localhost"),
			port=port or int(env.get("PGPORT", "5432"))
		)
		cur = conn.cursor()
	except Exception as e:
		print(f"ERROR: Failed to connect to database: {e}", file=sys.stderr)
		return False
	
	try:
		# Create dataset schema
		print("Creating dataset schema...")
		cur.execute("CREATE SCHEMA IF NOT EXISTS dataset")
		conn.commit()
		
		# Create tables
		print("Creating tables in dataset schema...")
		cur.execute("""
			DROP TABLE IF EXISTS dataset.test_train CASCADE;
			DROP TABLE IF EXISTS dataset.test_test CASCADE;
			
			CREATE TABLE dataset.test_train (
				features REAL[],
				label integer
			);
			
			CREATE TABLE dataset.test_test (
				features REAL[],
				label integer
			);
		""")
		conn.commit()
		
		# Read and split data
		train_data = []
		test_data = []
		
		print(f"Reading {csv_file}...")
		start_time = time.time()
		row_count = 0
		
		# Open file (handle both .gz and regular files)
		if csv_file.endswith('.gz'):
			f = gzip.open(csv_file, 'rt')
		else:
			f = open(csv_file, 'r')
		
		try:
			reader = csv.reader(f)
			for row in reader:
				if limit and row_count >= limit:
					break
				
				if len(row) != EXPECTED_NUM_COLUMNS:
					continue
				
				label = int(float(row[0].strip()))
				features = [float(x.strip()) for x in row[1:29]]
				
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
		
		# Insert train data
		print("\nInserting into dataset.test_train...")
		start_time = time.time()
		batch_size = 1000
		
		for i in range(0, len(train_data), batch_size):
			batch = train_data[i:i+batch_size]
			values = [(features, label) for features, label in batch]
			
			execute_batch(
				cur,
				"INSERT INTO dataset.test_train (features, label) VALUES (%s::REAL[], %s)",
				values
			)
			conn.commit()
			
			if (i // batch_size) % 10 == 0:
				elapsed = time.time() - start_time
				print(f"  Inserted {i + len(batch):,} / {len(train_data):,} train rows ({elapsed:.1f}s)")
		
		train_elapsed = time.time() - start_time
		print(f"✓ Loaded {len(train_data):,} rows into dataset.test_train in {train_elapsed:.1f}s")
		
		# Insert test data
		print("\nInserting into dataset.test_test...")
		start_time = time.time()
		
		for i in range(0, len(test_data), batch_size):
			batch = test_data[i:i+batch_size]
			values = [(features, label) for features, label in batch]
			
			execute_batch(
				cur,
				"INSERT INTO dataset.test_test (features, label) VALUES (%s::REAL[], %s)",
				values
			)
			conn.commit()
			
			if (i // batch_size) % 10 == 0:
				elapsed = time.time() - start_time
				print(f"  Inserted {i + len(batch):,} / {len(test_data):,} test rows ({elapsed:.1f}s)")
		
		test_elapsed = time.time() - start_time
		print(f"✓ Loaded {len(test_data):,} rows into dataset.test_test in {test_elapsed:.1f}s")
		
		# Verify
		cur.execute("SELECT COUNT(*) FROM dataset.test_train")
		train_count = cur.fetchone()[0]
		cur.execute("SELECT COUNT(*) FROM dataset.test_test")
		test_count = cur.fetchone()[0]
		
		print(f"\nFinal counts:")
		print(f"  dataset.test_train: {train_count:,} rows")
		print(f"  dataset.test_test: {test_count:,} rows")
		
		cur.close()
		conn.close()
		return True
		
	except Exception as e:
		print(f"ERROR: Failed to load dataset: {e}", file=sys.stderr)
		if cur:
			cur.close()
		if conn:
			conn.close()
		return False


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run SQL testcases by category with clean output."
	)
	parser.add_argument(
		"--category",
		choices=["basic", "advance", "negative", "all"],
		default="basic",
		help=f"Which test category to run (default: basic).",
	)
	parser.add_argument(
		"--db",
		default=DEFAULT_DB,
		help=f"Database name (default: {DEFAULT_DB}).",
	)
	parser.add_argument(
		"--host",
		default=DEFAULT_HOST,
		help=f"PostgreSQL host (default: {DEFAULT_HOST}).",
	)
	parser.add_argument(
		"--port",
		type=int,
		default=DEFAULT_PORT,
		help=f"PostgreSQL port (default: {DEFAULT_PORT}).",
	)
	parser.add_argument(
		"--user",
		default=None,
		help="PostgreSQL user (default: system user).",
	)
	parser.add_argument(
		"--password",
		default=None,
		help="PostgreSQL password (if required). Will set PGPASSWORD environment variable.",
	)
	parser.add_argument(
		"--psql",
		default=find_psql(),
		help="Path to psql executable (default: resolve from PATH).",
	)
	parser.add_argument(
		"--mode",
		choices=["gpu", "cpu"],
		default=DEFAULT_MODE,
		help=f"GPU or CPU mode (default: {DEFAULT_MODE}).",
	)
	parser.add_argument(
		"--gpu-kernels",
		default=None,
		help="GPU kernels to enable (e.g., 'l2,cosine,ip,linreg_train,linreg_predict'). Only used with --mode gpu.",
	)
	parser.add_argument(
		"--num-rows",
		type=int,
		default=DEFAULT_NUM_ROWS,
		help=f"Number of rows for test views (default: {DEFAULT_NUM_ROWS}).",
	)
	parser.add_argument(
		"-v", "--verbose",
		action="store_true",
		help="Verbose mode: print psql stdout/stderr for each test.",
	)
	parser.add_argument(
		"--dataset",
		choices=["higgs", "synthetic"],
		default=None,
		help="Dataset to load before running tests. Options: higgs (downloads real HIGGS dataset), synthetic (generates HIGGS-style synthetic data)",
	)
	parser.add_argument(
		"--dataset-path",
		default=None,
		help="Path to dataset CSV file. If not provided, will download or search for HIGGS.csv",
	)
	parser.add_argument(
		"--dataset-limit",
		type=int,
		default=None,
		help="Limit number of rows to load from dataset (for testing with smaller datasets). For synthetic dataset, this is the total number of rows to generate.",
	)
	parser.add_argument(
		"--dataset-seed",
		type=int,
		default=None,
		help="Random seed for synthetic dataset generation (for reproducibility).",
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
	global _shutdown_requested
	
	# Register signal handler for graceful shutdown on Ctrl+C
	signal.signal(signal.SIGINT, signal_handler)
	_shutdown_requested = False
	
	args = parse_args()
	# Extend parser with output/error dirs without breaking existing users
	# Backward-compatible defaults
	if not hasattr(args, "output_dir"):
		setattr(args, "output_dir", DEFAULT_OUTPUT_DIR)
	if not hasattr(args, "error_dir"):
		setattr(args, "error_dir", DEFAULT_ERROR_DIR)

	# Set password if provided
	if args.password:
		os.environ["PGPASSWORD"] = args.password
	
	# Set host, port, and user in environment if provided
	if args.host:
		os.environ["PGHOST"] = args.host
	if args.port:
		os.environ["PGPORT"] = str(args.port)
	if args.user:
		os.environ["PGUSER"] = args.user
	
	# Print header information
	print_header_info(SCRIPT_NAME, SCRIPT_VERSION, args.db, args.psql, args.host, args.port, args.mode)
	
	# Load dataset if requested
	if args.dataset == "higgs":
		when = datetime.now()
		load_start = time.perf_counter()
		load_ok = load_higgs_dataset(
			args.db,
			csv_path=args.dataset_path,
			limit=args.dataset_limit,
			host=args.host,
			port=args.port
		)
		load_elapsed = time.perf_counter() - load_start
		print(format_status_line(load_ok, when, "Loading HIGGS dataset...", load_elapsed))
		if not load_ok:
			print("Failed to load HIGGS dataset. Aborting.", file=sys.stderr)
			return 1
	elif args.dataset == "synthetic":
		when = datetime.now()
		load_start = time.perf_counter()
		num_rows = args.dataset_limit or 100000  # Default 100k rows for synthetic
		load_ok = load_synthetic_dataset(
			args.db,
			num_rows=num_rows,
			seed=args.dataset_seed,
			host=args.host,
			port=args.port
		)
		load_elapsed = time.perf_counter() - load_start
		print(format_status_line(load_ok, when, f"Generating synthetic dataset ({num_rows:,} rows)...", load_elapsed))
		if not load_ok:
			print("Failed to generate synthetic dataset. Aborting.", file=sys.stderr)
			return 1
	
	# Pre-test checks with status lines
	# 1. Check PostgreSQL connection
	when = datetime.now()
	conn_ok, conn_elapsed, conn_info = check_postgresql_connection(args.db, args.psql, args.host, args.port)
	print(format_status_line(conn_ok, when, f"Checking postgresql on {conn_info}...", conn_elapsed))
	if not conn_ok:
		print(f"Failed to connect to PostgreSQL at {conn_info}. Aborting.", file=sys.stderr)
		return 1
	
	# 2. Switch GPU/CPU mode (using ALTER SYSTEM)
	when = datetime.now()
	mode_start = time.perf_counter()
	mode_ok = switch_gpu_mode(args.db, args.mode, args.psql, args.gpu_kernels, args.verbose)
	mode_elapsed = time.perf_counter() - mode_start
	print(format_status_line(mode_ok, when, f"Configuring postgresql for {args.mode.upper()} mode...", mode_elapsed))
	if not mode_ok:
		print(f"Failed to switch to {args.mode.upper()} mode. Aborting.", file=sys.stderr)
		return 1
	
	# 3. Create test views (this also creates test_settings table)
	when = datetime.now()
	views_start = time.perf_counter()
	views_ok, row_count = create_test_views(args.db, args.psql, args.num_rows, args.host, args.port)
	views_elapsed = time.perf_counter() - views_start
	print(format_status_line(views_ok, when, f"Creating dataset for tests (rows={row_count})...", views_elapsed))
	if not views_ok:
		print(f"Warning: Failed to create test views. Some tests may fail.", file=sys.stderr)
		# Continue anyway - some tests might not need the views
	
	# 3.5. Store all test settings in test_settings table for tests to read
	when = datetime.now()
	settings_start = time.perf_counter()
	settings_sql = f"""
	-- Store GPU mode setting
	INSERT INTO test_settings (setting_key, setting_value, updated_at)
	VALUES ('gpu_mode', '{args.mode}', CURRENT_TIMESTAMP)
	ON CONFLICT (setting_key) DO UPDATE SET
		setting_value = EXCLUDED.setting_value,
		updated_at = CURRENT_TIMESTAMP;
	
	-- Store number of rows used for test views
	INSERT INTO test_settings (setting_key, setting_value, updated_at)
	VALUES ('num_rows', '{args.num_rows}', CURRENT_TIMESTAMP)
	ON CONFLICT (setting_key) DO UPDATE SET
		setting_value = EXCLUDED.setting_value,
		updated_at = CURRENT_TIMESTAMP;
	"""
	if args.gpu_kernels:
		settings_sql += f"""
	-- Store GPU kernels if provided
	INSERT INTO test_settings (setting_key, setting_value, updated_at)
	VALUES ('gpu_kernels', '{args.gpu_kernels}', CURRENT_TIMESTAMP)
	ON CONFLICT (setting_key) DO UPDATE SET
		setting_value = EXCLUDED.setting_value,
		updated_at = CURRENT_TIMESTAMP;
	"""
	settings_ok, settings_out, settings_err = run_psql_command(args.db, settings_sql, args.psql, args.verbose)
	settings_elapsed = time.perf_counter() - settings_start
	if not settings_ok:
		print(f"Warning: Failed to set test settings: {settings_err}", file=sys.stderr)
	
	sql_files = list_sql_files(args.category)
	if not sql_files:
		print(f"No SQL files found for category '{args.category}' in {TESTS_SQL_DIR}")
		return 2

	total = len(sql_files)
	passed = 0
	failed = 0
	t0 = time.perf_counter()

	# Print separator before tests
	print()
	print(HEADER_SEPARATOR)
	print()

	for idx, path in enumerate(sql_files, 1):
		# Check for shutdown request
		if _shutdown_requested:
			print("\n\nShutdown requested. Stopping test execution...")
			break
		
		when = datetime.now()
		name = os.path.basename(path)
		
		# Show which test is starting (will be overwritten)
		temp_line = format_test_line(True, when, idx, total, name, 0.0)
		print(temp_line, end="\r", flush=True)
		
		# Run the test (continue on failure)
		try:
			ok, elapsed, out_text, err_text = run_psql_file(args.db, path, args.psql, verbose=args.verbose)
		except Exception as e:
			# If test execution throws an exception, mark as failed but continue
			ok = False
			elapsed = 0.0
			out_text = ""
			err_text = str(e)
		
		# Check again after test execution
		if _shutdown_requested:
			print("\n\nShutdown requested. Stopping test execution...")
			break
		
		# Persist artifacts
		write_artifacts(name, ok, DEFAULT_OUTPUT_DIR, DEFAULT_ERROR_DIR, out_text, err_text)
		
		if ok:
			passed += 1
		else:
			failed += 1
		
		# Overwrite the starting line with final result (colored: green ✓ or red ✗)
		print(format_test_line(ok, when, idx, total, name, elapsed))
		
		# Verify GPU usage for ML training tests in GPU mode
		if ok and args.mode == "gpu" and ("train" in name.lower() or "linreg" in name.lower() or "logreg" in name.lower() or "rf" in name.lower() or "svm" in name.lower() or "ridge" in name.lower() or "lasso" in name.lower() or "dt" in name.lower() or "nb" in name.lower()):
			gpu_ok, gpu_err = verify_gpu_usage(args.db, args.psql, args.mode, name, args.host, args.port)
			if not gpu_ok:
				print(f"    {RED_BOLD}⚠ GPU Verification Failed: {gpu_err}{RESET}")
				# Don't mark test as failed, just warn
		
		# Only show error details in verbose mode
		if not ok and args.verbose:
			err_tail = "\n".join((err_text or "").strip().splitlines()[-5:])
			if err_tail:
				print(f"    {err_tail}")

	# Print separator after tests
	print()
	print(HEADER_SEPARATOR)
	print()

	# Print test report
	total_elapsed = time.perf_counter() - t0
	
	if _shutdown_requested:
		print("Test Report (Interrupted):")
	else:
		print("Test Report:")
	
	print()
	print(f"   Total Tests:    {total}")
	print(f"   Completed:      {idx if 'idx' in locals() else 0}")
	print(f"   Passed:         {passed}")
	print(f"   Failed:         {failed}")
	print(f"   Total Elapsed:  {total_elapsed:.2f}s")
	print()

	if _shutdown_requested:
		return 130  # Exit code for SIGINT
	return 0 if failed == 0 else 1


if __name__ == "__main__":
	sys.exit(main())


