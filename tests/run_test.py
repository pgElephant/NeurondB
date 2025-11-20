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
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

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
	
	# Check which source table exists (higgs.test_train/test_test, sample_train/sample_test, or test_train/test_test)
	try:
		# Check for higgs.test_train first (preferred), then sample_train, then test_train
		check_train_cmd = [
			psql_path, "-d", dbname, "-t", "-A", "-w",
			"-c", "SELECT table_schema || '.' || table_name FROM information_schema.tables WHERE (table_schema = 'higgs' AND table_name = 'test_train') OR (table_schema = 'public' AND table_name IN ('sample_train', 'test_train')) ORDER BY CASE WHEN table_schema = 'higgs' THEN 0 WHEN table_name = 'sample_train' THEN 1 ELSE 2 END LIMIT 1;"
		]
		
		train_proc = subprocess.Popen(check_train_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
		train_out, train_err = train_proc.communicate()
		
		if train_proc.returncode == 0 and train_out.strip():
			train_table_full = train_out.strip()
			
			# Determine corresponding test table
			if train_table_full.startswith('higgs.'):
				test_table_full = "higgs.test_test"
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
				# Create views with LIMIT num_rows
				# Convert REAL[] to vector if needed (when neurondb extension is available)
				# Try to cast to vector, fallback to original type if extension not available
				create_sql = f"""
DROP VIEW IF EXISTS test_train_view CASCADE;
DROP VIEW IF EXISTS test_test_view CASCADE;

CREATE VIEW test_train_view AS
SELECT features::vector(28) as features, label 
FROM {train_table_full} 
LIMIT {num_rows};

CREATE VIEW test_test_view AS
SELECT features::vector(28) as features, label 
FROM {test_table_full} 
LIMIT {num_rows};
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
						return True, row_count
		
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

	# Optionally set GPU kernels if provided
	if gpu_kernels:
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

	if verbose:
		print(f"Switched to {mode.upper()} mode successfully.")
	return True


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
	
	# 3. Create test views
	when = datetime.now()
	views_start = time.perf_counter()
	views_ok, row_count = create_test_views(args.db, args.psql, args.num_rows, args.host, args.port)
	views_elapsed = time.perf_counter() - views_start
	print(format_status_line(views_ok, when, f"Creating dataset for tests (rows={row_count})...", views_elapsed))
	if not views_ok:
		print(f"Warning: Failed to create test views. Some tests may fail.", file=sys.stderr)
		# Continue anyway - some tests might not need the views
	
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


