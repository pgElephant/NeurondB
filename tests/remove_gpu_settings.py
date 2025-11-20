#!/usr/bin/env python3
"""
Remove all GPU/CPU references from test SQL files.
GPU/CPU configuration is now handled via GUC (ALTER SYSTEM).
"""

import os
import re
import sys
from typing import List, Tuple


def remove_set_statements(content: str) -> str:
	"""Remove SET statements for GPU/CPU configuration."""
	patterns = [
		r'^\s*SET\s+neurondb\.gpu_enabled\s*=.*?;?\s*$',
		r'^\s*SET\s+neurondb\.gpu_kernels\s*=.*?;?\s*$',
		r'^\s*SET\s+neurondb\.automl\.use_gpu\s*=.*?;?\s*$',
	]
	
	for pattern in patterns:
		content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)
	
	return content


def remove_gpu_function_calls(content: str) -> str:
	"""Remove SELECT statements with GPU function calls."""
	patterns = [
		r'^\s*SELECT\s+neurondb_gpu_enable\(\)\s*.*?;?\s*$',
		r'^\s*SELECT\s+neurondb_gpu_info\(\)\s*.*?;?\s*$',
		r'^\s*SELECT\s+.*?neurondb_gpu_enable\(\)\s*AS\s+\w+.*?;?\s*$',
		r'^\s*SELECT\s+.*?neurondb_gpu_info\(\)\s*AS\s+\w+.*?;?\s*$',
	]
	
	for pattern in patterns:
		content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)
	
	return content


def remove_current_setting_refs(content: str) -> str:
	"""Remove current_setting() references for GPU settings."""
	patterns = [
		r"current_setting\('neurondb\.gpu_enabled'\)",
		r"current_setting\('neurondb\.gpu_kernels'\)",
	]
	
	for pattern in patterns:
		content = re.sub(pattern, '', content, flags=re.IGNORECASE)
	
	return content


def remove_gpu_comments(content: str) -> str:
	"""Remove comments and echo statements about GPU configuration."""
	patterns = [
		r'/\*.*?GPU.*?configuration.*?\*/',
		r'/\*.*?Register.*?GPU.*?kernels.*?\*/',
		r"\\echo\s+['\"]?.*?GPU.*?Configuration.*?['\"]?\s*",
	]
	
	for pattern in patterns:
		content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
	
	return content


def remove_gpu_headers(content: str) -> str:
	"""Remove headers mentioning GPU acceleration."""
	patterns = [
		r'\*.*?with GPU acceleration',
		r'\*.*?GPU acceleration',
	]
	
	for pattern in patterns:
		content = re.sub(pattern, '*', content, flags=re.IGNORECASE)
	
	return content


def clean_empty_lines(lines: List[str]) -> List[str]:
	"""Remove empty lines and clean up after GPU removal."""
	filtered = []
	skip_next = False
	
	for line in lines:
		if skip_next and not line.strip():
			skip_next = False
			continue
		
		line_stripped = line.strip()
		
		# Skip empty SELECT statements
		if line_stripped in ('SELECT;', 'SELECT'):
			skip_next = True
			continue
		
		# Skip lines with only commas
		if re.match(r'^\s*,\s*$', line):
			continue
		
		# Skip GPU configuration echo statements
		if '\\echo' in line and 'GPU' in line.upper():
			if any(word in line.lower() for word in ['configur', 'register', 'setup']):
				continue
		
		# Skip step comments about GPU configuration
		if re.search(r'Step.*[Cc]onfigur.*GPU', line, re.IGNORECASE):
			continue
		
		filtered.append(line)
	
	return filtered


def remove_gpu_references(file_path: str) -> bool:
	"""
	Remove all GPU/CPU references from SQL file.
	Returns True if file was modified, False otherwise.
	"""
	with open(file_path, 'r', encoding='utf-8') as f:
		content = f.read()
	
	original = content
	
	# Apply all removal steps
	content = remove_set_statements(content)
	content = remove_gpu_function_calls(content)
	content = remove_current_setting_refs(content)
	content = remove_gpu_comments(content)
	content = remove_gpu_headers(content)
	
	# Clean up lines
	lines = content.split('\n')
	lines = clean_empty_lines(lines)
	content = '\n'.join(lines)
	
	# Clean up multiple consecutive empty lines
	content = re.sub(r'\n{3,}', '\n\n', content)
	
	# Write if changed
	if content != original:
		with open(file_path, 'w', encoding='utf-8') as f:
			f.write(content)
		return True
	
	return False


def find_sql_files(base_dir: str) -> List[str]:
	"""Find all SQL files in basic, advance, and negative directories."""
	sql_dirs = [
		os.path.join(base_dir, 'sql', 'basic'),
		os.path.join(base_dir, 'sql', 'advance'),
		os.path.join(base_dir, 'sql', 'negative'),
	]
	
	files = []
	for sql_dir in sql_dirs:
		if not os.path.isdir(sql_dir):
			continue
		
		for root, dirs, filenames in os.walk(sql_dir):
			for filename in filenames:
				if filename.endswith('.sql'):
					files.append(os.path.join(root, filename))
	
	return sorted(files)


def main() -> int:
	"""Main entry point."""
	base_dir = os.path.dirname(__file__)
	sql_files = find_sql_files(base_dir)
	
	updated_count = 0
	for file_path in sql_files:
		if remove_gpu_references(file_path):
			rel_path = os.path.relpath(file_path, base_dir)
			print(f"Updated: {rel_path}")
			updated_count += 1
	
	print(f"\nTotal files updated: {updated_count}")
	return 0


if __name__ == '__main__':
	sys.exit(main())
