#!/usr/bin/env python3
"""
Remove hardcoded view creation code from test SQL files.
The test runner will inject this dynamically based on --num_rows.
"""

import os
import re
import sys

def remove_view_creation_blocks(file_path):
	"""
	Remove hardcoded view creation blocks from SQL file.
	Removes:
	- DO $$ blocks that check for sample_train/sample_test tables
	- DROP VIEW IF EXISTS test_train_view/test_test_view
	- CREATE VIEW test_train_view/test_test_view with LIMIT
	"""
	with open(file_path, 'r', encoding='utf-8') as f:
		content = f.read()
	
	original_content = content
	
	# Remove the DO $$ block that checks for sample_train/sample_test
	# Pattern: DO $$ ... BEGIN ... IF NOT EXISTS ... sample_train/sample_test ... END $$;
	do_block_pattern = r'DO\s+\$\$\s*BEGIN\s+IF\s+NOT\s+EXISTS\s*\(SELECT\s+1\s+FROM\s+information_schema\.tables\s+WHERE\s+table_schema\s*=\s*[\'"]public[\'"]\s+AND\s+table_name\s*=\s*[\'"]sample_(train|test)[\'"]\)\s+THEN.*?END\s+\$\$;'
	content = re.sub(do_block_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
	
	# Remove DROP VIEW statements for test views
	content = re.sub(r'DROP\s+VIEW\s+IF\s+EXISTS\s+test_(train|test)_view\s*;', '', content, flags=re.IGNORECASE)
	
	# Remove CREATE VIEW statements for test views
	# Pattern: CREATE VIEW test_train_view/test_test_view AS SELECT ... LIMIT X;
	create_view_pattern = r'CREATE\s+VIEW\s+test_(train|test)_view\s+AS\s+SELECT\s+features,\s+label\s+FROM\s+sample_(train|test)\s+LIMIT\s+\d+\s*;'
	content = re.sub(create_view_pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
	
	# Remove comment lines about creating views with X rows
	content = re.sub(r'--\s*Create\s+views\s+with\s+\d+\s+rows?\s+for\s+(basic|advance|negative)\s+tests?', '', content, flags=re.IGNORECASE)
	
	# Clean up multiple blank lines (more than 2 consecutive)
	content = re.sub(r'\n{3,}', '\n\n', content)
	
	# Only write if content changed
	if content != original_content:
		with open(file_path, 'w', encoding='utf-8') as f:
			f.write(content)
		return True
	return False

def main():
	# Find all SQL files in sql/basic, sql/advance, sql/negative
	base_dir = os.path.dirname(__file__)
	sql_dirs = [
		os.path.join(base_dir, 'sql', 'basic'),
		os.path.join(base_dir, 'sql', 'advance'),
		os.path.join(base_dir, 'sql', 'negative'),
	]
	
	total_updated = 0
	for sql_dir in sql_dirs:
		if not os.path.isdir(sql_dir):
			continue
		
		for root, dirs, files in os.walk(sql_dir):
			for file in files:
				if file.endswith('.sql'):
					file_path = os.path.join(root, file)
					if remove_view_creation_blocks(file_path):
						print(f"Updated: {os.path.relpath(file_path, base_dir)}")
						total_updated += 1
	
	print(f"\nTotal files updated: {total_updated}")
	return 0

if __name__ == '__main__':
	sys.exit(main())

