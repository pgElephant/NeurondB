#!/usr/bin/env python3
"""Fix text casts in RAG SQL files"""
import re
import glob

# Pattern to find string literals that need ::text cast
# Matches 'string' that's not already followed by ::text
pattern = r"'([^']+)'\s*(?!::text)"

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Add ::text after each string literal in neurondb_generate_embedding calls
    fixed = re.sub(
        r"neurondb_generate_embedding\(\s*'([^']+)'",
        r"neurondb_generate_embedding('\1'::text",
        content
    )
    
    with open(filepath, 'w') as f:
        f.write(fixed)
    print(f"✓ Fixed {filepath}")

# Fix all RAG SQL files
for filepath in glob.glob("*.sql"):
    if filepath not in ['000_run_all_tests.sql', '001_document_ingestion.sql', '002_generate_embeddings.sql', '999_cleanup.sql']:
        fix_file(filepath)

print("✅ All files fixed")

