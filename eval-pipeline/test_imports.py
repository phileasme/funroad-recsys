#!/usr/bin/env python3
import os
import sys

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Check if we're in eval-pipeline directory
in_eval_pipeline = os.path.basename(script_dir) == 'eval-pipeline'
print(f"In eval-pipeline directory: {in_eval_pipeline}")

# Add appropriate paths
if in_eval_pipeline:
    # We're already in eval-pipeline, use direct imports
    print("Using direct imports (already in eval-pipeline)")
    module_path = script_dir
else:
    # We're in project root, add eval-pipeline to path
    module_path = os.path.join(script_dir, 'eval-pipeline')
    print(f"Adding module path: {module_path}")
    sys.path.append(module_path)

# Check if the module path exists
if os.path.exists(module_path):
    print(f"Module path exists: {module_path}")
else:
    print(f"ERROR: Module path does not exist: {module_path}")
    sys.exit(1)

# List all Python files in the module path
print("\nPython files in module path:")
for file in os.listdir(module_path):
    if file.endswith('.py'):
        print(f"  - {file}")

# Try importing the modules - with hyphens converted to underscores
print("\nTrying imports with hyphens converted to underscores:")
try:
    module = __import__('process-gumroad-data'.replace('-', '_'))
    print("  ✓ process-gumroad-data")
except ImportError as e:
    print(f"  ✗ process-gumroad-data: {e}")

try:
    module = __import__('query-search-endpoint'.replace('-', '_'))
    print("  ✓ query-search-endpoint")
except ImportError as e:
    print(f"  ✗ query-search-endpoint: {e}")

try:
    module = __import__('generate-expert-rankings'.replace('-', '_'))
    print("  ✓ generate-expert-rankings")
except ImportError as e:
    print(f"  ✗ generate-expert-rankings: {e}")

try:
    module = __import__('calculate-metrics'.replace('-', '_'))
    print("  ✓ calculate-metrics")
except ImportError as e:
    print(f"  ✗ calculate-metrics: {e}")

# Print Python path
print("\nPython path:")
for p in sys.path:
    print(f"  - {p}")

print("\nTest complete.")