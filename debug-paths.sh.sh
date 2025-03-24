#!/bin/bash

# Debug script to verify paths and file locations
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: ${SCRIPT_DIR}"

# Check if running from eval-pipeline or root directory
if [[ "$(basename "${SCRIPT_DIR}")" == "eval-pipeline" ]]; then
  # Already in eval-pipeline directory
  EVAL_DIR="${SCRIPT_DIR}"
  ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
  echo "Running from eval-pipeline directory"
else
  # In root directory
  ROOT_DIR="${SCRIPT_DIR}"
  EVAL_DIR="${ROOT_DIR}/eval-pipeline"
  echo "Running from root directory"
fi

echo "Root directory: ${ROOT_DIR}"
echo "Eval directory: ${EVAL_DIR}"

# Check if eval-pipeline directory exists
if [ -d "${EVAL_DIR}" ]; then
  echo "eval-pipeline directory exists at: ${EVAL_DIR}"
else
  echo "ERROR: eval-pipeline directory not found at: ${EVAL_DIR}"
fi

# List Python files in eval directory
echo -e "\nPython files in eval directory:"
if [ -d "${EVAL_DIR}" ]; then
  for file in "${EVAL_DIR}"/*.py; do
    if [ -f "$file" ]; then
      echo "  - $(basename "${file}")"
    fi
  done
  
  if [ -z "$(ls -A "${EVAL_DIR}"/*.py 2>/dev/null)" ]; then
    echo "  No Python files found!"
  fi
else
  echo "  Cannot list files, directory doesn't exist"
fi

# Check for gumroad_data directory
GUMROAD_DATA="${ROOT_DIR}/gumroad_data"
echo -e "\nChecking for gumroad_data:"
if [ -d "${GUMROAD_DATA}" ]; then
  echo "gumroad_data directory exists at: ${GUMROAD_DATA}"
  echo "Contents (sample):"
  ls -la "${GUMROAD_DATA}" | head -n 5
else
  echo "ERROR: gumroad_data directory not found at: ${GUMROAD_DATA}"
fi

# Check Python interpreter
echo -e "\nPython interpreter:"
which python3
python3 --version

# Check for required Python packages
echo -e "\nChecking required Python packages:"
PACKAGES=("requests" "numpy" "pandas" "matplotlib" "seaborn" "anthropic")
for pkg in "${PACKAGES[@]}"; do
  if python3 -c "import ${pkg}" 2>/dev/null; then
    echo "  ✓ ${pkg}"
  else
    echo "  ✗ ${pkg} (not installed)"
  fi
done

echo -e "\nDebug complete."
