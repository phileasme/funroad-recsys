#!/bin/bash

# Run evaluation pipeline from root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if running from eval-pipeline or root directory
if [[ "$(basename "${SCRIPT_DIR}")" == "eval-pipeline" ]]; then
  # Already in eval-pipeline directory
  EVAL_DIR="${SCRIPT_DIR}"
  ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
else
  # In root directory
  ROOT_DIR="${SCRIPT_DIR}"
  EVAL_DIR="${ROOT_DIR}/eval-pipeline"
fi

echo "Root directory: ${ROOT_DIR}"
echo "Eval directory: ${EVAL_DIR}"

if [ ! -d "${EVAL_DIR}" ]; then
  echo "Error: eval-pipeline directory not found at ${EVAL_DIR}"
  exit 1
fi

# Default values
GUMROAD_DATA="${ROOT_DIR}/gumroad_data"
ENDPOINT_URL="http://localhost:8000/two_phase_optimized"
MAX_QUERIES=50

# Check for ANTHROPIC_API_KEY
if [ -z "${ANTHROPIC_API_KEY}" ]; then
  echo "Warning: ANTHROPIC_API_KEY environment variable is not set"
  echo "You'll need to provide it with --api-key parameter"
fi

# Parse command line arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --continue)
      CONTINUE="--continue"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --skip-gumroad)
      SKIP_GUMROAD="--skip-gumroad"
      shift
      ;;
    --skip-endpoint)
      SKIP_ENDPOINT="--skip-endpoint"
      shift
      ;;
    --skip-expert)
      SKIP_EXPERT="--skip-expert"
      shift
      ;;
    --skip-metrics)
      SKIP_METRICS="--skip-metrics" 
      shift
      ;;
    --gumroad-data)
      GUMROAD_DATA="$2"
      shift
      shift
      ;;
    --endpoint-url)
      ENDPOINT_URL="$2"
      shift
      shift
      ;;
    --max-queries)
      MAX_QUERIES="$2"
      shift
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Run the evaluation script
echo "Running evaluation pipeline..."
echo "Gumroad data: ${GUMROAD_DATA}"
echo "Endpoint URL: ${ENDPOINT_URL}"
echo "Max queries: ${MAX_QUERIES}"

# Check which version of the script exists
if [ -f "${EVAL_DIR}/run_evaluation.py" ]; then
  # Use the fixed version if it exists
  EVAL_SCRIPT="${EVAL_DIR}/run_evaluation.py"
elif [ -f "${EVAL_DIR}/run_evaluation.py" ]; then
  # Fall back to original version
  EVAL_SCRIPT="${EVAL_DIR}/run_evaluation.py"
else
  echo "Error: No evaluation script found in ${EVAL_DIR}"
  exit 1
fi

echo "Using evaluation script: ${EVAL_SCRIPT}"

python3 "${EVAL_SCRIPT}" \
  --gumroad-data "${GUMROAD_DATA}" \
  --endpoint-url "${ENDPOINT_URL}" \
  --max-queries "${MAX_QUERIES}" \
  ${CONTINUE} \
  ${OUTPUT_DIR:+--output-dir "${OUTPUT_DIR}"} \
  ${SKIP_GUMROAD} \
  ${SKIP_ENDPOINT} \
  ${SKIP_EXPERT} \
  ${SKIP_METRICS} \
  "${POSITIONAL_ARGS[@]}"

exit $?