# Search Engine Evaluation Pipeline

This folder contains scripts to evaluate and compare the performance of your custom search engine against Gumroad's search results. The evaluation uses expert rankings from Anthropic's Claude API to provide an objective quality assessment.

## Overview

The pipeline consists of the following steps:

1. **Process Gumroad Data**: Extract product results from Gumroad's search files
2. **Query Custom Search Endpoint**: Send the same queries to your search endpoint
3. **Generate Expert Rankings**: Use Anthropic's Claude to rank results from both sources
4. **Calculate Metrics**: Compute nDCG, Precision, Recall, and MRR for comparison

## Setup

### Prerequisites

- Python 3.8+
- Required Python packages:
  ```
  pip install requests numpy pandas matplotlib seaborn anthropic
  ```
- Anthropic API key (for expert rankings)

### Directory Structure

Place all files in a folder called `eval-pipeline` in your project root:

```
project_root/
├── gumroad_data/
│   ├── search_poker/
│   ├── search_digital_art/
│   └── ...
├── eval-pipeline/
│   ├── process_gumroad_data.py
│   ├── query_search_endpoint.py
│   ├── generate_expert_rankings.py
│   ├── calculate_metrics.py
│   └── run-evaluation.py
└── run-eval.sh
```

## Usage

### Run from Root Directory

Use the provided shell script from the project root:

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_api_key_here

# Run with default settings
./run-eval.sh

# Run with custom settings
./run-eval.sh \
  --gumroad-data /path/to/gumroad_data \
  --endpoint-url http://localhost:8000/two_phase_optimized \
  --max-queries 30 \
  --k-values 5 10 20 50
```

### Run Directly

You can also run the main Python script directly:

```bash
cd eval-pipeline
python run-evaluation.py \
  --gumroad-data ../gumroad_data \
  --endpoint-url http://localhost:8000/two_phase_optimized \
  --api-key your_api_key_here
```

### Command-Line Options

- `--gumroad-data`: Directory containing Gumroad data folders (default: './gumroad_data')
- `--endpoint-url`: URL of your search endpoint (default: 'http://localhost:8000/two_phase_optimized')
- `--api-key`: Anthropic API key (if not set as environment variable)
- `--skip-gumroad`: Skip Gumroad data processing step
- `--skip-endpoint`: Skip endpoint querying step
- `--skip-expert`: Skip expert ranking generation step
- `--max-queries`: Maximum number of queries to process (default: 50)
- `--k-values`: k values to calculate metrics at (default: 5 10 20 50)

## Output

The pipeline creates a timestamped directory with the following structure:

```
search_eval_20250324_120000/
├── gumroad_processed/
├── endpoint_results/
├── expert_rankings/
└── evaluation_results/
    ├── metrics_report.json
    ├── ndcg_comparison.png
    ├── precision_comparison.png
    ├── recall_comparison.png
    └── mrr_comparison.png
```

A summary of the results is printed to the console and also saved to `evaluation_pipeline.log`.

## Notes

- The products are randomized before being sent to the expert ranking API to avoid position bias.
- The source of each product is tracked, allowing analysis of any potential source bias in the expert rankings.
- For fairness, the evaluation uses the same set of queries for both search engines.
- Since the data is already filtered based on Gumroad results, absolute recall metrics may not represent performance against the full corpus.

## Metrics Explanation

- **nDCG@k**: Normalized Discounted Cumulative Gain - Measures the ranking quality, taking into account the position of relevant items.
- **Precision@k**: The fraction of retrieved items that are relevant.
- **Recall@k**: The fraction of relevant items that are retrieved.
- **MRR**: Mean Reciprocal Rank - The average of the reciprocal ranks of the first relevant item.
